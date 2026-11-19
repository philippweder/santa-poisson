import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from typing import ClassVar, Self

import gmsh
import numpy as np
import shapely
import shapely.affinity
import shapely.geometry.base
import shapely.geometry.polygon
from numpy.typing import NDArray
from scipy.special import expit  # pyright: ignore[reportMissingTypeStubs] scipy ships no stubs

from advent.shapes.triangle_mesh import TriangleMesh


@dataclass(frozen=True, slots=True, kw_only=True)
class Shape:
    """
    A planar region: one valid polygon, holes allowed, no two rings touching.

    A shape holds a value and its methods compute representations of it. Coordinate
    arrays put the axis first, shape ``(2, ...)``, so ``np.stack([X, Y])`` of a grid
    works directly and dolfinx's ``x`` of shape ``(3, n)`` is passed as ``x[:2]``.
    Set operations stay with shapely:
    ``Shape.from_geometry(a.polygon.difference(b.polygon))``.

    A point with a coordinate that is NaN or infinite has the value NaN.
    """

    GDIM: ClassVar[int] = 2
    """The number of coordinates of a point: shapes lie in a plane."""

    GMSH_TRIANGLE: ClassVar[int] = 2
    """The element type gmsh gives a three-node triangle."""

    GMSH_OPTIONS: ClassVar[tuple[tuple[str, float], ...]] = (
        ("General.Terminal", 0.0),
        ("Mesh.Algorithm", 6.0),
        ("Mesh.ElementOrder", 1.0),
        ("Mesh.RecombineAll", 0.0),
        ("Mesh.MeshSizeFactor", 1.0),
        ("Mesh.MeshSizeFromPoints", 1.0),
        ("Mesh.MeshSizeExtendFromBoundary", 1.0),
        ("Mesh.MeshSizeMin", 0.0),
        ("Mesh.MeshSizeMax", 1e22),
    )
    """
    The gmsh options ``mesh`` sets, with gmsh's defaults, and restores afterwards.

    Algorithm 6 is 2D Frontal-Delaunay; the rest silence gmsh and pin the element
    order, recombination and sizes. A caller's other options still apply.
    """

    GMSH_MODEL: ClassVar[str] = "advent_shape_mesh"
    """The name of the gmsh model ``mesh`` works in; gmsh allows a name used twice."""

    polygon: shapely.Polygon
    """The region: valid, non-empty, one polygon, holes allowed, rings apart."""

    def __post_init__(self) -> None:
        """
        Refuse anything but one valid, non-empty polygon whose rings do not touch.

        A hole that meets the shell, or two holes that share a point, are valid for
        shapely, but gmsh makes no triangles for such a polygon or does not return,
        so no shape is made of one.

        :raises ValueError: If the geometry is not a single polygon, is empty, is
            invalid (self-intersecting, or with a non-finite coordinate), or has two
            rings that touch.
        """
        if self.polygon.geom_type != "Polygon":
            raise ValueError(f"need one polygon, got a {self.polygon.geom_type}")
        if self.polygon.is_empty:
            raise ValueError("the polygon is empty")
        if not self.polygon.is_valid:
            reason = shapely.is_valid_reason(self.polygon)
            raise ValueError(f"the polygon is invalid: {reason}")
        touches = self._touch_points(self.polygon)
        if touches:
            raise ValueError(
                f"two rings of the polygon touch at {touches[0]}, which gmsh does not "
                "mesh: it makes no triangles or does not return; moving or scaling "
                "a shape can make rings touch by rounding"
            )

    @classmethod
    def from_geometry(cls, geometry: shapely.geometry.base.BaseGeometry) -> Self:
        """
        Return the shape of a polygon that a set operation made.

        Shapely types its set operations as returning its general base class, though
        the result is often one polygon, and this makes that narrowing explicit.

        :param geometry: The geometry, for example ``a.polygon.difference(b.polygon)``.
        :returns: The shape of ``geometry``.
        :raises ValueError: If ``geometry`` is not one valid, non-empty polygon whose
            rings do not touch.
        """
        if not isinstance(geometry, shapely.Polygon):
            raise ValueError(f"need one polygon, got a {geometry.geom_type}")
        return cls(polygon=geometry)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """The bounding box as ``(min_x, min_y, max_x, max_y)``."""
        min_x, min_y, max_x, max_y = self.polygon.bounds
        return (min_x, min_y, max_x, max_y)

    def scaled(self, factor: float) -> Self:
        """
        Return this shape scaled about the origin.

        :param factor: The scale factor in both axes.
        :returns: The scaled shape.
        :raises ValueError: If ``factor`` is not finite, or the result is not a valid
            polygon, as when ``factor`` is zero, or its rings touch.
        """
        if not math.isfinite(factor):
            raise ValueError(f"factor must be finite, got {factor}")

        scaled = shapely.affinity.scale(
            self.polygon, xfact=factor, yfact=factor, origin=(0.0, 0.0)
        )
        return type(self)(polygon=scaled)

    def translated(self, *, dx: float, dy: float) -> Self:
        """
        Return this shape moved by a vector.

        :param dx: The shift along the first axis.
        :param dy: The shift along the second axis.
        :returns: The translated shape.
        :raises ValueError: If a shift is not finite, or the result is not a valid
            polygon with apart rings, which rounding of the moved coordinates can break.
        """
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise ValueError(f"the shift must be finite, got dx={dx}, dy={dy}")

        moved = shapely.affinity.translate(self.polygon, xoff=dx, yoff=dy)
        return type(self)(polygon=moved)

    def normalised(self) -> Self:
        """
        Return this shape moved and scaled to the frame of the shape builders.

        The bounding box of the result is centred at the origin and its larger side
        has length 1.

        :returns: The shape translated by minus the centre of its bounding box, then
            scaled about the origin by the reciprocal of the larger side.
        :raises ValueError: If rounding of the moved coordinates makes two rings touch.
        """
        min_x, min_y, max_x, max_y = self.bounds
        centred = self.translated(dx=-0.5 * (min_x + max_x), dy=-0.5 * (min_y + max_y))
        return centred.scaled(1.0 / max(max_x - min_x, max_y - min_y))

    def signed_distance(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Return the signed distance of locations to the boundary of the region.

        The magnitude is the Euclidean distance to the boundary, holes included,
        computed by GEOS in double precision. The value is negative strictly inside,
        positive outside and zero on the boundary. It is NaN where a coordinate is not
        finite. It has kinks on the
        medial axis and is not smoothed.

        :param points: The points, shape ``(2, ...)``.
        :returns: The distances, shape ``points.shape[1:]``.
        :raises ValueError: If the first axis of ``points`` does not have length 2.
        """
        x, y = self._split(points)
        known = np.isfinite(x) & np.isfinite(y)
        x, y = x[known], y[known]
        # The annotation of shapely.distance leaves its keyword arguments untyped.
        distance = shapely.distance(  # pyright: ignore[reportUnknownMemberType] shapely stub
            self.polygon.boundary, shapely.points(x, y)
        )
        inside = shapely.contains_xy(self.polygon, x, y)
        result = np.full(known.shape, np.nan)
        result[known] = np.where(inside, -distance, distance)
        return result.reshape(np.shape(points)[1:])

    def indicator(
        self, points: NDArray[np.float64], *, width: float
    ) -> NDArray[np.float64]:
        """
        Return a smoothed indicator of the region.

        The value is ``(1 - tanh(phi / width)) / 2`` with ``phi`` the signed distance,
        computed as ``expit(-2 * phi / width)``. That is the same function, but it
        keeps the small values far outside accurate where ``1 - tanh`` rounds them
        away. Values lie in [0, 1], are exactly 0 or 1 far from the boundary, and are
        one half on it.

        :param points: The points, shape ``(2, ...)``.
        :param width: The length over which the indicator changes, in coordinate units.
        :returns: The indicator, shape ``points.shape[1:]``.
        :raises ValueError: If the first axis of ``points`` does not have length 2, or
            ``width`` is not finite and positive.
        """
        if not (math.isfinite(width) and width > 0.0):
            raise ValueError(f"width must be finite and positive, got {width}")

        phi = self.signed_distance(points)
        return np.asarray(expit(-2.0 * phi / width), dtype=np.float64)

    def mesh(self, *, size: float) -> TriangleMesh:
        """
        Return a triangle mesh of the region, made by gmsh.

        The mesh is of 2D Frontal-Delaunay type with target edge length ``size``, and
        its boundary is the polygon boundary. If gmsh is already initialised, the mesh
        is made in a model of its own. Whatever the call finds is restored on return,
        also when it raises: whether gmsh is initialised, the current model and the
        options the call sets.

        :param size: The target edge length, in coordinate units.
        :returns: The mesh, with counter-clockwise triangles.
        :raises ValueError: If ``size`` is not finite and positive, or if gmsh makes
            no triangles.
        """
        if not (math.isfinite(size) and size > 0.0):
            raise ValueError(f"size must be finite and positive, got {size}")

        polygon = shapely.geometry.polygon.orient(
            shapely.remove_repeated_points(self.polygon), sign=1.0
        )
        with self._gmsh_workspace():
            loops = [
                self._add_loop(shapely.get_coordinates(ring)[:-1], size)
                for ring in (polygon.exterior, *polygon.interiors)
            ]
            gmsh.model.geo.addPlaneSurface(loops)
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)
            tags, coordinates, _ = gmsh.model.mesh.getNodes(returnParametricCoord=False)
            _, corner_tags = gmsh.model.mesh.getElementsByType(self.GMSH_TRIANGLE)

        if corner_tags.size == 0:
            raise ValueError(f"gmsh made no triangles with size={size}")

        order = np.argsort(tags)
        points = coordinates.reshape(-1, 3)[order, : self.GDIM].T
        triangles = np.searchsorted(tags[order], corner_tags).reshape(-1, 3)
        return TriangleMesh(
            points=points,
            triangles=triangles.astype(np.int64),
            boundary_edges=self._boundary_edges(triangles),
        )

    @classmethod
    def _split(
        cls, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return the two coordinate rows of the argument, flattened.

        :param points: The points, shape ``(2, ...)``.
        :returns: The first and the second coordinate, each of shape ``(n,)``.
        :raises ValueError: If the first axis of ``points`` does not have length 2.
        """
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 0 or array.shape[0] != cls.GDIM:
            raise ValueError(
                f"points must have shape ({cls.GDIM}, ...), got {array.shape}; pass "
                "x[:2] for a 3D array"
            )
        x, y = array.reshape(cls.GDIM, -1)
        return x, y

    @staticmethod
    def _touch_points(polygon: shapely.Polygon) -> list[str]:
        """
        Find where two rings of a valid region meet.

        Rings of a valid region never cross, so two that intersect touch.

        :param polygon: The region, valid.
        :returns: The common part, as text, of each pair of rings that have a point in
            common: the shell and a hole, or two holes; empty when the rings are apart.
        """
        rings = (polygon.exterior, *polygon.interiors)
        return [
            shapely.intersection(first, second).wkt
            for first, second in combinations(rings, 2)
            if first.intersects(second)
        ]

    @staticmethod
    def _add_loop(vertices: NDArray[np.float64], size: float) -> int:
        """
        Add a closed polygonal curve loop to the current gmsh model.

        :param vertices: The distinct vertices in order, shape ``(n, 2)``.
        :param size: The target mesh size at every vertex.
        :returns: The tag of the curve loop.
        """
        point_tags = [gmsh.model.geo.addPoint(x, y, 0.0, size) for x, y in vertices]
        line_tags = [
            gmsh.model.geo.addLine(start, end)
            for start, end in zip(
                point_tags, point_tags[1:] + point_tags[:1], strict=True
            )
        ]
        return gmsh.model.geo.addCurveLoop(line_tags)

    @staticmethod
    def _boundary_edges(triangles: NDArray[np.intp]) -> NDArray[np.int64]:
        """
        Return the edges that belong to one triangle, directed as in that triangle.

        :param triangles: The vertex indices of the triangles, shape ``(m, 3)``.
        :returns: The directed boundary edges, shape ``(k, 2)``.
        """
        directed = np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        )
        _, inverse, counts = np.unique(
            np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
        )
        return directed[counts[inverse.ravel()] == 1].astype(np.int64)

    @classmethod
    @contextmanager
    def _gmsh_workspace(cls) -> Generator[None]:
        """
        Provide a fresh current gmsh model, and restore what the caller had.

        Gmsh is initialised if it was not, without reading configuration files so that
        the user's settings cannot change the mesh, and finalised again on exit. If it
        was initialised, a model of this call's own is made current for the duration,
        and the caller's current model and the options this call sets are put back.
        """
        was_initialised = bool(gmsh.isInitialized())
        if not was_initialised:
            gmsh.initialize(readConfigFiles=False)
        try:
            caller_model = gmsh.model.getCurrent()
            caller_options = {
                name: gmsh.option.getNumber(name) for name, _ in cls.GMSH_OPTIONS
            }
            gmsh.model.add(cls.GMSH_MODEL)
            try:
                for option, value in cls.GMSH_OPTIONS:
                    gmsh.option.setNumber(option, value)
                yield
            finally:
                gmsh.model.remove()
                gmsh.model.setCurrent(caller_model)
                for option, value in caller_options.items():
                    gmsh.option.setNumber(option, value)
        finally:
            if not was_initialised:
                gmsh.finalize()
