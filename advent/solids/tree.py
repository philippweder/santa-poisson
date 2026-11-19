import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar

import gmsh
import numpy as np
import shapely

from advent.shapes.shape import Shape
from advent.shapes.star import Star
from advent.solids.surface_mesh import SurfaceMesh
from advent.solids.tetrahedron_mesh import TetrahedronMesh


@dataclass(frozen=True, slots=True, kw_only=True)
class Tree:
    """
    A Christmas tree: a trunk, stacked truncated cones and a star on top.

    The axis is vertical, the base is at height 0 and the top of the star at height 1,
    so every length is a fraction of the height of the tree, and the bounding box is
    centred on the axis. The parts touch along faces and do not overlap: the trunk is
    a cylinder, each cone stands on the one below with the same height as the others,
    and the star stands on the top cone on a short stem. The star is a flat extrusion
    in the plane through the axis that holds the first horizontal axis.
    """

    STEM_HEIGHT: ClassVar[float] = 0.15
    """The height of the star's stem, as a multiple of ``star_size``."""

    STEM_WIDTH: ClassVar[float] = 0.2
    """The width of the star's stem, as a multiple of ``star_size``."""

    BODY_TAG: ClassVar[int] = 1
    """The tag of the trunk and the cones in ``TetrahedronMesh.tetrahedron_tags``."""

    STAR_TAG: ClassVar[int] = 2
    """The tag of the star, with its stem, in ``TetrahedronMesh.tetrahedron_tags``."""

    DIM_FACE: ClassVar[int] = 2
    """The dimension of a face, in the dimension and tag pairs of gmsh."""

    DIM_VOLUME: ClassVar[int] = 3
    """The dimension of a volume, in the dimension and tag pairs of gmsh."""

    GMSH_TETRAHEDRON: ClassVar[int] = 4
    """The element type gmsh gives a four-node tetrahedron."""

    GMSH_OPTIONS: ClassVar[tuple[tuple[str, float], ...]] = (
        ("General.Terminal", 0.0),
        ("Mesh.Algorithm", 6.0),
        ("Mesh.Algorithm3D", 1.0),
        ("Mesh.ElementOrder", 1.0),
        ("Mesh.RecombineAll", 0.0),
        ("Mesh.MeshSizeFactor", 1.0),
        ("Mesh.MeshSizeFromPoints", 1.0),
        ("Mesh.MeshSizeFromCurvature", 0.0),
        ("Mesh.MeshSizeExtendFromBoundary", 1.0),
        ("Mesh.MeshSizeMin", 0.0),
        ("Mesh.MeshSizeMax", 1e22),
    )
    """
    The gmsh options ``mesh`` sets and restores; gmsh's defaults, bar
    ``General.Terminal``, which starts at 1. Algorithm 6 is 2D Frontal-Delaunay and
    3D 1 is Delaunay; the rest silence gmsh and pin the order, recombination and
    sizes. ``mesh`` then sets ``Mesh.MeshSizeMax`` to its size.
    """

    GMSH_MODEL: ClassVar[str] = "advent_tree_mesh"
    """The name of the gmsh model ``mesh`` works in; gmsh allows a name used twice."""

    cone_count: int = 3
    """The number of stacked cones, at least 1."""

    cone_radius: float = 0.3
    """The radius of the base of the lowest cone."""

    cone_shrink: float = 0.65
    """The radius of the base of each cone over that of the cone below, in (0, 1]."""

    cone_taper: float = 0.4
    """
    The radius of the top of each cone over that of its base, in (0, 1).
    Not 1, since gmsh refuses a cone of two equal radii.
    """

    trunk_radius: float = 0.07
    """The radius of the trunk."""

    trunk_height: float = 0.15
    """The height of the trunk, which the lowest cone stands on."""

    star_size: float = 0.12
    """The width of the star, tip to tip; its height follows from the star's shape."""

    star_thickness: float = 0.03
    """The thickness of the star, perpendicular to its plane."""

    def __post_init__(self) -> None:
        """
        Refuse parameters that do not make one tree 1 high.

        :raises ValueError: If ``cone_count`` is not an integer of at least 1, a length
            is not finite and positive, ``cone_shrink`` is not in (0, 1],
            ``cone_taper`` is not in (0, 1), or the trunk and the star leave no height
            for the cones.
        """
        # The annotation says int, but a call can pass a float: range() then fails
        # later, and NaN and infinity pass an order test.
        if type(self.cone_count) is not int or self.cone_count < 1:
            raise ValueError(
                f"cone_count must be an integer of at least 1, got {self.cone_count}"
            )
        lengths = (
            ("cone_radius", self.cone_radius),
            ("trunk_radius", self.trunk_radius),
            ("trunk_height", self.trunk_height),
            ("star_size", self.star_size),
            ("star_thickness", self.star_thickness),
        )
        for name, value in lengths:
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if not 0.0 < self.cone_shrink <= 1.0:
            raise ValueError(f"cone_shrink must be in (0, 1], got {self.cone_shrink}")
        if not 0.0 < self.cone_taper < 1.0:
            raise ValueError(
                f"cone_taper must be in (0, 1), got {self.cone_taper}; gmsh refuses a "
                "cone of two equal radii"
            )
        if not self.trunk_height + self.star_height < 1.0:
            raise ValueError(
                "trunk_height and star_size leave no height for the cones, got "
                f"{self.trunk_height} and {self.star_size}, which need "
                f"{self.trunk_height + self.star_height} of the height 1"
            )

    @property
    def star_height(self) -> float:
        """The height of the star with its stem."""
        _, min_y, _, max_y = Star().build().scaled(self.star_size).bounds
        return max_y - min_y + self.STEM_HEIGHT * self.star_size

    @property
    def cone_height(self) -> float:
        """The height of each cone: what the trunk and the star leave, shared."""
        return (1.0 - self.trunk_height - self.star_height) / self.cone_count

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """
        The bounding box of the geometry, as ``(min_x, min_y, min_z, max_x, ...)``.

        The nodes of a mesh lie inside it, but a mesh of the curved parts is inscribed,
        so its own box can be smaller and off centre.
        """
        half_x = max(self.cone_radius, self.trunk_radius, 0.5 * self.star_size)
        half_y = max(self.cone_radius, self.trunk_radius, 0.5 * self.star_thickness)
        return (-half_x, -half_y, 0.0, half_x, half_y, 1.0)

    def mesh(self, *, size: float) -> TetrahedronMesh:
        """
        Return a tetrahedron mesh of the tree, made by gmsh.

        The mesh is of 3D Delaunay type. The cap on the target length of the
        tetrahedra, which gmsh does not enforce on their edges, is the argument. A
        cap above every radius gives the coarsest mesh. The mesh is conforming across
        the faces where the parts touch. The tetrahedra of the trunk and the cones
        are tagged ``BODY_TAG`` and those of the star ``STAR_TAG``.
        If gmsh is already initialised, the mesh is made in a model of its own, and
        the options outside ``GMSH_OPTIONS`` that the caller set, from a
        configuration file for one, still apply. Whatever the call finds is restored
        on return, also when it raises: whether gmsh is initialised, the current model
        and the options the call sets.

        :param size: The cap on the target size, in the units of the height 1.
        :returns: The mesh, with positively oriented tetrahedra.
        :raises ValueError: If ``size`` is not finite and positive, if gmsh fails, as
            it can for a thin cone and a coarse cap, if gmsh makes no tetrahedra in
            a volume, or if a tetrahedron it makes does not have positive
            orientation.
        """
        if not (math.isfinite(size) and size > 0.0):
            raise ValueError(f"size must be finite and positive, got {size}")

        try:
            with self._gmsh_workspace():
                gmsh.option.setNumber("Mesh.MeshSizeMax", size)
                body, star = self._add_parts()
                gmsh.model.mesh.generate(3)
                tags, coordinates, _ = gmsh.model.mesh.getNodes(
                    returnParametricCoord=False
                )
                volumes = (*body, *star)
                corners = [
                    gmsh.model.mesh.getElementsByType(self.GMSH_TETRAHEDRON, tag=v)[1]
                    for v in volumes
                ]
                parts = [self.BODY_TAG] * len(body) + [self.STAR_TAG] * len(star)
        # The gmsh API raises a bare Exception for a failure of OpenCASCADE or of the
        # mesher, so there is no narrower type to catch.
        except Exception as error:
            raise ValueError(
                f"gmsh failed on {self} with size={size}: {error}"
            ) from error

        empty = [v for v, c in zip(volumes, corners, strict=True) if c.size == 0]
        if empty:
            raise ValueError(
                f"gmsh made no tetrahedra in the volumes {empty} of {self} with "
                f"size={size}"
            )

        corner_tags = np.concatenate(corners)
        cell_tags = np.repeat(parts, [c.size // 4 for c in corners])
        used, inverse = np.unique(corner_tags, return_inverse=True)
        order = np.argsort(tags)
        points = coordinates.reshape(-1, 3)[order][np.searchsorted(tags[order], used)]
        return TetrahedronMesh.from_tetrahedra(
            points=points.T,
            tetrahedra=inverse.reshape(-1, 4).astype(np.int64),
            tetrahedron_tags=cell_tags.astype(np.int64),
        )

    def surface(self, *, size: float) -> SurfaceMesh:
        """
        Return the outer surface of the tree, the star included, for drawing.

        :param size: The cap on the target size, as for ``mesh``.
        :returns: The boundary faces of the mesh made with ``size``.
        :raises ValueError: As ``mesh`` does.
        """
        return self.mesh(size=size).surface()

    def star_part(self, *, size: float) -> TetrahedronMesh:
        """
        Return the mesh of the star alone, where it sits in the tree.

        The points and tetrahedra are those the star has in ``mesh`` with the same
        ``size``, with the points of the rest of the tree left out.

        :param size: The cap on the target size, as for ``mesh``.
        :returns: The mesh of the star, every tetrahedron tagged ``STAR_TAG``.
        :raises ValueError: As ``mesh`` does.
        """
        whole = self.mesh(size=size)
        star = whole.tetrahedra[whole.tetrahedron_tags == self.STAR_TAG]
        used, inverse = np.unique(star, return_inverse=True)
        return TetrahedronMesh.from_tetrahedra(
            points=whole.points[:, used],
            tetrahedra=inverse.reshape(-1, 4).astype(np.int64),
            tetrahedron_tags=np.full(len(star), self.STAR_TAG, dtype=np.int64),
        )

    def _star_outline(self) -> shapely.Polygon:
        """
        Return the star and its stem as a polygon in the plane of the star.

        :returns: The outline in the coordinates ``(x, z)``, its lowest point at the
            top of the highest cone and its highest point at 1.
        """
        star = Star().build().scaled(self.star_size).polygon
        stem_height = self.STEM_HEIGHT * self.star_size
        half_stem = 0.5 * self.STEM_WIDTH * self.star_size
        stem = shapely.box(-half_stem, star.bounds[1] - stem_height, half_stem, 0.0)
        outline = Shape.from_geometry(shapely.union_all([star, stem]))
        base = 1.0 - self.star_height
        return outline.translated(dx=0.0, dy=base - outline.bounds[1]).polygon

    def _add_parts(self) -> tuple[list[int], list[int]]:
        """
        Add the parts to the current gmsh model, fragmented where they touch.

        :returns: The tags of the volumes of the trunk and the cones, and those of the
            star.
        """
        occ = gmsh.model.occ
        trunk = occ.addCylinder(
            0.0, 0.0, 0.0, 0.0, 0.0, self.trunk_height, self.trunk_radius
        )
        body = [trunk]
        for level in range(self.cone_count):
            radius = self.cone_radius * self.cone_shrink**level
            body.append(
                occ.addCone(
                    0.0,
                    0.0,
                    self.trunk_height + level * self.cone_height,
                    0.0,
                    0.0,
                    self.cone_height,
                    radius,
                    self.cone_taper * radius,
                )
            )

        half_thickness = 0.5 * self.star_thickness
        corners = shapely.get_coordinates(self._star_outline().exterior)[:-1]
        points = [occ.addPoint(x, -half_thickness, z) for x, z in corners]
        lines = [
            occ.addLine(start, end)
            for start, end in zip(points, points[1:] + points[:1], strict=True)
        ]
        face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        extruded = occ.extrude([(self.DIM_FACE, face)], 0.0, self.star_thickness, 0.0)
        star = next(tag for dim, tag in extruded if dim == self.DIM_VOLUME)

        _, mapping = occ.fragment(
            [(self.DIM_VOLUME, tag) for tag in body], [(self.DIM_VOLUME, star)]
        )
        occ.synchronize()
        volumes = [[tag for _, tag in group] for group in mapping]
        return [tag for group in volumes[:-1] for tag in group], volumes[-1]

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
