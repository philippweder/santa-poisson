from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

# housestyle: ignore[type-checking-block] there is no cycle: dolfinx and mpi4py are
# named only in annotations, and importing them here would load both, and start MPI,
# with every import of this module.
if TYPE_CHECKING:
    import dolfinx.mesh
    from mpi4py import MPI


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class TriangleMesh:
    """
    A triangle mesh of a planar region, with its boundary edges.

    Equality is identity: the generated ``__eq__`` of a dataclass on arrays is
    ambiguous and its generated hash would fail. The arrays are read-only copies of the
    arguments. ``matplotlib.tri.Triangulation(*mesh.points, mesh.triangles)`` accepts
    the mesh directly.
    """

    GDIM: ClassVar[int] = 2
    """The number of coordinates of a point: the mesh lies in a plane."""

    TABLE_NDIM: ClassVar[int] = 2
    """The number of axes of each array: every array is a table."""

    TRIANGLE_VERTICES: ClassVar[int] = 3
    """The number of vertices of a triangle."""

    EDGE_VERTICES: ClassVar[int] = 2
    """The number of vertices of an edge."""

    points: NDArray[np.float64]
    """The vertex coordinates, shape ``(2, n)``, finite."""

    triangles: NDArray[np.int64]
    """The vertex indices of each triangle, shape ``(m, 3)``, counter-clockwise."""

    boundary_edges: NDArray[np.int64]
    """The vertex indices of each boundary edge, shape ``(k, 2)``."""

    def __post_init__(self) -> None:
        """
        Check the arrays and replace them by read-only copies.

        :raises ValueError: If an array has the wrong shape, an index array is not of
            integer dtype, a point is not finite, an index is out of range, or a
            triangle is not counter-clockwise.
        """
        points = np.array(self.points, dtype=np.float64)
        if points.ndim != self.TABLE_NDIM or points.shape[0] != self.GDIM:
            raise ValueError(
                f"points must have shape ({self.GDIM}, n), got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")

        triangles = self._index_array(
            self.triangles, "triangles", self.TRIANGLE_VERTICES
        )
        edges = self._index_array(
            self.boundary_edges, "boundary_edges", self.EDGE_VERTICES
        )
        count = points.shape[1]
        for name, indices in (("triangles", triangles), ("boundary_edges", edges)):
            if indices.size > 0 and (indices.min() < 0 or indices.max() >= count):
                raise ValueError(
                    f"{name} must index the {count} points, got indices from "
                    f"{indices.min()} to {indices.max()}"
                )

        corners = points[:, triangles]
        first = corners[:, :, 1] - corners[:, :, 0]
        second = corners[:, :, 2] - corners[:, :, 0]
        area = 0.5 * (first[0] * second[1] - first[1] * second[0])
        if not np.all(area > 0.0):
            bad = int(np.flatnonzero(~(area > 0.0))[0])
            raise ValueError(
                f"triangles must be counter-clockwise; triangle {bad} has signed area "
                f"{area[bad]}"
            )

        for name, array in (
            ("points", points),
            ("triangles", triangles),
            ("boundary_edges", edges),
        ):
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    @staticmethod
    def _index_array(
        values: NDArray[np.int64], name: str, width: int
    ) -> NDArray[np.int64]:
        """
        Return a copy of an index array after checking its dtype kind and shape.

        :param values: The array to copy.
        :param name: The field name, for the message.
        :param width: The required number of columns.
        :returns: The copy, as int64.
        :raises ValueError: If ``values`` is not an integer array of shape
            ``(rows, width)``.
        """
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be integers, got dtype {array.dtype}")
        if array.ndim != TriangleMesh.TABLE_NDIM or array.shape[1] != width:
            raise ValueError(f"{name} must have shape (m, {width}), got {array.shape}")
        return np.array(array, dtype=np.int64)

    def to_dolfinx(self, comm: "MPI.Comm") -> "dolfinx.mesh.Mesh[np.float64]":
        """
        Return the mesh as a dolfinx 0.11 mesh of first-order triangles.

        The cells and points are passed on rank 0 only, and dolfinx distributes them.
        Other dolfinx versions are not supported.

        :param comm: The MPI communicator the mesh lives on.
        :returns: The mesh, with geometric dimension 2.
        """
        # Deferred: dolfinx is heavy, and not every day needs it.
        import basix.ufl  # noqa: PLC0415 deferred heavy dependency, see above
        import dolfinx.mesh  # noqa: PLC0415 deferred heavy dependency, see above
        import ufl  # noqa: PLC0415 deferred heavy dependency, see above

        element = basix.ufl.element(
            "Lagrange", "triangle", 1, shape=(self.GDIM,), dtype=np.float64
        )
        if comm.rank == 0:
            cells, coordinates = self.triangles, self.points.T
        else:
            cells = np.empty((0, self.TRIANGLE_VERTICES), dtype=np.int64)
            coordinates = np.empty((0, self.GDIM), dtype=np.float64)
        # The signature of create_mesh leaves the mesh's scalar type argument open.
        return dolfinx.mesh.create_mesh(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] dolfinx annotation
            comm, cells, ufl.Mesh(element), coordinates
        )
