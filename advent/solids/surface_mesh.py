from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SurfaceMesh:
    """
    A triangle mesh of a closed surface in space, for drawing.

    Equality is identity, as for ``TriangleMesh``. The arrays are read-only copies of
    the arguments. ``pyvista.PolyData.from_regular_faces(mesh.points.T,
    mesh.triangles)`` accepts the mesh directly.
    """

    GDIM: ClassVar[int] = 3
    """The number of coordinates of a point: the surface lies in space."""

    TABLE_NDIM: ClassVar[int] = 2
    """The number of axes of each array: every array is a table."""

    TRIANGLE_VERTICES: ClassVar[int] = 3
    """The number of vertices of a triangle."""

    points: NDArray[np.float64]
    """The vertex coordinates, shape ``(3, n)``, finite."""

    triangles: NDArray[np.int64]
    """
    The vertex indices of each triangle, shape ``(m, 3)``.
    Counter-clockwise seen from outside, so the right-hand normal points out.
    """

    def __post_init__(self) -> None:
        """
        Check the arrays and replace them by read-only copies.

        :raises ValueError: If an array has the wrong shape, ``triangles`` is not of
            integer dtype, a point is not finite, or an index is out of range.
        """
        points = np.array(self.points, dtype=np.float64)
        if points.ndim != self.TABLE_NDIM or points.shape[0] != self.GDIM:
            raise ValueError(
                f"points must have shape ({self.GDIM}, n), got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")

        values = np.asarray(self.triangles)
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"triangles must be integers, got dtype {values.dtype}")
        if values.ndim != self.TABLE_NDIM or values.shape[1] != self.TRIANGLE_VERTICES:
            raise ValueError(
                f"triangles must have shape (m, {self.TRIANGLE_VERTICES}), got "
                f"{values.shape}"
            )
        triangles = np.array(values, dtype=np.int64)
        count = points.shape[1]
        if triangles.size > 0 and (triangles.min() < 0 or triangles.max() >= count):
            raise ValueError(
                f"triangles must index the {count} points, got indices from "
                f"{triangles.min()} to {triangles.max()}"
            )

        for name, array in (("points", points), ("triangles", triangles)):
            array.flags.writeable = False
            object.__setattr__(self, name, array)
