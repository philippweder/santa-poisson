from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from advent.solids.surface_mesh import SurfaceMesh


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class TetrahedronMesh:
    """
    A tetrahedron mesh of a solid, with its boundary faces and a tag for each cell.

    Equality is identity, as for ``TriangleMesh``: the generated ``__eq__`` of a
    dataclass on arrays is ambiguous. The arrays are read-only copies of the arguments.
    A tetrahedron ``(a, b, c, d)`` has positive orientation when the signed volume
    ``det[b - a, c - a, d - a] / 6`` is positive, the order of dolfinx's reference
    tetrahedron.
    """

    GDIM: ClassVar[int] = 3
    """The number of coordinates of a point: the mesh lies in space."""

    TABLE_NDIM: ClassVar[int] = 2
    """The number of axes of each table array."""

    TETRAHEDRON_VERTICES: ClassVar[int] = 4
    """The number of vertices of a tetrahedron."""

    FACE_VERTICES: ClassVar[int] = 3
    """The number of vertices of a face."""

    VOLUME_FACTOR: ClassVar[float] = 6.0
    """The signed volume of a tetrahedron is the determinant of its edges over this."""

    points: NDArray[np.float64]
    """The vertex coordinates, shape ``(3, n)``, finite."""

    tetrahedra: NDArray[np.int64]
    """The vertex indices of each tetrahedron, shape ``(m, 4)``, positively oriented."""

    boundary_faces: NDArray[np.int64]
    """
    The vertex indices of each boundary face, shape ``(k, 3)``.
    A face of exactly one tetrahedron, counter-clockwise seen from outside.
    """

    tetrahedron_tags: NDArray[np.int64]
    """The tag of the part each tetrahedron belongs to, shape ``(m,)``."""

    def __post_init__(self) -> None:
        """
        Check the arrays and replace them by read-only copies.

        :raises ValueError: If an array has the wrong shape, an index or tag array is
            not of integer dtype, a point is not finite, an index is out of range, or a
            tetrahedron does not have positive orientation.
        """
        points = np.array(self.points, dtype=np.float64)
        if points.ndim != self.TABLE_NDIM or points.shape[0] != self.GDIM:
            raise ValueError(
                f"points must have shape ({self.GDIM}, n), got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")

        tetrahedra = self._integers(
            self.tetrahedra, "tetrahedra", (-1, self.TETRAHEDRON_VERTICES)
        )
        faces = self._integers(
            self.boundary_faces, "boundary_faces", (-1, self.FACE_VERTICES)
        )
        tags = self._integers(self.tetrahedron_tags, "tetrahedron_tags", (-1,))
        count = points.shape[1]
        for name, indices in (("tetrahedra", tetrahedra), ("boundary_faces", faces)):
            if indices.size > 0 and (indices.min() < 0 or indices.max() >= count):
                raise ValueError(
                    f"{name} must index the {count} points, got indices from "
                    f"{indices.min()} to {indices.max()}"
                )
        if tags.shape[0] != tetrahedra.shape[0]:
            raise ValueError(
                f"tetrahedron_tags must have one tag for each of the "
                f"{tetrahedra.shape[0]} tetrahedra, got {tags.shape[0]}"
            )

        corners = points[:, tetrahedra]
        edges = [corners[:, :, k] - corners[:, :, 0] for k in (1, 2, 3)]
        volume = np.sum(np.cross(edges[0], edges[1], axis=0) * edges[2], axis=0)
        volume /= self.VOLUME_FACTOR
        if not np.all(volume > 0.0):
            bad = int(np.flatnonzero(~(volume > 0.0))[0])
            raise ValueError(
                f"tetrahedra must have positive orientation; tetrahedron {bad} has "
                f"signed volume {volume[bad]}"
            )

        for name, array in (
            ("points", points),
            ("tetrahedra", tetrahedra),
            ("boundary_faces", faces),
            ("tetrahedron_tags", tags),
        ):
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    @staticmethod
    def _integers(
        values: NDArray[np.int64], name: str, shape: Sequence[int]
    ) -> NDArray[np.int64]:
        """
        Return a copy of an integer array after checking its dtype kind and extent.

        :param values: The array to copy.
        :param name: The field name, for the message.
        :param shape: The required shape, with -1 for a free length.
        :returns: The copy, as int64.
        :raises ValueError: If ``values`` is not an integer array of the required
            extent.
        """
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be integers, got dtype {array.dtype}")
        if array.ndim != len(shape) or any(
            want not in (-1, got) for want, got in zip(shape, array.shape, strict=True)
        ):
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        return np.array(array, dtype=np.int64)

    @classmethod
    def from_tetrahedra(
        cls,
        *,
        points: NDArray[np.float64],
        tetrahedra: NDArray[np.int64],
        tetrahedron_tags: NDArray[np.int64],
    ) -> Self:
        """
        Return the mesh of the given cells, with its boundary faces found.

        A boundary face is a face of exactly one cell, directed outward as in that
        cell. For a positive tetrahedron ``(a, b, c, d)`` the faces opposite ``d``,
        ``c``, ``b`` and ``a`` wound ``(a, c, b)``, ``(a, b, d)``, ``(a, d, c)`` and
        ``(b, c, d)`` have their right-hand normal pointing out.

        :param points: The vertex coordinates, shape ``(3, n)``.
        :param tetrahedra: The vertex indices of the tetrahedra, shape ``(m, 4)``,
            positive orientation.
        :param tetrahedron_tags: The part of each tetrahedron, shape ``(m,)``.
        :returns: The mesh.
        :raises ValueError: As ``__init__`` does for these arrays.
        """
        cells = cls._integers(tetrahedra, "tetrahedra", (-1, cls.TETRAHEDRON_VERTICES))
        directed = np.concatenate(
            [
                cells[:, [0, 2, 1]],
                cells[:, [0, 1, 3]],
                cells[:, [0, 3, 2]],
                cells[:, 1:],
            ]
        )
        _, inverse, counts = np.unique(
            np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
        )
        return cls(
            points=points,
            tetrahedra=cells,
            boundary_faces=directed[counts[inverse.ravel()] == 1],
            tetrahedron_tags=tetrahedron_tags,
        )

    def surface(self) -> SurfaceMesh:
        """
        Return the boundary faces as a surface mesh of its own.

        :returns: The surface, with only the points that a boundary face uses, in the
            order of this mesh, and the faces renumbered to them.
        """
        used, inverse = np.unique(self.boundary_faces, return_inverse=True)
        return SurfaceMesh(
            points=self.points[:, used],
            triangles=inverse.reshape(-1, self.FACE_VERTICES).astype(np.int64),
        )
