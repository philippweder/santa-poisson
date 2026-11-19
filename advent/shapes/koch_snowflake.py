import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import shapely
from numpy.typing import NDArray

from advent.shapes.shape import Shape


@dataclass(frozen=True, slots=True, kw_only=True)
class KochSnowflake:
    """The Koch snowflake: a triangle whose sides grow bumps, level by level."""

    MAX_LEVELS: ClassVar[int] = 6
    """
    The largest ``levels``; the boundary then has ``3 * 4**6`` sides.

    A mesh sized to the shortest side has nine times the triangles per level, so a
    higher level makes a mesh too large for a day.
    """

    levels: int = 3
    """The number of times the middle third of every side is replaced by two sides."""

    def __post_init__(self) -> None:
        """
        Refuse a number of levels outside 0 to ``MAX_LEVELS``.

        :raises ValueError: If ``levels`` is negative or above ``MAX_LEVELS``.
        """
        if not 0 <= self.levels <= self.MAX_LEVELS:
            raise ValueError(
                f"levels must be between 0 and {self.MAX_LEVELS}, got {self.levels}"
            )

    def build(self) -> Shape:
        """
        Return the snowflake.

        The triangle is equilateral and its apex points up. The bumps stand on the
        middle third of each side and point outwards, and each has sides of that
        third's length. The boundary has ``3 * 4**levels`` sides.

        :returns: The snowflake, its bounding box centred at the origin and its larger
            side 1.
        """
        half_side = 0.5 * math.sqrt(3.0)
        vertices = np.array([[0.0, 1.0], [-half_side, -0.5], [half_side, -0.5]])
        for _ in range(self.levels):
            vertices = self._refined(vertices)
        return Shape(polygon=shapely.Polygon(vertices)).normalised()

    @staticmethod
    def _refined(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Return the corners of the polygon after every side has grown a bump.

        A side from ``p`` to ``q``, with ``d = q - p``, becomes ``p``, ``p + d / 3``,
        the tip, ``p + 2 d / 3``. The tip is the midpoint of the side moved by
        ``sqrt(3) / 6`` of ``d`` turned a quarter turn clockwise, which is outwards for
        a counter-clockwise polygon, and which makes the bump equilateral.

        :param vertices: The corners of a counter-clockwise polygon, shape ``(m, 2)``.
        :returns: The corners of the refined polygon, shape ``(4 * m, 2)``.
        """
        step = np.roll(vertices, -1, axis=0) - vertices
        outwards = np.stack([step[:, 1], -step[:, 0]], axis=1)
        tip = vertices + 0.5 * step + (math.sqrt(3.0) / 6.0) * outwards
        return np.stack(
            [vertices, vertices + step / 3.0, tip, vertices + 2.0 * step / 3.0], axis=1
        ).reshape(-1, 2)
