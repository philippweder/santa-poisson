import itertools
import math
from dataclasses import dataclass
from typing import ClassVar

import shapely
import shapely.affinity

from advent.shapes.shape import Shape


@dataclass(frozen=True, slots=True, kw_only=True)
class Noel:
    """
    The letters N, O, E and L, side by side in one frame, as four shapes.

    Each letter is 1 high before the common frame is applied: heights, widths and the
    gap are fractions of that height. N, E and L are straight-sided and
    ``WIDTH`` wide, and the O is a ring that is 1 wide, between two regular polygons.
    """

    WIDTH: ClassVar[float] = 0.6
    """The width of the N, the E and the L, as a fraction of the letter height."""

    MAX_STROKE: ClassVar[float] = 0.5 * WIDTH
    """
    The stroke at and above which ``__post_init__`` refuses.

    The N needs ``WIDTH > 2 * stroke`` for room between its bars, the strictest of
    the letters: the E needs ``3 * stroke < 1``, the O ``2 * stroke < 1``.
    """

    stroke: float = 0.18
    """The width of a stroke of every letter, as a fraction of the letter height."""

    gap: float = 0.12
    """The space between neighbouring letters, as a fraction of the letter height."""

    quad_segs: int = 16
    """The number of segments that approximate a quarter of each circle of the O."""

    def __post_init__(self) -> None:
        """
        Refuse sizes that do not make four separate letters, each one polygon.

        :raises ValueError: If ``stroke`` is not positive, finite and below
            ``MAX_STROKE``, ``gap`` is not positive and finite, or ``quad_segs`` is
            below 1.
        """
        if not 0.0 < self.stroke < math.inf:
            raise ValueError(f"stroke must be positive and finite, got {self.stroke}")
        if not self.stroke < self.MAX_STROKE:
            raise ValueError(
                f"stroke must be below {self.MAX_STROKE}, so that the N keeps room "
                f"between its bars, got {self.stroke}"
            )
        if not 0.0 < self.gap < math.inf:
            raise ValueError(f"gap must be positive and finite, got {self.gap}")
        if self.quad_segs < 1:
            raise ValueError(f"quad_segs must be at least 1, got {self.quad_segs}")

    def letters(self) -> tuple[Shape, Shape, Shape, Shape]:
        """
        Return the shapes of N, O, E and L in reading order.

        The four are placed left to right, ``gap`` apart, and share one frame: the
        bounding box of the four together is centred at the origin and its larger
        side is 1. The O has one hole and the others none.

        :returns: The shapes of N, O, E and L.
        :raises ValueError: If ``gap`` is so small against the width of the four that
            two neighbouring letters touch after rounding.
        """
        polygons: list[shapely.Polygon] = []
        left = 0.0
        for letter in (self._n(), self._o(), self._e(), self._l()):
            min_x, _, max_x, _ = letter.bounds
            polygons.append(shapely.affinity.translate(letter, xoff=left - min_x))
            left += max_x - min_x + self.gap
        for one, two in itertools.pairwise(polygons):
            if not one.distance(two) > 0.0:
                raise ValueError(
                    f"the letters touch: gap must be above the rounding of the layout, "
                    f"got {self.gap}"
                )
        min_x, min_y, max_x, max_y = shapely.union_all(polygons).bounds
        shift_x, shift_y = -0.5 * (min_x + max_x), -0.5 * (min_y + max_y)
        factor = 1.0 / max(max_x - min_x, max_y - min_y)
        n, o, e, ell = (
            Shape(polygon=polygon).translated(dx=shift_x, dy=shift_y).scaled(factor)
            for polygon in polygons
        )
        return (n, o, e, ell)

    def _n(self) -> shapely.Polygon:
        """
        Return the N: two bars and a diagonal from the top left to the bottom right.

        The diagonal has the horizontal thickness ``stroke``, so that its band meets
        the bars at the vertical thickness ``stroke / (WIDTH - stroke)``.

        :returns: The polygon, in the box from 0 to ``WIDTH`` and 0 to 1.
        """
        s, w = self.stroke, self.WIDTH
        thickness = s / (w - s)
        return shapely.Polygon(
            [
                (0.0, 0.0),
                (s, 0.0),
                (s, 1.0 - thickness),
                (w - s, 0.0),
                (w, 0.0),
                (w, 1.0),
                (w - s, 1.0),
                (w - s, thickness),
                (s, 1.0),
                (0.0, 1.0),
            ]
        )

    def _o(self) -> shapely.Polygon:
        """
        Return the O: a ring between two regular polygons of ``4 * quad_segs`` sides.

        :returns: The polygon, in the box from 0 to 1 in both axes.
        """
        centre = shapely.Point(0.5, 0.5)
        outer = centre.buffer(0.5, quad_segs=self.quad_segs)
        hole = centre.buffer(0.5 - self.stroke, quad_segs=self.quad_segs)
        return shapely.Polygon(outer.exterior, [hole.exterior])

    def _e(self) -> shapely.Polygon:
        """
        Return the E: a bar and three arms of equal length, the middle one centred.

        :returns: The polygon, in the box from 0 to ``WIDTH`` and 0 to 1.
        """
        s, w = self.stroke, self.WIDTH
        low, high = 0.5 - 0.5 * s, 0.5 + 0.5 * s
        return shapely.Polygon(
            [
                (0.0, 0.0),
                (w, 0.0),
                (w, s),
                (s, s),
                (s, low),
                (w, low),
                (w, high),
                (s, high),
                (s, 1.0 - s),
                (w, 1.0 - s),
                (w, 1.0),
                (0.0, 1.0),
            ]
        )

    def _l(self) -> shapely.Polygon:
        """
        Return the L: a bar and a foot.

        :returns: The polygon, in the box from 0 to ``WIDTH`` and 0 to 1.
        """
        s, w = self.stroke, self.WIDTH
        return shapely.Polygon(
            [(0.0, 0.0), (w, 0.0), (w, s), (s, s), (s, 1.0), (0.0, 1.0)]
        )
