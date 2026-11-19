import math
from dataclasses import dataclass
from itertools import pairwise

import shapely

from advent.shapes.shape import Shape


@dataclass(frozen=True, slots=True, kw_only=True)
class Snowman:
    """A snowman: three stacked discs, by default each smaller than the one below."""

    middle_ratio: float = 0.8
    """The radius of the middle disc over the radius of the bottom disc."""

    head_ratio: float = 0.6
    """The radius of the head over the radius of the bottom disc."""

    overlap: float = 0.5
    """
    How far neighbouring discs reach into each other, in (0, 1).

    The distance between two centres is the sum of the two radii times ``1 - overlap``.
    """

    quad_segs: int = 16
    """The number of segments that approximate a quarter of a circle."""

    def __post_init__(self) -> None:
        """
        Refuse radii that are not positive and discs that do not overlap.

        :raises ValueError: If a ratio is not positive, ``overlap`` is not in (0, 1) or
            ``quad_segs`` is below 1.
        """
        if not 0.0 < self.middle_ratio < math.inf:
            raise ValueError(
                f"middle_ratio must be positive and finite, got {self.middle_ratio}"
            )
        if not 0.0 < self.head_ratio < math.inf:
            raise ValueError(
                f"head_ratio must be positive and finite, got {self.head_ratio}"
            )
        if not 0.0 < self.overlap < 1.0:
            raise ValueError(f"overlap must be in (0, 1), got {self.overlap}")
        if self.quad_segs < 1:
            raise ValueError(f"quad_segs must be at least 1, got {self.quad_segs}")

    def build(self) -> Shape:
        """
        Return the snowman.

        :returns: The snowman, its bounding box centred at the origin and its larger
            side 1.
        """
        radii = (1.0, self.middle_ratio, self.head_ratio)
        centres = [0.0]
        for below, above in pairwise(radii):
            centres.append(centres[-1] + (below + above) * (1.0 - self.overlap))

        discs = [
            shapely.Point(0.0, centre).buffer(radius, quad_segs=self.quad_segs)
            for centre, radius in zip(centres, radii, strict=True)
        ]
        return Shape.from_geometry(shapely.union_all(discs)).normalised()
