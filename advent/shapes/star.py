import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import shapely

from advent.shapes.shape import Shape


@dataclass(frozen=True, slots=True, kw_only=True)
class Star:
    """A star with its tips on a circle and its notches on a smaller one."""

    MIN_TIPS: ClassVar[int] = 3
    """The fewest tips that make a star."""

    tips: int = 5
    """The number of tips; the first points up."""

    inner_ratio: float = 0.5 * (3.0 - math.sqrt(5.0))
    """
    The radius of the notches over the radius of the tips, in (0, 1).

    The default makes the edges of the five-point star straight lines of a pentagram.
    """

    def __post_init__(self) -> None:
        """
        Refuse a star with too few tips or a notch circle that is not inside.

        :raises ValueError: If ``tips`` is below 3 or ``inner_ratio`` is not in (0, 1).
        """
        if self.tips < self.MIN_TIPS:
            raise ValueError(f"tips must be at least {self.MIN_TIPS}, got {self.tips}")
        if not 0.0 < self.inner_ratio < 1.0:
            raise ValueError(f"inner_ratio must be in (0, 1), got {self.inner_ratio}")

    def build(self) -> Shape:
        """
        Return the star.

        :returns: The star, its bounding box centred at the origin and its larger
            side 1.
        """
        vertex = np.arange(2 * self.tips)
        angle = 0.5 * math.pi + math.pi * vertex / self.tips
        radius = np.where(vertex % 2 == 0, 1.0, self.inner_ratio)
        corners = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        return Shape(polygon=shapely.Polygon(corners)).normalised()
