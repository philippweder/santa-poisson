import math
from dataclasses import dataclass

import numpy as np
import shapely

from advent.shapes.shape import Shape


@dataclass(frozen=True, slots=True, kw_only=True)
class CandyCane:
    """A candy cane: a straight stick that ends in a half-circle hook."""

    hook_radius: float = 0.4
    """The radius of the hook's centre line, as a fraction of the stick's length."""

    thickness: float = 0.2
    """The width of the cane, as a fraction of the length of the stick."""

    quad_segs: int = 16
    """The number of segments that approximate a quarter of a circle."""

    def __post_init__(self) -> None:
        """
        Refuse a hook that the thickness of the cane would fill.

        :raises ValueError: If ``thickness`` is not positive, ``hook_radius`` is not
            above half of it, or ``quad_segs`` is below 1.
        """
        if not 0.0 < self.thickness < math.inf:
            raise ValueError(
                f"thickness must be positive and finite, got {self.thickness}"
            )
        if not 0.5 * self.thickness < self.hook_radius < math.inf:
            raise ValueError(
                f"hook_radius {self.hook_radius} must exceed half the thickness "
                f"{0.5 * self.thickness}"
            )
        if self.quad_segs < 1:
            raise ValueError(f"quad_segs must be at least 1, got {self.quad_segs}")

    def build(self) -> Shape:
        """
        Return the candy cane.

        :returns: The cane, standing upright with its hook to the right, its bounding
            box centred at the origin and its larger side 1.
        """
        angle = np.linspace(math.pi, 0.0, 2 * self.quad_segs + 1)
        hook = np.column_stack(
            [
                self.hook_radius + self.hook_radius * np.cos(angle),
                1.0 + self.hook_radius * np.sin(angle),
            ]
        )
        centre_line = shapely.LineString([(0.0, 0.0), *hook])
        outline = centre_line.buffer(
            0.5 * self.thickness,
            quad_segs=self.quad_segs,
            cap_style="round",
            join_style="mitre",
        )
        return Shape(polygon=outline).normalised()
