import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import shapely

from advent.shapes.shape import Shape


@dataclass(frozen=True, slots=True, kw_only=True)
class Snowflake:
    """A six-fold snowflake: six straight arms, each with pairs of side branches."""

    ARMS: ClassVar[int] = 6
    """The number of arms, evenly spaced; the first points along the first axis."""

    BRANCH_ANGLE: ClassVar[float] = math.pi / 3.0
    """The angle between a side branch and its arm, pointing outwards."""

    BRANCH_START: ClassVar[float] = 0.45
    """Where the first pair of branches leaves an arm, as a fraction of its length."""

    BRANCH_END: ClassVar[float] = 0.85
    """Where the last pair of branches leaves an arm, as a fraction of its length."""

    branches: int = 2
    """The number of pairs of side branches on each arm, spread from start to end."""

    branch_length: float = 0.3
    """The length of a side branch, as a fraction of the length of an arm."""

    arm_width: float = 0.06
    """The width of arms and branches, as a fraction of the length of an arm."""

    def __post_init__(self) -> None:
        """
        Refuse a negative number of branches and lengths that are not positive.

        :raises ValueError: If ``branches`` is negative, or ``branch_length`` or
            ``arm_width`` is not positive and finite.
        """
        if self.branches < 0:
            raise ValueError(f"branches must not be negative, got {self.branches}")
        if not 0.0 < self.branch_length < math.inf:
            raise ValueError(
                f"branch_length must be positive and finite, got {self.branch_length}"
            )
        if not 0.0 < self.arm_width < math.inf:
            raise ValueError(
                f"arm_width must be positive and finite, got {self.arm_width}"
            )

    def build(self) -> Shape:
        """
        Return the snowflake.

        :returns: The snowflake, its bounding box centred at the origin and its larger
            side 1.
        :raises ValueError: If the branches of neighbouring arms cross, which encloses
            holes, so that the snowflake is no longer one solid polygon.
        """
        origin = np.zeros(2)
        starts = np.linspace(self.BRANCH_START, self.BRANCH_END, self.branches)
        lines: list[shapely.LineString] = []
        for arm in range(self.ARMS):
            angle = 2.0 * math.pi * arm / self.ARMS
            along = np.array([math.cos(angle), math.sin(angle)])
            across = np.array([-along[1], along[0]])
            lines.append(shapely.LineString([origin, along]))
            for start in starts:
                for side in (1.0, -1.0):
                    direction = (
                        math.cos(self.BRANCH_ANGLE) * along
                        + side * math.sin(self.BRANCH_ANGLE) * across
                    )
                    root = start * along
                    lines.append(
                        shapely.LineString(
                            [root, root + self.branch_length * direction]
                        )
                    )

        skeleton = shapely.union_all(lines)
        outline = skeleton.buffer(
            0.5 * self.arm_width, cap_style="flat", join_style="mitre"
        )
        if len(outline.interiors) > 0:
            raise ValueError(
                "the branches of neighbouring arms cross for branch_length="
                f"{self.branch_length} and arm_width={self.arm_width}; lower "
                "branch_length or arm_width"
            )
        return Shape(polygon=outline).normalised()
