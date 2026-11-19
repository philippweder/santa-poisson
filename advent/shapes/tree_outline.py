import math
from dataclasses import dataclass

import shapely

from advent.shapes.shape import Shape


@dataclass(frozen=True, slots=True, kw_only=True)
class TreeOutline:
    """A tree outline: stacked triangles on a trunk, the upper ones narrower."""

    tiers: int = 3
    """The number of stacked triangles."""

    taper: float = 0.75
    """The width of a triangle over the width of the one below it, in (0, 1]."""

    overlap: float = 0.3
    """
    How far a triangle reaches down into the one below, in (0, 1).

    The value is a fraction of the height of the lower triangle, so a triangle starts
    at ``1 - overlap`` of that height.
    """

    tier_aspect: float = 0.8
    """The height of a triangle over its width."""

    trunk_width: float = 0.2
    """The width of the trunk over the width of the lowest triangle, in (0, 1]."""

    trunk_height: float = 0.15
    """The height of the trunk over the width of the lowest triangle."""

    def __post_init__(self) -> None:
        """
        Refuse parameters that do not make one stack of triangles on a trunk.

        :raises ValueError: If ``tiers`` is below 1, ``taper`` or ``trunk_width`` is not
            in (0, 1], ``overlap`` is not in (0, 1), or ``tier_aspect`` or
            ``trunk_height`` is not positive.
        """
        if self.tiers < 1:
            raise ValueError(f"tiers must be at least 1, got {self.tiers}")
        if not 0.0 < self.taper <= 1.0:
            raise ValueError(f"taper must be in (0, 1], got {self.taper}")
        if not 0.0 < self.overlap < 1.0:
            raise ValueError(f"overlap must be in (0, 1), got {self.overlap}")
        if not 0.0 < self.tier_aspect < math.inf:
            raise ValueError(
                f"tier_aspect must be positive and finite, got {self.tier_aspect}"
            )
        if not 0.0 < self.trunk_width <= 1.0:
            raise ValueError(f"trunk_width must be in (0, 1], got {self.trunk_width}")
        if not 0.0 < self.trunk_height < math.inf:
            raise ValueError(
                f"trunk_height must be positive and finite, got {self.trunk_height}"
            )

    def build(self) -> Shape:
        """
        Return the tree outline.

        :returns: The outline, its bounding box centred at the origin and its larger
            side 1.
        :raises ValueError: If the parts do not make one polygon, as for a very large
            ``tiers``, where the height of an upper tier is lost in the rounding of
            its base.
        """
        half_trunk = 0.5 * self.trunk_width
        parts = [
            shapely.box(-half_trunk, 0.0, half_trunk, self.trunk_height),
        ]
        base = self.trunk_height
        for tier in range(self.tiers):
            width = self.taper**tier
            height = self.tier_aspect * width
            parts.append(
                shapely.Polygon(
                    [
                        (-0.5 * width, base),
                        (0.5 * width, base),
                        (0.0, base + height),
                    ]
                )
            )
            base += (1.0 - self.overlap) * height
        return Shape.from_geometry(shapely.union_all(parts)).normalised()
