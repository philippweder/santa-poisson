from dataclasses import dataclass

import shapely
import shapely.geometry.base

from advent.shapes.shape import Shape


@dataclass(frozen=True, slots=True, kw_only=True)
class Candle:
    """
    A candle: a wax body, a wick and a teardrop flame, 1 high and centred at the origin.

    Heights are fractions of the height of the whole candle, widths too, since that
    height is 1 and the candle is the taller side.
    """

    body_width: float = 0.6
    """The width of the wax body, in (0, 1]."""

    flame_height: float = 0.3
    """The height of the flame, lowest point to tip; it must exceed ``flame_width``."""

    flame_width: float = 0.18
    """The width of the flame at its widest, in (0, 1], and wider than the wick."""

    wick_width: float = 0.03
    """The width of the wick; it must be below ``body_width`` and ``flame_width``."""

    wick_height: float = 0.07
    """The gap between the wax and the flame, which the wick spans."""

    quad_segs: int = 16
    """The number of segments that approximate a quarter of the flame's round base."""

    def __post_init__(self) -> None:
        """
        Refuse parameters that do not make one candle that is taller than wide.

        :raises ValueError: If a size is not positive, ``body_width`` or
            ``flame_width`` exceeds 1, ``flame_height`` is not above ``flame_width``,
            ``wick_width`` is not below ``body_width`` or ``flame_width``, the flame
            and the wick leave no room for the body, or ``quad_segs`` is below 1.
        """
        sizes = (
            ("body_width", self.body_width),
            ("flame_width", self.flame_width),
            ("wick_width", self.wick_width),
            ("wick_height", self.wick_height),
        )
        for name, value in sizes:
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.body_width > 1.0 or self.flame_width > 1.0:
            raise ValueError(
                "body_width and flame_width must not exceed the height 1, got "
                f"{self.body_width} and {self.flame_width}"
            )
        if not self.flame_height > self.flame_width:
            raise ValueError(
                f"flame_height {self.flame_height} must exceed flame_width "
                f"{self.flame_width}"
            )
        if not self.wick_width < self.body_width:
            raise ValueError(
                f"wick_width {self.wick_width} must be below body_width "
                f"{self.body_width}"
            )
        if not self.wick_width < self.flame_width:
            raise ValueError(
                f"wick_width {self.wick_width} must be below flame_width "
                f"{self.flame_width}"
            )
        if not self.flame_height + self.wick_height < 1.0:
            raise ValueError(
                "flame_height and wick_height leave no room for the body, got "
                f"{self.flame_height} and {self.wick_height}"
            )
        if self.quad_segs < 1:
            raise ValueError(f"quad_segs must be at least 1, got {self.quad_segs}")

    def build(self) -> Shape:
        """
        Return the candle with its flame.

        :returns: The candle, its bounding box centred at the origin and its height 1.
        """
        flame_bottom = 0.5 - self.flame_height
        wax_top = flame_bottom - self.wick_height
        half_body = 0.5 * self.body_width
        half_wick = 0.5 * self.wick_width
        body = shapely.box(-half_body, -0.5, half_body, wax_top)
        wick = shapely.box(
            -half_wick, wax_top, half_wick, flame_bottom + 0.5 * self.flame_width
        )
        return Shape.from_geometry(shapely.union_all([body, wick, self._flame()]))

    # housestyle: ignore[accessor-method] the design names the call, ``flame()``
    def flame(self) -> Shape:
        """
        Return the flame alone, where it sits in the candle.

        :returns: A teardrop, its tip at the top of the candle.
        """
        return Shape.from_geometry(self._flame())

    def _flame(self) -> shapely.geometry.base.BaseGeometry:
        """
        Return the flame outline: the convex hull of a disc and the tip above it.

        :returns: The outline, a polygon.
        """
        radius = 0.5 * self.flame_width
        centre_y = 0.5 - self.flame_height + radius
        disc = shapely.Point(0.0, centre_y).buffer(radius, quad_segs=self.quad_segs)
        return shapely.union_all([disc, shapely.Point(0.0, 0.5)]).convex_hull
