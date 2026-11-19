from enum import StrEnum

from matplotlib.colors import LinearSegmentedColormap

from advent.style.palette import Palette


class Colormap(StrEnum):
    """
    The sequential colour maps of the calendar; each value is the registered name.

    Both start at ``Palette.NIGHT_BLUE`` and end at ``Palette.SNOW_WHITE``. Their stop
    positions are proportional to the CIELAB lightness of the stops, and lightness
    rises monotonically along each map.
    """

    WARM = "advent-warm"
    COLD = "advent-cold"

    @property
    def colormap(self) -> LinearSegmentedColormap:
        """
        Return the colour map of this member, with 256 entries.

        :returns: A new ``LinearSegmentedColormap`` named like this member.
        """
        match self:
            case Colormap.WARM:
                stops = [
                    (0.00, Palette.NIGHT_BLUE),
                    (0.28, Palette.DEEP_RED),
                    (0.69, Palette.GOLD),
                    (1.00, Palette.SNOW_WHITE),
                ]
            case Colormap.COLD:
                stops = [
                    (0.00, Palette.NIGHT_BLUE),
                    (0.52, "#5B8DB8"),
                    (1.00, Palette.SNOW_WHITE),
                ]
        return LinearSegmentedColormap.from_list(self.value, stops, N=256)
