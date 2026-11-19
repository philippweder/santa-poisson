from pathlib import Path
from typing import ClassVar

import matplotlib
import matplotlib.style
from matplotlib.figure import Figure

from advent.style.colormap import Colormap
from advent.style.figure_size import FigureSize


class Style:
    """The look of every figure: colour maps, matplotlib settings and figure sizes."""

    DPI: ClassVar[int] = 200
    """Resolution of every figure; ``advent.mplstyle`` sets ``savefig.dpi`` to it."""

    WIDTH_PX: ClassVar[int] = 1600
    """Width in pixels of a saved figure, every ``FigureSize`` being 8.0 in wide."""

    MPLSTYLE: ClassVar[Path] = Path(__file__).with_name("advent.mplstyle")
    """The matplotlib style file that ``use`` applies."""

    @classmethod
    def use(cls) -> None:
        """
        Register both colour maps and apply ``advent.mplstyle``.

        A map already registered under its name and equal to the member's is left
        alone, so calling it again changes nothing and warns of nothing.
        """
        for member in Colormap:
            colormap = member.colormap
            if matplotlib.colormaps.get(member.value) != colormap:
                matplotlib.colormaps.register(colormap, name=member.value, force=True)
        matplotlib.style.use(cls.MPLSTYLE)

    @classmethod
    def figure(cls, size: FigureSize = FigureSize.STANDARD) -> Figure:
        """
        Return a new figure that pyplot does not manage.

        A figure built here is not registered with pyplot and needs no backend.

        :param size: The size of the figure.
        :returns: The figure, ``size.value[0]`` inches wide, at ``DPI`` dots per inch,
            with constrained layout.
        """
        return Figure(figsize=size.value, dpi=cls.DPI, layout="constrained")
