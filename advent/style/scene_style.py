from typing import ClassVar

from pyvista.plotting.themes import Theme

from advent.style.colormap import Colormap
from advent.style.palette import Palette
from advent.style.style import Style


class SceneStyle:
    """The look of every 3D scene: a pyvista theme in the palette of the calendar."""

    WINDOW_SIZE: ClassVar[tuple[int, int]] = (1600, 900)
    """Width and height in pixels of the plotter window that a theme sets."""

    @classmethod
    def theme(cls) -> Theme:
        """
        Return a new pyvista theme with the colours of the calendar.

        The background is ``Palette.SNOW_WHITE``, the font colour is
        ``Palette.NIGHT_BLUE``, the colour map is ``Colormap.WARM`` and the window is
        ``WINDOW_SIZE`` pixels. The colour maps are registered with matplotlib first,
        by ``Style.use``, which pyvista needs to resolve the name of one; so the theme
        works in a fresh process. That call also applies ``advent.mplstyle`` to the
        global settings of matplotlib.

        :returns: A theme to pass as ``theme=`` to ``pyvista.Plotter``. Each call
            returns a new one, and the global theme of pyvista is not changed.
        """
        Style.use()
        theme = Theme()
        theme.background = Palette.SNOW_WHITE
        theme.font.color = Palette.NIGHT_BLUE
        theme.cmap = Colormap.WARM.value
        theme.window_size = list(cls.WINDOW_SIZE)
        return theme
