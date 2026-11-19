from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image
from pyvista import Plotter

from advent.export.image_exporter import ImageExporter
from advent.export.output_folder import OutputFolder


@dataclass(frozen=True, slots=True, kw_only=True)
# housestyle: ignore[dataclass-unvalidated] any folder is legal; a write makes it
class SceneExporter:
    """Writes a day's 3D scene as one HTML file and its link preview as a PNG file."""

    SCENE_SUFFIX: ClassVar[str] = ".html"
    """The suffix of the scene file."""

    SCREENSHOT_SUFFIX: ClassVar[str] = ".png"
    """The suffix of the screenshot file."""

    output_dir: Path
    """The folder the files are written to."""

    def scene(self, plotter: Plotter, *, name: str) -> Path:
        """
        Write the scene of ``plotter`` as one self-contained HTML file.

        The file holds the viewer and the scene as inline script and data, and its
        markup has no ``src`` or ``href`` to a file or a URL. The viewer script itself
        builds favicon links to a remote site while it runs; a browser without a
        network ignores them. The export is made through ``plotter.trame.export_html``
        on a worker thread: trame-pyvista starts its server with ``nest_asyncio2``,
        which the environments do not install, when it is called on the main thread,
        and with a thread of its own when it is not. That server thread outlives the
        call, and the next export in the process reuses it.

        :param plotter: The plotter holding the scene; its window size is not changed.
        :param name: The file name, matching ``OutputFolder.NAME_PATTERN`` and ending
            in ``.html``.
        :returns: The path of the file written.
        :raises ValueError: If ``name`` is refused. The folder is not created then.
        """
        folder = OutputFolder(path=self.output_dir)
        path = folder.file(name, suffix=self.SCENE_SUFFIX)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(plotter.trame.export_html, path).result()
        return path

    def screenshot(self, plotter: Plotter, *, name: str) -> Path:
        """
        Write a link preview of the scene of ``plotter``, as a 1200 by 630 PNG.

        The screenshot is taken at the window size of ``plotter``, which is not
        changed, then cropped and resized as ``ImageExporter.fit_preview`` does.

        :param plotter: The plotter holding the scene; it must be off screen, or shown.
        :param name: The file name, matching ``OutputFolder.NAME_PATTERN`` and ending
            in ``.png``.
        :returns: The path of the file written.
        :raises ValueError: If ``name`` is refused. The folder is not created then.
        """
        folder = OutputFolder(path=self.output_dir)
        # Refuse a bad name before the plotter renders; ``file`` below checks again.
        folder.locate(name, suffix=self.SCREENSHOT_SUFFIX)

        # The stub of screenshot types its result as optional; it is an array here.
        pixels = np.asarray(plotter.screenshot(None, return_img=True), dtype=np.uint8)
        with Image.fromarray(pixels) as shot:
            preview = ImageExporter.fit_preview(shot)

        path = folder.file(name, suffix=self.SCREENSHOT_SUFFIX)
        preview.save(path, format="PNG")
        return path
