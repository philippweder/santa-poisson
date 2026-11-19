import io
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from matplotlib.figure import Figure
from PIL import Image

from advent.export.output_folder import OutputFolder
from advent.style.style import Style


@dataclass(frozen=True, slots=True, kw_only=True)
# housestyle: ignore[dataclass-unvalidated] any folder is legal; a write makes it
class ImageExporter:
    """Writes a day's figure and its link preview as PNG files."""

    PREVIEW_SIZE: ClassVar[tuple[int, int]] = (1200, 630)
    """The width and height in pixels of a link preview."""

    NEAREST_ONLY: ClassVar[tuple[str, ...]] = ("P", "1")
    """The Pillow modes that are resampled by nearest-neighbour, whatever the filter."""

    SUFFIX: ClassVar[str] = ".png"
    """The suffix of every file this exporter reads and writes."""

    output_dir: Path
    """The folder the files are written to, and the preview source is read from."""

    def png(self, figure: Figure, *, name: str) -> Path:
        """
        Save a ``Style`` plot as a PNG exactly ``Style.WIDTH_PX`` wide.

        The plot is saved at ``Style.DPI`` with no ``bbox_inches``, so its size in
        pixels is its size in inches times the dpi. It is rendered in memory first, so
        a refused plot leaves no file.

        :param figure: The plot, ``Style.WIDTH_PX / Style.DPI`` inches wide.
        :param name: The file name, matching ``OutputFolder.NAME_PATTERN`` and ending in
            ``.png``.
        :returns: The path of the file written.
        :raises ValueError: If ``name`` is refused, ``figure`` is not the right width in
            inches, or the rendered image is not ``Style.WIDTH_PX`` pixels wide, as when
            ``savefig.bbox`` is set to ``tight`` in the matplotlib settings. The folder
            is not created then.
        """
        width_in = float(figure.get_size_inches()[0])
        wanted_in = Style.WIDTH_PX / Style.DPI
        if width_in != wanted_in:
            raise ValueError(f"figure must be {wanted_in} in wide, got {width_in}")
        folder = OutputFolder(path=self.output_dir)
        # Refuse a bad name before the figure is rendered; ``file`` below checks again.
        folder.locate(name, suffix=self.SUFFIX)

        buffer = io.BytesIO()
        # The stub of savefig leaves its PathLike parameter unparameterised.
        figure.savefig(buffer, format="png", dpi=Style.DPI)  # pyright: ignore[reportUnknownMemberType] stub
        with Image.open(buffer) as image:
            width_px = image.width
        if width_px != Style.WIDTH_PX:
            raise ValueError(
                f"the image must be {Style.WIDTH_PX} px wide, got {width_px}"
            )

        path = folder.file(name, suffix=self.SUFFIX)
        path.write_bytes(buffer.getvalue())
        return path

    @classmethod
    def fit_preview(cls, image: Image.Image) -> Image.Image:
        """
        Crop ``image``, centred, to the aspect ratio of ``PREVIEW_SIZE`` and resize it.

        The side that is too long is cut equally at both ends, then the crop is resized
        to exactly that size with Lanczos resampling. Both steps are one resize with a
        crop box, which keeps the crop exact when it falls between pixels. A palette or
        1-bit ``image`` is converted to RGBA first, since Pillow resamples those modes
        with nearest-neighbour whatever the filter.

        :param image: The image to fit; it is not changed.
        :returns: The resized copy, of size ``PREVIEW_SIZE``.
        """
        out_w, out_h = cls.PREVIEW_SIZE
        if image.mode in cls.NEAREST_ONLY:
            image = image.convert("RGBA")
        width, height = image.size
        if width * out_h > height * out_w:
            crop_w, crop_h = height * out_w / out_h, float(height)
        else:
            crop_w, crop_h = float(width), width * out_h / out_w
        left, top = 0.5 * (width - crop_w), 0.5 * (height - crop_h)
        return image.resize(
            cls.PREVIEW_SIZE,
            Image.Resampling.LANCZOS,
            box=(left, top, left + crop_w, top + crop_h),
        )

    def preview(self, *, source: str, name: str) -> Path:
        """
        Write a link preview of a PNG in the output folder.

        The PNG is cropped and resized as ``fit_preview`` does.

        :param source: The PNG to read, in the output folder.
        :param name: The preview to write, by bare file name, in the output folder.
        :returns: The path of the file written.
        :raises ValueError: If ``source`` or ``name`` is refused, or they are the same,
            which would overwrite the PNG that is read. The folder is not created then.
        :raises FileNotFoundError: If the PNG to read is not in the folder.
        :raises PIL.UnidentifiedImageError: If that file is not an image.
        """
        if source == name:
            raise ValueError(f"the preview would overwrite its source {source!r}")
        folder = OutputFolder(path=self.output_dir)
        source_path = folder.locate(source, suffix=self.SUFFIX)
        folder.locate(name, suffix=self.SUFFIX)

        with Image.open(source_path) as opened:
            resized = self.fit_preview(opened)

        target = folder.file(name, suffix=self.SUFFIX)
        resized.save(target, format="PNG")
        return target
