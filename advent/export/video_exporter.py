import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from advent.export.output_folder import OutputFolder


@dataclass(frozen=True, slots=True, kw_only=True)
# housestyle: ignore[dataclass-unvalidated] any folder is legal; a write makes it
class VideoExporter:
    """Writes a day's looping video as an H.264 MP4 file, by piping frames to ffmpeg."""

    SUFFIX: ClassVar[str] = ".mp4"
    """The suffix of every file this exporter writes."""

    MAX_BYTES: ClassVar[int] = 10_000_000
    """
    The size, in bytes, that a video must stay below.

    A megabyte is 10**6 bytes, as in the size limits of ``advent check``.
    """

    ENCODER_FLAGS: ClassVar[tuple[str, ...]] = (
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
    )
    """
    The ffmpeg flags after the input: libx264, no sound, and ``+faststart``.

    ``yuv420p`` plays in every browser and needs even sides; ``+faststart`` puts the
    ``moov`` atom before ``mdat``. ``-crf`` and ``-preset`` apply to libx264 only.
    """

    FRAME_NDIM: ClassVar[int] = 3
    """The number of axes of a frame: height, width and colour channel."""

    CHANNELS: ClassVar[int] = 3
    """The length of the colour axis of a frame: red, green and blue."""

    output_dir: Path
    """The folder the video is written to."""

    def video(
        self, frames: Iterable[NDArray[np.uint8]], *, name: str, fps: int
    ) -> Path:
        """
        Encode ``frames`` as an MP4 below ``MAX_BYTES``.

        Every frame is checked before ffmpeg starts, so ``frames`` is read to the end,
        and held as bytes, first: the memory needed is the size of the raw video.
        Looping is the job of the page, not of the file. The output folder is created
        once the pictures are accepted, and stays if the encoding then fails.

        :param frames: The frames, in order: ``uint8`` arrays of shape
            ``(height, width, 3)`` in RGB, all of one size with even ``width`` and
            ``height``.
        :param name: The file name, matching ``OutputFolder.NAME_PATTERN`` and ending
            in ``.mp4``.
        :param fps: The rate in pictures per second, at least 1.
        :returns: The path of the file written.
        :raises ValueError: If ``name`` is refused, ``fps`` is below 1, there is no
            frame, a frame is not a ``uint8`` array of shape ``(height, width, 3)``,
            the pictures differ in size, or their width or height is odd or 0; the
            folder is not created and ffmpeg is not started then. Also if the file is
            ``MAX_BYTES`` or more, when it is removed; lower the frame size, the frame
            count or ``fps``.
        :raises RuntimeError: If ffmpeg fails, when the partial file is removed.
        :raises FileNotFoundError: If ffmpeg is not on the path.
        """
        folder = OutputFolder(path=self.output_dir)
        folder.locate(name, suffix=self.SUFFIX)
        if fps < 1:
            raise ValueError(f"fps must be at least 1, got {fps}")
        raw, width, height = self._raw(frames)

        path = folder.file(name, suffix=self.SUFFIX)
        command = [
            "ffmpeg",
            *("-y", "-loglevel", "error"),
            *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}"),
            *("-r", str(fps), "-i", "-"),
            *self.ENCODER_FLAGS,
            str(path),
        ]
        # The command is a list of our own strings, run without a shell.
        result = subprocess.run(  # noqa: S603 ffmpeg is looked up on the path
            command,
            input=raw,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            path.unlink(missing_ok=True)
            log = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg exited with status {result.returncode}: {log}")

        size = path.stat().st_size
        if size >= self.MAX_BYTES:
            path.unlink()
            raise ValueError(
                f"the video is {size} bytes, not below {self.MAX_BYTES}; lower the "
                f"frame size, the frame count or fps"
            )
        return path

    @classmethod
    def _raw(cls, frames: Iterable[NDArray[np.uint8]]) -> tuple[bytearray, int, int]:
        """
        Check the ``frames`` and join them into one buffer of raw RGB bytes.

        :param frames: The frames, as for ``video``.
        :returns: The bytes of every picture in order, the width and the height.
        :raises ValueError: As ``video`` does for the pictures.
        """
        raw = bytearray()
        shape: tuple[int, ...] | None = None
        for index, frame in enumerate(frames):
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                raise ValueError(f"frame {index} must be uint8, got {array.dtype}")
            if array.ndim != cls.FRAME_NDIM or array.shape[-1] != cls.CHANNELS:
                raise ValueError(
                    f"frame {index} must have shape (height, width, 3), "
                    f"got {array.shape}"
                )
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError(
                    f"frame {index} has shape {array.shape}, but frame 0 has {shape}"
                )
            raw += array.tobytes()

        if shape is None:
            raise ValueError("there is no frame to encode")
        height, width = shape[0], shape[1]
        if width % 2 != 0 or height % 2 != 0 or width == 0 or height == 0:
            raise ValueError(
                f"width and height must be even and not 0 for yuv420p, "
                f"got {width} by {height}"
            )
        return raw, width, height
