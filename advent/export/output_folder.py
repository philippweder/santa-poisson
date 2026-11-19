import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True, kw_only=True)
# housestyle: ignore[dataclass-unvalidated] any folder is legal; ``file`` creates it
class OutputFolder:
    """The folder a day's published files are written to, shared by the exporters."""

    NAME_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"[a-z0-9][a-z0-9_-]*\.(png|mp4|html)"
    )
    """
    What a published file name looks like, the whole name matching.

    The library enforces the pattern because the publish step accepts nothing else.
    """

    path: Path
    """The folder; it is created by the first call of ``file``."""

    def locate(self, name: str, *, suffix: str) -> Path:
        """
        Return the path of a file in the folder, creating nothing.

        :param name: A bare file name matching ``NAME_PATTERN``.
        :param suffix: The suffix the file must end in, with its dot, for example
            ``".png"``.
        :returns: The folder path joined with ``name``.
        :raises ValueError: If ``name`` does not match ``NAME_PATTERN``, or does not
            end in ``suffix``.
        """
        if self.NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"name must match {self.NAME_PATTERN.pattern}, got {name!r}"
            )
        if Path(name).suffix != suffix:
            raise ValueError(f"name must end in {suffix!r}, got {name!r}")

        return self.path / name

    def file(self, name: str, *, suffix: str) -> Path:
        """
        Return the path of a file in the folder, creating the folder on first use.

        :param name: A bare file name matching ``NAME_PATTERN``.
        :param suffix: The suffix the file must end in, with its dot, for example
            ``".png"``.
        :returns: The folder path joined with ``name``.
        :raises ValueError: If ``name`` does not match ``NAME_PATTERN``, or does not
            end in ``suffix``. Nothing is created then.
        """
        path = self.locate(name, suffix=suffix)
        self.path.mkdir(parents=True, exist_ok=True)
        return path
