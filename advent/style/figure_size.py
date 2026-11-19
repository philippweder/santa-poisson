from enum import Enum


class FigureSize(Enum):
    """The figure sizes of the calendar; each value is ``(width, height)`` in inches."""

    WIDE = (8.0, 4.5)
    STANDARD = (8.0, 6.0)
    SQUARE = (8.0, 8.0)
