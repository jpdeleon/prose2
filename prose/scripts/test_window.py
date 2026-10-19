"""Frame selection for ``--test_run`` on the raw MuSCAT / MuSCAT2 calibrators.

``--test_run`` used to mean "the first N frames". These helpers let it centre
the N-frame window on a specific FITS frame number instead (``--refid``), so a
test run can sample any part of the night.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

DEFAULT_TEST_RUN_FRAMES = 10

_MUSCAT2_NUMBER = re.compile(r"_(\d{6})(\d{4})(?:_calibrated)?\.fits$")
_TRAILING_NUMBER = re.compile(r"_(\d+)(?:_calibrated)?\.fits$")


def frame_number(path: str | Path) -> int | None:
    """FITS frame number of a raw or calibrated MuSCAT/MuSCAT2 file name.

    ``MCT20_2112260156.fits`` -> 156, ``MCT20_2112260156_calibrated.fits`` ->
    156, ``MSCT0_00000156.fits`` -> 156. ``None`` if the name has no number.
    """
    name = str(path)
    m = _MUSCAT2_NUMBER.search(name)
    if m:
        return int(m.group(2))
    m = _TRAILING_NUMBER.search(name)
    return int(m.group(1)) if m else None


def window_around_frame_number(
    files: Sequence[str | Path], center: int, n: int
) -> list:
    """*n* consecutive files centred on the frame numbered exactly *center*.

    Raises ``ValueError`` when *center* is not among *files*: the caller asked
    for a specific frame, and silently substituting the nearest one hides that.
    """
    for i, fp in enumerate(files):
        if frame_number(fp) == center:
            start = max(0, i - n // 2)
            return list(files[start : start + n])
    raise ValueError(f"frame number {center} not found among {len(files)} frames")


def select_test_run_frames(
    files: Sequence[str | Path], n: int, center: int | None = None
) -> list:
    """The ``--test_run`` subset of one band's *files*: the first *n*, or the
    *n*-frame window centred on frame number *center* when given."""
    if center is None:
        return list(files[:n])
    return window_around_frame_number(files, center, n)
