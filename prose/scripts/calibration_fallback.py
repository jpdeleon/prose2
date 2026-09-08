"""Search sibling night directories for calibration frames.

When a night's darks or flats for a band are missing, or present but not
exposure-matched (see ``calibrate_muscat2.select_darks_for_exposure``),
:func:`find_frames_in_other_nights` looks in nearby nights for a usable
replacement instead of failing or falling back to a mismatched local set.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from prose.utils import frames_from_obslog

# Mirrors calibrate_muscat2.EXPOSURE_RTOL/EXPOSURE_ATOL: only exact exposure
# matches are trustworthy without a master bias.
EXPOSURE_RTOL = 1e-3
EXPOSURE_ATOL = 1e-2


def _nearby_nights(data_dir: Path, max_days: int) -> list[str]:
    """Sibling night directories under *data_dir*'s parent, nearest first."""
    try:
        target_date = datetime.strptime(data_dir.name, "%y%m%d")
    except ValueError:
        return []

    candidates: list[tuple[int, str]] = []
    for night_dir in data_dir.parent.iterdir():
        if not night_dir.is_dir() or night_dir.name == data_dir.name:
            continue
        try:
            night_date = datetime.strptime(night_dir.name, "%y%m%d")
        except ValueError:
            continue
        days = abs((night_date - target_date).days)
        if days <= max_days:
            candidates.append((days, night_dir.name))
    candidates.sort()
    return [name for _, name in candidates]


def find_frames_in_other_nights(
    data_dir: Path,
    band: str,
    kind: str,
    band_from: Callable[[str | None, int | None], str | None],
    exposure: float | None = None,
    exposure_rtol: float = EXPOSURE_RTOL,
    exposure_atol: float = EXPOSURE_ATOL,
    max_days: int = 90,
    instrument: str | None = None,
) -> tuple[list[str], str | None]:
    """Search sibling nights (by calendar distance) for *kind* frames in *band*.

    *kind* is ``"DARK"`` or ``"FLAT"`` (matched against the obslog ``OBJECT``
    column, case-insensitively). Only nights with obslog metadata are searched
    (see :func:`prose.utils.frames_from_obslog`), nearest first, within
    *max_days* of *data_dir*'s own night. When *exposure* is given, only frames
    within *exposure_rtol*/*exposure_atol* of it qualify; otherwise any frame
    of *kind* in *band* does.

    Returns ``(paths, source_night)``; ``([], None)`` when nothing qualifies
    within the search window.
    """
    data_dir = Path(data_dir)
    instrument = (instrument or data_dir.parent.name).lower()
    kind = kind.strip().upper()

    for night_name in _nearby_nights(data_dir, max_days):
        candidate_dir = data_dir.parent / night_name
        records = frames_from_obslog(candidate_dir, instrument)
        if not records:
            continue

        matches = []
        for rec in records:
            if rec["object"].strip().upper() != kind:
                continue
            if band_from(rec["filter"], rec["ccd"]) != band:
                continue
            if exposure is not None and (
                rec["exposure"] is None
                or not np.isclose(
                    rec["exposure"], exposure, rtol=exposure_rtol, atol=exposure_atol
                )
            ):
                continue
            matches.append(rec["path"])

        if matches:
            return sorted(set(matches)), night_name

    return [], None
