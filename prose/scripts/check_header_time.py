"""Audit MuSCAT time-header conventions across the data archive.

MuSCAT/MuSCAT2/MuSCAT3/MuSCAT4 FITS headers do not store the observation time
the same way, and the convention has *changed over time* (e.g. early OAO-style
MuSCAT3 frames carry only ``MJD-STRT`` with a date-only ``DATE-OBS``, whereas
later LCO/BANZAI frames carry a full-timestamp ``DATE-OBS`` and ``MJD-OBS``).
Those assumptions are encoded in the ``prose2/data/*.telescope`` configs
(``keyword_jd`` + ``jd_scale``); when they drift, the time axis can silently
collapse (e.g. to midnight), corrupting BJD-TDB.

This script randomly samples dated directories per instrument (always including
the oldest and newest), inspects one science frame in each, and reports:

* the time keyword actually present in the header,
* whether ``DATE-OBS`` includes a time component,
* the telescope ``prose`` resolves the frame to, and
* a correctness check: the JD ``prose`` computes (after
  ``run_photometry.normalize_time_to_jd``) vs the header truth.

It then summarises the distinct conventions seen per instrument and flags any
that changed over the sampled time span, or any frame whose resolved JD is wrong.

Example
-------
::

    python -m prose.scripts.check_header_time
    python -m prose.scripts.check_header_time --samples 10 --no-check-jd
    python -m prose.scripts.check_header_time --instrument muscat3 --data-root /data
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys

from astropy.io import fits
from astropy.time import Time

# Header keywords that can carry the observation time, in order of preference.
# Single source of truth lives in ``run_photometry`` (also used to derive the
# output-file date); imported here so the audit and the pipeline never drift.
from prose.scripts.run_photometry import TIME_KEYS

# Default instrument -> data directory. Keys match the lowercase names used
# elsewhere in the pipeline; values are the on-disk archive roots.
DEFAULT_INSTRUMENT_DIRS = {
    "muscat1": "/data/MuSCAT",
    "muscat2": "/data/MuSCAT2",
    "muscat3": "/data/MuSCAT3",
    "muscat4": "/data/MuSCAT4",
}

# Tolerance (seconds) for the resolved-vs-truth JD comparison. MJD-OBS is often
# rounded to ~0.1 s, so a sub-2 s window avoids false positives.
JD_TOLERANCE_S = 2.0

_FITS_PATTERNS = ("MSCT?_*.fits", "MCT2?_*.fits", "MCT3?_*.fits", "*e91.fits", "*.fits")


def _sample_days(base: str, n: int, rng: random.Random) -> list[str]:
    """Return up to *n* dated subdirectory names, always including the oldest
    and newest, with the remainder sampled uniformly from the middle."""
    if not os.path.isdir(base):
        return []
    days = sorted(
        x
        for x in os.listdir(base)
        if re.fullmatch(r"\d{6,8}", x) and os.path.isdir(os.path.join(base, x))
    )
    if len(days) <= n:
        return days
    picks = {days[0], days[-1]}
    middle = days[1:-1]
    picks |= set(rng.sample(middle, min(n - len(picks), len(middle))))
    return sorted(picks)


def _find_frame(day_dir: str) -> str | None:
    """Return a representative (middle, non-tiny) FITS file in *day_dir*, looking
    one level deeper if the directory only holds sub-folders."""
    for pat in _FITS_PATTERNS:
        files = [f for f in glob.glob(os.path.join(day_dir, pat)) if os.path.getsize(f) > 1000]
        if files:
            return sorted(files)[len(files) // 2]
    for sub in sorted(glob.glob(os.path.join(day_dir, "*"))):
        if os.path.isdir(sub):
            found = _find_frame(sub)
            if found:
                return found
    return None


def _read_header(path: str):
    """Return the header carrying the image (handles ``.fits.fz`` compression)."""
    h = fits.getheader(path, 0)
    if h.get("NAXIS", 0) == 0:
        try:
            h = fits.getheader(path, 1)
        except IndexError:
            pass
    return h


def _primary_time_key(header) -> str | None:
    for k in TIME_KEYS:
        if k in header:
            return k
    return None


def _truth_jd(header) -> tuple[float | None, str]:
    """Best-effort 'true' JD from the raw header, and the source keyword."""
    if "MJD-STRT" in header:
        return float(header["MJD-STRT"]) + 2_400_000.5, "MJD-STRT"
    if "MJD-OBS" in header:
        return float(header["MJD-OBS"]) + 2_400_000.5, "MJD-OBS"
    do = header.get("DATE-OBS")
    if do and ("T" in str(do) or ":" in str(do)):
        try:
            return float(Time(do).jd), "DATE-OBS"
        except Exception:  # noqa: BLE001
            return None, "DATE-OBS?"
    return None, "none"


def _resolved_jd(path: str):
    """Return ``(telescope_name, jd_after_normalize)`` as prose would compute it,
    or ``(None, None)`` if prose is unavailable."""
    import numpy as np
    from prose import FITSImage
    from prose.scripts.run_photometry import normalize_time_to_jd

    im = FITSImage(path)
    jd = float(normalize_time_to_jd(np.array([im.jd]), im.telescope.jd_scale)[0])
    return im.telescope.name, jd


def audit_instrument(
    name: str, base: str, samples: int, check_jd: bool, rng: random.Random
) -> list[dict]:
    """Sample and inspect frames for one instrument; return per-day records."""
    rows: list[dict] = []
    for day in _sample_days(base, samples, rng):
        f = _find_frame(os.path.join(base, day))
        if f is None:
            rows.append({"day": day, "file": None})
            continue
        try:
            h = _read_header(f)
        except Exception as e:  # noqa: BLE001
            rows.append({"day": day, "file": os.path.basename(f), "error": str(e)})
            continue
        do = str(h.get("DATE-OBS", ""))
        row = {
            "day": day,
            "file": os.path.basename(f),
            "instrume": h.get("INSTRUME"),
            "telescop": h.get("TELESCOP"),
            "date_obs": do,
            "date_has_time": ("T" in do or ":" in do),
            "time_key": _primary_time_key(h),
        }
        if check_jd:
            truth, _ = _truth_jd(h)
            try:
                tel, jd = _resolved_jd(f)
                row["telescope"] = tel
                if truth is not None and jd is not None:
                    row["jd_dt_s"] = (jd - truth) * 86400.0
                    row["jd_ok"] = abs(row["jd_dt_s"]) <= JD_TOLERANCE_S
            except Exception as e:  # noqa: BLE001
                row["telescope"] = f"(prose error: {e})"
        rows.append(row)
    return rows


def _signature(row: dict) -> str | None:
    """A compact 'time convention' key used to detect drift over time."""
    if not row.get("file") or row.get("error"):
        return None
    return f"time_key={row.get('time_key')}, date_has_time={row.get('date_has_time')}"


def print_report(name: str, base: str, rows: list[dict], check_jd: bool) -> dict:
    print("=" * 80)
    print(f"{name}  ({base})")
    if not rows:
        print("  (no dated directories found)")
        return {"changed": False, "wrong": 0}

    for r in rows:
        if not r.get("file"):
            print(f"  {r['day']}: (no frame found)")
            continue
        if r.get("error"):
            print(f"  {r['day']}: {r['file']}  ERROR {r['error']}")
            continue
        line = (
            f"  {r['day']}: {r['file']}\n"
            f"        INSTRUME={r['instrume']!r} TELESCOP={r['telescop']!r}\n"
            f"        DATE-OBS={r['date_obs']!r} (has_time={r['date_has_time']}) "
            f"time_key={r['time_key']}"
        )
        if check_jd and "jd_ok" in r:
            verdict = "OK" if r["jd_ok"] else "WRONG"
            line += (
                f"\n        prose telescope={r.get('telescope')!r} "
                f"dt={r['jd_dt_s']:+.2f}s -> {verdict}"
            )
        elif check_jd:
            line += f"\n        prose telescope={r.get('telescope')!r} (no truth to compare)"
        print(line)

    sigs: list[tuple[str, str]] = [
        (r["day"], _signature(r)) for r in rows if _signature(r) is not None
    ]
    distinct = sorted({s for _, s in sigs})
    wrong = sum(1 for r in rows if r.get("jd_ok") is False)

    print("  " + "-" * 40)
    if len(distinct) > 1:
        print(f"  CONVENTION CHANGED over time — {len(distinct)} distinct signatures:")
        for s in distinct:
            days = [d for d, sg in sigs if sg == s]
            print(f"    - {s}   (e.g. {days[0]}..{days[-1]})")
    elif distinct:
        print(f"  convention stable: {distinct[0]}")
    if check_jd:
        if wrong:
            print(f"  *** {wrong} frame(s) resolve to a WRONG JD — check .telescope config ***")
        else:
            print("  all sampled frames resolve to a correct JD")
    return {"changed": len(distinct) > 1, "wrong": wrong}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data-root",
        "--data_root",
        default=None,
        help="Override the parent of the per-instrument dirs (default: built-in "
        "absolute paths). With this set, dirs are <root>/MuSCAT[2-4].",
    )
    ap.add_argument(
        "--instrument",
        action="append",
        choices=sorted(DEFAULT_INSTRUMENT_DIRS),
        help="Restrict to one or more instruments (repeatable). Default: all.",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of dated directories to sample per instrument (default: 6).",
    )
    ap.add_argument("--seed", type=int, default=7, help="RNG seed (default: 7).")
    ap.add_argument(
        "--no-check-jd",
        "--no_check_jd",
        dest="check_jd",
        action="store_false",
        help="Skip loading frames through prose to verify the resolved JD "
        "(header inspection only).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    instruments = args.instrument or sorted(DEFAULT_INSTRUMENT_DIRS)
    dirs = {}
    for inst in instruments:
        if args.data_root:
            suffix = DEFAULT_INSTRUMENT_DIRS[inst].rsplit("/", 1)[-1]
            dirs[inst] = os.path.join(args.data_root, suffix)
        else:
            dirs[inst] = DEFAULT_INSTRUMENT_DIRS[inst]

    any_changed = any_wrong = False
    for inst in instruments:
        rows = audit_instrument(inst, dirs[inst], args.samples, args.check_jd, rng)
        result = print_report(inst, dirs[inst], rows, args.check_jd)
        any_changed |= result["changed"]
        any_wrong |= result["wrong"] > 0

    print("=" * 80)
    print(
        f"summary: convention drift detected in >=1 instrument: {any_changed}; "
        f"wrong JD detected: {any_wrong}"
    )
    # Non-zero exit only when a frame resolves to a wrong JD (an actionable bug).
    return 1 if any_wrong else 0


if __name__ == "__main__":
    sys.exit(main())
