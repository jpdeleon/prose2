#!/usr/bin/env python
"""End-to-end prose photometry pipeline for LCO MuSCAT3/4 (and Sinistro) data.

This script reproduces the reduction demonstrated in
``notebooks/prose_muscat34_template.ipynb`` as a command-line tool. Given a
directory of *calibrated* (LCO BANZAI-reduced, e.g. ``*-e91.fits``) science
frames for a single target, it:

1. groups frames per photometric band for the requested target,
2. builds a per-band reference image and detects sources,
3. identifies the target via WCS / catalog cross-match,
4. sizes the sky annulus to avoid Gaia contaminants (>=10% of target flux)
   and sets aperture radii from the target FWHM up to that annulus,
5. runs aperture photometry in parallel (``prose.SequenceParallel``),
6. performs automatic differential photometry (Broeg et al. 2005),
7. converts GJD-UTC to BJD-TDB, and
8. writes per-band CSV / PNG / GIF products and multi-band summary
   plots plus a single ``.npz`` archive.

The output file naming and product set follow the layout used by
``eloy/docs/test_run/multiband_parallel_pipeline.py`` (used only as a
structural reference; the reduction itself is pure ``prose``)::

    {target}_{inst}_{band}_{date}_ref.png
    {target}_{inst}_{band}_{date}_apertures.png
    {target}_{inst}_{band}_{date}_alignment.png
    {target}_{inst}_{band}_{date}.gif
    {target}_{inst}_{band}_{date}.csv
    {target}_{inst}_{bands}_{date}_lightcurves.png
    {target}_{inst}_{bands}_{date}_raw_flux.png
    {target}_{inst}_{bands}_{date}_covariates.png
    {target}_{inst}_{bands}_{date}_stacks.png
    {target}_{inst}_{bands}_{date}.npz
    {iso-timestamp}.log

Example
-------
::

    python -m prose.scripts.run_photometry \
        --target_name TOI-6715 \
        --data_dir /data/MuSCAT4/250416 \
        --results_dir ./TOI-6715_250416 \
        --bands gp rp ip zs --ref_band gp


TODO: 
* add --calibrated flag which checks if data was calibrated by BANZAI already or pipeline needs to start from scratch
* add cutout plots to see zoomed comparison stars. Label which is target
* cleanup and fix notebooks
* add the ability to ingest single-band sinistro dataset
* why prose is superior in-memory multiprocessing but not eloy?
"""

from __future__ import annotations

import argparse

import csv
import logging
import os
import re
import sys
import time as time_module
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless: write figures without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.mast import Mast
from astroquery.simbad import Simbad
from rich.progress import track

from prose import FITSImage, Fluxes, Sequence, blocks
from prose import __version__ as _PROSE_VERSION
from prose.blocks import catalogs
from prose.core.block import Block
from prose.core.source import PointSource, Sources
from prose.core.sequence import SequenceParallel
from prose.scripts import calibrate_muscat, calibrate_muscat2
from prose.scripts.solve_wcs_astrometry import logger as _wcs_logger
from prose.scripts.solve_wcs_astrometry import inject_wcs_into_file, load_wcs_fits
from prose.utils import (
    LCO_SITES,
    PIXSCALES,
    _FILTER_ALIASES,
    coord_cache_path,
    frames_from_obslog,
    get_saturation_from_header,
    get_simbad_data,
    load_cached_df,
    read_filename_per_band,
    save_cached_df,
    z_scale,
)

# --------------------------- constants / defaults ---------------------------
# ignore if using BANZAI-reduced fits files
CAL_OBJECT_MAP = {"bias": "BIAS", "dark": "DARK", "flat": "FLAT"}

# Fallback observatory site (astropy EarthLocation site registry name) for
# instruments whose headers lack an LCO ``SITE`` keyword. Used for BJD-TDB
# barycentric correction when no site can be read from the header.
INSTRUMENT_SITES: dict[str, str] = {
    "muscat": "OAO",  # Okayama Astrophysical Observatory, Japan
    "muscat2": "teide",  # Teide Observatory (TCS 1.52m), Tenerife
}

INSTRUMENT_MAP: dict[tuple[str, str], str] = {
    ("ep06", "coj2m002"): "muscat4",
    ("ep07", "coj2m002"): "muscat4",
    ("ep08", "coj2m002"): "muscat4",
    ("ep09", "coj2m002"): "muscat4",
    ("ep04", "coj2m002"): "muscat3",  # g
    ("ep02", "coj2m002"): "muscat3",  # r
    ("ep03", "coj2m002"): "muscat3",  # i
    ("ep05", "coj2m002"): "muscat3",  # z
}

# Instrument-specific filter alias maps (INSTRUME header keyword -> canonical band name)
INSTRUMENT_FILTER_ALIASES: dict[str, dict[str, str]] = {
    "muscat": {"g": "gp", "r": "rp", "z_s": "zs"},  # only 3 bands, no i
    "muscat2": {
        "g": "gp",
        "r": "rp",
        "i": "ip",
        "z_s": "zs",
    },  # standardized to p names
}

DEFAULT_BROAD_BANDS = ["gp", "rp", "ip", "zs"]
DEFAULT_NARROW_BANDS = ["g_narrow", "Na_D", "i_narrow", "z_narrow"]
DEFAULT_BANDS = DEFAULT_BROAD_BANDS
DEFAULT_GIF_STRIDE = 10
TEST_RUN_FRAMES = 10  # frames per band used by --test_run
FPS = 5
GIF_MAX_PX = 512  # max GIF frame dimension [pix]; larger frames are downsampled

# reference-image / detection defaults (mirror the template notebook)
MAX_NUM_STARS = 10  # nth brightest stars to keep
DETECT_NUM_STARS_FACTOR = 1.5  # detect more stars initially to capture faint targets
CUTOUT_SIZE = 35  # cutout size of detected stars [pix]
CCD_TRIM_SIZE_YX = (0, 0)  # trim image edges [pix]
MIN_STAR_AREA = 10  # min detected-source area [pix]
MIN_STAR_SEPARATION = 10  # min separation between sources [pix]
ALIGN_DISCARD_TOLERANCE = 0.4  # cross-filter alignments can share <50% top sources
# Close-companion exclusion for the comparison-star pool. The auto threshold is
# conservative: keep detection deblending at 10 px, then reject candidate comps
# that still have a nearby detected/Gaia neighbour inside max(2 x FWHM, 3").
AVOID_NEARBY_STAR_AUTO = "auto"
AVOID_NEARBY_STAR_AUTO_FWHM = 2.0
AVOID_NEARBY_STAR_AUTO_MIN_ARCSEC = 3.0
AVOID_NEARBY_STAR_MATCH_ARCSEC = 1.5
# Exclude detected stars whose centroid is within this many pixels of any CCD
# edge from the comparison-star pool (the target is never excluded). ``None``
# means auto: half the cutout size, so the PSF cutout box stays fully on-chip.
# 0 disables edge exclusion.
EDGE_MARGIN_PIX = None

# Color scheme for plots. These constants reflect the default ``Greys`` cmap
# rendering, so downstream callers/tests can compare against the public values.
COLOR_TARGET = "crimson"  # distinct, visible on all grey levels
COLOR_APERTURE = "gold"  # complementary to target, high contrast
COLOR_SKY_ANNULUS = "yellow"  # visible on both dark and light sky
COLOR_SIMBAD_DEFAULT = "teal"  # works on dark image backgrounds
COLOR_SIMBAD_ECLBIN = "orange"  # warmer, more visible than default markers
COLOR_SOURCES = "lime"  # pops against dark source regions

# SIMBAD OTYPE substrings to highlight; all flagged types use the eclbin color.
_SIMBAD_FLAG_TYPES = {
    "eclbin": "simbad_eclbin",
    "eclipsing binary": "simbad_eclbin",
    "sb*": "simbad_eclbin",
}

_DARK_COLORS = {  # for light background
    "target": COLOR_TARGET,
    "aperture": COLOR_APERTURE,
    "sky_annulus": COLOR_SKY_ANNULUS,
    "simbad_default": COLOR_SIMBAD_DEFAULT,
    "simbad_eclbin": COLOR_SIMBAD_ECLBIN,
    "sources": COLOR_SOURCES,
}
_BRIGHT_COLORS = {  # for dark background
    "target": COLOR_TARGET,
    "aperture": "darkgreen",
    "sky_annulus": "yellow",
    "simbad_default": COLOR_SIMBAD_DEFAULT,
    "simbad_eclbin": COLOR_SIMBAD_ECLBIN,
    "sources": "darkmagenta",
}


def get_plot_colors(cmap: str = "Greys") -> dict:
    return dict(_BRIGHT_COLORS if not cmap.endswith("_r") else _DARK_COLORS)


# Gaia aperture-radii / sky-annulus heuristic
APER_STEP_PIX = 2  # spacing of aperture radii [pix]
CONTAM_DMAG = 2.5  # neighbour contaminates if Gmag - target < this (>=10% target flux)
CONTAM_MARGIN_PIX = 2  # keep annulus/aperture this far inside a contaminant [pix]
GAIA_CUTOUT = (200, 200)  # cutout around target for the Gaia query [pix]

# differential-photometry cleaning — None means the axis is not clipped
SIGMA_BKG = None
SIGMA_FWHM = None
SIGMA_DX = None
SIGMA_DY = None
BIN_SIZE_DAYS = 10 / 60 / 24  # 10-minute bins for plots

# Maximum pixel distance for cross-matching sources between bands.  When
# --ref_band is set, all bands are aligned to the same reference frame so the
# same physical star appears at nearly the same pixel position in every band.
# A tolerance of 5 px comfortably covers residual alignment offsets while
# avoiding spurious matches in crowded fields.
_CROSSMATCH_TOLERANCE_PX = 5.0
_TARGET_PIXEL_INFERENCE_TOLERANCE_PX = 25.0

# GJD->BJD sanity bound (light travel time should be well under this)
MAX_TIME_OFFSET_MIN = 2 * 8.4

# JD = MJD + this offset. Some instruments (e.g. MuSCAT2/TCS, keyword MJD-STRT)
# report their time axis in MJD; prose flags this via Telescope.jd_scale == "mjd".
MJD_TO_JD = 2_400_000.5

# Header keywords that can carry the observation time, in order of preference.
# The convention varies by instrument and epoch (e.g. MuSCAT2/TCS reports
# ``MJD-STRT`` with only a date-only ``DATE-OBS``), so a date derived from these
# numeric keys is more robust than trusting ``DATE-OBS`` alone. This is the
# single source of truth reused by ``check_header_time`` and ``date_from_header``.
TIME_KEYS: tuple[str, ...] = ("MJD-STRT", "MJD-OBS", "JD", "JD-STRT", "BJD")

# Band color map. Keyed by the full band token so Sloan broadband,
# narrow-band, and Johnson (B/V/R) filters each get a distinct color instead of
# collapsing onto the leading-letter color (which left B/V/R all black and every
# *_narrow sharing its broadband sibling's color).
BAND_COLORS = {
    # Sloan broadband
    "gp": "blue",
    "rp": "green",
    "ip": "orange",
    "zs": "red",
    # narrow-band (distinct from their broadband siblings)
    "g_narrow": "teal",
    "r_narrow": "olive",
    "i_narrow": "gold",
    "z_narrow": "maroon",
    "g_wide": "navy",
    "Na_D": "magenta",
    # Johnson
    "B": "cyan",
    "V": "purple",
    "R": "brown",
}

# Fallback for unmapped tokens: color by the leading Sloan letter, else black.
_BROADBAND_FALLBACK = {"g": "blue", "r": "green", "i": "orange", "z": "red"}


def band_color(band: str) -> str:
    if band in BAND_COLORS:
        return BAND_COLORS[band]
    return _BROADBAND_FALLBACK.get((band[:1] or "").lower(), "k")


logger = logging.getLogger("prose_run_photometry")


def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = log_exception


# ------------------------------- helpers -------------------------------


def setup_logger(outdir: Path, verbose: bool = False) -> Path:
    """Configure logging. The file always records INFO; the console shows
    INFO only when ``verbose`` is set, otherwise WARNING and above.
    """
    logger.setLevel(logging.INFO)
    log_path = outdir / f"{datetime.now().isoformat()}.log"
    fmt = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO if verbose else logging.WARNING)

    for handler in (file_handler, stream_handler):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    # --- startup banner (always first entry in the log) ---
    separator = "=" * 60
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info(separator)
    logger.info(f"prose v{_PROSE_VERSION}  |  {now_utc}")
    logger.info("command: photometry (run_photometry)")
    logger.info(separator)
    # --- end banner ---

    logger.info(f"log file: {log_path}")
    return log_path


def get_instrument(header) -> str:
    """Derive a short instrument label from a FITS header."""
    instrume = str(header.get("INSTRUME", "")).lower()
    if instrume in ("muscat", "muscat2"):
        return instrume
    telid = str(header.get("TELID", "")).lower()
    site = str(header.get("SITEID", "")).lower()
    if telid == "2m0a":
        return "muscat4" if site == "coj" else "muscat3"
    if telid == "1m0a":
        return "sinistro"
    return "unknown"


def _resolve_band(raw_filter: str, instrument: str, bands: list[str]) -> str | None:
    """Map raw filter value to canonical band name.

    Resolution chain: instrument-specific aliases -> global :data:`_FILTER_ALIASES`
    -> raw value.  Returns the band name if it is present in *bands*, else None.
    """
    aliases = INSTRUMENT_FILTER_ALIASES.get(instrument)
    if aliases:
        band = aliases.get(raw_filter)
        if band is not None and band in bands:
            return band
    band = _FILTER_ALIASES.get(raw_filter)
    if band is not None and band in bands:
        return band
    if raw_filter in bands:
        return raw_filter
    return None


def _date_from_time_keys(header) -> str:
    """Derive YYMMDD from the canonical numeric time keywords (:data:`TIME_KEYS`).

    Fallback for when the calendar-date keywords (``DAY-OBS``/``DATE-OBS``) are
    absent -- they can be missing on some epochs or stripped by upstream
    processing (e.g. WCS injection), whereas ``MJD-STRT`` survives. ``MJD-``
    keywords are shifted to JD before conversion. Returns ``""`` if no usable
    time keyword is found.
    """
    for key in TIME_KEYS:
        if key not in header:
            continue
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            continue
        jd = value + MJD_TO_JD if key.upper().startswith("MJD") else value
        try:
            return Time(jd, format="jd").datetime.strftime("%y%m%d")
        except (ValueError, OverflowError):
            continue
    return ""


def date_from_header(header) -> str:
    """Return YYMMDD for output-file naming, robust to header convention drift.

    Prefers the calendar-date keywords (``DAY-OBS`` then ``DATE-OBS``), handling
    both compact LCO-style values (``20250416``) and dashed values that may be
    non-zero-padded and carry a time component (e.g. MuSCAT2 ``2020-3-5`` ->
    ``200305``). When neither is usable, falls back to the canonical numeric
    time keywords (:data:`TIME_KEYS`, e.g. MuSCAT2's ``MJD-STRT``) -- see
    ``check_header_time`` for why ``DATE-OBS`` alone is unreliable.
    """
    raw = str(header.get("DAY-OBS", header.get("DATE-OBS", ""))).strip()
    # keep only the date portion if a time is appended (T or whitespace separated)
    date_part = raw.replace("T", " ").split()[0] if raw else ""
    if date_part:
        if "-" in date_part:
            try:
                year, month, day = (int(p) for p in date_part.split("-")[:3])
                return f"{year % 100:02d}{month:02d}{day:02d}"
            except ValueError:
                return date_part.replace("-", "")[2:]
        return date_part[2:]
    return _date_from_time_keys(header)


def build_stem(
    target: str,
    inst: str,
    date: str,
    band: str | None = None,
    site: str | None = None,
    confmode: str | None = None,
) -> str:
    target = target.replace(" ", "")
    inst_lower = inst.lower()
    suffix = ""
    if inst_lower == "sinistro" and confmode:
        confmode_str = str(confmode).lower()
        if "full" in confmode_str:
            suffix = "_full"

    if inst_lower == "sinistro" and site:
        site_str = site.lower()
        if site_str in ("lsc", "cpt", "coj", "tfn", "elp"):
            if band is None:
                return f"{target}_{inst}_{site_str}_{date}{suffix}"
            return f"{target}_{inst}_{site_str}_{band}_{date}{suffix}"
    if band is None:
        return f"{target}_{inst}_{date}{suffix}"
    return f"{target}_{inst}_{band}_{date}{suffix}"


def build_summary_stem(
    target: str,
    inst: str,
    date: str,
    bands: Iterable[str],
    site: str | None = None,
    confmode: str | None = None,
) -> str:
    """Build a summary-product stem scoped to the exact reduced band set."""
    band_token = "_".join(str(b).strip() for b in bands if str(b).strip())
    if not band_token:
        return build_stem(target, inst, date, site=site, confmode=confmode)
    return build_stem(target, inst, date, band_token, site=site, confmode=confmode)


def _savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {path}")


def parse_aper_grid(s: str) -> np.ndarray:
    """Parse ``"MIN,MAX,DR"`` into an aperture-radii grid (inclusive of MAX).

    Example: ``"10,20,2" -> [10, 12, 14, 16, 18, 20]``.
    """
    try:
        lo, hi, dr = (float(x) for x in s.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--aper_radii expects 'MIN,MAX,DR', got {s!r}"
        ) from exc
    if dr <= 0:
        raise argparse.ArgumentTypeError(f"--aper_radii step DR must be > 0, got {dr}")
    if hi < lo:
        raise argparse.ArgumentTypeError(
            f"--aper_radii MAX ({hi}) must be >= MIN ({lo})"
        )
    n = int(round((hi - lo) / dr))  # inclusive endpoint
    return lo + dr * np.arange(n + 1)


def parse_pair(s: str) -> tuple[float, float]:
    """Parse ``"RIN,ROUT"`` into a validated ``(rin, rout)`` tuple."""
    try:
        rin, rout = (float(x) for x in s.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--annulus expects 'RIN,ROUT', got {s!r}"
        ) from exc
    if rout <= rin:
        raise argparse.ArgumentTypeError(
            f"--annulus ROUT ({rout}) must be > RIN ({rin})"
        )
    return rin, rout


def parse_trim(s: str) -> tuple[int, int]:
    """Parse ``"Y,X"`` into a validated ``(y, x)`` trim tuple."""
    try:
        y, x = (int(v) for v in s.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--ccd_trim expects 'Y,X', got {s!r}"
        ) from exc
    if y < 0 or x < 0:
        raise argparse.ArgumentTypeError(f"--ccd_trim values must be >= 0, got {s!r}")
    return y, x


def parse_avoid_nearby_star(s: str) -> float | str:
    """Parse ``--avoid_nearby_star`` into an arcsec value or ``"auto"``."""
    s = str(s).strip().lower()
    if s in ("", AVOID_NEARBY_STAR_AUTO):
        return AVOID_NEARBY_STAR_AUTO
    try:
        value = float(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--avoid_nearby_star expects a positive arcsec value or 'auto', got {s!r}"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"--avoid_nearby_star must be > 0 arcsec, got {value}"
        )
    return value


def _zscale(data: np.ndarray) -> np.ndarray:
    """Return z-scaled image data clipped to the display interval [0, 1]."""
    scaled = np.asarray(z_scale(data), dtype=float)
    return np.clip(np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def aper_radii_pix(r: dict):
    """Aperture radii / annulus in pixels for plotting.

    When the reduction used ``--aper_unit fwhm`` (``r["scale"]`` is True) the
    stored radii are FWHM multiples; multiply by the reference FWHM to draw
    them in pixel coordinates.
    """
    radii = np.asarray(r["aper_radii"], dtype=float)
    rin, rout = float(r["rin"]), float(r["rout"])
    if r.get("scale"):
        fwhm = float(r["ref"].fwhm)
        return radii * fwhm, rin * fwhm, rout * fwhm
    return radii, rin, rout


def _format_pix_value(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value), rtol=0.0, atol=1e-6):
        return str(int(round(value)))
    return f"{value:.0f}".rstrip("0").rstrip(".")


def aperture_geometry_title(radii_pix, rin_pix: float, rout_pix: float) -> str:
    radii = np.asarray(radii_pix, dtype=float)
    rmin = radii[0] if len(radii) else np.nan
    rmax = radii[-1] if len(radii) else np.nan
    dr = np.median(np.diff(radii)) if len(radii) > 1 else 0.0
    return (
        "apertures: "
        f"r=({_format_pix_value(rmin)}, {_format_pix_value(rmax)}) "
        f"dr={_format_pix_value(dr)}; "
        f"annuli=({_format_pix_value(rin_pix)}, {_format_pix_value(rout_pix)}) pix"
    )


def _header_float(header, key: str) -> float:
    value = header.get(key)
    try:
        return float(value) if value is not None else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _header_airmass(header, keyword: str) -> float:
    """Airmass from a header keyword, guarding against telemetry dropouts.

    Some frames record a blank/zero pointing (e.g. SECZ=0, RA/DEC empty) when the
    telescope control system briefly loses tracking. Airmass is physically >= 1,
    so any non-finite or sub-unity value is treated as missing (NaN) rather than
    entering the design matrix as a spurious airmass=0 outlier.
    """
    value = _header_float(header, keyword)
    return value if np.isfinite(value) and value >= 1.0 else float("nan")


def ref_header_desc(ref, label: str, details: list[str] | None = None) -> str:
    airmass_keyword = (
        getattr(getattr(ref, "telescope", None), "keyword_airmass", "AIRMASS")
        or "AIRMASS"
    )
    focus = _header_float(ref.header, "FOCPOSN")
    z = _header_airmass(ref.header, airmass_keyword)
    exptime = _header_float(ref.header, "EXPTIME")

    focus_str = f"{focus:.1f}" if not np.isnan(focus) else "nan"
    z_str = f"{z:.1f}" if not np.isnan(z) else "nan"
    exptime_str = f"{exptime:.0f}" if not np.isnan(exptime) else "nan"

    parts = list(details or [])
    parts.extend([f"focus={focus_str}", f"airmass={z_str}", f"exptime={exptime_str}s"])
    return f"{label} ({' '.join(parts)})"


class MeasurePeaks(Block):
    def __init__(self, name=None):
        super().__init__(name=name)

    def run(self, image):
        peaks = []
        for x, y in image.sources.coords:
            ix, iy = int(round(x)), int(round(y))
            h = 5
            cutout = image.data[
                max(0, iy - h) : min(image.shape[0], iy + h + 1),
                max(0, ix - h) : min(image.shape[1], ix + h + 1),
            ]
            if cutout.size > 0:
                peaks.append(float(np.nanmax(cutout)))
            else:
                peaks.append(float("nan"))
        image.computed["peaks"] = np.array(peaks)


class FilterPointSources(Block):
    """Filter image sources to keep only PointSource instances."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def run(self, image):
        sources = image.sources.sources
        mask = np.array([isinstance(s, PointSource) for s in sources])
        filtered = [s for s, keep in zip(sources, mask) if keep]
        for i, s in enumerate(filtered):
            s.i = i
        image.sources = Sources(filtered, type="PointSource")


class MaskBadPixels(Block):
    """Mask bad pixels in image data using a bad pixel map.

    Bad pixels (hot pixels, dead pixels, cosmic rays) are marked as NaN
    so they don't contribute to aperture photometry. This allows the
    photometry to be performed with the remaining valid pixels while
    avoiding bias from detector defects.
    """

    def __init__(self, bad_pixel_map: np.ndarray | None = None, name=None):
        super().__init__(name=name)
        self.bad_pixel_map = bad_pixel_map

    def run(self, image):
        if self.bad_pixel_map is None:
            return

        # Ensure bad pixel map matches image dimensions
        if self.bad_pixel_map.shape != image.data.shape:
            logger.warning(
                f"bad pixel map shape {self.bad_pixel_map.shape} does not match "
                f"image shape {image.data.shape}; skipping bad pixel masking"
            )
            return

        # Mark bad pixels as NaN
        bad_mask = self.bad_pixel_map.astype(bool)
        n_masked = int(np.sum(bad_mask))
        if n_masked > 0:
            image.data[bad_mask] = np.nan
            logger.debug(f"masked {n_masked} bad pixels in {Path(image.path).name}")


# --------------------------- reference building ---------------------------


def reference_sequence(
    ccd_trim_size_yx: tuple[int, int] = CCD_TRIM_SIZE_YX,
    max_num_stars: int = MAX_NUM_STARS,
    min_star_separation: float = MIN_STAR_SEPARATION,
    cutout_size: int = CUTOUT_SIZE,
    min_area: int = MIN_STAR_AREA,
    bad_pixel_map: np.ndarray | None = None,
) -> Sequence:
    """Calibration sequence run on the per-band reference frame.

    Parameters
    ----------
    bad_pixel_map : np.ndarray, optional
        Boolean array marking bad pixels (True = bad). Applied to reference
        image to mask detector defects before source detection.
    """
    n_detect = max(int(max_num_stars * DETECT_NUM_STARS_FACTOR), max_num_stars + 5)
    return Sequence(
        [
            blocks.Trim(ccd_trim_size_yx),
            MaskBadPixels(bad_pixel_map),
            # blocks.AutoSourceDetection(
            blocks.PointSourceDetection(
                n=n_detect,
                min_area=min_area,
                min_separation=min_star_separation,
            ),
            # FilterPointSources(),
            blocks.Cutouts(shape=cutout_size, wcs=True),
            blocks.MedianEPSF(),
            blocks.psf.Gaussian2D(),
            blocks.CentroidQuadratic(),
            blocks.AperturePhotometry(),
            blocks.AnnulusBackground(),
        ]
    )


def find_target_index(ref: FITSImage, target_coord: SkyCoord) -> int:
    try:
        wcs = ref.wcs
        if wcs is not None and hasattr(wcs, "pixel_to_world"):
            coords = np.array([s.coords for s in ref.sources])
            if len(coords) > 0:
                stars_radec = wcs.pixel_to_world(*coords.T)
                if _skycoord_has_finite_data(stars_radec):
                    idx, d2d, _ = target_coord.match_to_catalog_sky(stars_radec)
                    if float(np.atleast_1d(d2d.arcsec)[0]) < 5.0:
                        return int(idx)
    except Exception as e:
        logger.warning(
            f"WCS-based target cross-match failed ({e}); use --tID for manual override"
        )
    logger.warning("defaulting to source 0 (brightest); verify with --tID")
    return 0


def _contaminant_seps(
    seps: np.ndarray, mags: np.ndarray, target_mag: float
) -> np.ndarray:
    """Separations [pix] of neighbours that contaminate the target.

    A neighbour contaminates when it contributes at least ~10% of the target
    flux, i.e. ``mag - target_mag < CONTAM_DMAG``. ``NaN`` magnitudes are
    treated as non-contaminating (too faint / unmeasured). The result is sorted
    ascending so callers can take the nearest contaminant first.
    """
    seps = np.asarray(seps, dtype=float)
    mags = np.asarray(mags, dtype=float)
    is_contaminant = (mags - target_mag) < CONTAM_DMAG  # NaN -> False
    contam = np.sort(seps[is_contaminant & np.isfinite(seps)])
    return contam


def _sky_annulus_pix(
    fwhm: float,
    contam_seps: np.ndarray,
    annulus_pix: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Inner/outer sky-annulus radii [pix] that avoid enclosing a contaminant.

    The annulus is nominally defined by ``annulus_pix`` (defaulting to ``(20.0, 30.0)``).
    If a contaminant falls within the nominal ring, the ring is
    shifted inward to sit just inside the nearest such source.
    """
    if annulus_pix is None:
        annulus_pix = (20.0, 30.0)
    rin_nom, rout_nom = map(float, annulus_pix)
    width = rout_nom - rin_nom
    rout = rout_nom
    intruding = contam_seps[contam_seps < rout + CONTAM_MARGIN_PIX]
    if len(intruding):
        rout = float(intruding.min()) - CONTAM_MARGIN_PIX
    rin = rout - width
    # keep the ring physical: positive inner radius above the minimum aperture
    rin = max(rin, fwhm + APER_STEP_PIX)
    rout = max(rout, rin + fwhm)
    return rin, rout


def _aperture_radii_pix(fwhm: float, rin: float) -> np.ndarray:
    """Aperture-radii grid [pix] from the target FWHM up to the inner annulus.

    Spaced by ``APER_STEP_PIX`` and always non-empty (a single ``[fwhm]`` radius
    when ``rin`` leaves no room).
    """
    radii = np.arange(fwhm, rin, APER_STEP_PIX)
    if len(radii) == 0:
        radii = np.array([fwhm])
    return radii


# in-memory Gaia cache: query once per run, reuse across the run's bands
_gaia_cache: dict = {}


def _gaia_catalog_df(ref, target_index, target_coord, pixscale):
    """Gaia catalog DataFrame around the target, querying live and refreshing the
    on-disk cache (``CACHE_DIR/gaia``); if the live query fails, fall back to a
    previous run's cached result. Returns ``None`` when neither is available.
    Queried once per process and reused across the run's bands.
    """
    cache_path = coord_cache_path("gaia", target_coord, GAIA_CUTOUT[0])
    if cache_path in _gaia_cache:  # same target, another band of this run
        return _gaia_cache[cache_path]

    df = load_cached_df(cache_path)
    if df is not None:
        _gaia_cache[cache_path] = df
        return df

    try:
        c = ref.copy()
        c.metadata["pixel_scale"] = float(pixscale)  # required before Gaia query
        c = catalogs.GaiaCatalog(mode="replace")(c)
        df = c.catalogs["gaia"]
        is_valid = False
        if df is not None and len(df) and target_coord is not None:
            from astropy.coordinates import SkyCoord

            gaia_coords = SkyCoord(df.ra.values, df.dec.values, unit="deg")
            min_sep = float(np.min(target_coord.separation(gaia_coords).arcsec))
            if min_sep <= 15.0:
                is_valid = True
            else:
                logger.warning(
                    f"Queried Gaia catalog center is far from target coordinates "
                    f"(min separation {min_sep:.1f} arcsec); skipping caching to prevent contamination."
                )
        if is_valid and save_cached_df(cache_path, df):  # refresh cache whenever online
            logger.info(f"cached Gaia result -> {cache_path}")
    except Exception as exc:  # noqa: BLE001 - degrade to cache then FWHM-only
        is_timeout = "timeout" in str(exc).lower() or "time out" in str(exc).lower()
        if is_timeout:
            logger.warning(f"Gaia query timed out: {exc}")
        df = load_cached_df(cache_path)
        if df is not None:
            logger.warning(
                f"Gaia query unavailable ({exc}); using cached result {cache_path}"
            )
        else:
            logger.warning(
                f"Gaia query unavailable ({exc}) and no cache; using FWHM-only geometry"
            )
    _gaia_cache[cache_path] = df  # store even None to avoid re-querying per band
    return df


def gaia_aperture_radii(
    ref: FITSImage,
    target_index: int,
    target_coord: SkyCoord,
    annulus_pix: tuple[float, float] | None = None,
):
    """Size the sky annulus and aperture radii from Gaia contamination.

    The minimum aperture is the target FWHM and the maximum is the inner
    sky-annulus radius; the annulus is placed to exclude any Gaia source
    contributing >=10% of the target flux (``CONTAM_DMAG``). The Gaia catalog is
    fetched via ``_gaia_catalog_df`` (live query, cached to disk, with an offline
    cache fallback); when no catalog is available the helpers run with no
    contaminants, giving an FWHM-only annulus.

    Returns ``(aper_radii, rin, rout)`` in pixels.
    """
    fwhm = float(ref.fwhm)
    pixscale = float(ref.header.get("PIXSCALE", ref.telescope.pixel_scale))
    df = _gaia_catalog_df(ref, target_index, target_coord, pixscale)

    contam = np.array([])
    if df is not None and len(df):
        try:
            gaia_coords = SkyCoord(df.ra.values, df.dec.values, unit="deg")
            sep_pix = target_coord.separation(gaia_coords).arcsec / pixscale
            mags = df.phot_g_mean_mag.values
            order = np.argsort(sep_pix)  # nearest first; row 0 = the target itself
            target_mag = float(mags[order][0])
            contam = _contaminant_seps(sep_pix[order][1:], mags[order][1:], target_mag)
        except Exception as exc:  # noqa: BLE001 - bad catalog must not be fatal
            logger.warning(
                f"contamination computation failed ({exc}); using FWHM-only geometry"
            )
            contam = np.array([])

    rin, rout = _sky_annulus_pix(fwhm, contam, annulus_pix=annulus_pix)
    aper_radii = _aperture_radii_pix(fwhm, rin)
    logger.info(
        f"apertures: {len(aper_radii)} radii in [{fwhm:.0f}, {rin:.0f}] px, "
        f"annulus ({rin:.0f}, {rout:.0f}) px, {len(contam)} contaminants (>=10% flux)"
    )
    return aper_radii, rin, rout


def _finite_world_to_pixel(wcs, coord: SkyCoord | None) -> tuple[float, float] | None:
    if wcs is None or coord is None or not hasattr(wcs, "world_to_pixel"):
        return None
    # image.wcs always returns a WCS() object (never None), even when the FITS
    # header has no celestial keywords. WCS(None) may still return finite pixel
    # coordinates for any input, so we must check has_celestial before trusting
    # the projection.
    if not getattr(wcs, "has_celestial", False):
        return None
    try:
        x, y = wcs.world_to_pixel(coord)
        x = float(np.asarray(x))
        y = float(np.asarray(y))
    except Exception:
        return None
    if not np.all(np.isfinite([x, y])):
        return None
    return x, y


def _wcs_can_project(wcs, coord: SkyCoord | None) -> bool:
    return _finite_world_to_pixel(wcs, coord) is not None


def _reference_file_has_usable_wcs(ref_file, target_coord: SkyCoord | None) -> bool:
    if target_coord is None:
        return False
    try:
        ref = FITSImage(ref_file)
        return _wcs_can_project(getattr(ref, "wcs", None), target_coord)
    except Exception as exc:  # noqa: BLE001 - WCS preflight must not fail the run
        logger.warning(f"{Path(ref_file).name}: WCS preflight failed ({exc})")
        return False


def _order_bands_for_target_id_inference(
    ordered_bands: list[str], ref_wcs_ok_by_band: dict[str, bool]
) -> list[str]:
    if not ref_wcs_ok_by_band:
        return ordered_bands
    if not any(ref_wcs_ok_by_band.values()) or all(ref_wcs_ok_by_band.values()):
        return ordered_bands
    return [band for band in ordered_bands if ref_wcs_ok_by_band.get(band, False)] + [
        band for band in ordered_bands if not ref_wcs_ok_by_band.get(band, False)
    ]


def _nearest_source_index(
    ref,
    pixel_position: np.ndarray | tuple[float, float] | list[float],
    max_distance: float = _TARGET_PIXEL_INFERENCE_TOLERANCE_PX,
) -> tuple[int, float]:
    positions = np.array([s.coords for s in ref.sources], dtype=float)
    if len(positions) == 0:
        raise ValueError("cannot infer target ID: reference has no detected sources")
    target_position = np.asarray(pixel_position, dtype=float)
    distances = np.linalg.norm(positions - target_position, axis=1)
    nearest = int(np.argmin(distances))
    distance = float(distances[nearest])
    if distance > max_distance:
        raise ValueError(
            f"inferred target position ({target_position[0]:.1f}, {target_position[1]:.1f}) "
            f"is {distance:.1f} px from the nearest detected source "
            f"(> {max_distance:.1f} px)"
        )
    return nearest, distance


def _target_pixel_override_for_band(
    manual_target_index: int | None,
    self_reference: bool,
    inferred_target_positions: list[np.ndarray],
    ref_has_usable_wcs: bool,
) -> np.ndarray | None:
    if manual_target_index is not None:
        return None
    if self_reference and inferred_target_positions and not ref_has_usable_wcs:
        return np.median(np.asarray(inferred_target_positions, dtype=float), axis=0)
    return None


def _image_center_xy(ref: FITSImage) -> tuple[float, float]:
    return ref.data.shape[1] / 2, ref.data.shape[0] / 2


def _target_pixel_or_center(
    ref: FITSImage, target_coord: SkyCoord
) -> tuple[float, float]:
    pix = _finite_world_to_pixel(getattr(ref, "wcs", None), target_coord)
    if pix is not None:
        return pix
    if getattr(ref, "wcs", None) is None:
        logger.warning(
            "WCS-based target localization unavailable; forcing target at image center"
        )
    else:
        logger.warning(
            "WCS-based target localization returned non-finite coordinates; "
            "forcing target at image center"
        )
    return _image_center_xy(ref)


def _skycoord_has_finite_data(coord) -> bool:
    if not isinstance(coord, SkyCoord):
        return False
    try:
        return bool(
            np.all(np.isfinite(coord.ra.deg)) and np.all(np.isfinite(coord.dec.deg))
        )
    except Exception:
        return False


def resolve_edge_margin(edge_margin: int | None, cutout_size: int) -> int:
    """Resolve the effective edge margin [pix].

    ``None`` means auto: half the cutout size, so a star sitting exactly at the
    margin still has its full PSF cutout box on-chip. An explicit value is used
    as-is; ``0`` (or negative) disables edge exclusion.
    """
    if edge_margin is None:
        return int(cutout_size) // 2
    return int(edge_margin)


def resolve_avoid_nearby_star_arcsec(
    avoid_nearby_star: float | str | None,
    fwhm_pix: float,
    pixscale: float,
) -> float | None:
    """Resolve ``--avoid_nearby_star`` to an effective arcsec threshold."""
    if avoid_nearby_star is None:
        return None
    if avoid_nearby_star == AVOID_NEARBY_STAR_AUTO:
        return max(
            AVOID_NEARBY_STAR_AUTO_FWHM * float(fwhm_pix) * float(pixscale),
            AVOID_NEARBY_STAR_AUTO_MIN_ARCSEC,
        )
    return float(avoid_nearby_star)


def _edge_source_indices(ref, margin: int, target_index: int) -> list[int]:
    """Indices of detected sources within ``margin`` px of any image border.

    Source centroids are ``(x, y)`` and ``ref.shape`` is ``(ny, nx)``. The
    ``target_index`` is always removed from the result so the target is never
    dropped from photometry, even when it sits near an edge. Returns a sorted
    list of comparison-star indices (empty when ``margin <= 0`` or no sources).
    """
    if margin <= 0:
        return []
    coords = np.array([s.coords for s in ref.sources], dtype=float)
    if coords.size == 0:
        return []
    ny, nx = ref.shape
    x, y = coords[:, 0], coords[:, 1]
    near = (x < margin) | (x > nx - 1 - margin) | (y < margin) | (y > ny - 1 - margin)
    if 0 <= target_index < near.size:
        near[target_index] = False
    return sorted(int(i) for i in np.nonzero(near)[0])


def _saturated_source_indices(
    ref, saturation_level: float | None, target_index: int
) -> list[int]:
    """Indices of detected sources whose peak pixel value exceeds saturation.

    Uses ``s.peak`` (set during source detection from ``region.intensity_max``).
    The ``target_index`` is always protected so the target is never dropped
    from photometry. Returns a sorted list of comparison-star indices (empty
    when ``saturation_level`` is ``None``, ``<= 0``, or no sources).
    """
    if saturation_level is None or saturation_level <= 0:
        return []
    n = len(ref.sources)
    if n == 0:
        return []
    peaks = np.array([s.peak for s in ref.sources])
    saturated = peaks >= saturation_level
    if 0 <= target_index < len(saturated):
        saturated[target_index] = False
    return sorted(int(i) for i in np.nonzero(saturated)[0])


def _nearby_detected_source_indices(
    ref,
    max_sep_pix: float,
    target_index: int,
) -> list[int]:
    """Detected-source indices whose nearest detected neighbour is too close."""
    if max_sep_pix <= 0:
        return []
    coords = np.array([s.coords for s in ref.sources], dtype=float)
    if coords.shape[0] < 2:
        return []
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nearest = np.asarray(dists[:, 1], dtype=float)
    near = np.isfinite(nearest) & (nearest <= float(max_sep_pix))
    if 0 <= target_index < near.size:
        near[target_index] = False
    return sorted(int(i) for i in np.nonzero(near)[0])


def _nearby_gaia_source_indices(
    ref,
    gaia_df,
    max_sep_arcsec: float,
    target_index: int,
) -> list[int]:
    """Detected-source indices with a close Gaia neighbour.

    A matched source is rejected when another Gaia source lies within
    ``max_sep_arcsec`` and is bright enough to contribute at least ~10% of the
    matched source flux (same ``CONTAM_DMAG`` criterion used for the target
    aperture heuristic). When Gaia magnitudes are unavailable, any neighbour
    within the separation threshold is considered contaminating.
    """
    if (
        gaia_df is None
        or not len(gaia_df)
        or getattr(ref, "wcs", None) is None
        or len(ref.sources) < 2
    ):
        return []
    try:
        coords = np.array([s.coords for s in ref.sources], dtype=float)
        source_coords = ref.wcs.pixel_to_world(*coords.T)
        if not _skycoord_has_finite_data(source_coords):
            return []
        gaia_coords = SkyCoord(gaia_df.ra.values, gaia_df.dec.values, unit="deg")
        match_idx, match_sep, _ = source_coords.match_to_catalog_sky(gaia_coords)
        match_sep_arcsec = np.asarray(np.atleast_1d(match_sep.arcsec), dtype=float)
        mags = None
        if "phot_g_mean_mag" in gaia_df:
            mags = np.asarray(gaia_df.phot_g_mean_mag.values, dtype=float)
        nearby: list[int] = []
        for src_idx, (gidx, sep_arcsec) in enumerate(zip(match_idx, match_sep_arcsec)):
            if src_idx == target_index or not np.isfinite(sep_arcsec):
                continue
            if sep_arcsec > AVOID_NEARBY_STAR_MATCH_ARCSEC:
                continue
            seps = np.asarray(
                gaia_coords[gidx].separation(gaia_coords).arcsec, dtype=float
            )
            mask = np.ones(len(seps), dtype=bool)
            mask[int(gidx)] = False
            if mags is not None and np.isfinite(mags[int(gidx)]):
                contam = _contaminant_seps(
                    seps[mask], mags[mask], float(mags[int(gidx)])
                )
                if len(contam) and float(contam.min()) <= float(max_sep_arcsec):
                    nearby.append(int(src_idx))
            else:
                neigh = np.sort(seps[mask & np.isfinite(seps)])
                if len(neigh) and float(neigh[0]) <= float(max_sep_arcsec):
                    nearby.append(int(src_idx))
        return sorted(set(nearby))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"nearby-star Gaia screening failed ({exc}); using detected-source fallback only"
        )
        return []


def build_reference(
    ref_file,
    target_coord,
    aper_radii=None,
    rin=None,
    rout=None,
    scale=False,
    ccd_trim_size_yx: tuple[int, int] = CCD_TRIM_SIZE_YX,
    max_num_stars: int = MAX_NUM_STARS,
    min_star_separation: float = MIN_STAR_SEPARATION,
    cutout_size: int = CUTOUT_SIZE,
    target_index_override: int | None = None,
    target_pixel_override: np.ndarray | tuple[float, float] | None = None,
    min_area: int = MIN_STAR_AREA,
    plot_gaia_sources: bool = False,
    edge_margin: int | None = EDGE_MARGIN_PIX,
    avoid_nearby_star: float | str | None = None,
    annulus_pix: tuple[float, float] | None = None,
    bad_pixel_map: np.ndarray | None = None,
):
    """Build the reference image, target index and aperture geometry.

    PIXSCALE and saturation are read from the reference image header
    (not hardcoded or taken from a probe frame).

    If ``aper_radii`` is provided, the explicit grid (and ``rin``/``rout``) is
    used and the Gaia heuristic is skipped. ``scale`` selects pixel
    (``False``) vs FWHM (``True``) units for the photometry blocks.

    If ``target_index_override`` is given, it bypasses the Gaia cross-match.
    If ``target_pixel_override`` is given, the nearest detected source to that
    pixel coordinate is used as the target.

    When ``plot_gaia_sources`` is set and the reference WCS can project the
    target coordinate, the Gaia catalog around the target is fetched (or reused
    from cache) and returned under ``gaia_df`` so the aperture/stack zoom plots
    can overlay Gaia source positions, even when the Gaia contamination
    heuristic is bypassed by an explicit aperture grid.

    Parameters
    ----------
    bad_pixel_map : np.ndarray, optional
        Boolean array marking bad pixels (True = bad pixel). Applied to reference
        image before source detection to mask detector defects.
    """
    ref = FITSImage(ref_file)
    instrument = get_instrument(ref.header)
    if instrument == "sinistro":
        confmode = str(ref.header.get("CONFMODE", ""))
        pix_key = "sinistro_2x2" if "2x2" in confmode else "sinistro_full"
    else:
        pix_key = instrument
    pixel_scale = float(ref.header.get("PIXSCALE", PIXSCALES.get(pix_key, 0.267)))
    band_key = ref.header.get("FILTER", "")
    try:
        sat_all = get_saturation_from_header(ref.header)
        saturation = sat_all.get(band_key) if isinstance(sat_all, dict) else None
    except Exception:  # noqa: BLE001
        saturation = None
    ref.telescope.saturation = saturation
    ref.telescope.pixel_scale = pixel_scale
    logger.info(
        f"{Path(ref_file).name}: PIXSCALE={pixel_scale} "
        f"saturation={saturation} instrument={instrument}"
    )
    reference_sequence(
        ccd_trim_size_yx=ccd_trim_size_yx,
        max_num_stars=max_num_stars,
        min_star_separation=min_star_separation,
        cutout_size=cutout_size,
        min_area=min_area,
        bad_pixel_map=bad_pixel_map,
    ).run(ref, show_progress=False)

    match_found = False
    if (
        target_index_override is None
        and target_pixel_override is None
        and target_coord is not None
        and ref.wcs is not None
    ):
        coords = np.array([s.coords for s in ref.sources])
        if len(coords) > 0:
            try:
                stars_radec = ref.wcs.pixel_to_world(*coords.T)
            except Exception as e:
                stars_radec = None
                logger.warning(
                    f"WCS-based source projection failed ({e}); "
                    "skipping astrometric match"
                )
            if _skycoord_has_finite_data(stars_radec):
                idx, d2d, _ = target_coord.match_to_catalog_sky(stars_radec)
                if float(np.atleast_1d(d2d.arcsec)[0]) < 5.0:
                    match_found = True
            else:
                logger.warning(
                    f"WCS returned non-SkyCoord from pixel_to_world "
                    f"(type={type(stars_radec).__name__}); skipping astrometric match"
                )

        if not match_found:
            logger.warning(
                "Target not found in detected sources (separation > 5 arcsec) "
                "and WCS-based localization failed. Falling back to source 0 "
                "(brightest detected star). Use --tID to override."
            )

    defaulted_to_brightest = False
    if target_index_override is None and target_pixel_override is None:
        defaulted_to_brightest = not match_found

    if target_index_override is not None:
        target_index = target_index_override
    elif target_pixel_override is not None:
        target_index, target_pixel_distance = _nearest_source_index(
            ref, target_pixel_override
        )
        logger.info(
            f"inferred target idx {target_index} from pixel position "
            f"({float(target_pixel_override[0]):.1f}, {float(target_pixel_override[1]):.1f}); "
            f"nearest source distance {target_pixel_distance:.1f} px"
        )
    else:
        target_index = find_target_index(ref, target_coord)
    # Validate against the actual number of kept sources (which may be fewer
    # than max_num_stars in sparse fields). An out-of-range target index would
    # otherwise surface as a cryptic IndexError deep inside auto_diff_1d.
    n_sources = len(ref.sources)
    if not (0 <= target_index < n_sources):
        raise ValueError(
            f"--tID {target_index} out of range: only {n_sources} sources kept "
            f"(valid 0..{n_sources - 1}); increase --max_num_stars or pick a "
            f"lower --tID"
        )
    aper_radii_was_custom = aper_radii is not None
    if aper_radii_was_custom:
        unit = "fwhm" if scale else "pix"
        logger.info(
            f"using custom apertures: {len(aper_radii)} radii in "
            f"[{aper_radii.min():g}, {aper_radii.max():g}] {unit}, "
            f"annulus ({rin:g}, {rout:g}) {unit}"
        )
    else:
        aper_radii, rin, rout = gaia_aperture_radii(
            ref, target_index, target_coord, annulus_pix=annulus_pix
        )
    # The default path above already queried (and cached) Gaia; for an explicit
    # aperture grid we only query when the overlay was requested. Reusing the
    # per-run cache means this never triggers a second network round-trip.
    gaia_df = None
    overlay_wcs_ok = _wcs_can_project(getattr(ref, "wcs", None), target_coord)
    if plot_gaia_sources and not overlay_wcs_ok:
        logger.warning("Gaia overlay skipped: reference image has no usable WCS")
    if (
        not aper_radii_was_custom or plot_gaia_sources or avoid_nearby_star is not None
    ) and overlay_wcs_ok:
        gaia_df = _gaia_catalog_df(ref, target_index, target_coord, pixel_scale)
    logger.info(
        f"reference {Path(ref_file).name}: FWHM {float(ref.fwhm):.2f} px, "
        f"target idx {target_index}"
    )
    margin = resolve_edge_margin(edge_margin, cutout_size)
    edge_cids = _edge_source_indices(ref, margin, target_index)
    if margin > 0:
        # The target is protected from edge_cids, but warn when it lands inside
        # the margin: its aperture/cutout may spill off-chip and it can drift out
        # of the FOV during the night — a data-quality heads-up, not a drop.
        ny, nx = ref.shape
        tx, ty = ref.sources[target_index].coords
        if tx < margin or tx > nx - 1 - margin or ty < margin or ty > ny - 1 - margin:
            logger.warning(
                f"target (idx {target_index}) is within {margin} px of a CCD "
                f"edge at ({tx:.0f}, {ty:.0f}); keeping it but its aperture may "
                f"clip the border"
            )
        if edge_cids:
            logger.info(
                f"edge exclusion ({margin} px margin): {len(edge_cids)} "
                f"comparison star(s) flagged near border: {edge_cids}"
            )
    nearby_cids: list[int] = []
    nearby_sep_arcsec = resolve_avoid_nearby_star_arcsec(
        avoid_nearby_star, float(ref.fwhm), pixel_scale
    )
    nearby_sep_pix = (
        float(nearby_sep_arcsec) / pixel_scale
        if nearby_sep_arcsec is not None
        else None
    )
    if nearby_sep_arcsec is not None:
        detected_nearby = _nearby_detected_source_indices(
            ref, nearby_sep_pix, target_index
        )
        gaia_nearby = _nearby_gaia_source_indices(
            ref, gaia_df, nearby_sep_arcsec, target_index
        )
        nearby_cids = sorted(set(detected_nearby) | set(gaia_nearby))
        if nearby_cids:
            mode = "Gaia+detected" if gaia_nearby else "detected"
            logger.info(
                f"nearby-star exclusion ({nearby_sep_arcsec:.2f} arcsec = "
                f"{nearby_sep_pix:.1f} px, {mode}): {len(nearby_cids)} "
                f"comparison star(s) flagged: {nearby_cids}"
            )
    saturated_cids = _saturated_source_indices(ref, saturation, target_index)
    if saturated_cids:
        logger.info(
            f"saturation exclusion ({saturation} ADU): {len(saturated_cids)} "
            f"comparison star(s) flagged as saturated: {saturated_cids}"
        )
    return dict(
        ref=ref,
        target_index=target_index,
        aper_radii=aper_radii,
        rin=rin,
        rout=rout,
        scale=scale,
        gaia_df=gaia_df,
        edge_cids=edge_cids,
        saturated_cids=saturated_cids,
        nearby_cids=nearby_cids,
        nearby_sep_arcsec=nearby_sep_arcsec,
        nearby_sep_pix=nearby_sep_pix,
        defaulted_to_brightest=defaulted_to_brightest,
    )


# --------------------------- per-band reduction ---------------------------


def photometry_sequence(
    ref,
    aper_radii,
    rin,
    rout,
    scale=False,
    ccd_trim_size_yx: tuple[int, int] = CCD_TRIM_SIZE_YX,
    max_num_stars: int = MAX_NUM_STARS,
    min_star_separation: float = MIN_STAR_SEPARATION,
    cutout_size: int = CUTOUT_SIZE,
    n_stars_align: int | None = None,
    target_index: int = 0,
    min_area: int = MIN_STAR_AREA,
    bad_pixel_map: np.ndarray | None = None,
) -> SequenceParallel:
    """Parallel per-image photometry sequence (mirrors the notebook).

    Parameters
    ----------
    bad_pixel_map : np.ndarray, optional
        Boolean array marking bad pixels (True = bad). Applied to all frames
        to mask detector defects before photometry.
    """
    if n_stars_align is None:
        n_stars_align = max_num_stars
    blocks_list = [
        blocks.Trim(ccd_trim_size_yx),
        MaskBadPixels(bad_pixel_map),
        # blocks.AutoSourceDetection(
        blocks.PointSourceDetection(
            n=max_num_stars,
            min_area=min_area,
            min_separation=min_star_separation,
            min_sources=2,
        ),
        # FilterPointSources(),
        blocks.Cutouts(shape=cutout_size),
        blocks.MedianEPSF(),
        blocks.Gaussian2D(ref),
    ]
    if n_stars_align >= 3:
        blocks_list.append(blocks.ComputeTransformTwirl(ref, n=n_stars_align))
        blocks_list.append(
            blocks.AlignReferenceSources(ref, discard_tolerance=ALIGN_DISCARD_TOLERANCE)
        )
    blocks_list.extend(
        [
            blocks.CentroidQuadratic(),
            blocks.AperturePhotometry(aper_radii, scale=scale),
            blocks.AnnulusBackground(rin=rin, rout=rout, scale=scale),
            MeasurePeaks(),
            blocks.Del("data", "cutouts"),
        ]
    )

    # Resolve the airmass header keyword from the telescope config so the header
    # keyword takes precedence over a hardcoded name (e.g. MuSCAT writes SECZ, not
    # AIRMASS). Falls back to "AIRMASS" when no telescope keyword is available.
    airmass_keyword = (
        getattr(getattr(ref, "telescope", None), "keyword_airmass", "AIRMASS")
        or "AIRMASS"
    )

    return SequenceParallel(
        blocks=blocks_list,
        data_blocks=[
            blocks.GetFluxes(
                "fwhm",
                airmass=lambda im: _header_airmass(im.header, airmass_keyword),
                dx=lambda im: (
                    im.transform.translation[0]
                    if hasattr(im, "transform") and im.transform is not None
                    else float("nan")
                ),
                dy=lambda im: (
                    im.transform.translation[1]
                    if hasattr(im, "transform") and im.transform is not None
                    else float("nan")
                ),
                peak=lambda im: (
                    im.computed["peaks"][target_index]
                    if "peaks" in im.computed
                    and len(im.computed["peaks"]) > target_index
                    else float("nan")
                ),
            ),
        ],
    )


def run_band(
    band,
    files,
    ref_file,
    target_coord,
    aper_radii=None,
    rin=None,
    rout=None,
    scale=False,
    ccd_trim_size_yx: tuple[int, int] = CCD_TRIM_SIZE_YX,
    max_num_stars: int = MAX_NUM_STARS,
    min_star_separation: float = MIN_STAR_SEPARATION,
    cutout_size: int = CUTOUT_SIZE,
    n_stars_align: int | None = None,
    target_index_override: int | None = None,
    target_pixel_override: np.ndarray | tuple[float, float] | None = None,
    cids: list[int] | None = None,
    avoid_cids: list[int] | None = None,
    ref_source_positions: np.ndarray | None = None,
    min_area: int = MIN_STAR_AREA,
    plot_gaia_sources: bool = False,
    edge_margin: int | None = EDGE_MARGIN_PIX,
    avoid_nearby_star: float | str | None = None,
    annulus_pix: tuple[float, float] | None = None,
    nan_imputation_method: str = "linear",
    bad_pixel_map: np.ndarray | None = None,
):
    """Full reduction for a single band. Returns a result dict or ``None``.
    PIXSCALE and saturation are read from the reference image header
    inside ``build_reference``.

    When ``avoid_cids`` is given and ``ref_source_positions`` is set, the
    index list is cross-matched from the reference band's source catalog
    (pixel positions) to this band's sources via nearest-neighbor search.
    When ``ref_source_positions`` is ``None`` (the reference band itself),
    indices apply directly.

    Parameters
    ----------
    bad_pixel_map : np.ndarray, optional
        Boolean array marking bad pixels (True = bad pixel). Applied during
        photometry to mask detector defects (hot pixels, dead pixels).
    """
    logger.info(f"[{band}] {len(files)} frames; building reference")
    reference = build_reference(
        ref_file,
        target_coord,
        aper_radii=aper_radii,
        rin=rin,
        rout=rout,
        scale=scale,
        ccd_trim_size_yx=ccd_trim_size_yx,
        max_num_stars=max_num_stars,
        min_star_separation=min_star_separation,
        cutout_size=cutout_size,
        target_index_override=target_index_override,
        target_pixel_override=target_pixel_override,
        min_area=min_area,
        plot_gaia_sources=plot_gaia_sources,
        edge_margin=edge_margin,
        avoid_nearby_star=avoid_nearby_star,
        annulus_pix=annulus_pix,
        bad_pixel_map=bad_pixel_map,
    )
    ref = reference["ref"]
    target_index = reference["target_index"]

    ref_band_star_ids: list[int] | None = None
    if ref_source_positions is not None:
        this_positions = np.array([s.coords for s in ref.sources])
        if len(this_positions) > 0:
            from scipy.spatial import KDTree

            tree_ref = KDTree(ref_source_positions)
            ref_band_star_ids = []
            for i, pos in enumerate(this_positions):
                dist, nearest = tree_ref.query(pos)
                if dist < _CROSSMATCH_TOLERANCE_PX:
                    ref_band_star_ids.append(int(nearest))
                else:
                    ref_band_star_ids.append(-1)

        if len(this_positions) and 0 <= target_index < len(ref_source_positions):
            from scipy.spatial import KDTree

            tree = KDTree(this_positions)
            target_dist, mapped_target = tree.query(ref_source_positions[target_index])
            if target_dist < _CROSSMATCH_TOLERANCE_PX:
                target_index = int(mapped_target)
            else:
                logger.warning(
                    f"[{band}] target index {reference['target_index']} from reference band "
                    f"maps {target_dist:.1f} px away in this band (> {_CROSSMATCH_TOLERANCE_PX} px); "
                    "using reference-band index"
                )

    # Cross-match avoid_cids from ref-band index space -> this band's indices
    # via nearest-neighbor KDTree (ref_source_positions is pre-populated from
    # the reference band, so all bands share the same alignment frame).
    mapped_avoid: list[int] | None = None
    if avoid_cids:
        if ref_source_positions is not None:
            this_positions = np.array([s.coords for s in ref.sources])
            from scipy.spatial import KDTree

            tree = KDTree(this_positions)
            mapped_avoid = []
            for idx in avoid_cids:
                if idx >= len(ref_source_positions):
                    logger.warning(
                        f"[{band}] avoid_cid {idx}: out of range "
                        f"(ref band has {len(ref_source_positions)} sources)"
                    )
                    continue
                dist, nearest = tree.query(ref_source_positions[idx])
                if dist < _CROSSMATCH_TOLERANCE_PX:
                    mapped_avoid.append(int(nearest))
                else:
                    logger.warning(
                        f"[{band}] avoid_cid {idx}: nearest source is "
                        f"{dist:.1f} px away (> {_CROSSMATCH_TOLERANCE_PX} px); skipping"
                    )
        else:
            mapped_avoid = list(avoid_cids)

    # Cross-match cids from ref-band index space -> this band's indices
    # via nearest-neighbor KDTree (ref_source_positions is pre-populated from
    # the reference band, so all bands share the same alignment frame).
    if cids is not None:
        if ref_source_positions is not None:
            this_positions = np.array([s.coords for s in ref.sources])
            from scipy.spatial import KDTree

            tree = KDTree(this_positions)
            mapped_cids = []
            for idx in cids:
                if idx >= len(ref_source_positions):
                    logger.warning(
                        f"[{band}] cID {idx}: out of range "
                        f"(ref band has {len(ref_source_positions)} sources)"
                    )
                    continue
                dist, nearest = tree.query(ref_source_positions[idx])
                if dist < _CROSSMATCH_TOLERANCE_PX:
                    mapped_cids.append(int(nearest))
                else:
                    logger.warning(
                        f"[{band}] cID {idx}: nearest source is "
                        f"{dist:.1f} px away (> {_CROSSMATCH_TOLERANCE_PX} px); skipping"
                    )
            cids = mapped_cids
        else:
            cids = list(cids)

        # Filter avoided indices from explicit comparison list
        if mapped_avoid:
            cids = [c for c in cids if c not in mapped_avoid]
            if not cids:
                logger.warning(
                    f"[{band}] all explicit cIDs are in avoid list; "
                    f"falling back to auto-selection"
                )
                cids = None

    # Merge edge-star indices (computed on this band's reference frame, so already
    # in this band's index space) into the avoid pool. Done after the ref-band
    # cross-match so edge stars are not double-mapped. The target is already
    # protected inside build_reference, so it can never appear here.
    edge_cids = reference.get("edge_cids") or []
    if edge_cids:
        current_avoid = set(mapped_avoid or [])
        target_idx = target_index
        n_sources = len(ref.sources)
        candidates = [
            i for i in range(n_sources) if i != target_idx and i not in current_avoid
        ]
        after_edge = [i for i in candidates if i not in set(edge_cids)]
        if candidates and not after_edge:
            logger.warning(
                f"[{band}] edge exclusion ({edge_margin} px margin) would remove all comparison stars; "
                f"relaxing edge exclusion to preserve comparison pool"
            )
        else:
            mapped_avoid = sorted(current_avoid | set(edge_cids))
            # Explicit-cid mode diffs against ``cids`` directly (bypassing the avoid
            # mask), so edge stars must also be stripped from an explicit list.
            if cids is not None:
                cids = [c for c in cids if c not in set(edge_cids)]
                if not cids:
                    logger.warning(
                        f"[{band}] all explicit cIDs are near a CCD edge; "
                        f"falling back to auto-selection"
                    )
                    cids = None

    nearby_cids = reference.get("nearby_cids") or []
    if nearby_cids:
        current_avoid = set(mapped_avoid or [])
        target_idx = target_index
        n_sources = len(ref.sources)
        candidates = [
            i for i in range(n_sources) if i != target_idx and i not in current_avoid
        ]
        after_nearby = [i for i in candidates if i not in set(nearby_cids)]
        nearby_desc = reference.get("nearby_sep_arcsec")
        if candidates and not after_nearby:
            logger.warning(
                f"[{band}] nearby-star exclusion ({nearby_desc:.2f} arcsec) would remove all "
                f"comparison stars; relaxing nearby-star exclusion to preserve comparison pool"
            )
        else:
            mapped_avoid = sorted(current_avoid | set(nearby_cids))
            if cids is not None:
                cids = [c for c in cids if c not in set(nearby_cids)]
                if not cids:
                    logger.warning(
                        f"[{band}] all explicit cIDs fail nearby-star exclusion; "
                        f"falling back to auto-selection"
                    )
                    cids = None

    saturated_cids = reference.get("saturated_cids") or []
    if saturated_cids:
        current_avoid = set(mapped_avoid or [])
        target_idx = target_index
        n_sources = len(ref.sources)
        candidates = [
            i for i in range(n_sources) if i != target_idx and i not in current_avoid
        ]
        after_sat = [i for i in candidates if i not in set(saturated_cids)]
        if candidates and not after_sat:
            logger.warning(
                f"[{band}] saturation exclusion would remove all comparison stars; "
                f"relaxing saturation exclusion to preserve comparison pool"
            )
        else:
            mapped_avoid = sorted(current_avoid | set(saturated_cids))
            if cids is not None:
                cids = [c for c in cids if c not in set(saturated_cids)]
                if not cids:
                    logger.warning(
                        f"[{band}] all explicit cIDs are saturated; "
                        f"falling back to auto-selection"
                    )
                    cids = None

    requested_n_stars_align = n_stars_align if n_stars_align else len(ref.sources)
    effective_n_stars_align = min(
        requested_n_stars_align, len(ref.sources), max_num_stars
    )
    if effective_n_stars_align < requested_n_stars_align:
        logger.info(
            f"[{band}] capping alignment stars from {requested_n_stars_align} "
            f"to {effective_n_stars_align} (max_num_stars={max_num_stars})"
        )

    phot = photometry_sequence(
        ref,
        reference["aper_radii"],
        reference["rin"],
        reference["rout"],
        scale=reference["scale"],
        ccd_trim_size_yx=ccd_trim_size_yx,
        max_num_stars=max_num_stars,
        min_star_separation=min_star_separation,
        cutout_size=cutout_size,
        n_stars_align=effective_n_stars_align,
        target_index=target_index,
        min_area=min_area,
        bad_pixel_map=bad_pixel_map,
    )
    phot.run(files)

    fluxes: Fluxes = phot.data[0].fluxes
    if fluxes is None:
        discard_summary = getattr(phot, "discards", None) or {}
        if discard_summary:
            summary = ", ".join(
                f"{block}={len(idxs)}" for block, idxs in discard_summary.items()
            )
            logger.warning(f"[{band}] all frames discarded by blocks: {summary}")
        logger.warning(
            f"[{band}] no valid frames (all discarded); skipping "
            f"(alignment/source matching likely failed)"
        )
        return None
    fluxes.target = target_index

    diff = differential_photometry(
        fluxes,
        target_index,
        cids=cids,
        avoid_cids=mapped_avoid,
        nan_imputation_method=nan_imputation_method,
    )
    if diff is None:
        logger.warning(f"[{band}] no valid frames after cleaning; skipping")
        return None

    # Map the selected comparison star IDs back to the chosen band's index space for consistent logging
    comps = getattr(diff, "comparisons", None)
    if comps is not None and ref_band_star_ids is not None:
        mapped_comps = []
        for c in comps:
            if c < len(ref_band_star_ids):
                mapped_id = ref_band_star_ids[c]
                if mapped_id != -1:
                    mapped_comps.append(mapped_id)
                else:
                    mapped_comps.append(f"{c}(unmapped)")
            else:
                mapped_comps.append(c)
        logger.info(
            f"[{band}] comparison stars mapped to chosen band's IDs: {mapped_comps}"
        )

    # normalize the time axis to JD (e.g. MuSCAT2/TCS reports MJD) so that BJD
    # correction, CSV export, and plots all operate on a true Julian Date.
    diff.time = normalize_time_to_jd(diff.time, ref.telescope.jd_scale)

    logger.info(f"[{band}] reduction complete: {len(diff.time)} points")
    return dict(
        band=band,
        ref=ref,
        files=files,
        fluxes=fluxes,
        diff=diff,
        avoid_cids=mapped_avoid,
        target_index=target_index,
        ref_band_star_ids=ref_band_star_ids,
        aper_radii=np.asarray(reference["aper_radii"]),
        rin=reference["rin"],
        rout=reference["rout"],
        scale=reference["scale"],
        gaia_df=reference.get("gaia_df"),
        defaulted_to_brightest=reference.get("defaulted_to_brightest", False),
    )


def differential_photometry(
    fluxes: Fluxes,
    target_index: int,
    cids: list[int] | None = None,
    avoid_cids: list[int] | None = None,
    nan_imputation_method: str = "linear",
):
    """Clean NaN comparison stars, sigma-clip, and run differential photometry.

    When ``cids`` is given only those stars are used as comparisons and the
    automatic selection (Broeg et al. 2005) is skipped.

    When ``avoid_cids`` is given, those stars are excluded from the comparison
    pool in both explicit and auto modes.  The caller is responsible for
    cross-mapping indices from the reference band to the current band.

    Parameters
    ----------
    nan_imputation_method : str, optional
        Method to impute NaN values in flux matrix. Default: "linear".
    """
    fluxes = fluxes.copy()
    fluxes.target = target_index

    # Impute NaNs before filtering, to recover partially-valid comparison stars
    if nan_imputation_method != "none":
        from prose.fluxes import impute_nans

        logger.info(
            f"imputing NaNs using method='{nan_imputation_method}' "
            f"before comparison star selection"
        )
        imputed_fluxes = impute_nans(fluxes.fluxes, method=nan_imputation_method)
        # Create new Fluxes instance with imputed data
        from copy import deepcopy

        fluxes_copy = deepcopy(fluxes)
        fluxes_copy.fluxes = imputed_fluxes
        fluxes = fluxes_copy

    n_sources = fluxes.fluxes.shape[1]
    if cids is not None:
        valid_cids = [c for c in cids if 0 <= c < n_sources]
        dropped = [c for c in cids if c not in valid_cids]
        if dropped:
            logger.warning(
                f"ignoring comparison ID(s) {dropped} out of range "
                f"(only {n_sources} sources detected); using {valid_cids or 'auto-selection'}"
            )
        cids = valid_cids
    if cids:
        mask = np.zeros(n_sources, dtype=bool)
        mask[target_index] = True
        mask[list(cids)] = True
        if avoid_cids:
            mask[list(avoid_cids)] = False
        fluxes = fluxes.mask_stars(mask)
        kept = int(np.sum(mask)) - 1 if mask[target_index] else int(np.sum(mask))
        logger.info(
            f"comparison pool (explicit cIDs): {kept} candidates "
            f"after masking target + explicit comparisons"
        )
    else:
        keep = ~np.any(np.isnan(fluxes.fluxes), axis=(0, 2))
        logger.info(
            f"comparison pool (auto): {int(np.sum(keep)) - 1 if keep[target_index] else int(np.sum(keep))} "
            f"candidates after NaN filtering"
        )
        if avoid_cids:
            keep = np.array(keep, dtype=bool)
            valid_avoid = [a for a in avoid_cids if 0 <= a < n_sources]
            if valid_avoid:
                keep[valid_avoid] = False
                logger.info(
                    f"comparison pool (auto): {int(np.sum(keep)) - 1 if keep[target_index] else int(np.sum(keep))} "
                    f"candidates after avoid_cids filtering (removed {valid_avoid})"
                )
        fluxes = fluxes.mask_stars(keep)
    n_before = len(fluxes.time) if fluxes.time is not None else 0
    sigma_kwargs = {
        k: v
        for k, v in dict(
            bkg=SIGMA_BKG, fwhm=SIGMA_FWHM, dx=SIGMA_DX, dy=SIGMA_DY
        ).items()
        if v is not None
    }
    fluxes = fluxes.sigma_clipping_data(**sigma_kwargs)
    n_after = len(fluxes.time) if fluxes.time is not None else 0
    clipped = n_before - n_after

    def _fmt(v):
        return str(v) if v is not None else "off"

    logger.info(
        f"!!! SIGMA CLIPPING: {clipped} / {n_before} frames clipped "
        f"(bkg={_fmt(SIGMA_BKG)}, fwhm={_fmt(SIGMA_FWHM)}, "
        f"dx={_fmt(SIGMA_DX)}, dy={_fmt(SIGMA_DY)}) !!!"
    )
    if fluxes.time is None or len(fluxes.time) == 0:
        return None
    if cids:
        return fluxes.diff(comps=np.array(cids))

    # Imputation is already done above, so skip it in autodiff
    diff = fluxes.autodiff(nan_imputation_method="none")
    comps = getattr(diff, "comparisons", None)
    if comps is not None:
        original_comps = list(comps)
        n_considered = int(np.sum(keep))
        if 0 <= target_index < len(keep) and keep[target_index]:
            n_considered -= 1
        logger.info(
            f"autodiff chosen comparison stars (based on Broeg et al. 2005): "
            f"{original_comps} out of {n_considered} stars considered"
        )
        logger.info(
            f"comparison pool (auto): final selected {len(original_comps)} comparison star(s)"
        )
    return diff


# --------------------------- time conversion ---------------------------


def normalize_time_to_jd(time: np.ndarray, jd_scale: str | None) -> np.ndarray:
    """Return the time axis in JD, converting from MJD when needed.

    prose stores each frame's time verbatim from ``Telescope.keyword_jd`` and
    records the scale in ``Telescope.jd_scale``. MuSCAT2/TCS reports MJD
    (keyword ``MJD-STRT``), so its axis must be shifted by :data:`MJD_TO_JD`
    before any JD-based time handling (BJD correction, CSV export, plots).
    """
    t = np.asarray(time, dtype=float)
    if jd_scale is not None and jd_scale.lower() == "mjd":
        finite = t[np.isfinite(t)]
        if len(finite) and np.nanmedian(finite) > MJD_TO_JD:
            return t
        return t + MJD_TO_JD
    return t


def compute_bjd_tdb(
    diff: Fluxes,
    header,
    target_coord: SkyCoord,
    use_barycorrpy: bool,
    instrument: str | None = None,
):
    """Convert the GJD-UTC time axis to BJD-TDB.

    The observatory location is resolved from the header ``SITE`` keyword
    (LCO-style) when present, otherwise from :data:`INSTRUMENT_SITES` keyed on
    *instrument* (e.g. MuSCAT/MuSCAT2 headers carry no ``SITE`` keyword).
    """
    from astropy.coordinates import EarthLocation

    site = header.get("SITE")
    if site is not None:
        # ``SITE`` may be an LCO node string (BANZAI headers, e.g. "LCOGT node
        # at Tenerife") that maps to an astropy site code via ``LCO_SITES``, or
        # already an astropy site code written by our calibration scripts
        # (muscat/muscat2, e.g. "teide"/"OAO"). Accept both.
        loc = EarthLocation.of_site(LCO_SITES.get(site, site))
    elif instrument in INSTRUMENT_SITES:
        loc = EarthLocation.of_site(INSTRUMENT_SITES[instrument])
        logger.info(
            f"no SITE keyword; using {instrument} site "
            f"'{INSTRUMENT_SITES[instrument]}' for BJD correction"
        )
    else:
        loc = None
        logger.warning(
            "SITE keyword not found in header; BJD correction without site location"
        )
    t = Time(diff.time, format="jd", scale="utc", location=loc)

    if use_barycorrpy:
        from barycorrpy import utc_tdb

        if loc is None:
            raise ValueError(
                "barycorrpy BJD correction requires an observatory location; "
                "no SITE keyword and no known site for instrument "
                f"{instrument!r}"
            )
        result = utc_tdb.JDUTC_to_BJDTDB(
            t.value,
            ra=target_coord.ra.deg,
            dec=target_coord.dec.deg,
            lat=loc.lat.deg,
            longi=loc.lon.deg,
            alt=loc.height.value,
        )
        bjd = np.asarray(result[0])
        offset = bjd - t.value
    else:
        lttd = t.light_travel_time(target_coord, kind="barycentric")
        bjd = (t.tdb + lttd).value
        offset = bjd - t.jd

    offset_min = float(np.median(offset)) * 24 * 60
    if offset_min > MAX_TIME_OFFSET_MIN:
        logger.warning(
            f"GJD->BJD offset {offset_min:.2f} min exceeds {MAX_TIME_OFFSET_MIN:.1f} min"
        )
    else:
        logger.info(f"GJD_UTC --> BJD_TDB offset {offset_min:.2f} min")
    return bjd


# --------------------------- CSV export ---------------------------

CSV_RENAME = {
    "time": "GJD_UTC",
    "flux": "Flux",
    "airmass": "Airmass",
    "dx": "Dx(pix)",
    "dy": "Dy(pix)",
    "bkg": "Bkg(ADU)",
    "fwhm": "FWHM(pix)",
    "peak": "Peak(ADU)",
}


def photometry_df(diff: Fluxes, bjd: np.ndarray) -> pd.DataFrame:
    df = diff.df.copy()
    df["BJD_TDB"] = bjd
    df["Err"] = diff.error
    df = df.rename(CSV_RENAME, axis=1)
    cols = df.columns.tolist()
    left = ["BJD_TDB", "Flux", "Err"]
    rest = [c for c in cols if c not in left]
    return df[left + rest]


# --------------------------- plots ---------------------------


def _overlay_gaia_sources(
    ax,
    cutout,
    gaia_df,
    target_coord=None,
    color="cyan",
    label_mag=True,
    marker_size=45,
    fontsize=4,
    legend_label="Gaia sources",
) -> int:
    """Overlay queried Gaia source positions on a target cutout.

    Gaia ``ra``/``dec`` are projected into the cutout's own pixel frame via its
    WCS and clipped to the cutout bounds, so the markers line up with whatever
    ``origin="lower"`` image the axis already shows.  When *target_coord* is
    given, the target itself is plotted with a distinct marker (no text label)
    and neighbouring sources are annotated with their **delta** G magnitude
    (``Gmag - target_Gmag``) instead of the absolute value.

    Failures (missing catalog, missing/invalid WCS) degrade to a no-op rather
    than breaking the figure.

    Returns the number of Gaia sources drawn.
    """
    cutout_wcs = getattr(cutout, "wcs", None)
    if (
        gaia_df is None
        or not len(gaia_df)
        or cutout_wcs is None
        or not getattr(cutout_wcs, "has_celestial", False)
    ):
        return 0
    try:
        coords = SkyCoord(gaia_df.ra.values, gaia_df.dec.values, unit="deg")
        x, y = cutout.wcs.world_to_pixel(coords)
    except Exception as exc:  # noqa: BLE001 - overlay must not break the plot
        logger.warning(f"Gaia overlay skipped ({exc})")
        return 0
    ny, nx = cutout.data.shape

    # Correct for WCS error: the cutout is centered on the target's refined
    # centroid (sources[i].coords), so the target should be at (nx/2, ny/2).
    # If the original FITS-header WCS has any error (it was never updated
    # after centroid refinement), the WCS-predicted position of target_coord
    # will be offset from the actual centroid.  Apply a uniform pixel shift
    # to align WCS-projected Gaia positions with the refined centroids.
    if target_coord is not None:
        try:
            tx, ty = cutout.wcs.world_to_pixel(target_coord)
            dx = nx / 2 - tx
            dy = ny / 2 - ty
            x += dx
            y += dy
        except Exception:  # noqa: BLE001
            pass
    inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    x, y = x[inside], y[inside]
    if not len(x):
        return 0

    # Identify the target among the visible Gaia sources (nearest to target_coord).
    target_mask = np.zeros(len(x), dtype=bool)
    target_mag = np.nan
    if target_coord is not None:
        try:
            seps = target_coord.separation(coords[inside])
            nearest = int(np.argmin(seps))
            target_mask[nearest] = True
            if "phot_g_mean_mag" in gaia_df:
                mags_all = np.asarray(gaia_df.phot_g_mean_mag.values, dtype=float)[
                    inside
                ]
                target_mag = mags_all[nearest]
        except Exception:  # noqa: BLE001
            pass

    neighbour = ~target_mask

    # Plot the target marker (no label).
    if target_mask.any():
        ax.scatter(
            x[target_mask],
            y[target_mask],
            marker="+",
            s=marker_size,
            c=color,
            lw=0.9,
            zorder=10,
        )
    # Plot neighbour markers with legend.
    if neighbour.any():
        ax.scatter(
            x[neighbour],
            y[neighbour],
            marker="+",
            s=marker_size,
            c=color,
            lw=0.9,
            zorder=9,
            label=legend_label,
        )

    if label_mag and "phot_g_mean_mag" in gaia_df:
        mags = np.asarray(gaia_df.phot_g_mean_mag.values, dtype=float)[inside]
        for i, (xi, yi, mag) in enumerate(zip(x, y, mags)):
            if target_mask[i] or not np.isfinite(mag):
                continue  # skip the target (no text label)
            if np.isfinite(target_mag):
                label_text = f"{mag - target_mag:.1f}"
            else:
                label_text = f"{mag:.1f}"
            ax.annotate(
                label_text,
                (xi, yi),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=fontsize,
                color=color,
                zorder=9,
            )
    return int(len(x))


def plot_ref_image(
    r,
    target_coord,
    instrument,
    path: Path,
    target_name: str = "",
    date: str = "",
    avoid_cids: list[int] | None = None,
    plot_gaia_sources: bool = False,
    simbad_df=None,
    cmap: str = "Greys",
) -> None:
    ref = r["ref"]
    if not target_name:
        target_name = ref.header.get("OBJECT", "")
    if not instrument:
        instrument = get_instrument(ref.header)
    if not date:
        date = date_from_header(ref.header)
    band = r["band"]
    target_id = r["target_index"]
    ref_band_star_ids = r.get("ref_band_star_ids")
    if ref_band_star_ids is not None and target_id < len(ref_band_star_ids):
        mapped_id = ref_band_star_ids[target_id]
        if mapped_id != -1:
            target_id = mapped_id

    # Only trust the reference WCS for celestial decorations (RA/Dec projection
    # and SIMBAD markers) when it can actually project the target. A rejected or
    # degenerate solve leaves ref.wcs as a non-None but unusable WCS, which would
    # otherwise draw a meaningless RA/Dec grid and mis-projected markers; fall
    # back to a plain pixel frame in that case (mirrors the Gaia overlay guard).
    wcs_ok = _wcs_can_project(getattr(ref, "wcs", None), target_coord)
    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection=ref.wcs) if wcs_ok else fig.add_subplot(111)

    colors = get_plot_colors(cmap)
    ref.show(ax=ax, frame=True, sources=False, cmap=cmap)
    target_idx = r["target_index"]
    tpix = None
    if 0 <= target_idx < len(ref.sources):
        tpix = ref.sources[target_idx].coords
    elif wcs_ok:
        tpix = ref.wcs.wcs_world2pix([[target_coord.ra.deg, target_coord.dec.deg]], 0)[
            0
        ]
    if tpix is not None:
        ax.scatter(tpix[0], tpix[1], s=120, ec=colors["target"], fc="none", zorder=10)
        label = "Target???" if r.get("defaulted_to_brightest", False) else "Target"
        ax.annotate(
            label,
            (tpix[0], tpix[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=colors["target"],
            zorder=10,
        )
    avoided = set(avoid_cids or [])
    plotted_source_ids = [i for i in range(len(ref.sources)) if i not in avoided]

    sources_to_plot = ref.sources[plotted_source_ids]
    if ref_band_star_ids is not None:
        sources_to_plot = sources_to_plot.copy()
        for s in sources_to_plot.sources:
            if s.i is not None and s.i < len(ref_band_star_ids):
                mapped_id = ref_band_star_ids[s.i]
                if mapped_id != -1:
                    s.i = mapped_id
    sources_to_plot.plot(ax=ax, c=colors["sources"])
    desc = ref_header_desc(ref, "reference frame")
    if not wcs_ok:
        desc += " (pixel frame; WCS unusable)"
    title = f"{target_name} | {instrument} | {date} | {band} | tID={target_id}\n{desc}"
    ax.set_title(title, y=1.08)

    # SIMBAD markers are WCS-projected, so only draw them when the WCS is usable.
    if wcs_ok:
        # Compute WCS offset from the target's refined centroid, same as
        # _overlay_gaia_sources does, so SIMBAD markers align with detected stars.
        wcs_offset = np.array([0.0, 0.0])
        try:
            wpix = ref.wcs.world_to_pixel(target_coord)
            if 0 <= target_idx < len(ref.sources):
                wcs_offset = ref.sources[target_idx].coords - wpix
        except Exception:  # noqa: BLE001
            pass

        simbad = (
            simbad_df
            if simbad_df is not None
            else get_simbad_data(target_coord, instrument)
        )
        if simbad is not None and not simbad.empty:
            simbad = simbad[simbad.OTYPE != "Star"]
            if not simbad.empty:
                simbad_coords = SkyCoord(
                    ra=simbad.RA, dec=simbad.DEC, unit=(u.hourangle, u.deg)
                )
                x_pix, y_pix = ref.wcs.wcs_world2pix(
                    np.column_stack([simbad_coords.ra.deg, simbad_coords.dec.deg]), 0
                ).T
                x_pix += wcs_offset[0]
                y_pix += wcs_offset[1]
                for xi, yi, label in zip(x_pix, y_pix, simbad.OTYPE):
                    color = colors["simbad_default"]
                    if plot_gaia_sources:
                        label_lower = str(label).lower()
                        for keyword, color_key in _SIMBAD_FLAG_TYPES.items():
                            if keyword in label_lower:
                                color = colors[color_key]
                                break
                    ax.scatter([xi], [yi], marker="D", s=40, ec=color, fc="none", lw=1)
                    ax.annotate(
                        label,
                        (xi, yi),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=5,
                        color=color,
                    )

    _savefig(fig, path)


def plot_apertures(
    r,
    path: Path,
    target_name: str = "",
    instrument: str = "",
    date: str = "",
    plot_gaia_sources: bool = False,
    target_coord=None,
    cmap: str = "Greys",
) -> None:
    ref = r["ref"]
    if not target_name:
        target_name = ref.header.get("OBJECT", "")
    if not instrument:
        instrument = get_instrument(ref.header)
    if not date:
        date = date_from_header(ref.header)
    band = r["band"]
    target_id = r["target_index"]
    ref_band_star_ids = r.get("ref_band_star_ids")
    if ref_band_star_ids is not None and target_id < len(ref_band_star_ids):
        mapped_id = ref_band_star_ids[target_id]
        if mapped_id != -1:
            target_id = mapped_id

    coords = ref.sources[r["target_index"]].coords
    c = ref.cutout(coords, GAIA_CUTOUT, reset_index=False)
    radii_pix, rin_pix, rout_pix = aper_radii_pix(r)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    colors = get_plot_colors(cmap)
    c.show(ax=ax, zscale=True, sources=False, cmap=cmap)

    target_source = next(
        (s for s in c.sources if s.i == r["target_index"]), c.sources[0]
    )
    for radius in radii_pix:
        target_source.plot(radius, label=False, c=colors["aperture"])
    target_source.plot(rin_pix, label=False, c=colors["sky_annulus"])
    target_source.plot(rout_pix, label=False, c=colors["sky_annulus"])
    if plot_gaia_sources:
        n = _overlay_gaia_sources(
            ax,
            c,
            r.get("gaia_df"),
            target_coord=target_coord,
            marker_size=100,
            fontsize=7,
            legend_label=r"Gaia ($\Delta$ mag)",
        )
        if n:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    desc = ref_header_desc(ref, "Target cutout")
    desc2 = aperture_geometry_title(radii_pix, rin_pix, rout_pix)
    title = f"{target_name} | {instrument} | {date} | {band} | tID={target_id}\n{desc}\n{desc2}"
    ax.set_title(title)
    _savefig(fig, path)


def plot_alignment(
    r,
    other_file,
    path: Path,
    target_name: str = "",
    instrument: str = "",
    date: str = "",
    target_index: int | None = None,
    ccd_trim_size_yx: tuple[int, int] = CCD_TRIM_SIZE_YX,
    max_num_stars: int = MAX_NUM_STARS,
    min_star_separation: float = MIN_STAR_SEPARATION,
    n_stars_align: int | None = None,
    min_area: int = MIN_STAR_AREA,
    cmap: str = "Greys",
) -> None:
    """Overlay the reference image with an aligned later frame (best effort)."""
    if n_stars_align is None:
        n_stars_align = max_num_stars
    from skimage.transform import warp

    ref = r["ref"]
    if not target_name:
        target_name = ref.header.get("OBJECT", "")
    if not instrument:
        instrument = get_instrument(ref.header)
    if not date:
        date = date_from_header(ref.header)
    if target_index is None:
        target_index = r["target_index"]
    band = r["band"]

    try:
        seq = Sequence(
            [
                blocks.Trim(ccd_trim_size_yx),
                # blocks.AutoSourceDetection(
                blocks.PointSourceDetection(
                    n=max_num_stars,
                    min_area=min_area,
                    min_separation=min_star_separation,
                ),
                # FilterPointSources(),
                blocks.ComputeTransformTwirl(ref, n=n_stars_align),
            ]
        )
        other = FITSImage(other_file, telescope=ref.telescope)
        seq.run(other, show_progress=False)
        raw = other.data.astype(float)
        aligned = warp(raw, other.transform.inverse, cval=np.median(raw))
    except Exception as exc:  # noqa: BLE001 - diagnostic plot only
        logger.warning(f"[{r['band']}] alignment plot failed: {exc}")
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(6, 3), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, img, title in zip(axes, (raw, aligned), ("raw", "aligned")):
        ax.imshow(z_scale(ref.data), cmap=cmap, origin="lower")
        ax.imshow(z_scale(img), cmap=cmap, origin="lower", alpha=0.5)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5, color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
    desc = ref_header_desc(ref, "alignment")
    title = (
        f"{target_name} | {instrument} | {date} | {band} | tID={target_index}\n{desc}"
    )
    fig.suptitle(title)
    _savefig(fig, path)


def _binned(time, flux, bin_size_days: float = BIN_SIZE_DAYS):
    from prose.utils import index_binning

    idxs = index_binning(time, bin_size_days)
    bt = np.array([time[i].mean() for i in idxs])
    bf = np.array([flux[i].mean() for i in idxs])
    be = np.array([flux[i].std() / np.sqrt(len(i)) for i in idxs])
    return bt, bf, be


def plot_lightcurves(
    band_results,
    path: Path,
    target_name: str,
    instrument: str,
    date: str,
    target_index: int,
    bin_size_days: float = BIN_SIZE_DAYS,
) -> None:
    bands = list(band_results.keys())
    fig, axes = plt.subplots(
        len(bands),
        1,
        figsize=(8, 2.4 * len(bands)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    t0 = int(np.asarray(band_results[bands[0]]["diff"].time)[0])
    for ax, band in zip(axes, bands):
        c = band_color(band)
        diff = band_results[band]["diff"]
        t, f = np.asarray(diff.time) - t0, np.asarray(diff.flux)
        ax.plot(t, f, ".", c="k", alpha=0.2)
        bt, bf, be = _binned(t, f, bin_size_days=bin_size_days)
        ax.errorbar(bt, bf, yerr=be, fmt="o", c=c)
        ax.set_ylabel(f"{band}\nDiff. flux")
    axes[-1].set_xlabel(f"time (JD) - {t0}")

    secax = axes[0].secondary_xaxis(
        location="top",
        functions=(
            lambda rel: rel + t0,
            lambda jd: jd - t0,
        ),
    )
    secax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda jd, _: Time(jd, format="jd").datetime.strftime("%m-%d\n%H:%M")
        )
    )
    secax.set_xlabel("UTC")
    fig.suptitle(f"{target_name} | {instrument} | {date} | tID={target_index}")
    _savefig(fig, path)


def plot_raw_flux(
    band_results,
    path: Path,
    target_name: str,
    instrument: str,
    date: str,
    target_index: int,
) -> None:
    """Plot comparison stars raw flux for each band."""
    bands = list(band_results.keys())
    fig, axes = plt.subplots(
        1,
        len(bands),
        figsize=(4 * len(bands), 7),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    t0 = int(np.asarray(band_results[bands[0]]["diff"].time)[0])
    for ax, band in zip(axes, bands):
        diff = band_results[band]["diff"]
        fluxes = band_results[band]["fluxes"]
        t = np.asarray(diff.time) - t0

        jd_scale = band_results[band]["ref"].telescope.jd_scale
        fluxes_time = normalize_time_to_jd(fluxes.time, jd_scale)
        mask = np.isin(fluxes_time, diff.time)

        comps = diff.comparisons
        stars = [diff.target]
        if comps is not None and len(comps) > 0:
            stars.extend([c for c in comps if c != diff.target])

        ref_band_star_ids = band_results[band].get("ref_band_star_ids")

        for j, i in enumerate(stars):
            y = fluxes.fluxes[diff.aperture, i][mask].copy()
            std_val = np.std(y)
            denom = std_val or 1e-12
            y = (y - np.mean(y)) / denom + 8 * j

            label_id = i
            if ref_band_star_ids is not None and i < len(ref_band_star_ids):
                mapped_id = ref_band_star_ids[i]
                if mapped_id != -1:
                    label_id = mapped_id

            ax.text(
                t.max(),
                np.mean(y) + 4,
                label_id if i != diff.target else "target",
                ha="right",
            )
            ax.plot(t, y, ".", c="0.8" if i != diff.target else band_color(band))

        ax.set_title(band)
        ax.set_xlabel(f"time (JD) - {t0}")
        ax.set_ylabel("raw flux (arbitrary units)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(f"{target_name} | {instrument} | {date} | tID={target_index}")
    _savefig(fig, path)


def plot_covariates(
    band_results,
    path: Path,
    target_name: str,
    instrument: str,
    date: str,
    target_index: int,
) -> None:
    bands = list(band_results.keys())
    fig, axes = plt.subplots(
        1, len(bands), figsize=(5 * len(bands), 8), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    signals = ["flux", "fwhm", "peak", "airmass", "bkg", "dx", "dy"]
    for ax, band in zip(axes, bands):
        bc = band_color(band)
        diff = band_results[band]["diff"]
        t = np.asarray(diff.time)
        t0 = int(t[0])
        t = t - t0
        for i, name in enumerate(signals):
            raw = np.asarray(diff.df[name], dtype=float).copy()
            std_val = np.std(raw)
            denom = std_val or 1e-12
            y = (raw - np.mean(raw)) / denom + 8 * i
            ax.text(t.max(), np.mean(y) + 4, f"{name} (std: {std_val:.2f})", ha="right")
            ax.plot(t, y, ".", c=bc if name == "flux" else "0.8")
        ax.set_xlabel(f"time (JD) - {t0}")
        ax.set_title(band)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    desc = ref_header_desc(band_results[bands[0]]["ref"], "stacks")
    fig.suptitle(f"{target_name} | {instrument} | {date} | tID={target_index}\n{desc}")
    _savefig(fig, path)


def _radial_profile(data, center):
    y, x = np.indices(data.shape)
    rr = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)
    tbin = np.bincount(rr.ravel(), data.ravel())
    nr = np.bincount(rr.ravel())
    return tbin / np.maximum(nr, 1)


def _radial_peak_profile(data, center):
    y, x = np.indices(data.shape)
    rr = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)
    num_bins = rr.max() + 1
    peaks = np.full(num_bins, -np.inf)
    np.maximum.at(peaks, rr.ravel(), data.ravel())
    peaks[peaks == -np.inf] = 0
    return peaks


def plot_stacks(
    band_results,
    path: Path,
    target_name: str,
    instrument: str,
    date: str,
    target_index: int,
    plot_gaia_sources: bool = False,
    target_coord=None,
    cmap: str = "Greys",
) -> None:
    """Per-band target cutout (from the reference image) plus radial profile."""
    bands = list(band_results.keys())
    fig, axes = plt.subplots(
        len(bands), 2, figsize=(7, 3 * len(bands)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for row, band in enumerate(bands):
        bc = band_color(band)
        r = band_results[band]
        ref = r["ref"]
        diff = r["diff"]
        c = ref.cutout(
            ref.sources[r["target_index"]].coords, GAIA_CUTOUT, reset_index=False
        )
        center = np.array(c.data.shape)[::-1] / 2

        colors = get_plot_colors(cmap)
        axes[row, 0].imshow(z_scale(c.data), cmap=cmap, origin="lower")
        axes[row, 0].set_title(f"target zoom ({band})")
        axes[row, 0].axis("off")
        if plot_gaia_sources:
            _overlay_gaia_sources(
                axes[row, 0],
                c,
                r.get("gaia_df"),
                target_coord=target_coord,
                label_mag=False,
            )

        radii_pix, rin_pix, rout_pix = aper_radii_pix(r)
        peaks = _radial_peak_profile(c.data, center)
        axes[row, 1].plot(peaks, ".", c="0.5", ms=4, alpha=0.5)
        axes[row, 1].plot(peaks, ls="--", c="0.5", alpha=0.5, label="peak")

        prof = _radial_profile(c.data, center)
        ax_twin = axes[row, 1].twinx()
        ax_twin.plot(prof, ".", c=bc, ms=6, alpha=0.5)
        ax_twin.plot(prof, c=bc, alpha=0.5, label="flux")
        ax_twin.set_yscale("log")
        ax_twin.set_ylabel("flux (ADU)")

        best = float(radii_pix[min(int(diff.aperture), len(radii_pix) - 1)])
        axes[row, 1].axvline(
            best, color=colors["aperture"], alpha=0.6, label=f"best: r={best:.0f}"
        )
        axes[row, 0].add_artist(
            plt.Circle(tuple(center), best, color=colors["aperture"], fill=False)
        )
        for radius in (rin_pix, rout_pix):
            axes[row, 1].axvline(
                radius, color=colors["sky_annulus"], ls="--", alpha=0.6
            )
            axes[row, 0].add_artist(
                plt.Circle(
                    tuple(center),
                    radius,
                    color=colors["sky_annulus"],
                    ls="--",
                    fill=False,
                )
            )
        saturation = getattr(ref.telescope, "saturation", None)
        if saturation is not None:
            axes[row, 1].axhline(
                saturation, color="k", ls="--", alpha=0.7, label="saturation"
            )
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_xlabel("radius (pixels)")
        axes[row, 1].set_ylabel("peak count (ADU)")

        # Combine legends from main and twin axes
        lines, labels = axes[row, 1].get_legend_handles_labels()
        lines_twin, labels_twin = ax_twin.get_legend_handles_labels()
        axes[row, 1].legend(lines + lines_twin, labels + labels_twin, loc="upper right")
    desc = ref_header_desc(band_results[bands[0]]["ref"], "stacks")
    fig.suptitle(f"{target_name} | {instrument} | {date} | tID={target_index}\n{desc}")
    _savefig(fig, path)


def plot_cutouts(
    r: dict,
    path: Path,
    target_name: str = "",
    instrument: str = "",
    band: str = "",
    date: str = "",
    max_num_stars: int = MAX_NUM_STARS,
    plot_gaia_sources: bool = False,
    target_coord=None,
    simbad_df=None,
    cmap: str = "Greys",
) -> None:
    ref = r["ref"]
    if not target_name:
        target_name = ref.header.get("OBJECT", "")
    if not instrument:
        instrument = get_instrument(ref.header)
    if not date:
        date = date_from_header(ref.header)
    if not band:
        band = r["band"]
    target_id = r.get("target_index")
    ref_band_star_ids = r.get("ref_band_star_ids")
    if (
        ref_band_star_ids is not None
        and target_id is not None
        and target_id < len(ref_band_star_ids)
    ):
        mapped_id = ref_band_star_ids[target_id]
        if mapped_id != -1:
            target_id = mapped_id

    cutouts = ref.computed.get("cutouts")
    if not cutouts:
        return

    target_idx = r.get("target_index")
    avoid_cids = r.get("avoid_cids")
    avoided = set(avoid_cids or [])

    # Filter out avoided stars, and make sure indices are valid
    candidates = [
        i for i in range(len(cutouts)) if i not in avoided and 0 <= i < len(cutouts)
    ]

    if (
        target_idx is not None
        and target_idx not in avoided
        and 0 <= target_idx < len(cutouts)
    ):
        top_candidates = candidates[:max_num_stars]
        if target_idx not in top_candidates:
            if len(top_candidates) == max_num_stars:
                top_candidates[-1] = target_idx
            else:
                top_candidates.append(target_idx)
        indices_to_plot = sorted(top_candidates)
    else:
        indices_to_plot = candidates[:max_num_stars]

    ncutouts = len(indices_to_plot)
    if ncutouts == 0:
        return

    # Check if WCS is OK for target_coord to query/project SIMBAD
    wcs_ok = target_coord is not None and _wcs_can_project(
        getattr(ref, "wcs", None), target_coord
    )
    simbad_coords_list = []
    if wcs_ok:
        wcs_offset = np.array([0.0, 0.0])
        try:
            wpix = ref.wcs.world_to_pixel(target_coord)
            if 0 <= target_idx < len(ref.sources):
                wcs_offset = ref.sources[target_idx].coords - wpix
        except Exception:  # noqa: BLE001
            pass

        simbad = (
            simbad_df
            if simbad_df is not None
            else get_simbad_data(target_coord, instrument)
        )
        if simbad is not None and not simbad.empty:
            simbad = simbad[simbad.OTYPE != "Star"]
            if not simbad.empty:
                try:
                    simbad_coords = SkyCoord(
                        ra=simbad.RA, dec=simbad.DEC, unit=(u.hourangle, u.deg)
                    )
                    x_pix_all, y_pix_all = ref.wcs.wcs_world2pix(
                        np.column_stack([simbad_coords.ra.deg, simbad_coords.dec.deg]),
                        0,
                    ).T
                    x_pix_all += wcs_offset[0]
                    y_pix_all += wcs_offset[1]
                    simbad_coords_list = list(zip(x_pix_all, y_pix_all, simbad.OTYPE))
                except Exception:  # noqa: BLE001
                    pass

    ncols = min(5, ncutouts)
    nrows = ncutouts // ncols if ncutouts % ncols == 0 else ncutouts // ncols + 1

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(2 * ncols, 2.5 * nrows), constrained_layout=True
    )
    ax = np.atleast_1d(axs).flatten()

    radii_pix, _, _ = aper_radii_pix(r)
    best = float(radii_pix[min(int(r["diff"].aperture), len(radii_pix) - 1)])

    colors = get_plot_colors(cmap)

    for i, idx in enumerate(indices_to_plot):
        img = cutouts[idx]
        img.show(ax=ax[i], cmap=cmap)
        ax[i].axis("off")

        if plot_gaia_sources and r.get("gaia_df") is not None:
            star_coord = None
            if ref.wcs is not None and hasattr(ref.wcs, "pixel_to_world"):
                star_coord = ref.wcs.pixel_to_world(*(ref.sources[idx].coords))
            _overlay_gaia_sources(
                ax[i],
                img,
                r.get("gaia_df"),
                target_coord=star_coord,
                label_mag=True,
            )

        # Plot SIMBAD objects if they fall inside this cutout
        if simbad_coords_list and hasattr(img, "origin") and img.origin is not None:
            x0, y0 = img.origin
            ny, nx = img.data.shape
            for xi, yi, label in simbad_coords_list:
                xi_c = xi - x0
                yi_c = yi - y0
                if 0 <= xi_c < nx and 0 <= yi_c < ny:
                    color = colors["simbad_default"]
                    if plot_gaia_sources:
                        label_lower = str(label).lower()
                        for keyword, color_key in _SIMBAD_FLAG_TYPES.items():
                            if keyword in label_lower:
                                color = colors[color_key]
                                break
                    ax[i].scatter(
                        [xi_c],
                        [yi_c],
                        marker="D",
                        s=40,
                        ec=color,
                        fc="none",
                        lw=1,
                        zorder=10,
                    )
                    ax[i].annotate(
                        label,
                        (xi_c, yi_c),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=5,
                        color=color,
                        zorder=10,
                    )

        center = np.array(img.data.shape)[::-1] / 2
        ax[i].add_artist(
            plt.Circle(
                tuple(center),
                best,
                color=colors["aperture"],
                fill=False,
                lw=1.5,
                alpha=0.8,
            )
        )
        peak = ref.sources[idx].peak
        is_target = " (Target)" if idx == target_idx else ""
        tcolor = "r" if idx == target_idx else "k"
        star_id_label = idx
        if ref_band_star_ids is not None and idx < len(ref_band_star_ids):
            mapped_id = ref_band_star_ids[idx]
            if mapped_id != -1:
                star_id_label = mapped_id
        ax[i].set_title(
            f"Star {star_id_label}{is_target}\npeak={peak:,.0f}", color=tcolor
        )

    for j in range(ncutouts, len(ax)):
        ax[j].axis("off")

    desc = ref_header_desc(ref, "cutouts")
    title = f"{target_name} | {instrument} | {date} | {band} | tID={target_id} | r={_format_pix_value(best)} pix\n{desc}"
    fig.suptitle(title)
    _savefig(fig, path)


# --------------------------- GIF ---------------------------


def _gif_frame(
    data: np.ndarray, label: str = "", max_px: int = GIF_MAX_PX, cmap: str = "Greys"
) -> np.ndarray:
    """Build one 8-bit RGB GIF frame from image data, matplotlib-free.

    The array is z-scaled, colormapped, flipped vertically to match matplotlib's
    ``origin="lower"`` display convention, downsampled so its longest side is
    ``max_px``, and stamped with ``label`` (e.g. ``DATE-OBS``) via PIL. This
    avoids the per-frame Figure/savefig round-trip that dominated runtime
    (see ``cprofile_results.txt``).
    """
    from PIL import Image, ImageDraw

    zscaled = _zscale(data)
    try:
        colormap = plt.get_cmap("gray" if cmap == "Greys" else cmap)
    except Exception:
        colormap = plt.get_cmap("gray")
    rgba = colormap(zscaled)
    arr = (rgba[:, :, :3] * 255).astype(np.uint8)
    arr = np.flipud(arr)  # mimic matplotlib origin="lower"
    frame = Image.fromarray(arr, mode="RGB")
    longest = max(frame.size)
    if longest > max_px:
        scale = max_px / longest
        frame = frame.resize(
            (max(1, round(frame.width * scale)), max(1, round(frame.height * scale))),
            Image.BILINEAR,
        )
    draw = ImageDraw.Draw(frame)
    width, height = frame.size
    grid_spacing = 50
    dot_spacing = 5
    pts = []
    # Horizontal grid lines
    for y in range(grid_spacing, height, grid_spacing):
        pts.extend((x, y) for x in range(0, width, dot_spacing))
    # Vertical grid lines
    for x in range(grid_spacing, width, grid_spacing):
        pts.extend((x, y) for y in range(0, height, dot_spacing))

    if pts:
        draw.point(pts, fill=(255, 255, 255))

    if label:
        draw.text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(frame)


def make_gif(files, path: Path, stride: int, cmap: str = "Greys") -> None:
    """Render a quick-look GIF per band without matplotlib."""
    import imageio.v2 as imageio

    sampled = files[:: max(1, stride)]
    if not sampled:
        return
    frames = []
    for fp in track(sampled, description=f"gif:{path.name}"):
        img = FITSImage(fp)
        frames.append(_gif_frame(img.data, img.header.get("DATE-OBS", ""), cmap=cmap))
    imageio.mimsave(path, frames, fps=FPS, loop=0)
    logger.info(f"wrote {path}")


# --------------------------- NPZ ---------------------------


def _npz_safe(v):
    arr = np.asarray(v)
    return arr if arr.dtype != object else np.array(v, dtype=object)


def save_all_bands_npz(
    band_results, bjds, path: Path, meta: dict | None = None
) -> None:
    out = {}
    for band, r in band_results.items():
        diff = r["diff"]
        out[f"{band}__time"] = _npz_safe(diff.time)
        out[f"{band}__bjd_tdb"] = _npz_safe(bjds[band])
        out[f"{band}__flux"] = _npz_safe(diff.flux)
        out[f"{band}__error"] = _npz_safe(diff.error)
        out[f"{band}__aper_radii"] = _npz_safe(r["aper_radii"])
        out[f"{band}__rin"] = np.array(r["rin"])
        out[f"{band}__rout"] = np.array(r["rout"])
        out[f"{band}__aper_unit"] = np.array("fwhm" if r.get("scale") else "pix")

        # Map target_index back to reference band if a reference band mapping is available
        target_idx = r["target_index"]
        ref_band_star_ids = r.get("ref_band_star_ids")
        if ref_band_star_ids is not None and target_idx < len(ref_band_star_ids):
            mapped_id = ref_band_star_ids[target_idx]
            if mapped_id != -1:
                target_idx = mapped_id
        out[f"{band}__target_index"] = np.array(target_idx)

        # Map comparisons back to reference band if a reference band mapping is available
        comps = getattr(diff, "comparisons", None)
        if comps is not None:
            if ref_band_star_ids is not None:
                mapped_comps = []
                for c in comps:
                    if c < len(ref_band_star_ids):
                        mapped_id = ref_band_star_ids[c]
                        if mapped_id != -1:
                            mapped_comps.append(mapped_id)
                out[f"{band}__comparisons"] = np.array(mapped_comps)
            else:
                out[f"{band}__comparisons"] = _npz_safe(comps)

        out[f"{band}__aperture"] = np.array(diff.aperture)
        out[f"{band}__wcs_header"] = np.array(r["ref"].wcs.to_header_string(relax=True))
        for key in ("fwhm", "airmass", "bkg", "dx", "dy", "peak"):
            if key in diff.data:
                out[f"{band}__data__{key}"] = _npz_safe(diff.data[key])
    if meta:
        out["__meta__"] = np.string_(str(meta))
        out["__prose_version__"] = np.string_(_PROSE_VERSION)
        out["__created__"] = np.string_(datetime.utcnow().isoformat())
    np.savez(path, **out)
    logger.info(f"wrote {path}")


# --------------------------- main ---------------------------


_NARROW_FILTER_NAMES = {"g_narrow", "Na_D", "i_narrow", "z_narrow"}


def _detect_narrow_bands(data_dir: Path, target_name: str) -> list[str]:
    """Peek at obslog or FITS headers to detect narrow-band filters.

    Returns ``DEFAULT_NARROW_BANDS`` if narrow filters are found, otherwise
    ``DEFAULT_BROAD_BANDS``.  Logs the decision so operators can override with
    an explicit ``--bands``.
    """
    # Shared obslog base, kept in sync with muscat-db's OBSLOG_BASE via the same
    # MUSCAT_OBSLOG_DIR env var (inherited from the launching muscat-db process,
    # or sourced from .env for manual runs). Default matches muscat-db (/ut3).
    obslog_base = os.environ.get("MUSCAT_OBSLOG_DIR", "/ut3/muscat/obslog")
    obslog_dir = Path(f"{obslog_base}/{data_dir.parent.name.lower()}/{data_dir.name}")
    if obslog_dir.is_dir():
        for ccd_csv in sorted(obslog_dir.glob("obslog-*-ccd?.csv")):
            with open(ccd_csv) as f:
                for row in csv.DictReader(f):
                    if row["OBJECT"] != target_name:
                        continue
                    raw = row["FILTER"].strip()
                    if raw in _NARROW_FILTER_NAMES:
                        logger.info(
                            f"obslog shows narrow-band filter {raw!r}; "
                            f"auto-using {DEFAULT_NARROW_BANDS}"
                        )
                        return DEFAULT_NARROW_BANDS
        return DEFAULT_BROAD_BANDS

    # fallback: scan first few FITS headers
    candidates = sorted(data_dir.glob("*.fits")) or sorted(data_dir.rglob("*.fits"))
    for fp in candidates[:50]:
        try:
            hdr = fits.getheader(fp)
            if str(hdr.get("OBJECT", "")).strip() != target_name:
                continue
            raw = str(hdr.get("FILTER", "")).strip()
            if raw in _NARROW_FILTER_NAMES:
                logger.info(
                    f"FITS header shows narrow-band filter {raw!r}; "
                    f"auto-using {DEFAULT_NARROW_BANDS}"
                )
                return DEFAULT_NARROW_BANDS
        except Exception:
            continue
    return DEFAULT_BROAD_BANDS


def _header_has_usable_wcs(header) -> bool:
    """Return True when a FITS header carries celestial WCS axes."""
    try:
        return bool(getattr(WCS(header), "has_celestial", False))
    except Exception:
        return False


def _inject_wcs_from_sidecars(
    *,
    calib_label: str,
    calib_dir: Path,
    calibrated_files: list[Path],
    active_bands: list[str],
    requested_bands: list[str],
    target_name: str,
    wcs_method: str,
) -> bool:
    """Inject cached per-band WCS sidecars into calibrated MuSCAT FITS files.

    Returns True only when every active band has a readable celestial WCS
    sidecar and every matching calibrated file was updated successfully.
    """
    sidecar_dir = calib_dir / ".wcs"
    sidecars = {b: sidecar_dir / f"{b}_{wcs_method}.wcs.fits" for b in active_bands}
    missing = [b for b, path in sidecars.items() if not path.exists()]
    if missing:
        logger.info(
            f"{calib_label}: cached {wcs_method} WCS sidecars missing for "
            f"{', '.join(missing)}"
        )
        return False

    band_files = read_filename_per_band(
        calibrated_files,
        requested_bands,
        target_name,
        filter_aliases=INSTRUMENT_FILTER_ALIASES.get(calib_label),
    )
    for band in active_bands:
        files = band_files.get(band, [])
        if not files:
            logger.warning(
                f"  {calib_label} [{band}]: no calibrated files matched target "
                f"{target_name}; cannot inject cached WCS"
            )
            return False

        wcs = load_wcs_fits(sidecars[band])
        if wcs is None:
            logger.warning(
                f"  {calib_label}: sidecar {sidecars[band]} unreadable; "
                f"re-calibrating"
            )
            return False

        for fp in files:
            if not inject_wcs_into_file(fp, wcs, method=wcs_method):
                logger.warning(
                    f"  {calib_label}: failed to inject WCS into {fp}; "
                    f"marking recalibration required"
                )
                return False

        logger.info(
            f"  {calib_label} [{band}]: injected cached {wcs_method} WCS into "
            f"{len(files)} files"
        )

    return True


def _calibrated_wcs_problems_by_band(
    *,
    calib_label: str,
    calibrated_files: list[Path],
    active_bands: list[str],
    requested_bands: list[str],
    target_name: str,
    wcs_method: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Inspect representative calibrated files for per-band WCS problems.

    Returns ``(missing, unreadable, no_wcs, wrong_method)`` band-name lists.
    """
    band_files = read_filename_per_band(
        calibrated_files,
        requested_bands,
        target_name,
        filter_aliases=INSTRUMENT_FILTER_ALIASES.get(calib_label),
    )
    missing: list[str] = []
    unreadable: list[str] = []
    no_wcs: list[str] = []
    wrong_method: list[str] = []

    for band in active_bands:
        files = band_files.get(band) or []
        if not files:
            missing.append(band)
            continue
        try:
            header = fits.getheader(files[0])
        except OSError:
            unreadable.append(band)
            continue
        if not _header_has_usable_wcs(header):
            no_wcs.append(band)
        elif header.get("WCSMTHD", "") != wcs_method:
            wrong_method.append(band)

    return missing, unreadable, no_wcs, wrong_method


def _fits_file_number(path: Path) -> int | None:
    """Extract the FITS frame number from a (calibrated) file path.

    Raw FITS:   MCT20_1911191480.fits                -> 1480
    Calibrated: MCT20_1911191480_calibrated.fits      -> 1480
    """
    m = re.search(r"_(\d{6})(\d{4})(?:_calibrated)?\.fits$", str(path))
    if m:
        return int(m.group(2))
    return None


def _find_frame_by_number(files: list[Path], number: int) -> int:
    """Return the index of the file whose FITS frame number is closest to *number*."""
    best_i, best_delta = 0, float("inf")
    for i, fp in enumerate(files):
        n = _fits_file_number(fp)
        if n is not None:
            d = abs(n - number)
            if d < best_delta:
                best_i, best_delta = i, d
    return best_i


def _normalize_toi_name(name: str) -> str:
    """Convert TOI names to MAST-compatible format.

    The obslog and FITS headers store zero-padded TOI names like
    ``TOI07475.01`` or ``TOI-05486.01``, but MAST/SIMBAD expects the
    standard ``TOI-7475`` format (no zero-padding, mandatory hyphen,
    no ``.01`` suffix).
    """
    m = re.match(r"^TOI-?0*(\d+)(?:\.\d+)?$", name, re.IGNORECASE)
    if m:
        return f"TOI-{m.group(1)}"
    return name


def _resolve_simbad_target(name: str) -> SkyCoord:
    simbad = Simbad()
    simbad.TIMEOUT = 30
    result = simbad.query_object(name)
    if result is None or len(result) == 0:
        raise ValueError(f"Simbad returned no result for '{name}'")
    return SkyCoord(result["ra"][0], result["dec"][0], unit=u.deg, frame="icrs")


def _calibration_args(
    args: argparse.Namespace, calib_dir: Path, bands: list[str] | None
) -> list[str]:
    calib_args = [
        "--data_dir",
        str(args.data_dir),
        "--target",
        args.target_name,
        "--output_dir",
        str(calib_dir),
    ]
    if bands:
        calib_args.extend(["--bands", *bands])
    calib_args.extend(["--solve_wcs", args.wcs_method])
    if args.test_run:
        calib_args.append("--test_run")
    if args.verbose:
        calib_args.append("--verbose")
    return calib_args


def _sinistro_modes_from_headers(
    data_dir: Path,
    glob_pattern: str,
    target_name: str,
    bands: list[str] | None,
) -> set[str]:
    """Return Sinistro CONFMODE values discoverable from FITS headers.

    This is intentionally header-only and best-effort so parser validation never
    triggers network work or full image reads. An empty set means "unknown",
    not "invalid".
    """
    files = sorted(data_dir.glob(glob_pattern))
    if not files:
        files = sorted(data_dir.rglob(glob_pattern))

    modes: set[str] = set()
    for file in files:
        try:
            header = fits.getheader(file)
        except Exception:
            continue
        if get_instrument(header) != "sinistro":
            continue
        if str(header.get("OBJECT", "")).strip() != target_name:
            continue
        raw_filter = str(header.get("FILTER", "")).strip()
        if bands and _resolve_band(raw_filter, "sinistro", bands) is None:
            continue
        confmode = str(header.get("CONFMODE", "")).strip().lower()
        if confmode:
            modes.add(confmode)
    return modes


def load_bad_pixel_map(
    source: str | None,
    ref_image: FITSImage | None = None,
) -> np.ndarray | None:
    """Load a bad pixel map from a FITS file or image header.

    Parameters
    ----------
    source : str, optional
        Path to bad pixel map FITS file or "header" to read from reference
        image header (requires ref_image to be provided).
    ref_image : FITSImage, optional
        Reference image to read bad pixel map from header (if source="header").

    Returns
    -------
    np.ndarray or None
        Boolean array marking bad pixels (True = bad), or None if not available.
    """
    if source is None:
        return None

    if source.lower() == "header":
        if ref_image is None:
            logger.warning(
                "bad pixel map source='header' but no reference image provided"
            )
            return None
        try:
            header = ref_image.header
            # Try common bad pixel map keywords
            for key in ("BADPIXEL", "BADPIX", "BPM", "BAD_PIXEL"):
                if key in header:
                    logger.info(f"found bad pixel map in header keyword '{key}'")
                    # Header typically stores as binary image data or checksum
                    # This is a placeholder - actual implementation depends on
                    # how your data stores bad pixels in the header
                    logger.warning(
                        f"bad pixel extraction from header keyword '{key}' "
                        "not yet implemented; skipping bad pixel masking"
                    )
                    return None
            logger.warning("no bad pixel map found in reference image header")
            return None
        except Exception as e:
            logger.warning(f"failed to read bad pixel map from header: {e}")
            return None

    # Load from FITS file
    try:
        filepath = Path(source)
        if not filepath.exists():
            logger.warning(f"bad pixel map file not found: {filepath}")
            return None

        with fits.open(filepath) as hdul:
            if len(hdul) == 0:
                logger.warning(f"bad pixel map file is empty: {filepath}")
                return None

            # Try to find data in primary HDU or first extension
            data = None
            for i, hdu in enumerate(hdul):
                if hdu.data is not None:
                    data = hdu.data
                    logger.info(f"loaded bad pixel map from {filepath} (HDU {i})")
                    break

            if data is None:
                logger.warning(f"no data found in bad pixel map file: {filepath}")
                return None

            # Ensure it's boolean or convertible to boolean
            bad_pixel_map = np.asarray(data, dtype=bool)
            logger.info(
                f"bad pixel map: shape={bad_pixel_map.shape}, "
                f"n_bad_pixels={int(np.sum(bad_pixel_map))}"
            )
            return bad_pixel_map

    except Exception as e:
        logger.error(f"failed to load bad pixel map from {source}: {e}")
        return None


def parse_args(argv=None) -> argparse.Namespace:
    mode_choices = ["central_2k_2x2", "full_frame"]

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target_name", "--target-name", required=True)
    ap.add_argument(
        "--target_coord",
        "--target-coord",
        nargs=2,
        default=None,
        metavar=("RA", "Dec"),
        help="Target sky coordinates RA Dec (e.g. '12:34:56.7' '38:47:04.5' "
        "or '187.5' '38.78'). When provided, bypasses MAST name resolution "
        "and uses these coordinates directly.",
    )
    ap.add_argument(
        "--tID",
        type=int,
        default=None,
        help="Target source index (default: auto-detected from Gaia).",
    )
    ap.add_argument(
        "--cID",
        type=int,
        nargs="+",
        default=None,
        help="Comparison star indices for differential photometry (default: auto-selected).",
    )
    ap.add_argument(
        "--avoid_cids",
        "--avoid-cids",
        type=int,
        nargs="+",
        default=None,
        help="Star indices to exclude as comparisons and from ref.png. "
        "Requires --ref_band; indices refer to the reference band's source "
        "catalog (default: none).",
    )
    ap.add_argument(
        "--data_dir",
        "--data-dir",
        required=True,
        type=Path,
        help="Directory of calibrated FITS frames (globbed recursively).",
    )
    ap.add_argument(
        "--results_dir",
        "--results-dir",
        dest="results_dir",
        required=True,
        type=Path,
        help="Full output directory for all products (CSV/PNG/GIF/NPZ/log).",
    )
    ap.add_argument("--bands", nargs="+", default=DEFAULT_BANDS)
    ap.add_argument(
        "--ref_band",
        "--ref-band",
        default=None,
        help="If given, all bands align to this band's frame. If omitted "
        "(default), each band self-references its own first frame "
        "(recommended for multi-camera instruments like MuSCAT3/4).",
    )
    ap.add_argument(
        "--refid",
        type=int,
        default=None,
        help="Reference-frame FITS file number (the 4-digit number after the "
        "date in the filename, e.g. 1480 for MCT20_1911191480.fits). "
        "Searches each band's science frames for the closest match "
        "(default: 0 for self-reference, middle frame when --ref_band is set).",
    )
    ap.add_argument(
        "--aper_radii",
        "--aper-radii",
        dest="aper_radii",
        type=parse_aper_grid,
        default=None,
        help="Custom aperture grid 'MIN,MAX,DR' (inclusive of MAX), e.g. "
        "'10,20,2'. Bypasses the Gaia heuristic. Requires --annulus.",
    )
    ap.add_argument(
        "--annulus",
        type=parse_pair,
        default=None,
        help="Background annulus 'RIN,ROUT' in the same unit as --aper_radii. "
        "Required when --aper_radii is given.",
    )
    ap.add_argument(
        "--aper_unit",
        "--aper-unit",
        dest="aper_unit",
        choices=["pix", "fwhm"],
        default="pix",
        help="Unit for --aper_radii/--annulus: 'pix' (default) or 'fwhm' "
        "(radii scaled by each image's FWHM).",
    )
    ap.add_argument(
        "--annulus_pix",
        "--annulus-pix",
        dest="annulus_pix",
        type=parse_pair,
        default=None,
        help="Custom fixed sky annulus 'RIN,ROUT' in pixels (default: 20,30). "
        "Bypasses the default FWHM-based nominal ring but still shifts "
        "inward to exclude contaminants.",
    )
    ap.add_argument("--glob", default="*.fits", help="FITS glob pattern.")
    ap.add_argument(
        "--site",
        default=None,
        help="Only reduce data from this site (only applicable for sinistro instrument).",
    )
    ap.add_argument(
        "--mode",
        default=None,
        choices=mode_choices,
        help="Only reduce data in this mode (only applicable for sinistro instrument).",
    )
    ap.add_argument(
        "--gif_stride",
        "--gif-stride",
        type=int,
        default=DEFAULT_GIF_STRIDE,
        help="Target number of frames to show in the quick-look GIF, equally-spaced in time (default: 10).",
    )
    ap.add_argument(
        "--gif",
        dest="make_gif",
        action="store_true",
        default=False,
        help="Render a quick-look GIF per band (off by default; GIF rendering "
        "is the slowest stage for batch reductions).",
    )
    ap.add_argument(
        "--use_barycorrpy",
        "--use-barycorrpy",
        action="store_true",
        help="Use barycorrpy for BJD-TDB (default: astropy light-travel).",
    )
    ap.add_argument(
        "--wcs_method",
        "--wcs-method",
        dest="wcs_method",
        choices=["twirl", "astrometry.net"],
        default="astrometry.net",
        help="WCS solving method for calibration: 'twirl' (twirl+Gaia, no API key "
        "needed) or 'astrometry.net' (nova.astrometry.net, requires ASTROMETRY_NET_API_KEY). "
        "Default: %(default)s.",
    )
    ap.add_argument(
        "--test_run",
        "--test-run",
        dest="test_run",
        action="store_true",
        help=f"Quick smoke test: use {TEST_RUN_FRAMES} frames per band "
        "centered on the --refid frame (or the first frames if unset). "
        "--refid is interpreted as a FITS file number.",
    )
    ap.add_argument(
        "--test_run_frames",
        "--test-run-frames",
        type=int,
        default=TEST_RUN_FRAMES,
        dest="test_run_frames",
        help=f"Number of frames per band in test-run mode (default: {TEST_RUN_FRAMES}). "
        "Together with --refid a window of this size is centered on the matched frame.",
    )
    ap.add_argument(
        "--min_star_separation",
        "--min-star-separation",
        type=float,
        default=MIN_STAR_SEPARATION,
        dest="min_star_separation",
        help="Minimum separation between detected sources in pixels "
        "(default: %(default)s).",
    )
    ap.add_argument(
        "--min_star_area",
        "--min-star-area",
        type=int,
        default=MIN_STAR_AREA,
        dest="min_star_area",
        help="Minimum area in pixels of detected sources (default: %(default)s).",
    )
    ap.add_argument(
        "--max_num_stars",
        "--max-num-stars",
        type=int,
        default=MAX_NUM_STARS,
        dest="max_num_stars",
        help="Number of brightest stars to keep for PSF modeling "
        "(default: %(default)s).",
    )
    ap.add_argument(
        "--n_stars_align",
        "--n-stars-align",
        type=int,
        default=None,
        dest="n_stars_align",
        help="Number of stars used for image alignment (default: same as "
        "--max_num_stars).",
    )
    ap.add_argument(
        "--cutout_size",
        "--cutout-size",
        type=int,
        default=CUTOUT_SIZE,
        dest="cutout_size",
        help="Side length in pixels of star cutouts (default: %(default)s).",
    )
    ap.add_argument(
        "--ccd_trim",
        "--ccd-trim",
        type=parse_trim,
        default=CCD_TRIM_SIZE_YX,
        dest="ccd_trim_size_yx",
        help="CCD edge trim 'Y,X' in pixels (default: %(default)s).",
    )
    ap.add_argument(
        "--edge_margin",
        "--edge-margin",
        type=int,
        default=None,
        dest="edge_margin",
        help="Exclude detected stars whose centroid is within this many pixels "
        "of any CCD edge from the comparison-star pool (the target is never "
        "excluded). Default: half of --cutout_size, so the PSF cutout box stays "
        "on-chip. Set 0 to disable.",
    )
    ap.add_argument(
        "--avoid_nearby_star",
        "--avoid-nearby-star",
        nargs="?",
        type=parse_avoid_nearby_star,
        const=AVOID_NEARBY_STAR_AUTO,
        default=None,
        dest="avoid_nearby_star",
        help="Reject detected stars from the comparison-star pool when a nearby "
        "companion is present. A numeric value is interpreted as the maximum "
        "companion separation in arcsec. Passing the flag with no value uses "
        "the conservative auto threshold max(2 x FWHM, 3 arcsec); without Gaia "
        "the same threshold is converted to pixels using PIXSCALE. Default: off.",
    )
    ap.add_argument(
        "--bin_size_minutes",
        "--bin-size-minutes",
        type=float,
        default=BIN_SIZE_DAYS * 24 * 60,
        dest="bin_size_minutes",
        help=f"Bin width in minutes for time-series plots "
        f"(default: %(default)g min = {BIN_SIZE_DAYS:.4f} days).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Log INFO messages to the console (default: warnings and errors "
        "only; the log file always records INFO).",
    )
    ap.add_argument(
        "--plot_gaia_sources",
        "--plot-gaia-sources",
        action="store_true",
        default=False,
        help="Mark Gaia source positions in aperture and stack zoom plots.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing products in --results_dir (default: abort if "
        "the directory already contains outputs).",
    )
    ap.add_argument(
        "--calib_dir",
        "--calib-dir",
        type=Path,
        default=None,
        help="Output directory for calibrated FITS files produced during the "
        "MuSCAT/MuSCAT2 (and future instrument) calibration step "
        "(default: <data_dir>_calibrated).",
    )
    ap.add_argument(
        "--sig_bkg",
        "--sig-bkg",
        type=float,
        default=None,
        dest="sig_bkg",
        help="Sigma threshold for sky background outlier clipping (default: disabled).",
    )
    ap.add_argument(
        "--sig_fwhm",
        "--sig-fwhm",
        type=float,
        default=None,
        dest="sig_fwhm",
        help="Sigma threshold for FWHM outlier clipping (default: disabled).",
    )
    ap.add_argument(
        "--sig_dx",
        "--sig-dx",
        type=float,
        default=None,
        dest="sig_dx",
        help="Sigma threshold for drift X outlier clipping (default: disabled).",
    )
    ap.add_argument(
        "--sig_dy",
        "--sig-dy",
        type=float,
        default=None,
        dest="sig_dy",
        help="Sigma threshold for drift Y outlier clipping (default: disabled).",
    )
    ap.add_argument(
        "--cmap",
        default="gray",
        help="Colormap for image display plots (default: 'gray'; use 'gray_r' to reverse).",
    )
    ap.add_argument(
        "--nan_imputation_method",
        "--nan-imputation-method",
        dest="nan_imputation_method",
        choices=["none", "mean", "median", "linear", "spline", "forward_fill"],
        default="linear",
        help="Method to impute NaN values in differential photometry: "
        "'none' (fail if NaNs present), 'mean' (star-wise mean), "
        "'median' (star-wise median), 'linear' (linear interpolation), "
        "'spline' (cubic spline), 'forward_fill' (last-observation-carried-forward). "
        "Default: %(default)s. The Broeg et al. 2005 algorithm requires no NaNs "
        "in the flux matrix.",
    )
    ap.add_argument(
        "--bad_pixel_map",
        "--bad-pixel-map",
        dest="bad_pixel_map",
        type=str,
        default=None,
        help="Path to bad pixel map FITS file, or 'header' to read from reference image "
        "header keyword. Boolean array where True marks bad pixels (hot, dead, etc.) "
        "that should be masked during photometry. Default: None (no masking).",
    )
    args = ap.parse_args(argv)

    if args.aper_radii is not None and args.annulus is None:
        ap.error("--annulus RIN,ROUT is required when --aper_radii is given")
    if args.aper_radii is None and (
        args.annulus is not None or args.aper_unit != "pix"
    ):
        ap.error("--annulus and --aper_unit only apply together with --aper_radii")
    if args.aper_radii is not None and args.aper_radii.max() > args.annulus[0]:
        ap.error(
            f"max aperture radius ({args.aper_radii.max():g}) must be <= "
            f"inner sky annulus radius ({args.annulus[0]:g})"
        )
    if args.mode is not None and args.data_dir.exists():
        modes = _sinistro_modes_from_headers(
            args.data_dir, args.glob, args.target_name, args.bands
        )
        mode_to_match = args.mode.lower()
        if modes and not any(
            mode_to_match == mode or mode_to_match in mode for mode in modes
        ):
            ap.error(
                f"--mode {args.mode!r} was not found in Sinistro data for "
                f"target={args.target_name!r}; available modes: {sorted(modes)}"
            )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    global SIGMA_BKG, SIGMA_FWHM, SIGMA_DX, SIGMA_DY
    SIGMA_BKG = args.sig_bkg
    SIGMA_FWHM = args.sig_fwhm
    SIGMA_DX = args.sig_dx
    SIGMA_DY = args.sig_dy

    assert args.tID not in (args.cID or []), (
        f"tID={args.tID} must not be in cID={args.cID}"
    )
    assert args.tID not in (args.avoid_cids or []), (
        f"tID={args.tID} must not be in avoid_cids={args.avoid_cids}"
    )

    # guard against clobbering an existing reduction (check before the log
    # file is created so the directory's own log does not count as a product)
    product_globs = ("*.csv", "*.npz", "*.png", "*.gif")
    existing = [p for g in product_globs for p in args.results_dir.glob(g)]
    if existing and not args.overwrite:
        print(
            f"error: {args.results_dir} already contains {len(existing)} product "
            "file(s); pass --overwrite to replace them.",
            file=sys.stderr,
        )
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(args.results_dir, verbose=args.verbose)
    t0 = time_module.time()

    logger.info(f"args: {vars(args)}")

    if args.bands is None:
        args.bands = _detect_narrow_bands(args.data_dir, args.target_name)
        logger.info(f"bands: {args.bands}")
    else:
        logger.info(f"bands: {args.bands} (explicit)")

    sciences = {}
    filepath_to_obslog = {}
    inst_obslog = args.data_dir.parent.name.lower()
    obslog_records = frames_from_obslog(args.data_dir, inst_obslog)
    if obslog_records is not None:
        logger.info(f"reading obslog: {len(obslog_records)} frames")
        for rec in obslog_records:
            if rec["object"] != args.target_name:
                continue
            band = _resolve_band(rec["filter"], inst_obslog, args.bands)
            if band is None:
                continue
            path = rec["path"]
            sciences.setdefault(band, []).append(path)
            filepath_to_obslog[path] = rec
        if sciences:
            logger.info(
                f"obslog: frames per band: "
                f"{ {b: len(sciences.get(b, [])) for b in args.bands} }"
            )
            missing = [b for b in args.bands if b not in sciences]
            if missing:
                logger.warning(f"obslog has no frames for bands: {missing}")
        else:
            logger.warning(
                "obslog found but no matching frames; falling back to header scan"
            )
            sciences = {}

    if not sciences:
        files = sorted(args.data_dir.glob(args.glob))
        if not files:
            files = sorted(args.data_dir.rglob(args.glob))
        logger.info(f"found {len(files)} FITS files in {args.data_dir}")
        if not files:
            logger.error("no FITS files found; aborting")
            return 1
        instrume = inst_obslog if inst_obslog in INSTRUMENT_FILTER_ALIASES else ""
        inst_aliases = INSTRUMENT_FILTER_ALIASES.get(instrume)
        sciences = read_filename_per_band(
            files, args.bands, args.target_name, filter_aliases=inst_aliases
        )

    active_bands = [b for b in args.bands if sciences.get(b)]
    if not active_bands:
        logger.error(f"no frames for target={args.target_name}; aborting")
        return 1

    # header metadata for naming, time conversion
    probe = FITSImage(sciences[active_bands[0]][0]).header
    instrument = get_instrument(probe)

    if args.site is not None and instrument != "sinistro":
        logger.error(
            f"--site can only be specified when instrument is 'sinistro' (found '{instrument}')"
        )
        return 1

    if args.mode is not None and instrument != "sinistro":
        logger.error(
            f"--mode can only be specified when instrument is 'sinistro' (found '{instrument}')"
        )
        return 1

    if instrument == "sinistro" and args.site:
        site_to_match = args.site.lower()
        allowed_sites = ("lsc", "cpt", "coj", "tfn", "elp")
        if site_to_match not in allowed_sites:
            logger.error(
                f"Invalid site '{args.site}' for sinistro. Must be one of {allowed_sites}"
            )
            return 1

        filtered_sciences = {}
        for b, fs in sciences.items():
            matching = []
            for f in fs:
                try:
                    hdr = fits.getheader(f)
                    file_site = str(hdr.get("SITEID") or hdr.get("SITE") or "").lower()
                    if file_site == site_to_match:
                        matching.append(f)
                except Exception as e:
                    logger.warning(f"Could not read header of {f}: {e}")
            if matching:
                filtered_sciences[b] = matching
        sciences = filtered_sciences

    if instrument == "sinistro" and args.site is None:
        unique_sites = set()
        for b, fs in sciences.items():
            for f in fs:
                try:
                    hdr = fits.getheader(f)
                    file_site = (
                        str(hdr.get("SITEID") or hdr.get("SITE") or "").strip().lower()
                    )
                except Exception as e:
                    logger.warning(f"Could not read header of {f}: {e}")
                    file_site = ""
                if file_site:
                    unique_sites.add(file_site)
        if len(unique_sites) > 1:
            raise ValueError(
                f"Multiple sites found in the dataset for sinistro: {sorted(unique_sites)}. "
                "Please specify --site to select one."
            )

    if instrument == "sinistro" and args.mode:
        mode_to_match = args.mode.lower()
        allowed_modes = ("central_2k_2x2", "full_frame")
        if mode_to_match not in allowed_modes:
            logger.error(
                f"Invalid mode '{args.mode}' for sinistro. Must be one of {allowed_modes}"
            )
            return 1

        filtered_sciences = {}
        for b, fs in sciences.items():
            matching = []
            for f in fs:
                rec = filepath_to_obslog.get(f)
                confmode = None
                if rec is not None:
                    confmode = (
                        rec.get("confmode") or rec.get("mode") or rec.get("CONFMODE")
                    )
                if not confmode:
                    try:
                        hdr = fits.getheader(f)
                        confmode = hdr.get("CONFMODE")
                    except Exception as e:
                        logger.warning(f"Could not read header of {f}: {e}")
                if confmode:
                    confmode_str = str(confmode).lower()
                    if mode_to_match == confmode_str or mode_to_match in confmode_str:
                        matching.append(f)
            if matching:
                filtered_sciences[b] = matching
        sciences = filtered_sciences

    if instrument == "sinistro" and args.mode is None:
        unique_modes = set()
        for b, fs in sciences.items():
            for f in fs:
                rec = filepath_to_obslog.get(f)
                confmode = None
                if rec is not None:
                    confmode = (
                        rec.get("confmode") or rec.get("mode") or rec.get("CONFMODE")
                    )
                if not confmode:
                    try:
                        hdr = fits.getheader(f)
                        confmode = hdr.get("CONFMODE")
                    except Exception as e:
                        logger.warning(f"Could not read header of {f}: {e}")
                if confmode:
                    unique_modes.add(str(confmode).strip().lower())
        if len(unique_modes) > 1:
            raise ValueError(
                f"Multiple configuration modes found in the dataset for sinistro: {list(unique_modes)}. "
                "Please specify --mode to select one."
            )

    if instrument != "muscat2" and instrument != "muscat" and args.test_run:
        nrf = args.test_run_frames
        if args.refid is not None:
            new_sciences = {}
            for b, fs in sciences.items():
                start = max(0, _find_frame_by_number(fs, args.refid) - nrf // 2)
                new_sciences[b] = fs[start : start + nrf]
            sciences = new_sciences
        else:
            sciences = {b: fs[:nrf] for b, fs in sciences.items()}
        logger.info(f"test-run: limiting to {nrf} frames per band (refid={args.refid})")

    active_bands = [b for b in args.bands if sciences.get(b)]
    if not active_bands:
        if instrument == "sinistro":
            details = []
            if args.site:
                details.append(f"site={args.site}")
            if args.mode:
                details.append(f"mode={args.mode}")
            details_str = " and ".join(details)
            logger.error(
                f"no frames for target={args.target_name} at {details_str}; aborting"
            )
            if args.mode:
                raise SystemExit(1)
        else:
            logger.error(f"no frames for target={args.target_name}; aborting")
        return 1

    probe = FITSImage(sciences[active_bands[0]][0]).header
    counts = {b: len(sciences.get(b, [])) for b in args.bands}
    logger.info(f"frames per band: {counts}")

    if instrument == "muscat2" or instrument == "muscat":
        is_muscat = instrument == "muscat"
        calib_label = "muscat" if is_muscat else "muscat2"
        calib_mod = calibrate_muscat if is_muscat else calibrate_muscat2
        default_bands = ["gp", "rp", "zs"] if is_muscat else ["gp", "rp", "ip", "zs"]

        if args.bands is None:
            args.bands = default_bands
            logger.info(f"{calib_label} bands: {args.bands}")
        calib_dir = args.calib_dir or args.data_dir.with_name(
            args.data_dir.name + "_calibrated"
        )
        calib_dir.mkdir(parents=True, exist_ok=True)
        calibrated_files = sorted(calib_dir.glob("*_calibrated.fits"))
        # An interrupted or disk-full calibration can leave zero-byte /
        # truncated FITS behind. Treat those as missing so we recalibrate
        # instead of crashing on a later fits.getheader(calibrated_files[0]).
        good_files = [f for f in calibrated_files if f.stat().st_size > 0]
        if len(good_files) != len(calibrated_files):
            logger.warning(
                f"{calib_label}: ignoring "
                f"{len(calibrated_files) - len(good_files)} empty/corrupt "
                f"calibrated frame(s); will recalibrate"
            )
        calibrated_files = good_files
        need_calib = not calibrated_files

        if not args.test_run and not need_calib:
            raw_count = sum(len(fs) for fs in sciences.values())
            if len(calibrated_files) < raw_count:
                logger.info(
                    f"{calib_label}: found {len(calibrated_files)} calibrated frames "
                    f"but {raw_count} raw frames; re-calibrating"
                )
                need_calib = True

        if calibrated_files and not need_calib:
            missing, unreadable, no_wcs, wrong_method = _calibrated_wcs_problems_by_band(
                calib_label=calib_label,
                calibrated_files=calibrated_files,
                active_bands=active_bands,
                requested_bands=args.bands,
                target_name=args.target_name,
                wcs_method=args.wcs_method,
            )
            if missing or unreadable:
                need_calib = True
                details = []
                if missing:
                    details.append(f"missing bands: {', '.join(missing)}")
                if unreadable:
                    details.append(f"unreadable headers: {', '.join(unreadable)}")
                logger.info(
                    f"{calib_label}: calibrated frames incomplete/unreadable "
                    f"({'; '.join(details)}); re-calibrating"
                )
            elif no_wcs or wrong_method:
                if _inject_wcs_from_sidecars(
                    calib_label=calib_label,
                    calib_dir=calib_dir,
                    calibrated_files=calibrated_files,
                    active_bands=active_bands,
                    requested_bands=args.bands,
                    target_name=args.target_name,
                    wcs_method=args.wcs_method,
                ):
                    details = []
                    if no_wcs:
                        details.append(f"lacked usable WCS: {', '.join(no_wcs)}")
                    if wrong_method:
                        details.append(
                            f"method changed to {args.wcs_method}: "
                            f"{', '.join(wrong_method)}"
                        )
                    logger.info(
                        f"{calib_label}: reused cached sidecar WCS "
                        f"({'; '.join(details)})"
                    )
                else:
                    need_calib = True
                    sidecar_dir = calib_dir / ".wcs"
                    logger.info(
                        f"{calib_label}: calibrated WCS needs repair "
                        f"(no usable WCS: {', '.join(no_wcs) or 'none'}; "
                        f"wrong method: {', '.join(wrong_method) or 'none'}); "
                        f"re-calibrating (sidecar files: "
                        f"{sum(1 for b in active_bands if (sidecar_dir / f'{b}_{args.wcs_method}.wcs.fits').exists())}"
                        f"/{len(active_bands)} bands)"
                    )
        if need_calib:
            # WCS solving only happens here (muscat/muscat2 calibration);
            # BANZAI-reduced muscat3/muscat4/sinistro never reach this branch.
            # 'astrometry.net' needs an Astrometry.net key; fail fast and point at twirl
            # rather than dying deep inside calibration after wasted work.
            if (
                args.wcs_method == "astrometry.net"
                and not os.environ.get("ASTROMETRY_NET_API_KEY", "").strip()
            ):
                logger.error(
                    f"{calib_label}: --wcs_method astrometry.net requires the "
                    "ASTROMETRY_NET_API_KEY environment variable, which is not "
                    "set. Either export ASTROMETRY_NET_API_KEY, or re-run with "
                    "--wcs_method twirl (twirl+Gaia, no API key needed)."
                )
                return 1
            logger.info(f"{calib_label}: running calibration")
            calibration_bands = (
                list(args.bands) if set(args.bands).issubset(default_bands) else None
            )
            calib_args = _calibration_args(args, calib_dir, calibration_bands)
            # Forward main logger's FileHandlers to the calibration and WCS loggers so details get written to the target's .log file
            calib_mod.logger.setLevel(logging.INFO)
            _wcs_logger.setLevel(logging.INFO)
            for h in logger.handlers:
                if isinstance(h, logging.FileHandler):
                    calib_mod.logger.addHandler(h)
                    _wcs_logger.addHandler(h)
            ret = calib_mod.main(calib_args)
            if ret != 0:
                logger.error(f"{calib_label} calibration failed")
                return 1
            calibrated_files = sorted(calib_dir.glob("*_calibrated.fits"))
        if not calibrated_files:
            logger.error(f"{calib_label}: no calibrated frames found")
            return 1
        logger.info(f"{calib_label}: {len(calibrated_files)} calibrated frames")
        sciences = read_filename_per_band(
            calibrated_files,
            args.bands,
            args.target_name,
            filter_aliases=INSTRUMENT_FILTER_ALIASES.get(calib_label),
        )
        if args.test_run:
            nrf = args.test_run_frames
            if args.refid is not None:
                new_sciences = {}
                for b, fs in sciences.items():
                    start = max(0, _find_frame_by_number(fs, args.refid) - nrf // 2)
                    new_sciences[b] = fs[start : start + nrf]
                sciences = new_sciences
            else:
                sciences = {b: fs[:nrf] for b, fs in sciences.items()}
        counts = {b: len(sciences.get(b, [])) for b in args.bands}
        logger.info(f"frames per band: {counts}")
        active_bands = [b for b in args.bands if sciences.get(b)]
        if not active_bands:
            logger.error(
                f"{calib_label}: no calibrated frames for target={args.target_name}; aborting"
            )
            return 1
        probe = FITSImage(sciences[active_bands[0]][0]).header
        instrument = get_instrument(probe)
    date = date_from_header(probe)
    logger.info(
        f"target={args.target_name} inst={instrument} date={date} "
        f"site={probe.get('SITE')}"
    )

    if args.target_coord is not None:
        ra_str, dec_str = args.target_coord
        if ":" in ra_str:
            target_coord = SkyCoord(
                ra_str, dec_str, frame="icrs", unit=(u.hourangle, u.deg)
            )
        else:
            target_coord = SkyCoord(ra_str, dec_str, frame="icrs", unit=u.deg)
        logger.info(f"target_coord from --target_coord: {target_coord}")
    else:
        mast_name = _normalize_toi_name(args.target_name)
        try:
            target_coord = Mast().resolve_object(mast_name)
        except Exception as e:
            logger.warning(
                f"MAST resolution failed for {mast_name}: {e}. Retrying with space-separated name."
            )
            try:
                target_coord = Mast().resolve_object(mast_name.replace("-", " "))
            except Exception:
                # Final fallback: try Simbad (resolves EPIC/K2 and other names
                # that MAST does not know).
                logger.warning(
                    f"MAST resolution failed for both '{mast_name}' and "
                    f"'{mast_name.replace('-', ' ')}'. Trying Simbad."
                )
                try:
                    target_coord = _resolve_simbad_target(args.target_name)
                    logger.info(f"target_coord resolved via Simbad: {target_coord}")
                except Exception as simbad_exc:
                    logger.error(
                        f"Simbad resolution also failed for '{args.target_name}': {simbad_exc}. "
                        "Use --target_coord to supply coordinates manually."
                    )
                    raise e
        logger.info(f"target radec: {target_coord}")

    # reference seeding: without --ref_band each band self-references its first
    # frame (within-camera alignment, correct for multi-camera instruments);
    # with --ref_band all bands align to that band's middle frame.
    self_reference = args.ref_band is None
    ref_band = None
    if self_reference:
        logger.info("reference: per-band self-reference (first frame of each band)")
    else:
        if args.ref_band not in active_bands:
            logger.warning(
                f"--ref_band {args.ref_band} has no frames; using {active_bands[0]}"
            )
        ref_band = args.ref_band if args.ref_band in active_bands else active_bands[0]
        logger.info(f"reference: all bands aligned to {ref_band}-band frame")

    # aperture spec shared by all bands; None radii -> per-band Gaia heuristic
    scale = args.aper_unit == "fwhm"
    arin, arout = args.annulus if args.annulus is not None else (None, None)

    if args.avoid_cids and self_reference:
        logger.warning(
            "--avoid_cids requires --ref_band to unify star indices across "
            "bands; ignoring --avoid_cids"
        )
        args.avoid_cids = None

    # When --avoid_cids is set and --ref_band is used, process the reference
    # band first and collect its source positions so they can be cross-matched
    # to the other bands' source catalogs (all bands are aligned to the same
    # reference frame, so pixel positions correspond).
    ordered_bands = list(active_bands)
    if not self_reference:
        ref_band = args.ref_band if args.ref_band in active_bands else active_bands[0]
        if ref_band in ordered_bands:
            ordered_bands.remove(ref_band)
            ordered_bands.insert(0, ref_band)

    band_results = {}
    failed_bands = []
    ref_source_positions: np.ndarray | None = None
    inferred_target_positions: list[np.ndarray] = []
    inferred_target_bands: list[str] = []

    # Load bad pixel map if provided
    bad_pixel_map = None
    if args.bad_pixel_map:
        # Load the reference image to get shape/metadata if needed
        ref_image = None
        if args.bad_pixel_map.lower() == "header" and sciences:
            try:
                first_band = ordered_bands[0]
                first_file = sciences[first_band][0]
                ref_image = FITSImage(first_file)
            except Exception as e:
                logger.warning(
                    f"could not load reference image for header extraction: {e}"
                )
        bad_pixel_map = load_bad_pixel_map(args.bad_pixel_map, ref_image=ref_image)

    selected_ref_files = {}
    for band in ordered_bands:
        band_files = sciences[band]
        if self_reference:
            ref_files, default_refid = band_files, 0
        else:
            ref_files, default_refid = sciences[ref_band], len(sciences[ref_band]) // 2
        if args.refid is not None:
            refid = _find_frame_by_number(ref_files, args.refid)
        else:
            refid = default_refid
        refid = min(refid, len(ref_files) - 1)
        selected_ref_files[band] = ref_files[refid]

    ref_wcs_ok_by_band: dict[str, bool] = {}
    if self_reference and args.tID is None and target_coord is not None:
        ref_wcs_ok_by_band = {
            band: _reference_file_has_usable_wcs(selected_ref_files[band], target_coord)
            for band in ordered_bands
        }
        inferred_order = _order_bands_for_target_id_inference(
            ordered_bands, ref_wcs_ok_by_band
        )
        if inferred_order != ordered_bands:
            ordered_bands = inferred_order
            logger.info(
                "target ID inference: processing WCS-solved bands first "
                f"({', '.join(b for b in ordered_bands if ref_wcs_ok_by_band.get(b, False))})"
            )

    for band in ordered_bands:
        band_files = sciences[band]
        target_pixel_override = _target_pixel_override_for_band(
            args.tID,
            self_reference,
            inferred_target_positions,
            ref_wcs_ok_by_band.get(band, True),
        )
        if target_pixel_override is not None:
            logger.info(
                f"[{band}] reference has no usable WCS; inferring target ID near "
                f"pixel ({target_pixel_override[0]:.1f}, {target_pixel_override[1]:.1f}) "
                f"from {','.join(inferred_target_bands)}-band target position(s)"
            )
        try:
            res = run_band(
                band,
                band_files,
                selected_ref_files[band],
                target_coord,
                aper_radii=args.aper_radii,
                rin=arin,
                rout=arout,
                scale=scale,
                ccd_trim_size_yx=args.ccd_trim_size_yx,
                max_num_stars=args.max_num_stars,
                min_star_separation=args.min_star_separation,
                cutout_size=args.cutout_size,
                n_stars_align=args.n_stars_align,
                target_index_override=args.tID,
                target_pixel_override=target_pixel_override,
                cids=args.cID,
                avoid_cids=args.avoid_cids,
                ref_source_positions=ref_source_positions,
                min_area=args.min_star_area,
                plot_gaia_sources=args.plot_gaia_sources,
                edge_margin=args.edge_margin,
                avoid_nearby_star=args.avoid_nearby_star,
                annulus_pix=args.annulus_pix,
                nan_imputation_method=args.nan_imputation_method,
                bad_pixel_map=bad_pixel_map,
            )
        except Exception as exc:  # noqa: BLE001 - one bad band must not kill the run
            logger.exception(f"[{band}] reduction failed: {exc}")
            failed_bands.append(band)
            continue
        if res is None:
            logger.error(f"[{band}] reduction produced no output; marking band failed")
            failed_bands.append(band)
            continue
        band_results[band] = res
        if (
            args.tID is None
            and self_reference
            and ref_wcs_ok_by_band.get(band, False)
            and not res.get("defaulted_to_brightest", False)
        ):
            target_pos = np.asarray(
                res["ref"].sources[int(res["target_index"])].coords, dtype=float
            )
            if not inferred_target_positions:
                inferred_target_positions.append(target_pos)
                inferred_target_bands.append(band)
                logger.info(
                    f"[{band}] target pixel ({target_pos[0]:.1f}, {target_pos[1]:.1f}) available "
                    "for bands without usable WCS"
                )
            else:
                inferred_pixel = np.median(
                    np.asarray(inferred_target_positions, dtype=float), axis=0
                )
                distance = float(np.linalg.norm(target_pos - inferred_pixel))
                if distance > _TARGET_PIXEL_INFERENCE_TOLERANCE_PX:
                    logger.warning(
                        f"[{band}] WCS-derived target pixel "
                        f"({target_pos[0]:.1f}, {target_pos[1]:.1f}) is "
                        f"{distance:.1f} px from previous inferred target position "
                        f"({inferred_pixel[0]:.1f}, {inferred_pixel[1]:.1f}); "
                        "excluding it from no-WCS inference"
                    )
                else:
                    inferred_target_positions.append(target_pos)
                    inferred_target_bands.append(band)
                    logger.info(
                        f"[{band}] target pixel ({target_pos[0]:.1f}, {target_pos[1]:.1f}) "
                        "added for no-WCS inference"
                    )
        if not self_reference and band == ref_band:
            ref_source_positions = np.array([s.coords for s in res["ref"].sources])

    elapsed = time_module.time() - t0
    if not band_results:
        logger.error(
            f"photometry FAILED: 0/{len(ordered_bands)} bands reduced "
            f"({elapsed:.0f}s elapsed)"
        )
        return 1

    site = None
    resolved_confmode = None
    if instrument == "sinistro" and probe is not None:
        site = probe.get("SITEID") or probe.get("SITE")
        if site:
            site = str(site).lower()
        if args.mode is not None:
            resolved_confmode = args.mode
        else:
            first_file = sciences[active_bands[0]][0]
            rec = filepath_to_obslog.get(first_file)
            if rec is not None:
                resolved_confmode = (
                    rec.get("confmode") or rec.get("mode") or rec.get("CONFMODE")
                )
            if not resolved_confmode:
                resolved_confmode = probe.get("CONFMODE")

    simbad_df = None
    if target_coord is not None:
        try:
            simbad_df = get_simbad_data(target_coord, instrument)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SIMBAD query failed: {exc}")

    stem_multi = build_summary_stem(
        args.target_name,
        instrument,
        date,
        active_bands,
        site=site,
        confmode=resolved_confmode,
    )
    bjds = {}
    for band, r in band_results.items():
        stem = build_stem(
            args.target_name,
            instrument,
            date,
            band,
            site=site,
            confmode=resolved_confmode,
        )
        bjds[band] = compute_bjd_tdb(
            r["diff"], r["ref"].header, target_coord, args.use_barycorrpy, instrument
        )
        if args.test_run:
            logger.info(f"test-run: skipping {stem}.csv")
        else:
            csv_path = args.results_dir / f"{stem}.csv"
            photometry_df(r["diff"], bjds[band]).to_csv(csv_path, index=False)
            logger.info(f"wrote {csv_path}")

        plot_ref_image(
            r,
            target_coord,
            instrument,
            args.results_dir / f"{stem}_ref.png",
            target_name=args.target_name,
            date=date,
            avoid_cids=r.get("avoid_cids"),
            plot_gaia_sources=args.plot_gaia_sources,
            simbad_df=simbad_df,
            cmap=args.cmap,
        )
        plot_apertures(
            r,
            args.results_dir / f"{stem}_apertures.png",
            target_name=args.target_name,
            instrument=instrument,
            date=date,
            plot_gaia_sources=args.plot_gaia_sources,
            target_coord=target_coord,
            cmap=args.cmap,
        )
        plot_alignment(
            r,
            r["files"][-1],
            args.results_dir / f"{stem}_alignment.png",
            args.target_name,
            instrument,
            date,
            target_index=r["target_index"],
            ccd_trim_size_yx=args.ccd_trim_size_yx,
            max_num_stars=args.max_num_stars,
            min_star_separation=args.min_star_separation,
            n_stars_align=args.n_stars_align,
            min_area=args.min_star_area,
            cmap=args.cmap,
        )
        plot_cutouts(
            r,
            args.results_dir / f"{stem}_cutouts.png",
            args.target_name,
            instrument,
            band,
            date,
            max_num_stars=args.max_num_stars,
            plot_gaia_sources=args.plot_gaia_sources,
            target_coord=target_coord,
            simbad_df=simbad_df,
            cmap=args.cmap,
        )
        if args.make_gif:
            stride_step = (
                1 if args.test_run else max(1, len(r["files"]) // args.gif_stride)
            )
            make_gif(
                r["files"],
                args.results_dir / f"{stem}.gif",
                stride_step,
                cmap=args.cmap,
            )

    # Compile nearby stars within the outer annulus once for the entire run
    if args.test_run:
        logger.info(f"test-run: skipping {stem_multi}_nearby_stars.csv")
    else:
        r_ref = None
        for r in band_results.values():
            if (
                r.get("gaia_df") is not None
                and len(r["gaia_df"])
                and getattr(r.get("ref"), "wcs", None) is not None
                and len(r["ref"].sources) > 0
            ):
                r_ref = r
                break

        nearby_stars_data = []
        gaia_df = None
        ref = None
        rout = None
        pixscale = None

        if r_ref is not None:
            gaia_df = r_ref["gaia_df"]
            ref = r_ref["ref"]
            rout = r_ref["rout"]
            pixscale = float(ref.telescope.pixel_scale)
        else:
            # Fallback: if the full-image Gaia query failed, attempt a target-centered query
            # with a small radius (e.g. 3 arcminutes) and a tight limit (e.g. 1000)
            # to retrieve nearby stars for plotting/contamination analysis.
            logger.info(
                "Attempting small-region fallback Gaia query around target coordinates..."
            )
            for r in band_results.values():
                if getattr(r.get("ref"), "wcs", None) is not None:
                    ref = r["ref"]
                    rout = r["rout"]
                    pixscale = float(ref.telescope.pixel_scale)
                    break
            if ref is not None and target_coord is not None:
                try:
                    from prose.utils import gaia_query

                    fallback_radius_deg = (
                        180.0 / 3600.0
                    )  # 180 arcseconds = 3 arcminutes
                    table = gaia_query(
                        target_coord, fallback_radius_deg * 2, "*", limit=1000
                    )
                    if table is not None and len(table) > 0:
                        table.rename_column("DESIGNATION", "id")
                        gaia_df = table.to_pandas()
                        logger.info(
                            f"Fallback small-region Gaia query succeeded: {len(gaia_df)} rows"
                        )
                except Exception as exc:
                    logger.warning(f"Fallback Gaia query failed: {exc}")

        if gaia_df is not None and ref is not None:
            try:
                gaia_coords = SkyCoord(
                    gaia_df.ra.values, gaia_df.dec.values, unit="deg"
                )
                target_idx_in_gaia = target_coord.separation(gaia_coords).argmin()
                target_g_mag = float(gaia_df.phot_g_mean_mag.values[target_idx_in_gaia])

                # Project detected sources to RA/Dec
                detected_pix_coords = np.array(
                    [s.coords for s in ref.sources], dtype=float
                )
                detected_coords = ref.wcs.pixel_to_world(*detected_pix_coords.T)

                # Match Gaia catalog to detected catalog
                match_idx, match_sep, _ = gaia_coords.match_to_catalog_sky(
                    detected_coords
                )

                for i in range(len(gaia_df)):
                    if i == target_idx_in_gaia:
                        continue

                    sep_arc = float(target_coord.separation(gaia_coords[i]).arcsec)
                    sep_p = sep_arc / pixscale

                    if sep_p <= rout:
                        g_mag = gaia_df.phot_g_mean_mag.values[i]
                        if np.isnan(g_mag) or np.isnan(target_g_mag):
                            delta_mag = np.nan
                            contam_ratio = np.nan
                        else:
                            delta_mag = float(g_mag - target_g_mag)
                            contam_ratio = float(10 ** (-delta_mag / 2.5) * 100)

                        det_sep_arc = float(match_sep[i].arcsec)
                        detected_str = "Y" if det_sep_arc <= 1.5 else "N"

                        source_id = (
                            gaia_df.source_id.values[i]
                            if "source_id" in gaia_df
                            else (gaia_df.id.values[i] if "id" in gaia_df else "")
                        )

                        nearby_stars_data.append(
                            {
                                "Separation (arcsec)": round(sep_arc, 3),
                                "Separation (pix)": round(sep_p, 2),
                                "Gaia delta mag": round(delta_mag, 3)
                                if not np.isnan(delta_mag)
                                else None,
                                "Detected (Y/N)": detected_str,
                                "Contamination Ratio (%)": round(contam_ratio, 4)
                                if not np.isnan(contam_ratio)
                                else None,
                                "Gaia Source ID": str(source_id),
                                "RA (deg)": round(float(gaia_df.ra.values[i]), 6),
                                "Dec (deg)": round(float(gaia_df.dec.values[i]), 6),
                                "Gaia G mag": round(float(g_mag), 3)
                                if not np.isnan(g_mag)
                                else None,
                            }
                        )
            except Exception as exc:
                logger.warning(f"Failed to compile nearby stars: {exc}")

        nearby_stars_data.sort(key=lambda x: x["Separation (arcsec)"])
        nearby_csv_path = args.results_dir / f"{stem_multi}_nearby_stars.csv"
        pd.DataFrame(nearby_stars_data).to_csv(nearby_csv_path, index=False)
        logger.info(f"wrote {nearby_csv_path}")

    bin_size_days = args.bin_size_minutes / (24 * 60)
    target_index = next(iter(band_results.values()))["target_index"]
    if args.test_run:
        logger.info("test-run: skipping combined lightcurve plot")
    else:
        plot_lightcurves(
            band_results,
            args.results_dir / f"{stem_multi}_lightcurves.png",
            args.target_name,
            instrument,
            date,
            target_index=target_index,
            bin_size_days=bin_size_days,
        )
    if args.test_run:
        logger.info("test-run: skipping raw flux and covariate plots")
    else:
        plot_raw_flux(
            band_results,
            args.results_dir / f"{stem_multi}_raw_flux.png",
            args.target_name,
            instrument,
            date,
            target_index=target_index,
        )
        plot_covariates(
            band_results,
            args.results_dir / f"{stem_multi}_covariates.png",
            args.target_name,
            instrument,
            date,
            target_index=target_index,
        )
    plot_stacks(
        band_results,
        args.results_dir / f"{stem_multi}_stacks.png",
        args.target_name,
        instrument,
        date,
        target_index=target_index,
        plot_gaia_sources=args.plot_gaia_sources,
        target_coord=target_coord,
        cmap=args.cmap,
    )
    if args.test_run:
        logger.info("test-run: skipping multi-band .npz archive")
    else:
        save_all_bands_npz(
            band_results, bjds, args.results_dir / f"{stem_multi}.npz", meta=vars(args)
        )

    # Dump the reference frame's full FITS header to a sidecar text file so it
    # can be inspected from the web GUI without downloading the data archive.
    # The saved .txt should be the actual header from the specified band and frame ID.
    try:
        target_ref_band = ref_band if ref_band is not None else active_bands[0]
        if self_reference:
            ref_files, default_refid = sciences[target_ref_band], 0
        else:
            ref_files, default_refid = (
                sciences[target_ref_band],
                len(sciences[target_ref_band]) // 2,
            )

        if args.refid is not None:
            refid = _find_frame_by_number(ref_files, args.refid)
        else:
            refid = default_refid
        refid = min(refid, len(ref_files) - 1)
        actual_ref_file = ref_files[refid]

        ref_header = fits.getheader(actual_ref_file)
        header_path = args.results_dir / f"{stem_multi}_ref_header.txt"
        header_path.write_text(
            ref_header.tostring(sep="\n", padding=False, endcard=False)
        )
        logger.info(f"wrote {header_path} from {actual_ref_file}")
    except Exception as e:
        logger.warning(f"could not write reference header: {e}")

    n_fail = len(failed_bands)
    if n_fail:
        logger.error(
            f"photometry PARTIAL FAILURE: {len(band_results)}/{len(ordered_bands)} "
            f"bands reduced ({elapsed:.0f}s elapsed); failed/skipped={failed_bands}"
        )
    else:
        logger.info(
            f"photometry SUCCEEDED: {len(band_results)}/{len(ordered_bands)} bands "
            f"({elapsed:.0f}s elapsed)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
