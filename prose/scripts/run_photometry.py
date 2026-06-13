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
    {target}_{inst}_{date}_lightcurves.png
    {target}_{inst}_{date}_covariates.png
    {target}_{inst}_{date}_stacks.png
    {target}_{inst}_{date}.npz
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
import sys
import time as time_module
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write figures without a display

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astroquery.mast import Mast
from rich.progress import track

from prose import FITSImage, Fluxes, Sequence, Telescope, blocks
from prose.blocks import catalogs
from prose.core.sequence import SequenceParallel
from prose.scripts import calibrate_muscat, calibrate_muscat2
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
)

# --------------------------- constants / defaults ---------------------------
# ignore if using BANZAI-reduced fits files
CAL_OBJECT_MAP = {"bias": "BIAS", "dark": "DARK", "flat": "FLAT"}

# Fallback observatory site (astropy/astroplan site registry name) for
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
DEFAULT_GIF_STRIDE = 100
TEST_RUN_FRAMES = 10  # frames per band used by --test_run
FPS = 5
GIF_MAX_PX = 512  # max GIF frame dimension [pix]; larger frames are downsampled

# reference-image / detection defaults (mirror the template notebook)
MAX_NUM_STARS = 10  # nth brightest stars to keep
CUTOUT_SIZE = 35  # cutout size of detected stars [pix]
CCD_TRIM_SIZE_YX = (0, 0)  # trim image edges [pix]
MIN_STAR_AREA = 10  # min detected-source area [pix]
MIN_STAR_SEPARATION = 10  # min separation between sources [pix]

# Gaia aperture-radii / sky-annulus heuristic
APER_STEP_PIX = 2  # spacing of aperture radii [pix]
ANNULUS_INNER_FWHM = 6  # nominal inner sky-annulus radius [FWHM]
ANNULUS_OUTER_FWHM = 10  # nominal outer sky-annulus radius [FWHM]
ANNULUS_MAX_PIX = 100  # clamp rout when FWHM is large (defocus) [pix]
CONTAM_DMAG = 2.5  # neighbour contaminates if Gmag - target < this (>=10% target flux)
CONTAM_MARGIN_PIX = 2  # keep annulus/aperture this far inside a contaminant [pix]
GAIA_CUTOUT = (200, 200)  # cutout around target for the Gaia query [pix]

# differential-photometry cleaning
SIGMA_BKG = 3
SIGMA_FWHM = 3
BIN_SIZE_DAYS = 10 / 60 / 24  # 10-minute bins for plots

# GJD->BJD sanity bound (light travel time should be well under this)
MAX_TIME_OFFSET_MIN = 2 * 8.4

# JD = MJD + this offset. Some instruments (e.g. MuSCAT2/TCS, keyword MJD-STRT)
# report their time axis in MJD; prose flags this via Telescope.jd_scale == "mjd".
MJD_TO_JD = 2_400_000.5

# band color map
BAND_COLORS = {"g": "blue", "r": "green", "i": "orange", "z": "red"}


def band_color(band: str) -> str:
    if band == "Na_D":
        return BAND_COLORS["r"]
    return BAND_COLORS.get(band[0], "k")


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


def date_from_header(header) -> str:
    """Return YYMMDD from the ``DAY-OBS`` or ``DATE-OBS`` keyword.

    Handles both compact LCO-style values (``20250416``) and dashed
    ``DATE-OBS`` values that may be non-zero-padded and carry a time component
    (e.g. MuSCAT2 ``2020-3-5`` -> ``200305``).
    """
    raw = str(header.get("DAY-OBS", header.get("DATE-OBS", ""))).strip()
    # keep only the date portion if a time is appended (T or whitespace separated)
    date_part = raw.replace("T", " ").split()[0] if raw else ""
    if "-" in date_part:
        try:
            year, month, day = (int(p) for p in date_part.split("-")[:3])
            return f"{year % 100:02d}{month:02d}{day:02d}"
        except ValueError:
            return date_part.replace("-", "")[2:]
    return date_part[2:]


def build_stem(target: str, inst: str, date: str, band: str | None = None) -> str:
    target = target.replace(" ", "")
    if band is None:
        return f"{target}_{inst}_{date}"
    return f"{target}_{inst}_{band}_{date}"


def _savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {path}")


def _zscale(data: np.ndarray) -> np.ndarray:
    vmin, vmax = ZScaleInterval().get_limits(data)
    return np.clip((data - vmin) / max(vmax - vmin, 1e-9), 0, 1)


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


# --------------------------- reference building ---------------------------


def reference_sequence(
    ccd_trim_size_yx: tuple[int, int] = CCD_TRIM_SIZE_YX,
    max_num_stars: int = MAX_NUM_STARS,
    min_star_separation: float = MIN_STAR_SEPARATION,
    cutout_size: int = CUTOUT_SIZE,
    min_area: int = MIN_STAR_AREA,
) -> Sequence:
    """Calibration sequence run on the per-band reference frame."""
    return Sequence(
        [
            blocks.Trim(ccd_trim_size_yx),
            blocks.PointSourceDetection(
                n=max_num_stars,
                min_area=min_area,
                min_separation=min_star_separation,
            ),
            blocks.Cutouts(shape=cutout_size),
            blocks.MedianEPSF(),
            blocks.psf.Moffat2D(),
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
            stars_radec = wcs.pixel_to_world(*coords.T)
            return int(target_coord.match_to_catalog_sky(stars_radec)[0])
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


def _sky_annulus_pix(fwhm: float, contam_seps: np.ndarray) -> tuple[float, float]:
    """Inner/outer sky-annulus radii [pix] that avoid enclosing a contaminant.

    The annulus is nominally ``ANNULUS_INNER_FWHM``-``ANNULUS_OUTER_FWHM`` times
    the FWHM, with ``rout`` clamped to ``ANNULUS_MAX_PIX`` when the FWHM is large
    (defocused). If a contaminant falls within the nominal ring, the ring is
    shifted inward to sit just inside the nearest such source.
    """
    width = (ANNULUS_OUTER_FWHM - ANNULUS_INNER_FWHM) * fwhm
    rout = min(ANNULUS_OUTER_FWHM * fwhm, ANNULUS_MAX_PIX)
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

    try:
        c = ref.cutout(ref.sources[target_index].coords, GAIA_CUTOUT)
        c.metadata["pixel_scale"] = pixscale * u.arcsec  # required before Gaia query
        c = catalogs.GaiaCatalog(mode="replace")(c)
        df = c.catalogs["gaia"]
        if save_cached_df(cache_path, df):  # refresh cache whenever online
            logger.info(f"cached Gaia result -> {cache_path}")
    except Exception as exc:  # noqa: BLE001 - degrade to cache then FWHM-only
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


def gaia_aperture_radii(ref: FITSImage, target_index: int, target_coord: SkyCoord):
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

    rin, rout = _sky_annulus_pix(fwhm, contam)
    aper_radii = _aperture_radii_pix(fwhm, rin)
    logger.info(
        f"apertures: {len(aper_radii)} radii in [{fwhm:.0f}, {rin:.0f}] px, "
        f"annulus ({rin:.0f}, {rout:.0f}) px, {len(contam)} contaminants (>=10% flux)"
    )
    return aper_radii, rin, rout


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
    min_area: int = MIN_STAR_AREA,
):
    """Build the reference image, target index and aperture geometry.

    PIXSCALE and saturation are read from the reference image header
    (not hardcoded or taken from a probe frame).

    If ``aper_radii`` is provided, the explicit grid (and ``rin``/``rout``) is
    used and the Gaia heuristic is skipped. ``scale`` selects pixel
    (``False``) vs FWHM (``True``) units for the photometry blocks.

    If ``target_index_override`` is given, it bypasses the Gaia cross-match.
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
    ).run(ref, show_progress=False)
    target_index = (
        target_index_override
        if target_index_override is not None
        else find_target_index(ref, target_coord)
    )
    if aper_radii is not None:
        unit = "fwhm" if scale else "pix"
        logger.info(
            f"using custom apertures: {len(aper_radii)} radii in "
            f"[{aper_radii.min():g}, {aper_radii.max():g}] {unit}, "
            f"annulus ({rin:g}, {rout:g}) {unit}"
        )
    else:
        aper_radii, rin, rout = gaia_aperture_radii(ref, target_index, target_coord)
    logger.info(
        f"reference {Path(ref_file).name}: FWHM {float(ref.fwhm):.2f} px, "
        f"target idx {target_index}"
    )
    return dict(
        ref=ref,
        target_index=target_index,
        aper_radii=aper_radii,
        rin=rin,
        rout=rout,
        scale=scale,
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
    min_area: int = MIN_STAR_AREA,
) -> SequenceParallel:
    """Parallel per-image photometry sequence (mirrors the notebook)."""
    if n_stars_align is None:
        n_stars_align = max_num_stars
    blocks_list = [
        blocks.Trim(ccd_trim_size_yx),
        blocks.PointSourceDetection(
            n=max_num_stars,
            min_area=min_area,
            min_separation=min_star_separation,
        ),
        blocks.Cutouts(shape=cutout_size),
        blocks.MedianEPSF(),
        blocks.Gaussian2D(ref),
    ]
    if n_stars_align >= 3:
        blocks_list.append(blocks.ComputeTransformTwirl(ref, n=n_stars_align))
        blocks_list.append(blocks.AlignReferenceSources(ref))
    blocks_list.extend(
        [
            blocks.CentroidQuadratic(),
            blocks.AperturePhotometry(aper_radii, scale=scale),
            blocks.AnnulusBackground(rin=rin, rout=rout, scale=scale),
            blocks.Del("data", "cutouts"),
        ]
    )

    return SequenceParallel(
        blocks=blocks_list,
        data_blocks=[
            blocks.GetFluxes(
                "fwhm",
                airmass=lambda im: im.header.get("AIRMASS", float("nan")),
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
                peak=lambda im: im.sources[0].peak,
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
    cids: list[int] | None = None,
    min_area: int = MIN_STAR_AREA,
):
    """Full reduction for a single band. Returns a result dict or ``None``.
    PIXSCALE and saturation are read from the reference image header
    inside ``build_reference``.
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
        min_area=min_area,
    )
    ref = reference["ref"]

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
        n_stars_align=min(n_stars_align, len(ref.sources))
        if n_stars_align
        else len(ref.sources),
        min_area=min_area,
    )
    phot.run(files)

    fluxes: Fluxes = phot.data[0].fluxes
    if fluxes is None:
        logger.warning(f"[{band}] no valid frames (all discarded); skipping")
        return None
    fluxes.target = reference["target_index"]

    diff = differential_photometry(fluxes, reference["target_index"], cids=cids)
    if diff is None:
        logger.warning(f"[{band}] no valid frames after cleaning; skipping")
        return None

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
        target_index=reference["target_index"],
        aper_radii=np.asarray(reference["aper_radii"]),
        rin=reference["rin"],
        rout=reference["rout"],
        scale=reference["scale"],
    )


def differential_photometry(
    fluxes: Fluxes, target_index: int, cids: list[int] | None = None
):
    """Clean NaN comparison stars, sigma-clip, and run differential photometry.

    When ``cids`` is given only those stars are used as comparisons and the
    automatic selection (Broeg et al. 2005) is skipped.
    """
    fluxes = fluxes.copy()
    fluxes.target = target_index
    if cids is not None:
        mask = np.zeros(fluxes.fluxes.shape[1], dtype=bool)
        mask[target_index] = True
        mask[list(cids)] = True
        fluxes = fluxes.mask_stars(mask)
    else:
        nan_stars = np.any(np.isnan(fluxes.fluxes), axis=(0, 2))
        fluxes = fluxes.mask_stars(~nan_stars)
    fluxes = fluxes.sigma_clipping_data(bkg=SIGMA_BKG, fwhm=SIGMA_FWHM)
    if fluxes.time is None or len(fluxes.time) == 0:
        return None
    if cids is not None:
        return fluxes.diff(comps=np.array(cids))
    return fluxes.autodiff()


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
    from astroplan import Observer

    site = header.get("SITE")
    if site is not None:
        obs_site = Observer.at_site(LCO_SITES[site])
    elif instrument in INSTRUMENT_SITES:
        obs_site = Observer.at_site(INSTRUMENT_SITES[instrument])
        logger.info(
            f"no SITE keyword; using {instrument} site "
            f"'{INSTRUMENT_SITES[instrument]}' for BJD correction"
        )
    else:
        obs_site = None
        logger.warning(
            "SITE keyword not found in header; BJD correction without site location"
        )
    loc = obs_site.location if obs_site is not None else None
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
        logger.info(f"GJD-TDB offset {offset_min:.2f} min")
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


def plot_ref_image(r, target_coord, instrument, path: Path) -> None:
    ref = r["ref"]
    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection=ref.wcs)
    ref.show(ax=ax, frame=True)
    ra_pix, dec_pix = ref.wcs.wcs_world2pix(
        [[target_coord.ra.deg, target_coord.dec.deg]], 0
    )[0]
    ax.scatter(ra_pix, dec_pix, s=120, ec="r", fc="none", zorder=10)
    ax.annotate(
        "Target",
        (ra_pix, dec_pix),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color="r",
        zorder=10,
    )
    ref.sources.plot(ax=ax, c="yellow")
    ax.set_title(f"{r['band']} reference", y=1.08)

    simbad = get_simbad_data(target_coord, instrument)
    if simbad is not None and not simbad.empty:
        simbad = simbad[simbad.OTYPE != "Star"]
        if not simbad.empty:
            simbad_coords = SkyCoord(
                ra=simbad.RA, dec=simbad.DEC, unit=(u.hourangle, u.deg)
            )
            x_pix, y_pix = ref.wcs.wcs_world2pix(
                np.column_stack([simbad_coords.ra.deg, simbad_coords.dec.deg]), 0
            ).T
            ax.scatter(x_pix, y_pix, marker="D", s=40, ec="cyan", fc="none", lw=1)
            for xi, yi, label in zip(x_pix, y_pix, simbad.OTYPE):
                ax.annotate(
                    label,
                    (xi, yi),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=5,
                    color="cyan",
                )

    _savefig(fig, path)


def plot_apertures(r, path: Path) -> None:
    ref = r["ref"]
    coords = ref.sources[r["target_index"]].coords
    c = ref.cutout(coords, GAIA_CUTOUT)
    radii_pix, rin_pix, rout_pix = aper_radii_pix(r)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    c.show(ax=ax, zscale=True, sources=False)
    for radius in radii_pix:
        c.sources[0].plot(radius, label=False, c="r")
    c.sources[0].plot(rin_pix, label=False, c="y")
    c.sources[0].plot(rout_pix, label=False, c="y")
    ax.set_title(f"{r['band']} apertures")
    _savefig(fig, path)


def plot_alignment(
    r,
    other_file,
    path: Path,
    target_name: str,
    instrument: str,
    date: str,
    target_index: int,
    ccd_trim_size_yx: tuple[int, int] = CCD_TRIM_SIZE_YX,
    max_num_stars: int = MAX_NUM_STARS,
    min_star_separation: float = MIN_STAR_SEPARATION,
    n_stars_align: int | None = None,
    min_area: int = MIN_STAR_AREA,
) -> None:
    """Overlay the reference image with an aligned later frame (best effort)."""
    if n_stars_align is None:
        n_stars_align = max_num_stars
    from skimage.transform import warp

    ref = r["ref"]
    try:
        seq = Sequence(
            [
                blocks.Trim(ccd_trim_size_yx),
                blocks.PointSourceDetection(
                    n=max_num_stars,
                    min_area=min_area,
                    min_separation=min_star_separation,
                ),
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
        ax.imshow(_zscale(ref.data), cmap="Greys_r", origin="lower")
        ax.imshow(_zscale(img), cmap="Reds_r", origin="lower", alpha=0.5)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5, color="white")
        ax.axis("off")
    fig.suptitle(f"{target_name} | {instrument} | {date} | tID={target_index}")
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
        1, len(bands), figsize=(4 * len(bands), 7), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    signals = ["flux", "fwhm", "airmass", "bkg", "dx", "dy"]
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
    fig.suptitle(f"{target_name} | {instrument} | {date} | tID={target_index}")
    _savefig(fig, path)


def _radial_profile(data, center):
    y, x = np.indices(data.shape)
    rr = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)
    tbin = np.bincount(rr.ravel(), data.ravel())
    nr = np.bincount(rr.ravel())
    return tbin / np.maximum(nr, 1)


def plot_stacks(
    band_results,
    path: Path,
    target_name: str,
    instrument: str,
    date: str,
    target_index: int,
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
        c = ref.cutout(ref.sources[r["target_index"]].coords, GAIA_CUTOUT)
        center = np.array(c.data.shape)[::-1] / 2
        axes[row, 0].imshow(_zscale(c.data), cmap="Greys_r", origin="lower")
        axes[row, 0].set_title(f"target zoom ({band})")
        axes[row, 0].axis("off")

        radii_pix, rin_pix, rout_pix = aper_radii_pix(r)
        prof = _radial_profile(c.data, center)
        axes[row, 1].plot(prof, ".", c=bc, ms=6)
        axes[row, 1].plot(prof, c=bc)
        best = float(radii_pix[min(int(diff.aperture), len(radii_pix) - 1)])
        axes[row, 1].axvline(best, color="r", alpha=0.6, label=f"best: r={best:.0f}")
        axes[row, 0].add_artist(plt.Circle(tuple(center), best, color="r", fill=False))
        for radius in (rin_pix, rout_pix):
            axes[row, 1].axvline(radius, color="y", ls="--", alpha=0.6)
            axes[row, 0].add_artist(
                plt.Circle(tuple(center), radius, color="y", ls="--", fill=False)
            )
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_xlabel("radius (pixels)")
        axes[row, 1].set_ylabel("flux (ADU)")
        axes[row, 1].legend()
    fig.suptitle(f"{target_name} | {instrument} | {date} | tID={target_index}")
    _savefig(fig, path)


# --------------------------- GIF ---------------------------


def _gif_frame(
    data: np.ndarray, label: str = "", max_px: int = GIF_MAX_PX
) -> np.ndarray:
    """Build one 8-bit RGB GIF frame from image data, matplotlib-free.

    The array is z-scaled to 0-255, flipped vertically to match matplotlib's
    ``origin="lower"`` display convention, downsampled so its longest side is
    ``max_px``, and stamped with ``label`` (e.g. ``DATE-OBS``) via PIL. This
    avoids the per-frame Figure/savefig round-trip that dominated runtime
    (see ``cprofile_results.txt``).
    """
    from PIL import Image, ImageDraw

    arr = (_zscale(data) * 255).astype(np.uint8)
    arr = np.flipud(arr)  # mimic matplotlib origin="lower"
    frame = Image.fromarray(arr, mode="L").convert("RGB")
    longest = max(frame.size)
    if longest > max_px:
        scale = max_px / longest
        frame = frame.resize(
            (max(1, round(frame.width * scale)), max(1, round(frame.height * scale))),
            Image.BILINEAR,
        )
    if label:
        ImageDraw.Draw(frame).text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(frame)


def make_gif(files, path: Path, stride: int) -> None:
    """Render a quick-look GIF per band without matplotlib."""
    sampled = files[:: max(1, stride)]
    if not sampled:
        return
    frames = []
    for fp in track(sampled, description=f"gif:{path.name}"):
        img = FITSImage(fp)
        frames.append(_gif_frame(img.data, img.header.get("DATE-OBS", "")))
    imageio.mimsave(path, frames, fps=FPS, loop=0)
    logger.info(f"wrote {path}")


# --------------------------- NPZ ---------------------------


def _npz_safe(v):
    arr = np.asarray(v)
    return arr if arr.dtype != object else np.array(v, dtype=object)


def save_all_bands_npz(band_results, bjds, path: Path) -> None:
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
        out[f"{band}__target_index"] = np.array(r["target_index"])
        out[f"{band}__aperture"] = np.array(diff.aperture)
        out[f"{band}__wcs_header"] = np.array(r["ref"].wcs.to_header_string())
        for key in ("fwhm", "airmass", "bkg", "dx", "dy", "peak"):
            if key in diff.data:
                out[f"{band}__data__{key}"] = _npz_safe(diff.data[key])
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
    obslog_dir = Path(
        f"/ut2/muscat/obslog/{data_dir.parent.name.lower()}/{data_dir.name}"
    )
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
            from astropy.io import fits

            hdr = fits.getheader(fp, memmap=True)
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


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target_name", "--target-name", required=True)
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
    ap.add_argument("--bands", nargs="+", default=None)
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
        help="Reference-frame index within a band (default: 0 for "
        "self-reference, middle frame when --ref_band is set).",
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
    ap.add_argument("--glob", default="*.fits", help="FITS glob pattern.")
    ap.add_argument("--gif_stride", "--gif-stride", type=int, default=DEFAULT_GIF_STRIDE)
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
        "--test_run",
        "--test-run",
        dest="test_run",
        action="store_true",
        help=f"Quick smoke test: use only the first {TEST_RUN_FRAMES} frames "
        "of each band.",
    )
    ap.add_argument(
        "--test_run_frames",
        "--test-run-frames",
        type=int,
        default=TEST_RUN_FRAMES,
        dest="test_run_frames",
        help=f"Number of frames per band in test-run mode (default: {TEST_RUN_FRAMES}).",
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
        "--muscat_calib_dir",
        "--muscat-calib-dir",
        type=Path,
        default=None,
        help="Output directory for MuSCAT calibrated FITS files "
        "(default: <data_dir>_calibrated).",
    )
    ap.add_argument(
        "--muscat2_calib_dir",
        "--muscat2-calib-dir",
        type=Path,
        default=None,
        help="Output directory for MuSCAT2 calibrated FITS files "
        "(default: <data_dir>_calibrated).",
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
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    assert args.tID not in (args.cID or []), (
        f"tID={args.tID} must not be in cID={args.cID}"
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

    if args.bands is None:
        args.bands = _detect_narrow_bands(args.data_dir, args.target_name)
        logger.info(f"bands: {args.bands}")
    else:
        logger.info(f"bands: {args.bands} (explicit)")

    sciences = {}
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
            sciences.setdefault(band, []).append(rec["path"])
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

    if args.test_run:
        nrf = args.test_run_frames
        sciences = {b: fs[:nrf] for b, fs in sciences.items()}
        logger.info(f"test-run: limiting to first {nrf} frames per band")
    counts = {b: len(sciences.get(b, [])) for b in args.bands}
    logger.info(f"frames per band: {counts}")
    active_bands = [b for b in args.bands if sciences.get(b)]
    if not active_bands:
        logger.error(f"no frames for target={args.target_name}; aborting")
        return 1

    # header metadata for naming, time conversion
    probe = FITSImage(sciences[active_bands[0]][0]).header
    instrument = get_instrument(probe)
    if instrument == "muscat2" or instrument == "muscat":
        is_muscat = instrument == "muscat"
        calib_label = "muscat" if is_muscat else "muscat2"
        calib_mod = calibrate_muscat if is_muscat else calibrate_muscat2
        calib_arg_name = "muscat_calib_dir" if is_muscat else "muscat2_calib_dir"
        default_bands = ["gp", "rp", "zs"] if is_muscat else ["gp", "rp", "ip", "zs"]

        if args.bands is None:
            args.bands = default_bands
            logger.info(f"{calib_label} bands: {args.bands}")
        calib_dir = getattr(args, calib_arg_name) or args.data_dir.with_name(
            args.data_dir.name + "_calibrated"
        )
        calib_dir.mkdir(parents=True, exist_ok=True)
        calibrated_files = sorted(calib_dir.glob("*_calibrated.fits"))
        need_calib = not calibrated_files

        if not args.test_run and not need_calib:
            raw_count = sum(len(fs) for fs in sciences.values())
            if len(calibrated_files) < raw_count:
                logger.info(
                    f"{calib_label}: found {len(calibrated_files)} calibrated frames "
                    f"but {raw_count} raw frames; re-calibrating"
                )
                need_calib = True

        if calibrated_files:
            h = fits.getheader(calibrated_files[0])
            if "CD1_1" not in h and "CDELT1" not in h:
                need_calib = True
                logger.info(
                    f"{calib_label}: existing calibrated frames lack WCS; re-calibrating"
                )
        if need_calib:
            logger.info(f"{calib_label}: running calibration")
            calib_args = [
                "--data_dir",
                str(args.data_dir),
                "--target",
                args.target_name,
                "--output_dir",
                str(calib_dir),
                "--solve_wcs",
            ]
            if args.test_run:
                calib_args.append("--test_run")
            if args.verbose:
                calib_args.append("--verbose")
            # Forward main logger's FileHandlers to the calibration logger so WCS and calibration details get written to the target's .log file
            calib_mod.logger.setLevel(logging.INFO)
            for h in logger.handlers:
                if isinstance(h, logging.FileHandler):
                    calib_mod.logger.addHandler(h)
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
            sciences = {b: fs[: args.test_run_frames] for b, fs in sciences.items()}
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

    try:
        target_coord = Mast().resolve_object(args.target_name)
    except Exception as e:
        logger.warning(f"MAST resolution failed for {args.target_name}: {e}. Retrying with space-separated name.")
        try:
            target_coord = Mast().resolve_object(args.target_name.replace("-", " "))
        except Exception:
            raise e
    logger.info(f"target radec: {target_coord}")

    # reference seeding: without --ref_band each band self-references its first
    # frame (within-camera alignment, correct for multi-camera instruments);
    # with --ref_band all bands align to that band's middle frame.
    self_reference = args.ref_band is None
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

    band_results = {}
    for band in active_bands:
        band_files = sciences[band]
        if self_reference:
            ref_files, default_refid = band_files, 0
        else:
            ref_files, default_refid = sciences[ref_band], len(sciences[ref_band]) // 2
        refid = args.refid if args.refid is not None else default_refid
        refid = min(refid, len(ref_files) - 1)
        try:
            res = run_band(
                band,
                band_files,
                ref_files[refid],
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
                cids=args.cID,
                min_area=args.min_star_area,
            )
        except Exception as exc:  # noqa: BLE001 - one bad band must not kill the run
            logger.exception(f"[{band}] reduction failed: {exc}")
            continue
        if res is not None:
            band_results[band] = res

    if not band_results:
        logger.error("no bands reduced successfully; aborting")
        return 1

    stem_multi = build_stem(args.target_name, instrument, date)
    bjds = {}
    for band, r in band_results.items():
        stem = build_stem(args.target_name, instrument, date, band)
        bjds[band] = compute_bjd_tdb(
            r["diff"], r["ref"].header, target_coord, args.use_barycorrpy, instrument
        )
        csv_path = args.results_dir / f"{stem}.csv"
        photometry_df(r["diff"], bjds[band]).to_csv(csv_path, index=False)
        logger.info(f"wrote {csv_path}")

        plot_ref_image(
            r, target_coord, instrument, args.results_dir / f"{stem}_ref.png"
        )
        plot_apertures(r, args.results_dir / f"{stem}_apertures.png")
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
        )
        if args.make_gif:
            stride = 1 if args.test_run else args.gif_stride
            make_gif(r["files"], args.results_dir / f"{stem}.gif", stride)

    bin_size_days = args.bin_size_minutes / (24 * 60)
    target_index = next(iter(band_results.values()))["target_index"]
    plot_lightcurves(
        band_results,
        args.results_dir / f"{stem_multi}_lightcurves.png",
        args.target_name,
        instrument,
        date,
        target_index=target_index,
        bin_size_days=bin_size_days,
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
    )
    save_all_bands_npz(band_results, bjds, args.results_dir / f"{stem_multi}.npz")

    elapsed = time_module.time() - t0
    logger.info(f"done  ({elapsed:.0f}s elapsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
