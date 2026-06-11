#!/usr/bin/env python
"""End-to-end prose photometry pipeline for LCO MuSCAT3/4 (and Sinistro) data.

This script reproduces the reduction demonstrated in
``notebooks/prose_muscat34_template.ipynb`` as a command-line tool. Given a
directory of *calibrated* (LCO BANZAI-reduced, e.g. ``*-e91.fits``) science
frames for a single target, it:

1. groups frames per photometric band for the requested target,
2. builds a per-band reference image and detects sources,
3. identifies the target via WCS / catalog cross-match,
4. sets aperture radii from the Gaia nearest-neighbour separation,
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
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astroquery.mast import Mast
from tqdm import tqdm

from prose import FITSImage, Fluxes, Sequence, Telescope, blocks
from prose.blocks import catalogs
from prose.core.sequence import SequenceParallel
from prose.utils import (
    LCO_SITES,
    PIXSCALES,
    get_saturation_from_header,
    get_simbad_data,
    read_filename_per_band,
)

# --------------------------- constants / defaults ---------------------------
# ignore if using BANZAI-reduced fits files
CAL_OBJECT_MAP = {"bias": "BIAS", "dark": "DARK", "flat": "FLAT"}

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

DEFAULT_BANDS = ["gp", "rp", "ip", "zs"]
DEFAULT_GIF_STRIDE = 100
TEST_RUN_FRAMES = 10  # frames per band used by --test_run
FPS = 5
GIF_MAX_PX = 512  # max GIF frame dimension [pix]; larger frames are downsampled

# reference-image / detection defaults (mirror the template notebook)
MAX_NUM_STARS = 10  # nth brightest stars to keep
CUTOUT_SIZE = 35  # cutout size of detected stars [pix]
CCD_TRIM_SIZE_YX = (10, 10)  # trim image edges [pix]
MIN_STAR_AREA = 100  # min detected-source area [pix]
MIN_STAR_SEPARATION = 10  # min separation between sources [pix]

# Gaia aperture-radii heuristic
APER_STEP_PIX = 2  # spacing of aperture radii [pix]
ANNULUS_GAP_PIX = 10  # gap between max aperture and inner annulus [pix]
ANNULUS_WIDTH_PIX = 20  # annulus width (rout - rin) [pix]
GAIA_CUTOUT = (200, 200)  # cutout around target for the Gaia query [pix]

# differential-photometry cleaning
SIGMA_BKG = 3
SIGMA_FWHM = 3
BIN_SIZE_DAYS = 10 / 60 / 24  # 10-minute bins for plots

# GJD->BJD sanity bound (light travel time should be well under this)
MAX_TIME_OFFSET_MIN = 2 * 8.4

# band color map
BAND_COLORS = {"g": "blue", "r": "green", "i": "orange", "z": "red"}


def band_color(band: str) -> str:
    if band == "NaD":
        return BAND_COLORS["r"]
    return BAND_COLORS.get(band[0], "k")


logger = logging.getLogger("prose_run_photometry")


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
    """Derive a short instrument label from an LCO header."""
    telid = str(header.get("TELID", "")).lower()
    site = str(header.get("SITEID", "")).lower()
    if telid == "2m0a":
        return "muscat4" if site == "coj" else "muscat3"
    if telid == "1m0a":
        return "sinistro"
    return "unknown"


def date_from_header(header) -> str:
    """Return YYMMDD from the LCO ``DAY-OBS`` keyword (e.g. 20250416 -> 250416)."""
    return str(header["DAY-OBS"]).replace("-", "")[2:]


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
) -> Sequence:
    """Calibration sequence run on the per-band reference frame."""
    return Sequence(
        [
            blocks.Trim(ccd_trim_size_yx),
            blocks.PointSourceDetection(
                n=max_num_stars,
                min_area=MIN_STAR_AREA,
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
    coords = np.array([s.coords for s in ref.sources])
    stars_radec = ref.wcs.pixel_to_world(*coords.T)
    return int(target_coord.match_to_catalog_sky(stars_radec)[0])


def gaia_aperture_radii(ref: FITSImage, target_index: int, target_coord: SkyCoord):
    """Set aperture radii from the Gaia nearest-neighbour separation.

    Returns ``(aper_radii, rin, rout)``. Falls back to fwhm-scaled default
    radii (and an annulus derived from the FWHM) if the Gaia query fails or
    the field is too sparse.
    """
    fwhm = float(ref.fwhm)
    aper_rad_min = max(1.0, round(fwhm))
    try:
        c = ref.cutout(ref.sources[target_index].coords, GAIA_CUTOUT)
        c = catalogs.GaiaCatalog(mode="replace")(c)
        df = c.catalogs["gaia"].copy()
        gaia_coords = SkyCoord(df.ra, df.dec, unit="deg")
        df["sep_arcsec"] = target_coord.separation(gaia_coords).arcsec
        pixscale = float(ref.header.get("PIXSCALE", ref.telescope.pixel_scale))
        df["sep_pix"] = (df["sep_arcsec"] / pixscale).round()
        df = df.sort_values(by="sep_arcsec").reset_index(drop=True)
        aper_rad_max = float(df.iloc[1].sep_pix)  # nearest neighbour (row 0 = target)
        if not np.isfinite(aper_rad_max) or aper_rad_max <= aper_rad_min:
            raise ValueError(
                f"nearest-neighbour radius {aper_rad_max} <= min {aper_rad_min}"
            )
        aper_radii = np.arange(aper_rad_min, aper_rad_max, APER_STEP_PIX)
        rin = aper_rad_max + ANNULUS_GAP_PIX
        rout = rin + ANNULUS_WIDTH_PIX
        logger.info(
            f"Gaia apertures: {len(aper_radii)} radii in "
            f"[{aper_rad_min:.0f}, {aper_rad_max:.0f}] px, annulus ({rin:.0f}, {rout:.0f})"
        )
        return aper_radii, rin, rout
    except Exception as exc:  # noqa: BLE001 - heuristic must degrade gracefully
        logger.warning(f"Gaia aperture heuristic failed ({exc}); using fwhm defaults")
        aper_radii = np.exp(np.linspace(np.log(0.1), np.log(6), 30)) * fwhm
        rin, rout = 8 * fwhm, 12 * fwhm
        if rout > 100:
            rin, rout = 80, 100
            logger.warning(f"clamping sky annulus to ({rin}, {rout})")
        aper_radii = aper_radii[aper_radii < rin]
        if len(aper_radii) == 0:
            aper_radii = np.array([rin - 1])
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
):
    """Build the reference image, target index and aperture geometry.

    PIXSCALE and saturation are read from the reference image header
    (not hardcoded or taken from a probe frame).

    If ``aper_radii`` is provided, the explicit grid (and ``rin``/``rout``) is
    used and the Gaia heuristic is skipped. ``scale`` selects pixel
    (``False``) vs FWHM (``True``) units for the photometry blocks.
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
    ref.telescope = Telescope(saturation=saturation, pixel_scale=pixel_scale)
    logger.info(
        f"{Path(ref_file).name}: PIXSCALE={pixel_scale} "
        f"saturation={saturation} instrument={instrument}"
    )
    reference_sequence(
        ccd_trim_size_yx=ccd_trim_size_yx,
        max_num_stars=max_num_stars,
        min_star_separation=min_star_separation,
        cutout_size=cutout_size,
    ).run(ref, show_progress=False)
    target_index = find_target_index(ref, target_coord)
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
) -> SequenceParallel:
    """Parallel per-image photometry sequence (mirrors the notebook)."""
    if n_stars_align is None:
        n_stars_align = max_num_stars
    return SequenceParallel(
        blocks=[
            blocks.Trim(ccd_trim_size_yx),
            blocks.PointSourceDetection(
                n=max_num_stars,
                min_area=MIN_STAR_AREA,
                min_separation=min_star_separation,
            ),
            blocks.Cutouts(shape=cutout_size),
            blocks.MedianEPSF(),
            blocks.Gaussian2D(ref),
            blocks.ComputeTransformTwirl(ref, n=n_stars_align),
            blocks.AlignReferenceSources(ref),
            blocks.CentroidQuadratic(),
            blocks.AperturePhotometry(aper_radii, scale=scale),
            blocks.AnnulusBackground(rin=rin, rout=rout, scale=scale),
            blocks.Del("data", "cutouts"),
        ],
        data_blocks=[
            blocks.GetFluxes(
                "fwhm",
                airmass=lambda im: im.header["AIRMASS"],
                dx=lambda im: im.transform.translation[0],
                dy=lambda im: im.transform.translation[1],
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
        n_stars_align=n_stars_align,
    )
    phot.run(files)

    fluxes: Fluxes = phot.data[0].fluxes
    fluxes.target = reference["target_index"]

    diff = differential_photometry(fluxes, reference["target_index"])
    if diff is None:
        logger.warning(f"[{band}] no valid frames after cleaning; skipping")
        return None

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


def differential_photometry(fluxes: Fluxes, target_index: int):
    """Clean NaN comparison stars, sigma-clip, and run automatic diff phot."""
    fluxes = fluxes.copy()
    fluxes.target = target_index
    nan_stars = np.any(np.isnan(fluxes.fluxes), axis=(0, 2))
    fluxes = fluxes.mask_stars(~nan_stars)
    fluxes = fluxes.sigma_clipping_data(bkg=SIGMA_BKG, fwhm=SIGMA_FWHM)
    if fluxes.time is None or len(fluxes.time) == 0:
        return None
    return fluxes.autodiff()


# --------------------------- time conversion ---------------------------


def compute_bjd_tdb(diff: Fluxes, header, target_coord: SkyCoord, use_barycorrpy: bool):
    """Convert the GJD-UTC time axis to BJD-TDB."""
    from astroplan import Observer

    site = header["SITE"]
    obs_site = Observer.at_site(LCO_SITES[site])
    t = Time(diff.time, format="jd", scale="utc", location=obs_site.location)

    if use_barycorrpy:
        from barycorrpy import utc_tdb

        result = utc_tdb.JDUTC_to_BJDTDB(
            t.value,
            ra=target_coord.ra.deg,
            dec=target_coord.dec.deg,
            lat=obs_site.location.lat.deg,
            longi=obs_site.location.lon.deg,
            alt=obs_site.location.height.value,
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
    df["Flux_Err"] = diff.error
    df = df.rename(CSV_RENAME, axis=1)
    cols = df.columns.tolist()
    left = ["BJD_TDB", "Flux", "Flux_Err"]
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
    ccd_trim_size_yx: tuple[int, int] = CCD_TRIM_SIZE_YX,
    max_num_stars: int = MAX_NUM_STARS,
    min_star_separation: float = MIN_STAR_SEPARATION,
    n_stars_align: int | None = None,
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
                    min_area=MIN_STAR_AREA,
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
        ax.axis("off")
    fig.suptitle(f"{target_name} | {instrument} | {date}")
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
        ax.plot(t, f, ".", c=c, alpha=0.5)
        bt, bf, be = _binned(t, f, bin_size_days=bin_size_days)
        ax.errorbar(bt, bf, yerr=be, fmt=".", c=c)
        ax.set_ylabel(f"{band}\nDiff. flux")
    axes[-1].set_xlabel(f"time (JD) - {t0}")
    fig.suptitle(f"{target_name} | {instrument} | {date}")
    _savefig(fig, path)


def plot_covariates(
    band_results, path: Path, target_name: str, instrument: str, date: str
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
    fig.suptitle(f"{target_name} | {instrument} | {date}")
    _savefig(fig, path)


def _radial_profile(data, center):
    y, x = np.indices(data.shape)
    rr = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)
    tbin = np.bincount(rr.ravel(), data.ravel())
    nr = np.bincount(rr.ravel())
    return tbin / np.maximum(nr, 1)


def plot_stacks(
    band_results, path: Path, target_name: str, instrument: str, date: str
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
    fig.suptitle(f"{target_name} | {instrument} | {date}")
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
    for fp in tqdm(sampled, desc=f"gif:{path.name}"):
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


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target_name", required=True)
    ap.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="Directory of calibrated FITS frames (globbed recursively).",
    )
    ap.add_argument(
        "--results_dir",
        dest="results_dir",
        required=True,
        type=Path,
        help="Full output directory for all products (CSV/PNG/GIF/NPZ/log).",
    )
    ap.add_argument("--bands", nargs="+", default=DEFAULT_BANDS)
    ap.add_argument(
        "--ref_band",
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
        dest="aper_unit",
        choices=["pix", "fwhm"],
        default="pix",
        help="Unit for --aper_radii/--annulus: 'pix' (default) or 'fwhm' "
        "(radii scaled by each image's FWHM).",
    )
    ap.add_argument("--glob", default="*.fits", help="FITS glob pattern.")
    ap.add_argument("--gif_stride", type=int, default=DEFAULT_GIF_STRIDE)
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
        action="store_true",
        help="Use barycorrpy for BJD-TDB (default: astropy light-travel).",
    )
    ap.add_argument(
        "--test_run",
        dest="test_run",
        action="store_true",
        help=f"Quick smoke test: use only the first {TEST_RUN_FRAMES} frames "
        "of each band.",
    )
    ap.add_argument(
        "--test_run_frames",
        type=int,
        default=TEST_RUN_FRAMES,
        dest="test_run_frames",
        help=f"Number of frames per band in test-run mode (default: {TEST_RUN_FRAMES}).",
    )
    ap.add_argument(
        "--min_star_separation",
        type=float,
        default=MIN_STAR_SEPARATION,
        dest="min_star_separation",
        help="Minimum separation between detected sources in pixels "
        "(default: %(default)s).",
    )
    ap.add_argument(
        "--max_num_stars",
        type=int,
        default=MAX_NUM_STARS,
        dest="max_num_stars",
        help="Number of brightest stars to keep for PSF modeling "
        "(default: %(default)s).",
    )
    ap.add_argument(
        "--n_stars_align",
        type=int,
        default=None,
        dest="n_stars_align",
        help="Number of stars used for image alignment (default: same as "
        "--max_num_stars).",
    )
    ap.add_argument(
        "--cutout_size",
        type=int,
        default=CUTOUT_SIZE,
        dest="cutout_size",
        help="Side length in pixels of star cutouts (default: %(default)s).",
    )
    ap.add_argument(
        "--ccd_trim",
        type=parse_trim,
        default=CCD_TRIM_SIZE_YX,
        dest="ccd_trim_size_yx",
        help="CCD edge trim 'Y,X' in pixels (default: %(default)s).",
    )
    ap.add_argument(
        "--bin_size_minutes",
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

    sciences = {}
    inst_obslog = args.data_dir.parent.name.lower()
    obslog_dir = Path(f"/ut2/muscat/obslog/{inst_obslog}/{args.data_dir.name}")
    if obslog_dir.is_dir():
        logger.info(f"reading obslog: {obslog_dir}")
        for ccd_csv in sorted(obslog_dir.glob("obslog-*-ccd?.csv")):
            with open(ccd_csv) as f:
                for row in csv.DictReader(f):
                    if (
                        row["OBJECT"] != args.target_name
                        or row["FILTER"] not in args.bands
                    ):
                        continue
                    fp = args.data_dir / f"{row['FRAME']}.fits"
                    if fp.is_file():
                        sciences.setdefault(row["FILTER"], []).append(str(fp))
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
        sciences = read_filename_per_band(files, args.bands, args.target_name)

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
    date = date_from_header(probe)
    logger.info(
        f"target={args.target_name} inst={instrument} date={date} "
        f"site={probe.get('SITE')}"
    )

    target_coord = Mast().resolve_object(args.target_name)
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
            r["diff"], r["ref"].header, target_coord, args.use_barycorrpy
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
            ccd_trim_size_yx=args.ccd_trim_size_yx,
            max_num_stars=args.max_num_stars,
            min_star_separation=args.min_star_separation,
            n_stars_align=args.n_stars_align,
        )
        if args.make_gif:
            stride = 1 if args.test_run else args.gif_stride
            make_gif(r["files"], args.results_dir / f"{stem}.gif", stride)

    bin_size_days = args.bin_size_minutes / (24 * 60)
    plot_lightcurves(
        band_results,
        args.results_dir / f"{stem_multi}_lightcurves.png",
        args.target_name,
        instrument,
        date,
        bin_size_days=bin_size_days,
    )
    plot_covariates(
        band_results,
        args.results_dir / f"{stem_multi}_covariates.png",
        args.target_name,
        instrument,
        date,
    )
    plot_stacks(
        band_results,
        args.results_dir / f"{stem_multi}_stacks.png",
        args.target_name,
        instrument,
        date,
    )
    save_all_bands_npz(band_results, bjds, args.results_dir / f"{stem_multi}.npz")

    elapsed = time_module.time() - t0
    logger.info(f"done  ({elapsed:.0f}s elapsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
