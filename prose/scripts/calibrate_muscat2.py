"""Calibrate raw MuSCAT2 data using darks and flats per band.

Given a raw data directory containing DARK, FLAT, and OBJECT frames
(identified from FITS ``OBJECT`` keyword), this script:

1. Groups calibration and science frames per CCD/band (g, r, i, z_s)
2. Builds master dark and master flat for each band
3. Calibrates all science frames (master-dark subtraction, flat division)
4. Optionally solves WCS astrometry (via twirl + Gaia) per band
5. Writes calibrated FITS files under ``<output_dir>/<band>/``

Example
-------
::

    python -m prose.scripts.calibrate_muscat2 \\
        --data_dir /data/MuSCAT2/250310 \\
        --target TOI00663.02 \\
        --output_dir /data/MuSCAT2/250310_calibrated

    python -m prose.scripts.calibrate_muscat2 \\
        --data_dir /data/MuSCAT2/250310 \\
        --target TOI00663.02 \\
        --output_dir /data/MuSCAT2/250310_calibrated \\
        --solve-wcs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from prose import FITSImage, blocks
from prose.console_utils import info

logger = logging.getLogger("calibrate_muscat2")

CCD_BANDS = {0: "g", 1: "r", 2: "i", 3: "z_s"}


def find_files(data_dir: Path) -> tuple[dict, dict]:
    """Scan FITS headers and return ``(darks, flats)`` per CCD.

    Each returned dict maps CCD index (0-3) to a list of file paths.
    """
    darks: dict[int, list[str]] = {c: [] for c in CCD_BANDS}
    flats: dict[int, list[str]] = {c: [] for c in CCD_BANDS}

    files = sorted(data_dir.glob("MCT2?_*.fits"))
    for fp in files:
        prefix = fp.stem.split("_")[0]
        ccd = int(prefix[4])
        hdr = fits.getheader(fp)
        obj = str(hdr.get("OBJECT", "")).strip()
        if obj == "DARK":
            darks[ccd].append(str(fp))
        elif obj == "FLAT":
            flats[ccd].append(str(fp))

    return darks, flats


def find_science_files(data_dir: Path, target: str) -> dict[int, list[str]]:
    """Scan FITS headers and return science files per CCD for *target*."""
    sciences: dict[int, list[str]] = {c: [] for c in CCD_BANDS}
    files = sorted(data_dir.glob("MCT2?_*.fits"))
    for fp in files:
        prefix = fp.stem.split("_")[0]
        ccd = int(prefix[4])
        hdr = fits.getheader(fp)
        obj = str(hdr.get("OBJECT", "")).strip()
        if obj == target:
            sciences[ccd].append(str(fp))
    return sciences


def _solve_wcs(image) -> object | None:
    """Solve WCS for a calibrated image using twirl + Gaia.

    Returns an ``astropy.wcs.WCS`` or *None* on failure.
    """
    import twirl

    from prose.blocks.catalogs import image_gaia_query
    from prose.blocks.detection import PointSourceDetection
    from twirl.geometry import sparsify

    detection = PointSourceDetection()
    detection.run(image)

    if len(image.sources) < 5:
        logger.warning(f"WCS: too few sources ({len(image.sources)}); skipping")
        return None

    pixel_coords = image.sources.coords.copy()

    radius = image.fov.min() / 12
    stars = (
        sparsify(
            pixel_coords * image.pixel_scale.to("arcmin").value,
            radius.to("arcmin").value,
        )
        / image.pixel_scale.to("arcmin").value
    )

    try:
        table = image_gaia_query(
            image, wcs=False, circular=True, fov=image.fov.max() * 1.2
        ).to_pandas()
    except Exception as e:
        logger.warning(f"WCS: Gaia query failed ({e}); skipping")
        return None

    gaias = np.array([table.ra, table.dec]).T
    gaias = gaias[~np.any(np.isnan(gaias), 1)]

    if len(gaias) < 5:
        logger.warning(f"WCS: too few Gaia stars ({len(gaias)}); skipping")
        return None

    sparse_gaias = sparsify(gaias, radius.to("deg").value)
    sparse_gaias = sparse_gaias[:30]

    try:
        wcs = twirl.compute_wcs(stars, sparse_gaias)
    except Exception as e:
        logger.warning(f"WCS: twirl.compute_wcs failed ({e})")
        return None

    return wcs


def calibrate_band(
    darks: list[str],
    flats: list[str],
    sciences: list[str],
    output_dir: Path,
    band: str,
    solve_wcs: bool = False,
) -> None:
    """Build master dark + flat and calibrate all science frames for one band."""
    band_dir = output_dir / band
    band_dir.mkdir(parents=True, exist_ok=True)

    if not darks or not flats:
        info(
            f"[{band}] missing calibration frames (darks={len(darks)} flats={len(flats)}); skipping"
        )
        return
    if not sciences:
        info(f"[{band}] no science frames; skipping")
        return

    info(
        f"[{band}] {len(darks)} darks, {len(flats)} flats, "
        f"{len(sciences)} science frames"
    )

    info(f"[{band}] building master calibration frames")
    calib = blocks.Calibration(
        darks=darks,
        flats=flats,
        verbose=True,
    )

    wcs = None
    if solve_wcs:
        info(f"[{band}] solving WCS on first science frame")
        first = FITSImage(sciences[0])
        calib.run(first)
        wcs = _solve_wcs(first)
        if wcs is not None:
            info(f"[{band}] WCS solved successfully")
        else:
            info(f"[{band}] WCS solving failed; continuing without astrometry")

    for i, fp in enumerate(tqdm(sciences, desc=f"[{band}] calibrating")):
        if solve_wcs and wcs is not None and i == 0:
            image = first
        else:
            image = FITSImage(fp)
            calib.run(image)

        out_path = band_dir / f"{Path(fp).stem}_calibrated.fits"
        hdu = fits.PrimaryHDU(image.data.astype(np.float32))
        hdu.header = image.header
        hdu.header["CALSTAGE"] = "calibrated"
        if wcs is not None:
            hdu.header.update(wcs.to_header())
        hdu.writeto(out_path, overwrite=True)

    info(f"[{band}] done  ({len(sciences)} frames -> {band_dir})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Raw MuSCAT2 data directory (contains MCT2?_*.fits files)",
    )
    ap.add_argument(
        "--target",
        default=None,
        help="Object name (e.g. TOI00663.02). If omitted, only master calibration "
        "frames are built.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for calibrated FITS files",
    )
    ap.add_argument("--verbose", action="store_true", help="Log to console")
    ap.add_argument(
        "--solve-wcs",
        action="store_true",
        help="Solve WCS astrometry via twirl + Gaia (requires network access). "
        "The WCS is solved once per band and applied to all frames.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO if args.verbose else logging.WARNING)
    logger.addHandler(handler)

    if not args.data_dir.is_dir():
        logger.error(f"data_dir not found: {args.data_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    darks, flats = find_files(args.data_dir)

    if args.target:
        sciences = find_science_files(args.data_dir, args.target)
    else:
        sciences = {c: [] for c in CCD_BANDS}

    for ccd, band in CCD_BANDS.items():
        calibrate_band(
            darks[ccd],
            flats[ccd],
            sciences[ccd],
            args.output_dir,
            band,
            solve_wcs=args.solve_wcs,
        )

    info("calibration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
