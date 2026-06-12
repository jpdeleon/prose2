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

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from prose import FITSImage, blocks
from prose.console_utils import info
from prose.core.sequence import SequenceParallel

logger = logging.getLogger("calibrate_muscat2")

CCD_BANDS = {0: "gp", 1: "rp", 2: "ip", 3: "zs"}


class SaveCalibratedFITS(blocks.Block):
    def __init__(self, output_dir, wcs=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.wcs = wcs
        self._parallel_friendly = True

    def run(self, image):
        fp = image.metadata.get("path")
        if fp is None:
            return
        out_path = self.output_dir / f"{Path(fp).stem}_calibrated.fits"
        hdu = fits.PrimaryHDU(image.data.astype(np.float32))
        hdu.header = image.header
        hdu.header["CALSTAGE"] = "calibrated"
        if self.wcs is not None:
            hdu.header.update(self.wcs.to_header())
        hdu.writeto(out_path, overwrite=True)


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
        if ccd not in CCD_BANDS:
            continue
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
        if ccd not in CCD_BANDS:
            continue
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

    try:
        detection = PointSourceDetection()
        detection.run(image)
    except Exception as e:
        logger.warning(f"WCS: source detection failed ({e})")
        return None

    if len(image.sources) < 5:
        logger.warning(f"WCS: too few sources ({len(image.sources)}); skipping")
        return None

    pixel_coords = image.sources.coords.copy()

    try:
        radius = image.fov.min() / 12
    except Exception as e:
        logger.warning(f"WCS: cannot compute FOV ({e}); skipping")
        return None

    stars: np.ndarray | None = None
    try:
        stars = (
            sparsify(
                pixel_coords * image.pixel_scale.to("arcmin").value,
                radius.to("arcmin").value,
            )
            / image.pixel_scale.to("arcmin").value
        )
    except Exception as e:
        logger.warning(f"WCS: sparsify failed ({e})")
        return None

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

    try:
        sparse_gaias = sparsify(gaias, radius.to("deg").value)
    except Exception as e:
        logger.warning(f"WCS: sparsify failed for Gaia stars ({e})")
        return None
    sparse_gaias = sparse_gaias[:30]

    try:
        wcs = twirl.compute_wcs(stars, sparse_gaias)
    except Exception as e:
        logger.warning(f"WCS: twirl.compute_wcs failed ({e})")
        return None

    return wcs


def save_master_plots(
    masters: dict[str, list[tuple[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    for master_type, items in masters.items():
        n = len(items)
        if n == 0:
            continue
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), squeeze=False)
        for ax, (band, data) in zip(axes[0], items):
            vmin, vmax = np.nanpercentile(data, [1, 99])
            im = ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(f"Master {master_type} — {band}")
            ax.set_xlabel("x (pix)")
            ax.set_ylabel("y (pix)")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        fig.savefig(
            output_dir / f"master_{master_type}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)


def calibrate_band(
    darks: list[str],
    flats: list[str],
    sciences: list[str],
    output_dir: Path,
    band: str,
    solve_wcs: bool = False,
    test_run: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build master dark + flat and calibrate all science frames for one band.

    Returns ``(master_dark, master_flat)`` arrays or ``(None, None)`` if skipped.
    """
    if test_run:
        sciences = sciences[:10]

    if not darks or not flats:
        info(
            f"[{band}] missing calibration frames (darks={len(darks)} flats={len(flats)}); skipping"
        )
        return None, None
    if not sciences:
        info(f"[{band}] no science frames; skipping")
        return None, None

    info(
        f"[{band}] {len(darks)} darks, {len(flats)} flats, "
        f"{len(sciences)} science frames"
    )

    info(f"[{band}] building master calibration frames")
    calib = blocks.Calibration(
        darks=darks,
        flats=flats,
        shared=True,
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

    seq = SequenceParallel(
        [
            calib,
            SaveCalibratedFITS(output_dir, wcs=wcs),
        ],
        name=f"[{band}] calibrating",
    )

    seq.run(sciences)

    info(f"[{band}] done  ({len(sciences)} frames -> {output_dir})")

    md = (
        calib.master_dark
        if hasattr(calib, "master_dark")
        else np.array(
            np.memmap(
                "__dark.array", dtype="float32", mode="r", shape=calib.shapes["dark"]
            )
        )
    )
    mf = (
        calib.master_flat
        if hasattr(calib, "master_flat")
        else np.array(
            np.memmap(
                "__flat.array", dtype="float32", mode="r", shape=calib.shapes["flat"]
            )
        )
    )

    return md, mf


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data_dir",
        "--data-dir",
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
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for calibrated FITS files",
    )
    ap.add_argument(
        "--test_run",
        "--test-run",
        action="store_true",
        help="Run on first 10 frames (darks, flats, sciences) per band for quick validation",
    )
    ap.add_argument("--verbose", action="store_true", help="Log to console")
    ap.add_argument(
        "--solve_wcs",
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

    assert any(args.data_dir.glob("MCT2[0-3]_*.fits")), (
        f"data_dir {args.data_dir} contains no MuSCAT2 files matching MCT2[0-3]_*.fits. "
        "This script only supports MuSCAT2 data."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    darks, flats = find_files(args.data_dir)

    if args.target:
        sciences = find_science_files(args.data_dir, args.target)
    else:
        sciences = {c: [] for c in CCD_BANDS}

    master_darks: list[tuple[str, np.ndarray]] = []
    master_flats: list[tuple[str, np.ndarray]] = []
    for ccd, band in CCD_BANDS.items():
        md, mf = calibrate_band(
            darks[ccd],
            flats[ccd],
            sciences[ccd],
            args.output_dir,
            band,
            solve_wcs=args.solve_wcs,
            test_run=args.test_run,
        )
        if md is not None and mf is not None:
            master_darks.append((band, md))
            master_flats.append((band, mf))

    save_master_plots({"dark": master_darks, "flat": master_flats}, args.output_dir)
    info("calibration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
