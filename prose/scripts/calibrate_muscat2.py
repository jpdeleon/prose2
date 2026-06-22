"""Calibrate raw MuSCAT2 data using darks and flats per band.

Given a raw data directory containing DARK, FLAT, and OBJECT frames
(identified from FITS ``OBJECT`` keyword), this script:

1. Groups calibration and science frames per CCD/band (g, r, i, z_s)
2. Builds master dark and master flat for each band
3. Calibrates all science frames (master-dark subtraction, flat division)
4. Optionally solves WCS astrometry per band (twirl+Gaia or nova.astrometry.net)
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
        --solve-wcs twirl
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from prose import FITSImage, blocks, __version__ as PROSE_VERSION
from prose.console_utils import info
from prose.core.sequence import SequenceParallel
from prose.utils import frames_from_obslog, scan_fits_headers

logger = logging.getLogger("calibrate_muscat2")

CCD_BANDS = {0: "gp", 1: "rp", 2: "ip", 3: "zs"}

# Exposure-time matching for dark selection. ``blocks.Calibration`` rescales the
# master dark by the science exposure (``dark_rate * exp_time``) assuming a
# zero-bias pedestal. With no master bias supplied (the MuSCAT case), that model
# is only exact when the darks share the science frames' exposure, so we select
# exposure-matched darks rather than mixing exposures.
EXPOSURE_KEY = "EXPTIME"
EXPOSURE_RTOL = 1e-3
EXPOSURE_ATOL = 1e-2


class SaveCalibratedFITS(blocks.Block):
    def __init__(self, output_dir, wcs=None, wcs_method=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.wcs = wcs
        self.wcs_method = wcs_method
        self._parallel_friendly = True

    def run(self, image):
        fp = image.metadata.get("path")
        if fp is None:
            return
        out_path = self.output_dir / f"{Path(fp).stem}_calibrated.fits"
        hdu = fits.PrimaryHDU(image.data.astype(np.float32))
        header = image.header.copy()
        if "BZERO" in header:
            del header["BZERO"]
        if "BSCALE" in header:
            del header["BSCALE"]
        # Strip any stale WCS keywords from raw header
        for k in (
            "WCSMTHD",
            "PRSVERS",
            "WCSAXES",
            "CTYPE1",
            "CTYPE2",
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CD1_1",
            "CD1_2",
            "CD2_1",
            "CD2_2",
            "CDELT1",
            "CDELT2",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
            "CUNIT1",
            "CUNIT2",
            "EQUINOX",
            "RADESYS",
            "A_ORDER",
            "A_0_0",
            "A_0_1",
            "A_0_2",
            "A_1_0",
            "A_1_1",
            "A_2_0",
            "B_ORDER",
            "B_0_0",
            "B_0_1",
            "B_0_2",
            "B_1_0",
            "B_1_1",
            "B_2_0",
            "CPDIS1",
            "CPDIS2",
            "DCLOG1",
            "DCLOG2",
            "DCRDR1",
            "DCRDR2",
            "DCRLG1",
            "DCRLG2",
        ):
            header.pop(k, None)
        hdu.header = header
        hdu.header["CALSTAGE"] = "calibrated"
        hdu.header["WCSMTHD"] = self.wcs_method or ("twirl" if self.wcs else "none")
        hdu.header["PRSVERS"] = PROSE_VERSION
        if self.wcs is not None:
            hdu.header.update(self.wcs.to_header())
        hdu.writeto(out_path, overwrite=True)


def find_frames(data_dir: Path, target: str | None = None) -> tuple[dict, dict, dict]:
    """Return ``(darks, flats, sciences)`` mapping CCD index (0-3) to file paths.

    Resolution mirrors ``run_photometry``: the MuSCAT obslog is used when present
    (no FITS reads at all), otherwise a parallel header scan classifies frames by
    ``OBJECT``. ``sciences`` stays empty when *target* is ``None``.
    """
    darks: dict[int, list[str]] = {c: [] for c in CCD_BANDS}
    flats: dict[int, list[str]] = {c: [] for c in CCD_BANDS}
    sciences: dict[int, list[str]] = {c: [] for c in CCD_BANDS}

    def _bucket(ccd, obj: str, path: str) -> None:
        if ccd not in CCD_BANDS:
            return
        if obj == "DARK":
            darks[ccd].append(path)
        elif obj == "FLAT":
            flats[ccd].append(path)
        elif target is not None and obj == target:
            sciences[ccd].append(path)

    records = frames_from_obslog(data_dir)
    if records is not None:
        logger.info(f"obslog: {len(records)} frames (skipping header scan)")
        for rec in records:
            _bucket(rec["ccd"], rec["object"], rec["path"])
        return darks, flats, sciences

    files = sorted(Path(data_dir).glob("MCT2?_*.fits"))
    for fp, header in scan_fits_headers(
        files, keys=("OBJECT",), description="Scanning calibration files"
    ):
        prefix = Path(fp).stem.split("_")[0]
        try:
            ccd = int(prefix[4])
        except (IndexError, ValueError):
            continue
        _bucket(ccd, header.get("OBJECT", ""), fp)
    return darks, flats, sciences


def find_files(data_dir: Path) -> tuple[dict, dict]:
    """Return ``(darks, flats)`` per CCD (thin wrapper over :func:`find_frames`)."""
    darks, flats, _ = find_frames(data_dir)
    return darks, flats


def find_science_files(data_dir: Path, target: str) -> dict[int, list[str]]:
    """Return science files per CCD for *target* (see :func:`find_frames`)."""
    _, _, sciences = find_frames(data_dir, target)
    return sciences


def _solve_wcs(image) -> object | None:
    """Solve WCS for a calibrated image using twirl + Gaia.

    Returns an ``astropy.wcs.WCS`` or *None* on failure.
    """
    import twirl
    import astropy.wcs.utils as wcsutils
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time
    from prose.core.image import str_to_astropy_unit

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

    # Precess apparent coordinates to J2000 for Gaia query
    original_ra = image.metadata.get("ra")
    original_dec = image.metadata.get("dec")
    precessed = False
    try:
        jd_val = image.jd
        obstime = Time(jd_val, format="mjd" if jd_val < 2400000 else "jd")
        c_apparent = SkyCoord(image.ra, image.dec, frame=FK5(equinox=obstime))
        c_j2000 = c_apparent.icrs
        ra_unit = str_to_astropy_unit(image.metadata["ra_unit"])
        dec_unit = str_to_astropy_unit(image.metadata["dec_unit"])
        image.metadata["ra"] = c_j2000.ra.to(ra_unit).value
        image.metadata["dec"] = c_j2000.dec.to(dec_unit).value
        precessed = True
    except Exception as e:
        logger.warning(f"WCS: precession of apparent coordinates failed ({e})")

    try:
        table = image_gaia_query(
            image, wcs=False, circular=True, fov=image.fov.max() * 1.2
        ).to_pandas()
    except Exception as e:
        logger.warning(f"WCS: Gaia query failed ({e}); skipping")
        return None
    finally:
        # Restore original header coordinates in metadata if we precessed them
        if precessed:
            image.metadata["ra"] = original_ra
            image.metadata["dec"] = original_dec

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
        if wcs is not None:
            scales = wcsutils.proj_plane_pixel_scales(wcs) * 3600.0
            if not (0.38 < scales[0] < 0.55 and 0.38 < scales[1] < 0.55):
                logger.warning(
                    f"WCS: solved pixel scales {scales} deviate significantly"
                    " from expected ~0.44 arcsec/pixel; rejecting WCS solution"
                )
                wcs = None
    except Exception as e:
        logger.warning(f"WCS: twirl.compute_wcs failed ({e})")
        wcs = None

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


def read_exposures(paths: list[str]) -> dict[str, float | None]:
    """Map each FITS path to its exposure (float), or ``None`` if unresolved.

    Mirrors :func:`find_frames`: the obslog ``EXPTIME (s)`` column is preferred
    (no FITS reads), and a parallel header scan resolves only the frames the
    obslog does not cover.
    """
    exposures: dict[str, float | None] = {}
    if not paths:
        return exposures

    records = frames_from_obslog(Path(paths[0]).parent)
    obslog = (
        {rec["path"]: rec.get("exposure") for rec in records}
        if records is not None
        else {}
    )

    missing = [fp for fp in paths if obslog.get(fp) is None]
    for fp in paths:
        exposures[fp] = obslog.get(fp)

    if missing:
        for fp, header in scan_fits_headers(
            missing, keys=(EXPOSURE_KEY,), description="Reading exposure times"
        ):
            raw = header.get(EXPOSURE_KEY, "")
            try:
                exposures[fp] = float(raw)
            except (TypeError, ValueError):
                exposures[fp] = None
    return exposures


def group_by_exposure(
    paths: list[str], exposures: dict[str, float | None]
) -> dict[float | None, list[str]]:
    """Group *paths* by rounded exposure; unknown exposures bucket under ``None``."""
    groups: dict[float | None, list[str]] = defaultdict(list)
    for fp in paths:
        exp = exposures.get(fp)
        groups[None if exp is None else round(exp, 3)].append(fp)
    return groups


def select_darks_for_exposure(
    darks: list[str], science_exposure: float | None, band: str = "?"
) -> tuple[list[str], str]:
    """Return the darks whose exposure matches *science_exposure*.

    Falls back to all darks (with a warning) when the science exposure is unknown
    or no dark matches it, because ``blocks.Calibration`` rescales the master dark
    by exposure assuming a zero-bias pedestal — only exposure-matched darks are
    guaranteed correct when no master bias is supplied. Returns ``(darks, status)``
    where status is one of ``matched``, ``no-darks``, ``science-exposure-unknown``
    or ``no-match``.
    """
    if not darks:
        return darks, "no-darks"

    exposures = read_exposures(darks)

    if science_exposure is None:
        logger.warning(
            f"[{band}] science exposure unknown; using all {len(darks)} darks "
            "without exposure matching"
        )
        return darks, "science-exposure-unknown"

    matched = [
        fp
        for fp in darks
        if exposures.get(fp) is not None
        and np.isclose(
            exposures[fp], science_exposure, rtol=EXPOSURE_RTOL, atol=EXPOSURE_ATOL
        )
    ]
    if matched:
        if len(matched) < len(darks):
            others = sorted(
                {
                    round(e, 3)
                    for e in exposures.values()
                    if e is not None
                    and not np.isclose(
                        e, science_exposure, rtol=EXPOSURE_RTOL, atol=EXPOSURE_ATOL
                    )
                }
            )
            info(
                f"[{band}] {len(matched)}/{len(darks)} darks match science exposure "
                f"{science_exposure:g}s (ignoring darks at {others}s)"
            )
        return matched, "matched"

    available = sorted({e for e in exposures.values() if e is not None})
    logger.warning(
        f"[{band}] no darks match science exposure {science_exposure:g}s "
        f"(dark exposures present: {available}s). Master dark will be "
        "exposure-rescaled assuming a zero-bias pedestal; supply exposure-matched "
        "darks or a master bias to avoid a scaling error."
    )
    return darks, "no-match"


def _wcs_sidecar_path(output_dir: Path, band: str, method: str) -> Path:
    p = output_dir / ".wcs" / f"{band}_{method}.wcs.fits"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def calibrate_band(
    darks: list[str],
    flats: list[str],
    sciences: list[str],
    output_dir: Path,
    band: str,
    solve_wcs: str | bool | None = None,
    test_run: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build master dark + flat and calibrate all science frames for one band.

    Parameters
    ----------
    solve_wcs:
        ``None`` = no WCS solving, ``"twirl"`` = twirl+Gaia, ``"nova"`` = astrometry.net.

    Returns ``(master_dark, master_flat)`` arrays or ``(None, None)`` if skipped.
    """
    if solve_wcs is True:  # backward compat: old bool True -> nova
        solve_wcs = "nova"
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

    # Select darks matching the science exposure so the master dark is not built
    # from a mix of exposures (see ``select_darks_for_exposure``).
    sci_exposures = read_exposures(sciences[:1])
    science_exposure = next(iter(sci_exposures.values()), None)
    darks, _ = select_darks_for_exposure(darks, science_exposure, band)

    info(f"[{band}] building master calibration frames")
    calib = blocks.Calibration(
        darks=darks,
        flats=flats,
        shared=True,
        verbose=True,
    )

    if solve_wcs == "twirl":
        info(f"[{band}] solving WCS on first science frame via twirl+Gaia")
        first = FITSImage(sciences[0])
        calib.run(first)
        wcs = _solve_wcs(first)
        if wcs is not None:
            info(f"[{band}] WCS solved successfully via twirl")
            sp = _wcs_sidecar_path(output_dir, band, "twirl")
            hdu = fits.PrimaryHDU()
            hdu.header.update(wcs.to_header(relax=True))
            hdu.writeto(str(sp), overwrite=True)
        else:
            info(
                f"[{band}] WCS solving failed via twirl; continuing without astrometry"
            )
        seq = SequenceParallel(
            [calib, SaveCalibratedFITS(output_dir, wcs=wcs, wcs_method="twirl")],
            name=f"[{band}] calibrating",
        )
        seq.run(sciences)

    elif solve_wcs == "nova":
        seq = SequenceParallel(
            [calib, SaveCalibratedFITS(output_dir, wcs_method="nova")],
            name=f"[{band}] calibrating",
        )
        seq.run(sciences)
        info(f"[{band}] solving WCS on first calibrated frame via astrometry.net")
        from prose.scripts.solve_wcs_astrometry import (
            _api_key,
            inject_wcs_into_file,
            upload_and_solve,
            validate_wcs,
        )

        try:
            api_key = _api_key()
        except RuntimeError as e:
            logger.warning(f"[{band}] {e}; skipping WCS")
        else:
            calibrated_files = sorted(output_dir.glob("*_calibrated.fits"))
            if calibrated_files:
                wcs = upload_and_solve(calibrated_files[0], api_key)
                if wcs is not None and validate_wcs(wcs, "muscat2"):
                    for fp in calibrated_files:
                        inject_wcs_into_file(fp, wcs)
                    sp = _wcs_sidecar_path(output_dir, band, "nova")
                    hdu = fits.PrimaryHDU()
                    hdu.header.update(wcs.to_header(relax=True))
                    hdu.writeto(str(sp), overwrite=True)
                    info(
                        f"[{band}] WCS solved via astrometry.net and "
                        f"injected into {len(calibrated_files)} files"
                    )
                else:
                    info(
                        f"[{band}] WCS solving failed via astrometry.net; continuing without astrometry"
                    )
            else:
                logger.warning(f"[{band}] no calibrated files found; skipping WCS")

    else:
        seq = SequenceParallel(
            [calib, SaveCalibratedFITS(output_dir)],
            name=f"[{band}] calibrating",
        )
        seq.run(sciences)

    info(f"[{band}] done  ({len(sciences)} frames -> {output_dir})")

    md = (
        calib.master_dark
        if hasattr(calib, "master_dark")
        else np.array(
            np.memmap(
                calib._cal_paths["dark"],
                dtype="float32",
                mode="r",
                shape=calib.shapes["dark"],
            )
        )
    )
    mf = (
        calib.master_flat
        if hasattr(calib, "master_flat")
        else np.array(
            np.memmap(
                calib._cal_paths["flat"],
                dtype="float32",
                mode="r",
                shape=calib.shapes["flat"],
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
        "--bands",
        nargs="+",
        choices=sorted(set(CCD_BANDS.values())),
        default=None,
        help="Bands to calibrate. Defaults to all MuSCAT2 bands.",
    )
    ap.add_argument(
        "--solve_wcs",
        "--solve-wcs",
        nargs="?",
        const="twirl",
        choices=["twirl", "nova"],
        default=None,
        help="Solve WCS astrometry: 'twirl' (twirl+Gaia, default when flag is given "
        "without a value) or 'nova' (nova.astrometry.net, requires API key in "
        "$ASTROMETRY_NET_API_KEY). The WCS is solved once per band and applied "
        "to all calibrated frames. Omit the flag entirely to skip WCS solving.",
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

    darks, flats, sciences = find_frames(args.data_dir, args.target)

    master_darks: list[tuple[str, np.ndarray]] = []
    master_flats: list[tuple[str, np.ndarray]] = []
    requested_bands = set(args.bands) if args.bands else None
    for ccd, band in CCD_BANDS.items():
        if requested_bands is not None and band not in requested_bands:
            continue
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
