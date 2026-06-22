"""Solve WCS for a FITS image via nova.astrometry.net and apply to science frames.

Given one representative FITS file (typically the first calibrated science frame),
this script:

1. Uploads the FITS file to https://nova.astrometry.net for blind plate solving.
2. Polls until the submission succeeds (or times out).
3. Downloads and saves the solved WCS FITS file.
4. Injects the solved WCS header keywords into every science FITS file in the
   specified directory (or a given explicit list), so that downstream tools
   (e.g. ``twirl`` for target and guide-star finding) can use accurate
   astrometry.

The API key is read from the ``ASTROMETRY_NET_API_KEY`` environment variable or
from ``~/.astropy/config/astroquery.cfg`` (the astroquery default location).

Example
-------
::

    # Solve one MuSCAT2 frame and apply its WCS to MCT20 science frames:
    python -m prose.scripts.solve_wcs_astrometry \\
        --fits /mnt_ut3/raid_ut3/data/MuSCAT2/260310/MCT20_2603100239.fits \\
        --out_dir /path/to/science/copies \\
        --instrument muscat2

    # Solve and write the WCS file next to the input without patching any dir:
    python -m prose.scripts.solve_wcs_astrometry \\
        --fits /mnt_ut3/raid_ut3/data/MuSCAT2/260310/MCT20_2603100239.fits

Notes
-----
* The solved WCS is validated against the expected MuSCAT2 pixel scale
  (~0.44 arcsec/pix); images that deviate significantly are rejected.
* All science frames in the output directory that share the same CCD prefix
  (first 5 chars of the filename stem, e.g. ``MCT20``) as the probe frame
  receive the WCS headers.  Pass ``--pattern`` to override the glob.
  ``run_photometry`` then uses that WCS to identify the target in its reference
  source catalog and ``ComputeTransformTwirl`` uses the catalog stars as guides
  to align the remaining frames.
* Uploads to nova.astrometry.net are **public** by default.  Set
  ``--publicly_visible n`` to keep submissions private (requires a paid
  account).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import warnings

from astropy.io import fits
from astropy.utils.decorators import AstropyDeprecationWarning
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from prose import __version__ as PROSE_VERSION

logger = logging.getLogger("solve_wcs_astrometry")

# ---------------------------------------------------------------------------
# Instrument-specific pixel-scale ranges for WCS validation [arcsec/pix]
# ---------------------------------------------------------------------------
PIXEL_SCALE_RANGES: dict[str, tuple[float, float]] = {
    "muscat": (0.31, 0.42),
    "muscat2": (0.38, 0.55),
    "muscat3": (0.22, 0.32),
    "muscat4": (0.22, 0.32),
    "sinistro": (0.25, 0.85),  # full or 2×2 binning
}

# WCS header keywords that carry the astrometric solution
_WCS_KEYS = (
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
    "B_ORDER",
    "B_0_0",
    "B_0_1",
    "B_0_2",
    "AP_ORDER",
    "BP_ORDER",
)

# Observation-epoch keywords that astropy's ``WCS.to_header`` emits as WCS
# "auxiliary" cards but which record *when* the frame was taken, not the
# celestial solution. They must survive WCS replacement: the downstream
# output-file date is derived from these (see ``run_photometry.date_from_header``
# / ``TIME_KEYS``), so deleting them silently collapses the filename date.
_PROTECTED_TIME_KEYS = frozenset(
    {
        "DATE-OBS", "DATE-BEG", "DATE-AVG", "DATE-END", "DATEREF", "DATE",
        "MJD-OBS", "MJD-BEG", "MJD-AVG", "MJD-END", "MJDREF", "MJD-STRT",
        "JD", "JD-STRT", "BJD", "TIMESYS",
    }
)

DEFAULT_TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _api_key() -> str:
    """Return the Astrometry.net API key from env or astroquery config."""
    key = os.environ.get("ASTROMETRY_NET_API_KEY", "").strip()
    if key:
        return key
    try:
        from astroquery.astrometry_net import AstrometryNet as _AN

        an = _AN()
        key = getattr(an, "api_key", "") or ""
        if key:
            return key
    except Exception:
        pass
    raise RuntimeError(
        "No Astrometry.net API key found.  Set the ASTROMETRY_NET_API_KEY "
        "environment variable or configure astroquery:\n"
        "  from astroquery.astrometry_net import AstrometryNet\n"
        "  AstrometryNet.login('your_api_key')"
    )


def _header_hint(fits_path: Path) -> dict:
    """Extract RA/Dec/pixel-scale hints from the FITS header if available."""
    hints: dict = {}
    try:
        hdr = fits.getheader(str(fits_path))
    except Exception:
        return hints

    def _parse_coord(val) -> float | None:
        """Parse sexagesimal (H:M:S or D:M:S) or decimal string to degrees."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        parts = val.replace("+", "").replace("-", " ").split()
        sign = -1 if val.startswith("-") else 1
        try:
            if ":" in val:
                segs = val.lstrip("+-").split(":")
                d = abs(float(segs[0]))
                m = float(segs[1]) if len(segs) > 1 else 0
                s = float(segs[2]) if len(segs) > 2 else 0
                return sign * (d + m / 60 + s / 3600)
            return float(parts[0])
        except (ValueError, IndexError):
            return None

    ra_raw = hdr.get("RA", hdr.get("RA_OBJ", hdr.get("CRVAL1")))
    dec_raw = hdr.get("DEC", hdr.get("DEC_OBJ", hdr.get("CRVAL2")))
    ra = _parse_coord(ra_raw)
    dec = _parse_coord(dec_raw)

    # MuSCAT2 stores RA in hours (H:M:S) → convert to degrees
    if ra is not None and ra <= 24 and ":" in str(ra_raw):
        ra = ra * 15.0

    if ra is not None:
        hints["center_ra"] = float(ra)
    if dec is not None:
        hints["center_dec"] = float(dec)
    if ra is not None and dec is not None:
        hints["radius"] = 0.5  # search radius in degrees

    instrume = str(hdr.get("INSTRUME", "")).lower()
    from prose.utils import PIXSCALES

    pixscale = PIXSCALES.get(instrume)
    if pixscale is not None:
        hints["scale_est"] = pixscale
        # Astrometry.net expresses scale_err as a percentage, not a fraction.
        hints["scale_err"] = 20.0
        hints["scale_units"] = "arcsecperpix"

    return hints


def upload_and_solve(
    fits_path: Path,
    api_key: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    publicly_visible: str = "y",
    allow_commercial_use: str = "n",
) -> WCS | None:
    """Upload *fits_path* to nova.astrometry.net and return the solved WCS.

    Parameters
    ----------
    fits_path:
        Path to the FITS file to solve.
    api_key:
        Your nova.astrometry.net API key.
    timeout:
        Maximum seconds to wait for the solve to complete.
    publicly_visible:
        ``"y"`` (default) or ``"n"`` — visibility of the uploaded image.
    allow_commercial_use:
        ``"n"`` (default) — commercial-use licensing of the upload.

    Returns
    -------
    WCS | None
        The solved :class:`~astropy.wcs.WCS` object, or *None* on failure.
    """
    an = AstrometryNet()
    an.api_key = api_key

    hints = _header_hint(fits_path)
    logger.info(f"Uploading {fits_path.name} to nova.astrometry.net …")
    if hints:
        logger.info(f"  hints: {hints}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyDeprecationWarning)
            wcs_header = an.solve_from_image(
                str(fits_path),
                # photutils is a prose dependency. Without this flag astroquery
                # uploads a locally extracted source list instead of the FITS file.
                force_image_upload=True,
                solve_timeout=int(timeout),
                verbose=False,
                publicly_visible=publicly_visible,
                allow_commercial_use=allow_commercial_use,
                **hints,
            )
    except Exception as exc:
        logger.error(f"Astrometry.net solve failed: {exc}")
        return None

    if not wcs_header:
        logger.warning("Astrometry.net returned no WCS solution.")
        return None

    try:
        wcs = WCS(wcs_header)
    except Exception as exc:
        logger.error(f"Cannot construct WCS from solved header: {exc}")
        return None

    logger.info("Plate solve succeeded.")
    return wcs


def validate_wcs(
    wcs: WCS,
    instrument: str | None = None,
) -> bool:
    """Return *True* if the WCS pixel scale matches the instrument expectation.

    Parameters
    ----------
    wcs:
        The candidate WCS solution.
    instrument:
        Instrument name (lower-case) for pixel-scale lookup.  When ``None``
        or unknown, no pixel-scale check is applied (always returns *True*).
    """
    if not getattr(wcs, "has_celestial", False):
        logger.warning("WCS validation: no celestial axes found.")
        return False

    try:
        import astropy.wcs.utils as wcsutils

        scales = wcsutils.proj_plane_pixel_scales(wcs) * 3600.0  # arcsec/pix
    except Exception as exc:
        logger.warning(f"WCS validation: cannot compute pixel scale ({exc}).")
        return True  # benefit of the doubt

    lo, hi = PIXEL_SCALE_RANGES.get(instrument or "", (0.01, 10.0))
    if not (lo < scales[0] < hi and lo < scales[1] < hi):
        logger.warning(
            f"WCS validation: pixel scales {scales} arcsec/pix outside "
            f"expected ({lo}, {hi}) for instrument '{instrument}'; "
            "rejecting solution."
        )
        return False

    logger.info(
        f"WCS validation: pixel scales {scales[0]:.4f}, "
        f"{scales[1]:.4f} arcsec/pix — OK."
    )
    return True


def save_wcs_fits(wcs: WCS, output_path: Path) -> None:
    """Write the WCS solution as a minimal FITS file with only WCS headers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU()
    hdu.header.update(wcs.to_header(relax=True))
    hdu.writeto(str(output_path), overwrite=True)
    logger.info(f"WCS solution written to {output_path}")


def load_wcs_fits(wcs_path: Path) -> WCS | None:
    """Load a WCS object from a FITS file saved by :func:`save_wcs_fits`."""
    try:
        hdr = fits.getheader(str(wcs_path))
        wcs = WCS(hdr)
        if not getattr(wcs, "has_celestial", False):
            logger.warning(f"WCS loaded from {wcs_path} has no celestial axes.")
            return None
        return wcs
    except Exception as exc:
        logger.error(f"Cannot load WCS from {wcs_path}: {exc}")
        return None


def inject_wcs_into_file(fits_path: Path, wcs: WCS) -> bool:
    """Overwrite *fits_path* in-place, inserting WCS header keywords.

    Returns *True* on success, *False* on error.
    """
    try:
        with fits.open(str(fits_path), mode="update") as hdul:
            hdr = hdul[0].header
            # Remove stale WCS keywords before inserting the solved ones.
            # ``WCS(hdr).to_header`` also reports observation-epoch keys
            # (e.g. DATE-OBS, MJD-OBS) as WCS aux cards; exclude those so the
            # observation date is preserved across WCS replacement.
            stale_keys = set(_WCS_KEYS)
            try:
                stale_keys.update(WCS(hdr).to_header(relax=True).keys())
            except Exception:
                pass
            stale_keys -= _PROTECTED_TIME_KEYS
            for key in stale_keys:
                if key in hdr:
                    del hdr[key]
            hdr.update(wcs.to_header(relax=True))
            hdr["WCSMTHD"] = "astrometry.net"
            hdr["PRSVERS"] = PROSE_VERSION
        return True
    except Exception as exc:
        logger.error(f"Failed to inject WCS into {fits_path}: {exc}")
        return False


def find_science_files(
    directory: Path,
    pattern: str = "*.fits",
) -> list[Path]:
    """Return sorted FITS paths matching *pattern* inside *directory*."""
    return sorted(directory.glob(pattern))


def apply_wcs_to_directory(
    wcs: WCS,
    directory: Path,
    pattern: str = "*.fits",
) -> tuple[int, int]:
    """Inject *wcs* into all FITS files matching *pattern* in *directory*.

    Returns ``(n_ok, n_fail)``.
    """
    files = find_science_files(directory, pattern)
    if not files:
        logger.warning(f"No files matching '{pattern}' in {directory}.")
        return 0, 0

    n_ok = n_fail = 0
    for fp in files:
        if inject_wcs_into_file(fp, wcs):
            n_ok += 1
        else:
            n_fail += 1
    logger.info(
        f"WCS injected into {n_ok}/{n_ok + n_fail} files in {directory} "
        f"(pattern='{pattern}')."
    )
    return n_ok, n_fail


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def solve_and_apply(
    fits_path: Path,
    *,
    api_key: str | None = None,
    out_dir: Path | None = None,
    pattern: str | None = None,
    instrument: str | None = None,
    wcs_output: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    publicly_visible: str = "y",
    allow_commercial_use: str = "n",
    dry_run: bool = False,
) -> WCS | None:
    """Upload *fits_path*, solve WCS, and optionally inject into science frames.

    Parameters
    ----------
    fits_path:
        The probe FITS file to solve (e.g. first calibrated science frame).
    api_key:
        Astrometry.net API key.  Falls back to :func:`_api_key` when *None*.
    out_dir:
        Directory whose FITS files receive the solved WCS.  When *None* no
        injection is performed.
    pattern:
        Glob pattern to filter files in *out_dir*.  Defaults to the CCD
        prefix of *fits_path* (e.g. ``MCT20*.fits``).
    instrument:
        Instrument name used for pixel-scale validation.  Auto-detected from
        the FITS header when *None*.
    wcs_output:
        If provided, the solved WCS is saved as a FITS file here (in addition
        to being injected into science frames).
    timeout:
        Max seconds to wait for astrometry.net.
    publicly_visible:
        Image visibility on nova.astrometry.net (``"y"`` or ``"n"``).
    allow_commercial_use:
        Commercial-use licensing (``"n"`` or ``"y"``).
    dry_run:
        When *True*, only upload/save the solution; do not inject science files.

    Returns
    -------
    WCS | None
        The solved WCS, or *None* if solving failed.
    """
    if api_key is None:
        api_key = _api_key()

    # Auto-detect instrument if not supplied
    if instrument is None:
        try:
            hdr = fits.getheader(str(fits_path))
            instrument = str(hdr.get("INSTRUME", "")).lower() or None
        except Exception:
            pass

    wcs = upload_and_solve(
        fits_path,
        api_key,
        timeout=timeout,
        publicly_visible=publicly_visible,
        allow_commercial_use=allow_commercial_use,
    )

    if wcs is None:
        return None

    if not validate_wcs(wcs, instrument):
        logger.warning("WCS solution rejected after validation; aborting.")
        return None

    if wcs_output is not None:
        save_wcs_fits(wcs, wcs_output)

    if dry_run or out_dir is None:
        if dry_run:
            logger.info("dry_run=True — no files modified.")
        return wcs

    # Determine default glob pattern from the probe file's CCD prefix
    if pattern is None:
        stem = fits_path.stem  # e.g. MCT20_2603100239_calibrated
        prefix = stem.split("_")[0]  # MCT20
        pattern = f"{prefix}*.fits"
        logger.info(f"Auto-detected file pattern: '{pattern}'")

    apply_wcs_to_directory(wcs, out_dir, pattern)
    return wcs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--fits",
        type=Path,
        required=True,
        help="Path to the FITS file to upload and solve (probe frame).",
    )
    ap.add_argument(
        "--api_key",
        "--api-key",
        default=None,
        help=(
            "Astrometry.net API key.  Defaults to $ASTROMETRY_NET_API_KEY or "
            "the astroquery config file."
        ),
    )
    ap.add_argument(
        "--out_dir",
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory of science FITS files to inject the WCS into.  "
            "When omitted, only the solved WCS FITS is written (see --wcs_output)."
        ),
    )
    ap.add_argument(
        "--pattern",
        default=None,
        help=(
            "Glob pattern to select science files in --out_dir "
            "(default: <CCD_prefix>*.fits derived from --fits)."
        ),
    )
    ap.add_argument(
        "--instrument",
        default=None,
        choices=list(PIXEL_SCALE_RANGES),
        help="Instrument name for pixel-scale validation (auto-detected when omitted).",
    )
    ap.add_argument(
        "--wcs_output",
        "--wcs-output",
        type=Path,
        default=None,
        help="Save the solved WCS to this FITS file.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Max seconds to wait for astrometry.net (default: {DEFAULT_TIMEOUT_S}).",
    )
    ap.add_argument(
        "--publicly_visible",
        "--publicly-visible",
        choices=["y", "n"],
        default="y",
        help="Image visibility on nova.astrometry.net (default: y).",
    )
    ap.add_argument(
        "--allow_commercial_use",
        "--allow-commercial-use",
        choices=["y", "n"],
        default="n",
        help="Commercial-use licensing of the upload (default: n).",
    )
    ap.add_argument(
        "--dry_run",
        "--dry-run",
        action="store_true",
        help="Solve and save the WCS, but do not inject science FITS files.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Emit INFO messages to stderr.",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stderr,
    )
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)

    if not args.fits.is_file():
        logger.error(f"FITS file not found: {args.fits}")
        return 1

    api_key = args.api_key
    if api_key is None:
        try:
            api_key = _api_key()
        except RuntimeError as exc:
            logger.error(str(exc))
            return 1

    # Always persist the downloaded solution. This also makes a solve useful
    # without --out-dir and gives downstream/reference-building code a stable
    # artifact that can be reused without another upload.
    wcs_output = args.wcs_output or args.fits.with_name(f"{args.fits.stem}_wcs.fits")

    wcs = solve_and_apply(
        args.fits,
        api_key=api_key,
        out_dir=args.out_dir,
        pattern=args.pattern,
        instrument=args.instrument,
        wcs_output=wcs_output,
        timeout=args.timeout,
        publicly_visible=args.publicly_visible,
        allow_commercial_use=args.allow_commercial_use,
        dry_run=args.dry_run,
    )

    if wcs is None:
        logger.error("WCS solving failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
