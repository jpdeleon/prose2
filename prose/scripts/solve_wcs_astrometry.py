"""Solve WCS for a FITS image locally or via nova.astrometry.net and apply to science frames.

Given one representative FITS file (typically the first calibrated science frame),
this script:

1. Uploads the FITS file to https://nova.astrometry.net for blind plate solving.
2. Polls until the submission succeeds (or times out).
3. Downloads and saves the solved WCS FITS file.
4. Injects the solved WCS header keywords into every science FITS file in the
   specified directory (or a given explicit list), so that downstream tools
   (e.g. ``twirl`` for target and guide-star finding) can use accurate
   astrometry.

With ``--backend local``, the installed ``solve-field`` executable and local
index files replace steps 1-3; no API key or upload is used. Local standalone
solves write a WCS sidecar only, leaving input images unchanged. Use
``--astrometry-config`` to select index paths.

The nova API key is read from the ``ASTROMETRY_NET_API_KEY`` environment variable or
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
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

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
        "DATE-OBS",
        "DATE-BEG",
        "DATE-AVG",
        "DATE-END",
        "DATEREF",
        "DATE",
        "MJD-OBS",
        "MJD-BEG",
        "MJD-AVG",
        "MJD-END",
        "MJDREF",
        "MJD-STRT",
        "JD",
        "JD-STRT",
        "BJD",
        "TIMESYS",
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
        sign = -1 if val.startswith("-") else 1
        try:
            if ":" in val:
                segs = val.lstrip("+-").split(":")
                d = abs(float(segs[0]))
                m = float(segs[1]) if len(segs) > 1 else 0
                s = float(segs[2]) if len(segs) > 2 else 0
                return sign * (d + m / 60 + s / 3600)
            return float(val)
        except (ValueError, IndexError):
            return None

    ra_raw = hdr.get("RA", hdr.get("RA_OBJ", hdr.get("CRVAL1")))
    dec_raw = hdr.get("DEC", hdr.get("DEC_OBJ", hdr.get("CRVAL2")))
    ra = _parse_coord(ra_raw)
    dec = _parse_coord(dec_raw)

    # MuSCAT2 stores RA in hours (H:M:S) → convert to degrees
    if ra is not None and ra <= 24 and ":" in str(ra_raw):
        ra = ra * 15.0

    if ra is not None and math.isfinite(ra) and 0 <= ra < 360:
        hints["center_ra"] = float(ra)
    if dec is not None and math.isfinite(dec) and -90 <= dec <= 90:
        hints["center_dec"] = float(dec)
    if "center_ra" in hints and "center_dec" in hints:
        hints["radius"] = 0.5  # search radius in degrees

    instrume = str(hdr.get("INSTRUME", "")).lower()
    from prose.utils import PIXSCALES

    # Prefer an explicit on-sky header scale (already accounts for binning).
    pixscale = None
    for key in ("PIXSCALE", "SECPIX", "CCDSCALE"):
        try:
            value = float(hdr[key])
            if math.isfinite(value) and value > 0:
                pixscale = value
                break
        except (KeyError, TypeError, ValueError):
            continue
    if pixscale is None:
        pixscale = PIXSCALES.get(instrume)
    if pixscale is not None:
        hints["scale_est"] = pixscale
        # Astrometry.net expresses scale_err as a percentage, not a fraction.
        hints["scale_err"] = 20.0
        hints["scale_units"] = "arcsecperpix"

    return hints


NOVA_STATUS_URL = "https://nova.astrometry.net/status/{}"

# Frames tried per band before giving up: first, middle and last.
MAX_SOLVE_FRAMES = 3


def _submission_url(submission_id) -> str:
    if submission_id is None:
        return "submission id unknown"
    return f"submission {submission_id}: {NOVA_STATUS_URL.format(submission_id)}"


def candidate_frames(files: Sequence[Path], n: int = MAX_SOLVE_FRAMES) -> list[Path]:
    """Frames to try for a plate solve: first, middle, last (at most *n*),
    without duplicates, in that order. A bad first frame (focus, clouds,
    tracking) then no longer fails the whole band."""
    if not files:
        return []
    indices = [0, len(files) // 2, len(files) - 1]
    picked: list[int] = []
    for i in indices:
        if i not in picked:
            picked.append(i)
    return [files[i] for i in picked[:n]]


def solve_first_valid(
    files: Sequence[Path],
    api_key: str | None,
    instrument: str | None,
    *,
    backend: str = "nova",
    **local_options,
) -> WCS | None:
    """Plate-solve the first of :func:`candidate_frames` that yields a WCS
    passing :func:`validate_wcs`; ``None`` if every attempt fails.

    The returned WCS comes from whichever frame solved, which may not be the
    first; callers inject it into every frame, as they already did for the
    first-frame solution, so a small pointing drift between frames carries
    through the same way.
    """
    if backend not in ("nova", "local"):
        raise ValueError(f"Unknown astrometry backend: {backend}")
    for fp in candidate_frames(files):
        wcs = (
            solve_local(fp, **local_options)
            if backend == "local"
            else upload_and_solve(fp, api_key)
        )
        valid = (
            _validate_local_wcs(wcs, fp, instrument)
            if backend == "local"
            else wcs is not None and validate_wcs(wcs, instrument)
        )
        if valid:
            return wcs
        logger.warning(f"Plate solve did not succeed on {fp.name}; trying next frame.")
    return None


def _positive_timeout(value) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return value


def add_local_solver_arguments(parser, *, include_timeout=True) -> None:
    """Shared options for standalone solving, calibration and photometry."""
    parser.add_argument(
        "--solve-field",
        default="solve-field",
        help="Local solve-field executable (name or path).",
    )
    parser.add_argument(
        "--astrometry-config",
        type=Path,
        default=None,
        help="Local astrometry.net backend config with index paths.",
    )
    parser.add_argument(
        "--astrometry-temp-dir",
        type=Path,
        default=None,
        help="Local solver workspace parent (default: ~/temp).",
    )
    if include_timeout:
        parser.add_argument(
            "--astrometry-timeout",
            type=_positive_timeout,
            default=DEFAULT_TIMEOUT_S,
            help="Seconds per local attempt, including lock wait (default: %(default)s).",
        )


def local_solver_options(args) -> dict:
    """Map the common CLI namespace to local solver keyword arguments."""
    return dict(
        executable=args.solve_field,
        config=args.astrometry_config,
        temp_dir=args.astrometry_temp_dir,
        timeout=args.astrometry_timeout,
    )


def _validate_local_wcs(wcs, fits_path, instrument) -> bool:
    if wcs is None or not wcs.has_celestial:
        return False
    scale = _header_hint(fits_path).get("scale_est")
    if scale is None:
        return validate_wcs(wcs, instrument)
    from astropy.wcs.utils import proj_plane_pixel_scales

    try:
        scales = proj_plane_pixel_scales(wcs) * 3600
        valid = all(
            math.isfinite(s) and 0.8 * scale <= s <= 1.2 * scale for s in scales
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        logger.warning("Local WCS rejected: pixel scale inconsistent with input hints.")
    return valid


def solve_local(
    fits_path: Path,
    *,
    executable: str = "solve-field",
    config: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    temp_dir: Path | None = None,
) -> WCS | None:
    """Run the POSIX astrometry.net solver offline, without editing the input.

    One local solve runs at a time per user/host. The wall deadline includes
    waiting for that lock and kills the solver process group on expiry. All
    intermediate products are isolated below *temp_dir* (default ~/temp).
    PROSE_ASTROMETRY_LOCK can select a shared lock path for service accounts.
    """
    timeout = _positive_timeout(timeout)
    deadline = time.monotonic() + timeout
    if os.name != "posix":
        logger.error("Local astrometry.net solving requires a POSIX host.")
        return None
    import fcntl

    binary = shutil.which(str(executable))
    if binary is None:
        logger.error("Local solve-field executable not found: %s", executable)
        return None
    fits_path = Path(fits_path).expanduser().resolve()
    if not fits_path.is_file():
        logger.error("Local solver input not found: %s", fits_path)
        return None
    if config is not None:
        config = Path(config).expanduser().resolve()
        if not config.is_file():
            logger.error("Local astrometry config not found: %s", config)
            return None
    parent = Path(temp_dir or Path.home() / "temp").expanduser().resolve()
    lock_path = Path(
        os.environ.get(
            "PROSE_ASTROMETRY_LOCK", str(Path.home() / "temp/prose-astrometry.lock")
        )
    ).expanduser()
    try:
        parent.mkdir(parents=True, exist_ok=True)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.error(
                            "Local astrometry timeout waiting for solver lock."
                        )
                        return None
                    time.sleep(min(0.1, remaining))
            with tempfile.TemporaryDirectory(
                prefix="prose-astrometry-", dir=parent
            ) as directory:
                work = Path(directory)
                wcs_file, solved_file = work / "solution.wcs", work / "solution.solved"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error("Local astrometry timeout before solver launch.")
                    return None
                command = [
                    str(Path(binary).resolve()),
                    "--no-plots",
                    "--new-fits",
                    "none",
                    "--dir",
                    str(work),
                    "--temp-dir",
                    str(work),
                    "--wcs",
                    str(wcs_file),
                    "--solved",
                    str(solved_file),
                    "--cpulimit",
                    str(max(1, math.ceil(remaining))),
                ]
                if config is not None:
                    command.extend(["--backend-config", str(config)])
                hints = _header_hint(fits_path)
                if "radius" in hints:
                    for key, option in (
                        ("center_ra", "--ra"),
                        ("center_dec", "--dec"),
                        ("radius", "--radius"),
                    ):
                        command.extend([option, str(hints[key])])
                if "scale_est" in hints:
                    scale = hints["scale_est"]
                    command.extend(
                        [
                            "--scale-units",
                            "arcsecperpix",
                            "--scale-low",
                            str(scale * 0.8),
                            "--scale-high",
                            str(scale * 1.2),
                        ]
                    )
                command.append(str(fits_path))
                logger.info(
                    "Solving %s locally (timeout %.1f s)", fits_path.name, remaining
                )
                env = dict(os.environ, TMPDIR=str(work))
                with (work / "solver.log").open(
                    "w+", encoding="utf-8", errors="replace"
                ) as log:
                    proc = None
                    timed_out = False
                    previous_handler = None

                    def interrupted(signum, frame):
                        raise SystemExit(128 + signum)

                    if threading.current_thread() is threading.main_thread():
                        previous_handler = signal.signal(signal.SIGTERM, interrupted)
                    try:
                        proc = subprocess.Popen(
                            command,
                            cwd=work,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        try:
                            proc.wait(timeout=max(0, deadline - time.monotonic()))
                        except subprocess.TimeoutExpired:
                            timed_out = True
                            logger.error(
                                "Local astrometry timeout after %.1f s", timeout
                            )
                    finally:
                        if proc is not None:
                            # Also stop surviving engine/source-extractor descendants.
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            proc.wait()
                        if previous_handler is not None:
                            signal.signal(signal.SIGTERM, previous_handler)
                    log.seek(0, os.SEEK_END)
                    log.seek(max(0, log.tell() - 8192))
                    tail = log.read()
                    if (
                        timed_out
                        or proc.returncode
                        or not solved_file.is_file()
                        or solved_file.read_bytes() != b"\x01"
                    ):
                        logger.warning(
                            "Local plate solve failed (exit %s): %s",
                            proc.returncode,
                            tail,
                        )
                        return None
                    logger.debug("Local solve-field output: %s", tail)
                return load_wcs_fits(wcs_file)
    except OSError as exc:
        logger.error("Cannot run local astrometry.net solver: %s", exc)
        return None


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
            wcs_header, submission_id = an.solve_from_image(
                str(fits_path),
                # photutils is a prose dependency. Without this flag astroquery
                # uploads a locally extracted source list instead of the FITS file.
                force_image_upload=True,
                solve_timeout=int(timeout),
                verbose=False,
                publicly_visible=publicly_visible,
                allow_commercial_use=allow_commercial_use,
                # Keep the submission id so a failure can be looked up on nova.
                return_submission_id=True,
                **hints,
            )
    except TimeoutError as exc:
        # astroquery raises TimeoutError(message, submission_id).
        sub_id = exc.args[1] if len(exc.args) > 1 else None
        logger.error(
            f"Astrometry.net solve timed out after {int(timeout)} s "
            f"({_submission_url(sub_id)})"
        )
        return None
    except Exception as exc:
        logger.error(f"Astrometry.net solve failed: {exc}")
        return None

    if not wcs_header:
        logger.warning(
            "Astrometry.net solver reported failure; no WCS solution "
            f"({_submission_url(submission_id)})."
        )
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


def inject_wcs_into_file(
    fits_path: Path, wcs: WCS, method: str = "astrometry.net"
) -> bool:
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
            hdr["WCSMTHD"] = method
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
    backend: str = "nova",
    executable: str = "solve-field",
    config: Path | None = None,
    temp_dir: Path | None = None,
) -> WCS | None:
    """Solve WCS locally or on nova; optionally inject nova solutions into copies.

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
    backend:
        ``"nova"`` (default) uploads; ``"local"`` runs offline and saves only
        a sidecar. Local *out_dir* requires *dry_run* to prevent input edits.
    executable, config, temp_dir:
        Local solve-field executable, backend configuration and workspace parent.
    dry_run:
        When *True*, only upload/save the solution; do not inject science files.

    Returns
    -------
    WCS | None
        The solved WCS, or *None* if solving failed.
    """
    if backend not in ("nova", "local"):
        raise ValueError(f"Unknown astrometry backend: {backend}")
    if backend == "local" and out_dir is not None and not dry_run:
        raise ValueError(
            "Local solving writes sidecars only; omit out_dir or use dry_run"
        )
    if (
        backend == "local"
        and wcs_output is not None
        and Path(wcs_output).resolve() == Path(fits_path).resolve()
    ):
        raise ValueError("Local WCS output must differ from the input FITS file")
    if backend == "nova" and api_key is None:
        api_key = _api_key()

    # Auto-detect instrument if not supplied
    if instrument is None:
        try:
            hdr = fits.getheader(str(fits_path))
            instrument = str(hdr.get("INSTRUME", "")).lower() or None
        except Exception:
            pass

    if backend == "local":
        wcs = solve_local(
            fits_path,
            executable=executable,
            config=config,
            timeout=timeout,
            temp_dir=temp_dir,
        )
    else:
        wcs = upload_and_solve(
            fits_path,
            api_key,
            timeout=timeout,
            publicly_visible=publicly_visible,
            allow_commercial_use=allow_commercial_use,
        )

    if wcs is None:
        return None

    if not (
        _validate_local_wcs(wcs, fits_path, instrument)
        if backend == "local"
        else validate_wcs(wcs, instrument)
    ):
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
        "--backend",
        choices=["nova", "local"],
        default="nova",
        help="Astrometry.net backend (default: nova). Local never uploads.",
    )
    add_local_solver_arguments(ap, include_timeout=False)
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
        type=_positive_timeout,
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

    if args.backend == "local" and args.out_dir is not None and not args.dry_run:
        logger.error(
            "Local solving writes sidecars only; omit --out-dir or use --dry-run."
        )
        return 1
    api_key = args.api_key
    if args.backend == "nova" and api_key is None:
        try:
            api_key = _api_key()
        except RuntimeError as exc:
            logger.error(str(exc))
            return 1

    # Always persist the downloaded solution. This also makes a solve useful
    # without --out-dir and gives downstream/reference-building code a stable
    # artifact that can be reused without another upload.
    wcs_output = args.wcs_output or args.fits.with_name(f"{args.fits.stem}_wcs.fits")

    if args.backend == "local" and wcs_output.resolve() == args.fits.resolve():
        logger.error("Local WCS output must differ from the input FITS file.")
        return 1
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
        backend=args.backend,
        executable=args.solve_field,
        config=args.astrometry_config,
        temp_dir=args.astrometry_temp_dir,
    )

    if wcs is None:
        logger.error("WCS solving failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
