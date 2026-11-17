"""kSIRIUS (Kagoshima University 1m) instrument constants and WCS preparation.

The KU 1m telescope at Iriki Observatory images simultaneously with an
optical camera (``INSTRMEN=optcam``: g, i; 1044x1032 px, ~0.62"/px) and the
kSIRIUS near-infrared camera (``INSTRMEN=kSIRIUS``: J, H, Ks; 320x255 px,
~0.694"/px). The archived frames are already sky-subtracted, so no
calibration is needed, but they carry no WCS.

Single frames hold only a handful of stars (the NIR field is ~3.7'x2.9'), too
few for a reliable blind solve. Each band is therefore solved once, on a
median stack of frames around the middle of the sequence, against Gaia:
twirl first, then (for the sparse NIR fields, ~4-8 stars) a search over
rotation and parity at the camera's known pixel scale. The pointing is stable
to ~1-2 px through a night, so that one solution is written into WCS-tagged
copies of every frame of the band (the raw data directory is never
modified). The solution and the copies are cached and reused on later runs.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, Mapping

import astropy.units as u
import astropy.wcs.utils as wcsutils
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger("prose_run_photometry")

# --------------------------------------------------------------- identity
KSIRIUS_TELESCOPE_PREFIX = "ku 1m"  # TELESCOP = 'KU 1m telescope'
KSIRIUS_CAMERAS = frozenset({"optcam", "ksirius"})  # INSTRMEN values, lower-cased
KSIRIUS_BANDS = ("gp", "ip", "J", "H", "Ks")
KSIRIUS_FILTER_ALIASES = {"g": "gp", "i": "ip", "J": "J", "H": "H", "Ks": "Ks"}
# Pixel scale [arcsec/px] per camera (INSTRMEN), measured by plate-solving
# /data/kSIRIUS: J with astrometry.net (0.694), i/g stacks with twirl (0.62).
# Mirrored by pixel_scale in data/ksirius_*.telescope (a test keeps them equal).
KSIRIUS_PIXEL_SCALES = {"optcam": 0.62, "ksirius": 0.694}
# Iriki Observatory (lat [deg], lon [deg E], height [m]). Approximate to
# ~100 m, which shifts BJD by well under a millisecond.
KSIRIUS_LOCATION = (31.7479, 130.4399, 574.0)

# ------------------------------------------------------------- WCS solving
WCS_METHOD = "ksirius-stack"
WCS_SUFFIX = "_wcs.fits"
STACK_NFRAMES = 40  # frames median-combined for the per-band solve
EDGE_TRIM_PIX = 16  # optcam frames have a ~10 px border of garbage values
DETECT_SIGMA = 5.0
DETECT_MIN_AREA = 10  # [pix]; rejects hot pixels and cosmic-ray residue
MIN_SOURCE_AXIS_PIX = 1.0  # sep semi-axes below this are not stellar
MAX_DETECTIONS = 30
GAIA_RADIUS_FACTOR = 0.75  # query radius / field diagonal (covers pointing error)
GAIA_MAX_STARS = (15, 25, 40, 60)  # brightest-N Gaia subsets tried by twirl
SCALE_TOLERANCE = 0.1  # solved scale must be within 10% of the telescope file's
MATCH_TOLERANCE_PIX = 2.0
MIN_MATCHES = 4
# Known-scale fallback for sparse fields (the 3.7' kSIRIUS field shows ~4-8
# stars, too few for twirl's asterism matching): search rotation x parity,
# vote on the translation, keep a unique best match.
ROTATION_STEP_DEG = 0.5
VOTE_BIN_PIX = 4.0
POINTING_MARGIN_ARCSEC = 90.0  # Gaia kept within half-diagonal + this
KNOWN_SCALE_GAIA_STARS = 40
MIN_MATCH_FRACTION = 0.6  # of detected stars
DISTINCT_ROTATION_DEG = 3.0
DISTINCT_SHIFT_PIX = 10.0

GaiaFn = Callable[[SkyCoord, u.Quantity], np.ndarray]


def is_ksirius_header(header) -> bool:
    """True for frames from either KU 1m camera (optcam or kSIRIUS)."""
    telescope = str(header.get("TELESCOP", "")).strip().lower()
    camera = str(header.get("INSTRMEN", "")).strip().lower()
    return telescope.startswith(KSIRIUS_TELESCOPE_PREFIX) and camera in KSIRIUS_CAMERAS


def median_stack(files: list[str], nframes: int = STACK_NFRAMES) -> np.ndarray:
    """Median of up to *nframes* frames centred on the middle of *files*."""
    n = max(1, min(nframes, len(files)))
    start = max(0, len(files) // 2 - n // 2)
    frames = [fits.getdata(f).astype(np.float64) for f in files[start : start + n]]
    return np.nanmedian(frames, axis=0)


def detect_stars(image: np.ndarray, trim: int = EDGE_TRIM_PIX) -> np.ndarray:
    """Pixel (x, y) of stellar sources, brightest first, ignoring the border."""
    import sep

    data = np.nan_to_num(np.asarray(image, dtype=np.float64))
    ny, nx = data.shape
    core = np.ascontiguousarray(data[trim : ny - trim, trim : nx - trim])
    background = sep.Background(core)
    residual = core - background.back()
    sources = sep.extract(
        residual, DETECT_SIGMA, err=background.globalrms, minarea=DETECT_MIN_AREA
    )
    stellar = sources[
        (sources["a"] > MIN_SOURCE_AXIS_PIX) & (sources["b"] > MIN_SOURCE_AXIS_PIX)
    ]
    stellar = stellar[np.argsort(-stellar["flux"])][:MAX_DETECTIONS]
    return np.column_stack([stellar["x"] + trim, stellar["y"] + trim])


def wcs_pixel_scale(wcs: WCS) -> float:
    """Mean pixel scale of *wcs* in arcsec/px."""
    return float(np.mean(wcsutils.proj_plane_pixel_scales(wcs)) * 3600.0)


def _count_matches(wcs: WCS, xy: np.ndarray, radec: np.ndarray) -> int:
    projected = np.column_stack(wcs.world_to_pixel_values(radec[:, 0], radec[:, 1]))
    distances = np.linalg.norm(projected[:, None, :] - xy[None, :, :], axis=2)
    return int(np.sum(distances.min(axis=0) < MATCH_TOLERANCE_PIX))


def _matches_needed(n_detected: int) -> int:
    """Matched stars required to accept a solution of *n_detected* stars."""
    return max(MIN_MATCHES, int(np.ceil(MIN_MATCH_FRACTION * n_detected)))


def solve_wcs(xy: np.ndarray, radec: np.ndarray, expected_scale: float) -> WCS | None:
    """Best twirl solution matching *xy* to Gaia *radec* (bright first).

    Several brightest-N Gaia subsets are tried, since a sparse field may
    only line up with the right subset. The solution with the most stars
    matched within ``MATCH_TOLERANCE_PIX`` wins; it is rejected unless its
    scale is within ``SCALE_TOLERANCE`` of *expected_scale* and it matches
    at least ``MIN_MATCHES`` and ``MIN_MATCH_FRACTION`` of the detections.
    """
    import twirl

    if len(xy) < MIN_MATCHES or len(radec) < MIN_MATCHES:
        logger.warning(
            f"WCS: too few stars to solve ({len(xy)} detected, {len(radec)} Gaia)"
        )
        return None
    best: tuple[int, WCS] | None = None
    for n_gaia in GAIA_MAX_STARS:
        try:
            wcs = twirl.compute_wcs(xy, radec[:n_gaia])
        except Exception as exc:  # noqa: BLE001 - twirl raises bare errors on no match
            logger.debug(f"WCS: twirl failed with {n_gaia} Gaia stars ({exc})")
            continue
        scale = wcs_pixel_scale(wcs)
        if abs(scale / expected_scale - 1) > SCALE_TOLERANCE:
            logger.debug(f'WCS: scale {scale:.3f}"/px rejected (n_gaia={n_gaia})')
            continue
        matches = _count_matches(wcs, xy, radec)
        if best is None or matches > best[0]:
            best = (matches, wcs)
    needed = _matches_needed(len(xy))
    if best is None or best[0] < needed:
        found = 0 if best is None else best[0]
        logger.warning(
            f'WCS: no solution near {expected_scale}"/px with >= {needed} '
            f"matched stars (best: {found})"
        )
        return None
    logger.info(
        f"WCS: solved, {best[0]}/{len(xy)} stars matched, "
        f'scale {wcs_pixel_scale(best[1]):.4f}"/px'
    )
    return best[1]


def _tangent_arcsec(radec: np.ndarray, center: SkyCoord) -> np.ndarray:
    """Gnomonic (xi, eta) [arcsec] of *radec* about *center*."""
    ra0, dec0 = center.ra.rad, center.dec.rad
    ra, dec = np.deg2rad(radec[:, 0]), np.deg2rad(radec[:, 1])
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cos_c
    eta = (
        np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    ) / cos_c
    return np.rad2deg(np.column_stack([xi, eta])) * 3600.0


def _rotation(angle_deg: float) -> np.ndarray:
    t = np.deg2rad(angle_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _vote_shift(xy: np.ndarray, model: np.ndarray) -> tuple[int, np.ndarray]:
    """Translation aligning the most *model* points with *xy*, and its count."""
    diffs = (xy[:, None, :] - model[None, :, :]).reshape(-1, 2)
    best = (0, np.zeros(2))
    for offset in (0.0, VOTE_BIN_PIX / 2):  # two grids so no peak straddles bins
        keys = np.floor((diffs + offset) / VOTE_BIN_PIX).astype(int)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        shift = diffs[inverse.ravel() == np.argmax(counts)].mean(axis=0)
        dist = np.linalg.norm(xy[:, None] - (model + shift)[None], axis=2).min(axis=1)
        n = int(np.sum(dist < MATCH_TOLERANCE_PIX))
        if n > best[0]:
            best = (n, shift)
    return best


def _known_scale_wcs(
    center: SkyCoord,
    scale: float,
    parity: int,
    angle: float,
    origin: np.ndarray,
) -> WCS:
    """TAN WCS with pixel = origin + R(angle) @ diag(parity, 1) @ (xi, eta) / scale."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [center.ra.deg, center.dec.deg]
    wcs.wcs.crpix = list(origin + 1.0)  # FITS pixels are 1-based
    flip = np.diag([parity, 1.0])
    wcs.wcs.cd = (scale / 3600.0) * flip @ _rotation(angle).T
    return wcs


def solve_wcs_known_scale(
    xy: np.ndarray,
    radec: np.ndarray,
    center: SkyCoord,
    scale: float,
    shape: tuple[int, int],
) -> WCS | None:
    """Rotation/parity/translation solve at a fixed *scale* for sparse fields.

    Gaia stars within the field (plus ``POINTING_MARGIN_ARCSEC``) are rotated
    through 360 deg in ``ROTATION_STEP_DEG`` steps for both parities; each
    orientation votes on the translation. The best orientation must match at
    least ``MIN_MATCHES`` and ``MIN_MATCH_FRACTION`` of the detected stars,
    and strictly more than any distinct alternative (different parity,
    rotation or translation) -- a tie means the field is ambiguous.
    """
    ny, nx = shape
    if len(xy) < MIN_MATCHES:
        logger.warning(f"WCS: too few stars for a known-scale solve ({len(xy)})")
        return None
    uv = _tangent_arcsec(radec, center)
    half_diagonal = 0.5 * np.hypot(nx, ny) * scale + POINTING_MARGIN_ARCSEC
    uv = uv[np.hypot(*uv.T) < half_diagonal][:KNOWN_SCALE_GAIA_STARS] / scale
    if len(uv) < MIN_MATCHES:
        logger.warning(f"WCS: too few Gaia stars in the field ({len(uv)})")
        return None
    centre_pix = np.array([nx / 2, ny / 2])
    xy_c = xy - centre_pix
    candidates = []
    for parity in (-1, 1):
        flipped = uv * [parity, 1]
        for angle in np.arange(0.0, 360.0, ROTATION_STEP_DEG):
            n, shift = _vote_shift(xy_c, flipped @ _rotation(angle).T)
            candidates.append((n, parity, float(angle), shift))
    candidates.sort(key=lambda c: -c[0])
    n_best, parity, angle, shift = candidates[0]
    rival = next(
        (
            c[0]
            for c in candidates[1:]
            if c[1] != parity
            or abs((c[2] - angle + 180) % 360 - 180) > DISTINCT_ROTATION_DEG
            or np.hypot(*(c[3] - shift)) > DISTINCT_SHIFT_PIX
        ),
        0,
    )
    needed = _matches_needed(len(xy))
    if n_best < needed or n_best <= rival:
        logger.warning(
            f"WCS: known-scale solve ambiguous or weak ({n_best}/{len(xy)} matched, "
            f"need {needed}; best distinct alternative {rival})"
        )
        return None
    offset_arcsec = float(np.hypot(*shift)) * scale
    if offset_arcsec > POINTING_MARGIN_ARCSEC:
        logger.warning(
            f'WCS: known-scale solve rejected: {offset_arcsec:.0f}" from the header '
            f'pointing (max {POINTING_MARGIN_ARCSEC:.0f}")'
        )
        return None
    wcs = _known_scale_wcs(center, scale, parity, angle, centre_pix + shift)
    logger.info(
        f"WCS: known-scale solve, {n_best}/{len(xy)} stars matched "
        f"(rotation {angle:.1f} deg, parity {parity:+d}, next best {rival})"
    )
    return wcs


def query_gaia(center: SkyCoord, radius: u.Quantity) -> np.ndarray:
    """Gaia (ra, dec) [deg] within *radius* of *center*, brightest first."""
    from prose.utils import gaia_query

    table = gaia_query(center, 2 * radius, "ra", "dec", "phot_g_mean_mag")
    radec = np.column_stack([np.asarray(table["ra"]), np.asarray(table["dec"])])
    return radec[np.all(np.isfinite(radec), axis=1)]


def _pointing(header) -> SkyCoord:
    """Header pointing; sexagesimal RA/DEC strings are hours/degrees."""
    ra, dec = header["RA"], header["DEC"]
    if isinstance(ra, str) and ":" in ra:
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    return SkyCoord(float(ra), float(dec), unit=u.deg)


def solve_band(
    files: list[str], expected_scale: float, gaia_fn: GaiaFn = query_gaia
) -> WCS | None:
    """Solve one band on a median stack of its frames, or ``None``."""
    stack = median_stack(files)
    xy = detect_stars(stack)
    header = fits.getheader(files[0])
    ny, nx = stack.shape
    radius = GAIA_RADIUS_FACTOR * np.hypot(nx, ny) * expected_scale * u.arcsec
    pointing = _pointing(header)
    try:
        radec = gaia_fn(pointing, radius)
    except Exception as exc:  # noqa: BLE001 - network/catalog errors vary
        logger.warning(f"WCS: Gaia query failed ({exc})")
        return None
    wcs = solve_wcs(xy, radec, expected_scale)
    if wcs is None:
        logger.info("WCS: twirl failed; trying the known-scale sparse-field solve")
        wcs = solve_wcs_known_scale(xy, radec, pointing, expected_scale, stack.shape)
    return wcs


def _copy_path(src: str, out_dir: Path) -> Path:
    name = Path(src).name
    stem = name[: -len(".fits")] if name.endswith(".fits") else name
    return out_dir / f"{stem}{WCS_SUFFIX}"


def _same_wcs(a: WCS, b: WCS) -> bool:
    """True when two celestial WCS solutions are numerically the same."""
    return bool(
        np.allclose(a.wcs.crval, b.wcs.crval, rtol=0, atol=1e-9)
        and np.allclose(a.wcs.crpix, b.wcs.crpix, rtol=0, atol=1e-6)
        and np.allclose(a.pixel_scale_matrix, b.pixel_scale_matrix, rtol=1e-9, atol=0)
    )


def _copy_matches(path: Path, wcs: WCS) -> bool:
    """True when *path* is a copy from this module carrying exactly *wcs*."""
    try:
        header = fits.getheader(path)
    except (OSError, ValueError):
        return False
    if header.get("WCSMTHD") != WCS_METHOD:
        return False
    existing = WCS(header)
    return existing.has_celestial and _same_wcs(existing, wcs)


def write_wcs_copies(files: Iterable[str], wcs: WCS, out_dir: Path) -> list[str]:
    """Copy *files* into *out_dir* with *wcs* and ``PIXSCALE`` in the header.

    Copies already carrying this exact WCS are kept; any other copy (e.g. one
    written before a re-solve) is rewritten.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = wcs.to_header(relax=True)
    scale = round(wcs_pixel_scale(wcs), 4)
    copies = []
    for src in files:
        dst = _copy_path(src, out_dir)
        if _copy_matches(dst, wcs):
            copies.append(str(dst))
            continue
        tmp = dst.with_suffix(".tmp")
        shutil.copyfile(src, tmp)
        with fits.open(tmp, mode="update") as hdul:
            header = hdul[0].header
            header.update(cards)
            header["PIXSCALE"] = (scale, "[arcsec/px] from the WCS solution")
            header["WCSMTHD"] = (WCS_METHOD, "median-stack Gaia solve")
        tmp.replace(dst)  # atomic: an interrupted run never leaves a half file
        copies.append(str(dst))
    return copies


def _expected_scale(path: str) -> float:
    """Nominal pixel scale of the camera that took *path*."""
    camera = str(fits.getheader(path).get("INSTRMEN", "")).strip().lower()
    return KSIRIUS_PIXEL_SCALES[camera]


def _safe_token(text: str) -> str:
    """Filesystem-safe token for an OBJECT name."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return token or "unknown"


def sidecar_path(out_dir: Path, band: str, target: str = "unknown") -> Path:
    """Where a target's per-band solved WCS is cached for later runs.

    Keyed by target as well as band: one data directory holds several targets
    per night, each at its own pointing.
    """
    name = f"{_safe_token(target)}_{band}_{WCS_METHOD}.wcs.fits"
    return Path(out_dir) / ".wcs" / name


def _save_sidecar(wcs: WCS, path: Path, pointing: SkyCoord, shape) -> None:
    """Cache *wcs* with the pointing and frame shape it was solved for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = wcs.to_header(relax=True)
    header["KSPNTRA"] = (pointing.ra.deg, "[deg] header pointing at solve time")
    header["KSPNTDE"] = (pointing.dec.deg, "[deg] header pointing at solve time")
    header["KSNAXIS1"] = (int(shape[1]), "frame width at solve time")
    header["KSNAXIS2"] = (int(shape[0]), "frame height at solve time")
    fits.PrimaryHDU(header=header).writeto(path, overwrite=True)


def _load_sidecar(path: Path, pointing: SkyCoord, shape) -> WCS | None:
    """Cached WCS, or ``None`` if missing or solved for another field/camera."""
    try:
        header = fits.getheader(path)
        cached = SkyCoord(header["KSPNTRA"], header["KSPNTDE"], unit=u.deg)
        cached_shape = (int(header["KSNAXIS2"]), int(header["KSNAXIS1"]))
    except (OSError, KeyError, ValueError):
        return None
    if cached_shape != tuple(shape):
        logger.info(
            f"ignoring cached WCS {path}: frame shape {cached_shape} != {tuple(shape)}"
        )
        return None
    separation = cached.separation(pointing).arcsec
    if separation > POINTING_MARGIN_ARCSEC:
        logger.info(
            f'ignoring cached WCS {path}: pointing differs by {separation:.0f}"'
        )
        return None
    wcs = WCS(header)
    return wcs if wcs.has_celestial else None


def _band_wcs(
    band: str,
    solve_files: list[str],
    out_dir: Path,
    gaia_fn: GaiaFn,
) -> WCS | None:
    """The band's cached WCS if it fits this field, else a fresh solve."""
    header = fits.getheader(solve_files[0])
    pointing = _pointing(header)
    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    sidecar = sidecar_path(out_dir, band, header.get("OBJECT", "unknown"))
    wcs = _load_sidecar(sidecar, pointing, shape)
    if wcs is not None:
        logger.info(f"[{band}] reusing cached WCS {sidecar}")
        return wcs
    expected = _expected_scale(solve_files[0])
    logger.info(
        f"[{band}] solving WCS on a median stack of "
        f'{min(STACK_NFRAMES, len(solve_files))} frames (expected {expected}"/px)'
    )
    wcs = solve_band(solve_files, expected, gaia_fn=gaia_fn)
    if wcs is not None:
        _save_sidecar(wcs, sidecar, pointing, shape)
    return wcs


# Failures preparing one band (unreadable/ragged frames, missing header keys,
# missing optional solver packages, an unwritable output directory) fall back
# to that band's original frames instead of aborting the run.
_BAND_PREP_ERRORS = (OSError, KeyError, ValueError, ImportError, RuntimeError)


def prepare_ksirius_frames(
    sciences: Mapping[str, list[str]],
    out_dir: Path,
    gaia_fn: GaiaFn = query_gaia,
    solve_sciences: Mapping[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Per band, WCS-tagged copies of the frames in *out_dir*.

    Each band is solved once, on a median stack drawn from
    ``solve_sciences[band]`` when given (e.g. the whole night while
    ``--test_run`` reduces only a window) and otherwise from *sciences*. The
    solution is cached per target and band under ``out_dir/.wcs`` together
    with the pointing and frame shape it was solved for, and is only reused
    for a matching field. Copies carrying that exact WCS are kept, so later
    runs neither re-solve nor re-copy. A band that cannot be solved or
    written keeps its original (WCS-less) frames, so the run can still
    proceed with ``--tID``.
    """
    out_dir = Path(out_dir)
    prepared: dict[str, list[str]] = {}
    for band, files in sciences.items():
        if not files:
            prepared[band] = list(files)
            continue
        solve_files = list((solve_sciences or {}).get(band) or files)
        try:
            wcs = _band_wcs(band, solve_files, out_dir, gaia_fn)
            if wcs is not None:
                prepared[band] = write_wcs_copies(files, wcs, out_dir)
                continue
            reason = "WCS solve failed"
        except _BAND_PREP_ERRORS as exc:
            reason = f"WCS preparation failed ({type(exc).__name__}: {exc})"
        logger.warning(
            f"[{band}] {reason}; using original frames without WCS "
            "(set --tID to identify the target)"
        )
        prepared[band] = list(files)
    return prepared
