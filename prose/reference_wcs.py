"""Check a reference frame's WCS against Gaia, and refit it when they disagree.

Target identification reads the target's pixel position from the reference
frame's WCS. That WCS can be wrong while still looking valid: MuSCAT2
calibration copies one astrometry.net solution into every frame, so a
drifted reference frame carries a WCS that is off by the drift (#94), and
BANZAI writes approximate solutions with ``WCSERR != 0`` (#77).

:func:`verify_reference_wcs` projects Gaia stars through the header WCS and
counts the detected sources that land on one. When too few do, it solves the
frame against Gaia with :func:`solve_wcs`, a pair-matching search that needs
only the pixel scale and works on fields with a handful of stars, and uses
that solution when it passes the same check (#95).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.spatial import cKDTree

#: A detection matches a Gaia star projected within this many pixels.
MATCH_TOLERANCE_PIX = 3.0
#: Fewer detections than this cannot verify a WCS.
MIN_VERIFY_DETECTIONS = 2
#: A WCS passes when at least this fraction of the detections match Gaia.
MIN_MATCH_FRACTION = 0.5
#: Gaia stars considered per detection (brightest first) when checking.
GAIA_PER_DETECTION = 3
#: A refit needs at least this many matched stars to be accepted.
MIN_SOLVE_MATCHES = 4
#: Brightest detections and Gaia stars paired when generating hypotheses.
MAX_SOLVE_DETECTIONS = 12
MAX_SOLVE_GAIA = 60
#: Looser tolerance for scoring a hypothesis before least-squares refinement.
HYPOTHESIS_TOLERANCE_PIX = 6.0
#: Pair lengths agree within this many pixels plus this fraction of the length.
PAIR_LENGTH_TOLERANCE_PIX = 3.0
PAIR_LENGTH_TOLERANCE_FRACTION = 0.02
#: A refit whose pixel scale differs from the nominal one by more is rejected.
MAX_SCALE_DEVIATION = 0.05
#: Hypotheses refined by least squares, best scored first.
MAX_REFINED_HYPOTHESES = 200


@dataclass(frozen=True)
class WcsCheck:
    """How many detected sources a WCS places on a Gaia star."""

    n_matched: int
    n_detections: int

    @property
    def is_verifiable(self) -> bool:
        return self.n_detections >= MIN_VERIFY_DETECTIONS

    @property
    def required(self) -> int:
        return max(
            MIN_VERIFY_DETECTIONS, math.ceil(MIN_MATCH_FRACTION * self.n_detections)
        )

    @property
    def passed(self) -> bool:
        return self.is_verifiable and self.n_matched >= self.required


@dataclass(frozen=True)
class WcsVerdict:
    """The WCS to use for target identification, and why."""

    wcs: Optional[WCS]
    source: str  # "header", "gaia_refit", or "unverified"
    trusted: bool
    header_check: Optional[WcsCheck] = None
    refit_check: Optional[WcsCheck] = None
    notes: tuple = field(default_factory=tuple)

    def summary(self) -> str:
        def fmt(check):
            return f"{check.n_matched}/{check.n_detections}" if check else "n/a"

        text = (
            f"reference WCS {self.source} "
            f"({'trusted' if self.trusted else 'not trusted'}); "
            f"Gaia matches: header {fmt(self.header_check)}"
        )
        if self.refit_check is not None:
            text += f", refit {fmt(self.refit_check)}"
        if self.notes:
            text += f"; {'; '.join(self.notes)}"
        return text


def has_celestial_wcs(wcs) -> bool:
    return wcs is not None and bool(getattr(wcs, "has_celestial", False))


def _matched_count(points: np.ndarray, targets: np.ndarray, tolerance: float) -> int:
    """Number of distinct *targets* within *tolerance* of some point."""
    if len(points) == 0 or len(targets) == 0:
        return 0
    distances, indices = cKDTree(targets).query(points)
    return len(set(indices[distances <= tolerance].tolist()))


def _in_frame(xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= -0.5)
        & (xy[:, 0] <= width - 0.5)
        & (xy[:, 1] >= -0.5)
        & (xy[:, 1] <= height - 0.5)
    )


def check_wcs(
    wcs,
    pixel_coords: np.ndarray,
    gaia: SkyCoord,
    shape: tuple[int, int],
    tolerance: float = MATCH_TOLERANCE_PIX,
) -> WcsCheck:
    """Count detections that *wcs* places on one of the brightest Gaia stars.

    *gaia* must be ordered brightest first. Only stars that *wcs* projects
    into the frame are considered, up to ``GAIA_PER_DETECTION`` per detection,
    so a fainter Gaia star does not stand in for an undetected bright one.
    """
    pixel_coords = np.asarray(pixel_coords, dtype=float).reshape(-1, 2)
    n_detections = len(pixel_coords)
    if not has_celestial_wcs(wcs) or len(gaia) == 0:
        return WcsCheck(0, n_detections)
    try:
        x, y = wcs.world_to_pixel(gaia)
    except Exception:  # noqa: BLE001 - an unusable WCS simply fails the check
        return WcsCheck(0, n_detections)
    projected = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    projected = projected[_in_frame(projected, shape)]
    projected = projected[: max(GAIA_PER_DETECTION * n_detections, 10)]
    return WcsCheck(_matched_count(pixel_coords, projected, tolerance), n_detections)


def _tangent_plane_deg(gaia: SkyCoord, center: SkyCoord) -> np.ndarray:
    """Gnomonic offsets of *gaia* about *center*, in degrees (x east, y north)."""
    plane = _tan_wcs(center, crpix=(1.0, 1.0), cd=np.eye(2))
    x, y = plane.world_to_pixel(gaia)
    return np.column_stack([np.asarray(x, float), np.asarray(y, float)])


def _tan_wcs(center: SkyCoord, crpix, cd: np.ndarray) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [center.ra.deg, center.dec.deg]
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.cd = np.asarray(cd, dtype=float)
    return wcs


def _fit_similarity(src: np.ndarray, dst: np.ndarray, flip: int):
    """Least-squares ``dst = M @ src + t`` with M a rotation-scale (and flip)."""
    s = src * np.array([1.0, flip])
    n = len(s)
    design = np.zeros((2 * n, 4))
    design[0::2] = np.column_stack([s[:, 0], -s[:, 1], np.ones(n), np.zeros(n)])
    design[1::2] = np.column_stack([s[:, 1], s[:, 0], np.zeros(n), np.ones(n)])
    (a, b, tx, ty), *_ = np.linalg.lstsq(design, dst.reshape(-1), rcond=None)
    matrix = np.array([[a, -b], [b, a]]) @ np.diag([1.0, flip])
    return matrix, np.array([tx, ty])


def _pair_hypotheses(det: np.ndarray, plane: np.ndarray):
    """Similarity transforms mapping a Gaia pair onto a detection pair.

    *plane* is in nominal pixels, so a true pair has equal lengths. Yields
    ``(matrix, translation)`` for both parities.
    """
    di, dj = np.triu_indices(len(det), k=1)
    gk, gl = np.nonzero(~np.eye(len(plane), dtype=bool))
    d_vec = det[dj] - det[di]
    g_vec = plane[gl] - plane[gk]
    d_len = np.hypot(*d_vec.T)
    g_len = np.hypot(*g_vec.T)
    tolerance = PAIR_LENGTH_TOLERANCE_PIX + PAIR_LENGTH_TOLERANCE_FRACTION * d_len
    p, q = np.nonzero(np.abs(d_len[:, None] - g_len[None, :]) <= tolerance[:, None])
    for flip in (1, -1):
        g_flip = g_vec[q] * np.array([1.0, flip])
        angle = np.arctan2(d_vec[p, 1], d_vec[p, 0]) - np.arctan2(
            g_flip[:, 1], g_flip[:, 0]
        )
        cos, sin = np.cos(angle), np.sin(angle)
        for c, s, i, k in zip(cos, sin, di[p], gk[q]):
            matrix = np.array([[c, -s], [s, c]]) @ np.diag([1.0, flip])
            yield matrix, det[i] - matrix @ plane[k], flip


def solve_wcs(
    pixel_coords: np.ndarray,
    gaia: SkyCoord,
    center: SkyCoord,
    pixel_scale_deg: float,
    shape: tuple[int, int],
    tolerance: float = MATCH_TOLERANCE_PIX,
) -> Optional[WCS]:
    """Solve a TAN WCS from detections and Gaia, knowing only the pixel scale.

    Every pair of bright detections is matched with every Gaia pair of the
    same length, in both parities, giving a rotation and offset. Each
    hypothesis is scored by how many detections land on a Gaia star, the best
    are refined by least squares on their matches, and the winner must match
    at least ``MIN_SOLVE_MATCHES`` stars and beat every distinct solution.
    Returns ``None`` when no unambiguous solution exists.
    """
    pixel_coords = np.asarray(pixel_coords, dtype=float).reshape(-1, 2)
    det = pixel_coords[:MAX_SOLVE_DETECTIONS]
    gaia = gaia[:MAX_SOLVE_GAIA]
    if len(det) < 2 or len(gaia) < 2 or len(pixel_coords) < MIN_SOLVE_MATCHES:
        return None
    plane = _tangent_plane_deg(gaia, center) / pixel_scale_deg
    det_tree = cKDTree(pixel_coords)

    scored = []
    for matrix, translation, flip in _pair_hypotheses(det, plane):
        projected = plane @ matrix.T + translation
        distances, _ = det_tree.query(projected)
        score = int(np.count_nonzero(distances <= HYPOTHESIS_TOLERANCE_PIX))
        if score >= MIN_SOLVE_MATCHES:
            scored.append((score, matrix, translation, flip))
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])

    solutions = []  # (n_matched, matrix, translation)
    for _, matrix, translation, flip in scored[:MAX_REFINED_HYPOTHESES]:
        refined = _refine(pixel_coords, plane, matrix, translation, flip, tolerance)
        if refined is not None:
            solutions.append(refined)
    if not solutions:
        return None
    solutions.sort(key=lambda item: -item[0])
    best_count, best_matrix, best_translation = solutions[0]
    if best_count < MIN_SOLVE_MATCHES:
        return None
    # Map the winning field's corners back to tangent-plane coordinates.
    # Corners bound the displacement of two affine transforms over the frame.
    height, width = shape
    corners = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]]
    )
    sky_corners = np.linalg.solve(best_matrix, (corners - best_translation).T).T
    for count, matrix, translation in solutions[1:]:
        if count < best_count:
            break
        projected = sky_corners @ matrix.T + translation
        if np.any(np.linalg.norm(projected - corners, axis=1) > 2 * tolerance):
            return None  # equally supported solutions disagree on the sky
    # best_matrix maps nominal-pixel offsets to pixels; CD maps pixels to degrees.
    cd = np.linalg.inv(best_matrix / pixel_scale_deg)
    return _tan_wcs(center, crpix=best_translation + 1.0, cd=cd)


def _refine(pixel_coords, plane, matrix, translation, flip, tolerance):
    """Least-squares refit on a hypothesis's matches; ``None`` if implausible."""
    for match_tolerance in (HYPOTHESIS_TOLERANCE_PIX, tolerance):
        projected = plane @ matrix.T + translation
        distances, det_index = cKDTree(pixel_coords).query(projected)
        keep = distances <= match_tolerance
        if np.count_nonzero(keep) < 2:
            return None
        matrix, translation = _fit_similarity(
            plane[keep], pixel_coords[det_index[keep]], flip
        )
        scale = math.sqrt(abs(np.linalg.det(matrix)))
        if abs(scale - 1.0) > MAX_SCALE_DEVIATION:
            return None
    projected = plane @ matrix.T + translation
    return (
        _matched_count(projected, pixel_coords, tolerance),
        matrix,
        translation,
    )


def verify_reference_wcs(
    header_wcs,
    pixel_coords: np.ndarray,
    gaia: Optional[SkyCoord],
    center: SkyCoord,
    pixel_scale_deg: float,
    shape: tuple[int, int],
) -> WcsVerdict:
    """Choose the WCS for target identification: the header's if Gaia agrees,
    otherwise a Gaia refit that passes the same check, otherwise the header's
    marked as not trusted."""
    pixel_coords = np.asarray(pixel_coords, dtype=float).reshape(-1, 2)
    if gaia is None or len(gaia) == 0:
        return WcsVerdict(header_wcs, "unverified", False, notes=("no Gaia catalog",))
    header_check = check_wcs(header_wcs, pixel_coords, gaia, shape)
    if header_check.passed:
        return WcsVerdict(header_wcs, "header", True, header_check)
    if not header_check.is_verifiable:
        return WcsVerdict(
            header_wcs,
            "unverified",
            False,
            header_check,
            notes=("too few detections to check",),
        )
    try:
        solved = solve_wcs(pixel_coords, gaia, center, pixel_scale_deg, shape)
    except Exception as exc:  # noqa: BLE001 - preserve the completed header check
        return WcsVerdict(
            header_wcs,
            "header",
            False,
            header_check,
            notes=(f"Gaia refit failed: {exc}",),
        )
    if solved is None:
        return WcsVerdict(
            header_wcs, "header", False, header_check, notes=("Gaia refit failed",)
        )
    refit_check = check_wcs(solved, pixel_coords, gaia, shape)
    if not refit_check.passed:
        return WcsVerdict(
            header_wcs,
            "header",
            False,
            header_check,
            refit_check,
            notes=("Gaia refit did not pass the check",),
        )
    return WcsVerdict(solved, "gaia_refit", True, header_check, refit_check)
