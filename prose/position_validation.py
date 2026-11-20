"""Independent checks of aperture positions against the pre-alignment catalog."""

import numpy as np
from scipy.spatial import cKDTree

from prose.core.block import Block


def _celestial_wcs(image):
    wcs = getattr(image, "wcs", None)
    flag = (getattr(image, "header", None) or {}).get("WCSERR", 0)
    if str(flag).strip() in {"0", "0.0"} and getattr(wcs, "has_celestial", False):
        return wcs
    return None


def _field_shift(reference, detections, tolerance):
    """Vote on a translation with the brightest sources, independently of Twirl."""
    # Bound the number of hypotheses for crowded fields. All detections are
    # still available for the individual aperture checks.
    ref = reference[:15]
    det = detections[:15]
    if min(len(ref), len(det)) < 2:
        return None
    offsets = (det[None, :, :] - ref[:, None, :]).reshape(-1, 2)
    distances, ids = cKDTree(det).query(ref[None, :, :] + offsets[:, None, :])
    _, reverse_ids = cKDTree(ref).query(det[None, :, :] - offsets[:, None, :])
    mutual = np.take_along_axis(reverse_ids, ids, axis=1) == np.arange(len(ref))
    matches = mutual & (distances <= tolerance)
    support = matches.sum(axis=1)
    best = np.flatnonzero(support == support.max())
    if support[best[0]] < 2:
        return None
    errors = np.where(matches, distances, 0).sum(axis=1) / support.clip(min=1)
    winner = best[np.argmin(errors[best])]
    if np.any(np.linalg.norm(offsets[best] - offsets[winner], axis=1) > 2 * tolerance):
        return None  # distinct translations explain the same number of stars
    ok = matches[winner]
    return np.median(det[ids[winner, ok]] - ref[ok], axis=0)


def _associations(predicted, detections, tolerance):
    distances, ids = cKDTree(detections).query(predicted, k=2)
    reverse, _ = cKDTree(predicted).query(detections, k=2)
    close = distances[:, 0] <= tolerance
    unique = close & (distances[:, 1] > tolerance) & (reverse[ids[:, 0], 1] > tolerance)
    return distances[:, 0], ids[:, 0], close, unique


# Stored alongside the two separations and their respective tolerances.
POSITION_REASONS = np.array(
    [
        "verified",
        "unregistered_field",
        "missing_detection",
        "ambiguous_identity",
        "missing_center",
        "center_on_other_detection",
        "center_too_far",
    ]
)


class MatchExposureSources(Block):
    """Record independently identified detections before alignment replaces them."""

    def __init__(self, ref, coords=None, coordinate_image=None):
        super().__init__()
        self._parallel_friendly = True
        self.reference = np.array(
            ref.sources.coords if coords is None else coords, dtype=float
        )
        self.worker_shared_arrays = (self.reference,)
        self.reference_fwhm = getattr(ref, "fwhm", np.nan)
        wcs = _celestial_wcs(ref if coordinate_image is None else coordinate_image)
        self.sky = wcs.pixel_to_world(*self.reference.T) if wcs is not None else None

    @classmethod
    def from_native_frame(cls, ref, image, required):
        """Transfer shared IDs once using a registered, unresampled native image."""
        if image.discard or getattr(image, "transform", None) is None:
            return None
        predicted = image.transform.inverse(ref.sources.coords)
        det = image.sources.coords
        if (
            not len(det)
            or not np.isfinite(predicted).all()
            or not np.isfinite(det).all()
        ):
            return None
        width = getattr(ref, "fwhm", np.nan)
        tolerance = max(3.0, width) if np.isfinite(width) else 3.0
        distances, ids, _, unique = _associations(predicted, det, tolerance)
        # Demand independent detections at three well-aligned reference positions
        # before transferring IDs, and a unique association for required sources.
        if (
            np.count_nonzero(unique & (distances <= max(3.0, tolerance / 2))) < 3
            or not unique[required].all()
        ):
            return None
        # Off-chip and undetected IDs retain their geometric predictions. Every
        # exposure must still supply an independent detection for a used ID.
        coords = predicted.copy()
        coords[unique] = det[ids[unique]]
        return cls(ref, coords=coords, coordinate_image=image)

    def run(self, image):
        det = np.asarray(image.sources.coords, dtype=float).reshape(-1, 2)
        det = det[np.isfinite(det).all(axis=1)]
        expected = self.reference.copy()
        wcs = _celestial_wcs(image)
        if self.sky is not None and wcs is not None:
            expected = np.column_stack(wcs.world_to_pixel(self.sky))
        finite = np.isfinite(expected).all(axis=1)
        # A failed science-frame PSF fit can inflate FWHM by an order of
        # magnitude. Bound it by the reference width before matching.
        widths = np.asarray([self.reference_fwhm, image.fwhm], dtype=float)
        widths = widths[np.isfinite(widths) & (widths > 0)]
        width = float(widths.min()) if widths.size else 0.0
        tolerance = max(3.0, 0.5 * width)
        matched = np.full_like(expected, np.nan)
        image.set("position_matches", matched)
        image.set("position_detections", det)
        image.set("position_tolerance", max(3.0, width))
        checks = np.full((len(expected), 5), np.nan)
        checks[:, 0] = 1 if len(det) else 2
        checks[:, 3:] = [tolerance, image.position_tolerance]
        image.set("position_checks", checks)
        image.set("position_candidates", np.full_like(expected, np.nan))
        if not finite.any() or not len(det):
            return
        predicted = expected[finite]
        shift = _field_shift(predicted, det, tolerance)
        if shift is None:
            if self.sky is None or wcs is None:
                return
        else:
            predicted = predicted + shift

        # Reject blends and competing IDs. A detection within tolerance of
        # two reference positions cannot establish either star's identity.
        distance, indices, close, unique = _associations(predicted, det, tolerance)
        checks[finite, 0] = np.where(unique, 0, np.where(close, 3, 2))
        checks[finite, 1] = distance
        image.position_candidates[finite] = det[indices]
        matched[np.flatnonzero(finite)[unique]] = det[indices[unique]]


def position_checks(image):
    """Finalize one diagnostic row per source using the actual aperture centers."""
    checks = image.position_checks.copy()
    centers = image.sources.coords
    checks[:, 2] = np.linalg.norm(centers - image.position_candidates, axis=1)
    matched = checks[:, 0] == 0
    finite = np.isfinite(centers).all(axis=1)
    checks[matched & ~finite, 0] = 4
    compare = matched & finite
    if compare.any():
        distances, nearest = cKDTree(image.position_detections).query(
            centers[compare], k=2
        )
        same = (distances[:, 0] < distances[:, 1]) & (
            image.position_detections[nearest[:, 0]] == image.position_matches[compare]
        ).all(axis=1)
        checks[compare, 0] = np.where(
            ~same, 5, np.where(distances[:, 0] > image.position_tolerance, 6, 0)
        )
    return checks


def verified_positions(image):
    return position_checks(image)[:, 0] == 0
