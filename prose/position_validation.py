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


class MatchExposureSources(Block):
    """Record independently identified detections before alignment replaces them."""

    def __init__(self, ref):
        super().__init__()
        self._parallel_friendly = True
        self.reference = np.array(ref.sources.coords, dtype=float)
        self.worker_shared_arrays = (self.reference,)
        self.reference_fwhm = getattr(ref, "fwhm", np.nan)
        wcs = _celestial_wcs(ref)
        self.sky = wcs.pixel_to_world(*self.reference.T) if wcs is not None else None

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
        distances, indices = cKDTree(det).query(predicted, k=2)
        close = distances[:, 0] <= tolerance
        reverse, _ = cKDTree(predicted).query(det, k=2)
        unique = (
            close
            & (distances[:, 1] > tolerance)
            & (reverse[indices[:, 0], 1] > tolerance)
        )
        matched[np.flatnonzero(finite)[unique]] = det[indices[unique, 0]]


def verified_positions(image):
    """Compare the final photometry centers with independent frame detections."""
    centers = image.sources.coords
    matches = image.position_matches
    valid = np.isfinite(centers).all(axis=1) & np.isfinite(matches).all(axis=1)
    if valid.any():
        # Refining a centroid must not move the aperture onto another detection.
        tree = cKDTree(image.position_detections)
        distances, nearest = tree.query(centers[valid], k=2)
        valid[valid] = (
            (distances[:, 0] <= image.position_tolerance)
            & (distances[:, 0] < distances[:, 1])
            & (image.position_detections[nearest[:, 0]] == matches[valid]).all(axis=1)
        )
    return valid
