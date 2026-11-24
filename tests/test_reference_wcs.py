"""Reference WCS verification against Gaia and pair-matching refit (#95)."""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from prose.reference_wcs import (
    WcsCheck,
    check_wcs,
    solve_wcs,
    verify_reference_wcs,
)

SHAPE = (1024, 1024)
SCALE_DEG = 0.44 / 3600  # MuSCAT2
TARGET = SkyCoord(117.026769, 50.225147, unit="deg")


def _true_wcs(rotation_deg=12.0, flip=False, crpix=(512.0, 512.0)):
    theta = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    parity = np.diag([-1.0, 1.0]) if not flip else np.eye(2)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [TARGET.ra.deg, TARGET.dec.deg]
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.cd = SCALE_DEG * rot @ parity
    return wcs


def _field(wcs, n_gaia=200, n_detected=15, seed=0, noise=0.3):
    """Gaia stars (brightest first) around the target, and the brightest
    *n_detected* of those inside the frame as detections."""
    rng = np.random.default_rng(seed)
    radius = 1.3 * SHAPE[0] * SCALE_DEG
    ra = TARGET.ra.deg + rng.uniform(-radius, radius, n_gaia) / np.cos(
        np.deg2rad(TARGET.dec.deg)
    )
    dec = TARGET.dec.deg + rng.uniform(-radius, radius, n_gaia)
    gaia = SkyCoord(ra, dec, unit="deg")
    x, y = wcs.world_to_pixel(gaia)
    xy = np.column_stack([x, y])
    inside = (xy >= 10).all(axis=1) & (xy <= SHAPE[0] - 10).all(axis=1)
    detected = xy[inside][:n_detected] + rng.normal(0, noise, (n_detected, 2))
    return gaia, detected


def _target_pixel(wcs):
    return np.array([float(v) for v in wcs.world_to_pixel(TARGET)])


def _shifted(wcs, dx, dy):
    copy = wcs.deepcopy()
    copy.wcs.crpix = [wcs.wcs.crpix[0] - dx, wcs.wcs.crpix[1] - dy]
    return copy


# --- check -----------------------------------------------------------------


def test_correct_wcs_passes_the_gaia_check():
    wcs = _true_wcs()
    gaia, detected = _field(wcs)

    check = check_wcs(wcs, detected, gaia, SHAPE)

    assert check.passed
    assert check.n_matched == len(detected)


def test_copied_wcs_off_by_the_drift_fails_the_gaia_check():
    """#94: the reference frame drifted (38.8, 125.0) px from the solved frame."""
    wcs = _true_wcs()
    gaia, detected = _field(wcs)

    check = check_wcs(_shifted(wcs, 38.8, 125.0), detected, gaia, SHAPE)

    assert not check.passed


def test_missing_celestial_wcs_fails_the_check():
    gaia, detected = _field(_true_wcs())

    assert not check_wcs(WCS(naxis=2), detected, gaia, SHAPE).passed


@pytest.mark.parametrize(
    ("n_matched", "n_detections", "passed"),
    [(8, 15, True), (7, 15, False), (2, 2, True), (1, 1, False), (2, 3, True)],
)
def test_check_requires_half_the_detections_and_at_least_two(
    n_matched, n_detections, passed
):
    assert WcsCheck(n_matched, n_detections).passed is passed


# --- solve -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("rotation_deg", "flip"), [(12.0, False), (-97.0, False), (180.0, True)]
)
def test_solve_recovers_the_target_pixel(rotation_deg, flip):
    wcs = _true_wcs(rotation_deg=rotation_deg, flip=flip, crpix=(430.0, 610.0))
    gaia, detected = _field(wcs, seed=1)

    solved = solve_wcs(detected, gaia, TARGET, SCALE_DEG, SHAPE)

    assert solved is not None
    np.testing.assert_allclose(_target_pixel(solved), _target_pixel(wcs), atol=1.0)


def test_solve_tolerates_spurious_detections_in_a_sparse_field():
    """Like TOI00782.01: a dozen detections, only about half are Gaia stars."""
    wcs = _true_wcs(rotation_deg=33.0)
    gaia, detected = _field(wcs, n_detected=6, seed=2)
    rng = np.random.default_rng(9)
    spurious = rng.uniform(20, SHAPE[0] - 20, (6, 2))
    mixed = np.vstack([detected[:3], spurious[:3], detected[3:], spurious[3:]])

    solved = solve_wcs(mixed, gaia, TARGET, SCALE_DEG, SHAPE)

    assert solved is not None
    np.testing.assert_allclose(_target_pixel(solved), _target_pixel(wcs), atol=1.0)


def test_solve_gives_up_with_too_few_stars():
    wcs = _true_wcs()
    gaia, detected = _field(wcs, n_detected=3)

    assert solve_wcs(detected, gaia, TARGET, SCALE_DEG, SHAPE) is None


def test_solve_gives_up_when_nothing_matches():
    gaia, _ = _field(_true_wcs())
    rng = np.random.default_rng(4)
    noise = rng.uniform(0, SHAPE[0], (12, 2))

    assert solve_wcs(noise, gaia, TARGET, SCALE_DEG, SHAPE) is None


# --- verify ----------------------------------------------------------------


def _verify(header_wcs, detected, gaia):
    return verify_reference_wcs(header_wcs, detected, gaia, TARGET, SCALE_DEG, SHAPE)


def test_header_wcs_that_agrees_with_gaia_is_kept():
    wcs = _true_wcs()
    gaia, detected = _field(wcs)

    verdict = _verify(wcs, detected, gaia)

    assert verdict.source == "header"
    assert verdict.trusted
    assert verdict.wcs is wcs


def test_drifted_header_wcs_is_replaced_by_the_gaia_refit():
    wcs = _true_wcs()
    gaia, detected = _field(wcs)

    verdict = _verify(_shifted(wcs, 38.8, 125.0), detected, gaia)

    assert verdict.source == "gaia_refit"
    assert verdict.trusted
    np.testing.assert_allclose(_target_pixel(verdict.wcs), _target_pixel(wcs), atol=1.0)
    assert "refit" in verdict.summary()


def test_frame_without_a_wcs_is_solved_from_gaia():
    wcs = _true_wcs(rotation_deg=-40.0)
    gaia, detected = _field(wcs, seed=5)

    verdict = _verify(WCS(naxis=2), detected, gaia)

    assert verdict.source == "gaia_refit"
    np.testing.assert_allclose(_target_pixel(verdict.wcs), _target_pixel(wcs), atol=1.0)


def test_unsolvable_frame_keeps_the_header_wcs_untrusted():
    wcs = _true_wcs()
    gaia, _ = _field(wcs)
    noise = np.random.default_rng(6).uniform(0, SHAPE[0], (12, 2))

    verdict = _verify(wcs, noise, gaia)

    assert verdict.wcs is wcs
    assert not verdict.trusted


def test_without_gaia_the_header_wcs_is_unverified():
    wcs = _true_wcs()
    _, detected = _field(wcs)

    verdict = _verify(wcs, detected, None)

    assert verdict.source == "unverified"
    assert not verdict.trusted
    assert verdict.wcs is wcs


def test_single_detection_cannot_verify_the_wcs():
    wcs = _true_wcs()
    gaia, detected = _field(wcs, n_detected=1)

    verdict = _verify(wcs, detected, gaia)

    assert verdict.source == "unverified"
    assert not verdict.trusted


@pytest.mark.parametrize(
    "offsets",
    [
        [(100, 20), (-100, 20), (37, -90), (-37, -90)],  # mirror symmetry
        [(80, 20), (-80, -20), (37, -90), (-37, 90)],  # half-turn symmetry
    ],
)
def test_tied_orientations_about_the_same_target_are_ambiguous(offsets):
    wcs = _true_wcs()
    detected = np.asarray(offsets) + _target_pixel(wcs)
    gaia = wcs.pixel_to_world(*detected.T)
    verdict = _verify(None, detected, gaia)
    assert not verdict.trusted
    assert verdict.wcs is None
