"""A WCS-derived frame transform is used only when the detected stars agree.

MuSCAT/MuSCAT2 calibration solves one WCS per band and writes it into every
frame, although the pointing drifts during the night. Such a copied WCS maps
each frame onto the reference with a near-identity transform. The alignment
block must reject it when the frame's stars contradict it and fall through to
the star-verified translation.
"""

import numpy as np
import pytest
from astropy.wcs import WCS

from prose import Image, blocks
from prose.core import Sources

REF_COORDS = np.array([[120.0, 130.0], [240.0, 280.0], [320.0, 160.0]])
PIXEL_SCALE_DEG = 0.435 / 3600  # MuSCAT2


def _tan_wcs(crpix=(200.0, 200.0), rotation=0.0):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 20.0]
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.cdelt = [-PIXEL_SCALE_DEG, PIXEL_SCALE_DEG]
    c, s = np.cos(rotation), np.sin(rotation)
    wcs.wcs.pc = [[c, -s], [s, c]]
    return wcs


def _image(coords, wcs=None):
    img = Image(np.zeros((400, 400)), _sources=Sources(np.asarray(coords, float)))
    if wcs is not None:
        img.wcs = wcs
    return img


def _stars_seen_through(frame_wcs, ref_wcs, ref_coords=REF_COORDS):
    """Pixel positions of the reference stars in a frame with *frame_wcs*."""
    sky = ref_wcs.pixel_to_world(ref_coords[:, 0], ref_coords[:, 1])
    x, y = frame_wcs.world_to_pixel(sky)
    return np.column_stack([x, y])


def _run(ref_coords, ref_wcs, frame_coords, frame_wcs):
    ref = _image(ref_coords, ref_wcs)
    frame = _image(frame_coords, frame_wcs)
    blocks.ComputeTransformTwirl(ref, n=len(ref_coords)).run(frame)
    return frame


def test_copied_wcs_is_rejected_in_favour_of_the_star_shift():
    # Arrange: the telescope drifted 4 px, but the frame carries the reference's WCS.
    drift = np.array([4.0, 1.0])
    wcs = _tan_wcs()

    # Act
    frame = _run(REF_COORDS, wcs, REF_COORDS + drift, wcs.deepcopy())

    # Assert: the copied WCS would give zero translation; the stars give -drift.
    assert not frame.discard
    np.testing.assert_allclose(frame.transform.translation, -drift, atol=1e-6)


def test_wcs_consistent_with_the_stars_is_used():
    ref_wcs = _tan_wcs()
    frame_wcs = _tan_wcs(crpix=(206.0, 197.0), rotation=0.004)

    frame = _run(
        REF_COORDS, ref_wcs, _stars_seen_through(frame_wcs, ref_wcs), frame_wcs
    )

    assert not frame.discard
    # Only the WCS strategy can recover a rotation.
    assert abs(frame.transform.rotation) == pytest.approx(0.004, abs=1e-6)


def test_wcs_is_kept_when_the_frame_has_too_few_stars_to_check_it():
    ref_wcs = _tan_wcs()
    frame_wcs = _tan_wcs(crpix=(206.0, 197.0))
    one_star = _stars_seen_through(frame_wcs, ref_wcs)[:1]

    frame = _run(REF_COORDS, ref_wcs, one_star, frame_wcs)

    assert not frame.discard
    np.testing.assert_allclose(frame.transform.translation, [-6.0, 3.0], atol=1e-6)


def test_contradicted_wcs_without_a_verified_shift_discards_the_frame():
    wcs = _tan_wcs()
    unrelated = np.array([[15.0, 20.0], [380.0, 40.0], [60.0, 370.0]])

    frame = _run(REF_COORDS, wcs, unrelated, wcs.deepcopy())

    assert frame.discard is True


def test_wcs_within_the_match_tolerance_is_accepted():
    # Independent solutions differ slightly; 1 px disagreement is not a contradiction.
    ref_wcs = _tan_wcs()
    frame_wcs = _tan_wcs(crpix=(206.0, 197.0), rotation=0.004)
    stars = _stars_seen_through(frame_wcs, ref_wcs) + np.array([1.0, -0.5])

    frame = _run(REF_COORDS, ref_wcs, stars, frame_wcs)

    assert abs(frame.transform.rotation) == pytest.approx(0.004, abs=1e-6)
