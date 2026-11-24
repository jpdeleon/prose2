"""Translation-only alignment fallback for sparse fields without celestial WCS.

When twirl cannot solve (fewer than four reference stars, or fewer than five
detections in a frame) and the frames carry no celestial WCS, the frame is
aligned by a translation that at least two stars agree on. Anything less
certain is discarded rather than accepted with a guessed shift.
"""

import numpy as np
import pytest

from prose import Image, blocks
from prose.blocks.geometry import match_translation
from prose.core import Sources

REF_COORDS = np.array([[120.0, 130.0], [240.0, 280.0], [320.0, 160.0]])
OFFSET = np.array([6.5, -3.25])


def _image(coords):
    return Image(np.zeros((400, 400)), _sources=Sources(np.asarray(coords, float)))


def _align(ref_coords, frame_coords, n=None):
    ref = _image(ref_coords)
    frame = _image(frame_coords)
    blocks.ComputeTransformTwirl(ref, n=n or len(ref_coords)).run(frame)
    return frame


def test_match_translation_recovers_shift_from_reordered_stars():
    # Arrange
    frame = (REF_COORDS + OFFSET)[::-1]

    # Act
    shift = match_translation(REF_COORDS, frame)

    # Assert: the shift maps frame pixels onto reference pixels
    np.testing.assert_allclose(shift, -OFFSET, atol=1e-9)


def test_match_translation_returns_none_without_agreement():
    """The old xyshift returned (0, 0) here; no agreement must mean no shift."""
    frame = np.array([[10.0, 10.0], [200.0, 50.0], [60.0, 390.0]])

    assert match_translation(REF_COORDS, frame) is None


def test_match_translation_requires_two_agreeing_stars():
    # Only one star of the frame is a real star: any single pair "agrees".
    frame = np.array([REF_COORDS[0] + OFFSET, [5.0, 5.0], [390.0, 390.0]])

    assert match_translation(REF_COORDS, frame) is None


def test_match_translation_rejects_ambiguous_shifts():
    # Evenly spaced stars: shifts of +5 and -5 px each match two stars.
    ref = np.array([[100.0, 100.0], [110.0, 100.0], [120.0, 100.0]])
    frame = np.array([[105.0, 100.0], [115.0, 100.0]])

    assert match_translation(ref, frame) is None


def test_match_translation_averages_matched_offsets():
    rng = np.random.default_rng(0)
    frame = REF_COORDS + OFFSET + rng.normal(0, 0.2, REF_COORDS.shape)

    shift = match_translation(REF_COORDS, frame)

    np.testing.assert_allclose(shift, -(frame - REF_COORDS).mean(axis=0), atol=1e-9)


def test_three_star_frame_without_wcs_is_aligned_in_reference_order():
    frame = _align(REF_COORDS, (REF_COORDS + OFFSET)[::-1])
    blocks.AlignReferenceSources(_image(REF_COORDS)).run(frame)

    assert not frame.discard
    np.testing.assert_allclose(frame.transform.translation, -OFFSET, atol=1e-9)
    np.testing.assert_allclose(frame.sources.coords, REF_COORDS + OFFSET, atol=1e-6)


def test_frame_that_lost_a_star_and_gained_a_hot_pixel_is_aligned():
    frame_coords = np.vstack([(REF_COORDS + OFFSET)[:2], [[30.0, 370.0]]])

    frame = _align(REF_COORDS, frame_coords)

    assert not frame.discard
    np.testing.assert_allclose(frame.transform.translation, -OFFSET, atol=1e-9)


def test_frame_with_two_detections_is_aligned():
    frame = _align(REF_COORDS, (REF_COORDS + OFFSET)[:2])

    assert not frame.discard
    np.testing.assert_allclose(frame.transform.translation, -OFFSET, atol=1e-9)


@pytest.mark.parametrize(
    "frame_coords",
    [
        pytest.param([[10.0, 10.0], [200.0, 50.0], [60.0, 390.0]], id="no-agreement"),
        pytest.param([[126.5, 126.75]], id="single-detection"),
        pytest.param(np.empty((0, 2)), id="no-detections"),
    ],
)
def test_unverifiable_frame_is_discarded(frame_coords):
    frame = _align(REF_COORDS, frame_coords)

    assert frame.discard is True


def test_sparse_frame_of_a_rich_reference_falls_back_to_translation():
    """A frame with fewer than five detections skips twirl even when the
    reference is rich enough for quads (e.g. a cloudy frame)."""
    rng = np.random.default_rng(3)
    ref_coords = rng.uniform(20, 380, (12, 2))
    frame_coords = (ref_coords + OFFSET)[[0, 3, 7]]

    frame = _align(ref_coords, frame_coords, n=12)

    assert not frame.discard
    np.testing.assert_allclose(frame.transform.translation, -OFFSET, atol=1e-9)


def test_failing_twirl_and_wcs_fall_through_to_translation(monkeypatch):
    """An exception in an earlier strategy must not discard an alignable frame."""
    from types import SimpleNamespace

    rng = np.random.default_rng(5)
    ref_coords = rng.uniform(20, 380, (8, 2))
    frame = _image(ref_coords + OFFSET)
    block = blocks.ComputeTransformTwirl(_image(ref_coords), n=8)

    def boom(*args, **kwargs):
        raise RuntimeError("solver failure")

    # Celestial-looking WCS objects whose pixel_to_world is missing (raises).
    broken_wcs = SimpleNamespace(has_celestial=True)
    block.reference_image = SimpleNamespace(data=None, wcs=broken_wcs)
    monkeypatch.setattr(Image, "wcs", property(lambda self: broken_wcs))
    monkeypatch.setattr(block, "solve", boom)
    block.run(frame)

    assert not frame.discard
    np.testing.assert_allclose(frame.transform.translation, -OFFSET, atol=1e-9)


def test_celestial_wcs_takes_precedence_over_translation():
    from astropy.wcs import WCS

    ref = _image(REF_COORDS)
    frame = _image((REF_COORDS + OFFSET)[::-1])
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = [-0.001, 0.001]
    ref.wcs = wcs
    frame.wcs = wcs.deepcopy()
    # The WCS says the frame is rotated, which a pure translation cannot express.
    frame.wcs.wcs.pc = [[np.cos(0.01), -np.sin(0.01)], [np.sin(0.01), np.cos(0.01)]]

    blocks.ComputeTransformTwirl(ref, n=3).run(frame)

    assert not frame.discard
    # A translation has zero rotation; the WCS path recovers the 0.01 rad.
    assert abs(frame.transform.rotation) == pytest.approx(0.01, abs=1e-6)
