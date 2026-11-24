"""An undetected target is added at its position under a trusted WCS (#96).

When no detection lies within 5 arcsec of the target but the reference WCS
passed the Gaia check (header or refit), the target is added at its WCS
pixel, as ``--tID <kept count>`` does by hand. An untrusted WCS still fails
the band, since it would place the aperture on blank sky or another star.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from prose.reference_wcs import WcsCheck, WcsVerdict
from prose.scripts import run_photometry as rp
from tests.scripts.test_run_photometry import _patch_ref_seq, _write_minimal_fits

RADII = dict(aper_radii=np.array([3.0, 4.0, 5.0]), rin=8.0, rout=12.0)


def _verdict(trusted):
    return WcsVerdict(
        wcs=None,
        source="header" if trusted else "unverified",
        trusted=trusted,
        header_check=WcsCheck(2 if trusted else 0, 2),
    )


@pytest.fixture
def frame(tmp_path, monkeypatch):
    """A 20x20 frame with detections at (5, 5) and (10, 10)."""
    _patch_ref_seq(monkeypatch)
    return _write_minimal_fits(tmp_path)


def _coord_at(fpath, x, y):
    return WCS(fits.getheader(fpath)).pixel_to_world(x, y)


def _use_verdict(monkeypatch, trusted):
    monkeypatch.setattr(rp, "check_reference_wcs", lambda *a, **k: _verdict(trusted))


def test_undetected_target_is_added_under_a_trusted_wcs(frame, monkeypatch):
    _use_verdict(monkeypatch, trusted=True)

    result = rp.build_reference(frame, _coord_at(frame, 15.0, 4.0), **RADII)

    assert result["identification_method"] == "added_at_trusted_wcs"
    assert result["target_index"] == 2
    assert len(result["ref"].sources) == 3


def test_untrusted_wcs_still_fails_the_band(frame, monkeypatch):
    _use_verdict(monkeypatch, trusted=False)

    with pytest.raises(rp.TargetNotFoundError):
        rp.build_reference(frame, _coord_at(frame, 15.0, 4.0), **RADII)


def test_target_outside_the_frame_is_not_added(frame, monkeypatch):
    _use_verdict(monkeypatch, trusted=True)

    with pytest.raises(rp.TargetNotFoundError):
        rp.build_reference(frame, _coord_at(frame, 60.0, 60.0), **RADII)


def test_detected_target_is_matched_not_added(frame, monkeypatch):
    _use_verdict(monkeypatch, trusted=True)

    result = rp.build_reference(frame, _coord_at(frame, 10.0, 10.0), **RADII)

    assert result["identification_method"] == "wcs_match"
    assert result["target_index"] == 1


def test_trusted_wcs_takes_precedence_over_another_bands_pixel(frame, monkeypatch):
    """The other-band fallback picks the nearest detection, which for an
    undetected target is a different star."""
    _use_verdict(monkeypatch, trusted=True)

    result = rp.build_reference(
        frame,
        _coord_at(frame, 15.0, 4.0),
        target_pixel_fallback=(10.0, 10.0),
        **RADII,
    )

    assert result["identification_method"] == "added_at_trusted_wcs"


@pytest.mark.parametrize(
    ("xy", "inside"),
    [
        ((15.0, 4.0), True),
        ((-0.4, 19.4), True),
        ((20.0, 5.0), False),
        ((5.0, -1.0), False),
    ],
)
def test_trusted_target_pixel_requires_a_position_inside_the_frame(frame, xy, inside):
    ref = rp.FITSImage(frame)
    pixel = rp.trusted_target_pixel(ref, _coord_at(frame, *xy), _verdict(True))

    assert (pixel is not None) is inside


def test_trusted_target_pixel_needs_a_verdict(frame):
    ref = rp.FITSImage(frame)

    assert rp.trusted_target_pixel(ref, _coord_at(frame, 15.0, 4.0), None) is None


@pytest.mark.parametrize("flag", [0, "0", 4, None])
def test_banzai_status_controls_matching_and_band_order(frame, monkeypatch, flag):
    if flag is not None:
        fits.setval(frame, "WCSERR", value=flag)
    monkeypatch.setattr(rp, "_gaia_field_coords", lambda *a: None)
    target = _coord_at(frame, 10, 10)
    assert rp._reference_file_has_usable_wcs(frame, target) == (flag != 4)
    if flag == 4:
        with pytest.raises(rp.TargetNotFoundError):
            rp.build_reference(frame, target, **RADII)
    else:
        result = rp.build_reference(frame, target, **RADII)
        assert result["identification_method"] == "wcs_match"

    faint = _coord_at(frame, 15, 4)
    if flag in (0, "0"):
        result = rp.build_reference(frame, faint, **RADII)
        assert result["identification_method"] == "added_at_trusted_wcs"
        np.testing.assert_allclose(result["ref"].sources[2].coords, [15, 4], atol=1e-6)
    else:
        with pytest.raises(rp.TargetNotFoundError):
            rp.build_reference(frame, faint, **RADII)


def test_gaia_contradiction_overrides_banzai_success(frame, monkeypatch):
    fits.setval(frame, "WCSERR", value=0)
    _use_verdict(monkeypatch, trusted=False)
    with pytest.raises(rp.TargetNotFoundError):
        rp.build_reference(frame, _coord_at(frame, 10, 10), **RADII)


def test_failed_header_allows_independent_gaia_refit(frame, monkeypatch):
    sky = _coord_at(frame, 15, 4)
    solved = WCS(fits.getheader(frame))
    fits.setval(frame, "WCSERR", value=4)

    def refit(header_wcs, *args):
        assert header_wcs is None
        return WcsVerdict(solved, "gaia_refit", True)

    monkeypatch.setattr(rp, "_gaia_field_coords", lambda *a: None)
    monkeypatch.setattr(rp, "verify_reference_wcs", refit)
    result = rp.build_reference(frame, sky, **RADII)
    assert result["identification_method"] == "added_at_trusted_wcs"
    np.testing.assert_allclose(result["ref"].sources[2].coords, [15, 4], atol=1e-6)
