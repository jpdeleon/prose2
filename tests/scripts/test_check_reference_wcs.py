"""run_photometry replaces a reference WCS that Gaia contradicts (#94, #95)."""

import numpy as np
import pytest
from astropy.wcs import WCS

from prose import Image
from prose.core import Sources
from prose.scripts import run_photometry as rp
from tests.test_reference_wcs import (
    SCALE_DEG,
    SHAPE,
    TARGET,
    _field,
    _shifted,
    _target_pixel,
    _true_wcs,
)

PIXEL_SCALE_ARCSEC = SCALE_DEG * 3600


def _reference(detected, wcs):
    ref = Image(np.zeros(SHAPE), _sources=Sources(detected))
    if wcs is not None:
        ref.wcs = wcs
    return ref


@pytest.fixture
def field(monkeypatch):
    wcs = _true_wcs()
    gaia, detected = _field(wcs)
    monkeypatch.setattr(rp, "_gaia_field_coords", lambda *a, **k: gaia)
    return wcs, detected


def test_drifted_reference_wcs_is_replaced_on_the_image(field):
    true_wcs, detected = field
    ref = _reference(detected, _shifted(true_wcs, 38.8, 125.0))

    verdict = rp.check_reference_wcs(ref, TARGET, PIXEL_SCALE_ARCSEC, "ref.fits")

    assert verdict.source == "gaia_refit"
    np.testing.assert_allclose(
        _target_pixel(ref.wcs), _target_pixel(true_wcs), atol=1.0
    )


def test_reference_without_wcs_gets_the_gaia_solution(field):
    true_wcs, detected = field
    ref = _reference(detected, None)

    rp.check_reference_wcs(ref, TARGET, PIXEL_SCALE_ARCSEC, "ref.fits")

    np.testing.assert_allclose(
        _target_pixel(ref.wcs), _target_pixel(true_wcs), atol=1.0
    )


def test_correct_reference_wcs_is_left_alone(field):
    true_wcs, detected = field
    ref = _reference(detected, true_wcs)

    verdict = rp.check_reference_wcs(ref, TARGET, PIXEL_SCALE_ARCSEC, "ref.fits")

    assert verdict.source == "header"
    assert verdict.trusted
    np.testing.assert_allclose(
        _target_pixel(ref.wcs), _target_pixel(true_wcs), atol=1e-6
    )


def test_gaia_outage_leaves_the_header_wcs_untrusted(monkeypatch):
    wcs = _true_wcs()
    _, detected = _field(wcs)
    monkeypatch.setattr(rp, "_gaia_field_coords", lambda *a, **k: None)
    ref = _reference(detected, wcs)

    verdict = rp.check_reference_wcs(ref, TARGET, PIXEL_SCALE_ARCSEC, "ref.fits")

    assert verdict.source == "unverified"
    assert not verdict.trusted
    np.testing.assert_allclose(_target_pixel(ref.wcs), _target_pixel(wcs), atol=1e-6)


def test_failing_check_never_raises(monkeypatch):
    wcs = _true_wcs()
    _, detected = _field(wcs)

    def boom(*args, **kwargs):
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(rp, "_gaia_field_coords", boom)
    ref = _reference(detected, wcs)

    verdict = rp.check_reference_wcs(ref, TARGET, PIXEL_SCALE_ARCSEC, "ref.fits")

    assert not verdict.trusted
    assert "catalog exploded" in verdict.summary()


@pytest.mark.gaia_field  # the real _gaia_field_coords, with gaia_query faked
def test_gaia_field_query_is_cached_per_run(monkeypatch, tmp_path):
    calls = []

    class Table:
        def to_pandas(self):
            import pandas as pd

            return pd.DataFrame({"ra": [TARGET.ra.deg], "dec": [TARGET.dec.deg]})

    def fake_query(*args, **kwargs):
        calls.append(args)
        return Table()

    monkeypatch.setattr(rp, "gaia_query", fake_query)
    monkeypatch.setattr(rp, "_gaia_field_cache", {})
    monkeypatch.setattr(rp, "_gaia_df_at_epoch", lambda df, t: df)
    monkeypatch.setattr(
        rp, "coord_cache_path", lambda *a, **k: tmp_path / "gaia_field.csv"
    )
    ref = _reference(np.zeros((0, 2)), WCS(naxis=2))

    first = rp._gaia_field_coords(ref, TARGET, PIXEL_SCALE_ARCSEC)
    second = rp._gaia_field_coords(ref, TARGET, PIXEL_SCALE_ARCSEC)

    assert len(calls) == 1
    assert second is first
    # 1.5 frame diagonals across, centred on the target
    expected_fov = 1.5 * np.hypot(*SHAPE) * SCALE_DEG
    assert calls[0][1] == pytest.approx(expected_fov)


@pytest.mark.parametrize("fallback", [None, (200, 200), (5, 5)])
def test_rejected_header_cannot_identify_a_neighbour(tmp_path, monkeypatch, fallback):
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from tests.scripts.test_run_photometry import _patch_ref_seq, _write_minimal_fits

    _patch_ref_seq(monkeypatch)
    path = _write_minimal_fits(tmp_path)
    # Source 1 coincides with the target only under the contradicted header.
    target = WCS(fits.getheader(path)).pixel_to_world(10, 10)
    gaia = SkyCoord([50, 51], [20, 21], unit="deg")
    monkeypatch.setattr(rp, "_gaia_field_coords", lambda *args: gaia)
    kwargs = dict(
        aper_radii=np.array([3.0]), rin=8, rout=12, target_pixel_fallback=fallback
    )
    if fallback == (5, 5):
        result = rp.build_reference(path, target, **kwargs)
        assert result["identification_method"] == "other_band_pixel"
        assert result["target_index"] == 0
    else:
        with pytest.raises(rp.TargetNotFoundError, match="header 0/2") as error:
            rp.build_reference(path, target, **kwargs)
        assert len(error.value.ref.sources) == 2
