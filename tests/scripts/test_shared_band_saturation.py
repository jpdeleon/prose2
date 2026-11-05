"""Shared catalogs must not transfer one band's saturation decisions to another."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from prose import Fluxes, Image
from prose.core.source import Sources
from prose.scripts import run_photometry as rp


COORDS = np.array(
    [[20, 20], [50, 20], [80, 25], [25, 65], [60, 55], [80, 80]], dtype=float
)


def _frames(tmp_path, limit=800, unit="electrons", offset=(2, 3)):
    yy, xx = np.indices((110, 110))
    data = np.ones((110, 110)) * 10
    for (x, y), peak in zip(COORDS + offset, [2000, 900, 500, 400, 300, 200]):
        data += peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4)
    header = fits.Header(
        dict(TELID="2m0a", SITEID="ogg", FILTER="ip", SATURATE=limit, BUNIT=unit)
    )
    path = tmp_path / "own.fits"
    fits.writeto(path, data, header, overwrite=True)
    ref = Image(data=np.zeros_like(data))
    ref.sources = Sources(COORDS)
    return path, ref


def _screen(path, ref, radius=4):
    return rp._band_saturation_indices(
        [path],
        ref,
        0,
        radius,
        ccd_trim_size_yx=(0, 0),
        max_num_stars=6,
        min_star_separation=5,
        min_area=3,
        bad_pixel_map=None,
    )


@pytest.mark.parametrize("unit", ["electrons", "ADU"])
def test_native_band_limit_preserves_shared_ids_and_target(tmp_path, unit, caplog):
    path, ref = _frames(tmp_path, limit=1200, unit=unit)
    assert _screen(path, ref) == []  # g-like limit: comparison 1 is safe
    fits.setval(path, "SATURATE", value=800)
    assert _screen(path, ref) == [1]  # same shared ID is saturated in i
    np.testing.assert_array_equal(ref.sources.coords, COORDS)
    assert "target 0 is saturated" in caplog.text


def test_peripheral_saturated_pixel_inside_aperture_is_screened(tmp_path):
    path, ref = _frames(tmp_path)
    # Saturation on a defocused ring, outside the old 11x11 peak box.
    with fits.open(path, mode="update") as hdus:
        hdus[0].data[58, 69] = 1000  # source 4 at (62,58), radius 7
    assert _screen(path, ref, radius=8) == [1, 4]


def test_unknown_limit_does_not_borrow_reference_threshold(tmp_path, caplog):
    path, ref = _frames(tmp_path)
    fits.delval(path, "SATURATE")
    ref.telescope = SimpleNamespace(saturation=100)
    assert _screen(path, ref) == []
    assert "per-band saturation unknown" in caplog.text


def test_failed_alignment_stops_screening(tmp_path, monkeypatch):
    path, ref = _frames(tmp_path)

    def fail(image):
        image.discard = True

    monkeypatch.setattr(
        rp.blocks, "ComputeTransformTwirl", lambda *a, **k: SimpleNamespace(run=fail)
    )
    with pytest.raises(ValueError, match="alignment failed"):
        _screen(path, ref)


@pytest.mark.parametrize("explicit", [None, [1, 2], [1]])
def test_run_band_replaces_shared_mask_for_auto_and_explicit_comparisons(
    tmp_path, monkeypatch, explicit
):
    path, ref = _frames(tmp_path)
    ref.telescope = SimpleNamespace(jd_scale="jd")
    reference = dict(
        ref=ref,
        target_index=0,
        aper_radii=[4],
        rin=8,
        rout=12,
        scale=False,
        saturated_cids=[2],
    )
    monkeypatch.setattr(rp, "build_reference", lambda *a, **k: reference)
    phot = SimpleNamespace(
        run=lambda files: None, data=[SimpleNamespace(fluxes=Fluxes(np.ones((6, 4))))]
    )
    monkeypatch.setattr(rp, "photometry_sequence", lambda *a, **k: phot)
    captured = {}

    def diff(fluxes, target_index, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(time=np.arange(4, dtype=float))

    monkeypatch.setattr(rp, "differential_photometry", diff)
    result = rp.run_band(
        "ip",
        [path],
        path,
        None,
        cids=explicit,
        shared_reference=True,
        max_num_stars=6,
        min_area=3,
        min_star_separation=5,
    )
    assert captured["avoid_cids"] == [1]
    assert captured["cids"] == ([2] if explicit == [1, 2] else None)
    assert result["saturated_cids"] == [1]


def test_all_comparisons_saturated_is_not_relaxed(tmp_path, monkeypatch):
    path, ref = _frames(tmp_path, limit=100)
    monkeypatch.setattr(
        rp,
        "build_reference",
        lambda *a, **k: dict(ref=ref, target_index=0, aper_radii=[4], scale=False),
    )
    with pytest.raises(ValueError, match="no usable comparison stars"):
        rp.run_band(
            "ip",
            [path],
            path,
            None,
            shared_reference=True,
            max_num_stars=6,
            min_area=3,
            min_star_separation=5,
        )
