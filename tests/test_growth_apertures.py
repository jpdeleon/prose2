from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from prose import Image
from prose.core.source import PointSource, Sources
from prose.scripts import run_photometry as rp


def _field(sigma, ring=0):
    y, x = np.indices((240, 240))
    r = np.hypot(x - 120, y - 120)
    image = Image(3000 + 1500 * np.exp(-0.5 * ((r - ring) / sigma) ** 2))
    image.sources = Sources(
        [PointSource(coords=np.array([120, 120]), a=ring + 2 * sigma)]
    )
    return image


@pytest.mark.parametrize("sigma,ring,expected", [(12, 0, 37), (2, 12, 17)])
def test_growth_radius_and_fit_window(sigma, ring, expected):
    # For a circular Gaussian, r99 = sigma * sqrt(2 log 100).
    # A thin Gaussian ring encloses 99% just beyond its outer 2-sigma wing.
    image = _field(sigma, ring)
    assert rp._growth_radius(image, 35) == pytest.approx(expected, abs=1)
    rp.ProfileCutouts(shape=35).run(image)
    assert image.cutouts[0].data.shape[0] > 2 * expected


def test_native_widest_frame_controls_annulus(tmp_path, monkeypatch):
    ref = _field(6)
    ref.fwhm = 1800.0  # deliberately unreliable reference fit
    ref.header["PIXSCALE"] = 1.0
    ref.telescope = SimpleNamespace(pixel_scale=1)
    path = tmp_path / "wider.fits"
    fits.writeto(path, _field(12).data)
    widest = rp._widest_growth_radius(
        [path],
        "other-band.fits",
        35,
        (0, 0),
        None,
        rp.blocks.PointSourceDetection(n=5, min_area=10),
    )
    assert widest == pytest.approx(37, abs=1)
    monkeypatch.setattr(rp, "_gaia_catalog_df", lambda *a: None)
    r99 = rp._growth_radius(ref, 35)
    grid, rin, rout = rp.gaia_aperture_radii(ref, 0, None, r99=r99, widest=widest)
    assert rin >= widest + 2
    assert len(grid) > 2 and np.all(np.diff(grid) > 0)
    assert grid[0] < r99 < grid[-1] < rin < rout


def _reference(monkeypatch, neighbour_sep_arcsec=None):
    import pandas as pd
    from astropy.coordinates import SkyCoord

    ref = _field(6)
    ref.fwhm = 5.0
    ref.header["PIXSCALE"] = 1.0
    ref.telescope = SimpleNamespace(pixel_scale=1)
    target = SkyCoord(10.0, 0.0, unit="deg")
    dec = [0.0]
    if neighbour_sep_arcsec is not None:
        dec.append(neighbour_sep_arcsec / 3600)
    catalog = pd.DataFrame(
        {"ra": [10.0] * len(dec), "dec": dec, "phot_g_mean_mag": [12.0] * len(dec)}
    )
    monkeypatch.setattr(rp, "_gaia_catalog_df", lambda *a: catalog)
    return ref, target, rp._growth_radius(ref, 35)


def test_annulus_stays_inside_equal_brightness_neighbour(monkeypatch):
    # An equal-magnitude neighbour 40 px away contaminates; the ring must end
    # at least 2 px (CONTAM_MARGIN_PIX) before it yet start beyond r99 + 2.
    ref, target, r99 = _reference(monkeypatch, 40)
    grid, rin, rout = rp.gaia_aperture_radii(ref, 0, target, r99=r99)
    # The 40 px separation comes from spherical trig, so allow float rounding.
    assert rout <= 40 - 2 + 1e-9
    assert rin >= r99 + 2
    assert grid[-1] < rin


def test_close_neighbour_narrows_annulus(monkeypatch):
    # r99 (~18 px) + 2 + a 10 px ring cannot fit before a neighbour at 30 px,
    # but an 8 px ring (>= MIN_ANNULUS_WIDTH_PIX) between r99 + 2 and 28 px does.
    ref, target, r99 = _reference(monkeypatch, 30)
    grid, rin, rout = rp.gaia_aperture_radii(ref, 0, target, r99=r99)
    assert rin == pytest.approx(r99 + 2)
    assert rout <= 30 - 2 + 1e-9
    assert rout - rin >= rp.MIN_ANNULUS_WIDTH_PIX
    assert grid[-1] < rin


def test_no_room_for_growth_ring_falls_back_to_fwhm(monkeypatch):
    # A neighbour at 23 px leaves < MIN_ANNULUS_WIDTH_PIX outside r99 + 2, so
    # the band keeps the FWHM-based ring instead of failing.
    ref, target, r99 = _reference(monkeypatch, 23)
    grid, rin, rout = rp.gaia_aperture_radii(ref, 0, target, r99=r99)
    assert rout <= 23 - 2 + 1e-9
    assert rin < r99
    assert grid[0] == pytest.approx(ref.fwhm)


def test_neighbour_inside_fwhm_ring_still_raises(monkeypatch):
    ref, target, r99 = _reference(monkeypatch, 10)
    with pytest.raises(ValueError, match="uncontaminated sky annulus"):
        rp.gaia_aperture_radii(ref, 0, target, r99=r99)


def test_growth_sample_is_bounded_even_and_skips_reference():
    files = [f"f{i:03d}.fits" for i in range(100)]
    sample = rp._growth_sample(files, "f000.fits")
    assert len(sample) == rp.GROWTH_SAMPLE_FRAMES
    assert "f000.fits" not in sample
    assert sample[0] == "f001.fits" and sample[-1] == "f099.fits"
    assert rp._growth_sample(files[:5], "f000.fits") == files[1:5]


def test_single_wide_frame_does_not_set_widest(tmp_path):
    paths = []
    for i, sigma in enumerate([6] * 9 + [12]):
        path = tmp_path / f"frame{i}.fits"
        fits.writeto(path, _field(sigma).data)
        paths.append(path)
    widest = rp._widest_growth_radius(
        paths,
        "ref.fits",
        35,
        (0, 0),
        None,
        rp.blocks.PointSourceDetection(n=5, min_area=10),
    )
    # Nine ~18 px frames and one ~37 px outlier: p90 stays near the bulk.
    assert 17 <= widest < 25


def test_explicit_annulus_clearing_profile_is_kept(monkeypatch):
    ref, target, r99 = _reference(monkeypatch)
    _, rin, rout = rp.gaia_aperture_radii(
        ref, 0, target, annulus_pix=(25.0, 35.0), r99=r99
    )
    assert (rin, rout) == (25.0, 35.0)


def test_explicit_annulus_inside_profile_moves_outside_it(monkeypatch):
    ref, target, r99 = _reference(monkeypatch)
    _, rin, rout = rp.gaia_aperture_radii(
        ref, 0, target, annulus_pix=(10.0, 20.0), r99=r99
    )
    assert rin >= r99 + 2
    assert rout - rin == pytest.approx(10.0)
