"""Separation between the largest photometry aperture and the inner sky annulus."""

from types import SimpleNamespace

import numpy as np
import pytest

from prose.scripts import run_photometry as rp

BASE_ARGS = ["--target_name", "X", "--data_dir", "data", "--results_dir", "out"]
CUSTOM_GRID = ["--aper_radii", "5,15,5", "--annulus", "20,30"]


def _geometry(fwhm, contam, gap, annulus_pix=None):
    rin, rout = rp._sky_annulus_pix(
        fwhm, np.asarray(contam, dtype=float), annulus_pix=annulus_pix, gap=gap
    )
    return rp._aperture_radii_pix(fwhm, rin, gap=gap), rin, rout


def test_default_gap_exceeds_aperture_step():
    assert rp.ANNULUS_GAP_PIX > rp.APER_STEP_PIX


def test_default_gap_moves_nominal_ring_out_and_keeps_apertures():
    radii, rin, rout = _geometry(5.0, [], rp.ANNULUS_GAP_PIX)

    assert (rin, rout) == (20 + rp.ANNULUS_GAP_PIX, 30 + rp.ANNULUS_GAP_PIX)
    np.testing.assert_array_equal(radii, rp._aperture_radii_pix(5.0, 20.0))
    assert rin - radii.max() >= rp.ANNULUS_GAP_PIX


@pytest.mark.parametrize("gap", [0.0, 3.0, 8.0, 12.5])
def test_gap_is_honoured_without_contaminants(gap):
    radii, rin, _ = _geometry(5.0, [], gap)

    assert rin - radii.max() >= gap


def test_explicit_annulus_is_kept_and_apertures_shrink():
    radii, rin, rout = _geometry(5.0, [], 8.0, annulus_pix=(30, 50))

    assert (rin, rout) == (30.0, 50.0)
    assert rin - radii.max() >= 8.0


def test_contaminant_limited_ring_still_honours_gap():
    radii, rin, rout = _geometry(3.0, [30.0], 5.0)

    assert rout == 30.0 - rp.CONTAM_MARGIN_PIX
    assert rin - radii.max() >= 5.0


def test_gap_is_reduced_when_neighbour_leaves_no_room():
    # With a 5 px gap the ring cannot clear the neighbour; a smaller one can.
    radii, rin, rout = _geometry(5.0, [16.0], 5.0)

    assert rout <= 16.0 - rp.CONTAM_MARGIN_PIX
    np.testing.assert_array_equal(radii, [5.0])
    assert rp.APER_STEP_PIX <= rin - radii.max() < 5.0


def test_impossible_ring_still_raises():
    with pytest.raises(ValueError, match="uncontaminated sky annulus"):
        _geometry(20.0, [25.0], 5.0)


def test_gaia_aperture_radii_applies_gap(monkeypatch):
    class Reference:
        fwhm = 5.0
        header = {"PIXSCALE": 0.4}
        telescope = SimpleNamespace(pixel_scale=0.4)

    monkeypatch.setattr(rp, "_gaia_catalog_df", lambda *a, **k: None)

    radii, rin, _ = rp.gaia_aperture_radii(Reference(), 0, None, annulus_gap=9.0)

    assert rin == 29.0
    assert rin - radii.max() >= 9.0


@pytest.mark.parametrize("flag", ["--annulus_gap", "--annulus-gap"])
def test_cli_accepts_gap(flag):
    assert rp.parse_args([*BASE_ARGS, flag, "7.5"]).annulus_gap == 7.5


def test_cli_default_gap():
    assert rp.parse_args(BASE_ARGS).annulus_gap == rp.ANNULUS_GAP_PIX


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "wide"])
def test_cli_rejects_invalid_gap(value):
    with pytest.raises(SystemExit):
        rp.parse_args([*BASE_ARGS, "--annulus_gap", value])


def test_cli_rejects_gap_with_custom_grid():
    with pytest.raises(SystemExit):
        rp.parse_args([*BASE_ARGS, *CUSTOM_GRID, "--annulus_gap", "5"])


def test_cli_custom_grid_without_gap_is_unchanged():
    args = rp.parse_args([*BASE_ARGS, *CUSTOM_GRID])

    assert args.aper_radii.max() == 15
