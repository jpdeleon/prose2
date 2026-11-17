"""--save_flux_datacube: the per-star, per-aperture flux cube kept after
differential photometry, and re-differencing it with other comparison stars
or another aperture without re-running photometry."""

import numpy as np
import pytest

from prose import datacube as dc
from prose.fluxes import Fluxes
from prose.scripts import run_photometry as rp

N = 300
N_STARS = 8
N_APER = 4
TARGET = 0


def _fluxes(seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(60000, 15000, N_STARS)[None, :, None]
    aper = np.linspace(0.8, 1.1, N_APER)[:, None, None]
    trend = 1 + 0.02 * np.sin(np.linspace(0, 3, N))  # shared transparency
    flux = base * aper * trend * (1 + rng.normal(0, 0.002, (N_APER, N_STARS, N)))
    return Fluxes(
        fluxes=flux,
        errors=np.sqrt(flux),
        time=2460000.0 + np.arange(N) / 1440.0,
        data={"bkg": rng.normal(100, 1, N)},
    )


def _cube(diff, ref_band_star_ids=None, band="g"):
    return dc.band_cube_from_reduction(
        band=band,
        diff=diff,
        bjd_tdb=np.asarray(diff.time) + 0.001,
        aper_radii=np.arange(1, N_APER + 1, dtype=float) * 3,
        aper_unit="pix",
        target_index=TARGET,
        ref_band_star_ids=ref_band_star_ids,
    )


# --------------------- capture in differential_photometry ---------------------


def test_datacube_is_not_kept_unless_requested():
    diff = rp.differential_photometry(_fluxes(), TARGET)
    assert "flux_datacube" not in diff.metadata


def test_datacube_keeps_every_star_unmasked_and_marks_the_pool():
    diff = rp.differential_photometry(
        _fluxes(), TARGET, avoid_cids=[5], keep_datacube=True
    )
    raw = diff.metadata["flux_datacube"]
    assert raw["fluxes"].shape == (N_APER, N_STARS, len(diff.time))
    # stars outside the pool are kept with real values, not prose's -1 mask
    assert np.all(raw["fluxes"] > 0)
    selectable = raw["selectable"]
    assert not selectable[TARGET]
    assert not selectable[5]
    assert selectable[[1, 2, 3, 4, 6, 7]].all()


def test_explicit_cids_still_keep_the_whole_auto_pool_selectable():
    diff = rp.differential_photometry(
        _fluxes(), TARGET, cids=[1, 2], keep_datacube=True
    )
    selectable = diff.metadata["flux_datacube"]["selectable"]
    assert selectable[[1, 2, 3, 4, 5, 6, 7]].all()


def test_sparse_star_is_not_selectable():
    fluxes = _fluxes()
    fluxes.fluxes[:, 3, : N // 2] = np.nan
    diff = rp.differential_photometry(fluxes, TARGET, keep_datacube=True)
    assert not diff.metadata["flux_datacube"]["selectable"][3]


def test_datacube_frames_follow_the_pipeline_frame_clipping(monkeypatch):
    fluxes = _fluxes()
    fluxes.fluxes[:, TARGET, 50] *= 1.3  # outlier rejected by --sig_flux
    fluxes.fluxes[:, TARGET, 80] = np.nan  # dropped: no target measurement
    monkeypatch.setattr(rp, "SIGMA_FLUX", 4.0)
    diff = rp.differential_photometry(fluxes, TARGET, keep_datacube=True)
    raw = diff.metadata["flux_datacube"]
    assert len(diff.time) < N - 1
    assert raw["fluxes"].shape[-1] == len(diff.time)
    np.testing.assert_array_equal(raw["time"], diff.time)


# ------------------------------- rebuild default -------------------------------


def test_default_lightcurve_reproduces_autodiff_broeg_result():
    diff = rp.differential_photometry(_fluxes(), TARGET, keep_datacube=True)
    flux, err = dc.default_lightcurve(_cube(diff))
    np.testing.assert_allclose(flux, diff.flux, rtol=2e-6)
    np.testing.assert_allclose(err, diff.error, rtol=2e-5)


def test_equal_weight_rediff_reproduces_explicit_cid_run():
    diff = rp.differential_photometry(
        _fluxes(), TARGET, cids=[2, 4, 6], keep_datacube=True
    )
    flux, err = dc.rediff(_cube(diff), [2, 4, 6], diff.aperture)
    np.testing.assert_allclose(flux, diff.flux, rtol=2e-6)
    np.testing.assert_allclose(err, diff.error, rtol=2e-5)


def test_rediff_other_aperture_differs_from_default_aperture():
    diff = rp.differential_photometry(
        _fluxes(), TARGET, cids=[2, 4], keep_datacube=True
    )
    other = (diff.aperture + 1) % N_APER
    a, _ = dc.rediff(_cube(diff), [2, 4], diff.aperture)
    b, _ = dc.rediff(_cube(diff), [2, 4], other)
    assert not np.allclose(a, b)


def test_default_comparisons_are_ordered_by_weight_in_display_ids():
    diff = rp.differential_photometry(_fluxes(), TARGET, keep_datacube=True)
    cube = _cube(diff)
    np.testing.assert_array_equal(cube.default_comparisons, diff.comparisons)


# ------------------------------- star ID mapping -------------------------------


def test_ids_are_reference_band_ids_under_ref_band():
    # band-local star k is reference-band star 10 + k; star 7 is unmatched
    mapping = np.array([10, 11, 12, 13, 14, 15, 16, -1])
    diff = rp.differential_photometry(
        _fluxes(), TARGET, cids=[2, 4], keep_datacube=True
    )
    cube = _cube(diff, ref_band_star_ids=mapping)
    flux, _ = dc.rediff(cube, [12, 14], diff.aperture)
    np.testing.assert_allclose(flux, diff.flux, rtol=2e-6)
    assert 17 not in cube.selectable_ids
    assert cube.target_id == 10


def test_short_reference_band_mapping_pads_as_unmatched():
    mapping = np.array([10, 11, 12])  # shorter than the star axis
    diff = rp.differential_photometry(_fluxes(), TARGET, keep_datacube=True)
    cube = _cube(diff, ref_band_star_ids=mapping)
    assert list(cube.selectable_ids) == [11, 12]


@pytest.mark.parametrize(
    "comps, aperture, match",
    [
        ([], 0, "at least one"),
        ([0], 0, "target"),
        ([99], 0, "unknown"),
        ([2, 2], 0, "duplicate"),
        ([5], 0, "not selectable"),
        ([2], N_APER, "aperture"),
        ([2], -1, "aperture"),
    ],
)
def test_rediff_rejects_invalid_selection(comps, aperture, match):
    diff = rp.differential_photometry(
        _fluxes(), TARGET, avoid_cids=[5], keep_datacube=True
    )
    with pytest.raises(ValueError, match=match):
        dc.rediff(_cube(diff), comps, aperture)


# ---------------------------------- save / load ----------------------------------


def test_save_load_roundtrip_is_float32_and_pickle_free(tmp_path):
    diff = rp.differential_photometry(_fluxes(), TARGET, keep_datacube=True)
    cube = _cube(diff)
    path = tmp_path / "x_flux_datacube.npz"
    dc.save_datacube(path, [cube], meta={"target_name": "T"})

    with np.load(path, allow_pickle=False) as z:
        assert z["g__fluxes"].dtype == np.float32
    loaded = dc.load_datacube(path)
    back = loaded.bands["g"]
    assert loaded.meta["target_name"] == "T"
    assert back.aper_unit == "pix"
    np.testing.assert_array_equal(back.star_ids, cube.star_ids)
    np.testing.assert_array_equal(back.selectable, cube.selectable)
    np.testing.assert_array_equal(back.bjd_tdb, cube.bjd_tdb)
    flux, _ = dc.default_lightcurve(back)
    np.testing.assert_allclose(flux, diff.flux, rtol=2e-6)


def test_load_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, __schema__=np.array(999), __bands__=np.array(["g"]))
    with pytest.raises(ValueError, match="schema"):
        dc.load_datacube(path)


def test_metrics_report_std_and_point_to_point_scatter_in_ppt():
    flux = np.array([1.0, 1.002, 0.998, 1.0])
    m = dc.scatter_metrics(flux)
    assert m["std_ppt"] == pytest.approx(np.std(flux, ddof=1) * 1e3)
    assert m["p2p_ppt"] == pytest.approx(
        np.std(np.diff(flux), ddof=1) / np.sqrt(2) * 1e3
    )
    assert m["n"] == 4


# ------------------------------ run_photometry wiring ------------------------------


def _parse(*extra):
    return rp.parse_args(
        ["--target_name", "x", "--data_dir", ".", "--results_dir", ".", *extra]
    )


def test_save_flux_datacube_is_off_by_default():
    assert _parse().save_flux_datacube is False
    assert _parse("--save_flux_datacube").save_flux_datacube is True
    assert _parse("--save-flux-datacube").save_flux_datacube is True


def test_save_flux_datacube_writes_bands_that_kept_a_cube(tmp_path):
    kept = rp.differential_photometry(_fluxes(), TARGET, keep_datacube=True)
    dropped = rp.differential_photometry(_fluxes(seed=1), TARGET)
    band_results = {
        b: dict(
            diff=d,
            aper_radii=np.arange(1, N_APER + 1, dtype=float),
            scale=True,
            target_index=TARGET,
            ref_band_star_ids=None,
        )
        for b, d in (("g", kept), ("r", dropped))
    }
    bjds = {b: np.asarray(r["diff"].time) for b, r in band_results.items()}
    path = tmp_path / "x_flux_datacube.npz"
    rp.save_flux_datacube(band_results, bjds, path, meta={"a": 1})
    loaded = dc.load_datacube(path)
    assert list(loaded.bands) == ["g"]
    assert loaded.bands["g"].aper_unit == "fwhm"


def test_save_flux_datacube_writes_nothing_without_cubes(tmp_path):
    diff = rp.differential_photometry(_fluxes(), TARGET)
    band_results = {
        "g": dict(diff=diff, aper_radii=np.ones(N_APER), target_index=TARGET)
    }
    path = tmp_path / "x_flux_datacube.npz"
    rp.save_flux_datacube(band_results, {"g": diff.time}, path)
    assert not path.exists()
