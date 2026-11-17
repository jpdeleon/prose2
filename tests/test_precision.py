"""Opt-in comparison ensemble, aperture criterion, per-star clipping and noise
budget (muscat-team/muscatdb#445)."""

import logging

import numpy as np
import pytest

from prose import precision as pr
from prose.fluxes import Fluxes, auto_diff_1d, diff
from prose.scripts import run_photometry as rp

N = 400
N_STARS = 10
N_APER = 3
TARGET = 0


def _matrix(seed=0, n_stars=N_STARS, noise=None, means=None):
    """(star, time) fluxes with shared transparency and photon-like noise."""
    rng = np.random.default_rng(seed)
    means = np.linspace(20000, 4000, n_stars) if means is None else np.asarray(means)
    trend = 1 + 0.03 * np.sin(np.linspace(0, 4, N))
    sigma = 1 / np.sqrt(means) if noise is None else np.asarray(noise)
    flux = (
        means[:, None] * trend * (1 + rng.normal(0, 1, (n_stars, N)) * sigma[:, None])
    )
    return flux, means[:, None] * sigma[:, None] * np.ones((1, N))


def _cube(seed=0, aperture_noise=(2e-3, 1e-3, 3e-3)):
    """(aperture, star, time) cube whose apertures differ only in noise level."""
    rng = np.random.default_rng(seed)
    means = np.linspace(20000, 6000, N_STARS)
    trend = 1 + 0.03 * np.sin(np.linspace(0, 4, N))
    noise = np.asarray(aperture_noise)[:, None, None]
    flux = (
        means[None, :, None]
        * trend
        * (1 + rng.normal(0, 1, (N_APER, N_STARS, N)) * noise)
    )
    errors = means[None, :, None] * noise * np.ones_like(flux)
    return flux, errors


# ----------------------------------------------------------------- options


def test_default_options_are_prose_behaviour():
    opts = pr.PrecisionOptions()
    assert opts.is_default
    assert opts.clip_sigma is None
    assert not pr.PrecisionOptions(ensemble="bright").is_default
    assert not pr.PrecisionOptions(criterion="check").is_default


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ensemble": "all"},
        {"criterion": "binned"},
        {"min_flux_fraction": -0.1},
        {"disagree_factor": 1.0},
        {"clip_sigma": 0},
        {"clip_window": 2},
    ],
)
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        pr.PrecisionOptions(**kwargs)


# ----------------------------------------------------------------- helpers


def test_ptp_scatter_measures_white_noise_and_ignores_trends():
    rng = np.random.default_rng(1)
    white = rng.normal(0, 1e-3, 20000)
    trend = np.linspace(0, 0.05, 20000)
    assert pr.ptp_scatter(white) == pytest.approx(1e-3, rel=0.03)
    assert pr.ptp_scatter(white + trend) == pytest.approx(1e-3, rel=0.03)


def test_eligible_comparisons_skip_target_masked_and_nonfinite_stars():
    flux, _ = _matrix()
    flux[3] = -1  # Fluxes.mask_stars sentinel
    flux[4, 10] = np.nan
    eligible = pr.eligible_comparisons(flux, TARGET)
    assert not eligible[[TARGET, 3, 4]].any()
    assert eligible.sum() == N_STARS - 3


def test_predicted_target_scatter_adds_ensemble_noise_in_quadrature():
    # two equal-weight comparisons of 2e-3 each: ensemble adds 2e-3 / sqrt(2)
    got = pr.predicted_target_scatter(
        1e-3, np.array([0, 2e-3, 2e-3]), np.array([0, 1, 1])
    )
    assert got == pytest.approx(np.sqrt(1e-3**2 + 2e-3**2 / 2))
    assert pr.predicted_target_scatter(1e-3, np.ones(3), np.zeros(3)) == np.inf


# ---------------------------------------------------------- bright ensemble


def test_bright_ensemble_uses_every_bright_star_weighted_by_flux():
    means = np.array([10000, 20000, 15000, 8000, 5000, 2000, 1000])
    flux, err = _matrix(means=means, n_stars=len(means))
    w = pr.bright_ensemble_weights(flux, TARGET, err, min_flux_fraction=0.3)
    # 2000 and 1000 are fainter than 0.3 x 10000
    assert np.flatnonzero(w).tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(
        w[1:5] / w[1], flux[1:5].mean(-1) / flux[1].mean(), rtol=1e-12
    )


def test_bright_ensemble_equals_summing_raw_comparison_fluxes():
    flux, err = _matrix()
    w = pr.bright_ensemble_weights(flux, TARGET, err, min_flux_fraction=0.0)
    lc, _ = diff(flux, w)
    comps = w > 0
    summed = flux[TARGET] / flux[comps].sum(axis=0)
    target_lc = lc.reshape(flux.shape)[TARGET]
    # identical up to a constant normalization
    np.testing.assert_allclose(
        target_lc / target_lc.mean(), summed / summed.mean(), rtol=1e-12
    )


def test_bright_ensemble_drops_a_comparison_that_disagrees():
    flux, err = _matrix(seed=2)
    variable = 3
    # slow variable: invisible to a point-to-point metric, not to the std
    flux[variable] *= 1 + 0.03 * np.sin(np.linspace(0, 6, N))
    w = pr.bright_ensemble_weights(flux, TARGET, err, min_flux_fraction=0.0)
    assert w[variable] == 0
    assert np.count_nonzero(w) == N_STARS - 2


def test_bright_ensemble_falls_back_to_all_stars_when_none_is_bright():
    means = np.array([100000, 5000, 4000, 3000])
    flux, err = _matrix(means=means, n_stars=4)
    w = pr.bright_ensemble_weights(flux, TARGET, err, min_flux_fraction=0.3)
    assert np.flatnonzero(w).tolist() == [1, 2, 3]


def test_bright_ensemble_without_errors_uses_photon_scaling():
    flux, _ = _matrix(seed=3)
    w = pr.bright_ensemble_weights(flux, TARGET, None, min_flux_fraction=0.0)
    assert np.count_nonzero(w) == N_STARS - 1


def test_bright_ensemble_rejects_no_comparisons():
    flux, err = _matrix(n_stars=2)
    flux[1] = -1
    with pytest.raises(ValueError, match="no valid comparison"):
        pr.bright_ensemble_weights(flux, TARGET, err)


# ----------------------------------------------------------- check criterion


def _with_transit(flux):
    out = flux.copy()
    out[TARGET, N // 3 : N // 2] *= 0.99
    return out


def test_greedy_check_weights_never_look_at_the_target_series():
    flux, err = _matrix(seed=4)
    rng = np.random.default_rng(9)
    noisy_target = flux.copy()
    noisy_target[TARGET] *= 1 + rng.normal(0, 0.01, N)
    w = pr.greedy_check_weights(flux, TARGET, err)
    np.testing.assert_array_equal(
        w, pr.greedy_check_weights(_with_transit(flux), TARGET, err)
    )
    np.testing.assert_array_equal(w, pr.greedy_check_weights(noisy_target, TARGET, err))
    assert w[TARGET] == 0 and np.count_nonzero(w) >= 1


def test_greedy_check_weights_choose_the_best_predicted_ensemble():
    flux, err = _matrix(seed=5)
    w = pr.greedy_check_weights(flux, TARGET, err)
    eligible = pr.eligible_comparisons(flux, TARGET)
    broeg = np.zeros(N_STARS)
    broeg[eligible] = pr.broeg_weights(flux[eligible])  # comparisons only
    sigma = pr.comparison_noise(flux, broeg)
    t_err = float(np.median(err[TARGET]) / flux[TARGET].mean())
    best = pr.predicted_target_scatter(t_err, sigma, w)
    order = [j for j in np.argsort(broeg)[::-1] if broeg[j] > 0]
    for count in range(1, len(order) + 1):
        prefix = np.zeros_like(broeg)
        prefix[order[:count]] = broeg[order[:count]]
        assert best <= pr.predicted_target_scatter(t_err, sigma, prefix) + 1e-15


def test_greedy_check_weights_need_errors_and_handle_a_lone_comparison():
    flux, err = _matrix(n_stars=2)
    with pytest.raises(ValueError, match="needs flux errors"):
        pr.greedy_check_weights(flux, TARGET, None)
    np.testing.assert_array_equal(pr.greedy_check_weights(flux, TARGET, err), [0, 1])


def test_check_aperture_index_picks_the_quietest_aperture():
    flux, errors = _cube()
    w = np.ones((N_APER, N_STARS))
    w[:, TARGET] = 0
    assert pr.check_aperture_index(flux, errors, w, TARGET) == 1


def test_check_aperture_index_ignores_the_target_series():
    flux, errors = _cube(seed=6)
    w = np.ones((N_APER, N_STARS))
    w[:, TARGET] = 0
    spoiled = flux.copy()
    spoiled[1, TARGET] *= 1 + np.random.default_rng(0).normal(0, 0.05, N)
    assert pr.check_aperture_index(spoiled, errors, w, TARGET) == 1
    with pytest.raises(ValueError, match="needs flux errors"):
        pr.check_aperture_index(flux, None, w, TARGET)


def test_precision_diff_matches_auto_diff_shapes():
    flux, errors = _cube()
    opts = pr.PrecisionOptions(ensemble="bright", min_flux_fraction=0.0)
    df, w, derr = pr.precision_diff(flux, TARGET, errors, opts)
    assert df.shape == flux.shape and derr.shape == errors.shape
    assert w.shape == (N_APER, N_STARS) and np.all(w[:, TARGET] == 0)
    ref, _, _ = auto_diff_1d(flux[0], TARGET, errors[0])
    assert ref.shape == df[0].shape


def test_ensemble_weights_refuses_the_default_combination():
    flux, err = _matrix()
    with pytest.raises(ValueError, match="prose's own auto_diff"):
        pr.ensemble_weights(flux, TARGET, err, pr.PrecisionOptions())


# ---------------------------------------------------------------- clipping


def test_clip_star_outliers_replaces_a_comparison_spike_only():
    flux, errors = _cube(seed=7)
    spiked = flux.copy()
    spiked[:, 4, 200] *= 1.03  # cosmic ray in a comparison star
    spiked[:, TARGET, 100] *= 1.03  # target spike: not ours to clip
    stars = np.ones(N_STARS, dtype=bool)
    stars[TARGET] = False
    out, flagged = pr.clip_star_outliers(spiked, stars, sigma=5)
    assert flagged[4, 200] and not flagged[TARGET].any()
    assert flagged.sum() <= 3  # essentially no false positives on clean data
    np.testing.assert_allclose(out[:, 4, 200], flux[:, 4, 200], rtol=0.01)
    np.testing.assert_array_equal(out[:, TARGET], spiked[:, TARGET])
    np.testing.assert_array_equal(
        spiked[:, 4, 200], flux[:, 4, 200] * 1.03
    )  # input intact


def test_clip_star_outliers_needs_several_stars():
    flux, _ = _cube()
    stars = np.zeros(N_STARS, dtype=bool)
    stars[[1, 2]] = True
    out, flagged = pr.clip_star_outliers(flux, stars, sigma=5)
    assert not flagged.any()
    np.testing.assert_array_equal(out, flux)
    assert out is not flux


# ------------------------------------------------------------ noise budget


def test_noise_budget_photon_only():
    rng = np.random.default_rng(8)
    flux = 1e6
    n = 20000  # enough points for a 3% point-to-point scatter estimate
    err = np.full(n, 1.5e-3)
    lc = 1 + rng.normal(0, 1.5e-3, n)
    b = pr.noise_budget(
        flux,
        sky=0,
        area=100,
        sky_area=500,
        gain=1,
        read_noise=0,
        diff_flux=lc,
        diff_error=err,
    )
    assert b["photon_ppt"] == pytest.approx(1.0)
    assert b["sky_ppt"] == 0 and b["read_ppt"] == 0
    assert b["expected_ppt"] == pytest.approx(1.5)
    assert b["comparison_ppt"] == pytest.approx(np.sqrt(1.5**2 - 1.0))
    assert b["ratio"] == pytest.approx(1.0, abs=0.05)


def test_noise_budget_sky_and_read_terms_follow_the_error_model():
    b = pr.noise_budget(
        1e4,
        sky=100,
        area=50,
        sky_area=1000,
        gain=2,
        read_noise=10,
        diff_flux=np.ones(10),
        diff_error=np.full(10, 0.1),
    )
    rn2 = (10 / 2) ** 2
    sky_var = 50 * 100 / 2 + 50**2 * np.pi / 2 * (100 / 2 + rn2) / 1000
    assert b["sky_ppt"] == pytest.approx(1e3 * np.sqrt(sky_var) / 1e4)
    assert b["read_ppt"] == pytest.approx(1e3 * np.sqrt(50 * rn2) / 1e4)


# ------------------------------------------------- differential_photometry


def _fluxes(seed=0, aperture_noise=(2e-3, 1e-3, 3e-3)):
    flux, errors = _cube(seed, aperture_noise)
    return Fluxes(
        fluxes=flux,
        errors=errors,
        time=2460000.0 + np.arange(N) / 1440.0,
        data={"bkg": np.full(N, 100.0)},
    )


def test_default_precision_is_identical_to_no_precision():
    a = rp.differential_photometry(_fluxes(), TARGET)
    b = rp.differential_photometry(_fluxes(), TARGET, precision=pr.PrecisionOptions())
    np.testing.assert_array_equal(a.fluxes, b.fluxes)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.aperture == b.aperture
    assert a.metadata["target_raw_flux"].shape == (N_APER,)


def test_bright_ensemble_in_differential_photometry_respects_avoid_cids():
    opts = pr.PrecisionOptions(ensemble="bright", min_flux_fraction=0.0)
    d = rp.differential_photometry(_fluxes(), TARGET, avoid_cids=[5], precision=opts)
    w = d.weights[d.aperture]
    assert w[5] == 0 and w[TARGET] == 0
    assert np.count_nonzero(w) == N_STARS - 2


def test_check_criterion_selects_aperture_in_auto_and_explicit_modes():
    opts = pr.PrecisionOptions(criterion="check")
    assert rp.differential_photometry(_fluxes(), TARGET, precision=opts).aperture == 1
    explicit = rp.differential_photometry(
        _fluxes(), TARGET, cids=[1, 2, 3], precision=opts
    )
    assert explicit.aperture == 1
    assert np.flatnonzero(explicit.weights[1]).tolist() == [1, 2, 3]


def test_star_clipping_reaches_the_light_curve_and_the_datacube(caplog):
    f = _fluxes(seed=3)
    f.fluxes[:, 2, 150] *= 1.05
    opts = pr.PrecisionOptions(clip_sigma=5)
    with caplog.at_level(logging.INFO):
        d = rp.differential_photometry(f, TARGET, precision=opts, keep_datacube=True)
    assert "per-star clipping" in caplog.text and "2: 1" in caplog.text
    cube = d.metadata["flux_datacube"]["fluxes"]
    assert cube[:, 2, 150] == pytest.approx(f.fluxes[:, 2, 150] / 1.05, rel=0.01)


# ------------------------------------------------------- run_photometry CLI


def _parse(*extra):
    return rp.parse_args(
        ["--target_name", "x", "--data_dir", ".", "--results_dir", ".", *extra]
    )


def test_cli_defaults_build_default_options():
    assert rp.precision_options_from_args(_parse()) == pr.PrecisionOptions()


def test_cli_flags_build_options():
    args = _parse(
        "--comp_ensemble",
        "bright",
        "--comp-criterion",
        "check",
        "--comp_min_flux_frac",
        "0.5",
        "--clip_star_sigma",
        "5",
        "--clip_star_window",
        "21",
    )
    assert rp.precision_options_from_args(args) == pr.PrecisionOptions(
        ensemble="bright",
        criterion="check",
        min_flux_fraction=0.5,
        clip_sigma=5,
        clip_window=21,
    )


@pytest.mark.parametrize(
    "extra", [("--clip_star_sigma", "-1"), ("--clip_star_window", "1")]
)
def test_cli_rejects_invalid_precision_values(extra):
    with pytest.raises(SystemExit):
        _parse(*extra)


def test_band_noise_budget_logs_and_lands_in_the_npz(tmp_path, caplog):
    class _Ref:
        fwhm = 5.0
        gain = 2.0
        read_noise = 8.0

        class wcs:  # noqa: N801 - minimal stand-in for the npz writer
            @staticmethod
            def to_header_string(relax=True):
                return ""

    d = rp.differential_photometry(_fluxes(), TARGET)
    d.apertures = np.tile([5.0, 8.0, 11.0], (len(d.time), 1))
    reference = {"rin": 20.0, "rout": 30.0, "scale": False}
    with caplog.at_level(logging.INFO):
        budget = rp._band_noise_budget("g", d, _Ref(), reference)
    assert "noise per point (ppt)" in caplog.text
    assert set(budget) >= {"photon_ppt", "comparison_ppt", "expected_ppt", "ratio"}
    result = {
        "diff": d,
        "aper_radii": np.array([5.0, 8.0, 11.0]),
        "rin": 20.0,
        "rout": 30.0,
        "target_index": TARGET,
        "ref": _Ref(),
        "noise_budget": budget,
    }
    path = tmp_path / "x.npz"
    rp.save_all_bands_npz({"g": result}, {"g": np.asarray(d.time)}, path)
    with np.load(path) as npz:
        assert float(npz["g__noise__expected_ppt"]) == pytest.approx(
            budget["expected_ppt"]
        )


def test_band_noise_budget_degrades_to_none_when_inputs_are_missing(caplog):
    d = rp.differential_photometry(_fluxes(), TARGET)
    del d.metadata["target_raw_flux"]
    assert (
        rp._band_noise_budget("g", d, object(), {"rin": 1, "rout": 2, "scale": False})
        is None
    )
    assert "noise budget unavailable" in caplog.text


@pytest.mark.parametrize(
    "opts",
    [
        pr.PrecisionOptions(ensemble="bright", min_flux_fraction=0.0),
        pr.PrecisionOptions(criterion="check", clip_sigma=5),
        pr.PrecisionOptions(ensemble="bright", criterion="check", clip_sigma=5),
    ],
)
def test_datacube_default_rebuild_matches_precision_runs(opts):
    from prose import datacube as dc

    f = _fluxes(seed=4)
    f.fluxes[:, 3, 120] *= 1.05
    d = rp.differential_photometry(f, TARGET, keep_datacube=True, precision=opts)
    cube = dc.band_cube_from_reduction(
        band="g",
        diff=d,
        bjd_tdb=np.asarray(d.time),
        aper_radii=np.array([5.0, 8.0, 11.0]),
        aper_unit="pix",
        target_index=TARGET,
        ref_band_star_ids=None,
    )
    flux, err = dc.default_lightcurve(cube)
    np.testing.assert_allclose(flux, d.flux, rtol=2e-6)
    np.testing.assert_allclose(err, d.error, rtol=2e-5)
