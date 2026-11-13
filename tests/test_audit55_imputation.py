"""#55 A3, A4: comparison-star imputation is capped, never extrapolates, follows
time, and explicit --cID comparisons are validated like the automatic pool."""

import logging

import numpy as np
import pytest

from prose.fluxes import Fluxes, impute_nans
from prose.scripts import run_photometry as rp

N = 200
TARGET = 0


def _fluxes(n_stars=6, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(40000, 20000, n_stars)[None, :, None]
    trend = 1 + 0.01 * np.sin(np.linspace(0, 3, N))  # shared transparency
    flux = base * trend * (1 + rng.normal(0, 0.001, (2, n_stars, N)))
    return Fluxes(
        fluxes=flux,
        errors=np.sqrt(flux),
        time=2460000.0 + np.arange(N) / 1440.0,
    )


def _nan_out(fluxes, star, frames):
    out = fluxes.copy()
    out.fluxes[:, star, frames] = np.nan
    out.errors[:, star, frames] = np.nan
    return out


# ------------------------------ impute_nans ------------------------------


def test_linear_imputation_holds_edge_value_instead_of_extrapolating():
    series = np.array([[1.0, 2.0, 3.0, 4.0, np.nan, np.nan]])
    out = impute_nans(series, method="linear")
    np.testing.assert_array_equal(out[0, 4:], [4.0, 4.0])


def test_linear_imputation_interpolates_in_time_not_index():
    x = np.array([0.0, 1.0, 2.0, 10.0])
    series = np.array([[0.0, 1.0, np.nan, 10.0]])  # value equals time
    out = impute_nans(series, method="linear", x=x)
    assert out[0, 2] == pytest.approx(2.0)


def test_spline_imputation_holds_edges():
    series = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, np.nan]])
    out = impute_nans(series, method="spline")
    assert out[0, 5] == pytest.approx(5.0)


# ------------------------ differential_photometry ------------------------


def test_sparse_comparison_is_excluded_from_auto_pool(caplog):
    fluxes = _nan_out(_fluxes(), star=3, frames=slice(40, 200))  # 80% missing
    fluxes = _nan_out(fluxes, star=4, frames=[5, 6, 7])  # 1.5% missing
    with caplog.at_level(logging.INFO, logger="prose_run_photometry"):
        diff = rp.differential_photometry(fluxes, TARGET)

    assert 3 not in list(diff.comparisons)
    assert 4 in list(diff.comparisons)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "missing in more than 10%" in messages and "3" in messages
    assert "imputed" in messages and "4: 3" in messages


def test_errors_are_imputed_with_fluxes():
    fluxes = _nan_out(_fluxes(), star=4, frames=[5, 6, 7])
    diff = rp.differential_photometry(fluxes, TARGET)
    assert np.all(np.isfinite(diff.flux))
    assert np.all(np.isfinite(diff.error))


def test_explicit_sparse_comparison_is_rejected(caplog):
    fluxes = _nan_out(_fluxes(), star=3, frames=slice(40, 200))
    with caplog.at_level(logging.WARNING, logger="prose_run_photometry"):
        diff = rp.differential_photometry(fluxes, TARGET, cids=[2, 3])

    np.testing.assert_array_equal(np.flatnonzero(diff.weights[0]), [2])
    assert np.all(np.isfinite(diff.flux))
    assert any(
        "rejecting explicit comparison" in r.getMessage() for r in caplog.records
    )


def test_explicit_nan_comparison_rejected_without_imputation(caplog):
    fluxes = _nan_out(_fluxes(), star=2, frames=[10])
    with caplog.at_level(logging.WARNING, logger="prose_run_photometry"):
        diff = rp.differential_photometry(
            fluxes, TARGET, cids=[2, 5], nan_imputation_method="none"
        )

    np.testing.assert_array_equal(np.flatnonzero(diff.weights[0]), [5])
    assert np.all(np.isfinite(diff.flux))
    assert any("[2]" in r.getMessage() for r in caplog.records)


def test_all_explicit_comparisons_rejected_falls_back_to_auto(caplog):
    fluxes = _nan_out(_fluxes(), star=3, frames=slice(40, 200))
    with caplog.at_level(logging.WARNING, logger="prose_run_photometry"):
        diff = rp.differential_photometry(fluxes, TARGET, cids=[3])

    comps = list(diff.comparisons)
    assert comps and 3 not in comps
    assert np.all(np.isfinite(diff.flux))
