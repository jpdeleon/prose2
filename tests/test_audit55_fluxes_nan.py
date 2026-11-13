"""#55 A5, A6, A8: NaN handling and robust statistics in prose.fluxes."""

import numpy as np
import pytest

from prose.fluxes import Fluxes, optimal_flux

N = 200


def _apertures(noise=(0.003, 0.001, 0.002), seed=0):
    """Differential fluxes (aperture, time); aperture 1 is the quietest."""
    rng = np.random.default_rng(seed)
    return np.array([1.0 + rng.normal(0, s, N) for s in noise])


# --------------------------- A5: optimal_flux ---------------------------


def test_optimal_flux_picks_quietest_aperture():
    assert optimal_flux(_apertures()) == 1


def test_optimal_flux_single_nan_does_not_reset_to_index_0():
    fluxes = _apertures()
    fluxes[2, 10] = np.nan
    assert optimal_flux(fluxes) == 1


def test_optimal_flux_clips_negative_outliers():
    # Aperture 0 is quietest but has one large negative outlier; clipping it
    # must be symmetric, so aperture 0 still wins.
    fluxes = _apertures(noise=(0.001, 0.002, 0.003))
    fluxes[0, 50] -= 0.5
    assert optimal_flux(fluxes) == 0


def test_optimal_flux_robust_to_many_outliers():
    # 10% of frames carry +0.5 outliers in aperture 0. They inflate the std
    # enough to hide themselves from a 4-sigma std cut; a MAD-based cut
    # still removes them.
    fluxes = _apertures(noise=(0.001, 0.002, 0.003))
    fluxes[0, ::10] += 0.5
    assert optimal_flux(fluxes) == 0


def test_optimal_flux_raises_without_finite_criterion():
    fluxes = np.full((3, N), np.nan)
    with pytest.raises(ValueError, match="no finite"):
        optimal_flux(fluxes)


# ------------------------ A6: sigma_clipping_data ------------------------


def _fluxes_with(**data):
    rng = np.random.default_rng(1)
    return Fluxes(
        fluxes=1.0 + rng.normal(0, 0.001, (1, 2, N)),
        time=np.arange(N, dtype=float),
        data={k: np.asarray(v, dtype=float) for k, v in data.items()},
    )


def test_sigma_clipping_keeps_frames_for_constant_column():
    out = _fluxes_with(dx=np.zeros(N)).sigma_clipping_data(dx=3)
    assert len(out.time) == N


def test_sigma_clipping_keeps_frames_for_all_nan_column():
    out = _fluxes_with(dx=np.full(N, np.nan)).sigma_clipping_data(dx=3)
    assert len(out.time) == N


def test_sigma_clipping_does_not_drop_isolated_nan():
    dx = np.random.default_rng(2).normal(0, 0.1, N)
    dx[7] = np.nan
    out = _fluxes_with(dx=dx).sigma_clipping_data(dx=5)
    assert 7.0 in out.time


def test_sigma_clipping_removes_outliers_robustly():
    dx = np.random.default_rng(3).normal(0, 0.1, N)
    dx[::10] = 5.0  # 10% outliers would inflate a mean/std threshold
    out = _fluxes_with(dx=dx).sigma_clipping_data(dx=3)
    assert not np.any(np.isin(out.time, np.arange(0, N, 10)))
    # A 3-sigma cut may also take the odd Gaussian frame, but no more.
    assert len(out.time) >= N - N // 10 - 3


# --------------------------- A8: mask_stars ---------------------------


def _weighted_fluxes():
    fluxes = Fluxes(
        fluxes=np.ones((2, 3, N)),
        errors=np.full((2, 3, N), 0.01),
        time=np.arange(N, dtype=float),
    )
    fluxes.weights = np.ones((2, 3))
    return fluxes


def test_mask_stars_zeroes_weights_not_errors():
    out = _weighted_fluxes().mask_stars(np.array([True, False, True]))
    np.testing.assert_array_equal(out.weights[:, 1], 0.0)
    np.testing.assert_array_equal(out.weights[:, [0, 2]], 1.0)
    np.testing.assert_array_equal(out.errors[:, 1], -1.0)


def test_mask_stars_without_indexing_slices_2d_weights_by_star():
    out = _weighted_fluxes().mask_stars(
        np.array([True, False, True]), keep_indexing=False
    )
    assert out.fluxes.shape == (2, 2, N)
    assert out.errors.shape == (2, 2, N)
    assert out.weights.shape == (2, 2)
