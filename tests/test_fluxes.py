import numpy as np
import pytest

from prose import Fluxes
from prose.fluxes import weights


def test_copy():
    x = np.random.rand(10)
    f = Fluxes(x, data={"test": 0})
    f2 = f.copy()
    assert f.flux is not f2.flux
    assert f.data is not f2.data
    assert f.data == f2.data


def test_1d():
    x = np.random.rand(10)
    f = Fluxes(x)
    f.flux

    # with errors
    with pytest.raises(AssertionError) as excinfo:
        f.error
    assert "errors not provided" in str(excinfo.value)

    f = Fluxes(fluxes=x, errors=x)
    f.error


def test_2d():
    x = np.random.rand(2, 10)
    f = Fluxes(x)
    with pytest.raises(AssertionError) as excinfo:
        f.flux
        f.error
    assert "target must be set" in str(excinfo.value)
    f.target = 0
    f.flux

    # with errors
    with pytest.raises(AssertionError) as excinfo:
        f.error
    assert "errors not provided" in str(excinfo.value)

    f = Fluxes(fluxes=x, errors=x, target=0)
    f.error


def test_3d():
    x = np.random.rand(3, 2, 10)
    f = Fluxes(x)
    with pytest.raises(AssertionError) as excinfo:
        f.flux
        f.error
    assert "target must be set" in str(excinfo.value)
    f.target = 0
    with pytest.raises(AssertionError) as excinfo:
        f.flux
        f.error
    assert "aperture must be set" in str(excinfo.value)
    f.aperture = 0
    f.flux

    # with errors
    with pytest.raises(AssertionError) as excinfo:
        f.error
    assert "errors not provided" in str(excinfo.value)

    f = Fluxes(fluxes=x, errors=x, target=0, aperture=0)
    f.error


def test_diff():
    x = np.random.uniform(0, 10000, size=(3, 2, 10))
    f = Fluxes(x)
    f.target = 1
    diff = f.autodiff()


def test_weights_single_comparison_star():
    fluxes = np.array(
        [
            [1.0, 1.1, 0.9, 1.05],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    w = weights(fluxes)
    assert w.shape == (2,)
    assert np.all(np.isfinite(w))


def test_sigma_clip_flux_poly():
    rng = np.random.default_rng(42)
    n = 500
    time = 2461200.0 + np.cumsum(rng.uniform(0.01, 0.02, n))
    # smooth quadratic trend the plain mean-based clip would mistake for scatter
    base = 40000.0 + 800.0 * (time - time.min()) + 50.0 * (time - time.min()) ** 2
    flux = base + rng.normal(0, 120, n)
    out_idx = [40, 150, 333]
    flux[out_idx] += rng.choice([-1, 1], 3) * rng.uniform(2500, 5000, 3)

    f = Fluxes(flux, time=time, errors=np.ones(n), target=0, aperture=0)
    mask = f.sigma_clip_flux_poly(sigma=6.0, degree=2, return_mask=True)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    # exactly the injected spikes are rejected, nothing else
    assert not np.any(mask[out_idx])
    assert np.count_nonzero(~mask) == 3
    # without detrending, the trend inflates sigma and nothing is rejected
    m = np.ones(n, dtype=bool)
    for _ in range(5):
        m &= np.abs(flux - np.nanmean(flux)) < np.nanstd(flux[m]) * 6.0
    assert np.count_nonzero(~m) == 0

    # masked instance shares the frame mask across the whole time axis
    clipped = f.sigma_clip_flux_poly(sigma=6.0, degree=2)
    assert len(clipped.time) == n - 3
    assert "bkg" not in clipped.data or len(clipped.data["bkg"]) == n - 3

    # too few points to fit: warn and leave flux unmasked
    g = Fluxes(flux[:2], time=time[:2], target=0, aperture=0)
    with pytest.warns(UserWarning, match="fewer than"):
        kept = g.sigma_clip_flux_poly(sigma=6.0, degree=2)
    assert len(kept.time) == 2

    # time-less 1D series falls back to frame index
    h = Fluxes(flux, target=0, aperture=0)
    mask_h = h.sigma_clip_flux_poly(sigma=6.0, degree=2, return_mask=True)
    assert np.count_nonzero(~mask_h) == 3


def test_sigma_clip_flux_poly_guards():
    with pytest.raises(ValueError, match="target and aperture"):
        Fluxes(np.random.rand(3, 50)).sigma_clip_flux_poly(sigma=6.0)
    with pytest.raises(ValueError, match="degree"):
        Fluxes(np.random.rand(50), target=0, aperture=0).sigma_clip_flux_poly(
            sigma=6.0, degree=-1
        )
