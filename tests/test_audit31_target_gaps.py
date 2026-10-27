import numpy as np
import pytest
from prose import Fluxes
from prose.scripts.run_photometry import differential_photometry


@pytest.mark.parametrize("method", ["none", "linear", "median"])
def test_target_gaps_are_removed_not_interpolated(method):
    f = 100 + np.random.default_rng(9).normal(size=(2, 4, 40))
    f[0, 0, 20] = np.nan
    f[1, 0, 21] = np.inf
    t = np.arange(40.0) ** 2
    obj = Fluxes(
        f,
        time=t,
        errors=np.ones_like(f),
        data={"airmass": np.arange(40.0)},
        apertures=np.ones((40, 2)),
    )
    out = differential_photometry(obj, 0, cids=[1, 2, 3], nan_imputation_method=method)
    np.testing.assert_equal(out.time, np.delete(t, [20, 21]))
    assert out.errors.shape[-1] == len(out.data["airmass"]) == len(out.apertures) == 38
    assert np.isfinite(out.flux).all()
    assert obj.fluxes.shape[-1] == 40
