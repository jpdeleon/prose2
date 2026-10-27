import numpy as np
import pytest
from prose import Fluxes
from prose.scripts.run_photometry import differential_photometry


@pytest.mark.parametrize("method", ["none", "linear"])
@pytest.mark.parametrize("cids", [None, [1, 2]])
def test_entirely_missing_target_never_becomes_a_lightcurve(method, cids):
    f = 100 + np.random.default_rng(9).normal(size=(2, 6, 100))
    f[:, 0] = np.nan
    obj = Fluxes(f, time=np.arange(100.0), errors=np.ones_like(f))
    assert (
        differential_photometry(obj, 0, cids=cids, nan_imputation_method=method) is None
    )
    assert np.isnan(obj.fluxes[:, 0]).all()
