import numpy as np
from prose import Fluxes


def test_binning_time_axis_and_heteroscedastic_errors():
    f = np.arange(30.0, dtype=float).reshape(2, 3, 5) + 1
    e = f / 10
    obj = Fluxes(
        f,
        time=np.arange(5.0),
        errors=e,
        data={"airmass": np.arange(5.0) + 1},
        apertures=np.ones((5, 2)),
    )
    out = obj.bin(2)
    assert out.fluxes.shape == out.errors.shape == (2, 3, 3)
    np.testing.assert_allclose(
        out.errors[..., 0], np.sqrt(e[..., 0] ** 2 + e[..., 1] ** 2) / 2
    )
    np.testing.assert_allclose(out.errors[..., -1], e[..., -1])
    assert out.data["airmass"].shape == (3,)
    assert out.apertures.shape == (3, 2)


def test_one_dimensional_binning():
    out = Fluxes(np.arange(8.0), errors=np.ones(8)).bin(2)
    np.testing.assert_allclose(out.error, np.sqrt(0.5))
