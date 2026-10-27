import numpy as np
from prose import Fluxes
from prose.fluxes import diff


def test_normalization_without_comparisons():
    data = np.array([[2.0, 4.0, 6.0], [10.0, 20.0, 30.0]])
    result, error = diff(data)
    np.testing.assert_allclose(result, data / data.mean(axis=-1, keepdims=True))
    assert error is None
    result, error = diff(data, errors=np.ones_like(data))
    np.testing.assert_allclose(
        error, np.ones_like(data) / data.mean(axis=-1, keepdims=True)
    )


def test_fluxes_normalization_public_api():
    obj = Fluxes(np.array([2.0, 3.0, 4.0]), errors=np.ones(3))
    out = obj.diff()
    np.testing.assert_allclose(out.flux, [2 / 3, 1, 4 / 3])
    np.testing.assert_allclose(out.error, 1 / 3)
