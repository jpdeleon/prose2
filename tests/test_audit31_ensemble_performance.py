import numpy as np
import pytest
from prose.fluxes import diff


@pytest.mark.parametrize("n", [2, 20, 100])
@pytest.mark.parametrize("apertures", [None, 3])
def test_ensemble_sum_matches_dense_reference(n, apertures):
    rng = np.random.default_rng(1)
    shape = (n, 100) if apertures is None else (apertures, n, 100)
    f = rng.uniform(10, 100, shape)
    w = rng.uniform(0.1, 1, shape[:-1])
    e = np.sqrt(f)
    w[..., 0] = 0
    z = f / f.mean(axis=-1, keepdims=True)
    sub = (1 - np.eye(n))[None, ...]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (sub @ (z * w[..., None])) / (w @ sub[0])[..., None]
        expected = z / c
        normalized_error = e / f.mean(axis=-1, keepdims=True)
        expected_error = np.sqrt(
            normalized_error**2
            + (sub @ (normalized_error**2 * w[..., None] ** 2))
            / (w**2 @ sub[0])[..., None]
        )
    result, error = diff(f, w, e)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(error, expected_error, atol=1e-12)


def test_dominant_weight_does_not_erase_small_comparison():
    f = np.array([[100.0, 101.0, 99.0], [50.0, 49.0, 51.0]])
    w = np.array([1e20, 1.0])
    out, errors = diff(f, w, np.ones_like(f))
    z = f / f.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(out[0, 0], z[0] / z[1])
    assert np.isfinite(errors).all()
