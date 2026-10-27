import numpy as np
from prose.fluxes import diff


def test_equal_comparison_uncertainty():
    values = np.ones((3, 20)) * 100
    _, errors = diff(values, np.array([0.0, 1.0, 1.0]), np.ones_like(values))
    np.testing.assert_allclose(errors[0, 0], np.sqrt(0.01**2 + 0.01**2 / 2))


def test_weighted_ratio_uncertainty_away_from_unity():
    values = np.array([[80.0, 120.0], [50.0, 150.0], [120.0, 80.0]])
    errors = np.array([[2.0, 4.0], [3.0, 5.0], [4.0, 6.0]])
    weights = np.array([0.0, 2.0, 3.0])
    y, e = diff(values, weights, errors)
    z = values / 100
    c = (2 * z[1] + 3 * z[2]) / 5
    vc = (4 * (errors[1] / 100) ** 2 + 9 * (errors[2] / 100) ** 2) / 25
    expected = np.sqrt((errors[0] / 100 / c) ** 2 + (z[0] / c**2) ** 2 * vc)
    np.testing.assert_allclose(e[0, 0], expected)
    np.testing.assert_allclose(y[0, 0], z[0] / c)
    np.testing.assert_allclose(diff(values, weights * 7, errors)[1], e, equal_nan=True)
