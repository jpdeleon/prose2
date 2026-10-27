import numpy as np
import pytest
import prose.fluxes as module
from prose.fluxes import auto_diff_1d


@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("constant_comparison", [False, True])
def test_single_comparison_is_retained(target, constant_comparison):
    f = 100 + np.random.default_rng(1).normal(size=(2, 100))
    if constant_comparison:
        f[1 - target] = 100
    out, w, _ = auto_diff_1d(f, i=target)
    assert w[target] == 0 and w[1 - target] > 0
    np.testing.assert_allclose(
        out[target],
        (f[target] / f[target].mean()) / (f[1 - target] / f[1 - target].mean()),
    )


def test_exhausted_search_keeps_last_improvement(monkeypatch):
    f = 100 + np.random.default_rng(2).normal(size=(4, 100))
    monkeypatch.setattr(module, "weights", lambda f: np.array([0.0, 3.0, 2.0, 1.0]))
    scores = iter([3.0, 2.0, 1.0])
    monkeypatch.setattr(
        module.utils, "binned_nanstd", lambda f: lambda x: np.full((1, 4), next(scores))
    )
    _, w, _ = auto_diff_1d(f, i=0)
    np.testing.assert_equal(w, [0.0, 3.0, 2.0, 1.0])
