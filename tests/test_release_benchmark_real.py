"""Fixed-aperture release check on real data (host only).

Runs the ``fixed`` datasets of benchmarks/datasets.yaml with two installs and
fails on any photometry change. Skipped unless both interpreters are given::

    PROSE_BENCH_BASELINE=~/miniconda3/envs/<previous>/bin/python \\
    PROSE_BENCH_CANDIDATE=~/miniconda3/envs/<candidate>/bin/python \\
    TMPDIR=~/temp uv run pytest -m benchmark tests/test_release_benchmark_real.py

For the full report (automatic selection too), run
``python -m prose.scripts.release_benchmark``; see benchmarks/README.md.
"""

import os
from pathlib import Path

import pytest

from prose.scripts import release_benchmark as rb

pytestmark = pytest.mark.benchmark

_ROOT = Path(__file__).resolve().parents[1]
_DATASETS = rb.load_datasets(_ROOT / "benchmarks/datasets.yaml")
_FIXED = [d for d in _DATASETS if d.kind == "fixed"]


@pytest.mark.parametrize("dataset", _FIXED, ids=[d.name for d in _FIXED])
def test_fixed_photometry_matches_previous_release(tmp_path, dataset):
    interpreters = {
        label: os.environ.get(f"PROSE_BENCH_{label.upper()}")
        for label in ("baseline", "candidate")
    }
    if not all(interpreters.values()):
        pytest.skip("set PROSE_BENCH_BASELINE and PROSE_BENCH_CANDIDATE")
    if not dataset.data_dir.is_dir():
        pytest.skip(f"{dataset.data_dir} not found")
    if dataset.calib_cache is not None and not dataset.calib_cache.is_dir():
        pytest.skip(f"{dataset.calib_cache} not found")
    runs = {
        label: rb.run_photometry(
            str(Path(python).expanduser()),
            dataset,
            tmp_path / label,
            2,
            rb.telescope_files(_ROOT / "data"),
        )
        for label, python in interpreters.items()
    }
    results = rb.evaluate(dataset, runs)
    failures = {r.band: r.failures for r in results if r.failures}
    assert not failures, failures
