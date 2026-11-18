"""Release benchmark harness on synthetic light curves (no raw data needed)."""

import json
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from prose.scripts import release_benchmark as rb

REPO_DATASETS = Path(__file__).resolve().parents[1] / "benchmarks/datasets.yaml"


def _curve(n=200, flux_offset=0.0, bjd_offset_s=0.0, seed=1):
    rng = np.random.default_rng(seed)
    gjd = 2460000.5 + np.arange(n) * 60 / 86400
    return pd.DataFrame(
        {
            "BJD_TDB": gjd + 0.003 + bjd_offset_s / 86400,
            "Flux": 1 + 1e-3 * rng.standard_normal(n) + flux_offset,
            "Err": np.full(n, 1e-3),
            "Peak(e-)": np.full(n, 1e4),
            "GJD_UTC": gjd,
        }
    )


VALID = {
    "name": "a",
    "kind": "fixed",
    "data_dir": "/data/X",
    "target_name": "T",
    "target_coord": "10.0 -20.0",
    "bands": ["gp"],
    "args": ["--cID", 1],
}


def _entry(**changes):
    """VALID with fields replaced; a value of None removes the field."""
    entry = {**VALID, **changes}
    return {key: value for key, value in entry.items() if value is not None}


def _write(tmp_path, *entries):
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump({"datasets": list(entries)}))
    return path


# --------------------------------------------------------------- datasets


def test_repository_datasets_load_and_cover_both_kinds():
    datasets = rb.load_datasets(REPO_DATASETS)
    assert {d.kind for d in datasets} == {"fixed", "auto"}
    assert len({d.name for d in datasets}) == len(datasets)


def test_load_datasets_reads_fields(tmp_path):
    (dataset,) = rb.load_datasets(_write(tmp_path, VALID))
    assert dataset.bands == ("gp",)
    assert dataset.args == ("--cID", "1")
    assert dataset.calib_cache is None


@pytest.mark.parametrize(
    "entries, message",
    [
        ((), "no 'datasets' list"),
        (("gp",), "must be a mapping"),
        ((_entry(kind="other"),), "kind must be"),
        ((_entry(bands=None),), "missing"),
        ((_entry(bands=[]),), "must not be empty"),
        ((_entry(target_coord=10.0),), "target_coord must be"),
        ((VALID, VALID), "duplicate"),
    ],
)
def test_load_datasets_rejects_invalid_entries(tmp_path, entries, message):
    with pytest.raises(ValueError, match=message):
        rb.load_datasets(_write(tmp_path, *entries))


def test_select_datasets_filters_and_rejects_unknown_names(tmp_path):
    datasets = rb.load_datasets(REPO_DATASETS)
    assert all(d.kind == "fixed" for d in rb.select_datasets(datasets, kind="fixed"))
    name = datasets[0].name
    assert [d.name for d in rb.select_datasets(datasets, only=[name])] == [name]
    with pytest.raises(SystemExit):
        rb.select_datasets(datasets, only=["nope"])


# ---------------------------------------------------------------- command


def test_negative_declination_is_not_read_as_a_flag():
    assert rb.coord_args("252.2 -44.2") == ["252.2", " -44.2"]
    assert rb.coord_args("252.2 44.2") == ["252.2", "44.2"]


def test_build_command_passes_dataset_args_and_private_calibration(tmp_path):
    (dataset,) = rb.load_datasets(_write(tmp_path, _entry(calib_cache="/c")))
    command = rb.build_command("/py", dataset, tmp_path, workers=3)
    assert command[:4] == ["/py", "-I", "-m", "prose.scripts.run_photometry"]
    assert command[command.index("--cID") + 1] == "1"
    assert command[command.index("--workers") + 1] == "3"
    assert command[command.index("--calib_dir") + 1] == str(tmp_path / "calibrated")
    assert command[command.index("--plots") + 1] == "none"


# ---------------------------------------------------------- fixed compare


def test_identical_fixed_curves_pass():
    assert rb.compare_fixed(_curve(), _curve()) == []


def test_fixed_compare_matches_frames_by_time_not_row():
    baseline = _curve()
    candidate = baseline.drop(index=[5]).sample(frac=1, random_state=0)
    candidate = candidate.sort_values("GJD_UTC").reset_index(drop=True)
    assert rb.compare_fixed(baseline, candidate) == []


def test_fixed_compare_fails_on_a_flux_change():
    (failure,) = rb.compare_fixed(_curve(), _curve(flux_offset=2e-3))
    assert "flux differs" in failure


def test_fixed_compare_allows_epoch_bjd_shift_but_not_more():
    assert rb.compare_fixed(_curve(), _curve(bjd_offset_s=0.01)) == []
    (failure,) = rb.compare_fixed(_curve(), _curve(bjd_offset_s=0.2))
    assert "BJD_TDB" in failure


def test_fixed_compare_fails_on_frame_loss_and_removed_columns():
    failures = rb.compare_fixed(_curve(), _curve().iloc[:150].drop(columns="Peak(e-)"))
    assert any("frames 200 -> 150" in f for f in failures)
    assert any("columns removed" in f for f in failures)


def test_fixed_compare_fails_on_unmatched_and_invalid_frames():
    candidate = _curve()
    candidate.loc[3, "GJD_UTC"] += 30 / 86400
    candidate.loc[7, "Err"] = 0
    failures = rb.compare_fixed(_curve(), candidate)
    assert any("no baseline counterpart" in f for f in failures)
    assert any("non-positive errors" in f for f in failures)


# ------------------------------------------------------------ auto review


def _summary(frames=200, ptp=1000.0, binned=300.0):
    return {"frames": frames, "ptp_ppm": ptp, "binned_ppm": binned}


def test_review_flags_noisier_or_missing_bands_only():
    assert rb.review_flags(_summary(), _summary(ptp=1100)) == []
    assert rb.review_flags(_summary(), _summary(binned=413)) == [
        "binned_ppm 300 -> 413"
    ]
    assert rb.review_flags(_summary(), None) == ["candidate produced no light curve"]
    assert rb.review_flags(None, _summary()) == []
    assert rb.review_flags(_summary(), _summary(frames=150)) == ["frames 200 -> 150"]


def test_binned_scatter_averages_ten_minute_bins():
    time_days = np.arange(600) * 10 / 86400  # 10-s cadence, 100 min
    flux = np.where(np.arange(600) < 300, 1.0, 1.01)
    assert rb.binned_scatter(time_days, flux) == pytest.approx(0.005, rel=0.05)
    assert np.isnan(rb.binned_scatter(time_days[:2], flux[:2]))


# ---------------------------------------------------------------- outputs


def _write_outputs(directory, band="gp", curve=None, radii=(8.0, 10.0), aperture=1):
    directory.mkdir(parents=True, exist_ok=True)
    (curve if curve is not None else _curve()).to_csv(
        directory / f"T_inst_{band}_200101.csv", index=False
    )
    pd.DataFrame({"x": [1]}).to_csv(
        directory / f"T_inst_{band}_200101_nearby_stars.csv"
    )
    np.savez(
        directory / "T_inst_200101.npz",
        **{
            f"{band}__aper_radii": np.array(radii),
            f"{band}__aperture": np.array(aperture),
            f"{band}__comparisons": np.array([2, 0]),
        },
    )


def test_band_outputs_reads_curve_and_selected_aperture(tmp_path):
    _write_outputs(tmp_path)
    curve, meta = rb.band_outputs(tmp_path, "gp")
    assert len(curve) == 200
    assert meta == {"aperture_pix": 10.0, "comparisons": [2, 0]}
    assert rb.band_outputs(tmp_path, "rp") == (None, {})


def test_evaluate_fails_fixed_band_when_target_was_not_located(tmp_path):
    (dataset,) = rb.load_datasets(_write(tmp_path, VALID))
    runs = {}
    for label in ("baseline", "candidate"):
        _write_outputs(tmp_path / label)
        log = "Target not found in detected sources" if label == "candidate" else ""
        runs[label] = rb.RunResult(tmp_path / label, 0, 1.0, log)
    (result,) = rb.evaluate(dataset, runs)
    assert result.failures == ("candidate log: 'Target not found'",)


def test_evaluate_reports_added_columns_and_nonzero_exit(tmp_path):
    (dataset,) = rb.load_datasets(_write(tmp_path, _entry(kind="auto")))
    _write_outputs(tmp_path / "baseline")
    _write_outputs(tmp_path / "candidate", curve=_curve().assign(New=1.0))
    runs = {
        "baseline": rb.RunResult(tmp_path / "baseline", 0, 1.0, ""),
        "candidate": rb.RunResult(tmp_path / "candidate", 3, 2.0, ""),
    }
    (result,) = rb.evaluate(dataset, runs)
    assert result.failures == () and result.flags == ()
    assert result.notes == ("candidate exit 3", "new columns: ['New']")
    report = rb.render_report([result], "/b", "/c", skipped=("gone",))
    assert "| a | auto | gp | 200 → 200 | 10.00 → 10.00 | (2, 0) → (2, 0) |" in report
    assert "Skipped (raw data not found): gone" in report


# ------------------------------------------------------------------- main

FAKE_ENGINE = """#!{python}
import sys
from pathlib import Path
import numpy as np
argv = sys.argv
out = Path(argv[argv.index("--results_dir") + 1])
assert argv[argv.index("--target_coord") + 2] == " -20.0"
n = 100
gjd = 2460000.5 + np.arange(n) * 60 / 86400
flux = 1 + 1e-3 * np.sin(np.arange(n)) + {offset}
rows = ["BJD_TDB,Flux,Err,GJD_UTC"] + [
    f"{{g + 0.003!r}},{{f!r}},0.001,{{g!r}}" for g, f in zip(gjd, flux)
]
(out / "T_x_gp_200101.csv").write_text("\\n".join(rows) + "\\n")
print("HOME", __import__("os").environ["HOME"])
"""


def _fake_engine(path, offset):
    path.write_text(FAKE_ENGINE.format(python=sys.executable, offset=offset))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.mark.parametrize("offset, code", [(0.0, 0), (5e-3, 1)])
def test_main_runs_both_installs_and_exits_on_fixed_failure(tmp_path, offset, code):
    data = tmp_path / "raw"
    data.mkdir()
    datasets = _write(tmp_path, _entry(data_dir=str(data)), _entry(name="missing"))
    baseline = _fake_engine(tmp_path / "baseline.py", 0.0)
    candidate = _fake_engine(tmp_path / "candidate.py", offset)
    telescopes = tmp_path / "telescopes"
    telescopes.mkdir()
    (telescopes / "muscat2_g.telescope").write_text("name: muscat2_g\n")
    out = tmp_path / "out"
    argv = ["--baseline", baseline, "--candidate", candidate,
            "--datasets", str(datasets), "--out", str(out),
            "--telescopes", str(telescopes)]  # fmt: skip
    assert rb.main(argv) == code
    report = (out / "report.md").read_text()
    assert "Skipped (raw data not found): missing" in report
    (record,) = json.loads((out / "report.json").read_text())
    assert bool(record["failures"]) == bool(code)
    log = (out / "a" / "candidate" / "run.log").read_text()
    assert f"HOME {out / 'a' / 'candidate' / 'home'}" in log
    for label in ("baseline", "candidate"):
        prose_home = out / "a" / label / "home" / ".prose"
        assert (prose_home / "muscat2_g.telescope").read_text() == "name: muscat2_g\n"


def test_repository_telescope_files_cover_every_instrument():
    names = {p.name for p in rb.telescope_files(REPO_DATASETS.parents[1] / "data")}
    for instrument in ("muscat_g", "muscat2_g", "muscat3_gp", "muscat4_gp",
                       "sinistro", "sbig", "qhy600"):  # fmt: skip
        assert f"{instrument}.telescope" in names


def test_main_refuses_a_directory_without_telescope_files(tmp_path):
    datasets = _write(tmp_path, VALID)
    argv = ["--baseline", sys.executable, "--candidate", sys.executable,
            "--datasets", str(datasets), "--out", str(tmp_path / "out"),
            "--telescopes", str(tmp_path)]  # fmt: skip
    with pytest.raises(ValueError, match="no \\*.telescope files"):
        rb.main(argv)
    assert not (tmp_path / "out").exists()
