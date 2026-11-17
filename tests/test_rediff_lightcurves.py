"""prose.scripts.rediff_lightcurves: inspect, test and apply a custom
comparison-star set / aperture from a run's flux datacube."""

import json

import numpy as np
import pandas as pd
import pytest

from prose import datacube as dc
from prose.scripts import rediff_lightcurves as rl
from prose.scripts import run_photometry as rp

from test_flux_datacube import TARGET, _cube, _fluxes

STEM = "TOI-0001_muscat3_{band}_250101"


def _results_dir(tmp_path, bands=("g", "r"), cids=None):
    """A run results dir with a datacube and the pipeline's band CSVs."""
    cubes = []
    for i, band in enumerate(bands):
        diff = rp.differential_photometry(
            _fluxes(seed=i), TARGET, cids=cids, keep_datacube=True
        )
        cube = _cube(diff, band=band)
        cubes.append(cube)
        pd.DataFrame(
            {
                "BJD_TDB": cube.bjd_tdb,
                "Flux": diff.flux,
                "Err": diff.error,
                "FWHM": np.full(len(diff.flux), 3.0),
            }
        ).to_csv(tmp_path / f"{STEM.format(band=band)}.csv", index=False)
    dc.save_datacube(tmp_path / "TOI-0001_muscat3_gr_250101_flux_datacube.npz", cubes)
    return tmp_path, {c.band: c for c in cubes}


def _run(argv):
    return rl.run(rl.parse_args([str(a) for a in argv]))


def test_parse_select():
    assert rl.parse_select("g=3,5,7@2") == ("g", [3, 5, 7], 2)
    for bad in ("g=3,5", "g@2", "=3@2", "g=a@2", "g=3@x"):
        with pytest.raises(ValueError):
            rl.parse_select(bad)


def test_info_lists_selectable_stars_and_defaults(tmp_path):
    rdir, cubes = _results_dir(tmp_path)
    out = _run([rdir, "--info"])
    g = out["bands"]["g"]
    assert g["target_id"] == 0
    assert g["selectable_ids"] == [int(i) for i in cubes["g"].selectable_ids]
    assert [c["id"] for c in g["default_comparisons"]] == list(
        cubes["g"].default_comparisons
    )
    assert sum(c["weight"] for c in g["default_comparisons"]) == pytest.approx(1)
    assert g["default_weighting"] == "broeg"
    assert len(g["aper_radii"]) == cubes["g"].fluxes.shape[0]


def test_info_reports_equal_weighting_for_explicit_cid_run(tmp_path):
    rdir, _ = _results_dir(tmp_path, cids=[2, 3])
    assert _run([rdir, "--info"])["bands"]["g"]["default_weighting"] == "equal"


def test_test_mode_reports_metrics_writes_preview_and_leaves_csvs(tmp_path):
    rdir, _ = _results_dir(tmp_path)
    csv = rdir / f"{STEM.format(band='g')}.csv"
    before = csv.read_bytes()
    preview = tmp_path / "out" / "preview.png"
    out = _run([rdir, "--select", "g=2,3@1", "--preview", preview])
    assert out["ok"] and not out["applied"]
    g = out["bands"]["g"]
    assert g["custom"]["comparisons"] == [2, 3]
    assert g["custom"]["aperture"] == 1
    assert g["custom"]["weighting"] == "equal"
    assert g["custom"]["metrics"]["p2p_ppt"] > 0
    assert g["default"]["metrics"]["p2p_ppt"] > 0
    assert preview.stat().st_size > 0
    assert csv.read_bytes() == before
    assert "r" not in out["bands"]  # only the selected band is evaluated


def test_apply_rewrites_flux_and_err_only_and_regenerates_summary(tmp_path):
    rdir, _ = _results_dir(tmp_path)
    saved = dc.load_datacube(
        rl.find_datacube(rdir)
    ).bands  # float32, as the script sees
    csv = rdir / f"{STEM.format(band='g')}.csv"
    r_csv = rdir / f"{STEM.format(band='r')}.csv"
    r_before = r_csv.read_bytes()
    out = _run(
        [
            rdir,
            "--select",
            "g=2,3@1",
            "--apply",
            "--target",
            "TOI-0001",
            "--inst",
            "muscat3",
            "--date",
            "250101",
        ]
    )
    assert out["applied"]
    df = pd.read_csv(csv)
    flux, err = dc.rediff(saved["g"], [2, 3], 1)
    np.testing.assert_allclose(df["Flux"], flux, rtol=1e-12)
    np.testing.assert_allclose(df["Err"], err, rtol=1e-12)
    np.testing.assert_array_equal(df["FWHM"], 3.0)
    assert r_csv.read_bytes() == r_before
    assert (rdir / out["summary_png"]).is_file()


def test_apply_refuses_when_csv_rows_no_longer_match_the_datacube(tmp_path):
    rdir, _ = _results_dir(tmp_path)
    csv = rdir / f"{STEM.format(band='g')}.csv"
    pd.read_csv(csv).iloc[5:].to_csv(csv, index=False)  # as after a post-process clip
    with pytest.raises(ValueError, match="post-process"):
        _run(
            [
                rdir,
                "--select",
                "g=2,3@1",
                "--apply",
                "--target",
                "TOI-0001",
                "--inst",
                "muscat3",
                "--date",
                "250101",
            ]
        )


def test_errors_are_reported_as_json(tmp_path, capsys):
    rdir, _ = _results_dir(tmp_path)
    assert rl.main([str(rdir), "--select", "g=0@1"]) == 1  # 0 is the target
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and "target" in out["error"]


def test_unknown_band_and_missing_datacube_are_errors(tmp_path):
    rdir, _ = _results_dir(tmp_path)
    with pytest.raises(ValueError, match="band"):
        _run([rdir, "--select", "z=2@1"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="datacube"):
        _run([empty, "--info"])


def test_select_or_info_is_required(tmp_path):
    rdir, _ = _results_dir(tmp_path)
    with pytest.raises(ValueError, match="--select"):
        _run([rdir])
