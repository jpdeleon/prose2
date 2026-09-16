"""Tests for ``prose.scripts.postprocess_lightcurves``.

These exercise the CSV-level sigma-clip helpers and the CLI without touching
real photometry output; every path used is a pytest ``tmp_path``.
"""

import numpy as np
import pandas as pd
import pytest

from prose.fluxes import Fluxes, flux_sigma_clip_mask
from prose.scripts import postprocess_lightcurves as pp
from prose.scripts.postprocess_lightcurves import read_lightcurve


def _make_csv(path, *, n=200, spike_idx=(10, 77), err=None):
    rng = np.random.default_rng(7)
    time = 2461200.0 + np.cumsum(rng.uniform(0.01, 0.02, n))
    flux = 1.0 + rng.normal(0, 0.001, n)
    flux[list(spike_idx)] += rng.choice([-1, 1], len(spike_idx)) * 0.02
    if err is None:
        err = np.full(n, 0.0015)
    pd.DataFrame({"BJD_TDB": time, "Flux": flux, "Err": err}).to_csv(path, index=False)


def _write_band_run(tmp_path, bands=("gp", "rp")):
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    rng = np.random.default_rng(11)
    stems = []
    for band in bands:
        stem = f"KIC_muscat2_{band}_260810"
        stems.append(stem)
        n = 120
        flux = 1.0 + rng.normal(0, 0.001, n)
        flux[5] += 0.04
        _make_csv(results_dir / f"{stem}.csv", n=n, err=None)
        df = pd.read_csv(results_dir / f"{stem}.csv")
        df["Flux"] = flux
        df.to_csv(results_dir / f"{stem}.csv", index=False)
    return results_dir, stems


@pytest.fixture
def band_run(tmp_path):
    results_dir, stems = _write_band_run(tmp_path)
    return results_dir, stems


def test_flux_sigma_clip_mask_parity_with_method():
    rng = np.random.default_rng(42)
    n = 500
    time = 2461200.0 + np.cumsum(rng.uniform(0.01, 0.02, n))
    base = 40000.0 + (time - time.min()) * 800.0 + (time - time.min()) ** 2 * 50.0
    flux = base + rng.normal(0, 120, n)
    out_idx = [40, 150, 333]
    flux[out_idx] += rng.choice([-1, 1], 3) * rng.uniform(2500, 5000, 3)

    f = Fluxes(flux, time=time, errors=np.ones(n), target=0, aperture=0)
    method_mask = f.sigma_clip_flux_poly(sigma=6.0, degree=2, return_mask=True)
    helper_mask = flux_sigma_clip_mask(time, flux, sigma=6.0, degree=2, iterations=5)
    assert np.array_equal(method_mask, helper_mask)
    assert np.count_nonzero(~helper_mask) == 3


def test_clip_df_rejects_spikes_keeps_nan_err(tmp_path):
    path = tmp_path / "TOI123_muscat2_g_260811.csv"
    n = 200
    err = np.full(n, 0.0015)
    err[3] = np.nan  # unknown error: kept
    err[99] = 0.0  # invalid error: rejected
    err[120] = -1.0  # invalid error: rejected
    _make_csv(path, n=n, err=err, spike_idx=(10, 77))
    df = read_lightcurve(path)
    mask, stats = pp.clip_df(df, sigma=5.0, degree=2, iterations=5)
    assert stats["n"] == n
    assert not mask[10]
    assert not mask[77]
    assert not mask[99]
    assert not mask[120]
    assert mask[3]
    assert stats["n_kept"] == int(mask.sum())
    assert stats["n_clipped"] == n - stats["n_kept"]


def test_clip_df_excludes_jd_range_before_fitting(tmp_path):
    n = 150
    time = 2461200.0 + np.arange(n) * 0.01
    flux = np.ones(n)
    flux[10] += 0.02  # a real outlier inside the kept range
    # A large instrumental drift confined to the tail: if it were included in
    # the trend fit it would pull the polynomial and change which in-range
    # points look like outliers.
    flux[120:] += np.linspace(0, 5.0, n - 120)
    err = np.full(n, 0.001)
    df = pd.DataFrame({"BJD_TDB": time, "Flux": flux, "Err": err})

    cutoff = time[119] + 0.005  # strictly between index 119 and 120
    mask, stats = pp.clip_df(
        df, sigma=5.0, degree=2, iterations=5, exclude_after_jd=cutoff
    )
    assert stats["n_excluded_range"] == n - 120
    assert not mask[120:].any()
    assert not mask[10]

    expected_in_range = flux_sigma_clip_mask(
        time[:120], flux[:120], sigma=5.0, degree=2, iterations=5
    )
    assert np.array_equal(mask[:120], expected_in_range)


def test_clip_df_excludes_before_and_after_jd(tmp_path):
    n = 100
    time = 2461200.0 + np.arange(n) * 0.01
    flux = np.ones(n)
    err = np.full(n, 0.001)
    df = pd.DataFrame({"BJD_TDB": time, "Flux": flux, "Err": err})

    mask, stats = pp.clip_df(
        df,
        sigma=5.0,
        degree=2,
        iterations=5,
        exclude_before_jd=time[10],
        exclude_after_jd=time[89],
    )
    assert stats["n_excluded_range"] == 20
    assert not mask[:10].any()
    assert mask[10:90].all()
    assert not mask[90:].any()


def test_clip_df_no_exclusion_reports_zero_excluded_range(tmp_path):
    path = tmp_path / "TOI123_muscat2_g_260811.csv"
    _make_csv(path)
    df = read_lightcurve(path)
    _, stats = pp.clip_df(df, sigma=5.0, degree=2, iterations=5)
    assert stats["n_excluded_range"] == 0


def test_clip_df_sigma_none_skips_clip_but_keeps_err_filter(tmp_path):
    n = 100
    time = 2461200.0 + np.arange(n) * 0.01
    flux = np.ones(n)
    flux[10] += 0.5  # would normally be clipped as a spike
    err = np.full(n, 0.001)
    err[20] = -1.0  # always rejected regardless of sigma
    df = pd.DataFrame({"BJD_TDB": time, "Flux": flux, "Err": err})

    mask, stats = pp.clip_df(df, sigma=None, degree=2, iterations=5)
    assert mask[10]  # the spike is kept: sigma-clip did not run
    assert not mask[20]  # invalid Err is still always rejected
    assert stats["n_clipped"] == 1
    assert stats["sigma"] is None


def test_clip_df_sigma_none_combines_with_jd_range(tmp_path):
    n = 100
    time = 2461200.0 + np.arange(n) * 0.01
    flux = np.ones(n)
    err = np.full(n, 0.001)
    df = pd.DataFrame({"BJD_TDB": time, "Flux": flux, "Err": err})

    cutoff = time[59] + 0.005
    mask, stats = pp.clip_df(
        df, sigma=None, degree=2, iterations=5, exclude_after_jd=cutoff
    )
    assert mask[:60].all()
    assert not mask[60:].any()
    assert stats["n_excluded_range"] == 40


def test_sigma_arg_parses_none_off_and_blank():
    assert pp._sigma_arg("none") is None
    assert pp._sigma_arg("None") is None
    assert pp._sigma_arg("off") is None
    assert pp._sigma_arg("") is None
    assert pp._sigma_arg("5.0") == 5.0


def test_parse_args_sigma_none_disables_clipping(tmp_path):
    args = pp.parse_args([str(tmp_path), "--sigma", "none"])
    assert args.sigma is None


def test_run_with_sigma_none_apply_only_excludes_range(tmp_path, band_run):
    results_dir, stems = band_run
    df0 = pd.read_csv(results_dir / f"{stems[0]}.csv")
    cutoff = float(df0["BJD_TDB"].iloc[99])  # keep all 120 rows of this band

    args = pp.parse_args(
        [
            str(results_dir),
            "--sigma",
            "none",
            "--exclude-after-jd",
            str(cutoff),
            "--apply",
            "--target",
            "KIC",
            "--inst",
            "muscat2",
            "--date",
            "260810",
        ]
    )
    report = pp.run(args)
    assert report["ok"] and report["applied"]
    assert report["sigma"] is None
    # band_run's fixture injects a spike at index 5 (well within range); with
    # sigma-clipping off it must survive, so only the JD-range cut (rows
    # after index 99) is reflected in n_clipped.
    assert report["files"][0]["n_clipped"] == report["files"][0]["n_excluded_range"]


def test_clip_df_falls_back_to_row_index_time(tmp_path):
    path = tmp_path / "TOI123_muscat2_g_260811.csv"
    rng = np.random.default_rng(3)
    flux = 1.0 + rng.normal(0, 0.001, 150)
    flux[9] += 0.05
    pd.DataFrame({"Flux": flux, "Err": np.full(150, 0.0015)}).to_csv(path, index=False)
    df = read_lightcurve(path)
    mask, _ = pp.clip_df(df, sigma=5.0, degree=2, iterations=5)
    assert mask.dtype == bool
    assert len(mask) == len(df)


def test_read_lightcurve_missing_flux_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"BJD_TDB": [1.0, 2.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Flux"):
        read_lightcurve(path)


def test_parse_stem_plain_multisite_and_narrow_band():
    plain = pp.parse_stem("TOI123_muscat2_g_260811")
    assert plain["target"] == "TOI123"
    assert plain["inst"] == "muscat2"
    assert plain["site"] is None
    assert plain["tel"] is None
    assert plain["band"] == "g"
    assert plain["date"] == "260811"

    multi = pp.parse_stem("TOI123_sbig_coj_tel0m4_g_narrow_260811_full")
    assert multi["site"] == "coj"
    assert multi["tel"] == "tel0m4"
    assert multi["band"] == "g_narrow"
    assert multi["date"] == "260811"
    assert multi["confmode"] == "full"

    narrow = pp.parse_stem("TOI123_muscat2_g_narrow_260811")
    assert narrow["band"] == "g_narrow"


def test_band_csvs_filters_other_csv_products(tmp_path):
    results_dir = tmp_path
    lightcurve = "TOI123_muscat2_g_260811.csv"
    _make_csv(results_dir / lightcurve)
    pd.DataFrame({"id": [1]}).to_csv(
        results_dir / "TOI123_muscat2_g_rp_ip_zs_260811_nearby_stars.csv", index=False
    )
    (results_dir / "notes.txt").write_text("x")
    found = [p.name for p in pp.band_csvs(results_dir)]
    assert found == [lightcurve]


def test_run_with_sigma_none_preview_writes_file_without_crashing(tmp_path, band_run):
    results_dir, _ = band_run
    preview_path = tmp_path / "preview.png"
    args = pp.parse_args(
        [str(results_dir), "--sigma", "none", "--preview", str(preview_path)]
    )
    report = pp.run(args)
    assert report["ok"]
    assert report["sigma"] is None
    assert preview_path.is_file()


def test_plot_preview_and_apply_end_to_end(tmp_path, band_run):
    results_dir, stems = band_run
    (results_dir / f"{stems[0]}_{stems[1]}_lightcurves.png").write_bytes(b"old")

    args = pp.parse_args(
        [
            str(results_dir),
            "--sigma",
            "5",
            "--degree",
            "2",
            "--preview",
            str(tmp_path / "preview.png"),
        ]
    )
    report = pp.run(args)
    assert report["ok"]
    assert report["n_files"] == 2
    assert report["files"][0]["n"] == 120
    assert report["files"][0]["n_clipped"] == 1
    assert (tmp_path / "preview.png").is_file()

    args = pp.parse_args(
        [
            str(results_dir),
            "--sigma",
            "5",
            "--degree",
            "2",
            "--apply",
            "--target",
            "KIC",
            "--inst",
            "muscat2",
            "--date",
            "260810",
        ]
    )
    report = pp.run(args)
    assert report["applied"]
    assert report["summary_png"] == f"{stems[0]}_{stems[1]}_lightcurves.png"
    df = pd.read_csv(results_dir / f"{stems[0]}.csv")
    assert len(df) == 119
    assert (results_dir / f"{stems[0]}_{stems[1]}_lightcurves.png").stat().st_size > 100


def test_apply_outliers_honored(tmp_path, band_run):
    results_dir, stems = band_run
    args = pp.parse_args(
        [
            str(results_dir),
            "--sigma",
            "5",
            "--degree",
            "2",
            "--apply",
            "--target",
            "KIC",
            "--inst",
            "muscat2",
            "--date",
            "260810",
        ]
    )
    report = pp.run(args)
    assert report["files"][0]["n_clipped"] == 1
    assert report["files"][0]["n_kept"] == 119


def test_apply_requires_context_args(tmp_path, band_run):
    results_dir, _ = band_run
    args = pp.parse_args([str(results_dir), "--apply"])
    with pytest.raises(ValueError, match="--apply requires"):
        pp.run(args)


def test_run_missing_results_dir():
    args = pp.parse_args(["/nonexistent/results"])
    with pytest.raises(FileNotFoundError):
        pp.run(args)


def test_parse_args_jd_exclusion_defaults_to_none(tmp_path):
    args = pp.parse_args([str(tmp_path)])
    assert args.exclude_before_jd is None
    assert args.exclude_after_jd is None


def test_run_rejects_before_not_less_than_after(tmp_path, band_run):
    results_dir, _ = band_run
    args = pp.parse_args(
        [
            str(results_dir),
            "--exclude-before-jd",
            "2461200.5",
            "--exclude-after-jd",
            "2461200.5",
        ]
    )
    with pytest.raises(ValueError, match="exclude-before-jd.*exclude-after-jd"):
        pp.run(args)


def test_run_with_jd_exclusion_apply_drops_rows_and_reports(tmp_path, band_run):
    results_dir, stems = band_run
    df0 = pd.read_csv(results_dir / f"{stems[0]}.csv")
    cutoff = float(df0["BJD_TDB"].iloc[59]) + 1e-6  # keep the first 60, drop the rest

    args = pp.parse_args(
        [
            str(results_dir),
            "--sigma",
            "5",
            "--degree",
            "2",
            "--exclude-after-jd",
            str(cutoff),
            "--apply",
            "--target",
            "KIC",
            "--inst",
            "muscat2",
            "--date",
            "260810",
        ]
    )
    report = pp.run(args)
    assert report["ok"] and report["applied"]
    assert report["exclude_after_jd"] == pytest.approx(cutoff)
    assert report["exclude_before_jd"] is None
    assert report["files"][0]["n_excluded_range"] == 60
    df_after = pd.read_csv(results_dir / f"{stems[0]}.csv")
    assert len(df_after) == report["files"][0]["n_kept"]
    assert len(df_after) <= 60


def test_run_with_jd_exclusion_preview_writes_file(tmp_path, band_run):
    results_dir, _ = band_run
    df0 = pd.read_csv(results_dir / "KIC_muscat2_gp_260810.csv")
    cutoff = float(df0["BJD_TDB"].iloc[59])
    preview_path = tmp_path / "preview.png"
    args = pp.parse_args(
        [
            str(results_dir),
            "--exclude-before-jd",
            str(cutoff),
            "--preview",
            str(preview_path),
        ]
    )
    report = pp.run(args)
    assert report["ok"]
    assert report["exclude_before_jd"] == pytest.approx(cutoff)
    assert preview_path.is_file()
