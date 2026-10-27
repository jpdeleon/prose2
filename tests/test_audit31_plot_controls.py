import pytest
from prose.scripts import run_photometry as rp


@pytest.mark.parametrize("group", ["all", "reference", "summary", "none"])
def test_plot_group_option(tmp_path, group):
    args = rp.parse_args(
        [
            "--target_name",
            "test",
            "--data_dir",
            str(tmp_path),
            "--results_dir",
            str(tmp_path / "out"),
            "--plots",
            group,
        ]
    )
    assert args.plots == group


@pytest.mark.parametrize("group", ["all", "reference", "summary", "none"])
def test_plot_groups_control_rendering_without_losing_data(
    tmp_path, monkeypatch, group
):
    import numpy as np
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    from prose import FITSImage, Fluxes
    from prose.core.source import Sources, PointSource

    w = WCS(naxis=2)
    w.wcs.crpix = [10, 10]
    w.wcs.cdelt = [0.01, 0.01]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    h = w.to_header()
    for k, v in {
        "TELID": "1m0a",
        "INSTRUME": "sinistro",
        "SITEID": "lsc",
        "CONFMODE": "central_2k_2x2",
        "OBJECT": "audit",
        "EXPTIME": 1.0,
        "FILTER": "gp",
        "JD": 2460000.0,
        "DATE-OBS": "2025-04-16T00:00:00",
    }.items():
        h[k] = v
    path = tmp_path / "science.fits"
    fits.writeto(path, np.ones((20, 20)), h)
    image = FITSImage(path)
    image.telescope.pixel_scale = 0.4
    image.sources = Sources(
        [PointSource(coords=np.array([10.0, 10.0]))], type="PointSource"
    )
    diff = Fluxes(
        np.array([1.0, 1.01, 0.99]),
        time=np.arange(3.0) + 2460000.0,
        errors=np.ones(3) * 0.01,
    )
    result = dict(
        ref=image,
        diff=diff,
        target_index=0,
        files=[path],
        gaia_df=pd.DataFrame(
            {"ra": [0.0], "dec": [0.0], "phot_g_mean_mag": [12.0], "source_id": ["1"]}
        ),
        rout=5.0,
    )
    monkeypatch.setattr(rp, "run_band", lambda *a, **kw: result)
    monkeypatch.setattr(rp, "compute_bjd_tdb", lambda *a, **kw: diff.time)
    monkeypatch.setattr(rp, "frames_from_obslog", lambda *a, **kw: None)
    calls = []
    names = [
        "plot_ref_image",
        "plot_apertures",
        "plot_alignment",
        "plot_cutouts",
        "plot_stacks",
        "plot_lightcurves",
        "plot_raw_flux",
        "plot_covariates",
        "save_all_bands_npz",
    ]
    for name in names:
        monkeypatch.setattr(rp, name, lambda *a, _name=name, **kw: calls.append(_name))
    monkeypatch.setattr(rp, "get_simbad_data", lambda *a: calls.append("simbad"))
    out = tmp_path / "out"
    code = rp.main(
        [
            "--target_name",
            "audit",
            "--target_coord",
            "0",
            "0",
            "--data_dir",
            str(tmp_path),
            "--results_dir",
            str(out),
            "--bands",
            "gp",
            "--glob",
            "*.fits",
            "--plots",
            group,
        ]
    )
    assert code == 0
    references = set(names[:5])
    summaries = set(names[5:8])
    assert (set(calls) & references) == (
        references if group in ("all", "reference") else set()
    )
    assert (set(calls) & summaries) == (
        summaries if group in ("all", "summary") else set()
    )
    assert "save_all_bands_npz" in calls
    assert (
        len([p for p in out.glob("*.csv") if not p.name.endswith("nearby_stars.csv")])
        == 1
    )
    assert ("simbad" in calls) == (group in ("all", "reference"))
