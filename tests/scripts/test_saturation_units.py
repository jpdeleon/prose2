"""Saturation is compared in stored pixel units, without a second gain change."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from prose.scripts import run_photometry as rp
from prose.utils import get_saturation_from_header, pixel_unit


@pytest.mark.parametrize(
    "site, band, saturation",
    [
        # SATURATE from MuSCAT3 2023-11-15 and MuSCAT4 2026-11-03 e91 frames.
        ("ogg", "gp", 121600),
        ("ogg", "rp", 120320),
        ("ogg", "ip", 82800),
        ("ogg", "zs", 84000),
        ("coj", "gp", 112000),
        ("coj", "rp", 117760),
        ("coj", "ip", 79407),
        ("coj", "zs", 147200),
    ],
)
def test_banzai_saturation_excludes_only_peaks_at_header_limit(site, band, saturation):
    header = fits.Header(
        dict(TELID="2m0a", SITEID=site, FILTER=band, GAIN=1.0, SATURATE=saturation)
    )
    instrument = "muscat3" if site == "ogg" else "muscat4"
    # The target remains protected even when saturated. A 70 ke- comparison
    # star is below every audited header limit, but above the old ADU limits.
    ref = SimpleNamespace(
        sources=[SimpleNamespace(peak=p) for p in (200000, 70000, saturation)]
    )
    limit, origin = rp.resolve_saturation(header, instrument, None)
    assert (limit, origin) == (saturation, "header")
    assert pixel_unit(header) == "e-"
    assert rp._saturated_source_indices(ref, limit, target_index=0) == [2]
    assert not rp._target_is_saturated(ref, 1, limit)
    assert rp._target_is_saturated(ref, 2, limit)


@pytest.mark.parametrize("site", ["ogg", "coj"])
@pytest.mark.parametrize("band", ["g_narrow", "Na_D", "i_narrow", "z_narrow"])
def test_muscat_narrowband_uses_frame_limit_without_maxlin(site, band):
    header = fits.Header(
        dict(TELID="2m0a", SITEID=site, FILTER=band, GAIN=1.0, SATURATE=112000)
    )
    assert get_saturation_from_header(header)[band] == 112000


@pytest.mark.parametrize(
    "telid, site", [("2m0a", "ogg"), ("1m0a", "coj"), ("0m4a", "coj")]
)
@pytest.mark.parametrize("gain, unit", [(1.0, "electrons"), (1.9, "ADU")])
def test_lco_header_limit_is_already_in_pixel_units(telid, site, gain, unit):
    header = fits.Header(
        dict(
            TELID=telid,
            SITEID=site,
            FILTER="gp",
            GAIN=gain,
            BUNIT=unit,
            SATURATE=64000,
            MAXLIN=60000,
        )
    )
    assert get_saturation_from_header(header)["gp"] == 64000


@pytest.mark.parametrize("bad_limit", [None, "invalid", 0, -1, np.nan, np.inf])
def test_invalid_lco_limit_is_unknown(bad_limit):
    header = dict(TELID="2m0a", SITEID="ogg", FILTER="gp", GAIN=1.0)
    if bad_limit is not None:
        header["SATURATE"] = bad_limit
    telescope = SimpleNamespace(name="LCO-2m/MuSCAT3_gp", saturation=63158)
    assert rp.resolve_saturation(header, "muscat3", telescope) == (None, "unknown")


@pytest.mark.parametrize(
    "telid, site, instrument, band",
    [
        ("1m0a", "coj", "sinistro", "V"),
        ("1m0a", "lsc", "sinistro", "rp*diffuser"),
        ("2m0a", "ogg", "muscat3", "unlisted"),
        ("0m4a", "elp", "qhy600", "Y"),
    ],
)
def test_lco_frame_limit_applies_to_any_filter(telid, site, instrument, band):
    header = fits.Header(
        dict(TELID=telid, SITEID=site, FILTER=band, GAIN=1.0, SATURATE=286000)
    )
    limit, origin = rp.resolve_saturation(header, instrument, None)
    ref = SimpleNamespace(
        sources=[SimpleNamespace(peak=p) for p in (1, 285000, 286000)]
    )
    assert (limit, origin) == (286000, "header")
    assert rp._saturated_source_indices(ref, limit, target_index=0) == [2]


def test_lco_frame_limit_does_not_require_filter_keyword():
    header = fits.Header(dict(TELID="1m0a", SITEID="coj", SATURATE=286000))
    assert rp.resolve_saturation(header, "sinistro", None) == (286000, "header")


def _display_reference(monkeypatch, shared_telescope, own_header, own_telescope):
    from prose import Image

    shared = Image(data=np.zeros((20, 20)))
    shared.telescope = shared_telescope
    frame = SimpleNamespace(header=own_header, telescope=own_telescope)
    monkeypatch.setattr(rp, "FITSImage", lambda _path: frame)
    monkeypatch.setattr(
        rp,
        "_frame_alignment_sequence",
        lambda **_kw: SimpleNamespace(run=lambda img, show_progress=False: None),
    )
    monkeypatch.setattr(
        rp, "_warp_onto_reference", lambda img, shared_ref: np.ones((20, 20))
    )
    display = rp._build_display_reference(
        ["frame.fits"],
        shared,
        ccd_trim_size_yx=(0, 0),
        max_num_stars=12,
        min_star_separation=5.0,
        cutout_size=5,
        min_area=3,
        bad_pixel_map=None,
        n_stack=1,
    )
    return display, shared


_MUSCAT3_ZS = dict(INSTRUME="ep03", TELID="2m0a", SITEID="ogg", FILTER="zs", GAIN=1.0)


@pytest.mark.parametrize(
    "saturate, expected",
    [(84000, 84000), (None, None), (0, None), (-1, None)],
    ids=["valid", "missing", "zero", "negative"],
)
def test_display_reference_uses_only_its_own_frame_limit(
    monkeypatch, saturate, expected
):
    header = dict(_MUSCAT3_ZS)
    if saturate is not None:
        header["SATURATE"] = saturate
    display, shared = _display_reference(
        monkeypatch,
        SimpleNamespace(name="LCO-2m/MuSCAT3_gp", saturation=121600),
        fits.Header(header),
        SimpleNamespace(name="LCO-2m/MuSCAT3_zs", saturation=55000),
    )
    assert display.telescope.saturation == expected
    assert shared.telescope.saturation == 121600


def test_display_reference_keeps_limit_for_unlisted_filter(monkeypatch):
    header = fits.Header(
        dict(TELID="1m0a", SITEID="coj", FILTER="V", GAIN=1.0, SATURATE=286000)
    )
    display, _ = _display_reference(
        monkeypatch,
        SimpleNamespace(name="LCO-1m_gp", saturation=244000),
        header,
        SimpleNamespace(name="LCO-1m_V", saturation=55000),
    )
    assert display.telescope.saturation == 286000


def test_display_reference_muscat2_falls_back_to_own_band_telescope(monkeypatch):
    display, _ = _display_reference(
        monkeypatch,
        SimpleNamespace(name="MuSCAT2_g", saturation=61000, _default=False),
        fits.Header(dict(INSTRUME="MuSCAT2", FILTER="i")),
        SimpleNamespace(name="MuSCAT2_i", saturation=62000, _default=False),
    )
    assert display.telescope.saturation == 62000


@pytest.mark.parametrize("instrument, limit", [("muscat", 55000), ("muscat2", 62000)])
def test_muscat_adu_comparison_exclusion(instrument, limit):
    header = fits.Header(dict(INSTRUME=instrument, FILTER="g"))
    telescope = SimpleNamespace(
        name=f"{instrument}_g", saturation=limit, _default=False
    )
    resolved, _ = rp.resolve_saturation(header, instrument, telescope)
    ref = SimpleNamespace(sources=[SimpleNamespace(peak=p) for p in (1, 54000, 63000)])
    assert pixel_unit(header) == "ADU"
    assert resolved == limit
    assert rp._saturated_source_indices(ref, resolved, target_index=0) == [2]


@pytest.mark.parametrize("unit", ["e-", "ADU"])
def test_csv_pixel_units_do_not_rescale_flux_or_diagnostics(unit):
    diff = SimpleNamespace(
        df=pd.DataFrame(
            dict(time=[2460000.0], flux=[1.01], bkg=[200.0], peak=[70000.0])
        ),
        error=np.array([0.001]),
    )
    result = rp.photometry_df(diff, np.array([2460000.001]), unit=unit)
    assert result[f"Bkg({unit})"].iloc[0] == 200.0
    assert result[f"Peak({unit})"].iloc[0] == 70000.0
    assert result["Flux"].iloc[0] == 1.01
    assert result["Err"].iloc[0] == 0.001
    assert result.columns.tolist()[:3] == ["BJD_TDB", "Flux", "Err"]


def test_quality_ranking_checks_muscat2_adu_limit(monkeypatch):
    from prose import Image
    from prose.core.source import PointSource, Sources

    frame = Image(data=np.zeros((10, 10)))
    frame.header = fits.Header(dict(INSTRUME="MuSCAT2", FILTER="g"))
    frame.telescope = SimpleNamespace(
        name="MuSCAT2_g", saturation=62000, _default=False
    )
    frame.metadata["path"] = "frame.fits"
    frame.fwhm = 3.0
    frame.sources = Sources([PointSource(coords=np.array([5.0, 5.0]), peak=63000)])
    monkeypatch.setattr(rp, "_target_index_or_none", lambda *a: 0)
    monkeypatch.setattr(rp, "_target_wcs_match_found", lambda *a: True)
    monkeypatch.setattr(
        rp, "reference_sequence", lambda **kw: SimpleNamespace(blocks=[])
    )

    class SerialCollector:
        def __init__(self, blocks, data_blocks, name):
            self.collector = data_blocks[0]

        def run(self, paths, **kwargs):
            self.collector.run(frame)

    monkeypatch.setattr(rp, "SequenceParallel", SerialCollector)
    result = rp.pixel_validate_candidates(["frame.fits"], [0], None, {})
    assert len(result) == 1
    assert result[0].target_saturated
