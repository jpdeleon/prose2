from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits import Header

from prose import CONFIG, FITSImage, Image, Telescope


def test_creation(name="test_telescope"):
    Telescope(name=name, save=True)
    assert name in CONFIG.build_telescopes_dict().keys()


def test_custom_header_date(tmp_path):
    im = Image()
    im.header = Header()

    keyword = "OBSTIME"
    value = "2023:02:16:19:38:54.250"

    im.header[keyword] = value
    im.writeto(tmp_path / "test.fits")

    telescope = Telescope(keyword_observation_date=keyword)
    telescope.date_string_format = "%Y:%m:%d:%H:%M:%S.%f"

    im = FITSImage(tmp_path / "test.fits", load_data=False, telescope=telescope)

    assert im.date == datetime(2023, 2, 16, 19, 38, 54, 250000)


def test_fitsimage_resolves_telescope_from_header(tmp_path):
    telescope_name = "test_telescope_resolve"
    Telescope(name=telescope_name, pixel_scale=0.42, save=True)

    data = np.ones((20, 20))
    hdr = fits.Header()
    hdr["INSTRUME"] = telescope_name
    hdr["TELESCOP"] = telescope_name
    hdr["EXPTIME"] = 1.0
    hdr["DATE-OBS"] = "2025-01-01T00:00:00"
    hdr["FILTER"] = "gp"
    hdr["JD"] = 2460000.0
    hdr["RA"] = 0.0
    hdr["DEC"] = 0.0
    fpath = tmp_path / "test.fits"
    fits.writeto(fpath, data, header=hdr)

    im = FITSImage(fpath, load_data=True)
    assert im.telescope.name == telescope_name
    assert abs(im.telescope.pixel_scale - 0.42) < 1e-6


def test_fitsimage_header_keywords_do_not_modify_telescope(tmp_path):
    telescope = Telescope(
        name="test_telescope_immutable",
        pixel_scale=0.42,
        gain=2.0,
        read_noise=10.0,
        saturation=50000,
    )

    data = np.ones((20, 20))
    hdr = fits.Header()
    hdr["INSTRUME"] = "some_instrument"
    hdr["TELESCOP"] = "some_telescope"
    hdr["EXPTIME"] = 99.0
    hdr["DATE-OBS"] = "2025-06-01T12:00:00"
    hdr["FILTER"] = "ip"
    hdr["JD"] = 2460000.0
    hdr["GAIN"] = 5.0
    fpath = tmp_path / "test.fits"
    fits.writeto(fpath, data, header=hdr)

    im = FITSImage(fpath, load_data=True, telescope=telescope)
    assert im.metadata["exposure"] == 99.0
    assert im.metadata["filter"] == "ip"
    assert im.telescope.name == "test_telescope_immutable"
    assert abs(im.telescope.pixel_scale - 0.42) < 1e-6
    assert abs(im.telescope.gain - 2.0) < 1e-6
    assert abs(im.telescope.read_noise - 10.0) < 1e-6
    assert abs(im.telescope.saturation - 50000) < 1e-6


def _write_fits(tmp_path, name="f.fits", **extra):
    hdr = fits.Header()
    hdr["INSTRUME"] = "some_instrument"
    hdr["TELESCOP"] = "some_telescope"
    hdr["EXPTIME"] = 10.0
    hdr["DATE-OBS"] = "2025-06-01T12:00:00"
    hdr["FILTER"] = "ip"
    hdr["JD"] = 2460000.0
    for key, value in extra.items():
        hdr[key] = value
    fpath = tmp_path / name
    fits.writeto(fpath, np.ones((20, 20)), header=hdr)
    return fpath


def test_header_gain_and_read_noise_override_when_keywords_are_set(tmp_path):
    # BANZAI headers carry per-frame GAIN (e-/ADU, 1.0 once calibrated) and
    # RDNOISE (e-), which depend on camera unit and readout mode; a single
    # .telescope value cannot represent that.
    telescope = Telescope(
        name="test_header_noise",
        gain=1.8,
        read_noise=6.0,
        keyword_gain="GAIN",
        keyword_read_noise="RDNOISE",
    )
    fpath = _write_fits(tmp_path, GAIN=1.0, RDNOISE=12.0)

    im = FITSImage(fpath, load_data=True, telescope=telescope)

    assert im.metadata["gain"] == 1.0
    assert im.metadata["read_noise"] == 12.0
    # the telescope definition itself is never mutated by header values
    assert im.telescope.gain == 1.8
    assert im.telescope.read_noise == 6.0


def test_header_read_noise_is_per_frame(tmp_path):
    telescope = Telescope(
        name="test_header_noise_per_frame",
        read_noise=6.0,
        keyword_read_noise="RDNOISE",
    )
    a = FITSImage(_write_fits(tmp_path, "a.fits", RDNOISE=7.8), telescope=telescope)
    b = FITSImage(_write_fits(tmp_path, "b.fits", RDNOISE=12.0), telescope=telescope)

    assert a.metadata["read_noise"] == 7.8
    assert b.metadata["read_noise"] == 12.0


def test_header_noise_keywords_are_opt_in(tmp_path):
    # without keyword_gain / keyword_read_noise the header is ignored, so
    # existing .telescope definitions behave exactly as before
    telescope = Telescope(name="test_header_noise_optin", gain=2.0, read_noise=10.0)
    fpath = _write_fits(tmp_path, GAIN=1.0, RDNOISE=12.0)

    im = FITSImage(fpath, telescope=telescope)

    assert im.metadata["gain"] == 2.0
    assert im.metadata["read_noise"] == 10.0


@pytest.mark.parametrize("bad", [None, "UNKNOWN", "N/A", 0.0, -1.0])
def test_header_noise_falls_back_to_telescope_when_value_unusable(tmp_path, bad):
    telescope = Telescope(
        name="test_header_noise_fallback",
        gain=1.8,
        read_noise=6.0,
        keyword_gain="GAIN",
        keyword_read_noise="RDNOISE",
    )
    extra = {} if bad is None else {"GAIN": bad, "RDNOISE": bad}
    fpath = _write_fits(tmp_path, **extra)

    im = FITSImage(fpath, telescope=telescope)

    assert im.metadata["gain"] == 1.8
    assert im.metadata["read_noise"] == 6.0


# LCO BANZAI-reduced instruments: headers carry per-frame GAIN (e-/ADU, 1.0) and
# RDNOISE (e-), verified on real archive frames of each instrument.
BANZAI_TELESCOPE_FILES = (
    ["sinistro", "sbig", "qhy600"]
    + [f"muscat3_{b}" for b in ("gp", "rp", "ip", "zs")]
    + [f"muscat4_{b}" for b in ("gp", "rp", "ip", "zs")]
)


@pytest.mark.parametrize("name", BANZAI_TELESCOPE_FILES)
def test_banzai_telescope_files_read_noise_and_gain_from_header(name):
    import yaml
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / f"{name}.telescope"
    spec = yaml.safe_load(path.read_text())
    assert spec["keyword_gain"] == "GAIN"
    assert spec["keyword_read_noise"] == "RDNOISE"


def test_header_positive_float_rejects_nan_and_inf():
    # FITS headers cannot store NaN, but dict-like headers can
    from prose.core.image import _header_positive_float

    assert _header_positive_float({"RDNOISE": float("nan")}, "RDNOISE", 6.0) == 6.0
    assert _header_positive_float({"RDNOISE": float("inf")}, "RDNOISE", 6.0) == 6.0
    assert _header_positive_float({"RDNOISE": 7.8}, "RDNOISE", 6.0) == 7.8
    assert _header_positive_float({"RDNOISE": 7.8}, None, 6.0) == 6.0


def _write_muscat2_frame(tmp_path, band):
    header = Header()
    header["INSTRUME"] = "MuSCAT2"
    header["TELESCOP"] = "TCS 1.52m tel."
    header["FILTER"] = band
    path = tmp_path / f"frame_{band}.fits"
    fits.writeto(path, np.zeros((4, 4)), header=header)
    return path


@pytest.fixture
def muscat2_config(monkeypatch):
    """Band telescopes sharing aliases; the alias keys point at the i band."""
    import prose.telescope as telescope_module

    variants = {
        band: {
            "name": f"MuSCAT2_{band}",
            "names": ["MuSCAT2", "muscat2", "TCS"],
            "gain": gain,
            "read_noise": rn,
        }
        for band, gain, rn in (
            ("g", 1.04, 4.08),
            ("r", 0.96, 3.96),
            ("i", 1.03, 4.02),
            ("zs", 1.02, 4.48),
        )
    }
    telescopes = {f"muscat2_{b}": t for b, t in variants.items()}
    telescopes.update(muscat2=variants["i"], tcs=variants["i"])
    fake = type(CONFIG).__new__(type(CONFIG))
    fake.telescopes_dict = telescopes
    monkeypatch.setattr(telescope_module, "CONFIG", fake)
    return variants


@pytest.mark.parametrize(
    "band, expected, gain, read_noise",
    [
        ("g", "MuSCAT2_g", 1.04, 4.08),
        ("r", "MuSCAT2_r", 0.96, 3.96),
        ("i", "MuSCAT2_i", 1.03, 4.02),
        ("z_s", "MuSCAT2_zs", 1.02, 4.48),
    ],
)
def test_fitsimage_uses_the_band_specific_telescope(
    tmp_path, muscat2_config, band, expected, gain, read_noise
):
    im = FITSImage(_write_muscat2_frame(tmp_path, band), load_data=False)

    assert im.telescope.name == expected
    assert im.gain == gain
    assert im.read_noise == read_noise


def test_fitsimage_keeps_alias_match_when_band_is_unknown(tmp_path, muscat2_config):
    im = FITSImage(_write_muscat2_frame(tmp_path, "Na_D"), load_data=False)
    assert im.telescope.name == "MuSCAT2_i"


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def repo_telescopes_config(monkeypatch):
    """CONFIG built from the repository's data/*.telescope files, alias keys
    resolved last-writer-wins like ConfigManager.build_telescopes_dict."""
    import yaml

    import prose.telescope as telescope_module

    telescopes = {}
    for path in sorted(DATA_DIR.glob("*.telescope")):
        definition = yaml.safe_load(path.read_text())
        telescopes[definition["name"].lower()] = definition
        for alias in definition.get("names") or ():
            telescopes[str(alias).lower()] = definition
    fake = type(CONFIG).__new__(type(CONFIG))
    fake.telescopes_dict = telescopes
    monkeypatch.setattr(telescope_module, "CONFIG", fake)


# Camera IDs observed in MuSCAT3 (ogg 2m0-01) and MuSCAT4 (coj 2m0-02) BANZAI
# headers; narrowband filters sit on the same channel as their broadband one.
@pytest.mark.parametrize(
    "instrume, telescop, band, expected",
    [
        ("ep04", "2m0-01", "gp", "LCO-2m/MuSCAT3_gp"),
        ("ep02", "2m0-01", "rp", "LCO-2m/MuSCAT3_rp"),
        ("ep03", "2m0-01", "ip", "LCO-2m/MuSCAT3_ip"),
        ("ep05", "2m0-01", "zs", "LCO-2m/MuSCAT3_zs"),
        ("ep04", "2m0-01", "g_narrow", "LCO-2m/MuSCAT3_gp"),
        ("ep02", "2m0-01", "Na_D", "LCO-2m/MuSCAT3_rp"),
        ("ep03", "2m0-01", "i_narrow", "LCO-2m/MuSCAT3_ip"),
        ("ep05", "2m0-01", "z_narrow", "LCO-2m/MuSCAT3_zs"),
        ("ep06", "2m0-02", "gp", "LCO-2m/MuSCAT4_gp"),
        ("ep07", "2m0-02", "rp", "LCO-2m/MuSCAT4_rp"),
        ("ep08", "2m0-02", "ip", "LCO-2m/MuSCAT4_ip"),
        ("ep09", "2m0-02", "zs", "LCO-2m/MuSCAT4_zs"),
    ],
)
def test_muscat3_4_camera_ids_match_their_channel_file(
    tmp_path, repo_telescopes_config, instrume, telescop, band, expected
):
    header = Header()
    header["INSTRUME"] = instrume
    header["TELESCOP"] = telescop
    header["FILTER"] = band
    header["GAIN"] = 1.0
    header["RDNOISE"] = 13.8
    path = tmp_path / f"{instrume}_{band}.fits"
    fits.writeto(path, np.zeros((4, 4)), header=header)

    im = FITSImage(path, load_data=False)

    assert im.telescope.name == expected
    # The matched file opts into per-frame header gain and read noise.
    assert (im.gain, im.read_noise) == (1.0, 13.8)


def test_each_camera_id_is_listed_by_exactly_one_telescope_file():
    import yaml

    owners = {}
    for path in sorted(DATA_DIR.glob("*.telescope")):
        for alias in yaml.safe_load(path.read_text()).get("names") or ():
            if str(alias).lower().startswith("ep0"):
                owners.setdefault(str(alias).lower(), []).append(path.stem)
    assert owners == {
        "ep02": ["muscat3_rp"],
        "ep03": ["muscat3_ip"],
        "ep04": ["muscat3_gp"],
        "ep05": ["muscat3_zs"],
        "ep06": ["muscat4_gp"],
        "ep07": ["muscat4_rp"],
        "ep08": ["muscat4_ip"],
        "ep09": ["muscat4_zs"],
    }
