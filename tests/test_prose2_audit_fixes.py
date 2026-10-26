"""Regression tests for bugs found in the prose2 audit."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.time import Time

from prose import Fluxes, FITSImage, Telescope
from prose.core.sequence import SequenceParallel
from prose.scripts import run_photometry as rp


def _write_frame(path, date_obs=None, mjd_strt=None):
    h = fits.Header()
    h["INSTRUME"] = "MuSCAT2"
    h["TELESCOP"] = "TCS 1.52m tel."
    h["FILTER"] = "g"
    h["EXPTIME"] = 10
    if date_obs is not None:
        h["DATE-OBS"] = date_obs
    if mjd_strt is not None:
        h["MJD-STRT"] = mjd_strt
    fits.writeto(path, np.zeros((4, 4)), header=h, overwrite=True)
    return path


# ---- 1 + 2: from_args resolves block names without eval, and leaves input alone


def test_from_args_builds_named_blocks():
    seq = SequenceParallel.from_args([{"block": "blocks.Cutouts", "shape": 10}])
    assert [type(b).__name__ for b in seq.blocks] == ["Cutouts"]


def test_from_args_does_not_mutate_its_input():
    args = [{"block": "blocks.Cutouts", "shape": 10}]
    SequenceParallel.from_args(args)
    assert args == [{"block": "blocks.Cutouts", "shape": 10}]
    # The same arguments must work on a second call.
    assert len(SequenceParallel.from_args(args).blocks) == 1


@pytest.mark.parametrize(
    "name",
    [
        "os.system",
        "blocks.__class__",
        "__import__('os').system('true')",
        "blocks.NoSuchBlock",
        "Cutouts",
    ],
)
def test_from_args_rejects_names_outside_prose_blocks(name):
    with pytest.raises(ValueError, match="unknown block"):
        SequenceParallel.from_args([{"block": name}])


# ---- 3: a frame without a usable date has no JD, instead of year 1800


def test_telescope_date_is_none_without_date_keyword():
    assert Telescope().date(fits.Header()) is None


def test_frame_without_date_has_no_jd(tmp_path, caplog):
    im = FITSImage(_write_frame(tmp_path / "nodate.fits"), load_data=False)
    assert im.jd is None
    assert "no usable DATE-OBS" in caplog.text


def test_frame_with_date_keeps_its_jd(tmp_path):
    path = _write_frame(tmp_path / "dated.fits", date_obs="2025-11-29T01:10:00")
    im = FITSImage(path, load_data=False)
    assert im.jd == pytest.approx(Time("2025-11-29T01:10:00").jd, abs=1e-6)


# ---- 4: MJD or JD is decided per value, not per series


def test_normalize_converts_only_the_mjd_values_in_a_mixed_series():
    jd_frame = Time("2025-11-29T01:10:00").jd
    mjd_frame = 60996.0417
    out = rp.normalize_time_to_jd(np.array([mjd_frame, jd_frame]), "mjd")
    assert out[0] == pytest.approx(mjd_frame + rp.MJD_TO_JD, abs=1e-6)
    assert out[1] == pytest.approx(jd_frame, abs=1e-6)


def test_normalize_mixed_series_when_the_mjd_frame_is_the_minority():
    jd_frames = np.array([Time("2025-11-29T01:10:00").jd + k for k in range(3)])
    out = rp.normalize_time_to_jd(np.r_[60996.0417, jd_frames], "mjd")
    assert out[0] == pytest.approx(60996.0417 + rp.MJD_TO_JD, abs=1e-6)
    np.testing.assert_allclose(out[1:], jd_frames, atol=1e-6)


def test_normalize_leaves_non_mjd_scales_alone():
    t = np.array([60996.0417, 2461008.5])
    np.testing.assert_array_equal(rp.normalize_time_to_jd(t, "utc"), t)
    np.testing.assert_array_equal(rp.normalize_time_to_jd(t, None), t)


# ---- 5: pickles and YAML from disk cannot run code


def _malicious_pickle(path, marker):
    class Payload:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    with open(path, "wb") as f:
        pickle.dump(Payload(), f)
    return path


def test_fluxes_load_rejects_pickled_code(tmp_path):
    marker = tmp_path / "pwned"
    bad = _malicious_pickle(tmp_path / "bad.fluxes", marker)
    with pytest.raises(pickle.UnpicklingError):
        Fluxes.load(bad)
    assert not marker.exists()


def test_fluxes_round_trip_still_loads(tmp_path):
    f = Fluxes(fluxes=np.ones((2, 1)), time=np.array([1.0, 2.0]))
    f.save(tmp_path / "ok.fluxes")
    loaded = Fluxes.load(tmp_path / "ok.fluxes")
    np.testing.assert_array_equal(loaded.time, f.time)
    np.testing.assert_array_equal(loaded.fluxes, f.fluxes)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point prose's ~/.prose at a temp dir and skip the Ballet download."""
    from prose.config import ConfigManager

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ConfigManager, "check_ballet", lambda self: None)
    monkeypatch.delenv("PROSE_CONFIG_FILE", raising=False)
    monkeypatch.delenv("PROSE_CONFIG_SHA256", raising=False)
    return tmp_path / ".prose"


def _payload(marker):
    # Valid YAML that runs `touch` under yaml.Loader (PyYAML 6 FullLoader
    # refuses it, so the telescope-file test is a guard, not a reproduction).
    return f"hack: !!python/object/apply:os.system ['touch {marker}']\n"


def test_home_config_file_does_not_execute_yaml_tags(tmp_path, isolated_home):
    from prose.config import ConfigManager

    isolated_home.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "pwned-config"
    (isolated_home / "config").write_text(_payload(marker) + "color: blue\n")
    try:
        ConfigManager()
    except yaml.YAMLError:
        pass
    assert not marker.exists()


def test_telescope_file_does_not_execute_yaml_tags(tmp_path, isolated_home):
    from prose.config import ConfigManager

    isolated_home.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "pwned-telescope"
    (isolated_home / "evil.telescope").write_text(_payload(marker) + "name: evil\n")
    try:
        ConfigManager()
    except yaml.YAMLError:
        pass
    assert not marker.exists()
