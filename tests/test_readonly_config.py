"""Explicit configuration must never initialize or modify the user directory."""

import copy
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_module(monkeypatch):
    # Load this module without importing prose and initializing its global CONFIG.
    package = ModuleType("prose")
    package.__path__ = [str(ROOT / "prose")]
    builtins = ModuleType("prose.builtins")
    builtins.built_in_telescopes = {"builtin": {"name": "Builtin"}}
    monkeypatch.setitem(sys.modules, "prose", package)
    monkeypatch.setitem(sys.modules, "prose.builtins", builtins)
    spec = importlib.util.spec_from_file_location(
        "_prose_config_under_test", ROOT / "prose/config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("PROSE_CONFIG_FILE", raising=False)
    monkeypatch.delenv("PROSE_CONFIG_SHA256", raising=False)
    return module


@pytest.fixture
def payload():
    camera = {
        "name": "Camera",
        "names": ("Camera", "Alias"),
        "gain": 1.8,
        "keyword_gain": "GAIN",
        "keyword_read_noise": "RDNOISE",
        "read_noise": 6.0,
        "trimming": (0, 2),
    }
    return {
        "schema": 1,
        "settings": {"color": "blue"},
        "telescopes": {"camera": camera, "alias": camera},
    }


@pytest.fixture
def configuration(tmp_path, payload):
    config = tmp_path / "canonical.yaml"
    config.write_text(yaml.dump(payload))
    return config


def test_explicit_load_has_no_implicit_home_reads_or_writes(
    config_module, configuration, payload, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("implicit user configuration or network accessed")

    monkeypatch.setattr(Path, "home", forbidden)
    monkeypatch.setattr(config_module.requests, "get", forbidden)
    original = configuration.read_bytes()
    config = config_module.ConfigManager(configuration)
    assert config.read_only
    assert config.config_file == configuration.resolve()
    assert config.config_sha256 == hashlib.sha256(original).hexdigest()
    assert config.config == payload["settings"]
    assert config.telescopes_dict == payload["telescopes"]
    assert isinstance(config.telescopes_dict["camera"]["trimming"], tuple)
    assert config.match_telescope_name("ALIAS")["gain"] == 1.8
    config.check_builtins_changes()
    assert configuration.read_bytes() == original
    assert list(configuration.parent.iterdir()) == [configuration]


@pytest.mark.parametrize(
    "method,args",
    [
        ("save", ()),
        ("set", ("color", "red")),
        ("save_telescope_file", ({"name": "Other"},)),
        ("update_builtins", ()),
        ("create_builtins_telescopes_files", ()),
        ("create_builtins_telescopes_files", (True,)),
        ("rename_id_files_to_telescopes", ()),
        ("check_ballet", ()),
    ],
)
def test_write_methods_fail_before_any_mutation(
    config_module, configuration, method, args
):
    config = config_module.ConfigManager(configuration)
    before = copy.deepcopy((config.config, config.telescopes_dict))
    original = configuration.read_bytes()
    with pytest.raises(RuntimeError, match="read-only"):
        getattr(config, method)(*args)
    assert (config.config, config.telescopes_dict) == before
    assert configuration.read_bytes() == original
    assert list(configuration.parent.iterdir()) == [configuration]


def test_loaded_snapshot_stays_stable_even_if_source_changes(
    config_module, configuration
):
    config = config_module.ConfigManager(configuration)
    digest = config.config_sha256
    configuration.unlink()
    config.config["color"] = "red"
    config.telescopes_dict["camera"]["gain"] = 99
    config.check_config_file(load=True)
    assert config.config == {"color": "blue"}
    rebuilt = config.build_telescopes_dict()
    assert rebuilt["camera"]["gain"] == 1.8
    rebuilt["camera"]["gain"] = 88
    assert config.build_telescopes_dict()["camera"]["gain"] == 1.8
    assert config.config_sha256 == digest


@pytest.mark.parametrize("alias", [" ", "  Camera  ", "MixedCase"])
def test_aliases_are_lowercased_but_not_stripped(
    config_module, tmp_path, payload, alias
):
    # Legacy builtin definitions include a literal space alias for Liverpool.
    payload["telescopes"] = {alias: payload["telescopes"]["camera"]}
    path = tmp_path / "aliases.yaml"
    path.write_text(yaml.dump(payload))
    config = config_module.ConfigManager(path)
    assert list(config.telescopes_dict) == [alias.lower()]
    assert config.match_telescope_name(alias)["gain"] == 1.8


def test_environment_selects_file_and_optional_expected_digest(
    config_module, configuration, monkeypatch
):
    digest = hashlib.sha256(configuration.read_bytes()).hexdigest()
    monkeypatch.setenv("PROSE_CONFIG_FILE", str(configuration))
    monkeypatch.setenv("PROSE_CONFIG_SHA256", digest)
    assert config_module.ConfigManager().config_sha256 == digest
    monkeypatch.setenv("PROSE_CONFIG_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        config_module.ConfigManager()
    # An explicitly constructed manager does not inherit another file's digest.
    assert config_module.ConfigManager(configuration).config_sha256 == digest


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64, "0" * 63])
def test_bad_expected_digest_fails(config_module, configuration, digest):
    with pytest.raises(ValueError, match="SHA-256"):
        config_module.ConfigManager(configuration, expected_sha256=digest)


def test_expected_digest_requires_explicit_file(config_module, monkeypatch):
    monkeypatch.setenv("PROSE_CONFIG_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="PROSE_CONFIG_FILE"):
        config_module.ConfigManager()


def test_missing_file_does_not_fall_back(config_module, tmp_path):
    missing = tmp_path / "absent" / "canonical.yaml"
    with pytest.raises(FileNotFoundError):
        config_module.ConfigManager(missing)
    assert not missing.parent.exists()


def test_empty_environment_file_is_not_a_request_for_defaults(
    config_module, monkeypatch
):
    monkeypatch.setenv("PROSE_CONFIG_FILE", "")
    with pytest.raises(ValueError, match="configuration file"):
        config_module.ConfigManager()


@pytest.mark.parametrize(
    "document",
    [
        "",
        "[]",
        "schema: 2\nsettings: {}\ntelescopes: {}",
        "schema: true\nsettings: {}\ntelescopes: {}",
        "schema: 1\nsettings: []\ntelescopes: {}",
        "schema: 1\nsettings: {}\ntelescopes: []",
        "schema: 1\nsettings: {}",
        "schema: 1\nsettings: {}\ntelescopes: {camera: null}",
        "schema: 1\nsettings: {}\ntelescopes: {camera: {gain: 1}}",
        "schema: 1\nsettings: {}\ntelescopes: {1: {name: Camera}}",
        "schema: 1\nsettings: {}\ntelescopes: {'': {name: Camera}}",
        "schema: 1\nschema: 2\nsettings: {}\ntelescopes: {}",
        "schema: 1\nsettings: {color: blue, color: red}\ntelescopes: {}",
        "schema: 1\nsettings: {}\ntelescopes: {a: {name: A}, a: {name: B}}",
        "schema: 1\nsettings: {}\ntelescopes: {A: {name: A}, a: {name: B}}",
        "schema: 1\nsettings: {}\ntelescopes: [",
        "!!python/object/apply:os.system ['exit 77']",
    ],
)
def test_invalid_configuration_fails_without_fallback(
    config_module, tmp_path, document
):
    config = tmp_path / "invalid.yaml"
    config.write_text(document)
    with pytest.raises(ValueError, match="configuration"):
        config_module.ConfigManager(config)
    assert list(tmp_path.iterdir()) == [config]


def test_legacy_initialization_is_unchanged(config_module, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    directory = tmp_path / ".prose"
    directory.mkdir()
    (directory / "centroid.h5").touch()
    config = config_module.ConfigManager()
    assert not config.read_only
    assert config.config_sha256 is None
    assert config.config_file == directory / "config"
    config.set("color", "red")
    assert yaml.safe_load(config.config_file.read_text()) == {"color": "red"}
    assert config.build_telescopes_dict() == {"builtin": {"name": "Builtin"}}


def _environment(configuration, tmp_path):
    env = dict(
        os.environ,
        PROSE_CONFIG_FILE=configuration.name,
        PYTHONPATH=str(ROOT),
        HOME=str(tmp_path / "unused-home"),
        MPLBACKEND="Agg",
    )
    env.pop("PROSE_CONFIG_SHA256", None)
    return env


def test_real_import_and_header_precedence(configuration, tmp_path):
    code = """
import json, os
import numpy as np
from astropy.io import fits
from prose import CONFIG, FITSImage, Telescope
header = fits.Header({'INSTRUME': 'Camera', 'TELESCOP': 'Camera', 'EXPTIME': 10.,
                      'DATE-OBS': '2026-01-01T00:00:00', 'JD': 2461041.,
                      'FILTER': 'gp', 'GAIN': 1., 'RDNOISE': 12.})
fits.writeto('frame.fits', np.ones((20, 20)), header)
image = FITSImage('frame.fits', telescope=Telescope.from_name('Alias'))
assert image.metadata['gain'] == 1.
assert image.metadata['read_noise'] == 12.
assert image.telescope.gain == 1.8
assert os.environ['PROSE_CONFIG_SHA256'] == CONFIG.config_sha256
assert os.environ['PROSE_CONFIG_FILE'] == str(CONFIG.config_file)
print(json.dumps({'digest': CONFIG.config_sha256, 'aliases': list(CONFIG.telescopes_dict)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=_environment(configuration, tmp_path),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(result.stdout)["digest"]
        == hashlib.sha256(configuration.read_bytes()).hexdigest()
    )
    assert not (tmp_path / "unused-home/.prose").exists()


@pytest.mark.parametrize("backend", ["multiprocessing", "multiprocess"])
@pytest.mark.parametrize("start_method", ["fork", "spawn", "forkserver"])
@pytest.mark.parametrize("changed", [False, True])
def test_workers_use_selected_file_and_reject_fresh_import_drift(
    configuration, tmp_path, backend, start_method, changed
):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable on this platform")
    original_digest = hashlib.sha256(configuration.read_bytes()).hexdigest()
    (tmp_path / "worker-cwd").mkdir()
    script = tmp_path / "worker_probe.py"
    script.write_text(f"""
import json, os
from pathlib import Path
import {backend} as mp

def worker(queue):
    try:
        os.chdir(Path(__file__).parent / 'worker-cwd')
        from prose import CONFIG
        queue.put({{'digest': CONFIG.config_sha256, 'gain': CONFIG.telescopes_dict['camera']['gain']}})
    except Exception as exc:
        queue.put({{'error': str(exc)}})

if __name__ == '__main__':
    from prose import CONFIG
    assert Path(os.environ['PROSE_CONFIG_FILE']).is_absolute()
    if {changed!r}:
        with Path(CONFIG.config_file).open('a') as stream:
            stream.write('\\n# changed during the run\\n')
    context = mp.get_context({start_method!r})
    queue = context.Queue()
    process = context.Process(target=worker, args=(queue,))
    process.start()
    result = queue.get(timeout=40)
    process.join(10)
    assert process.exitcode == 0
    print(json.dumps(result))
""")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=_environment(configuration, tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    if changed and start_method != "fork":
        assert "checksum" in response["error"]
    else:
        assert response == {
            "digest": original_digest,
            "gain": 1.8,
        }
    assert not (tmp_path / "unused-home/.prose").exists()


def _muscat2_variants():
    """Band files that all answer to the same instrument aliases, as in
    muscat-db's configuration (the alias keys point at the last band)."""
    variants = {}
    for band, gain in (("g", 1.04), ("r", 0.96), ("i", 1.03), ("zs", 1.02)):
        variants[f"muscat2_{band}"] = {
            "name": f"MuSCAT2_{band}",
            "names": ["MuSCAT2", "muscat2", "TCS"],
            "gain": gain,
        }
    telescopes = dict(variants)
    for alias in ("muscat2", "tcs"):
        telescopes[alias] = variants["muscat2_i"]
    return telescopes


@pytest.mark.parametrize(
    "alias, band, expected",
    [
        ("MuSCAT2", "g", "MuSCAT2_g"),
        ("muscat2", "r", "MuSCAT2_r"),
        ("TCS", "i", "MuSCAT2_i"),
        ("MuSCAT2", "z_s", "MuSCAT2_zs"),
        ("MuSCAT2", "zs", "MuSCAT2_zs"),
        ("MuSCAT2", " 'g' ", "MuSCAT2_g"),
    ],
)
def test_match_telescope_variant_picks_the_frames_band(
    config_module, tmp_path, alias, band, expected
):
    path = tmp_path / "config.yaml"
    payload = {"schema": 1, "settings": {}, "telescopes": _muscat2_variants()}
    path.write_text(yaml.dump(payload))
    config = config_module.ConfigManager(path)

    assert config.match_telescope_variant(alias, band)["name"] == expected


@pytest.mark.parametrize(
    "alias, band",
    [("MuSCAT2", "Na_D"), ("MuSCAT2", ""), ("MuSCAT2", None), ("Unknown", "g")],
)
def test_match_telescope_variant_returns_none_without_a_unique_band_match(
    config_module, tmp_path, alias, band
):
    path = tmp_path / "config.yaml"
    payload = {"schema": 1, "settings": {}, "telescopes": _muscat2_variants()}
    path.write_text(yaml.dump(payload))
    config = config_module.ConfigManager(path)

    assert config.match_telescope_variant(alias, band) is None
