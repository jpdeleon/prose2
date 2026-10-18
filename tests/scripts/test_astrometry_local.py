"""Offline local-solver contract tests using a real executable subprocess."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from prose.scripts import solve_wcs_astrometry as swa

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="local solver requires POSIX"
)


@pytest.fixture
def probe(tmp_path):
    path = tmp_path / "probe with spaces.fits"
    hdu = fits.PrimaryHDU(np.zeros((32, 32)))
    hdu.header.update(INSTRUME="MuSCAT2", RA="12:08:09", DEC="-00:05:58", PIXSCALE=0.5)
    hdu.writeto(path)
    return path


@pytest.fixture
def solver(tmp_path, monkeypatch):
    # Use a real child process so argv, environment, products and timeouts are exercised.
    path = tmp_path / "fake solve-field"
    path.write_text(
        f"#!{sys.executable}\n"
        + """
import os, sys, time, subprocess
from pathlib import Path
args = sys.argv[1:]
if os.environ.get('FAKE_HANG'):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    Path(os.environ['FAKE_PID']).write_text(str(child.pid))
    time.sleep(60)
if os.environ.get('FAKE_EXIT'):
    print('missing index files', flush=True)
    sys.exit(1)
from astropy.io import fits
from astropy.wcs import WCS
assert Path(os.environ['TMPDIR']).is_dir()
assert args[args.index('--new-fits') + 1] == 'none'
assert args[args.index('--temp-dir') + 1] == os.environ['TMPDIR']
assert float(args[args.index('--ra') + 1]) == 182.0375
assert float(args[args.index('--dec') + 1]) < 0
assert float(args[args.index('--scale-low') + 1]) == 0.4
assert float(args[args.index('--scale-high') + 1]) == 0.6
if os.environ.get('FAKE_CONFIG'):
    assert args[args.index('--backend-config') + 1] == os.environ['FAKE_CONFIG']
if not os.environ.get('FAKE_NO_SOLUTION'):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    scale = float(os.environ.get('FAKE_SCALE', '0.5')) / 3600
    w.wcs.cdelt = [-scale, scale]
    fits.PrimaryHDU(header=w.to_header()).writeto(args[args.index('--wcs') + 1])
    if os.environ.get('FAKE_CORRUPT'):
        Path(args[args.index('--wcs') + 1]).write_text('corrupt')
    Path(args[args.index('--solved') + 1]).write_bytes(b'\\x00' if os.environ.get('FAKE_UNSOLVED') else b'\\x01')
"""
    )
    path.chmod(0o755)
    monkeypatch.setenv("PROSE_ASTROMETRY_LOCK", str(tmp_path / "solver.lock"))
    return path


def test_local_sidecar_without_key_or_upload(probe, solver, tmp_path, monkeypatch):
    monkeypatch.setattr(swa, "_api_key", lambda: pytest.fail("local requested API key"))
    monkeypatch.setattr(
        swa, "upload_and_solve", lambda *a, **kw: pytest.fail("uploaded")
    )
    before = probe.read_bytes()
    sidecar = tmp_path / "solution.fits"
    assert (
        swa.main(
            [
                "--fits",
                str(probe),
                "--backend",
                "local",
                "--solve-field",
                str(solver),
                "--astrometry-temp-dir",
                str(tmp_path),
                "--wcs-output",
                str(sidecar),
            ]
        )
        == 0
    )
    assert WCS(fits.getheader(sidecar)).has_celestial
    assert probe.read_bytes() == before
    assert not list(tmp_path.glob("prose-astrometry-*"))


@pytest.mark.parametrize(
    "mode", ["FAKE_EXIT", "FAKE_NO_SOLUTION", "FAKE_CORRUPT", "FAKE_UNSOLVED"]
)
def test_unsolved_is_failure(probe, solver, tmp_path, monkeypatch, mode, caplog):
    monkeypatch.setenv(mode, "1")
    assert swa.solve_local(probe, executable=str(solver), temp_dir=tmp_path) is None
    if mode == "FAKE_EXIT":
        assert "missing index files" in caplog.text


def test_missing_executable_fails_without_upload(probe, tmp_path, caplog):
    assert (
        swa.solve_local(probe, executable=str(tmp_path / "absent"), temp_dir=tmp_path)
        is None
    )
    assert "solve-field" in caplog.text


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_rejected(probe, timeout):
    with pytest.raises(ValueError, match="timeout"):
        swa.solve_local(probe, timeout=timeout)


def test_missing_config_fails_before_launch(probe, solver, tmp_path, caplog):
    assert (
        swa.solve_local(
            probe, executable=str(solver), config=tmp_path / "absent", temp_dir=tmp_path
        )
        is None
    )
    assert "config" in caplog.text.lower()


def test_timeout_kills_solver_descendants(probe, solver, tmp_path, monkeypatch):
    pidfile = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_HANG", "1")
    monkeypatch.setenv("FAKE_PID", str(pidfile))
    start = time.monotonic()
    assert (
        swa.solve_local(probe, executable=str(solver), timeout=1, temp_dir=tmp_path)
        is None
    )
    assert time.monotonic() - start < 5
    assert pidfile.exists()
    pid = int(pidfile.read_text())
    # A killed descendant can briefly remain a zombie until init reaps it.
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_busy_lock_obeys_timeout(probe, solver, tmp_path, monkeypatch):
    import fcntl

    lock = tmp_path / "busy.lock"
    monkeypatch.setenv("PROSE_ASTROMETRY_LOCK", str(lock))
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        assert (
            swa.solve_local(
                probe, executable=str(solver), timeout=0.2, temp_dir=tmp_path
            )
            is None
        )
        assert time.monotonic() - start < 2


def test_header_hints_preserve_negative_dec_and_prefer_header_scale(probe):
    hints = swa._header_hint(probe)
    assert hints["center_ra"] == pytest.approx(182.0375)
    assert hints["center_dec"] == pytest.approx(-0.0994444444)
    assert hints["scale_est"] == 0.5


def test_decimal_negative_declination(probe):
    with fits.open(probe, mode="update") as hdus:
        hdus[0].header["DEC"] = "-20.125"
    assert swa._header_hint(probe)["center_dec"] == -20.125


def test_local_retry_rejects_bad_scale(probe, tmp_path, monkeypatch):
    files = [probe, tmp_path / "second.fits", tmp_path / "third.fits"]
    calls = []

    def solve(path, **kwargs):
        calls.append(path)
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        scale = (100 if len(calls) == 1 else 0.44) / 3600
        w.wcs.cdelt = [-scale, scale]
        return w

    monkeypatch.setattr(swa, "solve_local", solve)
    assert swa.solve_first_valid(files, None, "muscat2", backend="local") is not None
    assert calls == files[:2]


@pytest.mark.parametrize("module_name", ["calibrate_muscat", "calibrate_muscat2"])
def test_calibration_local_only_updates_current_generated_band(
    module_name, probe, tmp_path, monkeypatch
):
    from importlib import import_module

    module = import_module("prose.scripts." + module_name)
    foreign = tmp_path / "foreign.fits"
    fits.PrimaryHDU().writeto(foreign)
    before = foreign.read_bytes()
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.4 / 3600, 0.4 / 3600]
    monkeypatch.setattr(swa, "_api_key", lambda: pytest.fail("local requested API key"))
    monkeypatch.setattr(swa, "solve_local", lambda *a, **kw: w)
    module._solve_astrometry_net("gp", tmp_path, [probe], method="astrometry-local")
    assert foreign.read_bytes() == before
    assert fits.getheader(probe)["WCSMTHD"] == "astrometry-local"
    assert WCS(
        fits.getheader(tmp_path / ".wcs/gp_astrometry-local.wcs.fits")
    ).has_celestial


def test_photometry_forwards_local_options():
    from prose.scripts import run_photometry as rp

    args = SimpleNamespace(
        data_dir=Path("/data"),
        target_name="target",
        wcs_method="astrometry-local",
        test_run=False,
        verbose=False,
        astrometry_config=Path("/indexes/config"),
        astrometry_temp_dir=Path("/work"),
        solve_field="/bin/solve-field",
        astrometry_timeout=42,
    )
    argv = rp._calibration_args(args, Path("/out"), ["gp"])
    from prose.scripts import calibrate_muscat2 as cm

    parsed = cm.parse_args(argv)
    assert parsed.solve_wcs == "astrometry-local"
    assert parsed.astrometry_config == args.astrometry_config
    assert parsed.astrometry_temp_dir == args.astrometry_temp_dir
    assert parsed.solve_field == args.solve_field
    assert parsed.astrometry_timeout == 42


def test_config_path_with_spaces(probe, solver, tmp_path, monkeypatch):
    config = tmp_path / "backend config.cfg"
    config.write_text("autoindex\n")
    monkeypatch.setenv("FAKE_CONFIG", str(config))
    assert (
        swa.solve_local(probe, executable=str(solver), config=config, temp_dir=tmp_path)
        is not None
    )


def test_cli_refuses_to_overwrite_input(probe, solver):
    before = probe.read_bytes()
    assert (
        swa.main(
            [
                "--fits",
                str(probe),
                "--backend",
                "local",
                "--solve-field",
                str(solver),
                "--wcs-output",
                str(probe),
            ]
        )
        == 1
    )
    assert probe.read_bytes() == before


def test_sigterm_cleans_up_solver_children(probe, solver, tmp_path):
    pidfile = tmp_path / "terminated-child.pid"
    env = dict(
        os.environ,
        FAKE_HANG="1",
        FAKE_PID=str(pidfile),
        PROSE_ASTROMETRY_LOCK=str(tmp_path / "cancel.lock"),
    )
    code = (
        "from pathlib import Path; from prose.scripts.solve_wcs_astrometry import solve_local; "
        f"solve_local(Path({str(probe)!r}), executable={str(solver)!r}, temp_dir=Path({str(tmp_path)!r}))"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], env=env)
    try:
        deadline = time.monotonic() + 15
        while (
            not pidfile.exists() and proc.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert pidfile.exists()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
        assert proc.returncode == 128 + signal.SIGTERM
        stat = Path(f"/proc/{pidfile.read_text()}/stat")
        assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.skipif(
    not os.environ.get("PROSE_LOCAL_ASTROMETRY_TEST_FITS"),
    reason="requires a real frame and installed indexes",
)
def test_real_local_solver():
    """Opt-in scientific integration test; never downloads indexes or uploads."""
    executable = os.environ.get("PROSE_LOCAL_ASTROMETRY_TEST_SOLVER", "solve-field")
    if shutil.which(executable) is None:
        pytest.skip("solve-field is not installed")
    config = os.environ.get("PROSE_LOCAL_ASTROMETRY_TEST_CONFIG")
    path = Path(os.environ["PROSE_LOCAL_ASTROMETRY_TEST_FITS"])
    before = path.read_bytes()
    wcs = swa.solve_and_apply(
        path,
        backend="local",
        executable=executable,
        config=Path(config) if config else None,
    )
    assert wcs is not None and wcs.has_celestial
    assert path.read_bytes() == before


@pytest.mark.parametrize("module_name", ["calibrate_muscat", "calibrate_muscat2"])
def test_local_calibration_end_to_end(
    module_name, probe, solver, tmp_path, monkeypatch
):
    from importlib import import_module

    module = import_module("prose.scripts." + module_name)
    dark, flat = tmp_path / "dark.fits", tmp_path / "flat.fits"
    for path, level in ((dark, 10), (flat, 1000)):
        hdu = fits.PrimaryHDU(np.full((32, 32), level, dtype=np.float32))
        hdu.header.update(EXPTIME=1, FILTER="g", **{"MJD-STRT": 60000.0})
        hdu.writeto(path)
    with fits.open(probe, mode="update") as hdus:
        hdus[0].data[:] = 2000
        hdus[0].header.update(EXPTIME=1, FILTER="g", **{"MJD-STRT": 60000.0})
    before = probe.read_bytes()
    output = tmp_path / "calibrated"
    output.mkdir()
    monkeypatch.setattr(swa, "_api_key", lambda: pytest.fail("local requested API key"))
    module.calibrate_band(
        [str(dark)],
        [str(flat)],
        [str(probe)],
        output,
        "gp",
        solve_wcs="astrometry-local",
        local_astrometry=dict(executable=str(solver), temp_dir=tmp_path),
    )
    result = output / f"{probe.stem}_calibrated.fits"
    assert WCS(fits.getheader(result)).has_celestial
    assert fits.getheader(result)["WCSMTHD"] == "astrometry-local"
    assert (output / ".wcs/gp_astrometry-local.wcs.fits").exists()
    assert probe.read_bytes() == before
