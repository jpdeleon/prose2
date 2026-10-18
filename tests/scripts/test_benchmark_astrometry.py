"""Scientific agreement metrics and offline benchmark orchestration."""

import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS, Sip

from prose.scripts import benchmark_astrometry as bench


def make_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [16, 16]
    w.wcs.crval = [182.0375, -20]
    w.wcs.cdelt = [-0.44 / 3600, 0.44 / 3600]
    return w


@pytest.fixture
def frames(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"input {i}.fits"
        header = make_wcs().to_header()
        header.update(INSTRUME="MuSCAT2", PIXSCALE=0.44, RA="12:08:09", DEC="-20:00:00")
        fits.PrimaryHDU(np.full((32, 32), i, dtype=np.float32), header).writeto(p)
        paths.append(p)
    return paths


def test_identical_wcs_agree():
    metrics = bench.compare_wcs(make_wcs(), make_wcs(), (32, 32), grid_size=5)
    assert metrics["grid_samples"] == 25
    assert metrics["sky_max_arcsec"] == pytest.approx(0, abs=1e-8)
    assert metrics["pixel_rms"] == pytest.approx(0, abs=1e-7)


def test_known_pixel_offset_and_rotation():
    nova = make_wcs()
    nova.wcs.crpix += [3, 4]
    metrics = bench.compare_wcs(make_wcs(), nova, (32, 32), grid_size=5)
    assert metrics["pixel_rms"] == pytest.approx(5, abs=1e-6)
    assert metrics["sky_rms_arcsec"] == pytest.approx(2.2, abs=1e-5)
    rotated = make_wcs()
    rotated.wcs.pc = [[0, -1], [1, 0]]
    rotation = bench.compare_wcs(make_wcs(), rotated, (31, 31), grid_size=5)
    assert rotation["center_arcsec"] == pytest.approx(0, abs=1e-8)
    assert rotation["sky_max_arcsec"] > 10


def test_grid_detects_distortion_missed_at_center():
    distorted = make_wcs()
    a = np.zeros((3, 3))
    a[2, 0] = 0.001
    distorted.sip = Sip(a, np.zeros_like(a), None, None, distorted.wcs.crpix)
    distorted.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    metrics = bench.compare_wcs(make_wcs(), distorted, (31, 31), grid_size=5)
    assert metrics["center_arcsec"] == pytest.approx(0, abs=1e-8)
    assert metrics["sky_max_arcsec"] > 0.05


def test_probe_uses_same_pixels_without_existing_wcs(frames, tmp_path):
    before = frames[0].read_bytes()
    destination = tmp_path / "probe.fits"
    metadata = bench.prepare_probe(frames[0], destination)
    assert frames[0].read_bytes() == before
    assert metadata["shape"] == [32, 32]
    assert not WCS(fits.getheader(destination)).has_celestial
    assert fits.getheader(destination)["PIXSCALE"] == 0.44
    np.testing.assert_array_equal(fits.getdata(destination), fits.getdata(frames[0]))
    assert len(metadata["sha256"]) == 64


def test_compressed_image_with_primary_metadata(tmp_path):
    source = tmp_path / "compressed.fits.fz"
    primary = fits.PrimaryHDU()
    primary.header["RA"] = 182.0
    image = fits.CompImageHDU(np.ones((12, 16)), make_wcs().to_header())
    image.header["PIXSCALE"] = 0.44
    fits.HDUList([primary, image]).writeto(source)
    destination = tmp_path / "probe.fits"
    info = bench.prepare_probe(source, destination)
    assert info["source_hdu"] == 1
    assert info["shape"] == [12, 16]
    assert fits.getheader(destination)["RA"] == 182.0
    assert not WCS(fits.getheader(destination)).has_celestial


class Clock:
    def __init__(self):
        self.now = 0.0
        self.delays = []

    def perf_counter(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay


def test_benchmark_same_inputs_timing_metrics_and_rate_limit(
    frames, tmp_path, monkeypatch
):
    clock = Clock()
    monkeypatch.setattr(bench, "time", clock)
    calls = []

    def local(path, **kwargs):
        clock.now += 2
        calls.append(("local", path))
        assert not WCS(fits.getheader(path)).has_celestial
        return make_wcs()

    def nova(path, key, **kwargs):
        clock.now += 8
        calls.append(("nova", path))
        assert key == "secret-api-key"
        assert kwargs["publicly_visible"] == "n"
        w = make_wcs()
        w.wcs.crpix += [3, 4]
        return w

    monkeypatch.setattr(bench.swa, "solve_local", local)
    monkeypatch.setattr(bench.swa, "upload_and_solve", nova)
    output = tmp_path / "benchmark"
    report = bench.run_benchmark(
        frames, output, api_key="secret-api-key", online_delay=10
    )
    assert [b for b, p in calls] == ["local", "nova", "nova", "local"]
    assert calls[0][1] == calls[1][1]
    assert calls[2][1] == calls[3][1]
    assert clock.delays == [10]
    assert report["summary"]["local"]["valid"] == 2
    assert report["summary"]["nova"]["valid"] == 2
    assert report["summary"]["local"]["median_elapsed_s"] == 2
    assert report["summary"]["nova"]["median_elapsed_s"] == 8
    assert report["summary"]["paired_median_nova_over_local"] == 4
    assert report["results"][0]["agreement"]["pixel_rms"] == pytest.approx(5, abs=1e-6)
    text = (output / "results.json").read_text()
    assert "secret-api-key" not in text
    assert "NaN" not in text
    assert json.loads(text)["results"][0]["local"]["status"] == "valid"
    assert (output / "results.csv").exists()
    assert len(list((output / "wcs").glob("*.fits"))) == 4


def test_failure_is_recorded_and_does_not_abort(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(bench.swa, "solve_local", lambda *a, **kw: None)
    monkeypatch.setattr(bench.swa, "upload_and_solve", lambda *a, **kw: make_wcs())
    report = bench.run_benchmark(
        frames, tmp_path / "benchmark", api_key="key", online_delay=0
    )
    assert len(report["results"]) == 2
    assert report["summary"]["local"]["valid"] == 0
    assert report["summary"]["nova"]["valid"] == 2
    assert report["summary"]["paired_median_nova_over_local"] is None
    assert report["results"][0]["agreement"] is None
    assert report["results"][0]["local"]["status"] == "no_solution"


def test_invalid_solution_is_retained_but_excluded_from_pairs(
    frames, tmp_path, monkeypatch
):
    bad = make_wcs()
    bad.wcs.cdelt *= 100
    monkeypatch.setattr(bench.swa, "solve_local", lambda *a, **kw: bad)
    monkeypatch.setattr(bench.swa, "upload_and_solve", lambda *a, **kw: make_wcs())
    report = bench.run_benchmark(
        frames[:1], tmp_path / "benchmark", api_key="key", online_delay=0
    )
    assert report["summary"]["local"]["solved"] == 1
    assert report["summary"]["local"]["valid"] == 0
    assert report["results"][0]["local"]["status"] == "invalid"
    assert report["results"][0]["local"]["wcs"] is not None
    assert report["results"][0]["agreement"] is None


def test_input_limit_and_deduplication(frames):
    paths = bench.resolve_inputs(
        [str(frames[0].parent / "*.fits"), str(frames[0])], limit=1
    )
    assert paths == frames[:1]


def test_missing_solver_prevents_upload(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(bench.shutil, "which", lambda *a: None)
    monkeypatch.setattr(
        bench.swa, "_api_key", lambda: pytest.fail("checked credentials before solver")
    )
    assert (
        bench.main([str(frames[0]), "--output-dir", str(tmp_path / "benchmark")]) == 2
    )
    assert not (tmp_path / "benchmark").exists()


def test_existing_output_directory_is_preserved(frames, tmp_path, monkeypatch):
    output = tmp_path / "benchmark"
    output.mkdir()
    marker = output / "results.json"
    marker.write_text("existing results")
    monkeypatch.setattr(
        bench.swa, "solve_local", lambda *a, **kw: pytest.fail("solver ran")
    )
    with pytest.raises(FileExistsError):
        bench.run_benchmark(frames, output, api_key="key")
    assert marker.read_text() == "existing results"


def test_solver_exception_preserves_other_backend_and_redacts_key(
    frames, tmp_path, monkeypatch
):
    def unavailable(*args, **kwargs):
        raise RuntimeError("failed secret-api-key")

    monkeypatch.setattr(bench.swa, "solve_local", unavailable)
    monkeypatch.setattr(bench.swa, "upload_and_solve", lambda *a, **kw: make_wcs())
    output = tmp_path / "benchmark"
    report = bench.run_benchmark(frames[:1], output, api_key="secret-api-key")
    assert report["results"][0]["local"]["status"] == "error"
    assert report["results"][0]["nova"]["status"] == "valid"
    assert "secret-api-key" not in (output / "results.json").read_text()
    assert (
        "secret-api-key" not in next((output / "logs").glob("*local.log")).read_text()
    )


def test_nonfinite_wcs_does_not_produce_false_agreement():
    bad = make_wcs()
    bad.wcs.crval = [float("nan"), -20]
    with pytest.raises(ValueError, match="Non-finite"):
        bench.compare_wcs(make_wcs(), bad, (32, 32))


def test_ra_wrap_is_small_angular_difference():
    local, nova = make_wcs(), make_wcs()
    local.wcs.crval = [359.999, 0]
    nova.wcs.crval = [0.001, 0]
    metrics = bench.compare_wcs(local, nova, (31, 31))
    assert metrics["center_arcsec"] == pytest.approx(7.2, abs=1e-5)


def test_cli_writes_reproducible_report(frames, tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(bench.shutil, "which", lambda *a: "/test/solve-field")
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout="test-version", stderr="", returncode=0
        ),
    )
    monkeypatch.setattr(bench.swa, "_api_key", lambda: "private-key")
    monkeypatch.setattr(bench.swa, "solve_local", lambda *a, **kw: make_wcs())
    monkeypatch.setattr(bench.swa, "upload_and_solve", lambda *a, **kw: make_wcs())
    output = tmp_path / "benchmark"
    assert (
        bench.main(
            [str(frames[0]), "--output-dir", str(output), "--astrometry-timeout", "42"]
        )
        == 0
    )
    text = (output / "results.json").read_text()
    report = json.loads(text)
    assert report["status"] == "complete"
    assert report["metadata"]["local_version"] == "test-version"
    assert len(report["metadata"]["benchmark_sha256"]) == 64
    assert report["settings"]["timeout_s"] == 42
    assert report["summary"]["agreement"]["pairs"] == 1
    assert "private-key" not in text


def test_interruption_keeps_completed_attempt(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(bench.swa, "solve_local", lambda *a, **kw: make_wcs())

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(bench.swa, "upload_and_solve", interrupted)
    output = tmp_path / "benchmark"
    with pytest.raises(KeyboardInterrupt):
        bench.run_benchmark(frames[:1], output, api_key="key")
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "running"
    assert report["results"][0]["local"]["status"] == "valid"
    assert report["results"][0]["nova"] is None
    assert report["summary"]["local"]["attempts"] == 1
