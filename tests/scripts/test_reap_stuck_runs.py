import time

from prose.scripts import reap_stuck_runs as reap


def test_parse_results_dir_space_form():
    cmd = [
        "python",
        "-m",
        "prose.scripts.run_photometry",
        "--results_dir",
        "/out/x",
        "--gif",
    ]
    assert reap.parse_results_dir(cmd) == "/out/x"


def test_parse_results_dir_equals_form():
    cmd = ["python", "--results_dir=/out/y", "--verbose"]
    assert reap.parse_results_dir(cmd) == "/out/y"


def test_parse_results_dir_absent_returns_none():
    assert reap.parse_results_dir(["python", "-m", "x", "--data_dir", "/d"]) is None


def test_latest_activity_mtime_picks_newest(tmp_path):
    old = tmp_path / "a.log"
    new = tmp_path / "b.log"
    old.write_text("old")
    new.write_text("new")
    os_old = time.time() - 1000
    import os

    os.utime(old, (os_old, os_old))
    assert reap.latest_activity_mtime(str(tmp_path)) == new.stat().st_mtime


def test_latest_activity_mtime_missing_dir_is_none():
    assert reap.latest_activity_mtime("/no/such/dir/here") is None


def test_latest_activity_mtime_no_logs_is_none(tmp_path):
    (tmp_path / "data.fits").write_text("x")  # not a *.log
    assert reap.latest_activity_mtime(str(tmp_path)) is None


def test_is_stuck_true_when_log_stale_and_cpu_idle():
    now = 1_000_000.0
    assert reap.is_stuck(
        now=now,
        latest_mtime=now - 20 * 60,  # 20 min stale
        stale_seconds=15 * 60,
        cpu_percent=0.2,
        cpu_threshold=1.0,
    )


def test_is_stuck_false_when_cpu_busy():
    now = 1_000_000.0
    assert not reap.is_stuck(
        now=now,
        latest_mtime=now - 20 * 60,
        stale_seconds=15 * 60,
        cpu_percent=85.0,  # actively working
        cpu_threshold=1.0,
    )


def test_is_stuck_false_when_log_fresh():
    now = 1_000_000.0
    assert not reap.is_stuck(
        now=now,
        latest_mtime=now - 60,  # progressed a minute ago
        stale_seconds=15 * 60,
        cpu_percent=0.0,
        cpu_threshold=1.0,
    )


def test_is_stuck_false_when_no_log_signal():
    now = 1_000_000.0
    assert not reap.is_stuck(
        now=now,
        latest_mtime=None,
        stale_seconds=15 * 60,
        cpu_percent=0.0,
        cpu_threshold=1.0,
    )
