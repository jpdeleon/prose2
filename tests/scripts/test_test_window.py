from pathlib import Path

import pytest

from prose.scripts import run_photometry as rp
from prose.scripts.test_window import (
    frame_number,
    select_test_run_frames,
    window_around_frame_number,
)


def _raw(numbers, cam="MCT20"):
    return [f"/d/{cam}_211226{n:04d}.fits" for n in numbers]


def test_frame_number_parses_muscat2_raw_and_calibrated():
    assert frame_number("MCT20_2112260156.fits") == 156
    assert frame_number("MCT20_2112260156_calibrated.fits") == 156


def test_frame_number_parses_muscat_raw():
    assert frame_number("MSCT0_00000156.fits") == 156


def test_frame_number_none_without_number():
    assert frame_number("master_dark.png") is None


def test_select_test_run_frames_without_center_takes_first_n():
    files = _raw(range(1, 50))
    assert select_test_run_frames(files, 10) == files[:10]


def test_select_test_run_frames_centres_window_on_frame_number():
    files = _raw(range(1, 400))
    window = select_test_run_frames(files, 10, center=156)
    assert len(window) == 10
    assert [frame_number(f) for f in window] == list(range(151, 161))


def test_window_clamps_at_start_of_night():
    files = _raw(range(1, 100))
    window = window_around_frame_number(files, 2, 10)
    assert [frame_number(f) for f in window] == list(range(1, 11))


def test_window_raises_when_frame_absent_instead_of_snapping():
    files = _raw(range(1, 100))
    with pytest.raises(ValueError, match="156"):
        window_around_frame_number(files, 156, 10)


def _cache(numbers_by_cam):
    return [
        Path(f"MCT2{c}_211226{n:04d}_calibrated.fits")
        for c, nums in numbers_by_cam.items()
        for n in nums
    ]


def test_cache_has_frame_when_every_camera_covers_it():
    files = _cache({0: range(151, 161), 1: range(151, 161)})
    assert rp._calibrated_cache_has_frame(files, 156)


def test_cache_lacks_frame_from_earlier_window():
    files = _cache({0: range(56, 66), 1: range(56, 66)})
    assert not rp._calibrated_cache_has_frame(files, 156)


def test_cache_lacks_frame_when_one_camera_misses_it():
    files = _cache({0: range(151, 161), 1: range(56, 66)})
    assert not rp._calibrated_cache_has_frame(files, 156)


def test_cache_empty_has_no_frame():
    assert not rp._calibrated_cache_has_frame([], 156)


def _argv(*extra):
    return ["--target_name", "X", "--data_dir", "/d/211226", "--results_dir", "/r", *extra]


def test_calibration_args_forward_refid_as_test_run_center():
    args = rp.parse_args(_argv("--test_run", "--refid", "156", "--test_run_frames", "6"))
    calib_args = rp._calibration_args(args, Path("/c"), None)
    assert calib_args[calib_args.index("--test_run_center") + 1] == "156"
    assert calib_args[calib_args.index("--test_run_frames") + 1] == "6"


def test_calibration_args_omit_center_without_refid():
    args = rp.parse_args(_argv("--test_run"))
    calib_args = rp._calibration_args(args, Path("/c"), None)
    assert "--test_run_center" not in calib_args
