import numpy as np
import pytest

from prose import Image
from prose.blocks import AutoSourceDetection, PointSourceDetection
from prose.blocks.detection import _axis_ratio

STAR_XY = (40, 40)
TRAIL_ROW = 100


def _star_and_trail():
    """A round Gaussian star plus a long, thin horizontal trail."""
    rng = np.random.default_rng(0)
    y, x = np.indices((160, 160))
    star = 2000 * np.exp(
        -0.5 * ((x - STAR_XY[0]) ** 2 + (y - STAR_XY[1]) ** 2) / 3.0**2
    )
    trail = np.where((np.abs(y - TRAIL_ROW) <= 1) & (x >= 30) & (x <= 130), 1500.0, 0.0)
    return Image(100 + star + trail + rng.normal(0, 5, x.shape))


def _detected_rows(block):
    image = _star_and_trail()
    block.run(image)
    return sorted(int(round(s.coords[1])) for s in image.sources)


def test_default_keeps_elongated_detections():
    assert _detected_rows(PointSourceDetection()) == [STAR_XY[1], TRAIL_ROW]


def test_min_axis_ratio_rejects_trail_and_keeps_star():
    assert _detected_rows(PointSourceDetection(min_axis_ratio=0.5)) == [STAR_XY[1]]


def test_auto_source_detection_accepts_min_axis_ratio():
    assert _detected_rows(AutoSourceDetection(min_axis_ratio=0.5)) == [STAR_XY[1]]


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_min_axis_ratio_outside_unit_interval_raises(value):
    with pytest.raises(ValueError, match="min_axis_ratio"):
        PointSourceDetection(min_axis_ratio=value)


def test_single_pixel_region_counts_as_round():
    # axis_major_length is 0 for a one-pixel region; treat it as round
    # rather than dividing by zero.
    region = type("R", (), {"axis_major_length": 0.0, "axis_minor_length": 0.0})()
    assert _axis_ratio(region) == 1.0


def _trails_and_star(n_trails=3):
    """One round star plus several thin horizontal trails."""
    rng = np.random.default_rng(1)
    y, x = np.indices((200, 200))
    data = 100 + 2000 * np.exp(-0.5 * ((x - 40) ** 2 + (y - 40) ** 2) / 3.0**2)
    for k in range(n_trails):
        row = 90 + 40 * k
        data = data + np.where((np.abs(y - row) <= 1) & (x >= 30) & (x <= 170), 1500, 0)
    return Image(data + rng.normal(0, 5, x.shape))


def test_rejected_detections_are_recorded_on_the_image():
    image = _star_and_trail()
    PointSourceDetection(min_axis_ratio=0.5).run(image)
    (rejected,) = image.computed["axis_ratio_rejected"]
    assert round(rejected["y"]) == TRAIL_ROW
    assert rejected["axis_ratio"] < 0.1


def test_no_cut_records_no_rejections():
    image = _star_and_trail()
    PointSourceDetection().run(image)
    assert image.computed["axis_ratio_rejected"] == []


def test_rejections_are_logged_at_debug(caplog):
    caplog.set_level("DEBUG", logger="prose")
    PointSourceDetection(min_axis_ratio=0.5).run(_star_and_trail())
    assert "min_axis_ratio=0.5 rejected 1 detection" in caplog.text


def test_one_rejection_in_small_field_does_not_warn(caplog):
    caplog.set_level("WARNING", logger="prose")
    PointSourceDetection(min_axis_ratio=0.5).run(_star_and_trail())
    assert "min_axis_ratio" not in caplog.text


def test_warns_when_cut_removes_large_share_of_detections(caplog):
    # Poor tracking can push real stars under the cut; flag it loudly.
    caplog.set_level("WARNING", logger="prose")
    PointSourceDetection(min_axis_ratio=0.5).run(_trails_and_star(3))
    assert "min_axis_ratio=0.5 rejected 3 of 4 detections" in caplog.text
