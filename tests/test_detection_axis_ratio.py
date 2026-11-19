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
