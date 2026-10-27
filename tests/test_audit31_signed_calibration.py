import numpy as np
import pytest
from prose import Image, blocks


@pytest.mark.parametrize("shared", [False, True])
def test_calibration_preserves_signed_noise_and_invalid_pixels(shared):
    calibration = blocks.Calibration(shared=shared, verbose=False)
    try:
        image = Image(np.array([[-100.0, -2.0, 0.0, 2.0, np.inf, np.nan]]))
        calibration.run(image)
        np.testing.assert_equal(image.data, [[-100.0, -2.0, 0.0, 2.0, np.nan, np.nan]])
    finally:
        calibration.terminate()


def test_zero_flat_is_invalid_not_a_finite_measurement():
    calibration = blocks.Calibration(
        flats=np.array([[0.0, 1.0], [1.0, 1.0]]), verbose=False
    )
    image = Image(np.ones((2, 2)))
    calibration.run(image)
    assert np.isnan(image.data[0, 0])
