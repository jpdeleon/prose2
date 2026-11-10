"""Regression tests for MeasurePeaks (issue #53).

A defocused star is a ring; its brightest pixels sit on the ring. The peak must
be taken over the photometric aperture, not a fixed 11x11 box.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from prose.scripts import run_photometry as rp

IMAGE_SIZE = 121
RING_WIDTH = 1.5


def _donut(n: int, center: tuple[float, float], r_ring: float, amp: float = 1.0):
    y, x = np.mgrid[:n, :n]
    rr = np.hypot(x - center[0], y - center[1])
    return amp * np.exp(-0.5 * ((rr - r_ring) / RING_WIDTH) ** 2)


def _image(data: np.ndarray, coords, fwhm: float = 1.0):
    return SimpleNamespace(
        data=data,
        shape=data.shape,
        sources=SimpleNamespace(coords=np.asarray(coords, dtype=float)),
        fwhm=fwhm,
        computed={},
    )


@pytest.mark.parametrize("r_ring", [2, 4, 6, 8, 10, 12])
def test_peak_recovers_ring_maximum_for_defocused_star(r_ring):
    # Arrange: aperture radius covers the ring, as the pipeline's largest aperture does
    c = (IMAGE_SIZE - 1) / 2
    data = _donut(IMAGE_SIZE, (c, c), r_ring)
    image = _image(data, [(c, c)])

    # Act
    rp.MeasurePeaks(radius=r_ring + 3 * RING_WIDTH).run(image)

    # Assert
    assert image.computed["peaks"][0] / data.max() >= 0.99


def test_peak_scales_radius_with_fwhm_when_scale_true():
    c = (IMAGE_SIZE - 1) / 2
    data = _donut(IMAGE_SIZE, (c, c), r_ring=12)
    image = _image(data, [(c, c)], fwhm=8.0)

    rp.MeasurePeaks(radius=2.0, scale=True).run(image)  # 16 px

    assert image.computed["peaks"][0] / data.max() >= 0.99
    assert image.computed["peak_radius"] == pytest.approx(16.0)


def test_peak_excludes_brighter_neighbour_outside_aperture():
    # A brighter star ~19.8 px away (diagonal) sits inside a 15 px half-width
    # square box but outside a 15 px circular aperture.
    data = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    data[60, 60] = 100.0
    data[60 + 14, 60 + 14] = 1000.0  # ~19.8 px away, inside a 15 px square box
    image = _image(data, [(60, 60)])

    rp.MeasurePeaks(radius=15).run(image)

    assert image.computed["peaks"][0] == pytest.approx(100.0)


def test_peak_handles_source_at_image_edge_and_nan_pixels():
    data = np.full((20, 20), np.nan)
    data[0, 1] = 5.0
    image = _image(data, [(0.0, 0.0), (500.0, 500.0)])

    rp.MeasurePeaks(radius=3).run(image)

    peaks = image.computed["peaks"]
    assert peaks[0] == pytest.approx(5.0)
    assert np.isnan(peaks[1])  # off-image source has no pixels


def test_radius_never_below_one_pixel():
    data = np.zeros((11, 11))
    data[5, 6] = 7.0
    image = _image(data, [(5, 5)])

    rp.MeasurePeaks(radius=0.1).run(image)

    assert image.computed["peaks"][0] == pytest.approx(7.0)
    assert image.computed["peak_radius"] == pytest.approx(1.0)
