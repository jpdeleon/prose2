"""Test bad pixel masking in run_photometry script.

Tests the MaskBadPixels block that masks detector defects (hot pixels,
dead pixels) before photometry to prevent bias in flux measurements.
"""

import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits

from prose.scripts.run_photometry import MaskBadPixels, load_bad_pixel_map


class MockFITSImage:
    """Mock FITSImage for testing without needing real FITS files."""

    def __init__(self, data=None, path="test.fits"):
        self.data = data if data is not None else np.ones((100, 100)) * 1000.0
        self.path = path
        self.shape = self.data.shape


class TestMaskBadPixels:
    """Test bad pixel masking functionality."""

    def test_mask_bad_pixels_marks_as_nan(self):
        """Verify MaskBadPixels marks pixels as NaN."""
        # Create a simple test image
        image = MockFITSImage()

        # Create a bad pixel map (5% hot pixels scattered throughout)
        bad_pixel_map = np.zeros((100, 100), dtype=bool)
        bad_pixel_map[10:15, 20:25] = True  # 25 bad pixels
        bad_pixel_map[50, 75] = True  # 1 more bad pixel

        # Apply masking
        block = MaskBadPixels(bad_pixel_map)
        block.run(image)

        # Verify bad pixels are NaN
        assert np.sum(np.isnan(image.data)) == 26
        assert np.isnan(image.data[10, 20])
        assert np.isnan(image.data[50, 75])
        # Good pixels should remain unchanged
        assert image.data[0, 0] == 1000.0
        assert image.data[99, 99] == 1000.0

    def test_mask_bad_pixels_handles_none_map(self):
        """Verify MaskBadPixels handles None gracefully."""
        image = MockFITSImage()

        # Apply masking with None (should be no-op)
        block = MaskBadPixels(None)
        data_before = image.data.copy()
        block.run(image)

        # Data should be unchanged
        np.testing.assert_array_equal(image.data, data_before)

    def test_mask_bad_pixels_shape_mismatch_warning(self):
        """Verify shape mismatch is handled gracefully."""
        image = MockFITSImage()

        # Create mismatched bad pixel map
        bad_pixel_map = np.zeros((50, 50), dtype=bool)

        block = MaskBadPixels(bad_pixel_map)
        data_before = image.data.copy()
        block.run(image)

        # Data should be unchanged due to shape mismatch
        np.testing.assert_array_equal(image.data, data_before)

    def test_mask_bad_pixels_detects_hot_pixels(self):
        """Verify detection and masking of hot pixels."""
        # Create image with baseline noise
        data = np.random.normal(1000, 10, (512, 512))
        image = MockFITSImage(data=data)

        # Add hot pixels (high values)
        hot_pixel_positions = [(100, 100), (200, 200), (300, 300)]
        for y, x in hot_pixel_positions:
            image.data[y, x] = 5000  # Much higher than background

        # Create bad pixel map marking these as bad
        bad_pixel_map = np.zeros((512, 512), dtype=bool)
        for y, x in hot_pixel_positions:
            bad_pixel_map[y, x] = True

        # Apply masking
        block = MaskBadPixels(bad_pixel_map)
        block.run(image)

        # Hot pixels should now be NaN
        for y, x in hot_pixel_positions:
            assert np.isnan(image.data[y, x])

        # Background should be unaffected
        background = image.data[~np.isnan(image.data)]
        assert np.mean(background) < 2000  # Still around 1000, not 5000

    def test_bad_pixel_map_contiguous_region(self):
        """Verify masking of contiguous bad pixel regions (cosmic rays, etc)."""
        data = np.ones((256, 256)) * 500.0
        image = MockFITSImage(data=data)

        # Create a bad pixel map with a contiguous region (cosmic ray trail)
        bad_pixel_map = np.zeros((256, 256), dtype=bool)
        bad_pixel_map[50:55, 100:110] = True  # 50 bad pixels in a rectangle

        block = MaskBadPixels(bad_pixel_map)
        block.run(image)

        # Verify all bad pixels are masked
        assert np.sum(np.isnan(image.data)) == 50
        # Verify good pixels are unchanged
        assert image.data[0, 0] == 500.0
        assert image.data[255, 255] == 500.0

    def test_load_bad_pixel_map_from_file(self, tmp_path):
        """Test loading bad pixel map from FITS file."""
        # Create a temporary bad pixel map FITS file
        bad_pixel_data = np.random.rand(256, 256) > 0.95  # ~5% bad pixels
        fits_path = tmp_path / "bad_pixels.fits"

        # Create FITS file with write mode
        hdu = fits.PrimaryHDU(data=bad_pixel_data.astype(np.uint8))
        hdu.writeto(fits_path, overwrite=True)

        # Load the bad pixel map
        loaded_map = load_bad_pixel_map(str(fits_path))

        assert loaded_map is not None
        assert loaded_map.shape == bad_pixel_data.shape
        np.testing.assert_array_equal(loaded_map, bad_pixel_data)

    def test_load_bad_pixel_map_from_missing_file(self):
        """Test handling of missing bad pixel map file."""
        result = load_bad_pixel_map("/nonexistent/path/bad_pixels.fits")
        assert result is None

    def test_load_bad_pixel_map_from_header_stub(self):
        """Test header-based loading (stub - requires implementation)."""
        # This test is a placeholder for future implementation
        # of reading bad pixel maps from FITS headers
        result = load_bad_pixel_map("header", ref_image=None)
        assert result is None  # Currently unsupported without ref_image


class TestBadPixelIntegration:
    """Integration tests for bad pixel masking in photometry pipeline."""

    def test_bad_pixels_dont_contribute_to_aperture_flux(self):
        """Verify bad pixels don't bias aperture photometry measurements."""
        # This would be an integration test that creates synthetic data,
        # applies bad pixel masking, and verifies flux measurements are unbiased.
        # For now, it's a placeholder for more comprehensive integration tests.
        pass
