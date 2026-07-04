#!/usr/bin/env python
"""Example: Create and use a bad pixel map for run_photometry.

This example demonstrates:
1. Detecting bad pixels from a master bias/dark frame
2. Saving the map as a FITS file
3. Using it in the photometry pipeline
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
import subprocess
import sys


def create_bad_pixel_map_from_bias(
    bias_path: str,
    hot_pixel_threshold: float = 5000,
    dead_pixel_threshold: float = 100,
    output_path: str = "bad_pixels.fits",
):
    """Create a bad pixel map from a master bias frame.

    Parameters
    ----------
    bias_path : str
        Path to master bias FITS file
    hot_pixel_threshold : float
        ADU threshold for hot pixel detection (default: 5000)
    dead_pixel_threshold : float
        ADU threshold for dead pixel detection (default: 100)
    output_path : str
        Output FITS file for bad pixel map (default: bad_pixels.fits)
    """
    print(f"Loading master bias from {bias_path}")
    bias = fits.getdata(bias_path)
    print(f"Bias shape: {bias.shape}")

    # Identify hot pixels (very bright)
    hot_pixels = bias > hot_pixel_threshold
    n_hot = np.sum(hot_pixels)
    print(f"Found {n_hot} hot pixels (>{hot_pixel_threshold} ADU)")

    # Identify dead pixels (very dim)
    dead_pixels = bias < dead_pixel_threshold
    n_dead = np.sum(dead_pixels)
    print(f"Found {n_dead} dead pixels (<{dead_pixel_threshold} ADU)")

    # Combine into single bad pixel map
    bad_pixel_map = hot_pixels | dead_pixels
    n_bad = np.sum(bad_pixel_map)
    print(f"Total bad pixels: {n_bad} ({100*n_bad/bias.size:.2f}%)")

    # Save as FITS
    print(f"\nSaving bad pixel map to {output_path}")
    hdu = fits.PrimaryHDU(data=bad_pixel_map.astype(np.uint8))
    hdu.header.add_comment("Bad pixel map: 1=bad, 0=good")
    hdu.header["HOTPIX"] = hot_pixel_threshold
    hdu.header["DEADPIX"] = dead_pixel_threshold
    hdu.writeto(output_path, overwrite=True)

    return output_path


def create_bad_pixel_map_from_dark(
    dark_path: str,
    dark_exposure_time: float,
    hot_pixel_rate_threshold: float = 10.0,
    output_path: str = "bad_pixels.fits",
):
    """Create a bad pixel map from a master dark frame.

    Uses the dark current rate (e-/s) to identify hot pixels that accumulate
    signal faster than expected. This is particularly sensitive to hot pixels.

    Parameters
    ----------
    dark_path : str
        Path to master dark FITS file
    dark_exposure_time : float
        Exposure time of dark frame (seconds)
    hot_pixel_rate_threshold : float
        e-/s threshold for hot pixel detection (default: 10 e-/s)
    output_path : str
        Output FITS file for bad pixel map
    """
    print(f"Loading master dark from {dark_path}")
    dark = fits.getdata(dark_path)
    print(f"Dark shape: {dark.shape}, exposure time: {dark_exposure_time}s")

    # Compute dark current rate (e-/s)
    # Assumes dark is in electrons (after calibration)
    dark_rate = dark / dark_exposure_time

    # Identify hot pixels by excessive dark current
    hot_pixels = dark_rate > hot_pixel_rate_threshold
    n_hot = np.sum(hot_pixels)
    print(f"Found {n_hot} hot pixels (>{hot_pixel_rate_threshold} e-/s)")

    # Save
    print(f"Saving bad pixel map to {output_path}")
    hdu = fits.PrimaryHDU(data=hot_pixels.astype(np.uint8))
    hdu.header.add_comment("Bad pixel map from dark current: 1=bad, 0=good")
    hdu.header["DARKFILE"] = Path(dark_path).name
    hdu.header["EXPTIME"] = dark_exposure_time
    hdu.header["THRESHOLD"] = hot_pixel_rate_threshold
    hdu.writeto(output_path, overwrite=True)

    return output_path


def run_photometry_with_bad_pixels(
    target_name: str,
    data_dir: str,
    results_dir: str,
    bad_pixel_map: str,
    bands: list = None,
    ref_band: str = None,
):
    """Run photometry with bad pixel masking.

    Parameters
    ----------
    target_name : str
        Target name (e.g., "TOI-1234")
    data_dir : str
        Directory containing FITS files
    results_dir : str
        Output directory for results
    bad_pixel_map : str
        Path to bad pixel map FITS file
    bands : list, optional
        List of bands to reduce (default: ["gp", "rp", "ip", "zs"])
    ref_band : str, optional
        Reference band for multi-band alignment (default: first band)
    """
    if bands is None:
        bands = ["gp", "rp", "ip", "zs"]
    if ref_band is None:
        ref_band = bands[0]

    print("\n" + "=" * 60)
    print("Running photometry with bad pixel masking")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "prose.scripts.run_photometry",
        "--target_name",
        target_name,
        "--data_dir",
        data_dir,
        "--results_dir",
        results_dir,
        "--bands",
        *bands,
        "--ref_band",
        ref_band,
        "--bad_pixel_map",
        bad_pixel_map,
        # Optional: enable NaN imputation for extra robustness
        "--nan_imputation_method",
        "linear",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Example workflow: create and use bad pixel map."""
    print(__doc__)

    # Step 1: Create bad pixel map from master bias
    # (assuming you have a master bias file)
    bias_file = "calibration/master_bias.fits"
    if Path(bias_file).exists():
        bad_pixel_map = create_bad_pixel_map_from_bias(
            bias_file,
            hot_pixel_threshold=5000,
            dead_pixel_threshold=100,
            output_path="calibration/bad_pixels.fits",
        )
    else:
        print(f"\n⚠️  Master bias file not found: {bias_file}")
        print("   Skipping bad pixel map creation")
        print("   To use this feature, provide a master bias or dark frame")
        bad_pixel_map = None

    # Step 2: Run photometry with bad pixel masking
    if bad_pixel_map and Path(bad_pixel_map).exists():
        print("\n" + "=" * 60)

        # Example parameters (adjust to your data)
        target_name = "TOI-1234"
        data_dir = "./data"
        results_dir = f"./{target_name}_results"

        if Path(data_dir).exists():
            returncode = run_photometry_with_bad_pixels(
                target_name=target_name,
                data_dir=data_dir,
                results_dir=results_dir,
                bad_pixel_map=bad_pixel_map,
                bands=["gp", "rp", "ip", "zs"],
                ref_band="gp",
            )

            if returncode == 0:
                print("\n✓ Photometry completed successfully!")
                print(f"Results saved to {results_dir}")
            else:
                print(f"\n✗ Photometry failed with exit code {returncode}")
        else:
            print(f"\n⚠️  Data directory not found: {data_dir}")
            print("   To run photometry, provide a data directory with FITS files")

    print("\n" + "=" * 60)
    print("For more details, see docs/BAD_PIXEL_MASKING.md")


if __name__ == "__main__":
    main()
