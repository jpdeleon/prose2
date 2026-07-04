# Bad Pixel Masking in run_photometry

Bad pixel masking is an image-level preprocessing step that masks detector defects (hot pixels, dead pixels, cosmic rays) before aperture photometry. This prevents defective pixels from biasing flux measurements.

## Overview

The bad pixel masking feature works by:

1. **Loading a bad pixel map** - A boolean array where `True` marks defective pixels
2. **Applying to images** - Bad pixels are masked as NaN before source detection and photometry
3. **Aperture photometry** - The photometry algorithm naturally ignores NaN values during flux extraction

This approach is superior to post-hoc imputation because it:
- Prevents bad pixels from contaminating measurements at the source
- Allows aperture photometry to handle missing data robustly
- Maintains consistency across all frames and bands
- Preserves natural photometric uncertainties

## Creating a Bad Pixel Map

### From a FITS File

The simplest approach is to store your bad pixel map as a FITS image:

```python
import numpy as np
from astropy.io import fits

# Create bad pixel map (True = bad pixel)
bad_pixels = np.zeros((2048, 2048), dtype=bool)
bad_pixels[100:105, 200:210] = True  # Region of bad pixels
bad_pixels[1500, 1500] = True         # Single hot pixel

# Save as FITS
hdu = fits.PrimaryHDU(data=bad_pixels.astype(np.uint8))
hdu.writeto('bad_pixels.fits', overwrite=True)
```

The bad pixel map can be:
- **Binary (0/1)** or **boolean (True/False)** - both are supported
- **Any shape** - will be matched against frame dimensions at runtime
- **Stored in any HDU** - the first HDU with data is used

### Programmatically in the Calibration Pipeline

If you detect bad pixels during calibration, save them directly:

```python
# After bias/dark calibration
bias_frame = load_master_bias()
hot_pixel_threshold = 5000  # ADU

# Identify hot pixels
hot_pixels = bias_frame > hot_pixel_threshold
dead_pixels = bias_frame < 10  # Very low values

bad_pixel_map = hot_pixels | dead_pixels

# Save for use in photometry
hdu = fits.PrimaryHDU(data=bad_pixel_map.astype(np.uint8))
hdu.writeto('bad_pixels.fits', overwrite=True)
```

## Using Bad Pixel Map with run_photometry

### Command Line

Pass the bad pixel map file to `run_photometry`:

```bash
python -m prose.scripts.run_photometry \
    --target_name TOI-1234 \
    --data_dir ./data \
    --results_dir ./results \
    --bad_pixel_map ./calibration/bad_pixels.fits
```

### Expected Output

When a bad pixel map is loaded, you'll see log messages:

```
2025-07-04 12:34:56 - INFO: loaded bad pixel map from ./calibration/bad_pixels.fits (HDU 0)
2025-07-04 12:34:56 - INFO: bad pixel map: shape=(2048, 2048), n_bad_pixels=1247
[gp] masked 1247 bad pixels in frame_001.fits
[gp] masked 1247 bad pixels in frame_002.fits
```

## Implementation Details

### MaskBadPixels Block

The `MaskBadPixels` block is inserted early in both the reference image and science frame processing pipelines:

```
Trim → MaskBadPixels ← [NEW]
    ↓
PointSourceDetection
    ↓
Cutouts
    ↓
MedianEPSF / EPSF Measurement
    ↓
...aperture photometry...
```

### How It Works

1. **Shape validation** - The bad pixel map is checked against image dimensions
   - If shapes don't match, masking is skipped with a warning
   - This handles situations where calibration images have different geometry

2. **Pixel masking** - Bad pixels are set to NaN
   ```python
   image.data[bad_pixel_map.astype(bool)] = np.nan
   ```

3. **Automatic handling** - NaN values are naturally handled by:
   - **Source detection** - Pixels with NaN don't contribute to source detection
   - **Aperture photometry** - NaN pixels are excluded from flux sums (np.nansum)
   - **Background estimation** - NaN pixels are excluded from background statistics

### Multi-Band Consistency

When reducing multi-band data:

1. **Same bad pixel map for all bands** - The map is applied to each band's reference and science frames
2. **Per-band differences** - Bad pixels may affect bands differently due to different wavelengths
3. **Future enhancement** - Support for per-band bad pixel maps can be added if needed

## Combining with NaN Imputation

Bad pixel masking can be combined with NaN imputation for robustness:

```bash
python -m prose.scripts.run_photometry \
    --target_name TOI-1234 \
    --data_dir ./data \
    --results_dir ./results \
    --bad_pixel_map ./calibration/bad_pixels.fits \
    --nan_imputation_method linear
```

The order of operations:

1. **During photometry** - Bad pixels are masked as NaN
2. **After photometry** - Remaining NaNs (from other sources) are imputed using the specified method
3. **Final differential photometry** - Clean flux matrix is used for Broeg et al. 2005 algorithm

## Troubleshooting

### Bad pixel map not being applied

**Check:**
- File exists and is readable: `ls -l bad_pixels.fits`
- File format is valid: `fitsheader bad_pixels.fits`
- Shape matches your data: `python -c "from astropy.io import fits; print(fits.getdata('bad_pixels.fits').shape)"`

### Too many or too few pixels masked

**Check:**
- Bad pixel thresholds were appropriate for your detector
- You used the correct master bias/dark for your camera
- Bad pixels were stored in the first HDU

### Performance degradation

If masking is too aggressive:

1. Verify your bad pixel detection wasn't too sensitive
2. Consider only masking severely defective pixels (>99th percentile)
3. Use a more conservative threshold:
   ```python
   # Instead of
   hot_pixels = bias > 5000
   # Try
   hot_pixels = bias > np.percentile(bias, 99.9)
   ```

## API Reference

### MaskBadPixels Block

```python
from prose.scripts.run_photometry import MaskBadPixels

block = MaskBadPixels(bad_pixel_map=np.array(...))
block.run(image)
```

### load_bad_pixel_map Function

```python
from prose.scripts.run_photometry import load_bad_pixel_map

# From FITS file
bad_pixels = load_bad_pixel_map("bad_pixels.fits")

# From header (future enhancement)
bad_pixels = load_bad_pixel_map("header", ref_image=reference_image)
```

Returns `np.ndarray` or `None` if loading failed.

## Examples

### Detect and save bad pixels

```python
import numpy as np
from astropy.io import fits

# Load master bias
bias = fits.getdata('master_bias.fits')

# Identify outliers (hot/dead pixels)
median_val = np.median(bias)
std_val = np.std(bias)

hot = bias > median_val + 5*std_val  # 5-sigma hot pixels
dead = bias < median_val - 5*std_val # 5-sigma dead pixels
bad = hot | dead

# Save
hdu = fits.PrimaryHDU(data=bad.astype(np.uint8))
hdu.writeto('bad_pixels.fits', overwrite=True)
print(f"Found {np.sum(bad)} bad pixels")
```

### Use in photometry

```bash
python -m prose.scripts.run_photometry \
    --target_name TOI-1234 \
    --data_dir /data/2025-01-15 \
    --results_dir ./results \
    --bands gp rp ip zs \
    --bad_pixel_map ./calib/bad_pixels.fits
```

### Inspect results

```python
# Check how many pixels were masked per frame
import logging
logging.basicConfig(level=logging.DEBUG)

# Run photometry - debug output will show masking details
```

## References

- Bad pixels in astronomical imaging: https://en.wikipedia.org/wiki/Dead_pixel
- Aperture photometry with masked pixels: Photutils documentation
- MuSCAT detector characteristics: Narita et al. 2015
