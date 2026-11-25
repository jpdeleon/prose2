"""BANZAI positional reference selection (#77)."""

from astropy.io import fits
from astropy.wcs import WCS

from prose.scripts import run_photometry as rp
from tests.scripts.test_run_photometry import _write_minimal_fits


def test_nearest_solved_reference_or_original_default(tmp_path):
    files = [_write_minimal_fits(tmp_path, f"frame{i}.fits") for i in range(4)]
    target = WCS(fits.getheader(files[0])).pixel_to_world(10, 10)
    for fp in files:
        fits.setval(fp, "WCSERR", value=4)
    assert rp._solved_positional_reference(files, 2, target) == 2
    fits.setval(files[0], "WCSERR", value=0)
    fits.setval(files[3], "WCSERR", value=0)
    assert rp._solved_positional_reference(files, 2, target) == 3
    assert rp._solved_positional_reference(files, 0, target) == 0


def test_solved_reference_must_contain_target(tmp_path):
    files = [_write_minimal_fits(tmp_path, f"frame{i}.fits") for i in range(2)]
    target = WCS(fits.getheader(files[1])).pixel_to_world(10, 10)
    for fp in files:
        fits.setval(fp, "WCSERR", value=0)
    fits.setval(files[0], "CRPIX1", value=200)
    assert rp._solved_positional_reference(files, 0, target) == 1
