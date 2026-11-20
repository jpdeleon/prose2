import numpy as np
import pytest
from astropy.wcs import WCS

from prose import Image
from prose.position_validation import MatchExposureSources, verified_positions
from prose.scripts import run_photometry as rp
from prose.fluxes import Fluxes
from prose import datacube


def image_with(coords):
    im = Image(np.zeros((128, 128)))
    im.sources = np.asarray(coords, dtype=float)
    im.fwhm = 6.0
    return im


def test_check_uses_detections_before_transform_and_centers_after_centroid():
    coords = np.array([[12.0, 10.0], [90.0, 15.0], [65.0, 45.0], [95.0, 85.0]])
    frame_coords = coords + [5, -3]
    ref = image_with(coords)
    science = image_with(frame_coords[[3, 0, 1]])  # star 2 is missing
    MatchExposureSources(ref).run(science)
    centers = frame_coords.copy()
    centers[0] = frame_coords[3]  # an actual star with the wrong identity
    centers[1] += [1, -1]
    science.sources = centers
    np.testing.assert_equal(verified_positions(science), [False, True, False, True])


def test_refuses_two_equally_supported_translations():
    ref = image_with([[20, 20], [70, 20], [23, 63]])
    coords = ref.sources.coords
    science = image_with(np.vstack([coords, coords + [15, 10]]))
    MatchExposureSources(ref).run(science)
    science.sources = coords
    assert not verified_positions(science).any()


@pytest.mark.parametrize("bad_wcs", [False, True])
def test_rotated_field_uses_solved_wcs(bad_wcs):
    ref = image_with([[20, 10], [70, 15], [38, 62], [89, 90]])
    w = WCS(naxis=2)
    w.wcs.crpix = [64, 64]
    w.wcs.cdelt = [-0.001, 0.001]
    w.wcs.crval = [121, 45]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ref.wcs = w
    rotated = w.deepcopy()
    rotated.wcs.pc = [[0, -1], [1, 0]]
    coords = np.column_stack(
        rotated.world_to_pixel(w.pixel_to_world(*ref.sources.coords.T))
    )
    science = image_with(coords[::-1])
    science.wcs = rotated
    science.header["WCSERR"] = 4 if bad_wcs else 0
    MatchExposureSources(ref).run(science)
    science.sources = coords
    assert verified_positions(science).all() == (not bad_wcs)


@pytest.mark.parametrize("explicit", [True, False])
def test_bad_selected_positions_cannot_be_imputed_or_exported(explicit):
    rng = np.random.default_rng(27)
    vals = rng.normal([1000, 2000, 3000], [2, 2, 2], (40, 3)).T[None, ...]
    obs = Fluxes(
        target=0,
        fluxes=vals,
        time=np.arange(40.0),
        errors=np.sqrt(vals),
        data={"frame": np.arange(40)},
    )
    pos = np.ones((40, 3), bool)
    pos[4, 0] = pos[11, 1] = pos[12, 2] = False
    obs.fluxes[0, 0, 4] = obs.fluxes[0, 1, 11] = 1e8
    result = rp.differential_photometry(
        obs,
        0,
        cids=[1] if explicit else None,
        avoid_cids=[2],
        position_valid=pos,
        keep_datacube=True,
    )
    good = pos[:, :2].all(axis=1)
    np.testing.assert_equal(result.time, obs.time[good])
    np.testing.assert_equal(result.data["frame"], np.flatnonzero(good))
    assert "_position_valid" not in result.df.columns
    expected = obs.mask(good).diff(comps=[1])
    np.testing.assert_allclose(result.flux, expected.flux)
    np.testing.assert_allclose(result.error, expected.error)
    raw = result.metadata["flux_datacube"]
    assert np.isnan(raw["fluxes"][:, 2, result.time == 12]).all()
    assert np.isnan(raw["errors"][:, 2, result.time == 12]).all()
    cube = datacube.band_cube_from_reduction("g", result, result.time, [3], "pix", 0)
    np.testing.assert_allclose(
        datacube.default_lightcurve(cube)[0], result.flux, rtol=1e-6
    )
