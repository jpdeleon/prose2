import numpy as np
from prose import Image, blocks
from prose.core.source import PointSource, Sources
from prose.scripts.run_photometry import MaskBadPixels


def test_bad_pixel_partial_sums_are_flagged_not_exported():
    im = Image(np.ones((41, 41)) * 100, metadata={"jd": 1.0})
    im.path = "synthetic.fits"
    im.sources = Sources(
        [PointSource(coords=np.array([20.0, 20.0]))], type="PointSource"
    )
    mask = np.zeros((41, 41), bool)
    mask[20, 24] = True
    MaskBadPixels(mask).run(im)
    blocks.AperturePhotometry([2.0, 5.0], scale=False).run(im)
    blocks.AnnulusBackground(8, 10, scale=False).run(im)
    assert np.isfinite(im.aperture["fluxes"]).all()
    np.testing.assert_equal(im.aperture["valid"], [[True, False]])
    np.testing.assert_allclose(im.aperture["masked_area"], [[0.0, 1.0]])
    get = blocks.GetFluxes()
    get.run(im)
    get.terminate()
    assert np.isfinite(get.fluxes.fluxes[0]).all()
    assert np.isnan(get.fluxes.fluxes[1]).all()
    assert np.isnan(get.fluxes.errors[1]).all()
