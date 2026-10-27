import numpy as np
import pytest
from prose import Image, blocks
from prose.core.source import PointSource, Sources


@pytest.mark.parametrize("xy", [(0, 0), (0, 20), (20, 20)])
def test_uniform_sky_subtracts_to_zero_at_edges(xy):
    im = Image(np.ones((41, 41)) * 100, metadata={"jd": 1.0})
    im.sources = Sources(
        [PointSource(coords=np.array(xy, dtype=float))], type="PointSource"
    )
    blocks.AperturePhotometry([3.0, 5.0], scale=False).run(im)
    blocks.AnnulusBackground(8, 10, scale=False).run(im)
    np.testing.assert_allclose(im.annulus["median"], 100)
    get = blocks.GetFluxes()
    get.run(im)
    get.terminate()
    np.testing.assert_allclose(get.fluxes.fluxes, 0, atol=1e-10)
    assert np.all(im.aperture["truncated"] == (xy != (20, 20)))


def test_absent_sky_is_invalid():
    im = Image(np.ones((5, 5)) * 100)
    im.sources = Sources([PointSource(coords=np.array([2.0, 2.0]))], type="PointSource")
    blocks.AnnulusBackground(8, 10, scale=False).run(im)
    assert np.isnan(im.annulus["median"]).all()
