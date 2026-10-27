import numpy as np
import pytest
from prose import Image, blocks


@pytest.mark.parametrize("gain", [1.0, 2.0, 4.0])
def test_ccd_variance_in_input_units(gain):
    im = Image(np.zeros((3, 3)), metadata={"jd": 1.0, "gain": gain, "read_noise": 6.0})
    im.aperture = {"radii": np.array([2.0]), "fluxes": np.array([[1000.0]])}
    im.annulus = {
        "median": np.array([10.0]),
        "area": 100.0,
        "median_variance": np.array([0.25]),
    }
    get = blocks.GetFluxes()
    get.run(im)
    get.terminate()
    area = 4 * np.pi
    expected = 1000 / gain + area * (6 / gain) ** 2 + area**2 * 0.25
    np.testing.assert_allclose(get.fluxes.errors.item() ** 2, expected)


def test_median_sky_uncertainty_is_measured():
    from prose.core.source import PointSource, Sources

    im = Image(np.random.default_rng(3).normal(100, 10, (61, 61)))
    im.sources = Sources(
        [PointSource(coords=np.array([30.0, 30.0]))], type="PointSource"
    )
    blocks.AnnulusBackground(10, 20, scale=False).run(im)
    assert im.annulus["n_pixels"][0] > 100
    assert 0 < im.annulus["median_variance"][0] < 1
