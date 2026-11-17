import numpy as np
import pytest
from skimage import transform

from prose import Image, blocks
from prose.core import Sources


def test_transform_twirl_block(n=30):
    np.random.seed(n)
    original_transform = transform.AffineTransform(
        rotation=np.pi / 3, translation=(100, 100), scale=14.598
    )

    original_coords = np.random.rand(n, 2) * 1000
    original_image = Image(_sources=Sources(original_coords))

    transformed_image = Image(
        _sources=Sources(original_transform.inverse(original_coords))
    )

    block_transform = blocks.geometry.ComputeTransformTwirl(original_image)
    computed_transform = block_transform(transformed_image).transform

    assert computed_transform.rotation == pytest.approx(original_transform.rotation)
    assert computed_transform.translation == pytest.approx(
        original_transform.translation
    )
    assert computed_transform.scale == pytest.approx(original_transform.scale)


def test_transform_xyhift_block(n=100):
    np.random.seed(n)
    original_transform = transform.AffineTransform(translation=(132.45, 10.56))

    original_coords = np.random.rand(n, 2) * 1000
    original_image = Image(_sources=Sources(original_coords))

    transformed_image = Image(
        _sources=Sources(original_transform.inverse(original_coords))
    )

    block_transform = blocks.geometry.ComputeTransformXYShift(original_image)
    computed_transform = block_transform(transformed_image).transform

    assert computed_transform.translation == pytest.approx(
        original_transform.translation
    )


def test_align_reference_sources(n=30):
    np.random.seed(n)
    original_transform = transform.AffineTransform(
        rotation=np.pi / 3, translation=(100, 100), scale=14.598
    )

    original_coords = np.random.rand(n, 2) * 1000
    original_image = Image(_sources=Sources(original_coords))

    transformed_image = Image(
        _sources=Sources(original_transform(original_coords[0:6]))
    )

    block_align = blocks.alignment.AlignReferenceSources(original_image)
    blocks.ComputeTransformTwirl(original_image).run(transformed_image)
    block_align.run(transformed_image)
    computed_sources_coords = transformed_image.sources.coords

    np.testing.assert_allclose(
        transformed_image.transform(computed_sources_coords), original_coords
    )


def test_align_reference_sources_backward_compat(n=30):
    np.random.seed(n)
    original_transform = transform.AffineTransform(
        rotation=np.pi / 3, translation=(100, 100), scale=14.598
    )

    original_coords = np.random.rand(n, 2) * 1000
    original_image = Image(_sources=Sources(original_coords))

    transformed_image = Image(
        _sources=Sources(original_transform(original_coords[0:6]))
    )
    # backward compatibility
    transformed_image = Image(
        _sources=Sources(original_transform(original_coords[0:6]))
    )

    blocks.alignment.AlignReferenceSources(original_image).run(transformed_image)
    computed_sources_coords = transformed_image.sources.coords

    np.testing.assert_allclose(
        transformed_image.transform(computed_sources_coords), original_coords
    )


def test_align_reference_sources_tolerance_boundary():
    reference_coords = np.column_stack((np.arange(40.0), np.zeros(40)))
    image_coords = reference_coords.copy()
    image_coords[17:] += np.array([1000.0, 1000.0])

    strict = Image(_sources=Sources(image_coords))
    strict.transform = transform.AffineTransform()
    blocks.alignment.AlignReferenceSources(
        Image(_sources=Sources(reference_coords)), discard_tolerance=0.5
    ).run(strict)

    relaxed = Image(_sources=Sources(image_coords))
    relaxed.transform = transform.AffineTransform()
    blocks.alignment.AlignReferenceSources(
        Image(_sources=Sources(reference_coords)), discard_tolerance=0.4
    ).run(relaxed)

    assert strict.discard is True
    assert relaxed.discard is False


def _plate_solved_image(shape=(100, 120)):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [83.4, -6.8]
    wcs.wcs.crpix = [61.0, 51.0]  # 1-based -> 0-based pixel (60, 50)
    wcs.wcs.cdelt = [-0.62 / 3600, 0.62 / 3600]
    image = Image(np.zeros(shape))
    image.wcs = wcs
    return image


def test_trim_shifts_wcs_of_a_plate_solved_image():
    image = _plate_solved_image()
    assert image.plate_solved

    blocks.Trim((10, 16)).run(image)  # (y, x) trim, like --ccd_trim Y,X

    assert image.shape.tolist() == [100 - 2 * 10, 120 - 2 * 16]
    assert image.plate_solved
    x, y = image.wcs.world_to_pixel_values(83.4, -6.8)
    assert (float(x), float(y)) == pytest.approx((60 - 16, 50 - 10), abs=1e-6)


def test_trim_leaves_an_unsolved_image_without_wcs():
    image = Image(np.zeros((100, 120)))
    assert not image.plate_solved

    blocks.Trim(10).run(image)

    assert image.shape.tolist() == [80, 100]
    assert not image.plate_solved
