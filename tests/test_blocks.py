import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from prose import Block, Sequence, blocks, example_image
from prose.blocks.centroids import _PhotutilsCentroid
from prose.blocks.detection import _SourceDetection
from prose.blocks.psf import _PSFModelBase

image = blocks.PointSourceDetection()(example_image())
image_psf = image.copy()

Sequence([blocks.Cutouts(), blocks.MedianEPSF()]).run(image_psf)


def classes(module, sublcasses):
    class_members = inspect.getmembers(sys.modules[module], inspect.isclass)

    def mask(n, c):
        return issubclass(c, sublcasses) and n[0] != "_"

    return [c for n, c in class_members if mask(n, c)]


@pytest.mark.parametrize("block", classes("prose.blocks.detection", _SourceDetection))
def test_detection_blocks(block):
    block().run(image)


@pytest.mark.parametrize("block", classes("prose.blocks.centroids", _PhotutilsCentroid))
def test_centroids_blocks(block):
    block().run(image)


def test_centroid_ballet():
    from prose.blocks.centroids import CentroidBallet

    class ExactCenterModel:
        def centroid(self, cutouts):
            return np.tile([7.5, 7.5], (len(cutouts), 1))

    original = image_psf.sources.coords.copy()
    result = image_psf.copy()
    CentroidBallet(model=ExactCenterModel()).run(result)

    np.testing.assert_allclose(result.sources.coords, original)


def test_centroid_ballet_rejects_non_finite_and_large_displacements():
    from prose.blocks.centroids import CentroidBallet

    class InvalidModel:
        def centroid(self, cutouts):
            result = np.tile([7.5, 7.5], (len(cutouts), 1))
            result[0] = np.nan
            result[1] += 20
            return result

    result = image_psf.copy()
    original = result.sources.coords.copy()
    CentroidBallet(model=InvalidModel()).run(result)

    np.testing.assert_allclose(result.sources.coords[:2], original[:2])


def _centroid_test_image(data, initial, fwhm):
    from prose import Image
    from prose.core.source import PointSource, Sources

    result = Image(data=data)
    result.sources = Sources([PointSource(coords=np.asarray(initial, dtype=float))])
    result.fwhm = fwhm
    return result


def test_adaptive_centroid_uses_quadratic_for_compact_psf():
    from prose.blocks.centroids import AdaptiveCentroid

    yy, xx = np.mgrid[:81, :81]
    truth = np.array([40.35, 39.65])
    data = 1000 * np.exp(-((xx - truth[0]) ** 2 + (yy - truth[1]) ** 2) / (2 * 2**2))
    result = _centroid_test_image(data, [39.5, 40.5], fwhm=4.7)

    AdaptiveCentroid().run(result)

    assert result.centroid_methods[0] == "quadratic"
    assert result.centroid_valid[0]
    assert result.centroid_cutout == 21
    np.testing.assert_allclose(result.sources.coords[0], truth, atol=0.2)


def test_adaptive_centroid_uses_com_for_defocused_donut():
    from prose.blocks.centroids import AdaptiveCentroid

    yy, xx = np.mgrid[:121, :121]
    truth = np.array([60.4, 59.6])
    radius = np.hypot(xx - truth[0], yy - truth[1])
    data = 1000 * np.exp(-((radius - 13.0) ** 2) / (2 * 2.5**2))
    result = _centroid_test_image(data, [59.5, 60.5], fwhm=26.0)

    AdaptiveCentroid().run(result)

    assert result.centroid_methods[0] == "com"
    assert result.centroid_valid[0]
    assert result.centroid_cutout == 65
    np.testing.assert_allclose(result.sources.coords[0], truth, atol=0.2)


def test_adaptive_centroid_rejects_excessive_com_shift():
    from prose.blocks.centroids import AdaptiveCentroid

    yy, xx = np.mgrid[:101, :101]
    data = 1000 * np.exp(-((xx - 65) ** 2 + (yy - 50) ** 2) / (2 * 3**2))
    original = np.array([50.0, 50.0])
    result = _centroid_test_image(data, original, fwhm=20.0)

    AdaptiveCentroid().run(result)

    assert result.centroid_methods[0] == "region"
    assert not result.centroid_valid[0]
    np.testing.assert_array_equal(result.sources.coords[0], original)


@pytest.mark.parametrize("block", classes("prose.blocks.psf", _PSFModelBase))
def test_psf_blocks(block):
    if "JAX" in block.__name__:
        pytest.importorskip("jax")
    block().run(image_psf)


@pytest.mark.parametrize("d", [10, 50, 80, 100])
def test_sourcedetection_min_separation(d):
    from prose.blocks.detection import PointSourceDetection

    PointSourceDetection(min_separation=d).run(image)

    distances = np.linalg.norm(
        image.sources.coords - image.sources.coords[:, None], axis=-1
    )
    distances = np.where(np.eye(distances.shape[0]).astype(bool), np.nan, distances)
    distances = np.nanmin(distances, 0)
    np.testing.assert_allclose(distances > d, True)


@pytest.mark.parametrize("d", [10, 50, 80, 100])
def test_autosourcedetection_min_separation(d):
    # AutoSourceDetection.run() feeds clean() a raw ndarray of source objects
    # (unlike PointSourceDetection, which wraps them in a Sources collection).
    # Exercise the full detection path with min_separation enabled.
    from prose.blocks.detection import AutoSourceDetection

    im = image.copy()
    AutoSourceDetection(min_separation=d).run(im)

    distances = np.linalg.norm(im.sources.coords - im.sources.coords[:, None], axis=-1)
    distances = np.where(np.eye(distances.shape[0]).astype(bool), np.nan, distances)
    distances = np.nanmin(distances, 0)
    np.testing.assert_allclose(distances > d, True)


def test_clean_min_separation_accepts_ndarray():
    # Regression: with min_separation set, _SourceDetection.clean() must accept a
    # raw ndarray of source objects (the AutoSourceDetection/TraceDetection input).
    # It previously did `final_sources.coords`, raising AttributeError because an
    # ndarray has no `.coords` property (only a Sources collection does).
    from prose.blocks.detection import AutoSourceDetection
    from prose.core.source import PointSource

    sources = np.array(
        [
            PointSource(coords=np.array([0.0, 0.0]), peak=100.0),
            PointSource(coords=np.array([3.0, 0.0]), peak=50.0),  # within 5 px of first
            PointSource(coords=np.array([100.0, 100.0]), peak=10.0),
        ]
    )

    cleaned = AutoSourceDetection(min_separation=5).clean(sources)

    coords = np.array([s.coords for s in cleaned])
    # the dimmer of the two close sources (3, 0) is dropped; brightest + far survive
    assert len(cleaned) == 2
    np.testing.assert_array_equal(np.sort(coords[:, 0]), [0.0, 100.0])


def test_Trim():
    blocks.Trim(30).run(image.copy())


def test_Cutouts():
    im = blocks.Cutouts()(image)
    assert len(im._sources) == len(im.cutouts)


def test_ComputeTransform():
    from prose.blocks.geometry import ComputeTransform

    im = ComputeTransform(image.copy())(image.copy())
    assert np.allclose(im.transform, np.eye(3))


def test_MedianPSF():
    im = image.copy()
    blocks.Cutouts().run(im)
    blocks.MedianEPSF().run(im)


def test_AlignReferenceSources():
    im = image.copy()
    blocks.ComputeTransformTwirl(image.copy()).run(im)
    blocks.AlignReferenceSources(image.copy())(im)


def test_Get():
    image = example_image()
    image.a = 3
    image.b = 6
    image.header = {"C": 42}

    g = blocks.Get("a", "b", "keyword:C", arrays=False)
    g(image)
    assert g.values == {"a": [3], "b": [6], "c": [42]}


def test_peaks():
    im = image.copy()
    blocks.Peaks().run(im)


def test_LimitSources():
    from prose.core.source import PointSource, Sources

    im = image.copy()
    im.sources = Sources([PointSource(0, 0) for _ in range(2)])
    blocks.LimitSources().run(im)
    assert im.discard


def test_Del():
    im = image.copy()
    im.a = 3

    blocks.Del("a", "data").run(im)
    assert "a" not in im.computed
    assert im.data is None


def test_Apply():
    im = image.copy()
    im.a = 3

    def f(im):
        im.a += 1

    blocks.Apply(f).run(im)
    assert im.a == 4


def test_Calibration_with_arrays():
    from prose.blocks import Calibration

    im = image.copy()

    bias = np.ones_like(im.data) * 1
    dark = np.ones_like(im.data)
    flat = np.ones_like(im.data) * 0.5
    flat /= np.mean(flat)

    observed_flat = flat + bias + dark
    observed_dark = dark + bias

    # None
    expected = im.data
    Calibration().run(im)
    np.testing.assert_allclose(im.data, expected)

    # bias only
    im = image.copy()
    im.data = im.data + bias
    expected = im.data - bias
    Calibration(bias=bias).run(im)
    np.testing.assert_allclose(im.data, expected)

    # dark and bias only
    im = image.copy()
    im.data = im.data + bias + dark
    expected = im.data - bias - dark
    Calibration(darks=observed_dark, bias=bias).run(im)
    np.testing.assert_allclose(im.data, expected)

    # flat only
    im = image.copy()
    im.data = im.data * flat
    expected = im.data / flat
    Calibration(flats=flat).run(im)
    np.testing.assert_allclose(im.data, expected)

    # flat and bias only
    im = image.copy()
    im.data = (im.data * flat) + bias
    expected = (im.data - bias) / flat
    Calibration(bias=bias, flats=observed_flat).run(im)
    np.testing.assert_allclose(im.data, expected)

    # flat, dark and bias
    im = image.copy()
    im.data = (im.data * flat) + bias + dark
    expected = (im.data - bias - dark) / flat
    Calibration(bias=bias, flats=observed_flat, darks=observed_dark).run(im)
    np.testing.assert_allclose(im.data, expected)

    # empty lists and ndarray
    # this reproduce an observed bug
    im = image.copy()
    im.data = im.data + dark
    expected = im.data - dark
    Calibration(bias=np.array([], dtype=object), flats=[], darks=observed_dark).run(im)


def test_Calibration_with_files(tmp_path):
    from prose.blocks import Calibration

    im = image.copy()
    calib = image.copy()
    calib_path = tmp_path / "calib.fits"
    calib.writeto(calib_path)
    Calibration(bias=calib_path).run(im)
    Calibration(bias=[calib_path]).run(im)
    Calibration(bias=np.array([calib_path])).run(im)


def test_Calibration_only_allocates_and_cleans_shared_storage():
    from prose.blocks import Calibration

    local = Calibration()
    assert local._cal_dir is None

    shared = Calibration(shared=True)
    cal_dir = Path(shared._cal_dir)
    assert cal_dir.is_dir()
    Sequence([shared]).terminate()
    assert not cal_dir.exists()
    np.testing.assert_array_equal(shared.master_bias, [0.0])


def test_SequenceParallel_terminates_blocks():
    from prose.blocks import Calibration
    from prose.core.sequence import SequenceParallel

    shared = Calibration(shared=True)
    cal_dir = Path(shared._cal_dir)
    SequenceParallel([shared]).run([image.copy()], show_progress=False)
    assert not cal_dir.exists()


def test_SortSources():
    im = image_psf.copy()
    blocks.SortSources().run(im)
    peaks = [s.peak for s in im.sources]
    assert np.all(peaks[:-1] >= peaks[1:])


def test_require():
    im = image.copy()
    im.a = 0

    class Testa(Block):
        def __init__(self, name=None):
            super().__init__(name=name, read=["a"])

        def run(self, image):
            pass

    class Testab(Block):
        def __init__(self, name=None):
            super().__init__(name=name, read=["a", "b"])

        def run(self, image):
            pass

    Sequence([Testa()]).run(im)

    with pytest.raises(AttributeError, match="attribute 'b'"):
        Sequence([Testab()]).run(im)


def test_require_sources():
    im = image.copy()
    im.sources = None

    with pytest.raises(AttributeError, match="sources"):
        Sequence([blocks.Cutouts()]).run(im)


def test_Video(tmp_path):
    from prose.blocks import Video

    im = image.copy()
    im.sources = None

    Sequence([Video(tmp_path / "video.gif", fps=3)]).run([im, im, im])


def test_VideoPlot(tmp_path):
    from prose.blocks import VideoPlot

    def plot(image):
        image.show()

    im = image.copy()

    Sequence([VideoPlot(plot, tmp_path / "video.gif", fps=3)]).run([im, im, im])


def test_ComputeTransformTwirl_graceful_failure():
    from prose.blocks.geometry import ComputeTransformTwirl

    im1 = image.copy()
    im2 = image.copy()
    block = ComputeTransformTwirl(im1)

    def mock_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    block.solve = mock_solve
    block.run(im2)
    assert im2.discard is True
