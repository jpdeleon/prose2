"""Tests for ``prose.core.sequence.SequenceParallel``.

The synthetic tests always run. The real-data tests compare the parallel
photometry sequence used by ``run_photometry`` against a serial ``Sequence`` on
actual observations and are skipped when the data is not on disk. Point them
at other data with ``PROSE_TEST_MUSCAT2_DIR`` / ``PROSE_TEST_QHY600_DIR``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from prose import Block, FITSImage, Sequence, blocks, example_image
from prose.core.sequence import SequenceParallel

N_IMAGES = 8
N_REAL_FRAMES = 24

# Calibrated MuSCAT2 frames (no WCS in header) and BANZAI-reduced QHY600
# frames: one instrument per reduction path that run_photometry supports.
REAL_DATASETS = {
    "muscat2": (
        os.environ.get(
            "PROSE_TEST_MUSCAT2_DIR", "/ut2/jerome/ql/prose/muscat2/260128_calibrated"
        ),
        "MCT20_*_calibrated.fits",
    ),
    "qhy600": (
        os.environ.get("PROSE_TEST_QHY600_DIR", "/data/QHY600CMOS/261008"),
        "coj0m416-sq36-20261008-*-e91.fits",
    ),
}

_serializations = 0


class CountSerializations(Block):
    """Counts how many times the parent process pickles this block."""

    def __getstate__(self):
        global _serializations
        _serializations += 1
        return self.__dict__.copy()

    def run(self, image):
        pass


class RecordRuns(Block):
    """Records how many images this block instance had processed before."""

    def run(self, image):
        image.runs_before = self.runs


class RecordIndex(Block):
    """Stores the image index so the data block can check the order."""

    def run(self, image):
        image.recorded_i = image.i


class CollectIndices(Block):
    def __init__(self):
        super().__init__()
        self.indices = []

    def run(self, image):
        self.indices.append(image.recorded_i)


def _images(n=N_IMAGES):
    image = example_image()
    images = [image.copy() for _ in range(n)]
    for i, im in enumerate(images):
        im.metadata["jd"] = 2460000.0 + i
    return images


def test_blocks_are_serialized_once_per_run():
    global _serializations
    _serializations = 0
    SequenceParallel([CountSerializations()]).run(_images(), show_progress=False)
    assert _serializations == 1


def test_each_image_gets_fresh_block_state():
    seq = SequenceParallel([RecordRuns()], data_blocks=[blocks.Get("runs_before")])
    seq.run(_images(), show_progress=False)
    np.testing.assert_array_equal(seq.data[0].values["runs_before"], [0] * N_IMAGES)


def test_results_keep_input_order():
    collect = CollectIndices()
    SequenceParallel([RecordIndex()], data_blocks=[collect]).run(
        _images(), show_progress=False
    )
    assert collect.indices == list(range(N_IMAGES))


def test_matches_serial_sequence_on_simulated_images():
    def build():
        return [
            blocks.PointSourceDetection(n=10),
            blocks.AperturePhotometry(radii=[3, 5], scale=False),
            blocks.AnnulusBackground(rin=8, rout=12, scale=False),
        ]

    parallel = SequenceParallel(build(), data_blocks=[blocks.GetFluxes()])
    parallel.run(_images(), show_progress=False)
    serial = Sequence([*build(), blocks.GetFluxes()])
    serial.run(_images(), show_progress=False)
    np.testing.assert_array_equal(
        parallel.data[0].fluxes.fluxes, serial[-1].fluxes.fluxes
    )


def _real_frames(name):
    directory, pattern = REAL_DATASETS[name]
    files = sorted(Path(directory).glob(pattern))
    if len(files) <= N_REAL_FRAMES:
        pytest.skip(f"{name} data not available in {directory}")
    return files[: N_REAL_FRAMES + 1]


@pytest.mark.parametrize("name", list(REAL_DATASETS))
def test_photometry_sequence_matches_serial_on_real_data(name):
    from prose.scripts import run_photometry as rp

    files = _real_frames(name)
    ref = FITSImage(files[0])
    rp.reference_sequence().run(ref, show_progress=False)

    def build():
        return rp.photometry_sequence(
            ref, aper_radii=np.linspace(5, 15, 5), rin=20, rout=30
        )

    parallel = build()
    parallel.run(files[1:], show_progress=False)

    template = build()
    serial = Sequence([*template.blocks, *template.data.blocks])
    serial.run(files[1:], show_progress=False)

    par, ser = parallel.data[0].fluxes, serial[-1].fluxes
    assert par.fluxes.shape[-1] > 0
    np.testing.assert_array_equal(par.time, ser.time)
    np.testing.assert_array_equal(par.fluxes, ser.fluxes)
