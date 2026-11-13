"""#55 B1: one frame that raises must be discarded with its reason, not abort
the whole parallel sequence (and with it the band)."""

import numpy as np
import pytest
from prose import Block, Image, blocks
from prose.core.sequence import SequenceParallel


class Identify(Block):
    def run(self, image):
        image.identifier = image.metadata["id"]


class FailOn(Block):
    def __init__(self, bad_ids, **kwargs):
        super().__init__(**kwargs)
        self.bad_ids = set(bad_ids)

    def run(self, image):
        if image.metadata["id"] in self.bad_ids:
            raise OSError("truncated FITS")


def _images(n):
    return [Image(np.zeros((2, 2)), metadata={"id": i}) for i in range(n)]


def _sequence(bad_ids, processes=2):
    return SequenceParallel(
        [FailOn(bad_ids), Identify()],
        data_blocks=[blocks.Get("identifier")],
        processes=processes,
    )


@pytest.mark.parametrize("processes", [1, 2])
def test_failing_frame_is_discarded_and_others_kept(processes):
    seq = _sequence({3}, processes=processes)
    seq.run(_images(8), show_progress=False)

    np.testing.assert_equal(seq.data[0].identifier, [0, 1, 2, 4, 5, 6, 7])
    assert seq.discards == {"FailOn": ["3"]}
    assert len(seq.errors) == 1
    assert "truncated FITS" in seq.errors[0] and "OSError" in seq.errors[0]


def test_loader_failure_is_discarded(tmp_path):
    def loader(path):
        if "bad" in str(path):
            raise OSError("cannot read header")
        return Image(np.zeros((2, 2)), metadata={"id": 0})

    seq = SequenceParallel(
        [Identify()], data_blocks=[blocks.Get("identifier")], processes=1
    )
    seq.run(
        [str(tmp_path / "good.fits"), str(tmp_path / "bad.fits")],
        show_progress=False,
        loader=loader,
    )

    np.testing.assert_equal(seq.data[0].identifier, [0])
    assert seq.discards == {"load": ["1"]}
    assert "bad.fits" in seq.errors[0] and "cannot read header" in seq.errors[0]


def test_every_frame_failing_raises_with_first_error():
    seq = _sequence(set(range(4)))
    with pytest.raises(RuntimeError, match="truncated FITS"):
        seq.run(_images(4), show_progress=False)


def test_clean_run_records_no_errors():
    seq = _sequence(set())
    seq.run(_images(3), show_progress=False)
    assert seq.errors == []
    np.testing.assert_equal(seq.data[0].identifier, [0, 1, 2])
