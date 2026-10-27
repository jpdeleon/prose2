import numpy as np
from multiprocess.reduction import ForkingPickler
from prose import Block, Image
from prose.core import sequence


class RecordState(Block):
    def __init__(self):
        super().__init__()
        self.constant = np.ones(100)
        self.worker_shared_arrays = (self.constant,)
        self.mutable = []

    def run(self, image):
        image.previous = len(self.mutable)
        self.mutable.append(1)
        image.constant_id = id(self.constant)
        image.readonly = not self.constant.flags.writeable


def test_worker_reuses_constants_but_not_mutable_state():
    sequence._init_worker(bytes(ForkingPickler.dumps(([RecordState()], Image))))
    images = [
        sequence._run_all_in_worker((i, Image(np.zeros((2, 2))))) for i in range(3)
    ]
    assert [im.previous for im in images] == [0, 0, 0]
    assert len({im.constant_id for im in images}) == 1
    assert all(im.readonly for im in images)


def test_transform_does_not_retain_reference_pixels():
    from prose.blocks.geometry import ComputeTransformTwirl
    from prose.core.source import Sources, PointSource

    image = Image(np.zeros((128, 128)))
    image.sources = Sources(
        [
            PointSource(coords=xy)
            for xy in np.random.default_rng(1).uniform(10, 100, (6, 2))
        ],
        type="PointSource",
    )
    block = ComputeTransformTwirl(image, n=6)
    assert block.reference_image.data is None
    np.testing.assert_equal(block.ref, image.sources.coords)


def test_coordinate_only_reference_does_not_require_wcs():
    from types import SimpleNamespace

    from prose.blocks.geometry import ComputeTransformTwirl
    from prose.core.source import Sources

    reference = SimpleNamespace(
        sources=Sources(np.random.default_rng(2).uniform(10, 100, (6, 2)))
    )
    block = ComputeTransformTwirl(reference, n=6)
    assert block.reference_image.wcs is None
    np.testing.assert_equal(block.ref, reference.sources.coords)


def test_per_frame_closure_state_is_fresh():
    class ClosureBlock(Block):
        def __init__(self):
            super().__init__()
            state = []

            def record(image):
                image.previous = len(state)
                state.append(1)

            self.record = record

        def run(self, image):
            self.record(image)

    sequence._init_worker(bytes(ForkingPickler.dumps(([ClosureBlock()], Image))))
    images = [
        sequence._run_all_in_worker((i, Image(np.zeros((2, 2))))) for i in range(3)
    ]
    assert [im.previous for im in images] == [0, 0, 0]
