import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path

import multiprocess as mp
import numpy as np
from multiprocess.reduction import ForkingPickler
import yaml
from tabulate import tabulate

from prose.citations import citations as default_citations
from prose.console_utils import error, track, warning
from prose.core.image import Buffer, FITSImage, Image
from prose.utils import full_class_name


def progress(name, x, **kwargs):
    # rich keeps redirected logs clean on its own (one final line when stderr is
    # not a TTY); callers can still pass ``total`` / ``disable`` through.
    return track(
        x,
        description=name,
        total=kwargs.get("total"),
        disable=kwargs.get("disable", False),
    )


_BLOCK_NAME_PREFIX = "blocks."


def _resolve_block_class(name):
    """The ``prose.blocks`` class named by a ``"blocks.<Class>"`` string.

    Looked up by attribute, never evaluated: anything that is not a Block
    subclass in ``prose.blocks`` is rejected.
    """
    from prose import blocks as block_module
    from prose.core.block import Block

    text = str(name)
    class_name = text[len(_BLOCK_NAME_PREFIX) :]
    block_class = (
        getattr(block_module, class_name, None)
        if text.startswith(_BLOCK_NAME_PREFIX)
        else None
    )
    if not (isinstance(block_class, type) and issubclass(block_class, Block)):
        raise ValueError(f"unknown block {text!r}; expected 'blocks.<ClassName>'")
    return block_class


class Sequence:
    def __init__(self, blocks, name=None):
        """A sequence of :py:class:`Block` objects to sequentially process images

        Parameters
        ----------
        blocks : list
            list of :py:class:`Block` objects
        name : str, optional
            name of the sequence, by default None
        """
        self.name = name
        self.images = []
        self.blocks_dict = None
        self.blocks = blocks

        self.data = {}
        self.n_processed_images = None
        self.last_image = None

        # initially the buffer must have a size of max(front_size) + max(back_size) in
        # order to hold all images necessary to all blocks
        buffer_size = np.max([block.size for block in self.blocks])
        self.buffer = Buffer(buffer_size)

    def __getattr__(self, item):
        return self.blocks_dict[item]

    @property
    def blocks(self):
        """list of :py:class:`Block` objects

        Returns
        -------
        _type_
            _description_
        """
        return list(self.blocks_dict.values())

    @blocks.setter
    def blocks(self, blocks):
        self.blocks_dict = OrderedDict(
            {
                block.name if block.name is not None else "block{}".format(i): block
                for i, block in enumerate(blocks)
            }
        )

    def _set_blocks_in_sequence(self, in_sequence):
        for b in self.blocks:
            b.in_sequence = in_sequence

    def run(self, images, terminate=True, show_progress=True, loader=FITSImage):
        """Run the sequence

        Parameters
        ----------
        images : list, str, :py:class:`Image`
            :py:class:`Image` object or path (single or as a list) to be processed by the sequence
        terminate : bool, optional
            whether to run :py:class:`Sequence.terminate` at the end of the sequence, by default True
        show_progress : bool, optional
            whether to show a progress bar, by default True
        loader : Image sub-class, optional
            An Image sub-class to load images path(s) of provided as inputs, by default py:class:`Image`
        """
        self._set_blocks_in_sequence(True)
        self.images = images if not isinstance(images, (str, Path, Image)) else [images]
        assert len(self.images) != 0, "Empty array or no images provided"

        if not show_progress:

            def _p(x, **kwargs):
                return x

            self.progress = _p
        else:
            self.progress = partial(progress, self.name)

        if isinstance(self.images, list):
            if len(self.images) == 0:
                raise ValueError("No images to process")
        elif self.images is None:
            raise ValueError("No images to process")

        # run
        self.n_processed_images = 0
        self.discards = {}
        try:
            self._run(loader=loader)

            for block_name, discarded in self.discards.items():
                warning(
                    f"{block_name} discarded image{'s' if len(discarded)>1 else ''} {', '.join(discarded)}"
                )
        finally:
            if terminate:
                self.terminate()

    def _load(self, image, loader=FITSImage):
        _image = loader(image) if isinstance(image, (str, Path)) else image
        return _image

    def _run(self, loader=FITSImage):
        self.buffer.loader = partial(self._load, loader=loader)
        self.buffer.init(self.images)

        for i, buffer in enumerate(self.progress(self.buffer, total=len(self.images))):
            buffer.current.i = i
            self.last_image = buffer.current

            for block in self.blocks:
                block._run(buffer)
                # This allows to discard image in any Block
                if buffer.current is not None:
                    if buffer.current.discard:
                        self._add_discard(type(block).__name__, buffer.current.i)
                        break

            self.n_processed_images += 1

    def terminate(self):
        """Run the :py:class:`Block.terminate` method of all blocks"""
        for block in self.blocks:
            block._terminate()
        self._set_blocks_in_sequence(False)

    def __str__(self):
        rows = [
            [
                i,
                block.name,
                block.__class__.__name__,
                f"{block.processing_time:.3f} s ({(block.processing_time/self.processing_time)*100:.0f}%)",
            ]
            for i, block in enumerate(self.blocks)
        ]
        headers = ["index", "name", "type", "processing"]

        return tabulate(rows, headers, tablefmt="fancy_grid")

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def processing_time(self):
        """Total processing time of the sequence last run"""
        return np.sum([block.processing_time for block in self.blocks])

    def __getitem__(self, item):
        return self.blocks[item]

    # io
    # --

    def _add_discard(self, discard_block, i):
        if discard_block not in self.discards:
            self.discards[discard_block] = []
        self.discards[discard_block].append(str(i))

    @property
    def args(self):
        blocks = []
        for block in self.blocks:
            blocks.append({"block": full_class_name(block), **block.args})

        return blocks

    @classmethod
    def from_args(cls, args):

        blocks = []
        for block_dict in args:
            options = dict(block_dict)
            block_class = _resolve_block_class(options.pop("block"))
            blocks.append(block_class(**options))

        return cls(blocks)

    @property
    def params_str(self):
        return yaml.safe_dump(self.args, sort_keys=False)

    def citations(self):
        """Return the citations of the sequence"""

        # concatenate all blocks citations
        citations = []
        for block in self.blocks:
            citations += block.citations

        # remove duplicates
        citations = list(set(citations))
        citation_dict = {}

        for name in citations:
            if name[0] == "@":
                citation_dict[name] = name
            else:
                citation_dict[name] = default_citations[name]

        tex_citep = ", ".join(
            [
                f"{name} \citep{{{name}}}"
                for name in citation_dict.keys()
                if name not in ["prose", "astropy"]
            ]
        )
        tex_citep += " and astropy \citep{astropy}"
        tex = (
            f"This research made use of \\textsf{{prose}} \citep{{prose}} and its dependencies ({tex_citep})."
            ""
        )

        return tex, "\n\n".join(citation_dict.values())


class SequenceParallel(Sequence):
    """
    A multi-process :py:class:`Sequence` of blocks to be executed in parallel.

    .. caution::

        :py:class:`SequenceParallel` is an experimental feature and not all blocks are compatible with it.
        If you encounter issues (or for debugging) you should switch back to a normal :py:class:`Sequence`.

    The :code:`data_blocks` allow blocks carying large amount of data to be run sequentially
    so that they are not copied from one process to another.

    Parameters
    ----------
    blocks : list
        A list of blocks to be executed in parallel.
    data_blocks : list, optional
        A list of data blocks to be executed in parallel.
    name : str, optional
        A name for the sequence.
    """

    def __init__(self, blocks, data_blocks=None, name="", processes=None, chunksize=1):
        for label, value in (("processes", processes), ("chunksize", chunksize)):
            if value is None and label == "processes":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{label} must be a positive integer")
        self.processes = processes
        self.chunksize = chunksize
        super().__init__(blocks, name=name)
        if data_blocks is None:
            self.data = None
            self._has_data = False
        else:
            self.data = Sequence(data_blocks)
            self._has_data = True

    def check_data_blocks(self):
        bad_blocks = []
        for b in self.blocks:
            if b._data_block:
                bad_blocks.append(f"{b.__class__.__name__}")
        if len(bad_blocks) > 0:
            bad_blocks = ", ".join(list(np.unique(bad_blocks)))
            error(
                f"Data blocks [{bad_blocks}] cannot be used in MPSequence\n\nConsider using the data_blocks kwargs"
            )
            sys.exit()

    def _run(self, loader=FITSImage):
        self.check_data_blocks()

        self.n_processed_images = 0
        n = len(self.images)
        images_i = list(enumerate(self.images))
        # Serialize the blocks once here rather than once per image: they can
        # hold a full reference image, and re-pickling them for every task
        # made the parent process the bottleneck however many workers ran.
        payload = bytes(ForkingPickler.dumps((self.blocks, loader)))

        with mp.Pool(
            processes=self.processes, initializer=_init_worker, initargs=(payload,)
        ) as pool:
            for image in self.progress(
                pool.imap(_run_all_in_worker, images_i, chunksize=self.chunksize),
                total=n,
            ):
                if not image.discard:
                    if self._has_data:
                        self.data.run(image, terminate=False, show_progress=False)
                else:
                    self._add_discard(image.discard_block, image.i)

    def terminate(self):
        super().terminate()
        if self._has_data:
            self.data.terminate()


_worker_payload = None


def _init_worker(payload):
    global _worker_payload
    _worker_payload = payload


def _run_all_in_worker(image_i):
    # Unpickled per image so every image starts from fresh block state, as
    # it did when the blocks were sent along with each task.
    blocks, loader = ForkingPickler.loads(_worker_payload)
    return _run_all(image_i, blocks=blocks, loader=loader)


def _run_all(image_i, blocks=None, loader=None):
    i, image = image_i

    if isinstance(image, (str, Path)):
        image = loader(image)

    image.i = i

    for block in blocks:
        # This allows to discard image in any Block
        if image.discard:
            return image
        else:
            block._run(image)
            if image.discard:
                image.discard_block = type(block).__name__

    return image
