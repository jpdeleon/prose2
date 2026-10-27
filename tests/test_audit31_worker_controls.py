import numpy as np
import pytest
from prose import Block, Image, blocks
from prose.core.sequence import SequenceParallel


class Identify(Block):
    def run(self, image):
        image.identifier = image.metadata["id"]


@pytest.mark.parametrize("workers,chunks", [(1, 1), (1, 3), (2, 2)])
def test_worker_controls_preserve_order(workers, chunks):
    seq = SequenceParallel(
        [Identify()],
        data_blocks=[blocks.Get("identifier")],
        processes=workers,
        chunksize=chunks,
    )
    seq.run(
        [Image(np.zeros((2, 2)), metadata={"id": i}) for i in range(5)],
        show_progress=False,
    )
    np.testing.assert_equal(seq.data[0].identifier, np.arange(5))


@pytest.mark.parametrize(
    "kwargs", [{"processes": 0}, {"chunksize": -1}, {"processes": 1.5}]
)
def test_invalid_worker_configuration(kwargs):
    with pytest.raises(ValueError):
        SequenceParallel([Identify()], **kwargs)


def test_photometry_sequence_forwards_worker_controls(tmp_path):
    from prose.scripts import run_photometry as rp

    args = rp.parse_args(
        [
            "--target_name",
            "test",
            "--data_dir",
            str(tmp_path),
            "--results_dir",
            str(tmp_path / "out"),
            "--workers",
            "2",
            "--chunksize",
            "4",
        ]
    )
    image = Image(np.zeros((20, 20)))
    image.epsf = Image(np.zeros((5, 5)))
    image.epsf.params = {}
    seq = rp.photometry_sequence(
        image,
        [2.0],
        4.0,
        6.0,
        n_stars_align=0,
        workers=args.workers,
        chunksize=args.chunksize,
    )
    assert seq.processes == 2 and seq.chunksize == 4


def test_cli_worker_controls_reach_run_band(tmp_path, monkeypatch):
    from astropy.io import fits
    from astropy.wcs import WCS
    from prose.scripts import run_photometry as rp

    w = WCS(naxis=2)
    w.wcs.crpix = [10, 10]
    w.wcs.cdelt = [0.01, 0.01]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()
    for key, value in {
        "TELID": "1m0a",
        "INSTRUME": "sinistro",
        "SITEID": "lsc",
        "CONFMODE": "central_2k_2x2",
        "OBJECT": "audit",
        "EXPTIME": 1.0,
        "FILTER": "gp",
        "JD": 2460000.0,
        "DATE-OBS": "2025-04-16T00:00:00",
    }.items():
        header[key] = value
    fits.writeto(tmp_path / "science.fits", np.ones((20, 20)), header)

    received = []
    monkeypatch.setattr(rp, "run_band", lambda *a, **kw: received.append(kw))
    rp.main(
        [
            "--target_name",
            "audit",
            "--target_coord",
            "0",
            "0",
            "--data_dir",
            str(tmp_path),
            "--results_dir",
            str(tmp_path / "out"),
            "--bands",
            "gp",
            "--glob",
            "*.fits",
            "--workers",
            "2",
            "--chunksize",
            "4",
        ]
    )
    assert len(received) == 1
    assert received[0]["workers"] == 2 and received[0]["chunksize"] == 4
