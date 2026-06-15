import numpy as np
import pytest

from prose import FITSImage
from prose.core.image import Buffer, Image
from prose.core.source import PointSource, Sources
from prose.simulations import fits_image


def test_init_append(n=5):
    buffer = Buffer(n)
    init = np.random.randint(0, 20, size=20)
    buffer.init(init)
    np.testing.assert_equal(
        buffer.items[buffer.mid_index + 1 :], init[: buffer.mid_index]
    )
    buffer.append(4)
    assert buffer.items[-1] == 4


def test_buffer_iter():
    buffer = Buffer(5)
    data = np.random.randint(0, 20, 20)
    buffer.init(data)
    for i, buf in enumerate(buffer):
        assert buf.current == data[i]


def test_cutout(coords=(0, 0)):
    image = Image(data=np.random.rand(100, 100))
    im = image.cutout(coords, 5, wcs=False)
    assert im.data.shape == (5, 5)


def test_data_cutouts():
    image = Image(data=np.random.rand(100, 100))
    coords = np.random.rand(10, 2)
    cutouts = image.data_cutouts(coords, 5)


def test_plot_sources():
    image = Image(data=np.random.rand(100, 100))
    image.sources = Sources([PointSource(coords=(0, 0), i=i) for i in range(5)])
    image.show()
    image.sources[[0, 1, 3]].plot()
    image.sources[0].plot()
    # seen in a bug
    image.sources[np.int64(0)].plot()


def test_fitsimage(tmp_path):
    filename = tmp_path / "test.fits"
    fits_image(np.random.rand(100, 100), {}, filename)

    loaded_image = FITSImage(filename)
    assert "IMAGETYP" in dict(loaded_image.header)


def test_init_header():

    d = np.ones((512, 512))
    hdr = dict(FILTER="B")

    assert Image(d, header=hdr).header == hdr


def test_parse_sexagesimal_seconds_overflow():
    # MuSCAT2/TCS headers can write a seconds field >= 60 (e.g. +20:11:181),
    # which astropy's Angle rejects. We carry the overflow arithmetically.
    import astropy.units as u

    from prose.core.image import _parse_sexagesimal

    # DEC "+20:11:181" -> 20 + 11/60 + 181/3600 deg
    assert _parse_sexagesimal("+20:11:181", u.deg).deg == pytest.approx(20.233611, abs=1e-5)
    # RA "4:06:38" in hourangle -> (4 + 6/60 + 38/3600) * 15 deg
    assert _parse_sexagesimal("4:06:38", u.hourangle).deg == pytest.approx(61.658333, abs=1e-5)


def test_parse_sexagesimal_sign_and_canonical():
    import astropy.units as u

    from prose.core.image import _parse_sexagesimal

    # Negative declination keeps its sign.
    assert _parse_sexagesimal("-05:30:00", u.deg).deg == pytest.approx(-5.5)
    # A canonical (in-range) sexagesimal string still parses correctly.
    assert _parse_sexagesimal("+28:17:45", u.deg).deg == pytest.approx(28.295833, abs=1e-5)


def test_parse_sexagesimal_decimal_fallback():
    import astropy.units as u

    from prose.core.image import _parse_sexagesimal

    # Plain decimal strings (no colons) defer to astropy unchanged.
    assert _parse_sexagesimal("61.6", u.deg).deg == pytest.approx(61.6)
