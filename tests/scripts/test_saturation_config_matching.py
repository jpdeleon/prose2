"""A telescope name alone is not evidence of a configured saturation limit."""

import copy
import pickle
from dataclasses import asdict

import numpy as np
import pytest
from astropy.io import fits

from prose import FITSImage, Telescope
from prose.scripts.run_photometry import resolve_saturation
import prose.telescope as telescope_module


@pytest.mark.parametrize("instrument, limit", [("MuSCAT", 55000), ("MuSCAT2", 62000)])
@pytest.mark.parametrize("registered", [False, True])
def test_fits_config_match_controls_saturation(
    tmp_path, monkeypatch, instrument, limit, registered
):
    # Empty registry reproduces a fresh installation, without touching ~/.prose.
    registry = type(telescope_module.CONFIG).__new__(type(telescope_module.CONFIG))
    spec = dict(name=f"{instrument}_r", names=[instrument], saturation=limit)
    registry.telescopes_dict = {instrument: spec} if registered else {}
    monkeypatch.setattr(telescope_module, "CONFIG", registry)
    path = tmp_path / "frame.fits"
    header = fits.Header(dict(INSTRUME=instrument, TELESCOP=instrument, FILTER="r"))
    fits.writeto(path, np.zeros((4, 4)), header)
    frame = FITSImage(path, load_data=False)
    expected = (
        (float(limit), f"{instrument}_r telescope file")
        if registered
        else (None, "unknown")
    )
    assert frame.telescope._default is not registered
    for telescope in (
        frame.telescope,
        copy.deepcopy(frame.telescope),
        pickle.loads(pickle.dumps(frame.telescope)),
        Telescope.from_dict(asdict(frame.telescope)),
    ):
        assert resolve_saturation(header, instrument.lower(), telescope) == expected


@pytest.mark.parametrize("name", ["MuSCAT2", "MuSCAT2_r", "MuSCAT", "MuSCAT_g"])
def test_generic_telescope_cannot_impersonate_a_configured_name(name):
    instrument = name.split("_")[0].lower()
    telescope = Telescope(name=name)
    assert resolve_saturation({}, instrument, telescope) == (None, "unknown")


def test_valid_header_limit_does_not_require_a_config_match():
    header = dict(TELID="2m0a", SITEID="ogg", FILTER="gp", SATURATE=121600)
    assert resolve_saturation(header, "muscat3", Telescope()) == (121600, "header")
