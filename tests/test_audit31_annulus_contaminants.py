import numpy as np
import pytest
from prose.scripts import run_photometry as rp


def test_incompatible_sky_geometry_is_explicit():
    with pytest.raises(ValueError, match="cannot place an uncontaminated"):
        rp._sky_annulus_pix(20.0, np.array([25.0]))


def test_feasible_sky_geometry_excludes_neighbours():
    rin, rout = rp._sky_annulus_pix(2.0, np.array([25.0]))
    assert 0 < rin < rout <= 25 - rp.CONTAM_MARGIN_PIX


def test_no_contaminants_preserves_geometry():
    assert rp._sky_annulus_pix(2.0, np.array([])) == (20.0, 30.0)
