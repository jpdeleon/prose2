"""``gaia_query`` service fallback, query form and result columns (#102)."""

from types import SimpleNamespace

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.utils import tap

from prose import utils
from prose.blocks.catalogs import GaiaCatalog

ARI_URL = "https://gaia.ari.uni-heidelberg.de/tap"
VIZIER_URL = "https://tapvizier.u-strasbg.fr/TAPVizieR/tap"
CENTER = SkyCoord(150.0, 20.0, unit="deg")


def _result(designation="designation"):
    return Table(
        {
            designation: ["Gaia DR2 1"],
            "ra": [150.0] * u.deg,
            "dec": [20.0] * u.deg,
            "pmra": [1.0] * u.mas / u.yr,
            "pmdec": [1.0] * u.mas / u.yr,
            "parallax": [2.0] * u.mas,
        }
    )


@pytest.fixture
def services(monkeypatch):
    """Fake the three TAP services; ``fail`` names the ones that raise."""
    state = SimpleNamespace(calls=[], fail=set(), designation="designation")

    def launch(name, query):
        state.calls.append((name, query))
        if name in state.fail:
            raise ConnectionError(f"{name} down")
        return SimpleNamespace(get_results=lambda: _result(state.designation))

    class FakeTapPlus:
        def __init__(self, url):
            self.name = {ARI_URL: "ARI", VIZIER_URL: "VizieR"}[url]

        def launch_job(self, query):
            return launch(self.name, query)

    monkeypatch.setattr(utils, "_query_esa", lambda query: launch("ESA", query).get_results())
    monkeypatch.setattr(tap, "TapPlus", FakeTapPlus)
    return state


def _names(state):
    return [name for name, _ in state.calls]


@pytest.mark.parametrize(
    "down, expected",
    [
        (set(), ["ESA"]),
        ({"ESA"}, ["ESA", "ARI"]),
        ({"ESA", "ARI"}, ["ESA", "ARI", "VizieR"]),
    ],
)
def test_services_are_tried_in_order(services, down, expected):
    services.fail = down

    table = utils.gaia_query(CENTER, 0.1, "*")

    assert _names(services) == expected
    assert len(table) == 1


def test_raises_the_last_error_when_every_service_fails(services):
    services.fail = {"ESA", "ARI", "VizieR"}

    with pytest.raises(ConnectionError, match="VizieR down"):
        utils.gaia_query(CENTER, 0.1, "*")


def test_each_failed_service_is_logged(services, caplog):
    services.fail = {"ESA", "ARI"}

    utils.gaia_query(CENTER, 0.1, "*")

    assert "ESA Gaia TAP query failed (ESA down)" in caplog.text
    assert "ARI Gaia TAP query failed (ARI down)" in caplog.text


def test_circular_query_uses_the_indexable_cone_form(services):
    services.fail = {"ESA", "ARI"}

    utils.gaia_query(CENTER, 0.1, "*")

    for name, query in services.calls:
        assert (
            "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 150.0, 20.0, 0.05))"
            in query
        ), name


def test_ari_runs_the_official_query(services):
    services.fail = {"ESA"}

    utils.gaia_query(CENTER, 0.1, "*")

    (_, esa_query), (_, ari_query) = services.calls
    assert ari_query == esa_query
    assert "from gaiadr2.gaia_source " in ari_query


@pytest.mark.parametrize("spelling", ["designation", "DESIGNATION"])
def test_designation_column_has_one_spelling(services, spelling):
    services.designation = spelling

    table = utils.gaia_query(CENTER, 0.1, "*")

    assert "DESIGNATION" in table.colnames
    assert "designation" not in table.colnames


def test_gaia_catalog_accepts_a_lowercase_designation(services):
    services.fail = {"ESA"}
    image = SimpleNamespace(
        wcs=SimpleNamespace(pixel_to_world=lambda x, y: CENTER),
        shape=(100, 100),
        fov=[0.1, 0.1] * u.deg,
        date="2020-01-01T00:00:00",
    )

    table = GaiaCatalog().get_catalog(image)

    assert list(table["id"]) == ["Gaia DR2 1"]
