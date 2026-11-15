"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_gaia_target_resolution(request, monkeypatch):
    """Keep ``run_photometry.main`` offline: target identification queries
    Gaia (#55 A1). Tests of the resolver itself opt out with
    ``@pytest.mark.gaia_resolution``."""
    if request.node.get_closest_marker("gaia_resolution"):
        return
    from prose.scripts import run_photometry as rp

    monkeypatch.setattr(rp, "resolve_target_at_epoch", lambda coord, _time: coord)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gaia_resolution: run the real resolve_target_at_epoch"
    )
