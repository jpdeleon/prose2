"""Identify the target as a Gaia source and place it at the observation epoch.

Name resolvers (MAST, SIMBAD, ExoFOP) return J2000.0-epoch positions, while
detected sources sit where the star is on the night and ``GaiaCatalog``
propagates Gaia positions to the image date. A target with proper motion
above ~0.2 arcsec/yr therefore drifts more than the 5 arcsec match radius
between the two, and the nearest Gaia row to the stale position can be a
neighbour (#55 A1).

These helpers are pure (no network): the caller queries Gaia and passes the
rows in.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time

__all__ = [
    "GAIA_DR2_EPOCH",
    "RESOLVER_EPOCH",
    "TARGET_GAIA_MATCH_ARCSEC",
    "GaiaTarget",
    "gaia_target_row",
    "normalize_gaia_columns",
    "propagate_linear",
    "select_gaia_target",
    "track_separation_arcsec",
]

# ``prose.utils.gaia_query`` reads Gaia DR2 (official archive or the VizieR
# mirror), whose reference epoch is J2015.5.
GAIA_DR2_EPOCH = Time(2015.5, format="jyear")
# Epoch of the positions returned by MAST/SIMBAD name resolution.
RESOLVER_EPOCH = Time(2000.0, format="jyear")
# A Gaia source is the target only if it passes this close to the input
# coordinate (resolver/catalogue positions agree to well under 1 arcsec).
TARGET_GAIA_MATCH_ARCSEC = 2.0

# Canonical column -> accepted spellings (official archive, VizieR I/345/gaia2).
_COLUMN_ALIASES = {
    "source_id": ("source_id", "SOURCE_ID", "Source", "DESIGNATION", "id"),
    "ra": ("ra", "RA_ICRS"),
    "dec": ("dec", "DE_ICRS"),
    "pmra": ("pmra", "pmRA"),
    "pmdec": ("pmdec", "pmDE"),
    "phot_g_mean_mag": ("phot_g_mean_mag", "Gmag"),
}
_REQUIRED = ("ra", "dec")


def normalize_gaia_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with canonical Gaia column names.

    Missing optional columns are filled with NaN (proper motion, magnitude) or
    the row number (``source_id``). Raises ``ValueError`` without RA/Dec.
    """
    out = {}
    for canon, aliases in _COLUMN_ALIASES.items():
        found = next((a for a in aliases if a in df.columns), None)
        if found is not None:
            out[canon] = df[found].to_numpy()
        elif canon in _REQUIRED:
            raise ValueError(f"Gaia table has no {canon!r} column ({list(df.columns)})")
        elif canon == "source_id":
            out[canon] = np.arange(len(df))
        else:
            out[canon] = np.full(len(df), np.nan)
    result = pd.DataFrame(out)
    for col in ("ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"):
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(float)
    return result


def _finite_or_zero(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr), arr, 0.0)


def propagate_linear(
    coord: SkyCoord, pmra_masyr, pmdec_masyr, from_time: Time, to_time: Time
) -> SkyCoord:
    """Move ``coord`` along its proper motion from ``from_time`` to ``to_time``.

    ``pmra_masyr`` includes the cos(dec) factor (Gaia convention). Non-finite
    proper motions (two-parameter Gaia solutions) are treated as zero. The
    motion is linear on the sky, which is accurate to well below an arcsec over
    decades for every star except a handful with large radial velocity; it
    needs no parallax, so sources with missing or negative parallax work.
    """
    dt_yr = (to_time - from_time).to_value(u.yr)
    dra = _finite_or_zero(pmra_masyr) * dt_yr
    ddec = _finite_or_zero(pmdec_masyr) * dt_yr
    position_angle = np.arctan2(dra, ddec) * u.rad
    separation = np.hypot(dra, ddec) * u.mas
    return coord.directional_offset_by(position_angle, separation)


def track_separation_arcsec(
    target: SkyCoord,
    gaia: pd.DataFrame,
    start: Time,
    end: Time,
) -> np.ndarray:
    """Closest approach [arcsec] of each Gaia source's track to ``target``.

    Each source moves from its position at ``start`` to its position at
    ``end``; the separation is the distance from ``target`` to that segment on
    the tangent plane. This matches an input coordinate of any epoch within the
    interval without the caller having to know which epoch it is.
    """
    coords = SkyCoord(gaia["ra"].to_numpy(), gaia["dec"].to_numpy(), unit="deg")
    p0 = propagate_linear(coords, gaia["pmra"], gaia["pmdec"], GAIA_DR2_EPOCH, start)
    p1 = propagate_linear(coords, gaia["pmra"], gaia["pmdec"], GAIA_DR2_EPOCH, end)
    x0, y0 = (a.to_value(u.arcsec) for a in target.spherical_offsets_to(p0))
    x1, y1 = (a.to_value(u.arcsec) for a in target.spherical_offsets_to(p1))
    dx, dy = x1 - x0, y1 - y0
    length2 = dx**2 + dy**2
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(length2 > 0, -(x0 * dx + y0 * dy) / length2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(x0 + t * dx, y0 + t * dy)


@dataclass(frozen=True)
class GaiaTarget:
    """The Gaia source identified as the target."""

    source_id: str
    g_mag: float
    pmra_masyr: float
    pmdec_masyr: float
    track_sep_arcsec: float
    coord_gaia_epoch: SkyCoord
    # Gaia sources whose track passed the match gate (1 = unambiguous).
    n_candidates: int = 1

    @property
    def pm_total_masyr(self) -> float:
        return float(np.hypot(self.pmra_masyr, self.pmdec_masyr))

    def at(self, time: Time) -> SkyCoord:
        """Position at ``time``."""
        return propagate_linear(
            self.coord_gaia_epoch,
            self.pmra_masyr,
            self.pmdec_masyr,
            GAIA_DR2_EPOCH,
            time,
        )


def select_gaia_target(
    target: SkyCoord,
    gaia: pd.DataFrame,
    obs_time: Time,
    max_sep_arcsec: float = TARGET_GAIA_MATCH_ARCSEC,
) -> GaiaTarget | None:
    """Gaia source identified as ``target``, or ``None``.

    The input coordinate may be at J2000.0 (name resolvers), J2015.5 (Gaia),
    or the observation epoch (a user-measured position). A source is a
    candidate when its track between those epochs passes within
    ``max_sep_arcsec`` of ``target``. Candidates are ranked by their
    separation at whichever of the three epochs is closest, not by the track's
    closest approach: a fast mover that merely crossed the coordinate in, say,
    2008 must not beat a star that sits on it at one of the likely epochs.
    ``n_candidates`` reports ambiguity; the caller must not fall back to the
    nearest row when ``None`` is returned. ``gaia`` must use canonical column
    names (see :func:`normalize_gaia_columns`).
    """
    if gaia is None or not len(gaia):
        return None
    epochs = [RESOLVER_EPOCH, GAIA_DR2_EPOCH, obs_time]
    start = Time(min(e.jyear for e in epochs), format="jyear")
    end = Time(max(e.jyear for e in epochs), format="jyear")
    track = track_separation_arcsec(target, gaia, start, end)
    gate = np.isfinite(track) & (track <= max_sep_arcsec)
    if not gate.any():
        return None
    coords = SkyCoord(gaia["ra"].to_numpy(), gaia["dec"].to_numpy(), unit="deg")
    at_epochs = np.min(
        [
            target.separation(
                propagate_linear(coords, gaia["pmra"], gaia["pmdec"], GAIA_DR2_EPOCH, e)
            ).arcsec
            for e in epochs
        ],
        axis=0,
    )
    best = int(np.nanargmin(np.where(gate, at_epochs, np.inf)))
    row = gaia.iloc[best]
    return GaiaTarget(
        source_id=str(row["source_id"]),
        g_mag=float(row["phot_g_mean_mag"]),
        pmra_masyr=float(_finite_or_zero(row["pmra"])),
        pmdec_masyr=float(_finite_or_zero(row["pmdec"])),
        track_sep_arcsec=float(track[best]),
        coord_gaia_epoch=SkyCoord(row["ra"], row["dec"], unit="deg"),
        n_candidates=int(gate.sum()),
    )


def gaia_target_row(
    target: SkyCoord,
    gaia_coords: SkyCoord,
    max_sep_arcsec: float = TARGET_GAIA_MATCH_ARCSEC,
) -> int | None:
    """Index of the Gaia row within ``max_sep_arcsec`` of ``target``, else ``None``.

    Both sides must be at the same epoch. Replaces "nearest row is the
    target", which silently picks a neighbour when the target is absent or
    its position is stale.
    """
    if target is None or gaia_coords is None:
        return None
    if gaia_coords.isscalar or len(gaia_coords) == 0:
        return None
    seps = np.asarray(target.separation(gaia_coords).arcsec, dtype=float)
    if not np.isfinite(seps).any():
        return None
    best = int(np.nanargmin(seps))
    return best if seps[best] <= max_sep_arcsec else None
