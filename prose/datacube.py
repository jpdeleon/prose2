"""Per-star, per-aperture flux datacube kept after differential photometry.

``run_photometry --save_flux_datacube`` writes ``{stem}_flux_datacube.npz``
holding, per band, the flux cube (aperture, star, frame) that differential
photometry actually ran on: after the frame cuts (missing target, sigma
clipping) and NaN imputation, but before stars were masked out of the
comparison pool. Every detected star is kept, so a light curve can later be
re-differenced against any star the automatic selection could have chosen,
at any aperture, without re-running photometry.

Star IDs are the ones shown on the run's reference image: under
``--ref_band`` these are reference-band IDs (band-local stars without a
reference-band match get ID -1 and cannot be selected).

Re-differencing uses equal weights, which is what ``--cID`` does, so a
custom light curve here matches a rerun with those cIDs at that aperture.
The default light curve is rebuilt from the stored weights (Broeg et al.
2005 in auto mode), so it matches the pipeline's own CSV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from prose import __version__ as _PROSE_VERSION
from prose.fluxes import Fluxes, diff

SCHEMA_VERSION = 1
STORE_DTYPE = np.float32
PPT = 1e3


@dataclass(frozen=True)
class BandCube:
    """One band's datacube. ``fluxes``/``errors`` are (aperture, star, frame)."""

    band: str
    fluxes: np.ndarray
    errors: np.ndarray | None
    time: np.ndarray
    bjd_tdb: np.ndarray
    aper_radii: np.ndarray
    aper_unit: str
    target_index: int
    star_ids: np.ndarray
    selectable: np.ndarray
    weights: np.ndarray
    aperture: int

    @property
    def target_id(self) -> int:
        return int(self.star_ids[self.target_index])

    @property
    def selectable_ids(self) -> np.ndarray:
        return self.star_ids[self.selectable & (self.star_ids >= 0)]

    @property
    def default_comparisons(self) -> np.ndarray:
        """Default comparison IDs, most to least weighted."""
        w = self.weights[self.aperture]
        order = np.argsort(w)[::-1]
        local = order[w[order] > 0.0]
        local = local[local != self.target_index]
        return self.star_ids[local]

    def default_weights(self) -> dict[int, float]:
        """Display ID -> default weight, normalised to sum to one."""
        w = self.weights[self.aperture]
        local = [i for i in np.flatnonzero(w > 0.0) if i != self.target_index]
        total = float(np.sum(w[local])) or 1.0
        return {int(self.star_ids[i]): float(w[i]) / total for i in local}

    def to_local(self, ids: Iterable[int]) -> np.ndarray:
        """Display IDs -> band-local star indices, validated for re-differencing."""
        ids = [int(i) for i in ids]
        if not ids:
            raise ValueError("select at least one comparison star")
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate comparison IDs in {ids}")
        local = []
        for i in ids:
            if i < 0:
                raise ValueError(f"unknown star ID {i}")
            hits = np.flatnonzero(self.star_ids == i)
            if hits.size == 0:
                raise ValueError(f"unknown star ID {i} in band {self.band}")
            k = int(hits[0])
            if k == self.target_index:
                raise ValueError(f"star {i} is the target")
            if not self.selectable[k]:
                raise ValueError(
                    f"star {i} is not selectable in band {self.band} "
                    "(excluded, saturated, or too many missing frames)"
                )
            local.append(k)
        return np.array(local, dtype=int)

    def check_aperture(self, aperture: int) -> int:
        n = self.fluxes.shape[0]
        if not 0 <= int(aperture) < n:
            raise ValueError(f"aperture index {aperture} out of range 0..{n - 1}")
        return int(aperture)


@dataclass(frozen=True)
class Datacube:
    bands: dict[str, BandCube]
    meta: dict = field(default_factory=dict)
    prose_version: str = ""


def capture(full: Fluxes, final: Fluxes, selectable: np.ndarray) -> dict:
    """Arrays for the datacube from differential photometry's inputs.

    ``full`` is the cube after imputation with every star intact; ``final``
    is what differencing ran on, whose frames are a subset of ``full``'s.
    The final frame mask includes position checks for the contributing stars.
    """
    frames = np.isin(full.time, final.time)
    if int(frames.sum()) != len(final.time):
        raise ValueError("frame times are not unique; cannot align datacube frames")
    return {
        "fluxes": full.fluxes[..., frames],
        "errors": None if full.errors is None else full.errors[..., frames],
        "time": np.asarray(full.time)[frames],
        "selectable": np.asarray(selectable, dtype=bool),
    }


def _display_ids(n_stars: int, ref_band_star_ids) -> np.ndarray:
    if ref_band_star_ids is None:
        return np.arange(n_stars)
    ids = np.full(n_stars, -1, dtype=int)
    mapping = np.asarray(ref_band_star_ids, dtype=int)[:n_stars]
    ids[: len(mapping)] = mapping
    return ids


def band_cube_from_reduction(
    band: str,
    diff: Fluxes,
    bjd_tdb: np.ndarray,
    aper_radii: np.ndarray,
    aper_unit: str,
    target_index: int,
    ref_band_star_ids=None,
    raw: dict | None = None,
) -> BandCube:
    """Build a :class:`BandCube` from a band's reduction result.

    ``raw`` defaults to ``diff.metadata["flux_datacube"]``, set by
    ``differential_photometry(keep_datacube=True)``.
    """
    raw = raw if raw is not None else diff.metadata["flux_datacube"]
    fluxes = np.asarray(raw["fluxes"])
    n_aper, n_stars = fluxes.shape[:2]
    weights = np.asarray(diff.weights, dtype=float)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, (n_aper, n_stars))
    return BandCube(
        band=band,
        fluxes=fluxes,
        errors=None if raw["errors"] is None else np.asarray(raw["errors"]),
        time=np.asarray(raw["time"], dtype=float),
        bjd_tdb=np.asarray(bjd_tdb, dtype=float),
        aper_radii=np.asarray(aper_radii, dtype=float),
        aper_unit=str(aper_unit),
        target_index=int(target_index),
        star_ids=_display_ids(n_stars, ref_band_star_ids),
        selectable=np.asarray(raw["selectable"], dtype=bool),
        weights=np.array(weights),
        aperture=int(diff.aperture),
    )


def _difference(cube: BandCube, local: np.ndarray, aperture: int, weights: np.ndarray):
    stars = np.concatenate([[cube.target_index], local])
    sub = cube.fluxes[[aperture]][:, stars, :].astype(float)
    sub_err = (
        None
        if cube.errors is None
        else cube.errors[[aperture]][:, stars, :].astype(float)
    )
    flux, err = diff(sub, weights[None, :], sub_err)
    return flux[0, 0], (None if err is None else err[0, 0])


def rediff(cube: BandCube, comparison_ids: Iterable[int], aperture: int):
    """Target light curve against ``comparison_ids`` with equal weights.

    Returns ``(flux, error)``; ``error`` is ``None`` when the cube has none.
    """
    aperture = cube.check_aperture(aperture)
    local = cube.to_local(comparison_ids)
    weights = np.concatenate([[0.0], np.ones(len(local))])
    return _difference(cube, local, aperture, weights)


def default_lightcurve(cube: BandCube):
    """The pipeline's own light curve, rebuilt from the stored weights."""
    w = cube.weights[cube.aperture]
    local = np.array(
        [i for i in np.flatnonzero(w > 0.0) if i != cube.target_index], dtype=int
    )
    weights = np.concatenate([[w[cube.target_index]], w[local]])
    return _difference(cube, local, cube.aperture, weights)


def scatter_metrics(flux: np.ndarray) -> dict:
    """Standard deviation and point-to-point scatter, in ppt.

    The point-to-point scatter (std of successive differences / sqrt 2) is
    insensitive to transits and slow trends, so it compares comparison sets
    fairly on a light curve that has a signal in it.
    """
    f = np.asarray(flux, dtype=float)
    f = f[np.isfinite(f)]
    if f.size < 3:
        return {"n": int(f.size), "std_ppt": None, "p2p_ppt": None}
    return {
        "n": int(f.size),
        "std_ppt": float(np.std(f, ddof=1) * PPT),
        "p2p_ppt": float(np.std(np.diff(f), ddof=1) / np.sqrt(2) * PPT),
    }


# ----------------------------------- I/O -----------------------------------

_ARRAY_FIELDS = ("time", "bjd_tdb", "aper_radii", "star_ids", "selectable", "weights")


def save_datacube(
    path: Path, cubes: Iterable[BandCube], meta: dict | None = None
) -> None:
    cubes = list(cubes)
    out = {
        "__schema__": np.array(SCHEMA_VERSION),
        "__bands__": np.array([c.band for c in cubes]),
        "__prose_version__": np.array(str(_PROSE_VERSION)),
        "__meta__": np.array(json.dumps(meta or {}, default=str)),
    }
    for c in cubes:
        p = f"{c.band}__"
        out[p + "fluxes"] = c.fluxes.astype(STORE_DTYPE)
        if c.errors is not None:
            out[p + "errors"] = c.errors.astype(STORE_DTYPE)
        for name in _ARRAY_FIELDS:
            out[p + name] = np.asarray(getattr(c, name))
        out[p + "aper_unit"] = np.array(c.aper_unit)
        out[p + "target_index"] = np.array(c.target_index)
        out[p + "aperture"] = np.array(c.aperture)
    np.savez_compressed(path, **out)


def load_datacube(path: Path) -> Datacube:
    with np.load(path, allow_pickle=False) as z:
        schema = int(z["__schema__"])
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported flux datacube schema {schema} (expected {SCHEMA_VERSION})"
            )
        bands = {}
        for band in (str(b) for b in z["__bands__"]):
            p = f"{band}__"
            bands[band] = BandCube(
                band=band,
                fluxes=z[p + "fluxes"],
                errors=z[p + "errors"] if p + "errors" in z else None,
                aper_unit=str(z[p + "aper_unit"]),
                target_index=int(z[p + "target_index"]),
                aperture=int(z[p + "aperture"]),
                **{name: z[p + name] for name in _ARRAY_FIELDS},
            )
        return Datacube(
            bands=bands,
            meta=json.loads(str(z["__meta__"])),
            prose_version=str(z["__prose_version__"]),
        )
