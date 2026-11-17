"""Re-difference a finished run with other comparison stars or another aperture.

Reads the run's ``*_flux_datacube.npz`` (``run_photometry --save_flux_datacube``)
and, per selected band, rebuilds the target light curve against a chosen set
of comparison stars at a chosen aperture index, with equal weights (what
``--cID`` does). Three modes, each printing one JSON object on stdout:

* ``--info``: per band, the selectable star IDs, the default comparisons and
  their weights, and the aperture radii, so a caller can build a picker.
* ``--select BAND=ID,ID,...@APERTURE [--preview out.png]``: compare the
  custom light curve with the pipeline's default (scatter metrics and an
  overlay plot). Nothing in the run is written.
* ``--select ... --apply``: overwrite the ``Flux`` and ``Err`` columns of the
  selected bands' CSVs in place and regenerate the run's summary
  ``*_lightcurves.png``. Every other column is left as it is. The CSV rows
  must still be the datacube's frames, so a post-process clip has to be
  undone first.

Star IDs are those on the run's reference image (reference-band IDs under
``--ref_band``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prose import datacube as dc
from prose.scripts.postprocess_lightcurves import (
    ERR_KEY,
    FLUX_KEY,
    TIME_KEY,
    _band_of,
    _guess_min_time,
    _summary_png,
    band_csvs,
    plot_lightcurves_from_csv,
    read_lightcurve,
)
from prose.scripts.run_photometry import _binned, _savefig, band_color

logger = logging.getLogger("prose_rediff_lightcurves")

DATACUBE_GLOB = "*_flux_datacube.npz"
# CSV BJDs are written at full float64 precision; this only absorbs rounding.
BJD_MATCH_ATOL_DAYS = 1e-6
MAD_TO_SIGMA = 1.4826
OFFSET_SIGMAS = 8
YLIM_PERCENTILES = (0.5, 99.5)
_SELECT_RE = re.compile(r"^([^=@\s]+)=([0-9]+(?:,[0-9]+)*)@([0-9]+)$")


def parse_select(text: str) -> tuple[str, list[int], int]:
    """``'g=3,5,7@2'`` -> ``('g', [3, 5, 7], 2)``."""
    m = _SELECT_RE.match(text.strip())
    if m is None:
        raise ValueError(
            f"bad --select {text!r}; expected BAND=ID,ID,...@APERTURE_INDEX"
        )
    band, ids, aperture = m.groups()
    return band, [int(i) for i in ids.split(",")], int(aperture)


def find_datacube(results_dir: Path) -> Path:
    found = sorted(results_dir.glob(DATACUBE_GLOB))
    if not found:
        raise FileNotFoundError(
            f"no flux datacube ({DATACUBE_GLOB}) in {results_dir}; "
            "re-run photometry with --save_flux_datacube"
        )
    if len(found) > 1:
        raise ValueError(f"several flux datacubes in {results_dir}: {found}")
    return found[0]


def _weighting(cube: dc.BandCube) -> str:
    w = np.array(list(cube.default_weights().values()))
    return "equal" if w.size and np.allclose(w, w[0]) else "broeg"


def band_info(cube: dc.BandCube) -> dict:
    return {
        "target_id": cube.target_id,
        "n_frames": int(cube.fluxes.shape[-1]),
        "aper_radii": [float(r) for r in cube.aper_radii],
        "aper_unit": cube.aper_unit,
        "default_aperture": cube.aperture,
        "default_weighting": _weighting(cube),
        "default_comparisons": [
            {"id": i, "weight": w}
            for i, w in sorted(cube.default_weights().items(), key=lambda kv: -kv[1])
        ],
        "selectable_ids": [int(i) for i in cube.selectable_ids],
    }


def evaluate(cube: dc.BandCube, comparisons: list[int], aperture: int) -> dict:
    """Default and custom light curves of one band, with their metrics."""
    default_flux, _ = dc.default_lightcurve(cube)
    custom_flux, custom_err = dc.rediff(cube, comparisons, aperture)
    return {
        "default": {
            "comparisons": [int(i) for i in cube.default_comparisons],
            "aperture": cube.aperture,
            "aper_radius": float(cube.aper_radii[cube.aperture]),
            "weighting": _weighting(cube),
            "metrics": dc.scatter_metrics(default_flux),
        },
        "custom": {
            "comparisons": list(comparisons),
            "aperture": aperture,
            "aper_radius": float(cube.aper_radii[aperture]),
            "weighting": "equal",
            "metrics": dc.scatter_metrics(custom_flux),
        },
        "_arrays": (cube.bjd_tdb, default_flux, custom_flux, custom_err),
    }


def _label(kind: str, part: dict, unit: str) -> str:
    p2p = part["metrics"]["p2p_ppt"]
    p2p = "n/a" if p2p is None else f"{p2p:.3f} ppt"
    return (
        f"{kind}: cIDs {part['comparisons']}, r={part['aper_radius']:g} {unit}, "
        f"{part['weighting']} weights, p2p {p2p}"
    )


def plot_comparison(results: dict, cubes: dict, path: Path) -> None:
    """One panel per band: default on top, custom offset below, binned overlaid."""
    bands = list(results)
    fig, axes = plt.subplots(
        len(bands), 1, figsize=(9, 3.2 * len(bands)), constrained_layout=True
    )
    for ax, band in zip(np.atleast_1d(axes), bands):
        res = results[band]
        bjd, default_flux, custom_flux, _ = res["_arrays"]
        t0 = _guess_min_time(bjd)
        t = bjd - t0
        both = np.concatenate([default_flux, custom_flux])
        # robust spread, so single-frame outliers do not set offset or limits
        spread = MAD_TO_SIGMA * np.nanmedian(np.abs(both - np.nanmedian(both)))
        offset = OFFSET_SIGMAS * spread if np.isfinite(spread) and spread > 0 else 0.01
        shifted = np.concatenate([default_flux, custom_flux - offset])
        lo, hi = np.nanpercentile(shifted, YLIM_PERCENTILES)
        pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 2.5 * pad)  # headroom for the legend
        unit = cubes[band].aper_unit
        for flux, shift, color, kind in (
            (default_flux, 0.0, "k", "default"),
            (custom_flux, -offset, band_color(band), "custom"),
        ):
            ax.plot(t, flux + shift, ".", c="0.6", alpha=0.35, ms=3)
            bt, bf, be = _binned(t, flux + shift)
            ax.errorbar(
                bt,
                bf,
                yerr=be,
                fmt="o",
                c=color,
                ms=4,
                label=_label(kind, res[kind], unit),
            )
        ax.set_ylabel(f"{band}\nDiff. flux")
        ax.set_xlabel(f"BJD_TDB - {t0}")
        ax.legend(fontsize=7, loc="upper right")
    _savefig(fig, path)


def _band_csv(results_dir: Path, band: str) -> Path:
    hits = [p for p in band_csvs(results_dir) if _band_of(p) == band]
    if len(hits) != 1:
        raise ValueError(
            f"expected one CSV for band {band} in {results_dir}, found {hits}"
        )
    return hits[0]


def apply_selection(results_dir: Path, results: dict, args: argparse.Namespace):
    """Overwrite Flux/Err of each selected band's CSV; regenerate the summary."""
    prepared = []
    for band, res in results.items():
        path = _band_csv(results_dir, band)
        df = read_lightcurve(path)
        bjd, _, flux, err = res["_arrays"]
        csv_bjd = np.asarray(df[TIME_KEY], dtype=float)
        if len(csv_bjd) != len(bjd) or not np.allclose(
            csv_bjd, bjd, rtol=0, atol=BJD_MATCH_ATOL_DAYS
        ):
            raise ValueError(
                f"{path.name} rows no longer match the datacube frames "
                f"({len(csv_bjd)} vs {len(bjd)}); undo the post-process clip first"
            )
        new = df.assign(**{FLUX_KEY: flux, ERR_KEY: err if err is not None else np.nan})
        prepared.append((path, new))
    # validate every band before writing any, so a failure leaves the run untouched
    for path, new in prepared:
        new.to_csv(path, index=False)
        logger.info(f"re-differenced {path.name}")
    csvs = band_csvs(results_dir)
    fig_path = _summary_png(
        results_dir,
        csvs,
        args.target,
        args.inst,
        args.date,
        args.site or None,
        args.confmode or None,
        args.telescope or None,
    )
    subtitle = "; ".join(
        f"{b}: cIDs {r['custom']['comparisons']}, r={r['custom']['aper_radius']:g}"
        for b, r in results.items()
    )
    plot_lightcurves_from_csv(
        csvs,
        fig_path,
        args.target,
        args.inst,
        args.date,
        None,
        0,
        subtitle=f"re-differenced ({subtitle})",
    )
    return [p.name for p, _ in prepared], fig_path.name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m prose.scripts.rediff_lightcurves",
        description="Re-difference a run's light curves from its flux datacube.",
    )
    parser.add_argument("results_dir", type=Path, help="run results directory")
    parser.add_argument(
        "--datacube",
        type=Path,
        default=None,
        help=f"datacube path (default: the one {DATACUBE_GLOB} in results_dir)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="list selectable stars, default comparisons and apertures",
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="BAND=IDS@AP",
        help="comparison IDs and aperture index for a band, e.g. g=3,5,7@4; repeatable",
    )
    parser.add_argument(
        "--preview", type=Path, default=None, help="write comparison PNG"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="overwrite Flux/Err of the selected bands' CSVs in place",
    )
    parser.add_argument("--target", default="", help="target name (for --apply)")
    parser.add_argument("--inst", default="", help="instrument (for --apply)")
    parser.add_argument("--date", default="", help="date, 6 digits (for --apply)")
    parser.add_argument(
        "--site", default="", help="site code for multisite instruments"
    )
    parser.add_argument(
        "--confmode", default="", help="config-mode token (single/multi)"
    )
    parser.add_argument("--telescope", default="", help="telescope id for multisite")
    return parser.parse_args(argv)


def _selections(args: argparse.Namespace, cube: dc.Datacube) -> dict:
    selections = {}
    for text in args.select:
        band, ids, aperture = parse_select(text)
        if band not in cube.bands:
            raise ValueError(f"band {band!r} not in datacube (has {list(cube.bands)})")
        if band in selections:
            raise ValueError(f"band {band!r} selected twice")
        selections[band] = (ids, aperture)
    return selections


def run(args: argparse.Namespace) -> dict:
    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results dir not found: {results_dir}")
    cube = dc.load_datacube(args.datacube or find_datacube(results_dir))
    if args.info:
        return {
            "ok": True,
            "prose_version": cube.prose_version,
            "bands": {b: band_info(c) for b, c in cube.bands.items()},
        }
    if not args.select:
        raise ValueError("give --info or at least one --select")
    if args.apply and (not args.target or not args.inst or not args.date):
        raise ValueError("--apply requires --target, --inst and --date")
    results = {
        band: evaluate(cube.bands[band], ids, aperture)
        for band, (ids, aperture) in _selections(args, cube).items()
    }
    preview, written, summary_png = None, [], None
    if args.apply:
        written, summary_png = apply_selection(results_dir, results, args)
    elif args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(results, cube.bands, args.preview)
        preview = str(args.preview)
    return {
        "ok": True,
        "applied": bool(args.apply),
        "bands": {
            b: {k: v for k, v in r.items() if k != "_arrays"}
            for b, r in results.items()
        },
        "preview": preview,
        "written": written,
        "summary_png": summary_png,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        result = run(parse_args(argv))
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
