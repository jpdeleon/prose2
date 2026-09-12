"""Post-process existing per-band photometry CSV light-curves.

``run_photometry.py`` writes one CSV per band
(``{target}_{inst}[_{site}[_{tel}]]_{band}_{date6}[_full].csv``) carrying at
least ``BJD_TDB``, ``Flux`` and ``Err`` columns. This module applies the same
polynomial sigma-clip as :func:`Fluxes.sigma_clip_flux_poly` to the ``Flux``
column *after* the pipeline has finished, and either

* previews the frames that would be rejected (``--preview <path.png>``), or
* ``--apply``-s the clip: each CSV is overwritten in place so exactly one
  version is picked up by downstream transit fitting, and the run's summary
  ``*_lightcurves.png`` is regenerated from the clipped CSVs.

Rows whose ``Err`` is a finite non-positive value are always rejected. A NaN
``Err`` (the pipeline occasionally writes partial error columns) is kept, since
an outlier filter must not silently drop otherwise-real photometry. When
``BJD_TDB`` is absent or fully non-finite, the row index is used as the time
axis so the clip still runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

from prose.fluxes import flux_sigma_clip_mask
from prose.scripts.run_photometry import (
    MULTISITE_INSTRUMENTS,
    MULTISITE_SITES,
    _binned,
    _savefig,
    band_color,
    build_summary_stem,
    sort_bands_canonical,
)

logger = logging.getLogger("prose_postprocess_lightcurves")

TIME_KEY = "BJD_TDB"
FLUX_KEY = "Flux"
ERR_KEY = "Err"

# band-CSV names end with `_<date6>.csv` or `_<date6>_full.csv`; this excludes
# the run's `*_nearby_stars.csv` (different schema) and any other CSV product.
_BAND_CSV_RE = re.compile(r"^.*_[0-9]{6}(_full)?\.csv$")


def read_lightcurve(path: Path) -> pd.DataFrame:
    """Read a band light-curve CSV, guaranteeing ``Flux``/``Err`` columns."""
    df = pd.read_csv(path)
    if FLUX_KEY not in df.columns:
        raise ValueError(f"{path.name}: missing required column {FLUX_KEY!r}")
    if ERR_KEY not in df.columns:
        df[ERR_KEY] = np.nan
    return df


def _time_axis(df: pd.DataFrame) -> np.ndarray:
    if TIME_KEY in df.columns:
        time = np.asarray(df[TIME_KEY], dtype=float)
        if np.isfinite(time).any():
            return time
    return np.arange(len(df), dtype=float)


def clip_df(
    df: pd.DataFrame,
    sigma: float = 5.0,
    degree: int = 2,
    iterations: int = 5,
) -> tuple[np.ndarray, dict]:
    """Compute the keep-mask and a stats dict for one light-curve."""
    time = _time_axis(df)
    flux = np.asarray(df[FLUX_KEY], dtype=float)
    mask = flux_sigma_clip_mask(
        time, flux, sigma=sigma, degree=degree, iterations=iterations
    )
    err = pd.to_numeric(df[ERR_KEY], errors="coerce")
    mask = mask & ~(np.isfinite(err) & (err <= 0))
    n = len(df)
    n_clipped = int((~mask).sum())
    n_kept = n - n_clipped
    stats = {
        "file": "",
        "n": n,
        "n_clipped": n_clipped,
        "n_kept": n_kept,
        "kept_fraction": round(n_kept / n, 6) if n else 0.0,
        "sigma": sigma,
        "degree": degree,
        "iterations": iterations,
    }
    return mask, stats


def band_csvs(results_dir: Path) -> list[Path]:
    """Band light-curve CSVs in a results dir, sorted by name."""
    files = [p for p in sorted(results_dir.iterdir()) if p.is_file()]
    return [p for p in files if _BAND_CSV_RE.match(p.name)]


def analyze_csvs(
    results_dir: Path,
    csvs: list[Path],
    sigma: float,
    degree: int,
    iterations: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Clip every CSV; return ``{filename: keep-mask}`` and a per-file report."""
    masks: dict[str, np.ndarray] = {}
    report: list[dict] = []
    for path in csvs:
        df = read_lightcurve(path)
        mask, stats = clip_df(df, sigma, degree, iterations)
        stats["file"] = path.name
        masks[path.name] = mask
        report.append(stats)
    return masks, report


def _guess_min_time(time: np.ndarray) -> int:
    finite = time[np.isfinite(time)]
    return int(finite[0]) if len(finite) else 0


def parse_stem(stem: str) -> dict[str, Any]:
    """Recover ``(target, inst, site, tel, band, date)`` from a CSV stem.

    The stem follows ``build_stem``: ``target_inst[_{site}][_{tel}]_{band}_{date}``
    with an optional ``_full`` config-mode suffix. Bands may themselves contain
    underscores, so the site/tel tokens are consumed first and everything that
    remains before the date is the band.
    """
    confmode = ""
    date = ""
    site = None
    tel = None
    if "_" in stem:
        head, tail = stem.rsplit("_", 1)
        if tail == "full" and "_" in head:
            head, tail = head.rsplit("_", 1)
            confmode = "full"
        if len(tail) == 6 and tail.isdigit():
            date = tail
            head = head
        else:
            date = ""
            head = stem
    else:
        head = stem
    tokens = [t for t in head.split("_") if t]
    if len(tokens) < 3:
        target = tokens[0] if tokens else ""
        inst = tokens[1] if len(tokens) > 1 else ""
        band = ""
    else:
        target, inst = tokens[0], tokens[1]
        i = 2
        inst_key = inst.lower()
        if (
            inst_key in MULTISITE_INSTRUMENTS
            and i < len(tokens)
            and tokens[i] in MULTISITE_SITES.get(inst_key, ())
        ):
            site = tokens[i]
            i += 1
        if i < len(tokens) and tokens[i].startswith("tel"):
            tel = tokens[i]
            i += 1
        band = "_".join(tokens[i:])
    return {
        "target": target,
        "inst": inst,
        "site": site,
        "tel": tel,
        "band": band,
        "date": date,
        "confmode": confmode,
    }


def _band_of(path: Path) -> str:
    return parse_stem(path.name.rsplit(".csv", 1)[0])["band"]


def _stem(path: Path) -> str:
    return path.name.rsplit(".csv", 1)[0]


def plot_preview(
    csvs: list[Path],
    masks: dict[str, np.ndarray],
    sigma: float,
    degree: int,
    path: Path,
) -> None:
    """One panel per band: kept scatter in the band's color, rejected frames in red, the trend."""
    fig, axes = plt.subplots(
        len(csvs),
        1,
        figsize=(8, 2.6 * len(csvs)),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, csvs):
        df = read_lightcurve(p)
        mask = masks[p.name]
        time = _time_axis(df)
        flux = np.asarray(df[FLUX_KEY], dtype=float)
        t = time - _guess_min_time(time)
        c = band_color(_band_of(p))
        ax.plot(t[mask], flux[mask], ".", c=c, alpha=0.4, ms=4)
        ax.plot(t[~mask], flux[~mask], "rx", ms=7, mew=1.5)
        keep = mask & np.isfinite(time) & np.isfinite(flux)
        if keep.sum() > degree:
            # Fit against t (time shifted near zero), not raw time (BJD ~
            # 2.46e6): a degree >= 2 polyfit on unshifted BJD is catastrophically
            # ill-conditioned (numpy raises RankWarning) and the high-order
            # terms collapse to ~0, drawing what looks like a straight line
            # regardless of the requested degree. flux_sigma_clip_mask()
            # already fits on centered time for exactly this reason; mirror
            # that here so the preview curve matches the degree it claims.
            coeffs = np.polyfit(t[keep], flux[keep], degree)
            order = np.argsort(t[keep])
            # Fixed black trend line regardless of band color: some bands'
            # color (e.g. ip -> orange) would otherwise be indistinguishable
            # from a colored trend line.
            ax.plot(
                t[keep][order],
                np.polyval(coeffs, t[keep][order]),
                "-",
                c="k",
                lw=1,
                alpha=0.8,
            )
        ax.set_title(
            f"{p.name}  |  rejected {(~mask).sum()}/{len(mask)} "
            f"(sigma={sigma:g}, deg={degree})"
        )
        ax.set_ylabel("Flux")
    axes[-1].set_xlabel("time (JD)")
    _savefig(fig, path)


def plot_lightcurves_from_csv(
    csvs: list[Path],
    path: Path,
    target_name: str,
    instrument: str,
    date: str,
    sigma: float,
    degree: int,
    masks: dict[str, np.ndarray] | None = None,
) -> None:
    """Regenerate the run's summary light-curve figure from CSV lightcurves.

    All present rows of each CSV are plotted; ``masks`` is accepted for
    backward compatibility (the preview path) but no longer applied, since
    post-apply the CSVs already contain only the kept rows.
    """
    path_by_band = {_band_of(p): p for p in csvs}
    csvs = [path_by_band[b] for b in sort_bands_canonical(path_by_band)]
    prepared: list[tuple[np.ndarray, np.ndarray, Path]] = []
    for p in csvs:
        df = read_lightcurve(p)
        keep = masks[p.name] if masks is not None else None
        time = _time_axis(df)
        flux = np.asarray(df[FLUX_KEY], dtype=float)
        if keep is not None:
            keep = keep & np.isfinite(time) & np.isfinite(flux)
        else:
            keep = np.isfinite(time) & np.isfinite(flux)
        if keep.any():
            prepared.append((time[keep], flux[keep], p))
    if not prepared:
        logger.warning("no kept points after clipping; summary plot not written")
        return
    t0 = int(min(_guess_min_time(t) for t, _, _ in prepared))
    fig, axes = plt.subplots(
        len(prepared),
        1,
        figsize=(8, 2.4 * len(prepared)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (time, flux, p) in zip(axes, prepared):
        c = band_color(_band_of(p))
        t = time - t0
        ax.plot(t, flux, ".", c="k", alpha=0.2)
        bt, bf, be = _binned(t, flux)
        ax.errorbar(bt, bf, yerr=be, fmt="o", c=c)
        ax.set_ylabel(f"{_band_of(p)}\nDiff. flux")
    axes[-1].set_xlabel(f"time (JD) - {t0}")

    secax = axes[0].secondary_xaxis(
        location="top",
        functions=(lambda rel: rel + t0, lambda jd: jd - t0),
    )
    secax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda jd, _: Time(jd, format="jd").datetime.strftime("%m-%d\n%H:%M")
        )
    )
    secax.set_xlabel("UTC")
    fig.suptitle(
        f"{target_name} | {instrument} | {date} | post-processed "
        f"(sigma={sigma:g}, deg={degree})"
    )
    _savefig(fig, path)


def _summary_png(
    results_dir: Path,
    csvs: list[Path],
    target: str,
    inst: str,
    date: str,
    site: str | None,
    confmode: str | None,
    telescope: str | None,
) -> Path:
    existing = sorted(results_dir.glob("*_lightcurves.png"))
    if len(existing) == 1:
        return existing[0]
    bands = sort_bands_canonical([_band_of(p) for p in csvs])
    stem = build_summary_stem(
        target,
        inst,
        date,
        bands,
        site=site,
        confmode=confmode,
        telescope=telescope,
    )
    if len(existing) > 1:
        logger.warning(
            "multiple *_lightcurves.png candidates; rebuilding summary from run "
            f"context as {stem}_lightcurves.png"
        )
    return results_dir / f"{stem}_lightcurves.png"


def apply_clip(
    results_dir: Path,
    csvs: list[Path],
    masks: dict[str, np.ndarray],
    target: str,
    inst: str,
    date: str,
    sigma: float,
    degree: int,
    site: str | None = None,
    confmode: str | None = None,
    telescope: str | None = None,
) -> tuple[list[str], str | None]:
    """Overwrite each CSV in place and regenerate the summary light-curve."""
    written = []
    for p in csvs:
        df = read_lightcurve(p)
        mask = masks[p.name]
        dropped = int((~mask).sum())
        df = df.loc[mask].reset_index(drop=True)
        df.to_csv(p, index=False)
        written.append(f"{p.name}: dropped {dropped} rows")
        logger.info(f"post-processed {p.name}: kept {len(df)} / {len(mask)} rows")
    fig_path = _summary_png(
        results_dir, csvs, target, inst, date, site, confmode, telescope
    )
    plot_lightcurves_from_csv(csvs, fig_path, target, inst, date, sigma, degree)
    return written, fig_path.name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m prose.scripts.postprocess_lightcurves",
        description=(
            "Sigma-clip the Flux column of existing per-band photometry CSVs "
            "and optionally overwrite them in place."
        ),
    )
    parser.add_argument("results_dir", type=Path, help="run results directory")
    parser.add_argument(
        "--sigma", type=float, default=5.0, help="clip threshold in sigma"
    )
    parser.add_argument("--degree", type=int, default=2, help="trend polynomial degree")
    parser.add_argument(
        "--iterations", type=int, default=5, help="refit-and-clip passes"
    )
    parser.add_argument(
        "--preview", type=Path, default=None, help="write outlier preview PNG"
    )
    parser.add_argument("--apply", action="store_true", help="overwrite CSVs in place")
    parser.add_argument("--target", default="", help="target name")
    parser.add_argument("--inst", default="", help="instrument")
    parser.add_argument("--date", default="", help="date (6 digits)")
    parser.add_argument(
        "--site", default="", help="site code for multisite instruments"
    )
    parser.add_argument(
        "--confmode", default="", help="config-mode token (single/multi)"
    )
    parser.add_argument("--telescope", default="", help="telescope id for multisite")
    return parser.parse_args(argv)


def _write_preview(path: Path, csvs, masks, sigma, degree) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_preview(csvs, masks, sigma, degree, path)


def run(args: argparse.Namespace) -> dict:
    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results dir not found: {results_dir}")
    if args.apply and (not args.target or not args.inst or not args.date):
        raise ValueError("--apply requires --target, --inst and --date")
    csvs = band_csvs(results_dir)
    if not csvs:
        raise FileNotFoundError(f"no band light-curve CSVs found in {results_dir}")
    masks, report = analyze_csvs(
        results_dir, csvs, args.sigma, args.degree, args.iterations
    )
    applied = False
    preview = None
    written: list[str] = []
    summary_png = None
    if args.preview is not None:
        _write_preview(args.preview, csvs, masks, args.sigma, args.degree)
        preview = str(args.preview)
    elif args.apply:
        written, summary_png = apply_clip(
            results_dir,
            csvs,
            masks,
            args.target,
            args.inst,
            args.date,
            args.sigma,
            args.degree,
            site=args.site or None,
            confmode=args.confmode or None,
            telescope=args.telescope or None,
        )
        applied = True
    return {
        "ok": True,
        "results_dir": str(results_dir),
        "sigma": args.sigma,
        "degree": args.degree,
        "iterations": args.iterations,
        "applied": applied,
        "n_files": len(report),
        "files": report,
        "preview": preview,
        "summary_png": summary_png,
        "written": written,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
