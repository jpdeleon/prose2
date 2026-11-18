"""Compare a candidate prose install with the previous release on real data.

Runs ``run_photometry`` with two interpreters (``--baseline`` and
``--candidate``) over the datasets in a TOML file and writes a markdown report
and a JSON record. See ``benchmarks/README.md`` for the release procedure.

Two kinds of dataset:

* ``fixed``: one aperture and a fixed comparison ensemble, so differences come
  from the photometry, not from selection. Checked pass/fail
  (:func:`compare_fixed`); any failure makes the command exit 1.
* ``auto``: default selection, as users run it. Reported side by side;
  suspicious bands are flagged for review (:func:`review_flags`), not failed,
  because releases change selection on purpose.

Each run gets its own HOME. Its ``~/.prose`` holds the ``*.telescope`` files
from ``--telescopes`` (default: this checkout's ``data/``), the same files for
both installs, so differences come from the engine, not from whatever telescope
files a shared ``~/.prose`` happens to hold. Telescope changes between releases
are reviewed as a diff of ``data/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prose.precision import ptp_scatter
from prose.utils import index_binning

KINDS = ("fixed", "auto")
BIN_SIZE_DAYS = 10 / 60 / 24  # 10-minute bins
# Fixed check. BJD_TDB allows prose2 3.7's correction at the target's position
# on the observing night (TOI-3884 moves 4.4 arcsec by 2023, about 10 ms).
FLUX_ATOL = 1e-3
BJD_ATOL_S = 0.05
MATCH_TOLERANCE_S = 1e-3
MIN_FRAME_FRACTION = 0.99
# Auto review: flag a band whose scatter grows by more than this fraction.
SCATTER_REVIEW_FRACTION = 0.2
TARGET_MARKERS = ("Target not found", "defaulting to source")
RUN_TIMEOUT_S = 3 * 3600
# ~/.prose files copied into each private HOME so runs never download them.
HOME_SEED_FILES = ("centroid.h5",)


@dataclass(frozen=True)
class Dataset:
    name: str
    kind: str
    data_dir: Path
    target_name: str
    target_coord: str
    bands: tuple[str, ...]
    args: tuple[str, ...] = ()
    calib_cache: Path | None = None


@dataclass(frozen=True)
class RunResult:
    directory: Path
    returncode: int
    seconds: float
    log: str


@dataclass(frozen=True)
class BandResult:
    dataset: str
    kind: str
    band: str
    baseline: dict | None
    candidate: dict | None
    failures: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def load_datasets(path: Path) -> list[Dataset]:
    """Read and validate the ``datasets`` list of a YAML file."""
    document = yaml.safe_load(Path(path).read_text()) or {}
    entries = document.get("datasets") if isinstance(document, dict) else None
    if not entries or not isinstance(entries, list):
        raise ValueError(f"{path}: no 'datasets' list")
    datasets, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: dataset must be a mapping: {entry!r}")
        missing = {"name", "kind", "data_dir", "target_name", "target_coord", "bands"}
        missing -= entry.keys()
        if missing:
            raise ValueError(f"{path}: dataset missing {sorted(missing)}: {entry}")
        if entry["kind"] not in KINDS:
            raise ValueError(f"{entry['name']}: kind must be one of {KINDS}")
        if entry["name"] in names:
            raise ValueError(f"duplicate dataset name {entry['name']!r}")
        if not entry["bands"]:
            raise ValueError(f"{entry['name']}: bands must not be empty")
        coord = entry["target_coord"]
        if not isinstance(coord, str) or len(coord.split()) != 2:
            raise ValueError(f"{entry['name']}: target_coord must be 'RA Dec'")
        names.add(entry["name"])
        calib = entry.get("calib_cache")
        datasets.append(
            Dataset(
                name=entry["name"],
                kind=entry["kind"],
                data_dir=Path(entry["data_dir"]).expanduser(),
                target_name=entry["target_name"],
                target_coord=entry["target_coord"],
                bands=tuple(str(b) for b in entry["bands"]),
                args=tuple(str(a) for a in entry.get("args", ())),
                calib_cache=Path(calib).expanduser() if calib else None,
            )
        )
    return datasets


def coord_args(target_coord: str) -> list[str]:
    """``--target_coord`` values; a negative declination gets a leading space.

    argparse can read ``-44.2`` as an option flag; prose strips the space.
    """
    ra, dec = target_coord.split()
    return [ra, f" {dec}" if dec.startswith("-") else dec]


def build_command(
    python: str, dataset: Dataset, results_dir: Path, workers: int
) -> list[str]:
    command = [
        python, "-I", "-m", "prose.scripts.run_photometry",
        "--target_name", dataset.target_name,
        "--data_dir", str(dataset.data_dir),
        "--results_dir", str(results_dir),
        "--bands", *dataset.bands,
        "--target_coord", *coord_args(dataset.target_coord),
        *dataset.args,
        "--verbose", "--overwrite", "--plots", "none", "--workers", str(workers),
    ]  # fmt: skip
    if dataset.calib_cache is not None:
        command += ["--calib_dir", str(results_dir / "calibrated")]
    return command


def telescope_files(directory: Path) -> list[Path]:
    """The ``*.telescope`` files to install in every private HOME."""
    files = sorted(Path(directory).glob("*.telescope"))
    if not files:
        raise ValueError(f"no *.telescope files in {directory}")
    return files


def _private_home(directory: Path, seed: Path, telescopes: list[Path]) -> Path:
    home = directory / "home"
    (home / ".prose").mkdir(parents=True, exist_ok=True)
    for name in HOME_SEED_FILES:
        if (seed / name).is_file():
            shutil.copy2(seed / name, home / ".prose" / name)
    for path in telescopes:
        shutil.copy2(path, home / ".prose" / path.name)
    return home


def run_photometry(
    python: str,
    dataset: Dataset,
    directory: Path,
    workers: int,
    telescopes: list[Path],
) -> RunResult:
    """Run one install on one dataset in ``directory`` with a private HOME."""
    directory.mkdir(parents=True, exist_ok=True)
    if dataset.calib_cache is not None:
        # The engine may repair a calibration cache in place; work on a copy.
        shutil.copytree(
            dataset.calib_cache, directory / "calibrated", dirs_exist_ok=True
        )
    home = _private_home(directory, Path.home() / ".prose", telescopes)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PROSE_CONFIG")
    }
    env.update(HOME=str(home), PYTHONNOUSERSITE="1", TMPDIR=str(directory))
    command = build_command(python, dataset, directory, workers)
    log_path = directory / "run.log"
    start = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        try:
            returncode = subprocess.run(
                command, cwd=directory, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S,
            ).returncode  # fmt: skip
        except subprocess.TimeoutExpired:
            returncode = -1
            log.write(f"\nrelease_benchmark: timed out after {RUN_TIMEOUT_S} s\n")
    return RunResult(
        directory, returncode, time.monotonic() - start, log_path.read_text()
    )


def band_outputs(directory: Path, band: str) -> tuple[pd.DataFrame | None, dict]:
    """The band's light curve and its aperture/comparison metadata, if written."""
    csvs = [
        p for p in directory.glob("*.csv")
        if f"_{band}_" in p.name and not p.name.endswith("_nearby_stars.csv")
    ]  # fmt: skip
    if len(csvs) != 1:
        return None, {}
    meta = {}
    for npz in directory.glob("*.npz"):
        with np.load(npz, allow_pickle=False) as archive:
            key = f"{band}__aper_radii"
            if key in archive.files:
                radii = archive[key]
                index = (
                    int(archive[f"{band}__aperture"])
                    if f"{band}__aperture" in archive.files
                    else 0
                )
                meta["aperture_pix"] = float(radii[min(index, len(radii) - 1)])
                if f"{band}__comparisons" in archive.files:
                    meta["comparisons"] = [
                        int(c) for c in archive[f"{band}__comparisons"]
                    ]
                break
    return pd.read_csv(csvs[0]), meta


def binned_scatter(time_days: np.ndarray, flux: np.ndarray) -> float:
    """Standard deviation of 10-minute bin means (bins with 2+ frames)."""
    means = [
        flux[i].mean() for i in index_binning(time_days, BIN_SIZE_DAYS) if len(i) > 1
    ]
    return float(np.std(means)) if len(means) > 1 else float("nan")


def band_summary(curve: pd.DataFrame, meta: dict, seconds: float) -> dict:
    flux = curve["Flux"].to_numpy(dtype=float)
    time_days = curve["BJD_TDB"].to_numpy(dtype=float)
    return {
        "frames": len(curve),
        "aperture_pix": meta.get("aperture_pix"),
        "comparisons": meta.get("comparisons"),
        "ptp_ppm": float(ptp_scatter(flux)) * 1e6,
        "binned_ppm": binned_scatter(time_days, flux) * 1e6,
        "seconds": round(seconds, 1),
    }


def compare_fixed(baseline: pd.DataFrame, candidate: pd.DataFrame) -> list[str]:
    """Failures when a fixed-aperture, fixed-comparison light curve changed."""
    failures = []
    removed = set(baseline.columns) - set(candidate.columns)
    if removed:
        failures.append(f"columns removed: {sorted(removed)}")
    columns = ["BJD_TDB", "GJD_UTC", "Flux", "Err"]
    for label, curve in (("baseline", baseline), ("candidate", candidate)):
        if not set(columns) <= set(curve.columns):
            return failures + [f"{label} lacks {columns}"]
        if not np.isfinite(curve[columns].to_numpy(dtype=float)).all():
            failures.append(f"{label} has non-finite values")
        if not (np.diff(curve["GJD_UTC"]) > 0).all():
            failures.append(f"{label} times are not increasing")
        if not (curve["Err"] > 0).all():
            failures.append(f"{label} has non-positive errors")
    if len(candidate) < MIN_FRAME_FRACTION * len(baseline):
        failures.append(f"frames {len(baseline)} -> {len(candidate)}")
    # Both write exposure midpoints: match frames on time, never by row number.
    matched = pd.merge_asof(
        candidate.sort_values("GJD_UTC"), baseline.sort_values("GJD_UTC"),
        on="GJD_UTC", tolerance=MATCH_TOLERANCE_S / 86400, direction="nearest",
        suffixes=("_new", "_old"),
    )  # fmt: skip
    unmatched = int(matched["Flux_old"].isna().sum())
    if unmatched:
        failures.append(f"{unmatched} frames have no baseline counterpart")
    matched = matched.dropna(subset=["Flux_old"])
    if not matched["BJD_TDB_old"].is_unique:
        failures.append("two frames matched one baseline frame")
    dbjd = np.abs(matched["BJD_TDB_new"] - matched["BJD_TDB_old"]).max() * 86400
    if dbjd > BJD_ATOL_S:
        failures.append(f"BJD_TDB differs by up to {dbjd * 1e3:.1f} ms")
    dflux = np.abs(matched["Flux_new"] - matched["Flux_old"]).max()
    if dflux > FLUX_ATOL:
        failures.append(f"flux differs by up to {dflux:.6f}")
    return failures


def review_flags(baseline: dict | None, candidate: dict | None) -> list[str]:
    """Reasons an automatic-selection band needs a human look."""
    if candidate is None:
        return ["candidate produced no light curve"]
    if baseline is None:
        return []
    flags = []
    if candidate["frames"] < MIN_FRAME_FRACTION * baseline["frames"]:
        flags.append(f"frames {baseline['frames']} -> {candidate['frames']}")
    for key in ("ptp_ppm", "binned_ppm"):
        old, new = baseline[key], candidate[key]
        if np.isfinite(old) and old > 0 and new > (1 + SCATTER_REVIEW_FRACTION) * old:
            flags.append(f"{key} {old:.0f} -> {new:.0f}")
    return flags


def _target_problems(log: str) -> list[str]:
    return [f"log: {marker!r}" for marker in TARGET_MARKERS if marker in log]


def evaluate(dataset: Dataset, runs: dict[str, RunResult]) -> list[BandResult]:
    results = []
    problems = {label: _target_problems(run.log) for label, run in runs.items()}
    for band in dataset.bands:
        curves, summaries = {}, {}
        for label, run in runs.items():
            curve, meta = band_outputs(run.directory, band)
            curves[label] = curve
            summaries[label] = (
                None if curve is None else band_summary(curve, meta, run.seconds)
            )
        notes = [
            f"{label} exit {run.returncode}"
            for label, run in runs.items()
            if run.returncode
        ]
        added = set()
        if curves["baseline"] is not None and curves["candidate"] is not None:
            added = set(curves["candidate"].columns) - set(curves["baseline"].columns)
        if added:
            notes.append(f"new columns: {sorted(added)}")
        failures, flags = [], []
        target = [f"{label} {p}" for label, ps in problems.items() for p in ps]
        if dataset.kind == "fixed":
            missing = [label for label, curve in curves.items() if curve is None]
            if missing:
                failures.append(f"no light curve from {', '.join(missing)}")
            else:
                failures += compare_fixed(curves["baseline"], curves["candidate"])
            failures += target
        else:
            flags += (
                review_flags(summaries["baseline"], summaries["candidate"]) + target
            )
        results.append(
            BandResult(dataset.name, dataset.kind, band, summaries["baseline"],
                       summaries["candidate"], tuple(failures), tuple(flags), tuple(notes))
        )  # fmt: skip
    return results


def _cell(summary: dict | None, key: str) -> str:
    if summary is None:
        return "—"
    value = summary.get(key)
    if value is None:
        return "?"
    if isinstance(value, float):
        return (
            f"{value:.0f}"
            if key.endswith("ppm") or key == "seconds"
            else f"{value:.2f}"
        )
    if isinstance(value, list):
        return "(" + ", ".join(map(str, value)) + ")"
    return str(value)


def render_report(
    results: list[BandResult],
    baseline: str,
    candidate: str,
    skipped: tuple[str, ...] = (),
) -> str:
    lines = [
        "# prose release benchmark", "",
        f"- baseline: `{baseline}`", f"- candidate: `{candidate}`", "",
        "Each cell is baseline → candidate. ptp: robust point-to-point scatter; "
        "binned: std of 10-min bin means.", "",
        "| Dataset | Kind | Band | Frames | Aperture (px) | Comparisons | ptp (ppm) "
        "| 10-min binned (ppm) | Wall time (s) | Result |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    keys = ("frames", "aperture_pix", "comparisons", "ptp_ppm", "binned_ppm", "seconds")
    for r in results:
        pairs = [f"{_cell(r.baseline, k)} → {_cell(r.candidate, k)}" for k in keys]
        if r.failures:
            verdict = "**FAIL**: " + "; ".join(r.failures)
        elif r.flags:
            verdict = "review: " + "; ".join(r.flags)
        else:
            verdict = "ok"
        if r.notes:
            verdict += " (" + "; ".join(r.notes) + ")"
        lines.append(
            f"| {r.dataset} | {r.kind} | {r.band} | "
            + " | ".join(pairs)
            + f" | {verdict} |"
        )
    failed = sum(bool(r.failures) for r in results)
    flagged = sum(bool(r.flags) for r in results)
    lines += [
        "",
        f"{failed} fixed band(s) failed; {flagged} automatic band(s) flagged for review.",
    ]
    if skipped:
        lines += ["", "Skipped (raw data not found): " + ", ".join(skipped)]
    return "\n".join(lines) + "\n"


def _parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--baseline", required=True, help="python of the previous release's environment"
    )
    ap.add_argument(
        "--candidate", required=True, help="python of the candidate environment"
    )
    ap.add_argument("--datasets", type=Path, default=Path("benchmarks/datasets.yaml"))
    ap.add_argument(
        "--out", type=Path, required=True, help="output directory (must not exist)"
    )
    ap.add_argument("--only", nargs="+", help="dataset names to run (default: all)")
    ap.add_argument(
        "--kind", choices=KINDS, help="run only fixed or only auto datasets"
    )
    ap.add_argument("--workers", type=int, default=2, help="workers per photometry run")
    ap.add_argument(
        "--telescopes",
        type=Path,
        default=Path("data"),
        help="directory of *.telescope files installed for both runs",
    )
    return ap.parse_args(argv)


def select_datasets(datasets, only=None, kind=None) -> list[Dataset]:
    if only:
        unknown = set(only) - {d.name for d in datasets}
        if unknown:
            raise SystemExit(f"unknown datasets: {sorted(unknown)}")
        datasets = [d for d in datasets if d.name in only]
    return [d for d in datasets if kind is None or d.kind == kind]


def main(argv=None) -> int:
    args = _parse_args(argv)
    datasets = select_datasets(load_datasets(args.datasets), args.only, args.kind)
    for python in (args.baseline, args.candidate):
        if not Path(python).is_file():
            raise SystemExit(f"interpreter not found: {python}")
    telescopes = telescope_files(args.telescopes)
    args.out.mkdir(parents=True, exist_ok=False)
    results, skipped = [], []
    for dataset in datasets:
        if not dataset.data_dir.is_dir():
            print(f"skip {dataset.name}: {dataset.data_dir} not found", file=sys.stderr)
            skipped.append(dataset.name)
            continue
        runs = {}
        for label, python in (
            ("baseline", args.baseline),
            ("candidate", args.candidate),
        ):
            print(f"{dataset.name}: {label}", file=sys.stderr, flush=True)
            runs[label] = run_photometry(
                python,
                dataset,
                args.out / dataset.name / label,
                args.workers,
                telescopes,
            )
        results += evaluate(dataset, runs)
    report = render_report(results, args.baseline, args.candidate, tuple(skipped))
    (args.out / "report.md").write_text(report)
    (args.out / "report.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    print(report)
    return 1 if any(r.failures for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
