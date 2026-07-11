#!/usr/bin/env python3
"""Benchmark prose2 and AFPhot source detection on MuSCAT3 FITS frames.

This script runs the algorithms used by the production pipelines:

* prose2 ``PointSourceDetection``: global thresholding, scikit-image connected
  regions, weighted centroids, peak sorting, and minimum-separation cleaning.
  The small implementation below intentionally mirrors
  ``prose/blocks/detection.py`` so the benchmark does not import the complete
  prose/Photutils runtime.
* AFPhot ``starfind_centroid``: the production compiled executable with the
  MuSCAT3 CCD and detector parameter values.

Catalog agreement is not ground-truth completeness.  The output distinguishes
raw catalogs from the top-N catalogs actually retained by prose2.  It writes a
per-frame CSV, summary JSON, and optional matched-coordinate CSV.

Example
-------
UV_CACHE_DIR=/tmp/prose2-uv-cache uv run python \
  scripts/benchmark_detection_muscat3.py \
  '/data/MuSCAT3/260224/*-e91.fits' --limit 20 --output benchmark_detection

Recorded smoke benchmark
------------------------
Run on 2026-07-11 using the four MuSCAT3 BANZAI ``*-e91.fits`` frames from
``/data/MuSCAT3/220422`` (one frame from each camera), the production defaults,
a 5-pixel matching radius, and top-N = 10:

* all 4 frames completed successfully;
* median prose2 runtime: 0.434 s per frame;
* median AFPhot runtime: 0.270 s per frame;
* median raw detections: 43.5 for prose2 and 24.5 for AFPhot;
* median prose2 raw match fraction: 0.534;
* median AFPhot raw match fraction: 1.000;
* median raw matched-coordinate separation: 0.321 pixels; and
* median top-10 match fraction: 0.700 for both catalogs, with a median
  matched-coordinate separation of 0.060 pixels.

These figures measure agreement between detectors, not completeness or purity
against a truth catalog. Runtime includes FITS loading for both methods.

Recorded heavily defocused V1298 Tau benchmark
------------------------------------------------
The local MuSCAT observation database/obslog identified the 2023-11-15
MuSCAT3 V1298 Tau sequence as heavily defocused. The benchmark used the first
five V1298 Tau frames from each of the four cameras (20 BANZAI ``e91`` frames
in total). Their ``L1FWHM`` values span 5.53--7.46 arcsec, with a median of
6.91 arcsec, or 25.6 pixels at 0.27 arcsec/pixel. Using the same production
defaults, 5-pixel matching radius, and top-N = 10:

* all 20 frames completed successfully;
* median prose2 runtime: 0.572 s per frame;
* median AFPhot runtime: 0.556 s per frame;
* median raw detections: 12.5 for prose2 and 10.0 for AFPhot;
* median prose2 raw match fraction: 0.802;
* median AFPhot raw match fraction: 1.000;
* median raw matched-coordinate separation: 0.100 pixels; and
* median top-10 match fraction: 0.950 for prose2 and 1.000 for AFPhot, with a
  median matched-coordinate separation of 0.079 pixels.

Thus AFPhot's detections were a subset of prose2's within the chosen 5-pixel
radius on this sample, while prose2 reported a few additional candidates.
This remains a detector-agreement result, not a truth-catalog measurement of
which additional candidates are real stars.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops


AFPHOT_BINARY = Path("/ut3/muscat/reduction_afphot/tools/afphot/bin/starfind_centroid")
AFPHOT_PARAMS = Path(
    "/ut3/muscat/reduction_afphot/tools/params/param_muscat3/"
    "param-starfind_centroid.par"
)


@dataclass
class Catalog:
    coords: np.ndarray
    peaks: np.ndarray


@dataclass
class FrameResult:
    frame: str
    prose_seconds: float
    afphot_seconds: float
    prose_raw_count: int
    afphot_raw_count: int
    prose_kept_count: int
    afphot_top_count: int
    raw_matches: int
    raw_prose_match_fraction: float
    raw_afphot_match_fraction: float
    raw_median_separation_px: float
    top_matches: int
    top_prose_match_fraction: float
    top_afphot_match_fraction: float
    top_median_separation_px: float
    error: str = ""


def prose_detect(
    path: Path,
    threshold: float,
    min_area: float,
    minor_length: float,
    min_separation: float | None,
) -> Catalog:
    """Mirror PointSourceDetection.regions/from_region/clean before N slicing."""
    data = np.asarray(fits.getdata(path), dtype=float)
    flat = data.ravel()
    median = np.nanmedian(flat)
    # This deliberately preserves prose2's unusual one-standard-deviation
    # sample selection rather than replacing it with a conventional clip.
    sample = flat[np.abs(flat - median) < np.nanstd(flat)]
    absolute_threshold = threshold * np.nanstd(sample) + median
    regions = regionprops(label(data > absolute_threshold), data)
    regions = [
        region
        for region in regions
        if region.area >= min_area and region.axis_major_length >= minor_length
    ]

    coords = np.asarray(
        [region.centroid_weighted[::-1] for region in regions], dtype=float
    ).reshape((-1, 2))
    peaks = np.asarray([region.intensity_max for region in regions], dtype=float)
    order = np.argsort(peaks)[::-1]
    coords, peaks = coords[order], peaks[order]

    if min_separation and len(coords):
        keep = np.ones(len(coords), dtype=bool)
        for i in range(len(coords)):
            if not keep[i]:
                continue
            distances = np.linalg.norm(coords - coords[i], axis=1)
            keep[(np.arange(len(coords)) > i) & (distances < min_separation)] = False
        coords, peaks = coords[keep], peaks[keep]
    return Catalog(coords, peaks)


def afphot_detect(
    path: Path,
    binary: Path,
    params: Path,
    gain: float,
    read_noise: float,
    adu_low: float,
    adu_high: float,
) -> Catalog:
    command = [
        str(binary),
        "-frame",
        str(path),
        "-gain",
        str(gain),
        "-read_out_noise",
        str(read_noise),
        "-ADU_range",
        str(adu_low),
        str(adu_high),
        "-file",
        str(params),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    coords: list[tuple[float, float]] = []
    peaks: list[float] = []
    for line in completed.stdout.splitlines():
        if not re.match(r"^\s*\d+\s", line):
            continue
        fields = line.split()
        if len(fields) < 7:
            continue
        # AFPhot columns: ID, centroid x/y, integer-position x/y, flux, peak.
        coords.append((float(fields[1]), float(fields[2])))
        peaks.append(float(fields[6]))
    return Catalog(np.asarray(coords).reshape((-1, 2)), np.asarray(peaks))


def unique_matches(
    left: np.ndarray, right: np.ndarray, radius: float
) -> list[tuple[int, int, float]]:
    """Greedily select unique shortest pairs within radius."""
    if not len(left) or not len(right):
        return []
    tree = cKDTree(right)
    candidates: list[tuple[float, int, int]] = []
    for left_index, neighbor_indices in enumerate(tree.query_ball_point(left, radius)):
        for right_index in neighbor_indices:
            distance = float(np.linalg.norm(left[left_index] - right[right_index]))
            candidates.append((distance, left_index, right_index))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, left_index, right_index in sorted(candidates):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_index, right_index, distance))
    return matches


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def median_distance(matches: list[tuple[int, int, float]]) -> float:
    return float(np.median([match[2] for match in matches])) if matches else math.nan


def resolve_inputs(patterns: list[str], limit: int | None) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            paths.update(Path(item).resolve() for item in expanded)
        elif Path(pattern).is_file():
            paths.add(Path(pattern).resolve())
    ordered = sorted(paths)
    return ordered[:limit] if limit is not None else ordered


def summarize(results: list[FrameResult], arguments: argparse.Namespace) -> dict:
    successful = [result for result in results if not result.error]

    def values(field: str) -> np.ndarray:
        return np.asarray(
            [getattr(result, field) for result in successful], dtype=float
        )

    def median(field: str) -> float | None:
        array = values(field)
        finite = array[np.isfinite(array)]
        return float(np.median(finite)) if len(finite) else None

    return {
        "description": "Catalog agreement is not ground-truth completeness.",
        "frames_requested": len(results),
        "frames_successful": len(successful),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "median": {
            "prose_seconds": median("prose_seconds"),
            "afphot_seconds": median("afphot_seconds"),
            "prose_raw_count": median("prose_raw_count"),
            "afphot_raw_count": median("afphot_raw_count"),
            "raw_prose_match_fraction": median("raw_prose_match_fraction"),
            "raw_afphot_match_fraction": median("raw_afphot_match_fraction"),
            "raw_separation_px": median("raw_median_separation_px"),
            "top_prose_match_fraction": median("top_prose_match_fraction"),
            "top_afphot_match_fraction": median("top_afphot_match_fraction"),
            "top_separation_px": median("top_median_separation_px"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="FITS paths or glob patterns")
    parser.add_argument("--limit", type=int, help="use only the first N sorted frames")
    parser.add_argument("--output", type=Path, default=Path("benchmark_detection"))
    parser.add_argument("--match-radius", type=float, default=5.0)
    parser.add_argument("--prose-threshold", type=float, default=4.0)
    parser.add_argument("--prose-min-area", type=float, default=10.0)
    parser.add_argument("--prose-minor-length", type=float, default=0.0)
    parser.add_argument("--prose-min-separation", type=float, default=10.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--afphot-binary", type=Path, default=AFPHOT_BINARY)
    parser.add_argument("--afphot-params", type=Path, default=AFPHOT_PARAMS)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--read-noise", type=float, default=12.0)
    parser.add_argument("--adu-low", type=float, default=-1000.0)
    parser.add_argument("--adu-high", type=float, default=130000.0)
    parser.add_argument(
        "--write-matches", action="store_true", help="write matched coordinates"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = resolve_inputs(args.inputs, args.limit)
    if not frames:
        raise SystemExit("no input FITS frames matched")
    if not args.afphot_binary.is_file() or not args.afphot_params.is_file():
        raise SystemExit("AFPhot binary or parameter file does not exist")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[FrameResult] = []
    match_rows: list[dict] = []
    for index, frame in enumerate(frames, start=1):
        print(f"[{index}/{len(frames)}] {frame}", flush=True)
        try:
            started = time.perf_counter()
            prose = prose_detect(
                frame,
                args.prose_threshold,
                args.prose_min_area,
                args.prose_minor_length,
                args.prose_min_separation,
            )
            prose_seconds = time.perf_counter() - started

            started = time.perf_counter()
            afphot = afphot_detect(
                frame,
                args.afphot_binary,
                args.afphot_params,
                args.gain,
                args.read_noise,
                args.adu_low,
                args.adu_high,
            )
            afphot_seconds = time.perf_counter() - started

            raw = unique_matches(prose.coords, afphot.coords, args.match_radius)
            prose_top = prose.coords[: args.top_n]
            afphot_top = afphot.coords[: args.top_n]
            top = unique_matches(prose_top, afphot_top, args.match_radius)
            results.append(
                FrameResult(
                    str(frame),
                    prose_seconds,
                    afphot_seconds,
                    len(prose.coords),
                    len(afphot.coords),
                    len(prose_top),
                    len(afphot_top),
                    len(raw),
                    fraction(len(raw), len(prose.coords)),
                    fraction(len(raw), len(afphot.coords)),
                    median_distance(raw),
                    len(top),
                    fraction(len(top), len(prose_top)),
                    fraction(len(top), len(afphot_top)),
                    median_distance(top),
                )
            )
            if args.write_matches:
                for prose_index, afphot_index, distance in raw:
                    match_rows.append(
                        {
                            "frame": str(frame),
                            "prose_index": prose_index,
                            "afphot_index": afphot_index,
                            "prose_x": prose.coords[prose_index, 0],
                            "prose_y": prose.coords[prose_index, 1],
                            "afphot_x": afphot.coords[afphot_index, 0],
                            "afphot_y": afphot.coords[afphot_index, 1],
                            "separation_px": distance,
                        }
                    )
        except Exception as exc:  # continue across corrupt/problematic frames
            results.append(
                FrameResult(
                    str(frame),
                    math.nan,
                    math.nan,
                    0,
                    0,
                    0,
                    0,
                    0,
                    math.nan,
                    math.nan,
                    math.nan,
                    0,
                    math.nan,
                    math.nan,
                    math.nan,
                    f"{type(exc).__name__}: {exc}",
                )
            )

    result_path = args.output.with_suffix(".csv")
    with result_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    summary = summarize(results, args)
    summary_path = args.output.with_name(args.output.name + "_summary").with_suffix(
        ".json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.write_matches:
        match_path = args.output.with_name(args.output.name + "_matches").with_suffix(
            ".csv"
        )
        with match_path.open("w", newline="") as stream:
            fields = list(match_rows[0]) if match_rows else ["frame"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(match_rows)

    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"wrote {result_path} and {summary_path}")
    return 0 if summary["frames_successful"] == len(frames) else 1


if __name__ == "__main__":
    raise SystemExit(main())
