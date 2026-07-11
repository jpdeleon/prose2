#!/usr/bin/env python3
"""Benchmark centroid methods on matched sources in MuSCAT3 FITS frames.

The benchmark starts from prose2 production ``PointSourceDetection`` weighted
centroids, refines the same positions with Photutils ``centroid_quadratic``,
``centroid_com``, and ``centroid_2dg``, and matches them to positions emitted
by AFPhot's production ``starfind_centroid`` executable.

Photutils timing excludes FITS loading and source detection. AFPhot timing
necessarily includes FITS loading and detection because its executable does
not expose centroid refinement as a separate operation. Pairwise disagreement
is not astrometric truth; Gaussian2D is reported as a slower shape-model proxy.

This benchmark motivated replacing the former production-only quadratic step
with ``AdaptiveCentroid``. Quadratic remains included as the compact-PSF branch
and for historical comparison.

Example
-------
UV_CACHE_DIR=/tmp/prose2-uv-cache uv run python \
  scripts/benchmark_centroiding_muscat3.py \
  '/data/MuSCAT3/231115/ogg2m001-*-20231115-*-e91.fits' \
  --limit 20 --output centroid_v1298tau

Recorded benchmarks
-------------------
The focused sample used one 2022-04-22 BANZAI frame from each MuSCAT3 camera:
4 frames, 33 common sources, and median ``L1FWHM`` 1.63 arcsec. Median timing
per frame was 0.00488 s for quadratic, 0.00169 s for center-of-mass, 0.278 s
for Gaussian2D, and 0.318 s for the complete AFPhot detection/centroid process.
All Photutils results passed prose2's 10.5-pixel displacement limit. Median
disagreement with AFPhot was 0.214 pixels for quadratic, 0.173 pixels for
center-of-mass, and 0.133 pixels for Gaussian2D. Quadratic versus Gaussian2D
disagreement was 0.103 pixels, although quadratic had a long tail (maximum
8.13 pixels).

The heavily defocused sample used the first five 2023-11-15 V1298 Tau frames
from each camera: 20 frames, 192 common sources, and median ``L1FWHM`` 6.91
arcsec (25.6 pixels at 0.27 arcsec/pixel). Median timing was 0.00393 s for
quadratic, 0.00119 s for center-of-mass, 0.432 s for Gaussian2D, and 0.569 s
for complete AFPhot processing. Accepted fractions were 0.927 for quadratic,
1.000 for center-of-mass, and 0.776 for Gaussian2D. Median disagreement with
AFPhot was 7.73 pixels for quadratic, 0.381 pixels for center-of-mass, and 2.04
pixels for Gaussian2D. Quadratic versus Gaussian2D disagreement was 6.16
pixels.

These real-image results show that prose2's production quadratic centroid is
not shape-robust on the severely defocused V1298 Tau profiles. Center-of-mass
is much closer to AFPhot there and is faster, but agreement with AFPhot is not
ground-truth accuracy. Injection tests or astrometric/temporal residuals are
needed before declaring any method absolutely more accurate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from photutils.centroids import (
    centroid_2dg,
    centroid_com,
    centroid_quadratic,
    centroid_sources,
)

from benchmark_detection_muscat3 import (
    AFPHOT_BINARY,
    AFPHOT_PARAMS,
    afphot_detect,
    prose_detect,
    resolve_inputs,
    unique_matches,
)


METHODS = {
    "quadratic": centroid_quadratic,
    "com": centroid_com,
    "gaussian2d": centroid_2dg,
}


def distribution(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"median": None, "p90": None, "p95": None, "max": None}
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def refine(
    data: np.ndarray,
    initial: np.ndarray,
    function,
    cutout: int,
    limit: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not len(initial):
        return initial.copy(), np.empty(0, dtype=bool), 0.0
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        measured = np.asarray(
            centroid_sources(
                data,
                initial[:, 0],
                initial[:, 1],
                box_size=cutout,
                centroid_func=function,
            )
        ).T
    elapsed = time.perf_counter() - started
    shifts = np.linalg.norm(measured - initial, axis=1)
    accepted = np.all(np.isfinite(measured), axis=1) & (shifts < limit)
    final = initial.copy()
    final[accepted] = measured[accepted]
    return final, accepted, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="FITS paths or glob patterns")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("centroid_benchmark"))
    parser.add_argument("--match-radius", type=float, default=5.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--cutout", type=int, default=21)
    parser.add_argument("--centroid-limit", type=float, default=10.5)
    parser.add_argument("--prose-threshold", type=float, default=4.0)
    parser.add_argument("--prose-min-area", type=float, default=10.0)
    parser.add_argument("--prose-min-separation", type=float, default=10.0)
    parser.add_argument("--afphot-binary", type=Path, default=AFPHOT_BINARY)
    parser.add_argument("--afphot-params", type=Path, default=AFPHOT_PARAMS)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--read-noise", type=float, default=12.0)
    parser.add_argument("--adu-low", type=float, default=-1000.0)
    parser.add_argument("--adu-high", type=float, default=130000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = resolve_inputs(args.inputs, args.limit)
    if not frames:
        raise SystemExit("no input FITS frames matched")

    rows: list[dict] = []
    timings = {name: [] for name in METHODS}
    timings["afphot_total"] = []
    accepted_counts = {name: 0 for name in METHODS}
    attempts = 0
    disagreements = {
        "quadratic_afphot": [],
        "com_afphot": [],
        "gaussian2d_afphot": [],
        "quadratic_gaussian2d": [],
        "com_gaussian2d": [],
    }
    fwhm_arcsec: list[float] = []

    for frame_number, frame in enumerate(frames, start=1):
        data = np.asarray(fits.getdata(frame), dtype=float)
        header = fits.getheader(frame)
        if header.get("L1FWHM") is not None:
            fwhm_arcsec.append(float(header["L1FWHM"]))
        prose = prose_detect(
            frame,
            args.prose_threshold,
            args.prose_min_area,
            0.0,
            args.prose_min_separation,
        )
        initial = prose.coords[: args.top_n]

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
        timings["afphot_total"].append(time.perf_counter() - started)

        # Select stars detected by both systems before comparing centroid
        # definitions, avoiding a detection-completeness confound.
        matches = unique_matches(initial, afphot.coords, args.match_radius)
        if not matches:
            continue
        prose_indices = np.asarray([item[0] for item in matches], dtype=int)
        afphot_indices = np.asarray([item[1] for item in matches], dtype=int)
        common_initial = initial[prose_indices]
        afphot_coords = afphot.coords[afphot_indices]
        attempts += len(matches)

        refined = {}
        for name, function in METHODS.items():
            coords, accepted, elapsed = refine(
                data, common_initial, function, args.cutout, args.centroid_limit
            )
            refined[name] = coords
            accepted_counts[name] += int(accepted.sum())
            timings[name].append(elapsed)

        frame_disagreements = {
            "quadratic_afphot": np.linalg.norm(
                refined["quadratic"] - afphot_coords, axis=1
            ),
            "com_afphot": np.linalg.norm(refined["com"] - afphot_coords, axis=1),
            "gaussian2d_afphot": np.linalg.norm(
                refined["gaussian2d"] - afphot_coords, axis=1
            ),
            "quadratic_gaussian2d": np.linalg.norm(
                refined["quadratic"] - refined["gaussian2d"], axis=1
            ),
            "com_gaussian2d": np.linalg.norm(
                refined["com"] - refined["gaussian2d"], axis=1
            ),
        }
        for name, values in frame_disagreements.items():
            disagreements[name].extend(values.tolist())

        row = {
            "frame": str(frame),
            "matched_sources": len(matches),
            "l1fwhm_arcsec": header.get("L1FWHM", math.nan),
            **{f"{name}_seconds": timings[name][-1] for name in timings},
            **{
                f"{name}_median_px": float(np.median(values))
                for name, values in frame_disagreements.items()
            },
        }
        rows.append(row)
        print(f"[{frame_number}/{len(frames)}] {frame.name}: {len(matches)} matches")

    if not rows:
        raise SystemExit("no common sources were found")
    summary = {
        "description": "Pairwise centroid agreement; no method is ground truth.",
        "frames": len(rows),
        "matched_centroid_attempts": attempts,
        "l1fwhm_arcsec": distribution(fwhm_arcsec),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "timing_seconds_per_frame": {
            name: distribution(values) for name, values in timings.items()
        },
        "accepted_fraction": {
            name: accepted_counts[name] / attempts for name in METHODS
        },
        "pairwise_disagreement_px": {
            name: distribution(values) for name, values in disagreements.items()
        },
        "notes": [
            "Gaussian2D is a proxy, not ground truth.",
            "Photutils timings exclude FITS loading and detection.",
            "AFPhot total timing includes FITS loading and detection.",
            "Only sources detected by both pipelines within match-radius are compared.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    json_path = args.output.with_name(args.output.name + "_summary").with_suffix(
        ".json"
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"wrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
