#!/usr/bin/env python
"""Benchmark prose's Photutils and Ballet centroid blocks on MuSCAT3 FITS data.

Example
-------
uv run python scripts/compare_centroids_muscat3.py \
    /data/MuSCAT3/260119 --camera ep02 --frames 20 --stars 50

The benchmark uses the same PointSourceDetection coordinates as input to both
methods.  Photutils ``centroid_2dg`` is also evaluated as a slower reference;
it is not treated as ground truth, but agreement with it is a useful accuracy
proxy on real images.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from prose import FITSImage, blocks


METHODS = {
    "quadratic": blocks.CentroidQuadratic,
    "gaussian2d": blocks.CentroidGaussian2D,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--camera", default="ep02", help="MuSCAT3 camera ID")
    parser.add_argument("--glob", default="*-e91.fits")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--stars", type=int, default=50)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--cutout", type=int, default=21)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("centroid_comparison.json"))
    parser.add_argument("--csv", type=Path, default=Path("centroid_comparison.csv"))
    return parser.parse_args()


def selected_files(args):
    files = sorted(
        path for path in args.data_dir.rglob(args.glob) if args.camera in path.name
    )
    if not files:
        raise SystemExit(
            f"No {args.camera} files matching {args.glob!r} below {args.data_dir}"
        )
    if args.frames > 0 and len(files) > args.frames:
        # Evenly sample the night instead of measuring only adjacent exposures.
        indices = np.linspace(0, len(files) - 1, args.frames, dtype=int)
        files = [files[index] for index in indices]
    return files


def run_block(image, block):
    candidate = image.copy()
    before = candidate.sources.coords.copy()
    start = time.perf_counter()
    block.run(candidate)
    elapsed = time.perf_counter() - start
    after = candidate.sources.coords.copy()
    displacement = np.linalg.norm(after - before, axis=1)
    moved = displacement > 0
    return after, displacement, moved, elapsed


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "p90": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main():
    args = parse_args()
    from prose.blocks.centroids import CentroidBallet

    ballet = CentroidBallet(model_file=args.model_file)
    quadratic = METHODS["quadratic"](cutout=args.cutout)
    gaussian = METHODS["gaussian2d"](cutout=args.cutout)
    files = selected_files(args)

    rows = []
    timings = {name: [] for name in ("quadratic", "ballet", "gaussian2d")}
    agreements = {
        "quadratic_ballet": [],
        "quadratic_gaussian": [],
        "ballet_gaussian": [],
    }
    displacements = {name: [] for name in timings}
    moved = {name: 0 for name in timings}
    attempted = 0

    for frame_index, path in enumerate(files):
        image = FITSImage(path, skip_wcs=True)
        blocks.PointSourceDetection(n=args.stars, min_area=args.min_area).run(image)
        if not len(image.sources):
            continue

        results = {}
        for name, block in (
            ("quadratic", quadratic),
            ("ballet", ballet),
            ("gaussian2d", gaussian),
        ):
            coords, shift, was_moved, elapsed = run_block(image, block)
            results[name] = coords
            timings[name].append(elapsed)
            displacements[name].extend(shift)
            moved[name] += int(was_moved.sum())

        n_sources = len(image.sources)
        attempted += n_sources
        pair_values = {
            "quadratic_ballet": np.linalg.norm(
                results["quadratic"] - results["ballet"], axis=1
            ),
            "quadratic_gaussian": np.linalg.norm(
                results["quadratic"] - results["gaussian2d"], axis=1
            ),
            "ballet_gaussian": np.linalg.norm(
                results["ballet"] - results["gaussian2d"], axis=1
            ),
        }
        for name, values in pair_values.items():
            agreements[name].extend(values)

        rows.append(
            {
                "frame": frame_index,
                "file": str(path),
                "sources": n_sources,
                **{f"{name}_seconds": timings[name][-1] for name in timings},
                **{
                    f"{name}_median_px": float(np.nanmedian(values))
                    for name, values in pair_values.items()
                },
            }
        )
        print(f"[{frame_index + 1:>3}/{len(files)}] {path.name}: {n_sources} sources")

    if not attempted:
        raise SystemExit("No sources were detected in the selected frames")

    summary = {
        "dataset": str(args.data_dir),
        "camera": args.camera,
        "frames": len(rows),
        "centroid_attempts": attempted,
        "timing_seconds_per_frame": {
            name: distribution(values) for name, values in timings.items()
        },
        "ballet_to_quadratic_time_ratio": float(
            np.median(timings["ballet"]) / np.median(timings["quadratic"])
        ),
        "moved_fraction": {name: moved[name] / attempted for name in moved},
        "shift_from_detection_px": {
            name: distribution(values) for name, values in displacements.items()
        },
        "pairwise_disagreement_px": {
            name: distribution(values) for name, values in agreements.items()
        },
        "notes": [
            "Gaussian2D is an accuracy proxy, not ground truth.",
            "Timing excludes FITS loading and source detection.",
            "Moved fraction is not a rejection rate: an exact zero correction also counts as unchanged.",
            "Ballet timings include any JAX tracing/compilation caused by a new batch shape.",
        ],
    }

    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
