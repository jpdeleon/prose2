"""Compare local and nova astrometry.net on identical WCS-free FITS copies.

Run with ``uv run python -m prose.scripts.benchmark_astrometry --help``.
Online timing includes upload, queueing, solving and polling; WCS agreement
measures consistency between solutions, not accuracy against a truth catalog.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import astropy
import astroquery
import numpy as np
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_scales

from prose import __version__
from prose.scripts import solve_wcs_astrometry as swa

logger = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inputs(patterns: list[str], limit: int) -> list[Path]:
    """Expand explicitly selected files/globs in stable order, without duplicates."""
    files = set()
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        matches = [expanded] if Path(expanded).is_file() else glob.glob(expanded)
        matches = [Path(p).resolve() for p in matches if Path(p).is_file()]
        if not matches:
            raise ValueError(f"No files match: {pattern}")
        files.update(matches)
    return sorted(files)[:limit]


def prepare_probe(source: Path, destination: Path) -> dict:
    """Copy image pixels and only pointing/scale metadata; never reuse a WCS.

    A fresh header avoids leaving behind alternate WCS, SIP, or distortion-table
    keywords. Both backends receive this exact file, including the same hints.
    The first two-dimensional image HDU is used for compressed/multi-HDU files.
    """
    with fits.open(source, memmap=False) as hdus:
        for index, hdu in enumerate(hdus):
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
                if hdu.data is not None and hdu.data.ndim == 2:
                    break
        else:
            raise ValueError(f"No two-dimensional image in {source}")
        header = hdus[0].header.copy()
        header.update(hdu.header)
        clean = fits.Header()
        for key in (
            "INSTRUME",
            "RA",
            "DEC",
            "RA_OBJ",
            "DEC_OBJ",
            "PIXSCALE",
            "SECPIX",
            "CCDSCALE",
        ):
            if key in header:
                clean[key] = header[key]
        # A prior WCS centre may supply a pointing hint, never a full solution.
        for coord, alternate, wcs_key in (
            ("RA", "RA_OBJ", "CRVAL1"),
            ("DEC", "DEC_OBJ", "CRVAL2"),
        ):
            if coord not in clean and alternate not in clean and wcs_key in header:
                clean[coord] = header[wcs_key]
        shape = list(hdu.data.shape)
        if min(shape) < 2:
            raise ValueError(f"Image is too small for a comparison grid: {source}")
        fits.PrimaryHDU(data=hdu.data, header=clean).writeto(destination)
    return dict(
        path=str(source),
        sha256=file_sha256(source),
        source_hdu=index,
        shape=shape,
        probe=str(destination),
        probe_sha256=file_sha256(destination),
        hints=swa._header_hint(destination),
    )


def _finite_wcs(wcs) -> bool:
    # WCSLIB can turn NaN reference coordinates into finite-looking sky values.
    return all(
        np.all(np.isfinite(value))
        for value in (wcs.wcs.crval, wcs.wcs.crpix, wcs.pixel_scale_matrix)
    )


def compare_wcs(local, nova, shape, *, grid_size=5) -> dict:
    """Compare sky positions and local-sky -> nova-pixel offsets on a 0-based grid."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if not _finite_wcs(local) or not _finite_wcs(nova):
        raise ValueError("Non-finite WCS parameters")
    height, width = shape
    x, y = np.meshgrid(
        np.linspace(0, width - 1, grid_size), np.linspace(0, height - 1, grid_size)
    )
    x, y = x.ravel(), y.ravel()
    local_sky = local.pixel_to_world(x, y)
    nova_sky = nova.pixel_to_world(x, y)
    separation = np.asarray(local_sky.separation(nova_sky).arcsec)
    nx, ny = nova.world_to_pixel(local_sky)
    displacement = np.hypot(nx - x, ny - y)
    cx, cy = (width - 1) / 2, (height - 1) / 2
    center = float(
        local.pixel_to_world(cx, cy).separation(nova.pixel_to_world(cx, cy)).arcsec
    )
    if not (
        np.all(np.isfinite(separation))
        and np.all(np.isfinite(displacement))
        and math.isfinite(center)
    ):
        raise ValueError("Non-finite coordinates in WCS comparison")
    return dict(
        grid_samples=len(x),
        center_arcsec=center,
        sky_median_arcsec=float(np.median(separation)),
        sky_rms_arcsec=float(np.sqrt(np.mean(separation**2))),
        sky_p95_arcsec=float(np.percentile(separation, 95)),
        sky_max_arcsec=float(np.max(separation)),
        pixel_median=float(np.median(displacement)),
        pixel_rms=float(np.sqrt(np.mean(displacement**2))),
        pixel_max=float(np.max(displacement)),
    )


def _attempt(
    backend, probe, output, stem, *, api_key, timeout, local_options, publicly_visible
):
    log_path = output / "logs" / f"{stem}-{backend}.log"
    handler = logging.FileHandler(log_path)
    old_level = swa.logger.level
    swa.logger.addHandler(handler)
    swa.logger.setLevel(logging.INFO)
    wcs, error = None, None
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        if backend == "local":
            wcs = swa.solve_local(probe, **dict(local_options, timeout=timeout))
        else:
            wcs = swa.upload_and_solve(
                probe, api_key, timeout=timeout, publicly_visible=publicly_visible
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}".replace(api_key, "<redacted>")
        swa.logger.error("Benchmark attempt failed: %s", error)
    finally:
        elapsed = time.perf_counter() - start
        swa.logger.removeHandler(handler)
        swa.logger.setLevel(old_level)
        handler.close()
    record = dict(
        elapsed_s=elapsed,
        started_utc=started_utc,
        solved=wcs is not None,
        status="no_solution",
        error=error,
        log=str(log_path),
        wcs=None,
        scale_x_arcsec=None,
        scale_y_arcsec=None,
    )
    if error is not None:
        record["status"] = "error"
    elif wcs is not None:
        try:
            record["status"] = (
                "valid"
                if _finite_wcs(wcs) and swa._validate_local_wcs(wcs, probe, None)
                else "invalid"
            )
            if record["status"] == "invalid":
                record["error"] = (
                    "WCS failed finite-parameter, celestial-axis or pixel-scale validation"
                )
            wcs_path = output / "wcs" / f"{stem}-{backend}.fits"
            swa.save_wcs_fits(wcs, wcs_path)
            record["wcs"] = str(wcs_path)
            if wcs.has_celestial:
                scales = proj_plane_pixel_scales(wcs) * 3600
                record.update(
                    scale_x_arcsec=float(scales[0]), scale_y_arcsec=float(scales[1])
                )
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return record, wcs


def summarize(results) -> dict:
    summary = {}
    for backend in ("local", "nova"):
        attempts = [row[backend] for row in results if row[backend] is not None]
        elapsed = [r["elapsed_s"] for r in attempts]
        valid_times = [r["elapsed_s"] for r in attempts if r["status"] == "valid"]
        valid = len(valid_times)
        summary[backend] = dict(
            attempts=len(attempts),
            solved=sum(r["solved"] for r in attempts),
            valid=valid,
            valid_fraction=valid / len(attempts) if attempts else None,
            median_elapsed_s=float(np.median(elapsed)) if elapsed else None,
            p95_elapsed_s=float(np.percentile(elapsed, 95)) if elapsed else None,
            median_valid_elapsed_s=float(np.median(valid_times))
            if valid_times
            else None,
        )
    ratios = [
        r["nova"]["elapsed_s"] / r["local"]["elapsed_s"]
        for r in results
        if all(
            r[b] is not None and r[b]["status"] == "valid" for b in ("local", "nova")
        )
        and r["local"]["elapsed_s"] > 0
    ]
    summary["paired_valid"] = len(ratios)
    summary["paired_median_nova_over_local"] = (
        float(np.median(ratios)) if ratios else None
    )
    agreements = [r["agreement"] for r in results if r["agreement"] is not None]
    summary["agreement"] = dict(
        pairs=len(agreements),
        median_center_arcsec=float(np.median([a["center_arcsec"] for a in agreements]))
        if agreements
        else None,
        median_frame_rms_arcsec=float(
            np.median([a["sky_rms_arcsec"] for a in agreements])
        )
        if agreements
        else None,
        max_grid_arcsec=max(a["sky_max_arcsec"] for a in agreements)
        if agreements
        else None,
        median_frame_pixel_rms=float(np.median([a["pixel_rms"] for a in agreements]))
        if agreements
        else None,
        max_grid_pixels=max(a["pixel_max"] for a in agreements) if agreements else None,
    )
    return summary


def _rounded(value):
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _flatten(value, prefix=""):
    row = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            row.update(_flatten(item, name + "_"))
        else:
            row[name] = item
    return row


def write_report(report, output):
    """Checkpoint after each attempt; keep calculations at full precision."""
    report["summary"] = summarize(report["results"])
    rounded = _rounded(report)
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(rounded, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "results.json")
    rows = [_flatten(r) for r in rounded["results"]]
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        temporary = output / "results.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "results.csv")


def run_benchmark(
    paths,
    output,
    *,
    api_key,
    local_options=None,
    timeout=swa.DEFAULT_TIMEOUT_S,
    repeats=1,
    grid_size=5,
    online_delay=10,
    publicly_visible="n",
    metadata=None,
):
    """Run serial paired attempts. Failed solves remain in timing/success totals."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True)  # Refuse to overwrite a previous experiment.
    for name in ("inputs", "wcs", "logs"):
        (output / name).mkdir()
    report = dict(
        schema_version=1,
        status="running",
        created_utc=datetime.now(timezone.utc).isoformat(),
        metadata=metadata or {},
        settings=dict(
            timeout_s=timeout,
            repeats=repeats,
            grid_size=grid_size,
            online_delay_s=online_delay,
            publicly_visible=publicly_visible,
            validation="identical finite-parameter, celestial-axis and header-scale checks for both backends",
            timing="end-to-end backend wall time; excludes benchmark staging, validation and pacing",
            pixel_offsets="local sky positions transformed to nova pixels; zero-based image grid",
            accuracy="pairwise WCS agreement, not independent astrometric truth",
        ),
        inputs=[],
        results=[],
    )
    # Prepare every input before any upload so bad inputs fail without network work.
    for index, source in enumerate(paths):
        destination = output / "inputs" / f"{index:03d}.fits"
        report["inputs"].append(prepare_probe(Path(source).resolve(), destination))
    write_report(report, output)
    last_nova_end = None
    for index, frame in enumerate(report["inputs"]):
        for repeat in range(repeats):
            stem = f"{index:03d}-r{repeat:03d}"
            row = dict(
                frame=index,
                input=frame["path"],
                repeat=repeat,
                local=None,
                nova=None,
                agreement=None,
                comparison_error=None,
            )
            report["results"].append(row)
            solutions = {}
            # Alternate order across pairs; never parallelize external requests.
            order = (
                ("local", "nova")
                if (index * repeats + repeat) % 2 == 0
                else ("nova", "local")
            )
            row["order"] = list(order)
            for backend in order:
                if backend == "nova" and last_nova_end is not None:
                    wait = online_delay - (time.monotonic() - last_nova_end)
                    if wait > 0:
                        time.sleep(wait)
                logger.info("%s repeat %d: %s", frame["path"], repeat + 1, backend)
                row[backend], solutions[backend] = _attempt(
                    backend,
                    Path(frame["probe"]),
                    output,
                    stem,
                    api_key=api_key,
                    timeout=timeout,
                    local_options=local_options or {},
                    publicly_visible=publicly_visible,
                )
                if backend == "nova":
                    last_nova_end = time.monotonic()
                write_report(report, output)
            if all(row[b]["status"] == "valid" for b in order):
                try:
                    row["agreement"] = compare_wcs(
                        solutions["local"],
                        solutions["nova"],
                        frame["shape"],
                        grid_size=grid_size,
                    )
                except Exception as exc:
                    row["comparison_error"] = f"{type(exc).__name__}: {exc}"
            write_report(report, output)
    report["status"] = "complete"
    write_report(report, output)
    return report


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("must be positive")
    return value


def _nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("must be finite and nonnegative")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Calibrated FITS paths or quoted globs. Copies are uploaded to nova.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New experiment directory; must not exist.",
    )
    parser.add_argument("--limit", type=_positive_int, default=3)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--grid-size", type=_positive_int, default=5)
    parser.add_argument(
        "--online-delay",
        type=_nonnegative,
        default=10,
        help="Minimum seconds between completed and next nova request (default: 10).",
    )
    parser.add_argument("--publicly-visible", choices=["y", "n"], default="n")
    parser.add_argument(
        "--index-label",
        default=None,
        help="Catalog/release and scale/tile description for reproducibility.",
    )
    swa.add_local_solver_arguments(parser, include_timeout=False)
    parser.add_argument(
        "--astrometry-timeout",
        "--timeout",
        type=swa._positive_timeout,
        default=swa.DEFAULT_TIMEOUT_S,
        help="Timeout for both backends: local wall deadline and nova solve/poll budget (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        paths = resolve_inputs(args.inputs, args.limit)
        binary = shutil.which(args.solve_field)
        if binary is None:
            raise ValueError(f"Local solve-field not installed: {args.solve_field}")
        if args.astrometry_config is not None:
            args.astrometry_config = args.astrometry_config.expanduser().resolve()
            if not args.astrometry_config.is_file():
                raise ValueError(f"Index config not found: {args.astrometry_config}")
        key = swa._api_key()
        version = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        metadata = dict(
            benchmark_sha256=file_sha256(Path(__file__)),
            solver_module_sha256=file_sha256(Path(swa.__file__)),
            command=[
                sys.executable,
                "-m",
                "prose.scripts.benchmark_astrometry",
                *(argv if argv is not None else sys.argv[1:]),
            ],
            python=sys.version.split()[0],
            platform=platform.platform(),
            prose=__version__,
            numpy=np.__version__,
            astropy=astropy.__version__,
            astroquery=astroquery.__version__,
            local_executable=str(Path(binary).resolve()),
            local_version=(version.stdout or version.stderr).strip(),
            local_version_exit_code=version.returncode,
            config=str(args.astrometry_config) if args.astrometry_config else None,
            config_sha256=file_sha256(args.astrometry_config)
            if args.astrometry_config
            else None,
            index_label=args.index_label,
            nova_url="https://nova.astrometry.net",
            local_options=swa.local_solver_options(args),
        )
        report = run_benchmark(
            paths,
            args.output_dir,
            api_key=key,
            local_options=swa.local_solver_options(args),
            timeout=args.astrometry_timeout,
            repeats=args.repeats,
            grid_size=args.grid_size,
            online_delay=args.online_delay,
            publicly_visible=args.publicly_visible,
            metadata=metadata,
        )
    except KeyboardInterrupt:
        logger.warning(
            "Benchmark interrupted; completed attempts remain in the report."
        )
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        logger.error("%s", exc)
        return 2
    print(json.dumps(_rounded(report["summary"]), indent=2, allow_nan=False))
    print(f"Results: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
