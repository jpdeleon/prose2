"""Identify and (optionally) kill stuck MuSCAT photometry runs.

A "stuck" run is a ``run_photometry`` process (and its worker children) that is
**both** idle on the CPU and has stopped making progress, judged by the
timestamped ``*.log`` file the pipeline continuously writes into its
``--results_dir``. Process *age* alone is never used to decide — a legitimate
full run (thousands of frames × four bands) can run for hours — so we key on
*no log growth* plus *near-zero CPU*, which only a genuinely hung run exhibits.

This is a safety net, not the primary defence: the durable fix is bounding every
external query with a timeout (see :data:`prose.utils.NETWORK_TIMEOUT_S`). Use
this when an unforeseen hang (NFS stall, deadlock) still leaves orphans behind.

Safety
------
* **Dry-run by default** — prints what it *would* kill. Pass ``--apply`` to act.
* Scoped to the current user's processes only.
* Sends ``SIGTERM`` first, waits a grace period, then ``SIGKILL`` survivors —
  never leads with ``-9`` (a half-written FITS/CSV is worse than an orphan).
* Processes whose ``--results_dir`` cannot be determined are left alone unless
  ``--allow-no-log`` is given (then CPU + age are the only signals).

Examples
--------
::

    # Show what would be reaped (no changes)
    python -m prose.scripts.reap_stuck_runs

    # Actually kill runs idle with no log progress for 20+ minutes
    python -m prose.scripts.reap_stuck_runs --stale-minutes 20 --apply
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

DEFAULT_PATTERN = "prose.scripts.run_photometry"
LOG_GLOBS = ("*.log",)


def parse_results_dir(cmdline: list[str]) -> str | None:
    """Return the ``--results_dir`` value from a process command line, if any.

    Supports both ``--results_dir VALUE`` and ``--results_dir=VALUE`` forms.
    """
    for i, token in enumerate(cmdline):
        if token == "--results_dir" and i + 1 < len(cmdline):
            return cmdline[i + 1]
        if token.startswith("--results_dir="):
            return token.split("=", 1)[1]
    return None


def latest_activity_mtime(
    results_dir: str, globs: tuple[str, ...] = LOG_GLOBS
) -> float | None:
    """Return the newest mtime among matching files under *results_dir*.

    ``None`` when the directory is missing or contains no matching files.
    """
    base = Path(results_dir)
    if not base.is_dir():
        return None
    mtimes = [
        f.stat().st_mtime
        for pattern in globs
        for f in base.glob(pattern)
        if f.is_file()
    ]
    return max(mtimes) if mtimes else None


def is_stuck(
    *,
    now: float,
    latest_mtime: float | None,
    stale_seconds: float,
    cpu_percent: float,
    cpu_threshold: float,
) -> bool:
    """Decide whether a run is stuck.

    Stuck = log has not advanced for at least *stale_seconds* **and** CPU usage
    is below *cpu_threshold*. A missing log signal (``latest_mtime is None``) is
    treated as *not stuck* here; callers gate that case via ``--allow-no-log``.
    """
    if latest_mtime is None:
        return False
    log_is_stale = (now - latest_mtime) >= stale_seconds
    cpu_is_idle = cpu_percent < cpu_threshold
    return log_is_stale and cpu_is_idle


@dataclass
class RunGroup:
    """A run grouped by ``--results_dir`` with its processes and signals."""

    results_dir: str | None
    procs: list[psutil.Process]
    cpu_percent: float = 0.0
    latest_mtime: float | None = None
    min_age_s: float = 0.0


def _matching_processes(pattern: str, me: str) -> list[psutil.Process]:
    """Current user's live processes whose command line contains *pattern*.

    The reaper itself is excluded so it can never target its own process group.
    """
    self_pid = os.getpid()
    out: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "username", "cmdline"]):
        try:
            if proc.info["username"] != me or proc.pid == self_pid:
                continue
            cmdline = proc.info["cmdline"] or []
            joined = " ".join(cmdline)
            if pattern in joined and "reap_stuck_runs" not in joined:
                out.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def _group_runs(procs: list[psutil.Process]) -> list[RunGroup]:
    groups: dict[str | None, RunGroup] = {}
    for proc in procs:
        try:
            results_dir = parse_results_dir(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        groups.setdefault(results_dir, RunGroup(results_dir, [])).procs.append(proc)
    return list(groups.values())


def _sample(groups: list[RunGroup], sample_seconds: float, now: float) -> None:
    """Populate each group's CPU%, latest log mtime, and minimum process age."""
    for group in groups:
        for proc in group.procs:
            try:
                proc.cpu_percent(interval=None)  # prime psutil's internal counter
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    if sample_seconds > 0:
        time.sleep(sample_seconds)
    for group in groups:
        cpu_total = 0.0
        ages = []
        for proc in group.procs:
            try:
                cpu_total += proc.cpu_percent(interval=None)
                ages.append(now - proc.create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        group.cpu_percent = cpu_total
        group.min_age_s = min(ages) if ages else 0.0
        group.latest_mtime = (
            latest_activity_mtime(group.results_dir) if group.results_dir else None
        )


def _all_targets(group: RunGroup) -> list[psutil.Process]:
    """A group's processes plus all of their descendants, de-duplicated."""
    seen: dict[int, psutil.Process] = {}
    for proc in group.procs:
        seen[proc.pid] = proc
        try:
            for child in proc.children(recursive=True):
                seen[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return list(seen.values())


def _terminate(procs: list[psutil.Process], grace_seconds: float) -> None:
    for proc in procs:
        try:
            proc.terminate()  # SIGTERM
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _, alive = psutil.wait_procs(procs, timeout=grace_seconds)
    for proc in alive:
        try:
            proc.kill()  # SIGKILL survivors
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _fmt_age(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pattern", default=DEFAULT_PATTERN, help="cmdline substring to match"
    )
    p.add_argument(
        "--stale-minutes",
        type=float,
        default=15.0,
        help="kill only if the run's log has not grown for this many minutes",
    )
    p.add_argument(
        "--min-age-minutes",
        type=float,
        default=10.0,
        help="never touch processes younger than this",
    )
    p.add_argument(
        "--cpu-threshold",
        type=float,
        default=1.0,
        help="group CPU%% below which a run counts as idle",
    )
    p.add_argument(
        "--sample-seconds",
        type=float,
        default=2.0,
        help="window over which CPU usage is sampled",
    )
    p.add_argument(
        "--grace-seconds",
        type=float,
        default=10.0,
        help="seconds to wait after SIGTERM before SIGKILL",
    )
    p.add_argument(
        "--allow-no-log",
        action="store_true",
        help="also reap runs without a resolvable --results_dir (CPU+age only)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="actually send signals (default is a dry run)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    me = getpass.getuser()
    now = time.time()
    stale_seconds = args.stale_minutes * 60.0
    min_age_seconds = args.min_age_minutes * 60.0

    procs = _matching_processes(args.pattern, me)
    if not procs:
        print(f"no '{args.pattern}' processes found for user {me}")
        return 0

    groups = _group_runs(procs)
    _sample(groups, args.sample_seconds, now)

    stuck: list[RunGroup] = []
    for group in groups:
        if group.min_age_s < min_age_seconds:
            continue
        if group.results_dir is None:
            if args.allow_no_log and group.cpu_percent < args.cpu_threshold:
                stuck.append(group)
            continue
        if is_stuck(
            now=now,
            latest_mtime=group.latest_mtime,
            stale_seconds=stale_seconds,
            cpu_percent=group.cpu_percent,
            cpu_threshold=args.cpu_threshold,
        ):
            stuck.append(group)

    if not stuck:
        print(
            f"checked {len(groups)} run(s); none stuck "
            f"(stale>{args.stale_minutes:g}m, cpu<{args.cpu_threshold:g}%)"
        )
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    for group in stuck:
        targets = _all_targets(group)
        idle_min = (
            (now - group.latest_mtime) / 60.0 if group.latest_mtime else float("nan")
        )
        print(
            f"[{mode}] stuck run results_dir={group.results_dir} "
            f"pids={sorted(p.pid for p in targets)} "
            f"cpu={group.cpu_percent:.1f}% idle_log={idle_min:.1f}m "
            f"age={_fmt_age(group.min_age_s)}"
        )
        if args.apply:
            _terminate(targets, args.grace_seconds)

    if not args.apply:
        print("dry run: re-run with --apply to send SIGTERM/SIGKILL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
