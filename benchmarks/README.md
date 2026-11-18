# Release benchmark

Compares a candidate prose install with the previous release on real data, so
every release states what it changes in real light curves (#74). It runs on
the host that holds the raw data (ut2); GitHub-hosted runners cannot reach it.

## Datasets

`datasets.yaml` lists them. Two kinds:

- **fixed**: one aperture and a fixed comparison ensemble, so a difference
  comes from the photometry, not from selection. Pass/fail: frames matched on
  their midpoint timestamp must agree in flux to 0.001 and in BJD_TDB to 50 ms,
  at most 1% of frames may be lost, no baseline CSV column may disappear, and
  the target must be located in both runs.
- **auto**: default selection, as users run it. Reported side by side. A band
  is flagged for review, not failed, when the candidate writes no light curve,
  loses more than 1% of frames, its point-to-point or 10-min binned scatter
  grows by more than 20%, or the log reports a target-localization problem.
  Releases change selection on purpose (v3.7.0 dropped a comparison star that
  saturates during the night on V1298 Tau 231021 g, which raised the scatter),
  so a flag needs an explanation, not necessarily a fix.

Each run gets a private HOME whose `~/.prose` holds the `*.telescope` files
from `--telescopes` (default: this checkout's `data/`) plus `centroid.h5`
copied from your `~/.prose`. Both installs get the same files, so differences
come from the engine. The MuSCAT, Sinistro, SBIG and QHY600 definitions are
not part of a prose install, and a shared `~/.prose` holds whichever copies
were made by hand; without them MuSCAT2 frames lose their time and gain
keywords. Review telescope changes between releases as a diff of `data/`.
`PROSE_CONFIG_FILE` and `PROSE_CONFIG_SHA256` are removed from the
environment for the same reason.

## Release procedure

1. Install the release-please candidate into its own environment, next to the
   previous release's environment.
2. Run the benchmark (about 2 workers per run; the full set takes a few hours):

   ```bash
   TMPDIR=~/temp uv run python -m prose.scripts.release_benchmark \
       --baseline ~/miniconda3/envs/<previous>/bin/python \
       --candidate ~/miniconda3/envs/<candidate>/bin/python \
       --out ~/temp/bench-<version>
   ```

   `--kind fixed` or `--only <name> ...` runs a subset. The command exits 1 if
   any fixed band fails. The fixed check alone is also available as a pytest
   run: set `PROSE_BENCH_BASELINE` and `PROSE_BENCH_CANDIDATE` and run
   `uv run pytest -m benchmark tests/test_release_benchmark_real.py`.
3. Paste `report.md` into the release PR. Every fixed band must pass; give
   every flagged automatic band a one-line explanation.

Outputs per run (light curves, npz, `run.log`) stay under
`<out>/<dataset>/<baseline|candidate>/` for follow-up.
