# Seven-instrument local versus nova astrometry.net benchmark

Measured on 2026-10-18 at prose2 commit `97de72945cedfe35115e2d3395de52920144f489`
(the implementation in PR #17). This run contains 16 paired attempts,
one repetition per frame. Local returned a WCS passing the basic sanity checks in
14/16 attempts; nova did so in
14/16. These counts describe this sample and
network conditions, not instrument-wide success probabilities.

## Results

The median wall times below include failed attempts. Grid RMS ranges include only
frames with two WCS solutions passing the sanity checks. No failed attempt was
retried or replaced in this run. `valid` means finite celestial WCS parameters and
pixel scales within 20% of the header/instrument hint; it does not mean catalog-validated
accuracy. The literal `no_solution` status means the backend returned no WCS; the
failure notes and archived logs distinguish timeouts, unsuccessful searches, and
transport failures.

| Instrument | Local valid / attempts | Nova valid / attempts | Local median wall (s) | Nova median wall (s) | Paired grid RMS range (arcsec) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MuSCAT | 2/2 | 2/2 | 2.75 | 12.63 | 0.743–1.098 |
| MuSCAT2 | 3/4 | 3/4 | 10.37 | 13.60 | 0.659–3.146 |
| MuSCAT3 | 2/2 | 2/2 | 4.30 | 15.42 | 0.018–0.161 |
| MuSCAT4 | 2/2 | 1/2 | 4.32 | 16.46 | 1.174–1.174 |
| QHY600CMOS | 2/2 | 2/2 | 6.31 | 15.85 | 0.044–0.052 |
| SBIGSTL6303 | 1/2 | 2/2 | 94.91 | 14.41 | 0.737–0.737 |
| Sinistro | 2/2 | 2/2 | 14.05 | 19.43 | 1.152–1.210 |

| Instrument / camera | Band | Local status / wall (s) | Nova status / wall (s) | Grid RMS (arcsec) | Grid RMS (pixels) | Grid max (arcsec) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MuSCAT / MuSCAT | g | valid, 3.20 | valid, 13.22 | 0.743 | 2.078 | 1.383 |
| MuSCAT / MuSCAT | r | valid, 2.30 | valid, 12.05 | 1.098 | 3.059 | 1.665 |
| MuSCAT2 / MuSCAT2 | g | no_solution, 120.01 | valid, 9.73 | — | — | — |
| MuSCAT2 / MuSCAT2 | r | valid, 18.20 | valid, 13.27 | 3.146 | 7.240 | 7.497 |
| MuSCAT2 / MuSCAT2 | i | valid, 1.99 | valid, 15.32 | 0.659 | 1.514 | 1.980 |
| MuSCAT2 / MuSCAT2 | z_s | valid, 2.55 | no_solution, 13.93 | — | — | — |
| MuSCAT3 / ep02 | rp | valid, 4.31 | valid, 14.93 | 0.018 | 0.066 | 0.033 |
| MuSCAT3 / ep03 | ip | valid, 4.30 | valid, 15.91 | 0.161 | 0.603 | 0.404 |
| MuSCAT4 / ep06 | gp | valid, 4.31 | no_solution, 16.30 | — | — | — |
| MuSCAT4 / ep07 | rp | valid, 4.33 | valid, 16.61 | 1.174 | 4.388 | 2.364 |
| QHY600CMOS / sq36 | ip | valid, 5.96 | valid, 14.14 | 0.052 | 0.069 | 0.089 |
| QHY600CMOS / sq36 | ip | valid, 6.66 | valid, 17.56 | 0.044 | 0.060 | 0.110 |
| SBIGSTL6303 / kb24 | rp | no_solution, 87.48 | valid, 12.75 | — | — | — |
| SBIGSTL6303 / kb95 | rp | valid, 102.35 | valid, 16.07 | 0.737 | 1.293 | 1.860 |
| Sinistro / fa14 | gp | valid, 13.53 | valid, 19.56 | 1.152 | 2.955 | 2.538 |
| Sinistro / fa04 | gp | valid, 14.56 | valid, 19.31 | 1.210 | 3.101 | 2.484 |

## Failure and quality notes

- MuSCAT2 g: local reached its 120-second wall deadline (logged exit -9); nova
  solved the identical staged pixels. The log records 943 extracted local sources.
- MuSCAT2 z_s and MuSCAT4 g: nova returned no WCS after `RemoteDisconnected`
  (remote end closed the connection without a response). These are transport
  failures, not evidence that nova's solver rejected the images. Both solved locally.
- SBIG kb24: local completed without a solution in 87.48 s (exit 0, no solved
  marker/WCS); nova solved in 12.75 s. The other SBIG camera, kb95, solved
  locally in 102.35 s versus 16.07 s online, with 0.737 arcsec grid RMS disagreement.
- MuSCAT2 r: the two WCS solutions disagree by 3.146 arcsec RMS and 7.497 arcsec
  maximum on the grid (7.240 and 17.104 nova pixels). Both pass the basic checks.
  The cause and which solution is more accurate have not been established.
- The earlier [four-scale MuSCAT2 run](../2026-10-18-muscat2/results.json) also
  timed out locally on g and had the same r-band disagreement, while its online
  z_s attempt succeeded. This run additionally includes scales 5204–5206 and
  different online timings/solutions; it is a separate experiment.

## Sample and methods

[selection.json](selection.json) lists all original files, targets, filters,
cameras, dimensions, exposure times, and pointing coordinates. Select the most
recent numeric observing night available at selection time (a calibrated directory
for MuSCAT), prefer uncompressed copies, and take the first frame from each of the
first two distinct camera prefixes in lexical order. This gives different bands
for MuSCAT/3/4 and different site/camera examples for SBIG and Sinistro. For the
single QHY600 camera, take the first and midpoint frame. MuSCAT2 reuses the four
frames from the preceding benchmark. Selection was fixed before solving, without
screening for solver success. This is a small convenience sample: targets,
exposure times, focus, sites, and nights differ, so instrument rows are not a
controlled ranking of camera performance.

MuSCAT and MuSCAT2 inputs are existing calibrated products. The other five use
BANZAI RLEVEL=91 products. Both backends receive byte-identical fresh primary-image
FITS copies containing the original numeric pixels and only instrument, pointing,
and scale hints; original WCS/distortion headers are removed. The source hashes in
[source-integrity.json](source-integrity.json) were rechecked after all solves:
all originals are unchanged. Original science images and staged pixel arrays are
not included in this repository; reproducing the experiment requires those data.
Generated WCS metadata, JSON/CSV reports, and logs are included per instrument.

Local astrometry.net is version 0.98, Ubuntu package `0.98+dfsg-1build1`, installed
under `~/.local/opt/astrometry-0.98` with private native/Python helper dependencies.
The runner uses the prose conda environment (versions in each JSON report).
[host.json](host.json) records actual hardware, load snapshot, and sampling details;
the host is shared. CPU is an Intel Xeon Gold 5120 at 2.20 GHz with 28 visible
logical CPUs. The nova server software version, queue, and network conditions are
uncontrolled. OS/index and nova caches were not reset; the MuSCAT2 frames had
previous attempts. These are end-to-end measurements, not cold-cache CPU tests.

Local uses only Tycho-2 + Gaia-DR2 5200 LIGHT scales 5200–5206, with explicit
per-instrument configs covering every XY HEALPix tile at nside 2 within one degree
of each pointing: tiles 0, 1, 26, 30, 32, and 47. All 42 files (4.279 GiB) finished
downloading before solving. The installed 2MASS files are not used by these
configs. The FITS HEALPIX/HPNSIDE headers were checked and SHA256 hashes recorded in
[index-manifest.json](index-manifest.json), including the official download URLs;
the Gaia directory did not provide publisher checksums. This is regional coverage,
not a full-sky Gaia installation. The catalog is linked from the official
[index page](https://data.astrometry.net/) to the
[5200 LIGHT directory](https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/).

Solves run serially, alternating local/nova order across frames within each
instrument batch. Nova uploads are private, with a ten-second minimum cooldown
between completed online attempts and subsequent submissions within each batch.
Both backends receive a 120-second timeout setting. Local enforces a wall deadline
(including extraction/index loading); nova's solve/poll budget does not cap every
network operation. Timings exclude staging, validation, and cooldown, but include
local extraction/index access or online upload/queue/solve/polling respectively.

WCS comparison uses a 5-by-5 grid spanning the image at zero-based pixel centres.
Sky values are angular separations; pixel offsets transform the local sky solution
through the nova WCS. Pairwise agreement is not absolute astrometric accuracy:
there is no independent truth catalog validation in this benchmark. Do not infer
a general speedup or change the default backend from this small run.

## Reproduction

Each instrument directory contains `results.json`, `results.csv`, `astrometry.cfg`,
`logs/`, and generated `wcs/`. JSON reports record the exact command, input/probe
SHA256 hashes, source HDU, settings, per-attempt UTC times, package versions,
benchmark and solver module hashes, config hash, and local executable/version.
Config paths refer to the original host. On another machine, install the same
indexes, change paths to their locations, and supply the original calibrated
frames and an astrometry.net API key via the environment/configuration; keep the
key out of commands and artifacts. Use a new output directory for every run.

For example, on the original host, from the PR checkout in the prose environment:

```sh
uv run --no-sync python -m prose.scripts.benchmark_astrometry \
  /data/MuSCAT/260123_calibrated/MSCT0_2601230001_calibrated.fits \
  /data/MuSCAT/260123_calibrated/MSCT1_2601230001_calibrated.fits \
  --astrometry-config ~/temp/astrometry-multi-20261018/MuSCAT.cfg \
  --index-label 'Tycho-2 + Gaia-DR2 5200 LIGHT, scales 5200-5206, XY/nside2 tile 1' \
  --limit 2 --repeats 1 --astrometry-timeout 120 \
  --output-dir ~/temp/astrometry-muscat-rerun
```

## Archive validation

All 16 original source checksums and all staged-probe/config hashes matched after
the run. The 12 paired WCS comparisons were recomputed from the archived WCS FITS
metadata and reproduced the reported metrics within 0.00002. No API key appears
in the archived files. The four astrometry regression test modules passed with
96 tests passed and 2 live/platform tests skipped. No benchmark or solver code was
changed for this experiment.

CSV line endings and log trailing whitespace are normalized in this archive;
measurement values and diagnostic text are unchanged.
