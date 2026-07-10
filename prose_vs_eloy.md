# prose2 vs. Eloy photometry

The main aperture-photometry implementations are computationally equivalent. Neither has an inherent correctness or performance advantage for identical inputs.

| Aspect | prose2 | Eloy |
|---|---|---|
| Core implementation | `AperturePhotometry.run()` | `aperture_photometry()` |
| Apertures | Via `image.sources.apertures(r)` | Direct `CircularAperture(coords, r)` |
| Integration method | Photutils default | Photutils default |
| Output shape | `(sources, radii)` | `(sources, radii)` |
| Background subtraction | Separate pipeline stage | Separate function |
| FWHM scaling | Built in | Caller must implement |
| Metadata/state | Stores fluxes and radii on `Image` | Returns bare array |

Relevant implementations:

- `prose/blocks/photometry.py:38`
- `/ut2/jerome/github/research/project/ext_tools/eloy/src/eloy/photometry.py:14`

## Correctness

- On a synthetic 1024×1024 image with 200 sources and 30 radii, the outputs were bit-for-bit identical:
  - maximum absolute difference: `0.0`
  - identical shape and NaN placement
- Both inherit Photutils' default aperture behavior.
- Both propagate NaNs inside an aperture. Neither function supplies a mask, error map, or explicit integration method.
- Both calculate partial aperture overlap at image edges through Photutils. Fully off-image apertures become `NaN`.
- The annulus background calculations are also effectively identical, although prose2 additionally records annulus geometry and area.
- prose2 provides stronger integration correctness:
  - optional FWHM-relative radii;
  - declared required image fields;
  - radii retained with results;
  - pipeline-level bad-pixel masking;
  - an end-to-end synthetic photometry test.
- Eloy's function is easier to call independently, but correctness depends more heavily on the caller supplying properly ordered `(x, y)` coordinates and appropriate radii.

## Performance

Benchmark environment: NumPy 1.26.4 and Photutils 2.3.0.

- Eloy median: `0.229 s`
- prose2-equivalent core median: `0.239 s`
- The observed difference was about 4%, but run ranges overlapped almost completely (`~0.217–0.264 s`). This is timing noise rather than a meaningful performance difference.

Both implementations rebuild every aperture and invoke Photutils separately for every radius. Runtime therefore scales approximately with:

```text
number of radii × number of sources × aperture area
```

## Conclusion

Raw photometry correctness and speed are tied. prose2 is the safer production implementation because its wrapper preserves geometry and pipeline invariants and has substantially better testing. Eloy provides a thinner reusable API, but it is not faster in a defensible way.

The largest optimization opportunity in both implementations is avoiding repeated independent Photutils setup and work across the aperture-radius grid.

## Centroiding: Photutils versus Ballet

The main prose2 photometry pipeline currently uses Photutils
`centroid_quadratic`. Eloy offers Ballet, a JAX/Flax convolutional model, and
prose2's `CentroidBallet` block has been updated to use the same newer model and
`centroid_15x15.npz` weights.

### MuSCAT3 benchmark

The methods were compared on 10 evenly sampled MuSCAT3 `ep02`/`rp` frames from
`/data/MuSCAT3/260119`, comprising 462 detected sources. Timing excludes FITS
loading and source detection.

| Method | Median time per frame | Median disagreement from Gaussian2D |
|---|---:|---:|
| Photutils quadratic | `0.0149 s` | `6.28 px` |
| Ballet, without JIT | `2.88 s` | `2.71 px` |
| Photutils Gaussian2D | `0.705 s` | reference |

The un-jitted Ballet implementation was approximately 193 times slower than
quadratic Photutils on this CPU. Ballet agreed more closely with Gaussian2D,
but Gaussian2D is only an accuracy proxy, not ground truth. The several-pixel
disagreements merit further validation against injected-source simulations or
astrometric truth before claiming that either method is more accurate.

The reproducible benchmark and full results are stored in:

- `scripts/compare_centroids_muscat3.py`
- `muscat3_centroid_comparison.json`
- `muscat3_centroid_comparison.csv`

### Effect of JAX JIT

JIT was measured separately on one MuSCAT3 `ep02` frame with 49 sources:

| Ballet execution mode | Time per frame |
|---|---:|
| Current un-jitted implementation | `~2.88 s` |
| JIT compilation plus first inference | `2.02 s` |
| JIT steady-state median | `0.336 s` |

Steady-state JIT made Ballet approximately 8.6 times faster than its un-jitted
execution, but it remained about 22 times slower than quadratic Photutils on
this CPU.

JAX specializes compiled functions by input shape. Changing the number of
sources caused new compilations of approximately 1.8–2.1 seconds for the tested
40-, 41-, 42-, and 47-source batches. Reusing the compiled 49-source shape took
about 0.33 seconds.

An optimized pipeline should therefore JIT the model application, pad source
batches to a fixed `max_num_stars`, discard padded predictions afterward, and
warm up the model before processing or benchmarking. GPU execution may improve
Ballet's relative performance, but on the tested CPU quadratic Photutils
remains the substantially faster choice.
