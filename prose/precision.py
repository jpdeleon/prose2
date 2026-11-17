"""Opt-in comparison-ensemble, aperture and outlier choices for faint targets.

Default prose differential photometry (:func:`prose.fluxes.auto_diff_1d`)
adds comparison stars in Broeg-weight order until the *target's* binned
scatter stops improving, and picks the aperture with the lowest target
point-to-point scatter. On faint, defocused targets that keeps only a few
stars and tunes both choices on the target itself (muscat-team/muscatdb#445).

This module adds the alternatives used by ``run_photometry``:

* :func:`bright_ensemble_weights` uses every comparison star brighter than a
  fraction of the target, after removing stars that disagree with the rest.
  Its weights are proportional to each star's mean flux, which is the same
  as summing the stars' raw fluxes.
* :func:`greedy_check_weights` and :func:`check_aperture_index` choose the
  ensemble and the aperture from a predicted target scatter. The prediction
  uses the target's error model and the comparison stars' empirical noise,
  measured on their check light curves, and never the target's own scatter.
* :func:`clip_star_outliers` replaces single-star spikes (e.g. cosmic rays)
  in comparison stars, found against a running median of each star's check
  light curve, by interpolation in time.
* :func:`noise_budget` splits the expected target noise into photon, sky,
  read and comparison terms next to the achieved scatter.

Every array is ``(star, time)`` or ``(aperture, star, time)`` as in
:class:`prose.fluxes.Fluxes`. Inputs are never modified.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from prose.fluxes import MAD_TO_SIGMA, diff, weights as broeg_weights

ENSEMBLES = ("greedy", "bright")
CRITERIA = ("target", "check")

# Defaults of the opt-in options (see ``PrecisionOptions``).
DEFAULT_MIN_FLUX_FRACTION = 0.3
DEFAULT_DISAGREE_FACTOR = 2.0
DEFAULT_CLIP_WINDOW = 11
# The disagreement cut needs a median of several stars to be meaningful.
MIN_STARS_FOR_DISAGREEMENT = 3


@dataclass(frozen=True)
class PrecisionOptions:
    """Comparison-ensemble, aperture-criterion and star-clipping options.

    The defaults reproduce prose's own behaviour (``greedy`` + ``target``,
    no star clipping).
    """

    ensemble: str = "greedy"
    criterion: str = "target"
    min_flux_fraction: float = DEFAULT_MIN_FLUX_FRACTION
    disagree_factor: float = DEFAULT_DISAGREE_FACTOR
    clip_sigma: float | None = None
    clip_window: int = DEFAULT_CLIP_WINDOW

    def __post_init__(self):
        if self.ensemble not in ENSEMBLES:
            raise ValueError(
                f"ensemble must be one of {ENSEMBLES}, got {self.ensemble!r}"
            )
        if self.criterion not in CRITERIA:
            raise ValueError(
                f"criterion must be one of {CRITERIA}, got {self.criterion!r}"
            )
        if not 0 <= self.min_flux_fraction:
            raise ValueError("min_flux_fraction must be nonnegative")
        if not self.disagree_factor > 1:
            raise ValueError("disagree_factor must be greater than 1")
        if self.clip_sigma is not None and not self.clip_sigma > 0:
            raise ValueError("clip_sigma must be positive")
        if self.clip_window < 3:
            raise ValueError("clip_window must be at least 3 frames")

    @property
    def is_default(self) -> bool:
        """True when differencing is prose's own (``greedy`` + ``target``)."""
        return self.ensemble == "greedy" and self.criterion == "target"


# ------------------------------------------------------------------ helpers


def ptp_scatter(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Robust point-to-point scatter: MAD sigma of first differences / sqrt(2).

    Insensitive to slow trends and to isolated outliers.
    """
    d = np.diff(np.asarray(values, dtype=float), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        center = np.nanmedian(d, axis=axis, keepdims=True)
        mad = np.nanmedian(np.abs(d - center), axis=axis)
    return MAD_TO_SIGMA * mad / np.sqrt(2)


def robust_std(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """MAD-based standard deviation about the median (sensitive to variability)."""
    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        center = np.nanmedian(values, axis=axis, keepdims=True)
        return MAD_TO_SIGMA * np.nanmedian(np.abs(values - center), axis=axis)


def eligible_comparisons(fluxes: np.ndarray, target: int) -> np.ndarray:
    """Stars usable as comparisons in a ``(star, time)`` matrix.

    Masked stars (``Fluxes.mask_stars`` sets them to -1), non-finite stars
    and the target are excluded, as in :func:`prose.fluxes.auto_diff_1d`.
    """
    eligible = np.all(np.isfinite(fluxes), axis=-1) & (np.mean(fluxes, axis=-1) > 0)
    eligible[target] = False
    return eligible


def _relative_error(fluxes: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Median fractional error per star."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nanmedian(errors, axis=-1) / np.abs(np.nanmean(fluxes, axis=-1))


def comparison_noise(fluxes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Empirical fractional noise of every star's check light curve.

    A star's check light curve is its flux relative to the weighted ensemble
    of the *other* stars (its own weight is left out by :func:`diff`).
    """
    check, _ = diff(fluxes, weights)
    return ptp_scatter(check.reshape(fluxes.shape))


def predicted_target_scatter(
    target_error: float, comp_sigma: np.ndarray, weights: np.ndarray
) -> float:
    """Fractional target scatter predicted for a comparison ensemble.

    ``target_error`` is the target's own fractional error (error model only);
    the ensemble adds ``sum(w_j^2 sigma_j^2) / (sum w_j)^2``.
    """
    w = np.asarray(weights, dtype=float)
    total = np.sum(w)
    if total <= 0:
        return np.inf
    sigma = np.where(w > 0, comp_sigma, 0.0)
    ensemble_var = np.sum((w * sigma) ** 2) / total**2
    return float(np.sqrt(target_error**2 + ensemble_var))


# --------------------------------------------------------------- ensembles


def bright_ensemble_weights(
    fluxes: np.ndarray,
    target: int,
    errors: np.ndarray | None = None,
    min_flux_fraction: float = DEFAULT_MIN_FLUX_FRACTION,
    disagree_factor: float = DEFAULT_DISAGREE_FACTOR,
) -> np.ndarray:
    """Flux-proportional weights over every comparison bright enough.

    Candidates are eligible stars with a mean flux at least
    ``min_flux_fraction`` x the target's (all eligible stars if none is).
    Stars whose check-light-curve robust standard deviation, relative to
    their expected noise, exceeds ``disagree_factor`` x the candidates'
    median are dropped, and the cut is repeated until no star is dropped.
    The standard deviation (not the point-to-point scatter) is used so that
    slowly varying stars are caught as well as noisy ones.
    """
    eligible = eligible_comparisons(fluxes, target)
    if not np.any(eligible):
        raise ValueError("no valid comparison stars")
    mean = np.where(eligible, np.mean(fluxes, axis=-1), 0.0)
    candidates = eligible & (mean >= min_flux_fraction * np.mean(fluxes[target]))
    if not np.any(candidates):
        candidates = eligible
    expected = _expected_noise(fluxes, errors, mean)
    while np.count_nonzero(candidates) >= MIN_STARS_FOR_DISAGREEMENT:
        check, _ = diff(fluxes, mean * candidates)
        ratio = robust_std(check.reshape(fluxes.shape)) / expected
        threshold = disagree_factor * np.median(ratio[candidates])
        drop = candidates & ~(ratio <= threshold)  # NaN ratios are dropped too
        if not np.any(drop) or np.all(drop[candidates]):
            break
        candidates = candidates & ~drop
    return mean * candidates


def _expected_noise(fluxes, errors, mean) -> np.ndarray:
    """Expected fractional noise per star: the error model, else photon scaling."""
    if errors is not None:
        return _relative_error(fluxes, errors)
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(np.where(mean > 0, mean, np.nan))


def greedy_check_weights(
    fluxes: np.ndarray, target: int, errors: np.ndarray
) -> np.ndarray:
    """Broeg-ranked ensemble chosen by predicted, not measured, target scatter.

    Comparisons are ranked by Broeg weight as in :func:`auto_diff_1d`, but
    every ensemble size is scored with :func:`predicted_target_scatter`
    (comparison noise from check light curves against the full eligible
    ensemble) and the best one is kept, so the target series is never used.
    """
    if errors is None:
        raise ValueError("the 'check' criterion needs flux errors")
    eligible = eligible_comparisons(fluxes, target)
    if not np.any(eligible):
        raise ValueError("no valid comparison stars")
    if np.count_nonzero(eligible) == 1:
        return eligible.astype(float)  # a lone star has no check light curve
    # Broeg weights of the comparisons alone: the target must not shape them.
    w = np.zeros(fluxes.shape[0])
    w[eligible] = broeg_weights(fluxes[eligible])
    if not np.any(w > 0):
        w = eligible.astype(float)
    comp_sigma = comparison_noise(fluxes, w)
    target_error = float(_relative_error(fluxes, errors)[target])
    ranked = [j for j in np.argsort(w)[::-1] if w[j] > 0 and np.isfinite(comp_sigma[j])]
    if not ranked:
        raise ValueError("no comparison star has a finite check-light-curve scatter")
    best, best_score = None, np.inf
    for count in range(1, len(ranked) + 1):
        candidate = np.zeros_like(w)
        candidate[ranked[:count]] = w[ranked[:count]]
        score = predicted_target_scatter(target_error, comp_sigma, candidate)
        if score < best_score:
            best, best_score = candidate, score
    return best


def ensemble_weights(
    fluxes: np.ndarray,
    target: int,
    errors: np.ndarray | None,
    options: PrecisionOptions,
) -> np.ndarray:
    """Weights for one aperture's ``(star, time)`` matrix under ``options``."""
    if options.ensemble == "bright":
        return bright_ensemble_weights(
            fluxes,
            target,
            errors,
            min_flux_fraction=options.min_flux_fraction,
            disagree_factor=options.disagree_factor,
        )
    if options.criterion == "check":
        return greedy_check_weights(fluxes, target, errors)
    raise ValueError("greedy + target is prose's own auto_diff; call that instead")


# ---------------------------------------------------------------- aperture


def check_aperture_index(
    fluxes: np.ndarray, errors: np.ndarray, weights: np.ndarray, target: int
) -> int:
    """Aperture with the lowest predicted target scatter.

    Uses each aperture's target error model and the empirical noise of its
    comparison stars' check light curves; the target series is not used.
    """
    if errors is None:
        raise ValueError("the 'check' criterion needs flux errors")
    scores = np.full(fluxes.shape[0], np.inf)
    for k in range(fluxes.shape[0]):
        w = np.asarray(weights[k], dtype=float)
        comp_sigma = comparison_noise(fluxes[k], w)
        target_error = float(_relative_error(fluxes[k], errors[k])[target])
        if np.isfinite(target_error):
            scores[k] = predicted_target_scatter(target_error, comp_sigma, w)
    if not np.any(np.isfinite(scores)):
        raise ValueError("no aperture yields a finite predicted target scatter")
    return int(np.argmin(scores))


# ---------------------------------------------------------------- clipping


def clip_star_outliers(
    fluxes: np.ndarray,
    stars: np.ndarray,
    sigma: float,
    window: int = DEFAULT_CLIP_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace outliers in the check light curves of ``stars`` by interpolation.

    For every aperture each star is normalized and divided by the median of
    the normalized ``stars`` (robust to a spike in any one of them). A sample
    is an outlier when it lies more than ``sigma`` MAD-sigmas from a running
    median of ``window`` frames in any aperture; it is then replaced, in all
    apertures, by linear interpolation in frame order between the star's
    unflagged samples. Stars not in ``stars`` (e.g. the target) are untouched.

    Returns the new ``(aperture, star, time)`` fluxes and the ``(star, time)``
    boolean mask of replaced samples.
    """
    cube = np.asarray(fluxes, dtype=float)
    stars = np.asarray(stars, dtype=bool)
    flagged = np.zeros(cube.shape[1:], dtype=bool)
    if np.count_nonzero(stars) < MIN_STARS_FOR_DISAGREEMENT:
        return cube.copy(), flagged
    for k in range(cube.shape[0]):
        flagged |= _flag_aperture(cube[k], stars, sigma, window)
    return _interpolate_flagged(cube, flagged), flagged


def _flag_aperture(fluxes, stars, sigma, window) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = fluxes / np.nanmedian(fluxes, axis=-1, keepdims=True)
        check = norm / np.nanmedian(norm[stars], axis=0)
    residual = check - median_filter(check, size=(1, window), mode="nearest")
    # The MAD of a running-median residual underestimates its spread (the
    # residual is exactly 0 whenever a sample is its window's median), so the
    # point noise comes from first differences and is widened by the noise of
    # the window median itself, variance ~ pi / (2 * window).
    scatter = ptp_scatter(check)[:, None] * np.sqrt(1 + np.pi / (2 * window))
    with np.errstate(invalid="ignore"):
        out = np.abs(residual) > sigma * scatter
    out[~stars] = False
    return out


def _interpolate_flagged(cube: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    out = cube.copy()
    frames = np.arange(cube.shape[-1])
    for j in np.flatnonzero(flagged.any(axis=-1)):
        good = ~flagged[j]
        if np.count_nonzero(good) < 2:
            continue
        for k in range(cube.shape[0]):
            out[k, j, ~good] = np.interp(frames[~good], frames[good], cube[k, j, good])
    return out


# ------------------------------------------------------------ differencing


def precision_diff(
    fluxes: np.ndarray,
    target: int,
    errors: np.ndarray | None,
    options: PrecisionOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Differential photometry of an ``(aperture, star, time)`` cube.

    Same return values as :func:`prose.fluxes.auto_diff`: differential
    fluxes, ``(aperture, star)`` weights and differential errors.
    """
    weights = np.array(
        [
            ensemble_weights(
                fluxes[k], target, None if errors is None else errors[k], options
            )
            for k in range(fluxes.shape[0])
        ]
    )
    diff_fluxes, diff_errors = diff(fluxes, weights, errors)
    return (
        diff_fluxes.reshape(fluxes.shape),
        weights,
        (None if diff_errors is None else diff_errors.reshape(errors.shape)),
    )


# ------------------------------------------------------------ noise budget


def noise_budget(
    raw_flux: float,
    sky: float,
    area: float,
    sky_area: float,
    gain: float,
    read_noise: float,
    diff_flux: np.ndarray,
    diff_error: np.ndarray,
) -> dict[str, float]:
    """Expected per-point target noise by term, next to the achieved scatter.

    ``raw_flux`` (sky-subtracted, in counts), ``sky`` (counts/pixel) and the
    aperture and annulus areas (pixels) follow the error model of
    :class:`prose.blocks.utils.GetFluxes`. ``comparison`` is what the
    pipeline's own error (median ``diff_error``) adds on top of the target
    terms. Every value is in parts per thousand; ``ratio`` is achieved
    point-to-point scatter over the expected total.
    """
    flux = max(float(raw_flux), np.finfo(float).tiny)
    rn2 = (read_noise / gain) ** 2
    photon_var = max(raw_flux, 0.0) / gain
    sky_var = area * max(sky, 0.0) / gain
    sky_var += area**2 * (np.pi / 2) * (max(sky, 0.0) / gain + rn2) / sky_area
    read_var = area * rn2
    target_var = (photon_var + sky_var + read_var) / flux**2
    total = float(np.nanmedian(diff_error))
    comp = np.sqrt(max(total**2 - target_var, 0.0))
    achieved_ptp = float(ptp_scatter(diff_flux))
    ppt = 1e3
    return {
        "photon_ppt": ppt * np.sqrt(photon_var) / flux,
        "sky_ppt": ppt * np.sqrt(sky_var) / flux,
        "read_ppt": ppt * np.sqrt(read_var) / flux,
        "comparison_ppt": ppt * comp,
        "expected_ppt": ppt * total,
        "achieved_std_ppt": ppt * float(np.nanstd(diff_flux)),
        "achieved_ptp_ppt": ppt * achieved_ptp,
        "ratio": achieved_ptp / total if total > 0 else np.nan,
    }
