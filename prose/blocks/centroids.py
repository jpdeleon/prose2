import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from photutils.centroids import (
    centroid_2dg,
    centroid_com,
    centroid_quadratic,
    centroid_sources,
)

from prose import Block


__all__ = [
    "AdaptiveCentroid",
    "CentroidCOM",
    "CentroidGaussian2D",
    "CentroidQuadratic",
    "CentroidBallet",
]


def _odd_ceil(value):
    value = int(np.ceil(value))
    return value if value % 2 else value + 1


class AdaptiveCentroid(Block):
    """Refine centroids using a PSF-shape-aware production policy.

    Compact frames use Photutils' quadratic centroid with center-of-mass as a
    fallback. Broad/defocused frames use center-of-mass directly because a
    quadratic peak fit can lock onto one side of a donut-shaped PSF.

    Parameters
    ----------
    compact_cutout : int, optional
        Minimum centroid box size, by default 21 pixels.
    defocus_fraction : float, optional
        Use center-of-mass when FWHM reaches this fraction of
        ``compact_cutout``, by default 0.75.
    cutout_fwhm : float, optional
        Adaptive box size in FWHM units, by default 2.5.
    max_cutout : int, optional
        Upper bound on the adaptive box size, by default 101 pixels.
    shift_fwhm : float, optional
        Maximum accepted displacement in FWHM units, by default 0.25.
    min_shift : float, optional
        Minimum maximum displacement, by default 3 pixels.
    """

    def __init__(
        self,
        compact_cutout=21,
        defocus_fraction=0.75,
        cutout_fwhm=2.5,
        max_cutout=101,
        shift_fwhm=0.25,
        min_shift=3.0,
        name=None,
    ):
        super().__init__(name=name, read=["sources", "data", "fwhm"])
        self.compact_cutout = _odd_ceil(compact_cutout)
        self.defocus_fraction = defocus_fraction
        self.cutout_fwhm = cutout_fwhm
        self.max_cutout = _odd_ceil(max_cutout)
        self.shift_fwhm = shift_fwhm
        self.min_shift = min_shift

    @staticmethod
    def _measure(data, coords, cutout, centroid_func):
        if not len(coords):
            return coords.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyUserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.asarray(
                centroid_sources(
                    data,
                    coords[:, 0],
                    coords[:, 1],
                    box_size=cutout,
                    centroid_func=centroid_func,
                )
            ).T

    def run(self, image):
        original = image.sources.coords.copy()
        fwhm = float(image.fwhm)
        if not np.isfinite(fwhm) or fwhm <= 0:
            fwhm = self.compact_cutout / 3

        cutout = min(
            self.max_cutout,
            _odd_ceil(max(self.compact_cutout, self.cutout_fwhm * fwhm)),
        )
        max_shift = max(self.min_shift, self.shift_fwhm * fwhm)
        defocused = fwhm >= self.defocus_fraction * self.compact_cutout
        primary_func = centroid_com if defocused else centroid_quadratic
        primary_name = "com" if defocused else "quadratic"

        in_image = np.all(original < image.shape[::-1] - (1, 1), axis=1)
        in_image &= np.all(original > (0, 0), axis=1)
        final = original.copy()
        methods = np.full(len(original), "region", dtype="<U9")
        valid = np.zeros(len(original), dtype=bool)

        indices = np.flatnonzero(in_image)
        candidates = self._measure(image.data, original[in_image], cutout, primary_func)
        shifts = np.linalg.norm(candidates - original[in_image], axis=1)
        accepted = np.all(np.isfinite(candidates), axis=1) & (shifts < max_shift)
        final[indices[accepted]] = candidates[accepted]
        methods[indices[accepted]] = primary_name
        valid[indices[accepted]] = True

        # On compact frames, COM is a robust fallback when a quadratic peak fit
        # is invalid or jumps too far. Defocused frames already use COM.
        fallback_local = (
            np.flatnonzero(~accepted) if not defocused else np.empty(0, int)
        )
        if len(fallback_local):
            fallback = self._measure(
                image.data, original[indices[fallback_local]], cutout, centroid_com
            )
            fallback_shifts = np.linalg.norm(
                fallback - original[indices[fallback_local]], axis=1
            )
            fallback_ok = np.all(np.isfinite(fallback), axis=1) & (
                fallback_shifts < max_shift
            )
            accepted_indices = indices[fallback_local[fallback_ok]]
            final[accepted_indices] = fallback[fallback_ok]
            methods[accepted_indices] = "com"
            valid[accepted_indices] = True

        image.sources.coords = final
        image.centroid_methods = methods
        image.centroid_shifts = np.linalg.norm(final - original, axis=1)
        image.centroid_valid = valid
        image.centroid_cutout = cutout
        image.centroid_max_shift = max_shift

    @property
    def citations(self) -> list:
        return super().citations + ["photutils"]


class _PhotutilsCentroid(Block):
    def __init__(self, centroid_func, limit=None, cutout=21, name=None):
        """Photutils centroiding

        Parameters
        ----------
        centroid_func : function
            photutils.centroids function
        limit : int, optional
            maximum deviation from initial coordinate, by default `cutout/2`
        cutout : int, optional
            size of the cutout to be used for centroiding, by default 21
        """
        super().__init__(name=name, read=["sources", "data"])
        self.cutout = cutout
        self.centroid_func = centroid_func
        if limit is None:
            limit = cutout / 2
        self.limit = limit

    def run(self, image):
        # *%+#@ photutils check (see photutils.centroids.core code...)
        """
        Run centroiding on the given image.

        Parameters
        ----------
        image : prose.Image
            the image to be processed

        Notes
        -----
        The centroiding is done on the detected sources and the new positions are
        stored in the `sources` attribute of the image. The sources that are
        outside the image are ignored.

        The `limit` parameter is used to set the maximum deviation from the
        initial coordinate. If the deviation is larger than `limit`, the source
        is kept at its original position.

        """
        in_image = np.all(image.sources.coords < image.shape[::-1] - (1, 1), axis=1)
        in_image = np.logical_and(
            in_image, np.all(image.sources.coords > (0, 0), axis=1)
        )
        x, y = image.sources.coords[in_image].T.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyUserWarning)
            centroid_sources_coords = np.array(
                centroid_sources(
                    image.data,
                    x,
                    y,
                    box_size=self.cutout,
                    centroid_func=self.centroid_func,
                )
            ).T

        sources_coords = image.sources.coords.copy()
        sources_coords[in_image] = centroid_sources_coords
        in_limit = (
            np.linalg.norm(image.sources.coords - sources_coords, axis=1) < self.limit
        )
        final_sources_coords = image.sources.coords.copy()
        final_sources_coords[in_limit] = sources_coords[in_limit]
        image.sources.coords = final_sources_coords

    @property
    def citations(self) -> list:
        return super().citations + ["photutils"]


class CentroidCOM(_PhotutilsCentroid):
    """Centroiding using ``photutils.centroids.centroid_com``

    |read| ``Image.sources``

    |write| ``Image.sources``

    Parameters
    ----------
    limit : int, optional
        maximum deviation from initial coordinate, by default `cutout/2`
    cutout : int, optional
        size of the cutout to be used for centroiding, by default 21
    """

    def __init__(self, limit=None, cutout=21):
        super().__init__(centroid_func=centroid_com, limit=limit, cutout=cutout)


class CentroidGaussian2D(_PhotutilsCentroid):
    """Centroiding using ``photutils.centroids.centroid_2dg``

    |read| ``Image.sources``

    |write| ``Image.sources``

    Parameters
    ----------
    limit : int, optional
        maximum deviation from initial coordinate, by default `cutout/2`
    cutout : int, optional
        size of the cutout to be used for centroiding, by default 21
    """

    def __init__(self, limit=None, cutout=21):
        super().__init__(centroid_func=centroid_2dg, limit=limit, cutout=cutout)


class CentroidQuadratic(_PhotutilsCentroid):
    """Centroiding using ``photutils.centroids.centroid_quadratic``

    |read| ``Image.sources``

    |write| ``Image.sources``

    Parameters
    ----------
    limit : int, optional
        maximum deviation from initial coordinate, by default `cutout/2`
    cutout : int, optional
        size of the cutout to be used for centroiding, by default 21
    """

    def __init__(self, limit=None, cutout=21):
        super().__init__(centroid_func=centroid_quadratic, limit=limit, cutout=cutout)


class CentroidBallet(Block):
    """Centroid sources with the pretrained JAX/Flax Ballet CNN.

    Parameters
    ----------
    model_file : path-like, optional
        Eloy-compatible ``centroid_15x15.npz`` weights. When omitted, weights
        are downloaded from ``lgrcia/ballet`` on Hugging Face Hub.
    model : object, optional
        An initialized object exposing ``centroid(cutouts)``. Primarily useful
        for sharing a model between blocks or for testing.
    limit : float, optional
        Maximum accepted displacement in pixels, by default 7.5.
    """

    def __init__(self, model_file=None, model=None, limit=None, name=None):
        super().__init__(name=name, read=["sources", "data"])
        if model is not None and model_file is not None:
            raise ValueError("pass either model or model_file, not both")
        if model is None:
            from prose.ballet import Ballet

            model = Ballet(model_file=model_file)
        self.model = model
        self.cutout = 15
        self.limit = self.cutout / 2 if limit is None else limit

    def run(self, image):
        n = self.cutout
        in_image = np.all(image.sources.coords < image.shape[::-1] - (1, 1), axis=1)
        in_image = np.logical_and(
            in_image, np.all(image.sources.coords > (0, 0), axis=1)
        )
        in_image_coords = image.sources.coords[in_image].copy()
        cutouts = image.data_cutouts(in_image_coords, (n, n))
        cutouts_origins = in_image_coords - n / 2

        centroid_sources_coords = cutouts_origins + self.model.centroid(cutouts)
        # if coords is nan (any of x, y), keep old coord
        nan_mask = np.any(np.isnan(centroid_sources_coords), 1)
        centroid_sources_coords[nan_mask] = in_image_coords[nan_mask]

        # apply limit
        sources_coords = image.sources.coords.copy()
        sources_coords[in_image] = centroid_sources_coords
        in_limit = (
            np.linalg.norm(image.sources.coords - sources_coords, axis=1) < self.limit
        )
        final_sources_coords = image.sources.coords.copy()
        final_sources_coords[in_limit] = sources_coords[in_limit]
        image.sources.coords = final_sources_coords

    @property
    def citations(self):
        return super().citations + ["jax", "flax", "ballet"]
