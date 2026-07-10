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
    "CentroidCOM",
    "CentroidGaussian2D",
    "CentroidQuadratic",
    "CentroidBallet",
]


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
