import numpy as np
from astropy.wcs.utils import fit_wcs_from_points
from skimage.transform import warp
from twirl.match import count_cross_match

from prose.blocks.geometry import ComputeTransformTwirl
from prose.core import Block, Image

__all__ = ["TransformData", "AlignReferenceSources", "AlignReferenceWCS"]


# TODO test inverse
class TransformData(Block):
    def __init__(self, inverse=False, name=None):
        """Transform image data using its transform

        |read| :code:`Image.transform`

        |modify|

        Parameters
        ----------
        inverse: bool, optional
            whether to apply the inverse of the image transform by default False
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name, read=["transform"])
        self.inverse = inverse

    def run(self, image: Image):
        """
        Apply the image transformation to its data.

        This method uses the image's transformation matrix to warp the image data.
        If the `inverse` attribute is set to True, the inverse of the transformation
        is applied. The transformed data replaces the original data in the image.
        If a linear algebra error occurs (e.g., due to a singular matrix), the image
        is marked to be discarded.

        Parameters
        ----------
        image : Image
            The image object whose data is to be transformed.
        """

        try:
            image.data = warp(
                image.data,
                image.transform.inverse if self.inverse else image.transform,
                cval=np.nanmedian(image.data),
                output_shape=image.shape,
            )
        except np.linalg.LinAlgError:
            image.discard = True

    @property
    def citations(self) -> list:
        return super().citations + ["scikit-image"]


class AlignReferenceSources(Block):
    def __init__(
        self,
        reference: Image,
        name=None,
        verbose=False,
        discard_tolerance=0.5,
        match_tolerance=5,
    ):
        """Set Image sources to reference sources (from a reference Image)
        aligned to the Image

        |read| :code:`Image.transform`, :code:`Image.sources`

        |write| :code:`Image.sources`

        Parameters
        ----------
        reference : Image
            reference image containing sources
        name : _type_, optional
            _description_, by default None
        verbose : bool, optional
            _description_, by default False
        discard_tolerance: float, optional
            fraction of sources that needs to be matched before discarding image
        match_tolerance: float, optional
            maximum distance between matched sources in pixels, default 5
        """
        super().__init__(name, verbose, read=["transform", "sources"])
        self.reference_sources = reference.sources
        self._parallel_friendly = True
        self.discard_tolerance = discard_tolerance
        self.match_tolerance = match_tolerance
        self._transform_block = ComputeTransformTwirl(reference)

    def run(self, image: Image):
        """
        Align the sources of an image to the sources of a reference image.

        This method uses the transformation matrix of the image to align its sources
        with the sources of a reference image. If the image has not been transformed,
        the transformation is computed using a helper block. The alignment is performed
        by applying the inverse transformation to the coordinates of the reference sources.
        The image may be discarded if the number of matched sources is below a specified
        tolerance.

        Parameters
        ----------
        image : Image
            The image object whose sources are to be aligned.

        Raises
        ------
        AttributeError
            If the image does not have the necessary attributes.
        """
        if not image.discard:
            sources = self.reference_sources.copy()

            # backwards compatibility
            if "transform" not in image.computed:
                self._transform_block.run(image)

            new_sources_coords = image.transform.inverse(sources.coords.copy())

            # check if alignment potentially failed
            if self.discard_tolerance is not None:
                matches = count_cross_match(
                    new_sources_coords,
                    image.sources.coords,
                    tol=self.match_tolerance,
                )
                if matches < np.min(
                    [
                        self.discard_tolerance * len(image.sources),
                        len(self.reference_sources),
                    ]
                ):
                    image.discard = True

            sources.coords = new_sources_coords
            image.sources = sources

    @property
    def citations(self) -> list:
        return super().citations + ["scikit-image"]


class AlignReferenceWCS(Block):
    def __init__(self, reference: Image, name=None, verbose=False, n=6):
        """Create WCS based on a reference containing a valid WCS.

        To use this block, Image sources must match the sources from the reference (e.g. using AlignReferenceSources),
        i.e. same sources should be found at a given index in both images.

        |read| :code:`Image.sources`

        |write| :code:`Image.wcs`

        Parameters
        ----------
        reference : Image
            reference image containing a valid WCS
        n : int, optional
            number of stars used to match WCS, by default 6
        """
        super().__init__(name, verbose, read=["sources"])
        self.reference = reference
        assert reference.plate_solved, "reference must have valid WCS"
        self.n = n

    def run(self, image: Image):
        """
        Compute WCS based on a reference containing a valid WCS.

        To use this block, Image sources must match the sources from the reference (e.g. using AlignReferenceSources),
        i.e. same sources should be found at a given index in both images.

        Parameters
        ----------
        image: Image
            image to be plate solved
        """
        ref_skycoords = self.reference.wcs.pixel_to_world(
            *self.reference.sources.coords[0 : self.n].T
        )
        image.wcs = fit_wcs_from_points(
            image.sources.coords[0 : self.n].T, ref_skycoords
        )

    @property
    def citations(self) -> list:
        return super().citations + ["astropy"]
