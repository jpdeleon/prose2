import numpy as np
from astropy.stats import sigma_clip
from photutils.aperture import aperture_photometry

from prose import Block, Image

__all__ = ["AperturePhotometry", "AnnulusBackground"]


class AperturePhotometry(Block):
    def __init__(self, radii: np.ndarray = None, scale: bool = True, name=None):
        """Perform aperture photometry of each sources.

        |read| :code:`Image.data`, :code:`Image.sources`

        |write| :code:`Image.aperture`

        Parameters
        ----------
        radii : np.ndarray, optional
            apertures radii (definition varies depending on sources), by default None
        scale : bool, optional
            whether to scale radii with :code:`Image.fwhm` usually present in :code:`Image.epsf`, by default True
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name=name, read=["sources", "data"])
        if radii is None:
            # log-uniform
            self._radii = np.exp(np.linspace(np.log(0.1), np.log(12), 30))
        else:
            self._radii = radii
        self.scale = scale

        if self.scale:
            self.read.append("fwhm")

    def run(self, image: Image):
        """
        Perform aperture photometry on the given image and store the results.

        This method calculates the flux of each source within specified aperture
        radii. The radii can be scaled by the image's FWHM if specified. The
        calculated fluxes and their corresponding radii are stored in the image's
        aperture attribute.

        Parameters
        ----------
        image : Image
            The image object containing the data and sources on which to perform
            the aperture photometry.
        """
        if self.scale:
            radii = np.array(image.fwhm * self._radii)
        else:
            radii = np.array(self._radii)

        apertures = [image.sources.apertures(r) for r in radii]
        bad = ~np.isfinite(image.data)
        aperture_fluxes = np.array(
            [
                aperture_photometry(image.data, a, mask=bad)["aperture_sum"].data
                for a in apertures
            ]
        ).T
        areas = np.array([a.area_overlap(image.data, mask=bad) for a in apertures]).T
        geometric_areas = np.array([a.area_overlap(image.data) for a in apertures]).T
        missing = np.maximum(geometric_areas - areas, 0.0)
        # A masked sum is a useful diagnostic, not an estimate of missing
        # stellar light. Reject any aperture intersecting an invalid pixel.
        valid = np.isfinite(areas) & (areas > 0) & (missing < 1e-10)
        image.aperture = {
            "fluxes": aperture_fluxes,
            "radii": radii,
            "areas": areas,
            "valid": valid,
            "masked_area": missing,
        }

    @property
    def citations(self) -> list:
        return super().citations + ["photutils"]


class _AnnulusPhotometry(Block):
    def __init__(self, name=None, rin=5, rout=8, scale=True):
        super().__init__(name=name, read=["sources", "data"])
        self.rin = rin
        self.rout = rout
        self.scale = scale

    @property
    def citations(self) -> list:
        return super().citations + ["photutils"]


class AnnulusBackground(_AnnulusPhotometry):
    def __init__(
        self,
        rin: float = 5,
        rout: float = 8,
        sigma: float = 3,
        scale=True,
        name: str = None,
    ):
        """Estimate background around each source using an annulus aperture.

        |read| :code:`Image.data`, :code:`Image.sources`

        |write| :code:`Image.annulus`

        Parameters
        ----------
        rin : float, optional
            inner radius of the annulus, by default 5
        rout : float, optional
            outer radius of the annulus, by default 8
        sigma : float, optional
            sigma clipping applied to pixel within annulus before taking the median value, by default 3.
        scale : bool, optional
            whether to scale annulus to EPSF fwhm, by default True. If True, each image must contain an effective PSF and its model (e.g. using :py:class:`~prose.blocks.psf.MedianEPSF` and one of :py:class:`~prose.blocks.psf.Gaussian2D`)
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name=name, rin=rin, rout=rout, scale=scale)
        self.sigma = sigma

    def run(self, image: Image):
        """
        Estimate background around each source using an annulus aperture.

        Parameters
        ----------
        image : Image
            input image

        Notes
        -----
        If `scale` is True, the annulus radii are scaled with the fwhm of the effective PSF.
        The background is estimated using a sigma-clipped median (sigma={self.sigma}) of the pixel values within the annulus.
        """
        if self.scale:
            fwhm = image.fwhm
            rin = float(fwhm * self.rin)
            rout = float(fwhm * self.rout)
        else:
            rin = self.rin
            rout = self.rout

        annulus = image.sources.annulus(rin, rout)
        annulus_masks = annulus.to_mask(method="center")
        annulus_area = np.pi * (rout**2 - rin**2)

        bkg_median = []
        bkg_variance = []
        bkg_npix = []
        for mask in annulus_masks:
            values = mask.get_values(image.data, mask=~np.isfinite(image.data))
            clipped = sigma_clip(values, sigma=self.sigma).compressed()
            count = clipped.size
            bkg_npix.append(count)
            bkg_median.append(np.median(clipped) if count else np.nan)
            # Gaussian approximation for the variance of a sample median.
            bkg_variance.append(
                np.pi / 2 * np.var(clipped, ddof=1) / count if count > 1 else np.nan
            )

        image.computed["annulus"] = {
            "rin": rin,
            "rout": rout,
            "median": np.array(bkg_median),
            "sigma": self.sigma,
            "area": annulus_area,
            "median_variance": np.array(bkg_variance),
            "n_pixels": np.array(bkg_npix),
        }
