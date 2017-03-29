# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Core functionalities for building a PSF.
"""
from __future__ import absolute_import, division, print_function

__all__ = ['PSFBuilder', 'build_psf']


# TODO: Implement based on API from Erik Tollerud.
# TODO: The other modules might not be needed depending on how this pans out.
class PSFBuilder(object):
    """
    Class for building PSF.

    Parameters
    ----------
    model : `~astropy.modeling.Fittable2DModel`
        Initial model for the PSF.

    fitter : `~astropy.modeling.fitting.Fitter`
        Fitter for the PSF.

    recenterer : func
        Function for recentering a star on an image.
        Example API::

            def my_recenterer(star_image, x0, y0):
                x1, y1 = do_something(x0, y0)
                return x1, y1

    max_pdf_iters : int
        Maximum number of interations for PSF fitting.

    psf_total_flux : float or `None`
        If not `None`, PSF model is normalized such that its total flux
        is the given number (e.g., 1). Note that this might not mean the
        ePSF sums to 1; This simply allows aperture corrections to be
        folded into the model.

    """
    def __init__(self, model, fitter=None, recenterer=None, max_pdf_iters=10,
                 psf_total_flux=None):
        self.model = model
        self.fitter = fitter
        self.max_pdf_iters = max_pdf_iters
        self.psf_total_flux = psf_total_flux

    def __call__(self, images, psf_stars):
        """
        Run the builder.

        Parameters
        ----------
        images : list of `~astropy.nddata.NDData`
            Images for building PSF.

        psf_stars : `~astropy.table.Table`
            Table containing stars to use for building PSF.
            It must contain one of the following pair of columns;
            If both are present, the second pair is ignored:

            1. ``xcentroid`` and ``ycentroid``, which will be interpreted
               as X and Y (0-indexed) in the first image in ``images``.
               WCS is only required if there is more than one image.
            2. ``world_x`` and ``world_y``, which will use the WCS from each
               individual image in ``images`` to convert to pixel space.

        Returns
        -------
        psf_model : `~photutils.psf.models.FittableImageModel`
            PSF model that maps pixel locations to flux.

        psf_stars_perdither : list of `~astropy.table.Table`
            A list of tables corresponding to ``psf_stars``, with information
            like the exact pixel coordinates for that dither (computed from
            the WCS), how much that star in that dither was weighted in the
            final PSF, etc.

        """
        pass  # do magic here


# TODO: Remove this if extreme stretch goal too stretchy.
def build_psf(images):
    """
    Build PSF model from images using software default.

    Parameters
    ----------
    images : list of `~astropy.nddata.NDData`
        Images for building PSF.

    Returns
    -------
    psf_model : `~photutils.psf.models.FittableImageModel`
        PSF model that maps pixel locations to flux.

    psf_stars_perdither : list of `~astropy.table.Table`
        A list of tables corresponding to ``psf_stars``, with information
        like the exact pixel coordinates for that dither (computed from
        the WCS), how much that star in that dither was weighted in the
        final PSF, etc.

    """
    # psf_builder = PSFBuilder(model=EPSF() fitter=EPSFFitter())
    # psf_stars = MagicTable()
    # return psf_builder(images, psf_stars)
    pass
