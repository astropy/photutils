from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from ..models import FittableImageModel


__all__ = ['EPSFModel']


class EPSFModel(FittableImageModel):
    """
    A subclass of `FittableImageModel` that adds a ``pixel_scale``
    attribute.

    The ``pixel_scale`` can be used in conjunction with the `PSFStar`
    pixel scale when fitting (and building) the PSF.  This allows
    fitting (and building) a PSF using images of stars with different
    pixel scales.

    `EPSFModel` has the same parameters as `FittableImageModel` with the
    addition of a single parameter listed below.

    Parameters
    ----------
    pixel_scale : float or tuple of two floats, optional
        The pixel scale (in arbitrary units) of the PSF.  The
        ``pixel_scale`` can either be a single float or tuple of two
        floats of the form ``(x_pixscale, y_pixscale)``.  If
        ``pixel_scale`` is a scalar then the pixel scale will be the
        same for both the x and y axes.  The PSF ``pixel_scale`` is used
        in conjunction with the star pixel scale when building and
        fitting the PSF.  This allows for building (and fitting) a PSF
        using images of stars with different pixel scales (e.g. velocity
        aberrations).  Either ``oversampling`` or ``pixel_scale`` must
        be input.  If both are input, ``pixel_scale`` will override the
        input ``oversampling``.
    """

    def __init__(self, data, flux=1.0, x_0=0, y_0=0,
                 normalize=True, normalization_correction=1.0,
                 origin=None, oversampling=1., pixel_scale=1., fill_value=0.,
                 ikwargs={}):

        super(EPSFModel, self).__init__(
            data=data, flux=flux, x_0=x_0, y_0=y_0, normalize=normalize,
            normalization_correction=normalization_correction, origin=origin,
            fill_value=fill_value, ikwargs=ikwargs)

        self._pixel_scale = pixel_scale

    @property
    def pixel_scale(self):
        """
        The pixel scale (in arbitrary units) of the PSF.  The
        ``pixel_scale`` can either be a single float or tuple of two
        floats of the form ``(x_pixscale, y_pixscale)``.  If
        ``pixel_scale`` is a scalar then the pixel scale will be the
        same for both the x and y axes.  The PSF ``pixel_scale`` is used
        in conjunction with the `PSF_Star` pixel scale when building and
        fitting the PSF.  This allows for building and fitting a PSF
        using images of stars with different pixel scales (e.g. velocity
        aberrations).  Either ``oversampling`` or ``pixel_scale`` must
        be input.  If both are input, ``pixel_scale`` will override the
        input ``oversampling``.
        """

        return self._pixel_scale

    @pixel_scale.setter
    def pixel_scale(self, pixel_scale):
        if pixel_scale is not None:
            pixel_scale = np.atleast_1d(pixel_scale)
            if len(pixel_scale) == 1:
                pixel_scale = np.repeat(pixel_scale, 2).astype(float)
            elif len(pixel_scale) > 2:
                raise ValueError('pixel_scale must be a scalar or tuple '
                                 'of two floats.')

        self._pixel_scale = pixel_scale
