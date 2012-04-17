# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""

import math
import abc

import numpy as np

__all__ = ["aperture_circular", "aperture_elliptical", "annulus_circular",
           "annulus_elliptical", "Aperture", "EllipticalAnnulus", 
           "aperture_photometry"]


class Aperture(object):
    """An abstract base class for an arbitrary 2-d aperture.

    Derived classes should contain whatever internal data is needed to define
    the aperture, and provide methods 'encloses' and 'extent' (and optionally,
    'area').
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encloses(xx, yy):
        """Return a bool array representing which elements of an array
        are in an aperture.

        Parameters
        ----------
        xx : `~numpy.ndarray`
            x coordinate of each element
        yy : `~numpy.ndarray`
            y coordinate of each element

        Returns
        -------
        in_aper : `~numpy.ndarray` (bool)
            
        """
        return

    @abc.abstractmethod
    def extent():
        """Smallest parallelpiped containing aperture.
        
        Returns
        -------
        x_min, x_max, y_min, y_max: float
        """
        return

    def area():
        """Return area of aperture or None if too difficult to calculate.""" 
        return None

class EllipticalAnnulus(Aperture):
    """A class for representing an elliptical annulus."""

    def __init__(self, xc, yc, a_in, a_out, b_out, theta):
        """
        Parameters
        ----------
        xc, yc, a_in, a_out, b_out, theta: float
            Parameters defining the annulus. Respectively, the x coordinate 
            center, y coordinate center, the inner semimajor axis, the outer
            semimajor axis, the outer semiminor axis, and the position angle
            of the semimajor axis in radians.
        
        """

        self.xc = xc
        self.yc = yc
        self.a_in = a_in
        self.b_in = a_in * b_out / a_out
        self.a_out = a_out
        self.b_out = b_out
        self.theta = theta

        if not (self.a_out > self.a_in):
            raise ValueError('a_out must be greater than a_in')

    def extent(self):
        # It is possible to get a tighter bounding box...
        return (self.xc - self.a_out, self.xc + self.a_out,
                self.yc - self.a_out, self.yc + self.a_out)

    def encloses(self, xx, yy):
        xx_off = xx - self.xc
        yy_off = yy - self.yc
        numerator1 = (xx_off * math.cos(self.theta) - 
                      yy_off * math.sin(self.theta))
        numerator2 = (yy_off * math.cos(self.theta) + 
                      xx_off * math.sin(self.theta))
        inside_outer_ellipse = ((numerator1 / self.a_out) ** 2 +
                                (numerator2 / self.b_out) ** 2) < 1.
        outside_inner_ellipse = ((numerator1 / self.a_in) ** 2 +
                                 (numerator2 / self.b_in) ** 2) > 1.
        return (inside_outer_ellipse & outside_inner_ellipse)

    def area():
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)
        

def aperture_photometry(arr, aperture, bkgerr=None, gain=1., mask=None,
                        subpixels=5):
    """Sum flux within aperture(s).

    .. warning::
        Some options not yet implemented.

    Parameters
    ----------
    arr : array_like
        The 2-d array on which to perform photometry.
    aperture: an `Aperture`-like object or array of `Aperture`-like objects
        The apertures to use for photometry. For arrays of apertures, 1-d and
        2-d arrays are allowed and the following convention is followed:
        1-d
            Each aperture is associated with a different object
        2-d
            Axis=1 (trailing/fast axis) is associated with different objects.
            Axis=0 (leading/slow axis) is associated with multiple apertures
            for a single aperture.
        These rules are useful to avoid resampling the input array multiple
        times. (The input array is only resampled once for each object.)
    bkgerr : float or array_like, optional
        Background (sky) error, interpreted as Gaussian 1-sigma uncertainty
        per pixel. If ``None`` fluxerr is not returned.
    gain : float or array_like, optional
        Ratio of Counts (or electrons) per flux (units of the array),
        for the purpose of calculating Poisson error.
        ARRAY_LIKE NOT YET IMPLEMENTED.
    mask : array_like (bool), optional
        Mask to apply to the data.  NOT YET IMPLEMENTED.
    subpixels : int, optional
        Resample pixels by this factor (in each dimension) when summing
        flux in apertures. That is, each pixel is divided into
        ``subpixels ** 2`` subpixels.

    Returns
    -------
    flux : float or `~numpy.ndarray`
        Enclosed flux in annuli(s). The output shape matches the shape of
        the input 'aperture'.
    fluxerr : float or `~numpy.ndarray`
        Uncertainty in flux values. Not returned if bkgerr is `None`.

    .. note::
        The coordinates are zero-indexed, meaning that ``(x, y) = (0.,
        0.)`` corresponds to the center of the lower-left array
        element.  The value of arr[0, 0] is taken as the value over
        the range ``-0.5 < x <= 0.5``, ``-0.5 < y <= 0.5``. The array
        is thus defined over the range ``-0.5 < x <= arr.shape[1] -
        0.5``, ``-0.5 < y <= arr.shape[0] - 0.5``.

    """

    # Check input array type and dimension.
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        raise TypeError('Complex type not supported')
    if arr.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(arr.ndim))

    # Check 'aperture' shape and type
    apertures = np.asarray(aperture)  # numpy array eases shape testing
    output_shape = apertures.shape  # Original shape to return.
    if apertures.ndim > 2:
        raise ValueError('{0}-d aperture array not supported. '
                         'Only 2-d arrays supported.'.format(apertures.ndim))
    apertures = np.atleast_2d(apertures)  # Make shapes uniform internally
    for aperture in apertures.ravel():
        if not isinstance(aperture, Aperture):
            raise ValueError("'aperture' must be an instance of Aperture.")

    # Check background error shape, convert to variance for internal use.
    scalar_bkgvar = False
    bkgvar = bkgerr
    if bkgvar is not None:
        if np.isscalar(bkgvar):
            scalar_bkgvar = True
            bkgvar = bkgerr ** 2
        else:
            bkgvar = np.asarray(bkgvar) ** 2
            if bkgvar.shape != arr.shape:
                raise ValueError('bkgerr must match shape of input array')

    # Check that 'subpixels' is an int and is 1 or greater.
    subpixels = int(subpixels)
    if subpixels < 1:
        raise ValueError('an integer greater than 0 is required')
    subpixelsize = 1. / subpixels  # Size of subpixels in original pixels.
    subpixelarea = subpixelsize * subpixelsize

    # Initialize arrays to return.
    flux = np.zeros(apertures.shape, dtype=np.float)
    if bkgvar is not None:
        fluxvar = np.zeros(apertures.shape, dtype=np.float)

    # 'extents' will hold the extent of all apertures for a given object.
    extents = np.empty((apertures.shape[0], 4), dtype=np.float)

    for i in range(apertures.shape[1]):  # Loop over objects.

        # Fill 'extents' with extent of all apertures for this object.
        for j, aperture in enumerate(apertures[:, i]):
            extents[j] = aperture.extent()

        # Set array index extents to encompass all apertures for this object.
        x_min = int(extents[:, 0].min() + 0.5)
        x_max = int(extents[:, 1].max() + 1.5)
        y_min = int(extents[:, 2].min() + 0.5)
        y_max = int(extents[:, 3].max() + 1.5)

        # Check that at least part of the sub-array is in the image.
        if (x_min >= arr.shape[1] or x_max <= 0 or
            y_min >= arr.shape[0] or y_max <= 0):
            # TODO: flag all the apertures for this object
            continue

        # Limit sub-array to be within the image.
        x_min = max(x_min, 0)
        x_max = min(x_max, arr.shape[1])
        y_min = max(y_min, 0)
        y_max = min(y_max, arr.shape[0])

        # Get the sub-array of the image.
        subarr = arr[y_min:y_max, x_min:x_max]
        if subpixels > 1:
            subarr = np.repeat(np.repeat(subarr, subpixels, axis=0),
                               subpixels, axis=1)

        # Get the sub-array of background variance.
        if bkgvar is not None and not scalar_bkgvar:
            subbkgvar = bkgvar[y_min:y_max, x_min:x_max]
            if subpixels > 1:
                subbkgvar = np.repeat(np.repeat(subbkgvar, subpixels, axis=0),
                                      subpixels, axis=1)

        # Get the coordinates of each pixel in the sub-array in units of
        # the original image pixels.
        x_vals = np.arange(x_min - 0.5 + subpixelsize / 2.,
                           x_max - 0.5 + subpixelsize / 2.,
                           subpixelsize)
        y_vals = np.arange(y_min - 0.5 + subpixelsize / 2.,
                           y_max - 0.5 + subpixelsize / 2.,
                           subpixelsize)
        xx, yy = np.meshgrid(x_vals, y_vals)

        # Loop over apertures for this object.
        for j in range(apertures.shape[0]):
            in_aper = apertures[j, i].encloses(xx, yy)
            flux[j, i] = subarr[in_aper].sum() * subpixelarea
            if bkgvar is not None:
                if scalar_bkgvar:
                    area = apertures[j, i].area()
                    if area is None:
                        area = in_aper.sum() * subpixelarea
                    fluxvar[j, i] = (bkgvar * area + flux[j, i] / gain)
                else:
                    fluxvar[j, i] = (bkgvar[in_aper].sum() * subpixelarea +
                                     flux[j, i] / gain)

    # Reshape output array(s) to match input array of apertures.
    flux = flux.reshape(output_shape)
    if bkgvar is not None:
        fluxerr = np.sqrt(fluxvar)
        fluxerr = fluxerr.reshape(output_shape)

    # If input was a simple 'Aperture' instance, return floats.
    if isinstance(aperture, Aperture):
        if bkgvar is None:
            return flux[()]
        else:
            return flux[()], fluxerr[()]

    # Otherwise, input aperture was an array of some sort, so return arrays.
    if bkgvar is None:
        return flux
    else:
        return flux, fluxerr


def aperture_circular(arr, xc, yc, r, bkgerr=None, gain=1., mask=None,
                      maskmode='mirror', subpixels=5):
    """Return sum of array enclosed in circular aperture(s)."""
    pass


def aperture_elliptical(arr, xc, yc, a, b, theta, bkgerr=None, gain=1.,
                        mask=None, maskmode='mirror', subpixels=5):
    """Return sum of array enclosed in elliptical aperture(s)."""
    pass


def annulus_circular(arr, xc, yc, r_in, r_out, bkgerr=None, gain=1.,
                     mask=None, maskmode='mirror', subpixels=5):
    """Return sum of array enclosed in circular annuli."""
    pass


def annulus_elliptical(arr, xc, yc, a_in, a_out, b_out, theta,
                       bkgerr=None, gain=1., mask=None, maskmode='mirror',
                       subpixels=5):
    """Sum flux within elliptical annuli.

    Parameters
    ----------
    arr : array_like
        The 2-d array on which to perform photometry.
    xc, yc : float or list_like
        The x and y coordinates of the aperture center(s) in pixels.
    a_in, a_out, b_out, theta : array_like
        Parameters specifying the elliptical annuli: respectively the
        inner semimajor axis in pixels, the outer semimajor axis in
        pixels, the outer semiminor axis in pixels, and the
        position angle of the semimajor axis in radians. ``a_in``,
        ``a_out``, ``b_out``, ``theta`` must be broadcastable to the
        same shape. Different shapes are treated as follows:

        scalar
            The same parameters are applied to all objects in ``xc``, ``yc``
        1-d
            The parameters must be the same length as ``xc``, ``yc``
            and each object gets its own annulus parameters.
        2-d
            axis=1 corresponds to the object and must be either the same
            length as ``xc``, ``yc``, or 1. axis=0 specifies multiple
            annulus parameters for each object.
    bkgerr : float or array_like, optional
        Background (sky) error, interpreted as Gaussian 1-sigma uncertainty
        per pixel. If ``None`` fluxerr is not returned.
    gain : float or array_like, optional
        Ratio of Counts (or electrons) per flux (units of the array),
        for the purpose of calculating Poisson error.
        ARRAY_LIKE NOT YET IMPLEMENTED.
    mask : array_like (bool), optional
        Mask to apply to the data.  NOT YET IMPLEMENTED.
    subpixels : int, optional
        Resample pixels by this factor (in each dimension) when summing
        flux in apertures. That is, each pixel is divided into
        ``subpixels ** 2`` subpixels.

    Returns
    -------
    flux : float or `~numpy.ndarray`
        Enclosed flux in annuli(s).
    fluxerr : float or `~numpy.ndarray`
        Uncertainty in flux values.
    """
    if (np.isscalar(xc) and np.isscalar(yc) and np.isscalar(a_in) and
        np.isscalar(a_out) and np.isscalar(b_out) and np.isscalar(theta)):
        aperture = EllipticalAnnulus(xc, yc, a_in, a_out, b_out, theta)
    else:
        xc, yc, a_in, a_out, b_out, theta = \
            np.broadcast_arrays(xc, yc, a_in, a_out, b_out, theta)
        aperture = np.empty(xc.shape, dtype=object)
        for index in np.ndindex(xc.shape):
            aperture[index] = EllipticalAnnulus(
                xc[index], yc[index], a_in[index], 
                a_out[index], b_out[index], theta[index])

    return aperture_photometry(arr, aperture, bkgerr=bkgerr, gain=gain,
                               mask=mask, subpixels=subpixels)
