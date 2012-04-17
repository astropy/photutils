# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""

import math
import abc

import numpy as np

__all__ = ["aperture_circular", "aperture_elliptical", "annulus_circular",
           "annulus_elliptical"]


class ApertureSet(object):
    """An abstract base class for sets of arbitrary apertures.

    Sets of apertures allow for multiple objects and multiple
    apertures associated with each object.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encloses(i, j, xx, yy):
        """Return a bool array representing which elements of an array
        are in an aperture.

        Inputs
        ------
        i : int
            Index of object
        j : int
            Index of aperture
        xx : 2-d `~numpy.ndarray`
            x coordinates of array elements
        yy : 2-d `~numpy.ndarray`
            y coordinates of array elements
        """
        return

    @abc.abstractmethod
    def extent(i):
        """Return extent of all apertures for a single object."""
        return

    @abc.abstractproperty
    def nobjects():
        """Number of objects"""
        return

    @abc.abstractproperty
    def napertures():
        """Number of apertures per object."""
        return

    @abc.abstractproperty
    def shape():
        """(napertures, nobjects)"""
        return

    def area(i, j):
        """Return area of aperture or None if too difficult to calculate.""" 
        return None

class EllipticalAnnulusSet(Aperture):
    """A class for representing elliptical annuli"""

    def __init__(self, xc, yc, a_in, a_out, b_out, theta):

        # Make xc and yc into 1-d arrays. For clarity to user, 
        # do not broadcast.
        self.xc = np.atleast_1d(xc)
        self.yc = np.atleast_1d(yc)
        if self.xc.ndim > 1 or self.yc.ndim > 1:
            raise ValueError('xc and yc can be at most 1-d.')
        if self.xc.shape != self.yc.shape:
            raise ValueError('Lengths of xc and yc must match')

        # Broadcast annulus parameters to same 2-d shape.
        a_in, a_out, b_out, theta = np.broadcast_arrays(a_in, a_out, 
                                                        b_out, theta)
        a_in, a_out, b_out, theta = np.atleast_2d(a_in, a_out, b_out, theta)

        # Check that trailing dimension is 1 or matches number of objects.
        if a_in.shape[1] not in [1, xc.shape[0]]:
            raise ValueError('Trailing dimension of annulus parameters'
                             ' must be 1 or match number of objects')

        # Expand trailing dimension of annulus parameters to match objects.
        self.a_in, self.a_out, self.b_out, self.theta, xc2d = \
            np.broadcast_arrays(a_in, a_out, b_out, theta, xc)

        # Check that a_out > a_in for all annuli.
        if not (self.a_out > self.a_in).all():
            raise ValueError('a_out must be greater than a_in.')

    def extent(self, i):
        # We can't easily do a tight bounding box because there is no
        # guarantee that the multiple ellipses for each object are
        # concentric.
        r = max(self.a_out[:, i].max(), self.b_out[:, i].max())
        x_min = self.xc[i] - r
        x_max = self.xc[i] + r
        y_min = self.yc[i] - r
        y_max = self.yc[i] + r
        return x_min, x_max, y_min, y_max

    def encloses(self, i, j, xx, yy):
        costheta = math.cos(self.theta[j, i])
        sintheta = math.sin(self.theta[j, i])
        b_in = (self.b_out[j, i] * self.a_in[j, i] /
                self.a_out[j, i])
        xx_off = xx - self.xc[j, i]
        yy_off = yy - self.yc[j, i]
        numerator1 = (xx_off * costheta - yy_off * sintheta)
        numerator2 = (yy_off * costheta + xx_off * sintheta)
        idx_out = ((numerator1 / self.a_out[j, i]) ** 2 +
                   (numerator2 / self.b_out[j, i]) ** 2) < 1.
        idx_in = ((numerator1 / self.a_in[j, i]) ** 2 +
                  (numerator2 / b_in) ** 2) > 1.
        return (idx_out & idx_in)

    def nobjects():
        return self.xc.shape[0]

    def napertures():
        return self.a_in.shape[0]

    def shape():
        return self.a_in.shape

    def area(i, j):
        return math.pi * (self.a_out[j, i] * self.b_out[j, i] -
                          self.a_in[j, i] ** 2 * self.b_out[j, i] /
                          self.a_out[j, i])
        

def aperture_photometry(arr, apertureset, bkgerr=None, gain=1., mask=None,
                        subpixels=5)
    """Sum flux within aperture(s).

    .. warning::
        Some options not yet implemented.

    Parameters
    ----------
    arr : array_like
        The 2-d array on which to perform photometry.
    apertureset: an ApertureSet-derived object
        Contains the definition of the apertures.
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

    # Check input array type and dimension.
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        raise TypeError('Complex type not supported')
    if arr.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(arr.ndim))

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

    # Test dimension of object coordinate arrays so we know how to return.
    scalar_input = False
    if np.isscalar(xc) and np.isscalar(yc):
        scalar_input = True

    # Initialize arrays to return.
    flux = np.zeros(apertureset.shape, dtype=np.float)
    if bkgvar is not None:
        fluxvar = np.zeros(apertureset.shape, dtype=np.float)

    for i in range(apertureset.nobjects):  # Loop over objects.
        # Maximum extent of all ellipses for this object.
        # We can't easily do a tight bounding box because there is no
        # guarantee that the multiple ellipses for each object are
        # concentric.
        x_min, x_max, y_min, y_max = apertureset.extent(i)
        x_min = int(x_min + 0.5)
        x_max = int(x_max + 1.5)
        y_min = int(y_min + 0.5)
        y_max = int(y_max + 1.5)

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

        # Get the distance of each pixel in the sub-array from the object
        # center, in units of the original image pixels.
        x_vals = np.arange(x_min - 0.5 + subpixelsize / 2.,
                           x_max - 0.5 + subpixelsize / 2.,
                           subpixelsize)
        y_vals = np.arange(y_min - 0.5 + subpixelsize / 2.,
                           y_max - 0.5 + subpixelsize / 2.,
                           subpixelsize)
        xx, yy = np.meshgrid(x_vals, y_vals)

        # Loop over annuli for this object.
        for j in range(apertureset.napertures):
            idx = apertureset.encloses(i, j, xx, yy)
            flux[j, i] = subarr[idx].sum() * subpixelarea
            if bkgvar is not None:
                if scalar_bkgvar:
                    if apertureset.area(i, j) is not None:
                        fluxvar[j, i] = (bkgvar * apertureset.area(i, j) +
                                         flux[j, i] / gain)
                    else:
                        fluxvar[j, i] = (bkgvar * idx.sum() * subpixelarea +
                                         flux[j, i] / gain)
                else:
                    fluxvar[j, i] = (bkgvar[idx].sum() * subpixelarea +
                                     flux[j, i] / gain)

    if bkgvar is not None:
        fluxerr = np.sqrt(fluxvar)  # Convert back to error.

    # If the input xc, yc were scalars (single object), AND we only had
    # a single aperture, then and only then can we return a scalar:
    if scalar_input and flux.shape == (1, 1):
        if bkgvar is None:
            return flux[0, 0]
        else:
            return flux[0, 0], fluxerr[0, 0]

    # Otherwise, if we only had a single aperture for each object,
    # then we can return a 1-d array.
    elif flux.shape[0] == 1:
        if bkgvar is None:
            return np.reshape(flux, flux.shape[1])
        else:
            return (np.reshape(flux, flux.shape[1]),
                    np.reshape(fluxerr, flux.shape[1]))

    # Otherwise we have multiple apertures, so we need to return a 2-d array.
    else:
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

    .. warning::
        Some options not yet implemented.

    Parameters
    ----------
    arr : array_like
        The 2-d array on which to perform photometry.
    xc, yc : float or list_like
        The x and y coordinates of the aperture center(s) in pixels.
        If dimension is greater than 1, the array is flattened. The
        coordinates are zero-indexed, meaning that ``(x, y) = (0., 0.)``
        corresponds to the center of the lower-left array element.
        The value of arr[0, 0] is taken as the value over the range
        ``-0.5 < x <= 0.5``, ``-0.5 < y <= 0.5``. The array is thus
        defined over the range ``-0.5 < x <= arr.shape[1] - 0.5``,
        ``-0.5 < y <= arr.shape[0] - 0.5``.
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
    maskmode : {'mirror', 'zero'}, optional
        What to do with masked pixels.  NOT YET IMPLEMENTED.
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
    pass
