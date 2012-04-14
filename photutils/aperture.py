# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""

import math

import numpy as np

__all__ = ["aperture_circular", "aperture_elliptical", "annulus_circular",
           "annulus_elliptical"]


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
                raise ValueError('must match shape of input array')

    # Test dimension of object coordinate arrays so we know how to return.
    scalar_input = False
    if np.isscalar(xc) and np.isscalar(yc):
        scalar_input = True

    # Make xc and yc into 1-d arrays. For clarity to user, do not broadcast.
    xc = np.atleast_1d(xc)
    yc = np.atleast_1d(yc)
    if xc.ndim > 1 or yc.ndim > 1:
        raise ValueError('xc and yc can be at most 1-d.')
    if xc.shape != yc.shape:
        raise ValueError('Lengths of xc and yc must match')

    # Broadcast annulus parameters to same 2-d shape.
    a_in, a_out, b_out, theta = np.broadcast_arrays(a_in, a_out, b_out, theta)
    a_in, a_out, b_out, theta = np.atleast_2d(a_in, a_out, b_out, theta)

    # Check that trailing dimension is 1 or matches number of objects.
    if a_in.shape[1] not in [1, xc.shape[0]]:
        raise ValueError('Trailing dimension of annulus parameters'
                         ' must be 1 or match number of objects')

    # Expand trailing dimension of annulus parameters to match objects.
    a_in, a_out, b_out, theta, xc2d = \
        np.broadcast_arrays(a_in, a_out, b_out, theta, xc)

    # Check that a_out > a_in for all annuli.
    if not (a_out > a_in).all():
        raise ValueError('a_out must be greater than a_in.')

    # Check that 'subpixels' is an int and is 1 or greater.
    subpixels = int(subpixels)
    if subpixels < 1:
        raise ValueError('an integer greater than 0 is required')
    subpixelsize = 1. / subpixels  # Size of subpixels in original pixels.
    subpixelarea = subpixelsize * subpixelsize

    # Initialize arrays to return.
    flux = np.zeros(a_out.shape, dtype=np.float)
    if bkgvar is not None:
        fluxvar = np.zeros(a_out.shape, dtype=np.float)

    for i in range(xc.shape[0]):  # Loop over objects.
        # Maximum extent of all ellipses for this object.
        # We can't easily do a tight bounding box because there is no
        # guarantee that the multiple ellipses for each object are
        # concentric.
        r = max(a_out[:, i].max(), b_out[:, i].max())
        x_min = int(xc[i] - r + 0.5)
        x_max = int(xc[i] + r + 1.5)
        y_min = int(yc[i] - r + 0.5)
        y_max = int(yc[i] + r + 1.5)

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

        # Get the subarray of background variance.
        if bkgvar is not None and not scalar_bkgvar:
            subbkgvar = bkgvar[y_min:y_max, x_min:x_max]
            if subpixels > 1:
                subbkgvar = np.repeat(np.repeat(subbkgvar, subpixels, axis=0),
                                      subpixels, axis=1)

        # Get the distance of each pixel in the sub-array from the object
        # center, in units of the original image pixels.
        x_dist = np.arange(x_min - 0.5 + subpixelsize / 2. - xc[i],
                           x_max - 0.5 + subpixelsize / 2. - xc[i],
                           subpixelsize)
        y_dist = np.arange(y_min - 0.5 + subpixelsize / 2. - yc[i],
                           y_max - 0.5 + subpixelsize / 2. - yc[i],
                           subpixelsize)
        xx, yy = np.meshgrid(x_dist, y_dist)

        # Loop over annuli for this object.
        for j in range(a_out.shape[0]):
            costheta = math.cos(theta[j, i])
            sintheta = math.sin(theta[j, i])
            b_in = (b_out[j, i] * a_in[j, i] /
                    a_out[j, i])
            idx_out = (((xx * costheta - yy * sintheta) /
                        a_out[j, i]) ** 2 +
                       ((yy * costheta + xx * sintheta) /
                        b_out[j, i]) ** 2) < 1.
            idx_in = (((xx * costheta - yy * sintheta) /
                        a_in[j, i]) ** 2 +
                       ((yy * costheta + xx * sintheta) / b_in) ** 2) > 1.
            idx = (idx_out & idx_in)

            # TODO: speed this up for apertures (non-annuli)??
            #       (don't compute idx_in; idx = idx_out)
            # TODO: speed up further for circles??
            #       (simpler formulae for idx)

            flux[j, i] = subarr[idx].sum() * subpixelarea
            if bkgvar is not None:
                if scalar_bkgvar:
                    fluxvar[j, i] = (
                        bkgvar * math.pi *
                        (a_out[j, i] * b_out[j, i] - a_in[j, i] * b_in) +
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
