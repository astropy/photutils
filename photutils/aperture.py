# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""

import math
import abc
import copy

import numpy as np

__all__ = ["CircularAperture", "CircularAnnulus",
           "EllipticalAperture", "EllipticalAnnulus",
           "aperture_photometry",
           "aperture_circular", "aperture_elliptical",
           "annulus_circular", "annulus_elliptical"]


class Aperture(object):
    """An abstract base class for an arbitrary 2-d aperture.

    Derived classes should contain whatever internal data is needed to define
    the aperture, and provide methods 'encloses' and 'extent' (and optionally,
    'area').
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extent():
        """Extent of aperture relative to object center.

        Returns
        -------
        x_min, x_max, y_min, y_max: float
        """
        return

    @abc.abstractmethod
    def encloses(xx, yy):
        """Return a bool array representing which elements of an array
        are in an aperture.

        Parameters
        ----------
        xx : `~numpy.ndarray`
            x coordinate of each element, relative to object center
        yy : `~numpy.ndarray`
            y coordinate of each element, relative to object center

        Returns
        -------
        in_aper : `~numpy.ndarray` (bool)
            2-d bool array indicating whether each element is within the
            aperture.
        """
        return


class CircularAperture(Aperture):
    """A circular aperture.

    Parameters
    ----------
    r : float
        The radius of the aperture.
    """

    def __init__(self, r):
        if not (r >= 0.):
            raise ValueError('r must be non-negative')
        self.r = r

    def extent(self):
        """Extent of aperture relative to object center.

        Returns
        -------
        x_min, x_max, y_min, y_max: float
        """

        return (-self.r, self.r, -self.r, self.r)

    def encloses(self, xx, yy):
        """Return a bool array representing which elements of an array
        are in the aperture.

        Parameters
        ----------
        xx : `~numpy.ndarray`
            x coordinate of each element, relative to object center
        yy : `~numpy.ndarray`
            y coordinate of each element, relative to object center

        Returns
        -------
        in_aper : `~numpy.ndarray` (bool)
            bool array indicating whether each element is within the
            aperture.
        """
        return xx * xx + yy * yy < self.r * self.r


    def encloses_exact(self, x_edges, y_edges):
        """
        Return a float array giving the fraction of each pixel covered
        by the aperture.

        Parameters
        ----------
        x_edges : `~numpy.ndarray`
            x coordinates of the pixel edges (1-d)
        y_edges : `~numpy.ndarray`
            y coordinates of the pixel edges (1-d)

        Returns
        -------
        overlap_area : `~numpy.ndarray` (float)
            2-d array of overlapping fraction. This has dimensions
            (len(y_edges) - 1, len(x_edges) - 1))
        """
        from circular_exact import circular_overlap_grid
        return circular_overlap_grid(x_edges, y_edges, self.r)

    def area(self):
        """Return area enclosed by aperture.

        Returns
        -------
        area : float
            Area in pixels enclosed by aperture.
        """
        return math.pi * self.r ** 2


class CircularAnnulus(Aperture):
    """A circular annulus aperture.

    Parameters
    ----------
    r_in : float
        The inner radius of the annulus.
    r_out : float
        The outer radius of the annulus.
    """

    def __init__(self, r_in, r_out):
        if not (r_out > r_in):
            raise ValueError('r_out must be greater than r_in')
        if not (r_in >= 0.):
            raise ValueError('r_in must be non-negative')
        self.r_in = r_in
        self.r_out = r_out

    def extent(self):
        """Extent of aperture relative to object center.

        Returns
        -------
        x_min, x_max, y_min, y_max: float
        """

        return (-self.r_out, self.r_out, -self.r_out, self.r_out)

    def encloses(self, xx, yy):
        """Return a bool array representing which elements of an array
        are in the aperture.

        Parameters
        ----------
        xx : `~numpy.ndarray`
            x coordinate of each element, relative to object center
        yy : `~numpy.ndarray`
            y coordinate of each element, relative to object center

        Returns
        -------
        in_aper : `~numpy.ndarray` (bool)
            bool array indicating whether each element is within the
            aperture.
        """
        dist_sq = (xx ** 2 + yy ** 2)
        return ((dist_sq < self.r_out ** 2) &
                (dist_sq > self.r_in ** 2))

    def area(self):
        """Return area enclosed by aperture.

        Returns
        -------
        area : float
            Area in pixels enclosed by aperture.
        """
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)


class EllipticalAperture(Aperture):
    """An elliptical aperture.

    Parameters
    ----------
    a : float
        The semimajor axis.
    b : float
        The semiminor axis.
    theta : float
        The position angle of the semimajor axis in radians.
    """

    def __init__(self, a, b, theta):
        if a < 0 or b < 0:
            raise ValueError('a and b must be nonnegative.')
        self.a = a
        self.b = b
        self.theta = theta

    def extent(self):
        """Extent of aperture relative to object center.

        Returns
        -------
        x_min, x_max, y_min, y_max: float
        """

        # TODO: determine a tighter bounding box (may not be worth
        #       the computation?)
        r = max(self.a, self.b)
        return (-r, r, -r, r)

    def encloses(self, xx, yy):
        """Return a bool array representing which elements of an array
        are in the aperture.

        Parameters
        ----------
        xx : `~numpy.ndarray`
            x coordinate of each element, relative to object center
        yy : `~numpy.ndarray`
            y coordinate of each element, relative to object center

        Returns
        -------
        in_aper : `~numpy.ndarray` (bool)
            bool array indicating whether each element is within the
            aperture.
        """

        if self.a == 0 or self.b == 0:
            return np.zeros(xx.shape, dtype=np.bool)
        numerator1 = (xx * math.cos(self.theta) -
                      yy * math.sin(self.theta))
        numerator2 = (yy * math.cos(self.theta) +
                      xx * math.sin(self.theta))
        return (((numerator1 / self.a) ** 2 +
                 (numerator2 / self.b) ** 2) < 1.)

    def area(self):
        """Return area enclosed by aperture.

        Returns
        -------
        area : float
            Area in pixels enclosed by aperture.
        """
        return math.pi * self.a * self.b


class EllipticalAnnulus(Aperture):
    """An elliptical annulus aperture.

    Parameters
    ----------
    a_in : float
        The inner semimajor axis.
    a_out : float
        The outer semimajor axis.
    b_out : float
        The outer semiminor axis. (The inner semiminor axis is determined
        by scaling by a_in/a_out.)
    theta : float
        The position angle of the semimajor axis in radians.

    """

    def __init__(self, a_in, a_out, b_out, theta):
        if not (a_out > a_in):
            raise ValueError('a_out must be greater than a_in')
        if a_in < 0 or b_out < 0:
            raise ValueError('a_in and b_out must be non-negative')
        self.a_in = a_in
        self.b_in = a_in * b_out / a_out
        self.a_out = a_out
        self.b_out = b_out
        self.theta = theta

    def extent(self):
        """Extent of aperture relative to object center.

        Returns
        -------
        x_min, x_max, y_min, y_max: float
        """

        # TODO: determine a tighter bounding box (may not be worth
        #       the computation?)
        r = max(self.a_out, self.b_out)
        return (-r, r, -r, r)

    def encloses(self, xx, yy):
        """Return a bool array representing which elements of an array
        are in the aperture.

        Parameters
        ----------
        xx : `~numpy.ndarray`
            x coordinate of each element, relative to object center
        yy : `~numpy.ndarray`
            y coordinate of each element, relative to object center

        Returns
        -------
        in_aper : `~numpy.ndarray` (bool)
            bool array indicating whether each element is within the
            aperture.
        """
        if self.a_out == 0 or self.b_out == 0:
            return np.zeros(xx.shape, dtype=np.bool)
        numerator1 = (xx * math.cos(self.theta) -
                      yy * math.sin(self.theta))
        numerator2 = (yy * math.cos(self.theta) +
                      xx * math.sin(self.theta))
        inside_outer_ellipse = ((numerator1 / self.a_out) ** 2 +
                                (numerator2 / self.b_out) ** 2) < 1.
        if self.a_in == 0 or self.b_in == 0:
            return inside_outer_ellipse
        outside_inner_ellipse = ((numerator1 / self.a_in) ** 2 +
                                 (numerator2 / self.b_in) ** 2) > 1.
        return (inside_outer_ellipse & outside_inner_ellipse)

    def area(self):
        """Return area enclosed by aperture.

        Returns
        -------
        area : float
            Area in pixels enclosed by aperture.
        """
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)


def aperture_photometry(data, xc, yc, apertures, error=None, gain=None,
                        mask=None, subpixels=5, pixelwise_errors=True):
    r"""Sum flux within aperture(s).

    Multiple objects and multiple apertures per object can be specified.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.
    xc, yc : float or list_like
        The x and y coordinates of the object center(s). If list_like,
        the lengths must match.
    apertures : `Aperture` object or array of `Aperture` objects

        The apertures to use for photometry. If an array (of at most 2
        dimensions), the trailing dimension of the array must be
        broadcastable to N_objects (= `len(xc)`). In  other words,
        the trailing dimension must be equal to either 1 or N_objects. The
        following shapes are thus allowed:

        `()` or `(1,)` or `(1, 1)`
            The same single aperture is applied to all objects.
        `(N_objects,)` or `(1, N_objects)`
            Each object gets its own single aperture.
        `(N_apertures, 1)`
            The same `N_aper` apertures are applied to all objects.
        `(N_apertures, N_objects)`
            Each object gets its own set of N_apertures apertures.

        Note that for subpixel sampling, the input array is only
        resampled once for each object.
    error : float or array_like, optional
        Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
    gain : float or array_like, optional
        Ratio of counts (e.g., electrons or photons) to units of the data
        (e.g., ADU), for the purpose of calculating Poisson error from the
        object itself. If `gain` is `None` (default), `error` is assumed to
        include all uncertainty in each pixel. If `gain` is given, `error`
        is assumed to be the "background error" only (not accounting for
        Poisson error in the flux in the apertures).
    mask : array_like (bool), optional
        Mask to apply to the data. The value of masked pixels are replaced
        by the value of the pixel mirrored across the center of the object,
        if available. If unavailable, the value is set to zero.
    subpixels : int or 'exact', optional
        Resample pixels by this factor (in each dimension) when summing
        flux in apertures. That is, each pixel is divided into
        `subpixels ** 2` subpixels. This can also be set to 'exact' to
        indicate that the exact overlap fraction should be used (this is
        only available for aperture classes that define the
        ``overlap_area`` method)
    pixelwise_errors : bool, optional
        For error and/or gain arrays. If True, assume error and/or gain
        vary significantly within an aperture: sum contribution from each
        pixel. If False, assume error and gain do not vary significantly
        within an aperture. Use the single value of error and/or gain at
        the center of each aperture as the value for the entire aperture.
        Default is True.

    Returns
    -------
    flux : float or `~numpy.ndarray`
        Enclosed flux in aperture(s). If `xc` and `yc` are floats and
        there is a single aperture, a float is returned. If xc, yc are
        list_like and there is a single aperture per object, a 1-d
        array is returned. If there are multiple apertures per object,
        a 2-d array is returned.
    fluxerr : float or `~numpy.ndarray`
        Uncertainty in flux values. Only returned if error is not `None`.
    """

    from .utils.downsample import downsample

    # Check input array type and dimension.
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))

    # Note whether xc, yc are scalars so we can try to return scalars later.
    scalar_obj_centers = np.isscalar(xc) and np.isscalar(yc)

    # Check shapes of xc, yc
    xc = np.atleast_1d(xc)
    yc = np.atleast_1d(yc)
    if xc.ndim > 1 or yc.ndim > 1:
        raise ValueError('Only 1-d arrays supported for object centers.')
    if xc.shape[0] != yc.shape[0]:
        raise ValueError('length of xc and yc must match')
    n_obj = xc.shape[0]

    # Check 'apertures' dimensions and type
    apertures = np.atleast_2d(apertures)
    if apertures.ndim > 2:
        raise ValueError('{0}-d aperture array not supported. '
                         'Only 2-d arrays supported.'.format(apertures.ndim))
    for aperture in apertures.ravel():
        if not isinstance(aperture, Aperture):
            raise TypeError("'aperture' must be an instance of Aperture.")
    n_aper = apertures.shape[0]

    # Check 'apertures' shape and expand trailing dimension to match N_obj
    # if necessary.
    if apertures.shape[1] not in [1, n_obj]:
        raise ValueError("trailing dimension of 'apertures' must be 1 or "
                         "match length of xc, yc")
    if apertures.shape[1] != n_obj:
        # We will not use xc2d. This is just for broadcasting 'apertures':
        apertures, xc2d = np.broadcast_arrays(apertures, xc)

    # Check whether we really need to calculate pixelwise errors, even if
    # requested. (If neither error nor gain is an array, we don't need to.)
    if ((error is None) or
        (np.isscalar(error) and gain is None) or
        (np.isscalar(error) and np.isscalar(gain))):
        pixelwise_errors = False

    # Check error shape.
    if error is not None:
        if np.isscalar(error):
            error, data = np.broadcast_arrays(error, data)
        if error.shape != data.shape:
            raise ValueError('shapes of error array and data array must'
                             ' match')

    # Check gain shape.
    if gain is not None:
        # Gain doesn't do anything without error set, so raise an exception.
        # (TODO: instead, should we just set gain = None and ignore it?)
        if error is None:
            raise ValueError('gain requires error')
        if np.isscalar(gain):
            gain, data = np.broadcast_arrays(gain, data)
        if gain.shape != data.shape:
            raise ValueError('shapes of gain array and data array must match')

    # Check mask shape and type.
    if mask is not None:
        mask = np.asarray(mask)
        if np.iscomplexobj(mask):
            raise TypeError('Complex type not supported')
        if mask.ndim != 2:
            raise ValueError('{0}-d array not supported. '
                             'Only 2-d arrays supported.'.format(mask.ndim))

    # Check that 'subpixels' is an int and is 1 or greater.
    if subpixels != 'exact':
        subpixels = int(subpixels)
        if subpixels < 1:
            raise ValueError('an integer greater than 0 is required')
        subpixelsize = 1. / subpixels  # Size of subpixels in original pixels.

    # Initialize arrays to return.
    flux = np.zeros(apertures.shape, dtype=np.float)
    if error is not None:
        fluxerr = np.zeros(apertures.shape, dtype=np.float)

    # 'extents' will hold the extent of all apertures for a given object.
    extents = np.empty((n_aper, 4), dtype=np.float)

    for i in range(n_obj):  # Loop over objects.

        # Fill 'extents' with extent of all apertures for this object.
        for j in range(n_aper):
            extents[j] = apertures[j, i].extent()

        # Set array index extents to encompass all apertures for this object.
        x_min = int(xc[i] + extents[:, 0].min() + 0.5)
        x_max = int(xc[i] + extents[:, 1].max() + 1.5)
        y_min = int(yc[i] + extents[:, 2].min() + 0.5)
        y_max = int(yc[i] + extents[:, 3].max() + 1.5)

        # Check that at least part of the sub-array is in the image.
        if (x_min >= data.shape[1] or x_max <= 0 or
            y_min >= data.shape[0] or y_max <= 0):
            # TODO: flag all the apertures for this object
            continue

        # Limit sub-array to be within the image.
        x_min = max(x_min, 0)
        x_max = min(x_max, data.shape[1])
        y_min = max(y_min, 0)
        y_max = min(y_max, data.shape[0])

        # Get the sub-array of the image and error
        subdata = data[y_min:y_max, x_min:x_max]
        if pixelwise_errors:
            subvariance = error[y_min:y_max, x_min:x_max] ** 2
            # If gain is specified, add poisson noise from the counts above
            # the background
            if gain is not None:
                subgain = gain[y_min:y_max, x_min:x_max]
                subvariance += subdata / subgain

        if mask is not None:
            submask = mask[y_min:y_max, x_min:x_max]  # Get sub-mask.

            # Get a copy of the data, because we will edit it
            subdata = copy.deepcopy(subdata)

            # Coordinates of masked pixels in sub-array.
            y_masked, x_masked = np.nonzero(submask)

            # Corresponding coordinates mirrored across xc, yc
            x_mirror = (2 * (xc[i] - x_min) - x_masked + 0.5).view('int32')
            y_mirror = (2 * (yc[i] - y_min) - y_masked + 0.5).view('int32')

            # reset pixels that go out of the image.
            outofimage = ((x_mirror < 0) |
                          (y_mirror < 0) |
                          (x_mirror >= subdata.shape[1]) |
                          (y_mirror >= subdata.shape[0]))
            if outofimage.any():
                x_mirror[outofimage] = x_masked[outofimage]
                y_mirror[outofimage] = y_masked[outofimage]

            # Replace masked pixel values.
            subdata[y_masked, x_masked] = subdata[y_mirror, x_mirror]
            if pixelwise_errors:
                subvariance[y_masked, x_masked] = \
                    subvariance[y_mirror, x_mirror]

            # Set pixels that mirrored to another masked pixel to zero.
            # This will also set to zero any pixels that mirrored out of
            # the image.
            mirror_is_masked = mask[y_mirror, x_mirror]
            x_bad = x_masked[mirror_is_masked]
            y_bad = y_masked[mirror_is_masked]
            subdata[y_bad, x_bad] = 0.
            if pixelwise_errors:
                subvariance[y_bad, x_bad] = 0.

        # In this case we just compute the position of the pixel 'walls' in x and y
        if subpixels == 'exact':

            x_edges = np.linspace(x_min - xc[i] - 0.5,
                                  x_max - xc[i] - 0.5,
                                  subdata.shape[0] + 1)
            y_edges = np.linspace(y_min - yc[i] - 0.5,
                                  y_max - yc[i] - 0.5,
                                  subdata.shape[1] + 1)

        else:

            x_vals = np.arange(x_min - xc[i] - 0.5 + subpixelsize / 2.,
                               x_max - xc[i] - 0.5 + subpixelsize / 2.,
                               subpixelsize)
            y_vals = np.arange(y_min - yc[i] - 0.5 + subpixelsize / 2.,
                               y_max - yc[i] - 0.5 + subpixelsize / 2.,
                               subpixelsize)

            xx, yy = np.meshgrid(x_vals, y_vals)

        # Loop over apertures for this object.
        for j in range(apertures.shape[0]):

            # Find fraction of overlap between aperture and pixels
            if subpixels == 'exact':
                try:
                    fraction = apertures[j, i].encloses_exact(x_edges, y_edges)
                except AttributeError:
                    raise Exception("subpixels='exact' cannot be used for "
                                    "{0:s}".format(apertures[j, i].__class__.__name__))
            else:
                in_aper = apertures[j, i].encloses(xx, yy).astype(float)
                fraction = downsample(in_aper, subpixels)

            # Sum the flux in those pixels and assign it to the output array.
            flux[j, i] = np.sum(subdata * fraction)

            if error is not None:  # If given, calculate error on flux.

                # If pixelwise, we have to do this the slow way.
                if pixelwise_errors:
                    fluxvar = np.sum(subvariance * fraction)

                # Otherwise, assume error and gain are constant over whole
                # aperture.
                else:
                    local_error = error[int(yc[i] + 0.5), int(xc[i] + 0.5)]
                    if hasattr(apertures[j, i], 'area'):
                        area = apertures[j, i].area()
                    else:
                        area = np.sum(fraction)
                    fluxvar = local_error ** 2 * area
                    if gain is not None:
                        local_gain = gain[int(yc[i] + 0.5), int(xc[i] + 0.5)]
                        fluxvar += flux[j, i] / local_gain

                # Make sure variance is > 0 when converting to st. dev.
                fluxerr[j, i] = math.sqrt(max(fluxvar, 0.))

    # If input coordinates were scalars, return scalars (if single aperture)
    if scalar_obj_centers and n_aper == 1:
        if error is None:
            return flux[0, 0]
        else:
            return flux[0, 0], fluxerr[0, 0]

    # If we only had a single aperture per object, we can return 1-d arrays
    if n_aper == 1:
        if error is None:
            return flux[0]
        else:
            return flux[0], fluxerr[0]

    # Otherwise, return 2-d array
    if error is None:
        return flux
    else:
        return flux, fluxerr


def aperture_circular(data, xc, yc, r, error=None, gain=None, mask=None,
                      subpixels='exact', pixelwise_errors=True):
    r"""Sum flux within circular apertures.

    Multiple objects and multiple apertures per object can be specified.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.
    xc, yc : float or list_like
        The x and y coordinates of the object center(s). If list_like,
        the lengths must match.
    r : float or array_like
        Radius of the aperture(s). If an array (of at most 2
        dimensions), the trailing dimension of the array must be
        broadcastable to N_objects (= `len(xc)`). In  other words,
        the trailing dimension must be equal to either 1 or N_objects. The
        following shapes are thus allowed:

        `()` or `(1,)` or `(1, 1)`
            The same single aperture is applied to all objects.
        `(N_objects,)` or `(1, N_objects)`
            Each object gets its own single aperture.
        `(N_apertures, 1)`
            The same `N_aper` apertures are applied to all objects.
        `(N_apertures, N_objects)`
            Each object gets its own set of N_apertures apertures.

    error : float or array_like, optional
        Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
    gain : float or array_like, optional
        Ratio of counts (e.g., electrons or photons) to units of the data
        (e.g., ADU), for the purpose of calculating Poisson error from the
        object itself. If `gain` is `None` (default), `error` is assumed to
        include all uncertainty in each pixel. If `gain` is given, `error`
        is assumed to be the "background error" only (not accounting for
        Poisson error in the flux in the apertures).
    mask : array_like (bool), optional
        Mask to apply to the data. The value of masked pixels are replaced
        by the value of the pixel mirrored across the center of the object,
        if available. If unavailable, the value is set to zero.
    subpixels : int, optional
        Resample pixels by this factor (in each dimension) when summing
        flux in apertures. That is, each pixel is divided into
        `subpixels ** 2` subpixels. This can also be set to 'exact' to
        indicate that the exact overlap fraction should be used.
    pixelwise_errors : bool, optional
        For error and/or gain arrays. If True, assume error and/or gain
        vary significantly within an aperture: sum contribution from each
        pixel. If False, assume error and gain do not vary significantly
        within an aperture. Use the single value of error and/or gain at
        the center of each aperture as the value for the entire aperture.
        Default is True.

    Returns
    -------
    flux : float or `~numpy.ndarray`
        Enclosed flux in aperture(s). If `xc` and `yc` are floats and
        there is a single aperture, a float is returned. If xc, yc are
        list_like and there is a single aperture per object, a 1-d
        array is returned. If there are multiple apertures per object,
        a 2-d array is returned.
    fluxerr : float or `~numpy.ndarray`
        Uncertainty in flux values. Only returned if error is not `None`.

    See Also
    --------
    aperture_photometry
    """
    r = np.asarray(r)
    apertures = np.empty(r.shape, dtype=object)
    for index in np.ndindex(r.shape):
        apertures[index] = CircularAperture(r[index])
    return aperture_photometry(data, xc, yc, apertures, error=error,
                               gain=gain, mask=mask, subpixels=subpixels,
                               pixelwise_errors=pixelwise_errors)


def aperture_elliptical(data, xc, yc, a, b, theta, error=None, gain=None,
                        mask=None, subpixels=5, pixelwise_errors=True):
    r"""Sum flux within elliptical apertures.

    Multiple objects and multiple apertures per object can be specified.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.
    xc, yc : float or list_like
        The x and y coordinates of the object center(s). If list_like,
        the lengths must match.
    a, b, theta : float or array_like
        The parameters of the aperture(s): respectively, the
        semimajor, semiminor axes in pixels and the position angle in
        radians. If an array (of at most 2 dimensions), the trailing
        dimension of the array must be broadcastable to N_objects (=
        `len(xc)`). In other words, the trailing dimension must be
        equal to either 1 or N_objects. The following shapes are thus
        allowed:

        `()` or `(1,)` or `(1, 1)`
            The same single aperture is applied to all objects.
        `(N_objects,)` or `(1, N_objects)`
            Each object gets its own single aperture.
        `(N_apertures, 1)`
            The same `N_aper` apertures are applied to all objects.
        `(N_apertures, N_objects)`
            Each object gets its own set of N_apertures apertures.

    error : float or array_like, optional
        Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
    gain : float or array_like, optional
        Ratio of counts (e.g., electrons or photons) to units of the data
        (e.g., ADU), for the purpose of calculating Poisson error from the
        object itself. If `gain` is `None` (default), `error` is assumed to
        include all uncertainty in each pixel. If `gain` is given, `error`
        is assumed to be the "background error" only (not accounting for
        Poisson error in the flux in the apertures).
    mask : array_like (bool), optional
        Mask to apply to the data. The value of masked pixels are replaced
        by the value of the pixel mirrored across the center of the object,
        if available. If unavailable, the value is set to zero.
    subpixels : int, optional
        Resample pixels by this factor (in each dimension) when summing
        flux in apertures. That is, each pixel is divided into
        `subpixels ** 2` subpixels.
    pixelwise_errors : bool, optional
        For error and/or gain arrays. If True, assume error and/or gain
        vary significantly within an aperture: sum contribution from each
        pixel. If False, assume error and gain do not vary significantly
        within an aperture. Use the single value of error and/or gain at
        the center of each aperture as the value for the entire aperture.
        Default is True.

    Returns
    -------
    flux : float or `~numpy.ndarray`
        Enclosed flux in aperture(s). If `xc` and `yc` are floats and
        there is a single aperture, a float is returned. If xc, yc are
        list_like and there is a single aperture per object, a 1-d
        array is returned. If there are multiple apertures per object,
        a 2-d array is returned.
    fluxerr : float or `~numpy.ndarray`
        Uncertainty in flux values. Only returned if error is not `None`.

    See Also
    --------
    aperture_photometry
    """

    a, b, theta = np.broadcast_arrays(a, b, theta)
    apertures = np.empty(a.shape, dtype=object)
    for index in np.ndindex(a.shape):
        apertures[index] = EllipticalAperture(a[index], b[index], theta[index])
    return aperture_photometry(data, xc, yc, apertures, error=error,
                               gain=gain, mask=mask, subpixels=subpixels,
                               pixelwise_errors=pixelwise_errors)


def annulus_circular(data, xc, yc, r_in, r_out, error=None, gain=None,
                     mask=None, subpixels=5, pixelwise_errors=True):
    r"""Sum flux within circular annuli.

    Multiple objects and multiple apertures per object can be specified.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.
    xc, yc : float or list_like
        The x and y coordinates of the object center(s). If list_like,
        the lengths must match.
    r_in, r_out : float or array_like
        The parameters of the annuli: respectively, the inner and
        outer radii. If an array (of at most 2 dimensions), the
        trailing dimension of the array must be broadcastable to
        N_objects (= `len(xc)`). In other words, the trailing
        dimension must be equal to either 1 or N_objects. The
        following shapes are thus allowed:

        `()` or `(1,)` or `(1, 1)`
            The same single aperture is applied to all objects.
        `(N_objects,)` or `(1, N_objects)`
            Each object gets its own single aperture.
        `(N_apertures, 1)`
            The same `N_aper` apertures are applied to all objects.
        `(N_apertures, N_objects)`
            Each object gets its own set of N_apertures apertures.

    error : float or array_like, optional
        Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
    gain : float or array_like, optional
        Ratio of counts (e.g., electrons or photons) to units of the data
        (e.g., ADU), for the purpose of calculating Poisson error from the
        object itself. If `gain` is `None` (default), `error` is assumed to
        include all uncertainty in each pixel. If `gain` is given, `error`
        is assumed to be the "background error" only (not accounting for
        Poisson error in the flux in the apertures).
    mask : array_like (bool), optional
        Mask to apply to the data. The value of masked pixels are replaced
        by the value of the pixel mirrored across the center of the object,
        if available. If unavailable, the value is set to zero.
    subpixels : int, optional
        Resample pixels by this factor (in each dimension) when summing
        flux in apertures. That is, each pixel is divided into
        `subpixels ** 2` subpixels.
    pixelwise_errors : bool, optional
        For error and/or gain arrays. If True, assume error and/or gain
        vary significantly within an aperture: sum contribution from each
        pixel. If False, assume error and gain do not vary significantly
        within an aperture. Use the single value of error and/or gain at
        the center of each aperture as the value for the entire aperture.
        Default is True.

    Returns
    -------
    flux : float or `~numpy.ndarray`
        Enclosed flux in aperture(s). If `xc` and `yc` are floats and
        there is a single aperture, a float is returned. If xc, yc are
        list_like and there is a single aperture per object, a 1-d
        array is returned. If there are multiple apertures per object,
        a 2-d array is returned.
    fluxerr : float or `~numpy.ndarray`
        Uncertainty in flux values. Only returned if error is not `None`.

    See Also
    --------
    aperture_photometry
    """

    r_in, r_out = np.broadcast_arrays(r_in, r_out)
    apertures = np.empty(r_in.shape, dtype=object)
    for index in np.ndindex(r_in.shape):
        apertures[index] = CircularAnnulus(r_in[index], r_out[index])
    return aperture_photometry(data, xc, yc, apertures, error=error,
                               gain=gain, mask=mask, subpixels=subpixels,
                               pixelwise_errors=pixelwise_errors)


def annulus_elliptical(data, xc, yc, a_in, a_out, b_out, theta,
                       error=None, gain=None, mask=None, subpixels=5,
                        pixelwise_errors=True):
    r"""Sum flux within elliptical annuli.

    Multiple objects and multiple apertures per object can be specified.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.
    xc, yc : float or list_like
        The x and y coordinates of the object center(s). If list_like,
        the lengths must match.
    a_in, a_out, b_out, theta : float or array_like
        The parameters of the annuli: respectively, the inner and
        outer semimajor axis in pixels, the outer semiminor axis in
        pixels, the position angle in radians. If an array (of at most
        2 dimensions), the trailing dimension of the array must be
        broadcastable to N_objects (= `len(xc)`). In other words, the
        trailing dimension must be equal to either 1 or N_objects. The
        following shapes are thus allowed:

        `()` or `(1,)` or `(1, 1)`
            The same single aperture is applied to all objects.
        `(N_objects,)` or `(1, N_objects)`
            Each object gets its own single aperture.
        `(N_apertures, 1)`
            The same `N_aper` apertures are applied to all objects.
        `(N_apertures, N_objects)`
            Each object gets its own set of N_apertures apertures.

    error : float or array_like, optional
        Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
    gain : float or array_like, optional
        Ratio of counts (e.g., electrons or photons) to units of the data
        (e.g., ADU), for the purpose of calculating Poisson error from the
        object itself. If `gain` is `None` (default), `error` is assumed to
        include all uncertainty in each pixel. If `gain` is given, `error`
        is assumed to be the "background error" only (not accounting for
        Poisson error in the flux in the apertures).
    mask : array_like (bool), optional
        Mask to apply to the data. The value of masked pixels are replaced
        by the value of the pixel mirrored across the center of the object,
        if available. If unavailable, the value is set to zero.
    subpixels : int, optional
        Resample pixels by this factor (in each dimension) when summing
        flux in apertures. That is, each pixel is divided into
        `subpixels ** 2` subpixels.
    pixelwise_errors : bool, optional
        For error and/or gain arrays. If True, assume error and/or gain
        vary significantly within an aperture: sum contribution from each
        pixel. If False, assume error and gain do not vary significantly
        within an aperture. Use the single value of error and/or gain at
        the center of each aperture as the value for the entire aperture.
        Default is True.

    Returns
    -------
    flux : float or `~numpy.ndarray`
        Enclosed flux in aperture(s). If `xc` and `yc` are floats and
        there is a single aperture, a float is returned. If xc, yc are
        list_like and there is a single aperture per object, a 1-d
        array is returned. If there are multiple apertures per object,
        a 2-d array is returned.
    fluxerr : float or `~numpy.ndarray`
        Uncertainty in flux values. Only returned if error is not `None`.

    See Also
    --------
    aperture_photometry
    """

    a_in, a_out, b_out, theta = \
        np.broadcast_arrays(a_in, a_out, b_out, theta)
    apertures = np.empty(a_in.shape, dtype=object)
    for index in np.ndindex(a_in.shape):
        apertures[index] = EllipticalAnnulus(a_in[index], a_out[index],
                                             b_out[index], theta[index])
    return aperture_photometry(data, xc, yc, apertures, error=error,
                               gain=gain, mask=mask, subpixels=subpixels,
                               pixelwise_errors=pixelwise_errors)
