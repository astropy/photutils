# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""
from __future__ import division

import math
import abc
import copy

import numpy as np

__all__ = ["Aperture",
           "CircularAperture", "CircularAnnulus",
           "EllipticalAperture", "EllipticalAnnulus",
           "RectangularAperture",
           "aperture_photometry",
           "aperture_circular", "aperture_elliptical",
           "annulus_circular", "annulus_elliptical"]


class Aperture(object):
    """Abstract base class for an arbitrary 2-d aperture.

    Derived classes should contain whatever internal data is needed to define
    the aperture, and provide methods 'encloses' and 'extent' (and optionally,
    'area').
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extent(self):
        """Extent of aperture relative to object center.

        Returns
        -------
        x_min, x_max, y_min, y_max: float
            Extent of the aperture relative to object center.
        """
        return

    @abc.abstractmethod
    def encloses(self, x_min, x_max, y_min, y_max, nx, ny, method='center'):
        """Return a float array giving the fraction of each pixel covered
        by the aperture.

        Parameters
        ----------
        x_min, x_max : float
            x coordinates of outer edges of array, relative to object center.
        y_min, y_max : float
            y coordinates of outer edges of array, relative to object center.
        nx, ny : int
            dimensions of array
        method : str
            Which method to use for calculation. Available methods can
            differ between derived classes.

        Returns
        -------
        overlap_area : `~numpy.ndarray` (float)
            2-d array of shape (ny, nx) giving the fraction of each pixel
            covered by the aperture.
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
        return -self.r, self.r, -self.r, self.r


    def encloses(self, x_min, x_max, y_min, y_max, nx, ny,
                 method='exact', subpixels=5):
        if method == 'center':
            x_size = (x_max - x_min) / nx
            y_size = (y_max - y_min) / ny
            x_centers = np.arange(x_min + x_size / 2., x_max, x_size)
            y_centers = np.arange(y_min + y_size / 2., y_max, y_size)
            xx, yy = np.meshgrid(x_centers, y_centers)
            return xx * xx + yy * yy < self.r * self.r
        elif method == 'subpixel':
            from .circular_overlap import circular_overlap_grid
            return circular_overlap_grid(x_min, x_max, y_min, y_max, nx, ny,
                                         self.r, 0, subpixels)
        elif method == 'exact':
            from .circular_overlap import circular_overlap_grid
            return circular_overlap_grid(x_min, x_max, y_min, y_max, nx, ny,
                                         self.r, 1, 1)
        else:
            raise ValueError('{0} method not supported for aperture class {1}'
                             .format(method, self.__class__.__name__))


    def area(self):
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
        return (-self.r_out, self.r_out, -self.r_out, self.r_out)


    def encloses(self, x_min, x_max, y_min, y_max, nx, ny,
                 method='exact', subpixels=5):
        if method == 'center':
            x_size = (x_max - x_min) / nx
            y_size = (y_max - y_min) / ny
            x_centers = np.arange(x_min + x_size / 2., x_max, x_size)
            y_centers = np.arange(y_min + y_size / 2., y_max, y_size)
            xx, yy = np.meshgrid(x_centers, y_centers)
            dist_sq = xx * xx + yy * yy
            return (dist_sq < self.r_out * self.r_out) \
                & (dist_sq > self.r_in * self.r_in)
        elif method == 'subpixel':
            from .circular_overlap import circular_overlap_grid
            return (circular_overlap_grid(x_min, x_max, y_min, y_max, nx, ny,
                                          self.r_out, 0, subpixels) -
                    circular_overlap_grid(x_min, x_max, y_min, y_max, nx, ny,
                                          self.r_in, 0, subpixels))
        elif method == 'exact':
            from .circular_overlap import circular_overlap_grid
            return (circular_overlap_grid(x_min, x_max, y_min, y_max, nx, ny,
                                          self.r_out, 1, 1) -
                    circular_overlap_grid(x_min, x_max, y_min, y_max, nx, ny,
                                          self.r_in, 1, 1))
        else:
            raise ValueError('{0} method not supported for aperture class {1}'
                             .format(method, self.__class__.__name__))


    def area(self):
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
        The position angle of the semimajor axis in radians
        (counterclockwise).
    """

    def __init__(self, a, b, theta):
        if a < 0 or b < 0:
            raise ValueError('a and b must be nonnegative.')
        self.a = a
        self.b = b
        self.theta = theta

    def extent(self):
        r = max(self.a, self.b)
        return (-r, r, -r, r)


    def encloses(self, x_min, x_max, y_min, y_max, nx, ny,
                 method='subpixel', subpixels=5):

        # Shortcut to avoid divide-by-zero errors.
        if self.a == 0 or self.b == 0:
            return np.zeros((ny, nx), dtype=np.float)

        if method == 'center' or method == 'subpixel':
            if method == 'center': subpixels = 1
            x_size = (x_max - x_min) / (nx * subpixels)
            y_size = (y_max - y_min) / (ny * subpixels)
            x_centers = np.arange(x_min + x_size / 2., x_max, x_size)
            y_centers = np.arange(y_min + y_size / 2., y_max, y_size)
            xx, yy = np.meshgrid(x_centers, y_centers)
            numerator1 = (xx * math.cos(self.theta) +
                          yy * math.sin(self.theta))
            numerator2 = (yy * math.cos(self.theta) -
                          xx * math.sin(self.theta))
            in_aper = (((numerator1 / self.a) ** 2 +
                        (numerator2 / self.b) ** 2) < 1.).astype(float)
            if method == 'center':
                return in_aper
            else:
                from .utils.downsample import downsample
                return downsample(in_aper, subpixels)

        elif method == 'exact':
            from .elliptical_exact import elliptical_overlap_grid
            x_edges = np.linspace(x_min, x_max, nx + 1)
            y_edges = np.linspace(y_min, y_max, ny + 1)
            return elliptical_overlap_grid(x_edges, y_edges, self.a, self.b,
                                           self.theta)
        else:
            raise ValueError('{0} method not supported for aperture class {1}'
                             .format(method, self.__class__.__name__))


    def area(self):
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
        (counterclockwise).
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
        r = max(self.a_out, self.b_out)
        return (-r, r, -r, r)


    def encloses(self, x_min, x_max, y_min, y_max, nx, ny,
                 method='subpixel', subpixels=5):

        # Shortcut to avoid divide-by-zero errors.
        if self.a_out == 0 or self.b_out == 0:
            return np.zeros((ny, nx), dtype=np.float)

        if method == 'center' or method == 'subpixel':
            if method == 'center': subpixels = 1
            x_size = (x_max - x_min) / (nx * subpixels)
            y_size = (y_max - y_min) / (ny * subpixels)
            x_centers = np.arange(x_min + x_size / 2., x_max, x_size)
            y_centers = np.arange(y_min + y_size / 2., y_max, y_size)
            xx, yy = np.meshgrid(x_centers, y_centers)

            numerator1 = (xx * math.cos(self.theta) +
                          yy * math.sin(self.theta))
            numerator2 = (yy * math.cos(self.theta) -
                          xx * math.sin(self.theta))
            inside_outer_ellipse = ((numerator1 / self.a_out) ** 2 +
                                    (numerator2 / self.b_out) ** 2) < 1.
            if self.a_in == 0 or self.b_in == 0:
                in_aper = inside_outer_ellipse.astype(float)
            else:
                outside_inner_ellipse = ((numerator1 / self.a_in) ** 2 +
                                         (numerator2 / self.b_in) ** 2) > 1.
                in_aper = (inside_outer_ellipse &
                           outside_inner_ellipse).astype(float)

            if method == 'center':
                return in_aper
            else:
                from .utils.downsample import downsample
                return downsample(in_aper, subpixels)

        elif method == 'exact':
            from .elliptical_exact import elliptical_overlap_grid
            x_edges = np.linspace(x_min, x_max, nx + 1)
            y_edges = np.linspace(y_min, y_max, ny + 1)
            return (elliptical_overlap_grid(x_edges, y_edges, self.a_out,
                                            self.b_out, self.theta) -
                    elliptical_overlap_grid(x_edges, y_edges, self.a_in,
                                            self.b_in, self.theta))
        else:
            raise ValueError('{0} method not supported for aperture class {1}'
                             .format(method, self.__class__.__name__))


    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)


class RectangularAperture(Aperture):
    """A rectangular aperture.

    Parameters
    ----------
    w : float
        The full width of the aperture (at theta = 0, this is the "x" axis).
    h : float
        The full height of the aperture (at theta = 0, this is the "y" axis).
    theta : float
        The position angle of the semimajor axis in radians
        (counterclockwise).
    """

    def __init__(self, w, h, theta):
        if w < 0 or h < 0:
            raise ValueError('w and h must be nonnegative.')
        self.w = w
        self.h = h
        self.theta = theta


    def extent(self):
        r = max(self.h, self.w) * 2 ** -0.5
        #this is an overestimate by up to sqrt(2) unless theta = 45 deg
        return (-r, r, -r, r)


    def encloses(self, x_min, x_max, y_min, y_max, nx, ny,
                 method='subpixel', subpixels=5):

        # Shortcut to avoid divide-by-zero errors.
        if self.w == 0 or self.h == 0:
            return np.zeros((ny, nx), dtype=np.float)

        if method in ('center', 'subpixel'):
            if method == 'center':
                subpixels = 1

            x_size = (x_max - x_min) / (nx * subpixels)
            y_size = (y_max - y_min) / (ny * subpixels)

            x_centers = np.arange(x_min + x_size / 2., x_max, x_size)
            y_centers = np.arange(y_min + y_size / 2., y_max, y_size)

            xx, yy = np.meshgrid(x_centers, y_centers)

            newx = (xx * math.cos(self.theta) +
                    yy * math.sin(self.theta))
            newy = (yy * math.cos(self.theta) -
                    xx * math.sin(self.theta))

            halfw = self.w / 2
            halfh = self.h / 2
            in_aper = ((-halfw < newx) & (newx < halfw) &
                       (-halfh < newy) & (newy < halfh)).astype(float)

            if method == 'center':
                return in_aper
            else:
                from .utils.downsample import downsample
                return downsample(in_aper, subpixels)

        elif method == 'exact':
            raise NotImplementedError('exact method not yet supported for '
                                      'RectangularAperture')
        else:
            raise ValueError('{0} method not supported for aperture class {1}'
                             .format(method, self.__class__.__name__))

    def area(self):
        return self.w * self.h


doc_template = ("""\
    {desc}

    Multiple objects and multiple apertures per object can be specified.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.
    xc, yc : float or list_like
        The x and y coordinates of the object center(s). If list_like,
        the lengths must match.
    {args}
        If an array (of at most 2 dimensions), the trailing dimension
        of the array must be broadcastable to match ``xc`` and ``yc``.
        That is, the trailing dimension must be either 1 or
        ``len(xc)``. The following shapes are thus allowed:

        ``()`` or ``(1,)`` or ``(1, 1)``
            The same single aperture is applied to all objects.
        ``(N_objects,)`` or ``(1, N_objects)``
            Each object gets its own single aperture.
        ``(N_apertures, 1)``
            The same ``N_aper`` apertures are applied to all objects.
        ``(N_apertures, N_objects)``
            Each object gets its own set of ``N_apertures`` apertures.

        Note that for subpixel sampling, the input array is only
        resampled once for each object.
    error : float or array_like, optional
        Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
    gain : float or array_like, optional
        Ratio of counts (e.g., electrons or photons) to units of the data
        (e.g., ADU), for the purpose of calculating Poisson error from the
        object itself. If ``gain`` is `None` (default), ``error`` is assumed to
        include all uncertainty in each pixel. If ``gain`` is given, ``error``
        is assumed to be the "background error" only (not accounting for
        Poisson error in the flux in the apertures).
    mask : array_like (bool), optional
        Mask to apply to the data. The value of masked pixels are replaced
        by the value of the pixel mirrored across the center of the object,
        if available. If unavailable, the value is set to zero.
    method : str, optional
        Method to use for determining overlap between the aperture and pixels.
        Options include ['center', 'subpixel', 'exact'], but not all options
        are available for all types of apertures. More precise methods will
        generally be slower.

        'center'
            A pixel is considered to be entirely in or out of the aperture
            depending on whether its center is in or out of the aperture.
        'subpixel'
            A pixel is divided into subpixels and the center of each subpixel
            is tested (as above). With ``subpixels`` set to 1, this method is
            equivalent to 'center'.
        'exact' (default)
            The exact overlap between the aperture and each pixel is
            calculated.
    subpixels : int, optional
        For the 'subpixel' method, resample pixels by this factor (in
        each dimension). That is, each pixel is divided into
        ``subpixels ** 2`` subpixels.
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
        Enclosed flux in aperture(s). If ``xc`` and ``yc`` are floats and
        there is a single aperture, a float is returned. If ``xc``, ``yc`` are
        list_like and there is a single aperture per object, a 1-d
        array is returned. If there are multiple apertures per object,
        a 2-d array is returned.
    fluxerr : float or `~numpy.ndarray`
        Uncertainty in flux values. Only returned if error is not `None`.

    {seealso}
    """)

def aperture_photometry(data, xc, yc, apertures, error=None, gain=None,
                        mask=None, method='exact', subpixels=5,
                        pixelwise_errors=True):
    """Sum flux within aperture(s)."""

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
    if apertures.shape[1] != n_obj:
        raise ValueError("trailing dimension of 'apertures' must "
                         "match the length of xc, yc")

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
    if method == 'subpixel':
        subpixels = int(subpixels)
        if subpixels < 1:
            raise ValueError('subpixels: an integer greater than 0 is '
                             'required')

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
            mirror_is_masked = submask[y_mirror, x_mirror]
            x_bad = x_masked[mirror_is_masked]
            y_bad = y_masked[mirror_is_masked]
            subdata[y_bad, x_bad] = 0.
            if pixelwise_errors:
                subvariance[y_bad, x_bad] = 0.

        # Loop over apertures for this object.
        for j in range(apertures.shape[0]):

            # Find fraction of overlap between aperture and pixels
            fraction = apertures[j, i].encloses(
                x_min - xc[i] - 0.5, x_max - xc[i] - 0.5,
                y_min - yc[i] - 0.5, y_max - yc[i] - 0.5,
                subdata.shape[1], subdata.shape[0],
                method=method, subpixels=subpixels)

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

aperture_photometry.__doc__ = doc_template.format(
    desc=aperture_photometry.__doc__,
    args="""apertures : `~photutils.Aperture` or array thereof
        The apertures to use for photometry.""",
    seealso="")

def aperture_circular(data, xc, yc, r, error=None, gain=None, mask=None,
                      method='exact', subpixels=5, pixelwise_errors=True):
    """Sum flux within circular apertures."""

    r = np.asarray(r)
    if r.ndim > 0:
        apertures = np.empty(r.shape, dtype=object)
        for index in np.ndindex(*r.shape):
            apertures[index] = CircularAperture(r[index])
    else:
        apertures = CircularAperture(r)
    return aperture_photometry(data, xc, yc, apertures, error=error,
                               gain=gain, mask=mask, method=method,
                               subpixels=subpixels,
                               pixelwise_errors=pixelwise_errors)

aperture_circular.__doc__ = doc_template.format(
    desc=aperture_circular.__doc__,
    args="""r : float or array_like
        Radius of the aperture(s).""",
    seealso="""See Also
    --------
    aperture_photometry""")

def aperture_elliptical(data, xc, yc, a, b, theta, error=None, gain=None,
                        mask=None, method='exact', subpixels=5,
                        pixelwise_errors=True):
    """Sum flux within elliptical apertures."""

    a, b, theta = np.broadcast_arrays(a, b, theta)
    if a.ndim > 0:
        apertures = np.empty(a.shape, dtype=object)
        for index in np.ndindex(*a.shape):
            apertures[index] = EllipticalAperture(a[index], b[index], theta[index])
    else:
        apertures = EllipticalAperture(a, b, theta)
    return aperture_photometry(data, xc, yc, apertures, error=error,
                               gain=gain, mask=mask, method=method,
                               subpixels=subpixels,
                               pixelwise_errors=pixelwise_errors)

aperture_elliptical.__doc__ = doc_template.format(
    desc=aperture_elliptical.__doc__,
    args="""a, b, theta : float or array_like
        The parameters of the aperture(s): respectively, the semimajor,
        semiminor axes in pixels, and the position angle in radians
        (measured counterclockwise).""",
    seealso="""See Also
    --------
    aperture_photometry""")


def annulus_circular(data, xc, yc, r_in, r_out, error=None, gain=None,
                     mask=None, method='exact', subpixels=5,
                     pixelwise_errors=True):
    """Sum flux within circular annuli."""

    r_in, r_out = np.broadcast_arrays(r_in, r_out)
    if r_in.ndim > 0:
        apertures = np.empty(r_in.shape, dtype=object)
        for index in np.ndindex(*r_in.shape):
            apertures[index] = CircularAnnulus(r_in[index], r_out[index])
    else:
        apertures = CircularAnnulus(r_in, r_out)
    return aperture_photometry(data, xc, yc, apertures, error=error,
                               gain=gain, mask=mask, method=method,
                               subpixels=subpixels,
                               pixelwise_errors=pixelwise_errors)

annulus_circular.__doc__ = doc_template.format(
    desc=annulus_circular.__doc__,
    args="""r_in, r_out : float or array_like
        The parameters of the annuli: respectively, the inner and
        outer radii.""",
    seealso="""See Also
    --------
    aperture_photometry""")

def annulus_elliptical(data, xc, yc, a_in, a_out, b_out, theta,
                       error=None, gain=None, mask=None, method='exact',
                       subpixels=5, pixelwise_errors=True):
    """Sum flux within elliptical annuli."""

    a_in, a_out, b_out, theta = \
        np.broadcast_arrays(a_in, a_out, b_out, theta)
    if a_in.ndim > 0:
        apertures = np.empty(a_in.shape, dtype=object)
        for index in np.ndindex(*a_in.shape):
            apertures[index] = EllipticalAnnulus(a_in[index], a_out[index],
                                                 b_out[index], theta[index])
    else:
        apertures = EllipticalAnnulus(a_in, a_out, b_out, theta)
    return aperture_photometry(data, xc, yc, apertures, error=error,
                               gain=gain, mask=mask, method=method,
                               subpixels=subpixels,
                               pixelwise_errors=pixelwise_errors)

annulus_elliptical.__doc__ = doc_template.format(
    desc=annulus_elliptical.__doc__,
    args="""a_in, a_out, b_out, theta : float or array_like
        The parameters of the annuli: respectively, the inner and outer
        semimajor axis in pixels, the outer semiminor axis in pixels, and
        the position angle in radians (measured counterclockwise).""",
    seealso="""See Also
    --------
    aperture_photometry""")
