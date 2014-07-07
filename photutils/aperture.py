# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import abc
import copy
import numpy as np
from astropy.extern import six
import warnings
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ["Aperture",
           "CircularAperture", "CircularAnnulus",
           "EllipticalAperture", "EllipticalAnnulus",
           "RectangularAperture",
           "aperture_photometry"]


@six.add_metaclass(abc.ABCMeta)
class Aperture(object):
    """
    Abstract base class for an arbitrary 2-d aperture.

    Derived classes should contain whatever internal data is needed to define
    the aperture, and provide methods 'encloses' and 'extent' (and optionally,
    'area').
    """

    @abc.abstractmethod
    def extent(self):
        """
        Extent of apertures. In the case when part of an aperture's extent
        falls out of the actual data region,
        `~photutils.aperture_photometry` calls the ``encloses()`` method of
        the aperture with a redefined extent which has data coverage.

        Returns
        -------
        x_min, x_max, y_min, y_max: list of floats
            Extent of the apertures.
        """

    @abc.abstractmethod
    def encloses(self, extent, nx, ny, method='center'):
        """
        Returns and array of float arrays giving the fraction of each pixel
        covered by the apertures.

        Parameters
        ----------
        extent : sequence of integers
            x_min, x_max, y_min, y_max extent of the aperture
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

    @abc.abstractmethod
    def plot(self, **kwargs):
        """
        Plot the aperture(s) on the current matplotlib Axes instance.

        Parameters
        ----------
        kwargs
            Any keyword arguments accepted by `matplotlib.patches.Patch`.
        """


class CircularAperture(Aperture):
    """
    Circular aperture(s).

    Parameters
    ----------
    positions : tuple, or list, or array
        Center coordinates of the apertures as list or array of (x, y)
        pixelcoordinates.
    r : float
        The radius of the aperture.
    """

    def __init__(self, positions, r):
        try:
            self.r = float(r)
        except TypeError:
            raise TypeError('r must be numeric, received {0}'.format(type(r)))

        if r < 0:
            raise ValueError('r must be non-negative')

        if isinstance(positions, (list, tuple, np.ndarray)):
            self.positions = np.atleast_2d(positions)
        else:
            raise TypeError("List or array of (x,y) pixel coordinates is "
                            "expected got '{0}'.".format(positions))

        if self.positions.ndim > 2:
            raise ValueError('{0}-d position array not supported. Only 2-d '
                             'arrays supported.'.format(self.positions.ndim))

    def extent(self):
        extents = []
        centers = []
        for x, y in self.positions:
            extents.append((int(x - self.r + 0.5), int(x + self.r + 1.5),
                            int(y - self.r + 0.5), int(y + self.r + 1.5)))
            centers.append((x, x, y, y))

        self._centers = np.array(centers)
        return np.array(extents)

    def encloses(self, extent, nx, ny, method='exact', subpixels=5):
        x_min, x_max, y_min, y_max = extent
        if method == 'center':
            x_size = (x_max - x_min) / nx
            y_size = (y_max - y_min) / ny
            x_centers = np.arange(x_min + x_size / 2., x_max, x_size)
            y_centers = np.arange(y_min + y_size / 2., y_max, y_size)
            xx, yy = np.meshgrid(x_centers, y_centers)
            return xx * xx + yy * yy < self.r * self.r
        elif method == 'subpixel':
            from .circular_overlap import circular_overlap_grid
            return circular_overlap_grid(x_min, x_max, y_min, y_max,
                                         nx, ny, self.r, 0, subpixels)
        elif method == 'exact':
            from .circular_overlap import circular_overlap_grid
            return circular_overlap_grid(x_min, x_max, y_min, y_max,
                                         nx, ny, self.r, 1, 1)
        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

    def area(self):
        return math.pi * self.r ** 2

    def plot(self, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        ax = plt.gca()
        for position in self.positions:
            patch = mpatches.Circle(position, self.r, **kwargs)
            ax.add_patch(patch)


class CircularAnnulus(Aperture):
    """
    Circular annulus aperture.

    Parameters
    ----------
    positions : tuple, or list, or array
        Center coordinates of the apertures as list or array of (x, y)
        pixelcoordinates.
    r_in : float
        The inner radius of the annulus.
    r_out : float
        The outer radius of the annulus.
    """

    def __init__(self, positions, r_in, r_out):
        try:
            self.r_in = r_in
            self.r_out = r_out
        except TypeError:
            raise TypeError("'r_in' and 'r_out' must be numeric, received {0} "
                            "and {1}".format((type(r_in), type(r_out))))

        if not (r_out > r_in):
            raise ValueError('r_out must be greater than r_in')
        if r_in < 0:
            raise ValueError('r_in must be non-negative')

        if isinstance(positions, (list, tuple, np.ndarray)):
            self.positions = np.atleast_2d(positions)
        else:
            raise TypeError("List or array of (x,y) pixel coordinates is "
                            "expected got '{0}'.".format(positions))

        if self.positions.ndim > 2:
            raise ValueError('{0}-d position array not supported. Only 2-d '
                             'arrays supported.'.format(self.positions.ndim))

    def extent(self):
        extents = []
        centers = []
        for x, y in self.positions:
            extents.append((int(x - self.r_out + 0.5),
                            int(x + self.r_out + 1.5),
                            int(y - self.r_out + 0.5),
                            int(y + self.r_out + 1.5)))
            centers.append((x, x, y, y))

        self._centers = np.array(centers)
        return np.array(extents)

    def encloses(self, extent, nx, ny, method='exact', subpixels=5):
        x_min, x_max, y_min, y_max = extent
        if method == 'center':
            x_size = (x_max - x_min) / nx
            y_size = (y_max - y_min) / ny
            x_centers = np.arange(x_min + x_size / 2., x_max, x_size)
            y_centers = np.arange(y_min + y_size / 2., y_max, y_size)
            xx, yy = np.meshgrid(x_centers, y_centers)
            dist_sq = xx * xx + yy * yy
            return ((dist_sq < self.r_out * self.r_out) &
                    (dist_sq > self.r_in * self.r_in))
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
    """
    An elliptical aperture.

    Parameters
    ----------
    positions : tuple, or list, or array
        Center coordinates of the apertures as list or array of (x, y)
        pixelcoordinates.
    a : float
        The semimajor axis.
    b : float
        The semiminor axis.
    theta : float
        The position angle of the semimajor axis in radians
        (counterclockwise).
    """

    def __init__(self, positions, a, b, theta):
        try:
            self.a = float(a)
            self.b = float(b)
            self.theta = float(theta)
        except TypeError:
            raise TypeError("'a' and 'b' and 'theta' must be numeric, received"
                            "{0} and {1} and {2}."
                            .format((type(a), type(b), type(theta))))

        if a < 0 or b < 0:
            raise ValueError("'a' and 'b' must be non-negative.")

        if isinstance(positions, (list, tuple, np.ndarray)):
            self.positions = np.atleast_2d(positions)
        else:
            raise TypeError("List or array of (x,y) pixel coordinates is "
                            "expected got '{0}'.".format(positions))

        if self.positions.ndim > 2:
            raise ValueError('{0}-d position array not supported. Only 2-d '
                             'arrays supported.'.format(self.positions.ndim))

    def extent(self):
        r = max(self.a, self.b)
        extents = []
        centers = []
        for x, y in self.positions:
            extents.append((int(x - r + 0.5), int(x + r + 1.5),
                            int(y - r + 0.5), int(y + r + 1.5)))
            centers.append((x, x, y, y))

        self._centers = np.array(centers)
        return np.array(extents)

    def encloses(self, extent, nx, ny, method='exact', subpixels=5):
        # Shortcut to avoid divide-by-zero errors.
        if self.a == 0 or self.b == 0:
            return np.zeros((ny, nx), dtype=np.float)

        x_min, x_max, y_min, y_max = extent
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
            in_aper /= subpixels ** 2    # conserve aperture area

            if method == 'center':
                return in_aper
            else:
                from .utils import downsample
                return downsample(in_aper, subpixels)

        elif method == 'exact':
            from .elliptical_exact import elliptical_overlap_grid
            x_edges = np.linspace(x_min, x_max, nx + 1)
            y_edges = np.linspace(y_min, y_max, ny + 1)
            return elliptical_overlap_grid(x_edges, y_edges, self.a,
                                           self.b, self.theta)
        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

    def area(self):
        return math.pi * self.a * self.b

    def plot(self, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        ax = plt.gca()
        theta_deg = self.theta * 180. / np.pi
        for position in self.positions:
            patch = mpatches.Ellipse(position, self.a, self.b, theta_deg,
                                     **kwargs)
            ax.add_patch(patch)


class EllipticalAnnulus(Aperture):
    """
    An elliptical annulus aperture.

    Parameters
    ----------
    positions : tuple, or list, or array
        Center coordinates of the apertures as list or array of (x, y)
        pixelcoordinates.
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

    def __init__(self, positions, a_in, a_out, b_out, theta):
        try:
            self.a_in = float(a_in)
            self.a_out = float(a_out)
            self.b_out = float(b_out)
            self.theta = float(theta)
        except TypeError:
            raise TypeError("'a_in' and 'a_out' and 'b_out' and 'theta' must "
                            "be numeric, received {0} and {1} and {2} and {3}."
                            .format((type(a_in), type(a_out),
                                     type(b_out), type(theta))))

        if not (a_out > a_in):
            raise ValueError("'a_out' must be greater than 'a_in'")
        if a_in < 0 or b_out < 0:
            raise ValueError("'a_in' and 'b_out' must be non-negative")

        self.b_in = a_in * b_out / a_out

        if isinstance(positions, (list, tuple, np.ndarray)):
            self.positions = np.atleast_2d(positions)
        else:
            raise TypeError("List or array of (x,y) pixel coordinates is "
                            "expected got '{0}'.".format(positions))

        if self.positions.ndim > 2:
            raise ValueError('{0}-d position array not supported. Only 2-d '
                             'arrays supported.'.format(self.positions.ndim))

    def extent(self):
        r = max(self.a_out, self.b_out)
        extents = []
        centers = []
        for x, y in self.positions:
            extents.append((int(x - r + 0.5), int(x + r + 1.5),
                            int(y - r + 0.5), int(y + r + 1.5)))
            centers.append((x, x, y, y))

        self._centers = np.array(centers)
        return np.array(extents)

    def encloses(self, extent, nx, ny, method='subpixel', subpixels=5):

        # Shortcut to avoid divide-by-zero errors.
        if self.a_out == 0 or self.b_out == 0:
            return np.zeros((ny, nx), dtype=np.float)

        x_min, x_max, y_min, y_max = extent
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
            in_aper /= subpixels ** 2    # conserve aperture area

            if method == 'center':
                return in_aper
            else:
                from .utils import downsample
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
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)


class RectangularAperture(Aperture):
    """
    A rectangular aperture.

    Parameters
    ----------
    positions : tuple, or list, or array
        Center coordinates of the apertures as list or array of (x, y)
        pixelcoordinates.
    w : float
        The full width of the aperture (at theta = 0, this is the "x" axis).
    h : float
        The full height of the aperture (at theta = 0, this is the "y" axis).
    theta : float
        The position angle of the semimajor axis in radians
        (counterclockwise).
    """

    def __init__(self, positions, w, h, theta):
        try:
            self.w = float(w)
            self.h = float(h)
            self.theta = float(theta)
        except TypeError:
            raise TypeError("'w' and 'h' and 'theta' must "
                            "be numeric, received {0} and {1} and {2}."
                            .format((type(w), type(h), type(theta))))
        if w < 0 or h < 0:
            raise ValueError("'w' and 'h' must be nonnegative.")

        if isinstance(positions, (list, tuple, np.ndarray)):
            self.positions = np.atleast_2d(positions)
        else:
            raise TypeError("List or array of (x,y) pixel coordinates is "
                            "expected got '{0}'.".format(positions))

        if self.positions.ndim > 2:
            raise ValueError('{0}-d position array not supported. Only 2-d '
                             'arrays supported.'.format(self.positions.ndim))

    def extent(self):
        r = max(self.h, self.w) * 2 ** -0.5
        # this is an overestimate by up to sqrt(2) unless theta = 45 deg
        extents = []
        centers = []
        for x, y in self.positions:
            extents.append((int(x - r + 0.5), int(x + r + 1.5),
                            int(y - r + 0.5), int(y + r + 1.5)))
            centers.append((x, x, y, y))

        self._centers = np.array(centers)
        return np.array(extents)

    def encloses(self, extent, nx, ny, method='subpixel', subpixels=5):

        # Shortcut to avoid divide-by-zero errors.
        if self.w == 0 or self.h == 0:
            return np.zeros((ny, nx), dtype=np.float)

        x_min, x_max, y_min, y_max = extent
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

            in_aper /= subpixels ** 2

            if method == 'center':
                return in_aper
            else:
                from .utils import downsample
                return downsample(in_aper, subpixels)

        elif method == 'exact':
            raise NotImplementedError('exact method not yet supported for '
                                      'RectangularAperture')
        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

    def area(self):
        return self.w * self.h

    def plot(self, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        ax = plt.gca()
        hw = self.w / 2.
        hh = self.h / 2.
        sint = math.sin(self.theta)
        cost = math.cos(self.theta)
        dx = (hh * sint) - (hw * cost)
        dy = -(hh * cost) - (hw * sint)
        positions = self.positions + np.array([dx, dy])
        angle = self.theta * 180. / np.pi
        for position in positions:
            patch = mpatches.Rectangle(position, self.w, self.h, angle,
                                       **kwargs)
            ax.add_patch(patch)


doc_template = ("""\
    {desc}

    Multiple objects and multiple apertures per object can be specified.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.
    {args}
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
        For error and/or gain arrays. If `True`, assume error and/or gain
        vary significantly within an aperture: sum contribution from each
        pixel. If False, assume error and gain do not vary significantly
        within an aperture. Use the single value of error and/or gain at
        the center of each aperture as the value for the entire aperture.
        Default is `True`.

    Returns
    -------
    flux : `~numpy.ndarray`
        Enclosed flux in aperture(s).
    fluxerr : `~numpy.ndarray`
        Uncertainty in flux values. Only returned if error is not `None`.

    {seealso}
    """)


def aperture_photometry(data, apertures, error=None, gain=None,
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
    flux = np.zeros(len(apertures.positions), dtype=np.float)
    if error is not None:
        fluxerr = np.zeros(len(apertures.positions), dtype=np.float)

    extents = apertures.extent()
    # Loop over apertures.
    for j in range(len(apertures.positions)):

        # Limit sub-array to be within the image.

        extent = extents[j]

        # Check that at least part of the sub-array is in the image.
        if (extent[0] >= data.shape[1] or extent[1] <= 0 or
            extent[2] >= data.shape[0] or extent[3] <= 0):

            # TODO: flag these objects
            flux[j] = np.nan
            warnings.warn('The aperture at position ({0}, {1}) does not have '
                          'any overlap with the data'
                          .format(apertures.positions[j][0],
                                  apertures.positions[j][1]),
                          AstropyUserWarning)
            continue

        # TODO check whether it makes sense to have negative pixel
        # coordinate, one could imagine a stackes image where the reference
        # was a bit offset from some of the images? Or in those cases just
        # give Skycoord to the Aperture and it should deal with the
        # conversion for the actual case?
        extent[0] = max(extent[0], 0)
        extent[1] = min(extent[1], data.shape[1])
        extent[2] = max(extent[2], 0)
        extent[3] = min(extent[3], data.shape[0])

        # Get the sub-array of the image and error
        subdata = data[extent[2]:extent[3], extent[0]:extent[1]]

        if pixelwise_errors:
            subvariance = error[extent[2]:extent[3], extent[0]:extent[1]] ** 2
            # If gain is specified, add poisson noise from the counts above
            # the background
            if gain is not None:
                subgain = gain[extent[2]:extent[3], extent[0]:extent[1]]
                subvariance += subdata / subgain

        if mask is not None:
            submask = mask[extent[2]:extent[3], extent[0]:extent[1]]

            # Get a copy of the data, because we will edit it
            subdata = copy.deepcopy(subdata)

            # Coordinates of masked pixels in sub-array.
            y_masked, x_masked = np.nonzero(submask)

            # Corresponding coordinates mirrored across xc, yc
            x_mirror = (2 * (apertures.positions[j][0] - extent[0])
                        - x_masked + 0.5).astype('int32')
            y_mirror = (2 * (apertures.positions[j][1] - extent[2])
                        - y_masked + 0.5).astype('int32')

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

        # To keep the aperture reusable, define new extent for the actual
        # photometry rather than overwrite the aperture's extents parameter

        photom_extent = extent - apertures._centers[j] - 0.5

        # Find fraction of overlap between aperture and pixels
        fraction = apertures.encloses(photom_extent, subdata.shape[1],
                                      subdata.shape[0],
                                      method=method, subpixels=subpixels)

        # Sum the flux in those pixels and assign it to the output array.
        flux[j] = np.sum(subdata * fraction)

        if error is not None:  # If given, calculate error on flux.

            # If pixelwise, we have to do this the slow way.
            if pixelwise_errors:
                fluxvar = np.sum(subvariance * fraction)

                # Otherwise, assume error and gain are constant over whole
                # aperture.
            else:
                local_error = error[int((extent[2] + extent[3]) / 2 + 0.5),
                                    int((extent[0] + extent[1]) / 2 + 0.5)]
                if hasattr(apertures, 'area'):
                    area = apertures.area()
                else:
                    area = np.sum(fraction)
                fluxvar = local_error ** 2 * area
                if gain is not None:
                    local_gain = gain[int((extent[2] + extent[3]) / 2 + 0.5),
                                      int((extent[0] + extent[1]) / 2 + 0.5)]
                    fluxvar += flux[j] / local_gain

            # Make sure variance is > 0 when converting to st. dev.
            fluxerr[j] = math.sqrt(max(fluxvar, 0.))

    if error is None:
        return flux
    else:
        return flux, fluxerr

aperture_photometry.__doc__ = doc_template.format(
    desc=aperture_photometry.__doc__,
    args="""apertures : `~photutils.Aperture`
        The `~photutils.Aperture` object containing the apertures to use for
        photometry.""",
    seealso="")
