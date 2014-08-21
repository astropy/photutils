# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import abc
import numpy as np
import warnings
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.extern import six
from astropy.utils.exceptions import AstropyUserWarning
from .aperture_funcs import do_circular_photometry, do_elliptical_photometry, \
                            do_annulus_photometry

__all__ = ["Aperture",
           "CircularAperture", "CircularAnnulus",
           "EllipticalAperture", "EllipticalAnnulus",
           "RectangularAperture",
           "aperture_photometry"]


def _make_annulus_path(patch_inner, patch_outer):
    import matplotlib.path as mpath
    verts_inner = patch_inner.get_verts()
    verts_outer = patch_outer.get_verts()
    codes_inner = (np.ones(len(verts_inner), dtype=mpath.Path.code_type) *
                   mpath.Path.LINETO)
    codes_inner[0] = mpath.Path.MOVETO
    codes_outer = (np.ones(len(verts_outer), dtype=mpath.Path.code_type) *
                   mpath.Path.LINETO)
    codes_outer[0] = mpath.Path.MOVETO
    codes = np.concatenate((codes_inner, codes_outer))
    verts = np.concatenate((verts_inner, verts_outer[::-1]))
    return mpath.Path(verts, codes)


@six.add_metaclass(abc.ABCMeta)
class Aperture(object):
    """
    Abstract base class for an arbitrary 2-d aperture.

    Derived classes should contain whatever internal data is needed to
    define the aperture, and provide methods 'do_photometry' and 'extent'
    (and optionally, 'area').
    """

    @abc.abstractmethod
    def extent(self):
        """
        Extent of apertures. In the case when part of an aperture's extent
        falls out of the actual data region, the
        ``~photutils.Aperture.get_phot_extent`` method redefines the extent
        which has data coverage.

        Returns
        -------
        x_min, x_max, y_min, y_max: list of floats
            Extent of the apertures.
        """

    @abc.abstractmethod
    def plot(self, ax=None, fill=False, **kwargs):
        """
        Plot the aperture(s) on a matplotlib Axes instance.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` instance, optional
            If `None`, then the current ``Axes`` instance is used.

        fill : bool, optional
            Set whether to fill the aperture patch.  The default is
            `False`.

        kwargs
            Any keyword arguments accepted by `matplotlib.patches.Patch`.
        """

    @abc.abstractmethod
    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):
        """Sum flux within aperture(s)."""

    def get_phot_extents(self, data):
        """
        Get the photometry extents and check if the apertures is fully out
        of data.

        Parameters
        ----------
        data : array_like
            The 2-d array on which to perform photometry.

        Returns
        -------
        extents : dict
            The ``extents`` dictionary contains 3 elements:

            * ``'ood_filter'``
                A boolean array with `True` elements where the aperture is
                falling out of the data region.
            * ``'pixel_extent'``
                x_min, x_max, y_min, y_max : Refined extent of apertures with
                data coverage.
            * ``'phot_extent'``
                x_pmin, x_pmax, y_pmin, y_pmax: Extent centered to the 0, 0
                positions as required by the `~photutils.geometry` functions.

        """
        extents = self.extent()

        # Check if an aperture is fully out of data
        ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                                   extents[:, 1] <= 0)
        np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                      out=ood_filter)
        np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

        # TODO check whether it makes sense to have negative pixel
        # coordinate, one could imagine a stackes image where the reference
        # was a bit offset from some of the images? Or in those cases just
        # give Skycoord to the Aperture and it should deal with the
        # conversion for the actual case?
        x_min = np.maximum(extents[:, 0], 0)
        x_max = np.minimum(extents[:, 1], data.shape[1])
        y_min = np.maximum(extents[:, 2], 0)
        y_max = np.minimum(extents[:, 3], data.shape[0])

        x_pmin = x_min - self.positions[:, 0] - 0.5
        x_pmax = x_max - self.positions[:, 0] - 0.5
        y_pmin = y_min - self.positions[:, 1] - 0.5
        y_pmax = y_max - self.positions[:, 1] - 0.5

        # TODO: check whether any pixel is nan in data[y_min[i]:y_max[i],
        # x_min[i]:x_max[i])), if yes return something valid rather than nan

        return {'ood_filter': ood_filter,
                'pixel_extent': [x_min, x_max, y_min, y_max],
                'phot_extent': [x_pmin, x_pmax, y_pmin, y_pmax]}


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

        if isinstance(positions, u.Quantity):
            positions = positions.value
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
        for x, y in self.positions:
            extents.append((int(x - self.r + 0.5), int(x + self.r + 1.5),
                            int(y - self.r + 0.5), int(y + self.r + 1.5)))

        return np.array(extents)

    def area(self):
        return math.pi * self.r ** 2

    def plot(self, ax=None, fill=False, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        kwargs['fill'] = fill
        if ax is None:
            ax = plt.gca()
        for position in self.positions:
            patch = mpatches.Circle(position, self.r, **kwargs)
            ax.add_patch(patch)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):
        extents = super(CircularAperture, self).get_phot_extents(data)

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_circular_photometry(data, self.positions, extents,
                                      self.r, error=error, gain=gain,
                                      pixelwise_error=pixelwise_error,
                                      method=method,
                                      subpixels=subpixels)
        return flux


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

        if isinstance(positions, u.Quantity):
            positions = positions.value
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

    def area(self):
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):
        extents = super(CircularAnnulus, self).get_phot_extents(data)

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_annulus_photometry(data, self.positions, 'circular', extents,
                                     (self.r_in, ), (self.r_out, ),
                                     error=error, gain=gain,
                                     pixelwise_error=pixelwise_error,
                                     method=method,
                                     subpixels=subpixels)

        return flux

    def plot(self, ax=None, fill=False, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        kwargs['fill'] = fill
        if ax is None:
            ax = plt.gca()
        resolution = 20
        for position in self.positions:
            patch_inner = mpatches.CirclePolygon(position, self.r_in,
                                                 resolution=resolution)
            patch_outer = mpatches.CirclePolygon(position, self.r_out,
                                                 resolution=resolution)
            path = _make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)


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

        if isinstance(positions, u.Quantity):
            positions = positions.value
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

    def area(self):
        return math.pi * self.a * self.b

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):
        extents = super(EllipticalAperture, self).get_phot_extents(data)

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_elliptical_photometry(data, self.positions, extents,
                                        self.a, self.b, self.theta,
                                        error=error, gain=gain,
                                        pixelwise_error=pixelwise_error,
                                        method=method,
                                        subpixels=subpixels)
        return flux

    def plot(self, ax=None, fill=False, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        kwargs['fill'] = fill
        if ax is None:
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

        if isinstance(positions, u.Quantity):
            positions = positions.value
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

    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def plot(self, ax=None, fill=False, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        kwargs['fill'] = fill
        if ax is None:
            ax = plt.gca()
        theta_deg = self.theta * 180. / np.pi
        for position in self.positions:
            patch_inner = mpatches.Ellipse(position, self.a_in, self.b_in,
                                           theta_deg, **kwargs)
            patch_outer = mpatches.Ellipse(position, self.a_out, self.b_out,
                                           theta_deg, **kwargs)
            path = _make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):
        extents = super(EllipticalAnnulus, self).get_phot_extents(data)

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_annulus_photometry(data, self.positions, 'elliptical',
                                     extents,
                                     (self.a_in, self.b_in, self.theta),
                                     (self.a_out, self.b_out, self.theta),
                                     error=error, gain=gain,
                                     pixelwise_error=pixelwise_error,
                                     method=method, subpixels=subpixels)
        return flux


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

        if isinstance(positions, u.Quantity):
            positions = positions.value
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

    def area(self):
        return self.w * self.h

    def plot(self, ax=None, fill=False, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        kwargs['fill'] = fill
        if ax is None:
            ax = plt.gca()
        hw = self.w / 2.
        hh = self.h / 2.
        sint = math.sin(self.theta)
        cost = math.cos(self.theta)
        dx = (hh * sint) - (hw * cost)
        dy = -(hh * cost) - (hw * sint)
        positions = self.positions + np.array([dx, dy])
        theta_deg = self.theta * 180. / np.pi
        for position in positions:
            patch = mpatches.Rectangle(position, self.w, self.h, theta_deg,
                                       **kwargs)
            ax.add_patch(patch)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='subpixel', subpixels=5):

        extents = super(RectangularAperture, self).get_phot_extents(data)

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        ood_filter = extents['ood_filter']
        x_min, x_max, y_min, y_max = extents['pixel_extent']
        x_pmin, x_pmax, y_pmin, y_pmax = extents['phot_extent']

        flux = u.Quantity(np.zeros(len(self.positions), dtype=np.float),
                          unit=data.unit)

        # Check for invalid aperture
        if self.w == 0 or self.h == 0:
            return (flux, )

        # TODO: flag these objects
        if np.sum(ood_filter):
            flux[ood_filter] = np.nan
            warnings.warn("The aperture at position {0} does not have any "
                          "overlap with the data"
                          .format(self.positions[ood_filter]),
                          AstropyUserWarning)
            if np.sum(ood_filter) == len(self.positions):
                return (flux, )

        if error is not None:
            fluxvar = u.Quantity(np.zeros(len(self.positions), dtype=np.float),
                                 unit=error.unit ** 2)

        if method in ('center', 'subpixel'):
            if method == 'center': subpixels = 1
            if method == 'subpixel': from imageutils import downsample

            for i in range(len(flux)):
                x_size = ((x_pmax[i] - x_pmin[i]) /
                          (data[:, x_min[i]:x_max[i]].shape[1] * subpixels))
                y_size = ((y_pmax[i] - y_pmin[i]) /
                          (data[y_min[i]:y_max[i], :].shape[0] * subpixels))

                x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                      x_pmax[i], x_size)
                y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                      y_pmax[i], y_size)

                xx, yy = np.meshgrid(x_centers, y_centers)

                newx = (xx * math.cos(self.theta) +
                        yy * math.sin(self.theta))
                newy = (yy * math.cos(self.theta) -
                        xx * math.sin(self.theta))

                halfw = self.w / 2
                halfh = self.h / 2
                in_aper = (((-halfw < newx) & (newx < halfw) &
                            (-halfh < newy) & (newy < halfh)).astype(float)
                           / subpixels ** 2)

                if method == 'center':
                    if not np.isnan(flux[i]):
                        flux[i] = np.sum(data[y_min[i]:y_max[i],
                                              x_min[i]:x_max[i]] * in_aper)
                        if error is not None:
                            if pixelwise_error:
                                subvariance = error[y_min[i]:y_max[i],
                                                    x_min[i]:x_max[i]] ** 2
                                if gain is not None:
                                    subvariance += (data[y_min[i]:y_max[i],
                                                         x_min[i]:x_max[i]] /
                                                    gain[y_min[i]:y_max[i],
                                                         x_min[i]:x_max[i]])
                                # Make sure variance is > 0
                                fluxvar[i] = max(np.sum(subvariance * in_aper), 0)
                            else:
                                local_error = error[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                                    int((x_min[i] + x_max[i]) / 2 + 0.5)]
                                fluxvar[i] = max(local_error ** 2 * np.sum(in_aper), 0)
                                if gain is not None:
                                    local_gain = gain[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                                      int((x_min[i] + x_max[i]) / 2 + 0.5)]
                                    fluxvar[i] += flux[i] / local_gain
                else:
                    if not np.isnan(flux[i]):
                        if error is None:
                            flux[i] = np.sum(data[y_min[i]:y_max[i],
                                                  x_min[i]:x_max[i]] *
                                             downsample(in_aper, subpixels))
                        else:
                            fraction = downsample(in_aper, subpixels)
                            flux[i] = np.sum(data[y_min[i]:y_max[i],
                                                  x_min[i]:x_max[i]] * fraction)

                            if pixelwise_error:
                                subvariance = error[y_min[i]:y_max[i],
                                                    x_min[i]:x_max[i]] ** 2
                                if gain is not None:
                                    subvariance += (data[y_min[i]:y_max[i],
                                                         x_min[i]:x_max[i]] /
                                                    gain[y_min[i]:y_max[i],
                                                         x_min[i]:x_max[i]])
                                # Make sure variance is > 0
                                fluxvar[i] = max(np.sum(subvariance * fraction), 0)
                            else:
                                local_error = error[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                                    int((x_min[i] + x_max[i]) / 2 + 0.5)]
                                fluxvar[i] = max(local_error ** 2 * np.sum(fraction), 0)
                                if gain is not None:
                                    local_gain = gain[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                                      int((x_min[i] + x_max[i]) / 2 + 0.5)]
                                    fluxvar[i] += flux[i] / local_gain

        elif method == 'exact':
            raise NotImplementedError("'exact' method not yet supported for "
                                      "RectangularAperture")
        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        if error is None:
            return (flux, )
        else:
            return (flux, np.sqrt(fluxvar))


def aperture_photometry(data, positions, apertures, unit=None, wcs=None,
                        error=None, gain=None, mask=None, method='exact',
                        subpixels=5, pixelcoord=True, pixelwise_error=True,
                        mask_method='skip'):
    """
    Sum flux within an aperture at the given position(s).

    Parameters
    ----------
    data : array_like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
        The 2-d array on which to perform photometry. Units are used during
        the photometry, either provided along with the data array, or stored
        in the header keyword ``'BUNIT'``.
    positions : list, tuple, nd.array or `~astropy.coordinates.SkyCoord`
        Positions of the aperture centers, either in pixel or sky
        coordinates. If positions is `~astropy.coordinates.SkyCoord` or
        ``pixelcoord`` is `False` a wcs transformation is also needed to
        convert the input positions to pixel positions. If ``positions`` are
        sky positions but not an `~astropy.coordinates.SkyCoord` object, it
        need to be in the same celestial frame as the wcs transformation.
    apertures : tuple
        First element of the tuple is the mode, the currently supported ones
        are: ``'circular'``, ``'elliptical'``, ``'circular_annulus'``,
        ``'elliptical_annulus'``, ``'rectangular'``. The remaining (1 to 4)
        elements are the parameters for the given mode. Check the
        documentation of the relevant ``Aperture`` classes for more
        information.
    unit : `~astropy.units.UnitBase` instance, str
        An object that represents the unit associated with ``data``.  Must
        be an `~astropy.units.UnitBase` object or a string parseable by the
        :mod:`~astropy.units` package. An error is raised if ``data``
        already has a different unit.
    wcs : `~astropy.wcs.WCS`, optional
        Use this as the wcs transformation when either ``pixelcoord`` is
        `False` or ``positions`` is `~astropy.coordinates.SkyCoord`. It
        overrides any wcs transformation passed along with ``data`` either
        in the header or in an attribute.
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
        Mask to apply to the data.
    method : str, optional
        Method to use for determining overlap between the aperture and pixels.
        Options include ['center', 'subpixel', 'exact'], but not all options
        are available for all types of apertures. More precise methods will
        generally be slower.

        * ``'center'``
            A pixel is considered to be entirely in or out of the aperture
            depending on whether its center is in or out of the aperture.
        * ``'subpixel'``
            A pixel is divided into subpixels and the center of each
            subpixel is tested (as above). With ``subpixels`` set to 1, this
            method is equivalent to 'center'. Note that for subpixel
            sampling, the input array is only resampled once for each
            object.
        * ``'exact'`` (default)
            The exact overlap between the aperture and each pixel is
            calculated.
    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor (in
        each dimension). That is, each pixel is divided into
        ``subpixels ** 2`` subpixels.
    pixelcoord : bool, optional
        If `True` (default), assume ``positions`` are pixel positions. If
        `False`, assume the input positions are sky coordinates and uses the
        wcs transformation (provided either via ``wcs`` or along with
        ``data``) to convert them to pixel positions.
    pixelwise_error : bool, optional
        For error and/or gain arrays. If `True`, assume error and/or gain
        vary significantly within an aperture: sum contribution from each
        pixel. If `False`, assume error and gain do not vary significantly
        within an aperture. Use the single value of error and/or gain at
        the center of each aperture as the value for the entire aperture.
        Default is `True`.
    mask_method : str, optional
        Method to treat masked pixels. Currently supported methods:

        * ``'skip'``
            Leave out the masked pixels from all calculations.
        * ``'interpolation'``
            The value of the masked pixels are replaced by the mean value of
            the neighbouring non-masked pixels.

    Returns
    -------
    phot_table : `~astropy.table.Table`
        A table of the photometry with the following columns:

        * ``'aperture_sum'``: Sum of the values within the aperture.
        * ``'aperture_sum_err'``: Corresponding uncertainty in
          ``'aperture_sum'`` values.  Returned only if input ``error`` is not
          `None`.
        * ``'pixel_center'``: pixel coordinate pairs of the center of the
          apertures. Unit is pixel.
        * ``'input_center'``: input coordinate pairs as they were given in the
          ``positions`` parameter.

        The metadata of the table stores the version numbers of both astropy
        and photutils, as well as the calling arguments.
    aux_dict : dict
        Auxilary dictionary storing all the auxilary information
        available. The element are the following:

        * ``'apertures'``
            The `~photutils.Aperture` object containing the apertures to use
            during the photometry.

    """
    dataunit = None
    datamask = None
    wcs_transformation = wcs

    if isinstance(data, (fits.PrimaryHDU, fits.ImageHDU)):
        header = data.header
        data = data.data

        if 'BUNIT' in header:
            dataunit = header['BUNIT']

        # TODO check how a mask can be stored in the header, it seems like
        # full pixel masks are not supported by the FITS standard, look for
        # real life examples (e.g. header value stores the fits number of
        # fits extension where the pixelmask is stored?)
        if 'MASK' in header:
            datamask = header.mask

    elif isinstance(data, fits.HDUList):
        # TODO: do it in a 2d array, and thus get the light curves as a
        # side-product? Although it's not usual to store time series as
        # HDUList

        for i in range(len(data)):
            if data[i].data is not None:
                warnings.warn("Input data is a HDUList object, photometry is "
                              "onry run for the {0}. HDU."
                              .format(i), AstropyUserWarning)
                return aperture_photometry(data[i], positions, apertures, unit,
                                           wcs, error, gain, mask, method,
                                           subpixels, pixelcoord,
                                           pixelwise_error, mask_method)

    # this is basically for NDData inputs and alike
    elif hasattr(data, 'data') and not isinstance(data, np.ndarray):
        if data.wcs is not None and wcs_transformation is None:
            wcs_transformation = data.wcs
        datamask = data.mask

    if hasattr(data, 'unit'):
        dataunit = data.unit

    if unit is not None and dataunit is not None:
        if unit != dataunit:
            raise u.UnitsError('Unit of input data ({0}) and unit given by '
                               'unit argument ({1}) are not identical.'.
                               format(dataunit, unit))
        data = u.Quantity(data, unit=dataunit, copy=False)
    elif unit is None:
        if hasattr(data, 'unit'):
            data = u.Quantity(data, unit=dataunit, copy=False)
        else:
            data = u.Quantity(data, copy=False)
    else:
        data = u.Quantity(data, unit=unit, copy=False)

    if datamask is None:
        data.mask = datamask

    # Check input array type and dimension.
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))

    # Deal with the mask if it exist
    if mask is not None or datamask is not None:
        if mask is None:
            mask = datamask
        else:
            mask = np.asarray(mask)
            if np.iscomplexobj(mask):
                raise TypeError('Complex type not supported')
            if mask.ndim != 2:
                raise ValueError('{0}-d array not supported. '
                                 'Only 2-d arrays supported.'
                                 .format(mask.ndim))
            if mask.shape != data.shape:
                raise ValueError('Shapes of mask array and data array '
                                 'must match')

            if datamask is not None:
                mask *= datamask

        if mask_method == 'skip':
            data *= ~mask

        if mask_method == 'interpolation':
            for i, j in zip(*np.nonzero(mask)):
                y0, y1 = max(i - 1, 0), min(i + 2, data.shape[0])
                x0, x1 = max(j - 1, 0), min(j + 2, data.shape[1])
                data[i, j] = np.mean(data[y0:y1, x0:x1][~mask[y0:y1, x0:x1]])

    # Check whether we really need to calculate pixelwise errors, even if
    # requested. (If neither error nor gain is an array, we don't need to.)
    if ((error is None) or
        (np.isscalar(error) and gain is None) or
        (np.isscalar(error) and np.isscalar(gain))):
        pixelwise_error = False

    # Check error shape.
    if error is not None:
        if isinstance(error, u.Quantity):
            if np.isscalar(error.value):
                error = u.Quantity(np.broadcast_arrays(error, data),
                                   unit=error.unit)[0]
        elif np.isscalar(error):
            error = u.Quantity(np.broadcast_arrays(error, data),
                               unit=data.unit)[0]
        else:
            error = u.Quantity(error, unit=data.unit, copy=False)

        if error.shape != data.shape:
            raise ValueError('shapes of error array and data array must'
                             ' match')

    # Check gain shape.
    if gain is not None:
        # Gain doesn't do anything without error set, so raise an exception.
        # (TODO: instead, should we just set gain = None and ignore it?)
        if error is None:
            raise ValueError('gain requires error')

        if isinstance(gain, u.Quantity):
            if np.isscalar(gain.value):
                gain = u.Quantity(np.broadcast_arrays(gain, data),
                                  unit=gain.unit)[0]

        elif np.isscalar(gain):
            gain = np.broadcast_arrays(gain, data)[0]
        if gain.shape != data.shape:
            raise ValueError('shapes of gain array and data array must match')

    # Check that 'subpixels' is an int and is 1 or greater.
    if method == 'subpixel':
        subpixels = int(subpixels)
        if subpixels < 1:
            raise ValueError('subpixels: an integer greater than 0 is '
                             'required')

    if not pixelcoord or isinstance(positions, SkyCoord):

        from astropy.wcs import wcs

        try:
            from astropy.wcs.utils import wcs_to_celestial_frame
        except ImportError:  # Astropy < 1.0
            from .extern.wcs_utils import wcs_to_celestial_frame

        if wcs_transformation is None:
            wcs_transformation = wcs.WCS(header)

        # TODO this should be simplified once wcs_world2pix() supports
        # SkyCoord objects as input
        if isinstance(positions, SkyCoord):
            # Check which frame the wcs uses
            framename = wcs_to_celestial_frame(wcs_transformation).name
            frame = getattr(positions, framename)
            component_names = list(frame.representation_component_names.keys())[0:2]
            if len(positions.shape) > 0:
                positions_repr = u.Quantity(zip(getattr(frame,
                                                        component_names[0]).deg,
                                                getattr(frame,
                                                        component_names[1]).deg),
                                            unit=u.deg)
            else:
                positions_repr = (u.Quantity((getattr(frame,
                                                      component_names[0]).deg,
                                              getattr(frame,
                                                      component_names[1]).deg),
                                             unit=u.deg), )
        elif not isinstance(positions, u.Quantity):
            # TODO figure out the unit of the input positions for this case
            # TODO revise this once wcs_world2pix() accepts more input formats
            if len(positions) > 1 and not isinstance(positions, tuple):
                positions_repr = u.Quantity(positions, copy=False)
            else:
                positions_repr = (u.Quantity(positions, copy=False), )
        pixelpositions = u.Quantity(wcs_transformation.wcs_world2pix
                                    (positions_repr, 0), unit=u.pixel,
                                    copy=False)
    else:
        positions = u.Quantity(positions, unit=u.pixel, copy=False)
        pixelpositions = positions

    if apertures[0] == 'circular':
        ap = CircularAperture(pixelpositions.value, apertures[1])
    elif apertures[0] == 'circular_annulus':
        ap = CircularAnnulus(pixelpositions.value, *apertures[1:3])
    elif apertures[0] == 'elliptical':
        ap = EllipticalAperture(pixelpositions.value, *apertures[1:4])
    elif apertures[0] == 'elliptical_annulus':
        ap = EllipticalAnnulus(pixelpositions.value, *apertures[1:5])
    elif apertures[0] == 'rectangular':
        if method == 'exact':
            warnings.warn("'exact' method is not implemented, defaults to "
                          "'subpixel' instead", AstropyUserWarning)
            method = 'subpixel'
        ap = RectangularAperture(pixelpositions.value, *apertures[1:4])

    # Prepare version return data
    from astropy import __version__
    astropy_version = __version__

    from photutils import __version__
    photutils_version = __version__

    photometry_result = ap.do_photometry(data, method=method,
                                         subpixels=subpixels,
                                         error=error, gain=gain,
                                         pixelwise_error=pixelwise_error)
    if error is None:
        phot_col_names = ('aperture_sum', )
    else:
        phot_col_names = ('aperture_sum', 'aperture_sum_err')

    # Note: if wcs_transformation is None, 'pixel_center' will be the same
    # as 'input_center'

    # check whether single or multiple positions
    if len(pixelpositions) > 1 and pixelpositions[0].size >= 2:
        coord_columns = (pixelpositions, positions)
    else:
        coord_columns = ((pixelpositions,), (positions,))

    coord_col_names = ('pixel_center', 'input_center')

    return (Table(data=(photometry_result + coord_columns),
                  names=(phot_col_names + coord_col_names),
                  meta={'name': 'Aperture photometry results',
                        'version': 'astropy: {0}, photutils: {1}'
                        .format(astropy_version, photutils_version),
                        'calling_args': ('method={0}, subpixels={1}, '
                                         'error={2}, gain={3}, '
                                         'pixelwise_error={4}')
                        .format(method, subpixels, error is not None,
                                gain is not None, pixelwise_error)}),
            {'apertures': ap})
