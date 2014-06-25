# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import abc
import copy
import numpy as np
from astropy.table import Table
from astropy.extern import six
import warnings
from astropy.utils.exceptions import AstropyUserWarning
import astropy.units as u

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
        for x, y in self.positions:
            extents.append((int(x - self.r + 0.5), int(x + self.r + 1.5),
                            int(y - self.r + 0.5), int(y + self.r + 1.5)))

        return np.array(extents)

    def encloses(self):
        return

    def area(self):
        return math.pi * self.r ** 2

    def do_photometry(self, data, method='exact', subpixels=5):
        flux = np.zeros(len(self.positions), dtype=np.float)

        extents = self.extent()

        # Check if an aperture is fully out of data
        ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                                   extents[:, 1] <= 0)
        np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                      out=ood_filter)
        np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

        # TODO: flag these objects
        if np.sum(ood_filter):
            flux[ood_filter] = np.nan
            print("Position {0} and its aperture is out of the data"
                  " area".format(self.positions[ood_filter]))

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

        # Find fraction of overlap between aperture and pixels

        if method == 'center':
            for i in range(len(self.positions)):
                x_size = ((x_pmax[i] - x_pmin[i]) /
                          data[:, x_min[i]:x_max[i]].shape[1])
                y_size = ((y_pmax[i] - y_pmin[i]) /
                          data[y_min[i]:y_max[i], :].shape[0])

                x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                      x_pmax[i], x_size)
                y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                      y_pmax[i], y_size)
                xx, yy = np.meshgrid(x_centers, y_centers)
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 (xx * xx + yy * yy < self.r * self.r))

        elif method == 'subpixel':
            from .geometry import circular_overlap_grid
            for i in range(len(self.positions)):
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                       y_pmin[i], y_pmax[i],
                                                       x_pmax[i] - x_pmin[i],
                                                       y_pmax[i] - y_pmin[i],
                                                       self.r, 0, subpixels))

        elif method == 'exact':
            from .geometry import circular_overlap_grid
            for i in range(len(self.positions)):
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                       y_pmin[i], y_pmax[i],
                                                       x_pmax[i] - x_pmin[i],
                                                       y_pmax[i] - y_pmin[i],
                                                       self.r, 1, 1))
        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

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

    def do_photometry(self, data, method='exact', subpixels=5):
        flux = np.zeros(len(self.positions), dtype=np.float)

        extents = self.extent()
        for j in range(len(self.positions)):
            extent = extents[j]
            if (extent[0] >= data.shape[1] or extent[1] <= 0 or
                extent[2] >= data.shape[0] or extent[3] <= 0):

                # TODO: flag these objects
                flux[j] = np.nan
                print("Position {0} and its aperture is out of the data"
                      " area".format(self.positions[j]))
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

            photom_extent = extent - self._centers[j] - 0.5

            # Find fraction of overlap between aperture and pixels
            fraction = self.encloses(photom_extent, subdata.shape[1],
                                     subdata.shape[0],
                                     method=method, subpixels=subpixels)

            # Sum the flux in those pixels and assign it to the output array.
            flux[j] = np.sum(subdata * fraction)

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

    def encloses(self):
        return

    def area(self):
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)

    def do_photometry(self, data, method='exact', subpixels=5):
        flux = np.zeros(len(self.positions), dtype=np.float)

        extents = self.extent()

        # Check if an aperture is fully out of data
        ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                                   extents[:, 1] <= 0)
        np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                      out=ood_filter)
        np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

        # TODO: flag these objects
        if np.sum(ood_filter):
            flux[ood_filter] = np.nan
            print("Position {0} and its aperture is out of the data"
                  " area".format(self.positions[ood_filter]))

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

        # Find fraction of overlap between aperture and pixels

        if method == 'center':
            for i in range(len(self.positions)):
                x_size = ((x_pmax[i] - x_pmin[i]) /
                          data[:, x_min[i]:x_max[i]].shape[1])
                y_size = ((y_pmax[i] - y_pmin[i]) /
                          data[y_min[i]:y_max[i], :].shape[0])

                x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                      x_pmax[i], x_size)
                y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                      y_pmax[i], y_size)
                xx, yy = np.meshgrid(x_centers, y_centers)
                dist_sq = xx * xx + yy * yy
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 ((dist_sq < self.r_out * self.r_out) &
                                  (dist_sq > self.r_in * self.r_in)))

        elif method == 'subpixel':
            from .geometry import circular_overlap_grid
            for i in range(len(self.positions)):
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                       y_pmin[i], y_pmax[i],
                                                       x_pmax[i] - x_pmin[i],
                                                       y_pmax[i] - y_pmin[i],
                                                       self.r_out, 0, subpixels) -
                                 circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                       y_pmin[i], y_pmax[i],
                                                       x_pmax[i] - x_pmin[i],
                                                       y_pmax[i] - y_pmin[i],
                                                       self.r_in, 0, subpixels))

        elif method == 'exact':
            from .geometry import circular_overlap_grid
            for i in range(len(self.positions)):
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                       y_pmin[i], y_pmax[i],
                                                       x_pmax[i] - x_pmin[i],
                                                       y_pmax[i] - y_pmin[i],
                                                       self.r_out, 1, 1) -
                                 circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                       y_pmin[i], y_pmax[i],
                                                       x_pmax[i] - x_pmin[i],
                                                       y_pmax[i] - y_pmin[i],
                                                       self.r_in, 1, 1))
        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

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

    def encloses(self):
        return

    def area(self):
        return math.pi * self.a * self.b

    def do_photometry(self, data, method='exact', subpixels=5):
        flux = np.zeros(len(self.positions), dtype=np.float)

        # Shortcut to avoid divide-by-zero errors.
        if self.a == 0 or self.b == 0:
            return flux

        extents = self.extent()

        # Check if an aperture is fully out of data
        ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                                   extents[:, 1] <= 0)
        np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                      out=ood_filter)
        np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

        # TODO: flag these objects
        if np.sum(ood_filter):
            flux[ood_filter] = np.nan
            print("Position {0} and its aperture is out of the data"
                  " area".format(self.positions[ood_filter]))

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

        # Find fraction of overlap between aperture and pixels

        if method == 'center' or method == 'subpixel':
            if method == 'center': subpixels = 1
            for i in range(len(self.positions)):
                x_size = ((x_pmax[i] - x_pmin[i]) /
                          (data[:, x_min[i]:x_max[i]].shape[1] * subpixels))
                y_size = ((y_pmax[i] - y_pmin[i]) /
                          (data[y_min[i]:y_max[i], :].shape[0] * subpixels))

                x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                      x_pmax[i], x_size)
                y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                      y_pmax[i], y_size)
                xx, yy = np.meshgrid(x_centers, y_centers)
                numerator1 = (xx * math.cos(self.theta) +
                              yy * math.sin(self.theta))
                numerator2 = (yy * math.cos(self.theta) -
                              xx * math.sin(self.theta))

                in_aper = ((((numerator1 / self.a) ** 2 +
                             (numerator2 / self.b) ** 2) < 1.).astype(float)
                           / subpixels ** 2)

                if method == 'center':
                    flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                     in_aper)
                else:
                    from imageutils import downsample
                    flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                     downsample(in_aper, subpixels))

        elif method == 'exact':
            from .geometry import elliptical_overlap_grid
            for i in range(len(self.positions)):
                x_edges = np.linspace(x_pmin[i], x_pmax[i],
                                      data[:, x_min[i]:x_max[i]].shape[1] + 1)
                y_edges = np.linspace(y_pmin[i], y_pmax[i],
                                      data[y_min[i]:y_max[i], :].shape[0] + 1)
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 elliptical_overlap_grid(x_edges, y_edges,
                                                         self.a, self.b,
                                                         self.theta))

        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        return flux


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

    def encloses(self):
        return

    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def do_photometry(self, data, method='exact', subpixels=5):
        flux = np.zeros(len(self.positions), dtype=np.float)

        extents = self.extent()

        # Check if an aperture is fully out of data
        ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                                   extents[:, 1] <= 0)
        np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                      out=ood_filter)
        np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

        # TODO: flag these objects
        if np.sum(ood_filter):
            flux[ood_filter] = np.nan
            print("Position {0} and its aperture is out of the data"
                  " area".format(self.positions[ood_filter]))

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

        # Find fraction of overlap between aperture and pixels

        if method == 'center' or method == 'subpixel':
            if method == 'center': subpixels = 1
            for i in range(len(self.positions)):
                x_size = ((x_pmax[i] - x_pmin[i]) /
                          (data[:, x_min[i]:x_max[i]].shape[1] * subpixels))
                y_size = ((y_pmax[i] - y_pmin[i]) /
                          (data[y_min[i]:y_max[i], :].shape[0] * subpixels))

                x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                      x_pmax[i], x_size)
                y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                      y_pmax[i], y_size)
                xx, yy = np.meshgrid(x_centers, y_centers)
                numerator1 = (xx * math.cos(self.theta) +
                              yy * math.sin(self.theta))
                numerator2 = (yy * math.cos(self.theta) -
                              xx * math.sin(self.theta))

                in_aper = ((((numerator1 / self.a) ** 2 +
                             (numerator2 / self.b) ** 2) < 1.).astype(float)
                           / subpixels ** 2)

                if method == 'center':
                    flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                     in_aper)
                else:
                    from .utils import downsample
                    flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                     downsample(in_aper, subpixels))

        elif method == 'exact':
            from .elliptical_exact import elliptical_overlap_grid
            for i in range(len(self.positions)):
                x_edges = np.linspace(x_pmin[i], x_pmax[i],
                                      data[:, x_min[i]:x_max[i]].shape[1] + 1)
                y_edges = np.linspace(y_pmin[i], y_pmax[i],
                                      data[y_min[i]:y_max[i], :].shape[0] + 1)
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 elliptical_overlap_grid(x_edges, y_edges,
                                                         self.a, self.b,
                                                         self.theta))

        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

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
                from imageutils import downsample
                return downsample(in_aper, subpixels)

        elif method == 'exact':
            from .geometry import elliptical_overlap_grid
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


    def do_photometry(self, data, method='exact', subpixels=5):
        flux = np.zeros(len(self.positions), dtype=np.float)

        # Shortcut to avoid divide-by-zero errors.
        if self.a_out == 0 or self.b_out == 0:
            return flux

        extents = self.extent()

        # Check if an aperture is fully out of data
        ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                                   extents[:, 1] <= 0)
        np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                      out=ood_filter)
        np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

        # TODO: flag these objects
        if np.sum(ood_filter):
            flux[ood_filter] = np.nan
            print("Position {0} and its aperture is out of the data"
                  " area".format(self.positions[ood_filter]))

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

        # Find fraction of overlap between aperture and pixels

        if method == 'center' or method == 'subpixel':
            if method == 'center': subpixels = 1
            for i in range(len(self.positions)):
                x_size = ((x_pmax[i] - x_pmin[i]) /
                          (data[:, x_min[i]:x_max[i]].shape[1] * subpixels))
                y_size = ((y_pmax[i] - y_pmin[i]) /
                          (data[y_min[i]:y_max[i], :].shape[0] * subpixels))

                x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                      x_pmax[i], x_size)
                y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                      y_pmax[i], y_size)
                xx, yy = np.meshgrid(x_centers, y_centers)
                numerator1 = (xx * math.cos(self.theta) +
                              yy * math.sin(self.theta))
                numerator2 = (yy * math.cos(self.theta) -
                              xx * math.sin(self.theta))

                inside_outer_ellipse = ((((numerator1 / self.a_out) ** 2 +
                                          (numerator2 / self.b_out) ** 2) < 1.))
                if self.a_in == 0 or self.b_in == 0:
                    in_aper = inside_outer_ellipse.astype(float) / subpixels ** 2
                else:
                    outside_inner_ellipse = ((numerator1 / self.a_in) ** 2 +
                                             (numerator2 / self.b_in) ** 2) > 1.
                    in_aper = (inside_outer_ellipse &
                               outside_inner_ellipse).astype(float) / subpixels ** 2

                if method == 'center':
                    flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                     in_aper)
                else:
                    from .utils import downsample
                    flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                     downsample(in_aper, subpixels))

        elif method == 'exact':
            from .elliptical_exact import elliptical_overlap_grid
            for i in range(len(self.positions)):
                x_edges = np.linspace(x_pmin[i], x_pmax[i],
                                      data[:, x_min[i]:x_max[i]].shape[1] + 1)
                y_edges = np.linspace(y_pmin[i], y_pmax[i],
                                      data[y_min[i]:y_max[i], :].shape[0] + 1)
                flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                 (elliptical_overlap_grid(x_edges, y_edges,
                                                          self.a_in, self.b_in,
                                                          self.theta) -
                                  elliptical_overlap_grid(x_edges, y_edges,
                                                          self.a_in, self.b_in,
                                                          self.theta)))

        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

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
                from imageutils import downsample
                return downsample(in_aper, subpixels)

        elif method == 'exact':
            raise NotImplementedError('exact method not yet supported for '
                                      'RectangularAperture')
        else:
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

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


def aperture_photometry(data, positions, apertures, error=None, gain=None,
                        mask=None, method='exact', subpixels=5,
                        pixelwise_errors=True):
    """Sum flux within aperture(s)."""

    # Check input array type and dimension.
    data = u.Quantity(data)
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
        if isinstance(error, u.Quantity):
            if np.isscalar(error.value):
                error = u.Quantity(np.broadcast_arrays(error, data),
                                   unit=error.unit)[0]
        elif np.isscalar(error):
            error = np.broadcast_arrays(error, data)[0]

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

    # TODO check whether positions is wcs or pixel
    pixelpositions = positions

    if apertures[0] == 'circular':
        return CircularAperture(pixelpositions, apertures[1])\
            .do_photometry(data, method=method, subpixels=subpixels)
    elif apertures[0] == 'circular_annulus':
        return CircularAnnulus(pixelpositions, *apertures[1:3])\
            .do_photometry(data, method=method, subpixels=subpixels)
    elif apertures[0] == 'elliptical':
        return EllipticalAperture(pixelpositions, *apertures[1:4])\
            .do_photometry(data, method=method, subpixels=subpixels)
    elif apertures[0] == 'elliptical_annulus':
        return EllipticalAnnulus(pixelpositions, *apertures[1:5])\
            .do_photometry(data, method=method, subpixels=subpixels)


aperture_photometry.__doc__ = doc_template.format(
    desc=aperture_photometry.__doc__,
    args="""apertures : `~photutils.Aperture`
        The `~photutils.Aperture` object containing the apertures to use for
        photometry.""",
    seealso="")
