# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import abc
import numpy as np
import copy
import warnings
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.extern import six
from astropy.utils.misc import InheritDocstrings
from astropy.utils.exceptions import AstropyUserWarning
from .aperture_funcs import (do_circular_photometry, do_elliptical_photometry,
                             do_rectangular_photometry)
from .wcsutils import (skycoord_to_pixel, skycoord_to_pixel_scale_angle,
                       assert_angle_or_pixel, assert_angle)

__all__ = ['Aperture', 'SkyAperture', 'PixelAperture',
           'SkyCircularAperture', 'CircularAperture',
           'SkyCircularAnnulus', 'CircularAnnulus',
           'SkyEllipticalAperture', 'EllipticalAperture',
           'SkyEllipticalAnnulus', 'EllipticalAnnulus',
           'RectangularAperture', 'RectangularAnnulus',
           'aperture_photometry']


def _sanitize_pixel_positions(positions):

    if isinstance(positions, u.Quantity):
        if positions.unit is u.pixel:
            positions = positions.value
        else:
            raise u.UnitsError("positions should be in pixel units")

    if isinstance(positions, u.Quantity):
        positions = positions.value
    elif isinstance(positions, (list, tuple, np.ndarray)):
        positions = np.atleast_2d(positions)
    elif isinstance(positions, zip):
        # This is needed for zip to work seamlessly in Python 3
        positions = np.atleast_2d(list(positions))
    else:
        raise TypeError("List or array of (x,y) pixel coordinates is "
                        "expected got '{0}'.".format(positions))

    if positions.ndim > 2:
        raise ValueError('{0}-d position array not supported. Only 2-d '
                         'arrays supported.'.format(positions.ndim))

    return positions


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


class _ABCMetaAndInheritDocstrings(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(_ABCMetaAndInheritDocstrings)
class Aperture(object):
    """
    Abstract base class for all apertures.
    """


@six.add_metaclass(_ABCMetaAndInheritDocstrings)
class SkyAperture(Aperture):
    """
    Abstract base class for 2-d apertures defined in celestial coordinates.
    """


@six.add_metaclass(_ABCMetaAndInheritDocstrings)
class PixelAperture(Aperture):
    """
    Abstract base class for 2-d apertures defined in pixel coordinates.

    Derived classes should contain whatever internal data is needed to
    define the aperture, and provide the method `do_photometry` (and
    optionally, ``area``).
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
        """Sum flux within aperture(s).

        Parameters
        ----------
        data : array_like
            The 2-d array on which to perform photometry.
        error : array_like, optional
            Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
            ``error`` has to have the same shape as ``data``.
        gain : array_like, optional
            Ratio of counts (e.g., electrons or photons) to units of the
            data (e.g., ADU), for the purpose of calculating Poisson error from
            the object itself. ``gain`` has to have the same shape as ``data``.
        pixelwise_error : bool, optional
            For error and/or gain arrays. If `True`, assume error and/or gain
            vary significantly within an aperture: sum contribution from each
            pixel. If `False`, assume error and gain do not vary significantly
            within an aperture.
        method : str, optional
            Method to use for determining overlap between the aperture
            and pixels.  Options include ['center', 'subpixel',
            'exact'], but not all options are available for all types of
            apertures. More precise methods will generally be slower.

            * ``'center'``
                A pixel is considered to be entirely in or out of the
                aperture depending on whether its center is in or out of
                the aperture.

            * ``'subpixel'``
                A pixel is divided into subpixels and the center of each
                subpixel is tested (as above). With ``subpixels`` set to
                1, this method is equivalent to 'center'. Note that for
                subpixel sampling, the input array is only resampled
                once for each object.

            * ``'exact'`` (default)
                The exact overlap between the aperture and each pixel is
                calculated.
        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor (in
            each dimension). That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Sum of the values withint the aperture(s). Unit is kept to be the
            unit of the input ``data``.

        fluxvar :  `~astropy.units.Quantity`
            Corresponting uncertainity in the ``flux`` values. Returned only
            if input ``error`` is not `None`.
        """

    def area():
        """
        Area of aperture.

        Returns
        -------
        area : float
            Area of aperture.
        """


class SkyCircularAperture(SkyAperture):
    """
    Circular aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.
    r : `~astropy.units.Quantity`
        The radius of the aperture(s), either in angular or pixel units.
    """

    def __init__(self, positions, r):

        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('r', r)
        self.r = r

    def to_pixel(self, wcs):
        """
        Return a CircularAperture instance in pixel coordinates.
        """

        if self.r.unit.physical_type == 'angle':
            x, y, scale, angle = skycoord_to_pixel_scale_angle(self.positions, wcs)
            # TODO: no need to use the mean once we support arrays of aperture values
            r = (np.mean(scale) * self.r).to(u.pixel).value
        else:  # pixel
            x, y = skycoord_to_pixel(self.positions, wcs)
            r = self.r.value

        pixel_positions = np.array([x, y]).transpose()

        return CircularAperture(pixel_positions, r)


class CircularAperture(PixelAperture):
    """
    Circular aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` Numpy
        array, or an ``Nx2`` `~astropy.units.Quantity` in units of pixels.
    r : float
        The radius of the aperture(s), in pixels.

    Raises
    ------
    ValueError : `~.exceptions.ValueError`
        If the radius is negative.
    """

    def __init__(self, positions, r):

        try:
            self.r = float(r)
        except TypeError:
            raise TypeError('r must be numeric, received {0}'.format(type(r)))

        if r < 0:
            raise ValueError('r must be non-negative')

        self.positions = _sanitize_pixel_positions(positions)

    def area(self):
        return math.pi * self.r ** 2

    def plot(self, ax=None, fill=False, source_id=None, **kwargs):

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        kwargs['fill'] = fill

        if ax is None:
            ax = plt.gca()
        if source_id is None:
            positions = self.positions
        else:
            positions = self.positions[np.atleast_1d(source_id)]

        for position in positions:
            patch = mpatches.Circle(position, self.r, **kwargs)
            ax.add_patch(patch)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_circular_photometry(data, self.positions,
                                      self.r, error=error, gain=gain,
                                      pixelwise_error=pixelwise_error,
                                      method=method,
                                      subpixels=subpixels)
        return flux


class SkyCircularAnnulus(SkyAperture):
    """
    Circular annulus aperture(s), defined in sky coordinates.

    Parameters
    ----------

    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.
    r : `~astropy.units.Quantity`
        The radius of the aperture(s), either in angular or pixel units.
    """

    def __init__(self, positions, r_in, r_out):

        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('r_in', r_in)
        assert_angle_or_pixel('r_out', r_out)

        if r_in.unit.physical_type != r_out.unit.physical_type:
            raise ValueError("r_in and r_out should either both be angles or in pixels")

        self.r_in = r_in
        self.r_out = r_out

    def to_pixel(self, wcs):
        """
        Return a CircularAnnulus instance in pixel coordinates.
        """

        if self.r_in.unit.physical_type == 'angle':
            x, y, scale, angle = skycoord_to_pixel_scale_angle(self.positions, wcs)
            # TODO: no need to use the mean once we support arrays of aperture values
            r_in = (np.mean(scale) * self.r_in).to(u.pixel).value
            r_out = (np.mean(scale) * self.r_out).to(u.pixel).value
        else:  # pixel
            x, y = skycoord_to_pixel(self.positions, wcs)
            r_in = self.r_in.value
            r_out = self.r_out.value

        pixel_positions = np.array([x, y]).transpose()

        return CircularAnnulus(pixel_positions, r_in, r_out)


class CircularAnnulus(PixelAperture):
    """
    Circular annulus aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` Numpy
        array, or an ``Nx2`` `~astropy.units.Quantity` in units of pixels.
    r_in : float
        The inner radius of the annulus.
    r_out : float
        The outer radius of the annulus.

    Raises
    ------
    ValueError : `~.exceptions.ValueError`
        If inner radius (``r_in``) is greater than outer radius (``r_out``).
    ValueError : `~.exceptions.ValueError`
        If inner radius is negative.
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

        self.positions = _sanitize_pixel_positions(positions)

    def area(self):
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_circular_photometry(data, self.positions,
                                      self.r_out, error=error, gain=gain,
                                      pixelwise_error=pixelwise_error,
                                      method=method,
                                      subpixels=subpixels,
                                      r_in=self.r_in)

        return flux

    def plot(self, ax=None, fill=False, source_id=None, **kwargs):

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        kwargs['fill'] = fill

        if ax is None:
            ax = plt.gca()

        if source_id is None:
            positions = self.positions
        else:
            positions = self.positions[np.atleast_1d(source_id)]

        resolution = 20

        for position in positions:
            patch_inner = mpatches.CirclePolygon(position, self.r_in,
                                                 resolution=resolution)
            patch_outer = mpatches.CirclePolygon(position, self.r_out,
                                                 resolution=resolution)
            path = _make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)


class SkyEllipticalAperture(SkyAperture):
    """
    Elliptical aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.
    a : `~astropy.units.Quantity`
        The semimajor axis, either in angular or pixel units.
    b : `~astropy.units.Quantity`
        The semiminor axis, either in angular or pixel units.
    theta : `~astropy.units.Quantity`
        The position angle of the semimajor axis (counterclockwise), either
        in angular or pixel units.
    """

    def __init__(self, positions, a, b, theta):

        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('a', a)
        assert_angle_or_pixel('b', b)
        assert_angle('theta', theta)

        if a.unit.physical_type != b.unit.physical_type:
            raise ValueError("a and b should either both be angles or in pixels")

        self.a = a
        self.b = b
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Return a EllipticalAperture instance in pixel coordinates.
        """

        x, y, scale, angle = skycoord_to_pixel_scale_angle(self.positions, wcs)

        # TODO: no need to use the mean once we support arrays of aperture values
        if self.a.unit.physical_type == 'angle':
            a = (np.mean(scale) * self.a).to(u.pixel).value
            b = (np.mean(scale) * self.b).to(u.pixel).value
        else:  # pixel
            a = self.a.value
            b = self.b.value

        theta = (angle + self.theta).to(u.radian).value

        pixel_positions = np.array([x, y]).transpose()

        return EllipticalAperture(pixel_positions, a, b, theta)


class EllipticalAperture(PixelAperture):
    """
    Elliptical aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` Numpy
        array, or an ``Nx2`` `~astropy.units.Quantity` in units of pixels.
    a : float
        The semimajor axis.
    b : float
        The semiminor axis.
    theta : float
        The position angle of the semimajor axis in radians
        (counterclockwise).

    Raises
    ------
    ValueError : `~.exceptions.ValueError`
        If either axis (``a`` or ``b``) is negative.

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

        self.positions = _sanitize_pixel_positions(positions)

    def area(self):
        return math.pi * self.a * self.b

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_elliptical_photometry(data, self.positions,
                                        self.a, self.b, self.theta,
                                        error=error, gain=gain,
                                        pixelwise_error=pixelwise_error,
                                        method=method,
                                        subpixels=subpixels)
        return flux

    def plot(self, ax=None, fill=False, source_id=None, **kwargs):

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        kwargs['fill'] = fill

        if ax is None:
            ax = plt.gca()

        if source_id is None:
            positions = self.positions
        else:
            positions = self.positions[np.atleast_1d(source_id)]

        theta_deg = self.theta * 180. / np.pi
        for position in positions:
            patch = mpatches.Ellipse(position, 2.*self.a, 2.*self.b,
                                     theta_deg, **kwargs)
            ax.add_patch(patch)


class SkyEllipticalAnnulus(SkyAperture):
    """
    Elliptical annulus aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.
    a_in : `~astropy.units.Quantity`
        The inner semimajor axis, either in angular or pixel units.
    a_out : `~astropy.units.Quantity`
        The outer semimajor axis, either in angular or pixel units.
    b_out : `~astropy.units.Quantity`
        The outer semiminor axis, either in angular or pixel units. The inner
        semiminor axis is determined by scaling by a_in/a_out.
    theta : `~astropy.units.Quantity`
        The position angle of the semimajor axis (counterclockwise), either
        in angular or pixel units.
    """

    def __init__(self, positions, a_in, a_out, b_out, theta):

        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('a_in', a_in)
        assert_angle_or_pixel('a_out', a_out)
        assert_angle_or_pixel('b_out', b_out)
        assert_angle('theta', theta)

        if a_in.unit.physical_type != a_out.unit.physical_type:
            raise ValueError("a_in and a_out should either both be angles or in pixels")

        if a_out.unit.physical_type != b_out.unit.physical_type:
            raise ValueError("a_out and b_out should either both be angles or in pixels")

        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Return a EllipticalAnnulus instance in pixel coordinates.
        """

        x, y, scale, angle = skycoord_to_pixel_scale_angle(self.positions, wcs)

        # TODO: no need to use the mean once we support arrays of aperture values
        if self.a_in.unit.physical_type == 'angle':
            a_in = (np.mean(scale) * self.a_in).to(u.pixel).value
            a_out = (np.mean(scale) * self.a_out).to(u.pixel).value
            b_out = (np.mean(scale) * self.b_out).to(u.pixel).value
        else:
            a_in = self.a_in.value
            a_out = self.a_out.value
            b_out = self.b_out.value

        theta = (angle + self.theta).to(u.radian).value

        pixel_positions = np.array([x, y]).transpose()

        return EllipticalAnnulus(pixel_positions, a_in, a_out, b_out, theta)


class EllipticalAnnulus(PixelAperture):
    """
    Elliptical annulus aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` Numpy
        array, or an ``Nx2`` `~astropy.units.Quantity` in units of pixels.
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

    Raises
    ------
    ValueError : `~.exceptions.ValueError`
        If inner semimajor axis (``a_in``) is greater than outer semimajor
        axis (``a_out``).
    ValueError : `~.exceptions.ValueError`
        If either the inner semimajor axis (``a_in``) or the outer semiminor
        axis (``b_out``) is negative.
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

        self.positions = _sanitize_pixel_positions(positions)

    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def plot(self, ax=None, fill=False, source_id=None, **kwargs):

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        kwargs['fill'] = fill

        if ax is None:
            ax = plt.gca()

        if source_id is None:
            positions = self.positions
        else:
            positions = self.positions[np.atleast_1d(source_id)]

        theta_deg = self.theta * 180. / np.pi
        for position in positions:
            patch_inner = mpatches.Ellipse(position, 2.*self.a_in,
                                           2.*self.b_in, theta_deg, **kwargs)
            patch_outer = mpatches.Ellipse(position, 2.*self.a_out,
                                           2.*self.b_out, theta_deg, **kwargs)
            path = _make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_elliptical_photometry(data, self.positions,
                                        self.a_out, self.b_out, self.theta,
                                        error=error, gain=gain,
                                        pixelwise_error=pixelwise_error,
                                        method=method,
                                        subpixels=subpixels,
                                        a_in=self.a_in)

        return flux


class RectangularAperture(PixelAperture):
    """
    A rectangular aperture, defined in pixel coordinates.

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
        The position angle of the width side in radians
        (counterclockwise).

    Raises
    ------
    ValueError : `~.exceptions.ValueError`
        If either width (``w``) or height (``h``) is negative.
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

        self.positions = _sanitize_pixel_positions(positions)

    def area(self):
        return self.w * self.h

    def plot(self, ax=None, fill=False, source_id=None, **kwargs):

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        kwargs['fill'] = fill

        if ax is None:
            ax = plt.gca()

        if source_id is None:
            positions = self.positions
        else:
            positions = self.positions[np.atleast_1d(source_id)]

        hw = self.w / 2.
        hh = self.h / 2.
        sint = math.sin(self.theta)
        cost = math.cos(self.theta)
        dx = (hh * sint) - (hw * cost)
        dy = -(hh * cost) - (hw * sint)
        positions = positions + np.array([dx, dy])
        theta_deg = self.theta * 180. / np.pi
        for position in positions:
            patch = mpatches.Rectangle(position, self.w, self.h, theta_deg,
                                       **kwargs)
            ax.add_patch(patch)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='subpixel', subpixels=5):

        if method == 'exact':
            warnings.warn("'exact' method is not implemented, defaults to "
                          "'subpixel' method and subpixels=32 instead",
                          AstropyUserWarning)
            method = 'subpixel'
            subpixels = 32

        elif method not in ('center', 'subpixel'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_rectangular_photometry(data, self.positions,
                                         self.w, self.h, self.theta,
                                         error=error, gain=gain,
                                         pixelwise_error=pixelwise_error,
                                         method=method,
                                         subpixels=subpixels)
        return flux


class RectangularAnnulus(PixelAperture):
    """
    Rectangular annulus aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` Numpy
        array, or an ``Nx2`` `~astropy.units.Quantity` in units of pixels.
    w_in : float
        The inner full width of the aperture.
    w_out : float
        The outer full width of the aperture.
    h_out : float
        The outer full height of the aperture. (The inner full height is
        determined by scaling by w_in/w_out.)
    theta : float
        The position angle of the width side in radians.
        (counterclockwise).

    Raises
    ------
    ValueError : `~.exceptions.ValueError`
        If inner width (``w_in``) is greater than outer width (``w_out``).
    ValueError : `~.exceptions.ValueError`
        If either the inner width (``w_in``) or the outer height (``h_out``)
        is negative.
    """

    def __init__(self, positions, w_in, w_out, h_out, theta):
        try:
            self.w_in = float(w_in)
            self.w_out = float(w_out)
            self.h_out = float(h_out)
            self.theta = float(theta)
        except TypeError:
            raise TypeError("'w_in' and 'w_out' and 'h_out' and 'theta' must "
                            "be numeric, received {0} and {1} and {2} and {3}."
                            .format((type(w_in), type(w_out),
                                     type(h_out), type(theta))))

        if not (w_out > w_in):
            raise ValueError("'w_out' must be greater than 'w_in'")
        if w_in < 0 or h_out < 0:
            raise ValueError("'w_in' and 'h_out' must be non-negative")

        self.h_in = w_in * h_out / w_out

        self.positions = _sanitize_pixel_positions(positions)

    def area(self):
        """
        Returns
        -------
        area : float
            Area of aperture.
        """
        return self.w_out * self.h_out - self.w_in * self.h_in

    def plot(self, ax=None, fill=False, source_id=None, **kwargs):

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        kwargs['fill'] = fill

        if ax is None:
            ax = plt.gca()

        if source_id is None:
            positions = self.positions
        else:
            positions = self.positions[np.atleast_1d(source_id)]

        sint = math.sin(self.theta)
        cost = math.cos(self.theta)
        theta_deg = self.theta * 180. / np.pi

        hw_inner = self.w_in / 2.
        hh_inner = self.h_in / 2.
        dx_inner = (hh_inner * sint) - (hw_inner * cost)
        dy_inner = -(hh_inner * cost) - (hw_inner * sint)
        positions_inner = positions + np.array([dx_inner, dy_inner])
        hw_outer = self.w_out / 2.
        hh_outer = self.h_out / 2.
        dx_outer = (hh_outer * sint) - (hw_outer * cost)
        dy_outer = -(hh_outer * cost) - (hw_outer * sint)
        positions_outer = positions + np.array([dx_outer, dy_outer])

        for i, position_inner in enumerate(positions_inner):
            patch_inner = mpatches.Rectangle(position_inner, self.w, self.h,
                                             theta_deg, **kwargs)
            patch_outer = mpatches.Rectangle(positions_outer[i], self.w,
                                             self.h, theta_deg, **kwargs)
            path = _make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)

    def do_photometry(self, data, error=None, gain=None, pixelwise_error=True,
                      method='subpixel', subpixels=5):

        if method == 'exact':
            warnings.warn("'exact' method is not implemented, defaults to "
                          "'subpixel' instead", AstropyUserWarning)
            method = 'subpixel'

        elif method not in ('center', 'subpixel'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_rectangular_photometry(data, self.positions,
                                         self.w_out, self.h_out, self.theta,
                                         error=error, gain=gain,
                                         pixelwise_error=pixelwise_error,
                                         method=method, subpixels=subpixels,
                                         w_in=self.w_in)

        return flux


def aperture_photometry(data, apertures, unit=None, wcs=None,
                        error=None, gain=None, mask=None, method='exact',
                        subpixels=5, pixelwise_error=True):
    """
    Sum flux within an aperture at the given position(s).

    Parameters
    ----------
    data : array_like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
        The 2-d array on which to perform photometry. Units are used during
        the photometry, either provided along with the data array, or stored
        in the header keyword ``'BUNIT'``.
    apertures : `~photutils.Aperture` instance
        The apertures to use for the photometry.
    unit : `~astropy.units.UnitBase` instance, str
        An object that represents the unit associated with ``data``.  Must
        be an `~astropy.units.UnitBase` object or a string parseable by the
        :mod:`~astropy.units` package. An error is raised if ``data``
        already has a different unit.
    wcs : `~astropy.wcs.WCS`, optional
        Use this as the wcs transformation. It overrides any wcs transformation
        passed along with ``data`` either in the header or in an attribute.
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
        Mask to apply to the data.  Masked pixels are excluded/ignored.
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
    pixelwise_error : bool, optional
        For error and/or gain arrays. If `True`, assume error and/or gain
        vary significantly within an aperture: sum contribution from each
        pixel. If `False`, assume error and gain do not vary significantly
        within an aperture. Use the single value of error and/or gain at
        the center of each aperture as the value for the entire aperture.
        Default is `True`.

    Returns
    -------
    phot_table : `~astropy.table.Table`
        A table of the photometry with the following columns:

        * ``'aperture_sum'``: Sum of the values within the aperture.
        * ``'aperture_sum_err'``: Corresponding uncertainty in
          ``'aperture_sum'`` values.  Returned only if input ``error`` is not
          `None`.
        * ``'xcenter_pix'``, ``'ycenter_pix'``: x and y pixel coordinates of
          the center of the apertures. Unit is pixel.
        * ``'xcenter_input'``, ``'ycenter_input'``: input x and y coordinates
          as they were given in the input ``positions`` parameter.

        The metadata of the table stores the version numbers of both astropy
        and photutils, as well as the calling arguments.
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
                              "only run for the {0}. HDU."
                              .format(i), AstropyUserWarning)
                return aperture_photometry(data[i], apertures, unit,
                                           wcs, error, gain, mask, method,
                                           subpixels, pixelwise_error)

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
        if dataunit is not None:
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

    # Deal with the mask if it exists
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

        # masked values are replaced with zeros, so they do not contribute
        # to the aperture sums
        data = copy.deepcopy(data)    # do not modify input data
        data *= ~mask

    # Check whether we really need to calculate pixelwise errors, even if
    # requested. (If neither error nor gain is an array, we don't need to.)
    if ((error is None) or (np.isscalar(error) and gain is None) or
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

        # mask the error array, if necessary
        # masked values are replaced with zeros, so they do not contribute
        # to the sums
        if mask is not None:
            error = copy.deepcopy(error)    # do not modify input data
            error *= ~mask

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

    ap = apertures

    if isinstance(apertures, SkyAperture):
        positions = ap.positions
        if wcs_transformation is None:
            wcs_transformation = WCS(header)
        ap = ap.to_pixel(wcs_transformation)
        pixelpositions = ap.positions * u.pixel

        pixpos = np.transpose(pixelpositions)
        # check whether single or multiple positions
        if len(pixelpositions) > 1 and pixelpositions[0].size >= 2:
            coord_columns = (pixpos[0], pixpos[1], positions)
        else:
            coord_columns = ((pixpos[0],), (pixpos[1],), (positions,))
        coord_col_names = ('xcenter_pixel', 'ycenter_pixel', 'center_input')
    else:
        positions = ap.positions * u.pixel
        pixelpositions = ap.positions * u.pixel

        pixpos = np.transpose(pixelpositions)
        pos = np.transpose(positions)
        # check whether single or multiple positions
        if len(pixelpositions) > 1 and pixelpositions[0].size >= 2:
            coord_columns = (pixpos[0], pixpos[1], pos[0], pos[1])
        else:
            coord_columns = ((pixpos[0],), (pixpos[1],), (pos[0],), (pos[1],))
        coord_col_names = ('xcenter_pixel', 'ycenter_pixel', 'xcenter_input',
                           'ycenter_input')

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

    return Table(data=(photometry_result + coord_columns),
                 names=(phot_col_names + coord_col_names),
                 meta={'name': 'Aperture photometry results',
                       'version': 'astropy: {0}, photutils: {1}'
                       .format(astropy_version, photutils_version),
                       'calling_args': ('method={0}, subpixels={1}, '
                                        'error={2}, gain={3}, '
                                        'pixelwise_error={4}')
                       .format(method, subpixels, error is not None,
                               gain is not None, pixelwise_error)})
