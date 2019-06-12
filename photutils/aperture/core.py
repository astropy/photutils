# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc
import copy
import warnings
from collections import OrderedDict

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable
import astropy.units as u
from astropy.utils import deprecated
from astropy.utils.decorators import deprecated_renamed_argument
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from astropy.wcs import WCS
from astropy.wcs.utils import (skycoord_to_pixel, pixel_to_skycoord,
                               wcs_to_celestial_frame)

from ..utils import get_version_info
from ..utils.misc import _ABCMetaAndInheritDocstrings
from ..utils._wcs_helpers import _pixel_scale_angle_at_skycoord


__all__ = ['Aperture', 'SkyAperture', 'PixelAperture', 'aperture_photometry']


class Aperture(metaclass=_ABCMetaAndInheritDocstrings):
    """
    Abstract base class for all apertures.
    """

    def __len__(self):
        if self.isscalar:
            raise TypeError('Scalar {0!r} object has no len()'
                            .format(self.__class__.__name__))
        return self.shape[0]

    def __getitem__(self, index):
        kwargs = dict()
        for param in self._params:
            kwargs[param] = getattr(self, param)
        return self.__class__(self.positions[index], **kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def _positions_str(self, prefix=None):
        if isinstance(self, PixelAperture):
            return np.array2string(self.positions, separator=', ',
                                   prefix=prefix)
        elif isinstance(self, SkyAperture):
            return repr(self.positions)
        else:
            raise TypeError('Aperture must be a subclass of PixelAperture '
                            'or SkyAperture')

    def __repr__(self):
        prefix = '<{0}('.format(self.__class__.__name__)
        params = [self._positions_str(prefix)]
        for param in self._params:
            params.append('{0}={1}'.format(param, getattr(self, param)))
        params = ', '.join(params)

        return '{0}{1})>'.format(prefix, params)

    def __str__(self):
        prefix = 'positions'
        cls_info = [
            ('Aperture', self.__class__.__name__),
            (prefix, self._positions_str(prefix + ': '))]
        if self._params is not None:
            for param in self._params:
                cls_info.append((param, getattr(self, param)))
        fmt = ['{0}: {1}'.format(key, val) for key, val in cls_info]

        return '\n'.join(fmt)

    @property
    def shape(self):
        if isinstance(self.positions, SkyCoord):
            return self.positions.shape
        else:
            return self.positions.shape[:-1]

    @property
    def isscalar(self):
        return self.shape == ()

    @property
    @abc.abstractmethod
    def positions(self):
        """
        The aperture positions.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')


class PixelAperture(Aperture):
    """
    Abstract base class for apertures defined in pixel coordinates.
    """

    @property
    def _default_patch_properties(self):
        """
        A dictionary of default matplotlib.patches.Patch properties.
        """

        mpl_params = dict()

        # matplotlib.patches.Patch default is ``fill=True``
        mpl_params['fill'] = False

        return mpl_params

    @staticmethod
    def _translate_mask_mode(mode, subpixels, rectangle=False):
        if mode not in ('center', 'subpixel', 'exact'):
            raise ValueError('Invalid mask mode: {0}'.format(mode))

        if rectangle and mode == 'exact':
            warnings.warn('The "exact" method is not yet implemented for '
                          'rectangular apertures -- using "subpixel" method '
                          'with "subpixels=32"', AstropyUserWarning)
            mode = 'subpixel'
            subpixels = 32

        if mode == 'subpixels':
            if not isinstance(subpixels, int) or subpixels <= 0:
                raise ValueError('subpixels must be a strictly positive '
                                 'integer')

        if mode == 'center':
            use_exact = 0
            subpixels = 1
        elif mode == 'subpixel':
            use_exact = 0
        elif mode == 'exact':
            use_exact = 1
            subpixels = 1

        return use_exact, subpixels

    @property
    @abc.abstractmethod
    def bounding_boxes(self):
        """
        The minimal bounding box for the aperture.

        If the aperture is scalar then a single `~photutils.BoundingBox`
        is returned, otherwise a list of `~photutils.BoundingBox` is
        returned.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')

    @property
    def _centered_edges(self):
        """
        A list of ``(xmin, xmax, ymin, ymax)`` tuples, one for each
        position, of the pixel edges after recentering the aperture at
        the origin.

        These pixel edges are used by the low-level `photutils.geometry`
        functions.
        """

        edges = []
        for position, bbox in zip(np.atleast_2d(self.positions),
                                  np.atleast_1d(self.bounding_boxes)):
            xmin = bbox.ixmin - 0.5 - position[0]
            xmax = bbox.ixmax - 0.5 - position[0]
            ymin = bbox.iymin - 0.5 - position[1]
            ymax = bbox.iymax - 0.5 - position[1]
            edges.append((xmin, xmax, ymin, ymax))

        return edges

    @property
    def area(self):
        """
        Return the exact area of the aperture shape.

        Returns
        -------
        area : float
            The aperture area.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')

    @deprecated('0.7', alternative=('e.g. np.sum(aper.to_mask().data) for a '
                                    'scalar aperture'))
    def mask_area(self, method='exact', subpixels=5):
        """
        Return the area of the aperture mask.

        For ``method`` other than ``'exact'``, this area will be less
        than the exact analytical area (e.g. the ``area`` method).  Note
        that for these methods, the values can also differ because of
        fractional pixel positions.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        area : float
            A list of the mask area (one per position) of the aperture.
        """

        masks = self.to_mask(method=method, subpixels=subpixels)
        if self.isscalar:
            masks = (masks,)
        return [np.sum(mask.data) for mask in masks]

    @abc.abstractmethod
    def to_mask(self, method='exact', subpixels=5):
        """
        Return a mask for the aperture.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        mask : `~photutils.ApertureMask` or list of `~photutils.ApertureMask`
            A mask for the aperture.  If the aperture is scalar then a
            single `~photutils.ApertureMask` is returned, otherwise a
            list of `~photutils.ApertureMask` is returned.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')

    def _do_photometry(self, data, variance, method='exact', subpixels=5,
                       unit=None):

        aperture_sums = []
        aperture_sum_errs = []

        masks = self.to_mask(method=method, subpixels=subpixels)
        if self.isscalar:
            masks = (masks,)

        for apermask in masks:
            data_weighted = apermask.multiply(data)
            if data_weighted is None:
                aperture_sums.append(np.nan)
            else:
                aperture_sums.append(np.sum(data_weighted))

            if variance is not None:
                variance_weighted = apermask.multiply(variance)
                if variance_weighted is None:
                    aperture_sum_errs.append(np.nan)
                else:
                    aperture_sum_errs.append(
                        np.sqrt(np.sum(variance_weighted)))

        aperture_sums = np.array(aperture_sums)
        aperture_sum_errs = np.array(aperture_sum_errs)

        # apply units
        if unit is not None:
            aperture_sums = aperture_sums * unit  # can't use *= w/old numpy
            aperture_sum_errs = aperture_sum_errs * unit

        return aperture_sums, aperture_sum_errs

    @deprecated_renamed_argument('unit', None, '0.7')
    def do_photometry(self, data, error=None, mask=None, method='exact',
                      subpixels=5, unit=None):
        """
        Perform aperture photometry on the input data.

        Parameters
        ----------
        data : array_like or `~astropy.units.Quantity` instance
            The 2D array on which to perform photometry.  ``data``
            should be background subtracted.

        error : array_like or `~astropy.units.Quantity`, optional
            The pixel-wise Gaussian 1-sigma errors of the input
            ``data``.  ``error`` is assumed to include *all* sources of
            error, including the Poisson error of the sources (see
            `~photutils.utils.calc_total_error`) .  ``error`` must have
            the same shape as the input ``data``.

        mask : array_like (bool), optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked.  Masked data are excluded from all calculations.

        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        unit : `~astropy.units.UnitBase` object or str, optional
            Deprecated in v0.7.
            An object that represents the unit associated with the input
            ``data`` and ``error`` arrays.  Must be a
            `~astropy.units.UnitBase` object or a string parseable by
            the :mod:`~astropy.units` package.  If ``data`` or ``error``
            already have a different unit, the input ``unit`` will not
            be used and a warning will be raised.

        Returns
        -------
        aperture_sums : `~numpy.ndarray` or `~astropy.units.Quantity`
            The sums within each aperture.

        aperture_sum_errs : `~numpy.ndarray` or `~astropy.units.Quantity`
            The errors on the sums within each aperture.
        """

        # validate inputs
        data, error = _validate_inputs(data, error)

        # handle data, error, and unit inputs
        # output data and error are ndarray without units
        data, error, unit = _handle_units(data, error, unit)

        # compute variance and apply input mask
        data, variance = _prepare_photometry_data(data, error, mask)

        return self._do_photometry(data, variance, method=method,
                                   subpixels=subpixels, unit=unit)

    @staticmethod
    def _make_annulus_path(patch_inner, patch_outer):
        """
        Define a matplotlib annulus path from two patches.

        This preserves the cubic Bezier curves (CURVE4) of the aperture
        paths.
        """

        import matplotlib.path as mpath

        path_inner = patch_inner.get_path()
        transform_inner = patch_inner.get_transform()
        path_inner = transform_inner.transform_path(path_inner)

        path_outer = patch_outer.get_path()
        transform_outer = patch_outer.get_transform()
        path_outer = transform_outer.transform_path(path_outer)

        verts_inner = path_inner.vertices[:-1][::-1]
        verts_inner = np.concatenate((verts_inner, [verts_inner[-1]]))

        verts = np.vstack((path_outer.vertices, verts_inner))
        codes = np.hstack((path_outer.codes, path_inner.codes))

        return mpath.Path(verts, codes)

    def _define_patch_params(self, origin=(0, 0), indices=None, **kwargs):
        """
        Define the aperture patch position and set any default
        matplotlib patch keywords (e.g. ``fill=False``).

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        indices : int or array of int, optional
            The indices of the aperture positions to plot.

        kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        xy_positions : `~numpy.ndarray`
            The aperture patch positions.

        patch_params : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.
        """

        xy_positions = copy.deepcopy(np.atleast_2d(self.positions))
        if indices is not None:
            xy_positions = xy_positions[np.atleast_1d(indices)]

        xy_positions[:, 0] -= origin[0]
        xy_positions[:, 1] -= origin[1]

        patch_params = self._default_patch_properties
        patch_params.update(kwargs)

        return xy_positions, patch_params

    @abc.abstractmethod
    def _to_patch(self, origin=(0, 0), indices=None, **kwargs):
        """
        Return a `~matplotlib.patches.patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        indices : int or array of int, optional
            The indices of the aperture positions to plot.

        kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.patch` or list of `~matplotlib.patches.patch`
            A patch for the aperture.  If the aperture is scalar then a
            single `~matplotlib.patches.patch` is returned, otherwise a
            list of `~matplotlib.patches.patch` is returned.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')

    @deprecated_renamed_argument('ax', 'axes', '0.7')
    @deprecated_renamed_argument('indices', None, '0.7',
                                 alternative=('indices directly on the '
                                              'aperture object '
                                              '(e.g. aper[idx].plot())'))
    def plot(self, axes=None, origin=(0, 0), indices=None, **kwargs):
        """
        Plot the aperture on a matplotlib `~matplotlib.axes.Axes`
        instance.

        Parameters
        ----------
        axes : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot.  If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        indices : int, array of int, or `None`, optional
            The indices of the aperture position(s) to plot.  If `None`
            (default) then all aperture positions will be plotted.

        kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.
        """

        import matplotlib.pyplot as plt

        if axes is None:
            axes = plt.gca()

        patches = self._to_patch(origin=origin, indices=indices, **kwargs)
        if self.isscalar:
            patches = (patches,)

        for patch in patches:
            axes.add_patch(patch)

    def _to_sky_params(self, wcs, mode='all'):
        """
        Convert the pixel aperture parameters to those for a sky
        aperture.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        sky_params : `dict`
            A dictionary of parameters for an equivalent sky aperture.
        """

        sky_params = {}
        x, y = np.transpose(self.positions)
        sky_params['positions'] = pixel_to_skycoord(x, y, wcs, mode=mode)

        # The aperture object must have a single value for each shape
        # parameter so we must use a single pixel scale for all positions.
        # Here, we define the scale at the WCS CRVAL position.
        crval = SkyCoord(*wcs.wcs.crval, frame=wcs_to_celestial_frame(wcs),
                         unit=wcs.wcs.cunit)
        scale, angle = _pixel_scale_angle_at_skycoord(crval, wcs)

        params = self._params[:]
        theta_key = 'theta'
        if theta_key in self._params:
            sky_params[theta_key] = (self.theta * u.rad) - angle.to(u.rad)
            params.remove(theta_key)

        param_vals = [getattr(self, param) for param in params]
        for param, param_val in zip(params, param_vals):
            sky_params[param] = (param_val * u.pix * scale).to(u.arcsec)

        return sky_params

    @abc.abstractmethod
    def to_sky(self, wcs, mode='all'):
        """
        Convert the aperture to a `SkyAperture` object defined in
        celestial coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS` object
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `SkyAperture` object
            A `SkyAperture` object.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')


class SkyAperture(Aperture):
    """
    Abstract base class for all apertures defined in celestial
    coordinates.
    """

    def _to_pixel_params(self, wcs, mode='all'):
        """
        Convert the sky aperture parameters to those for a pixel
        aperture.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        pixel_params : `dict`
            A dictionary of parameters for an equivalent pixel aperture.
        """

        pixel_params = {}
        x, y = skycoord_to_pixel(self.positions, wcs, mode=mode)
        pixel_params['positions'] = np.array([x, y]).transpose()

        # The aperture object must have a single value for each shape
        # parameter so we must use a single pixel scale for all positions.
        # Here, we define the scale at the WCS CRVAL position.
        crval = SkyCoord(*wcs.wcs.crval, frame=wcs_to_celestial_frame(wcs),
                         unit=wcs.wcs.cunit)
        scale, angle = _pixel_scale_angle_at_skycoord(crval, wcs)

        params = self._params[:]
        theta_key = 'theta'
        if theta_key in self._params:
            pixel_params[theta_key] = (self.theta + angle).to(u.radian).value
            params.remove(theta_key)

        param_vals = [getattr(self, param) for param in params]
        if param_vals[0].unit.physical_type == 'angle':
            for param, param_val in zip(params, param_vals):
                pixel_params[param] = (param_val / scale).to(u.pixel).value
        else:    # pixels
            for param, param_val in zip(params, param_vals):
                pixel_params[param] = param_val.value

        return pixel_params

    @abc.abstractmethod
    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to a `PixelAperture` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS` object
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `PixelAperture` object
            A `PixelAperture` object.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')


def _handle_hdu_input(data):
    """
    Convert FITS HDU ``data`` to a `~numpy.ndarray` (and optional unit).

    Used to parse ``data`` input to `aperture_photometry`.

    Parameters
    ----------
    data : array_like, `~astropy.units.Quantity`, `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.HDUList`
        The 2D data array.

    Returns
    -------
    data : `~numpy.ndarray`
        The 2D data array.

    unit : `~astropy.unit.Unit` or `None`
        The unit for the data.
    """

    bunit = None

    if isinstance(data, (fits.PrimaryHDU, fits.ImageHDU, fits.HDUList)):
        warnings.warn('"astropy.io.fits.PrimaryHDU", '
                      '"astropy.io.fits.ImageHDU", and '
                      '"astropy.io.fits.HDUList" inputs are deprecated as of '
                      'v0.7 and will not be allowed in future versions.',
                      AstropyDeprecationWarning)

    if isinstance(data, fits.HDUList):
        for i, hdu in enumerate(data):
            if hdu.data is not None:
                warnings.warn('Input data is a HDUList object.  Doing '
                              'photometry only on the {0} HDU.'
                              .format(i), AstropyUserWarning)
                data = hdu
                break

    if isinstance(data, (fits.PrimaryHDU, fits.ImageHDU)):
        header = data.header
        data = data.data

        if 'BUNIT' in header:
            bunit = u.Unit(header['BUNIT'], parse_strict='warn')
            if isinstance(bunit, u.UnrecognizedUnit):
                warnings.warn('The BUNIT in the header of the input data is '
                              'not parseable as a valid unit.',
                              AstropyUserWarning)

    try:
        fits_wcs = WCS(header)
    except Exception:
        # A valid WCS was not found in the header.  Let the calling
        # application raise an exception if it needs a WCS.
        fits_wcs = None

    return data, bunit, fits_wcs


def _validate_inputs(data, error):
    """
    Validate inputs.

    ``data`` and ``error`` are converted to a `~numpy.ndarray`, if
    necessary.

    Used to parse inputs to `aperture_photometry` and
    `PixelAperture.do_photometry`.
    """

    data = np.asanyarray(data)
    if data.ndim != 2:
        raise ValueError('data must be a 2D array.')

    if error is not None:
        error = np.asanyarray(error)
        if error.shape != data.shape:
            raise ValueError('error and data must have the same shape.')

    return data, error


def _handle_units(data, error, unit):
    """
    Handle Quantity inputs and the ``unit`` keyword.

    Any units on ``data`` and ``error` are removed.  ``data`` and
    ``error`` are returned as `~numpy.ndarray`.  The returned ``unit``
    represents the unit for both ``data`` and ``error``.

    Used to parse inputs to `aperture_photometry` and
    `PixelAperture.do_photometry`.
    """

    if unit is not None:
        unit = u.Unit(unit, parse_strict='warn')
        if isinstance(unit, u.UnrecognizedUnit):
            warnings.warn('The input unit is not parseable as a valid '
                          'unit.', AstropyUserWarning)
            unit = None

    # check Quantity inputs
    inputs = (data, error)
    has_unit = [hasattr(x, 'unit') for x in inputs if x is not None]
    use_units = all(has_unit)
    if any(has_unit) and not use_units:
        raise ValueError('If data or error has units, then they both must '
                         'have the same units.')

    # handle Quantity inputs
    if use_units:
        if unit is not None and data.unit != unit:
            warnings.warn('The input unit does not agree with the data '
                          'unit.  Using the data unit.', AstropyUserWarning)
            unit = data.unit

        # strip data and error units for performance
        unit = data.unit
        data = data.value

        if error is not None:
            if unit != error.unit:
                raise ValueError('data and error must have the same units.')
            error = error.value

    return data, error, unit


def _prepare_photometry_data(data, error, mask):
    """
    Prepare data and error arrays for photometry.

    Error is converted to variance and masked values are set to zero in
    the output data and variance arrays.

    Used to parse inputs to `aperture_photometry` and
    `PixelAperture.do_photometry`.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The 2D array on which to perform photometry.

    error : `~numpy.ndarray` or `None`
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.

    mask : array_like (bool) or `None`
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    data : `~numpy.ndarray`
        The 2D array on which to perform photometry, where masked values
        have been set to zero.

    variance : `~numpy.ndarray` or `None`
        The pixel-wise Gaussian 1-sigma variance of the input ``data``,
        where masked values have been set to zero.
    """

    if error is not None:
        variance = error ** 2
    else:
        variance = None

    if mask is not None:
        mask = np.asanyarray(mask)
        if mask.shape != data.shape:
            raise ValueError('mask and data must have the same shape.')

        data = data.copy()  # do not modify input data
        data[mask] = 0.

        if variance is not None:
            variance[mask] = 0.

    return data, variance


@deprecated_renamed_argument('unit', None, '0.7')
def aperture_photometry(data, apertures, error=None, mask=None,
                        method='exact', subpixels=5, unit=None, wcs=None):
    """
    Perform aperture photometry on the input data by summing the flux
    within the given aperture(s).

    Parameters
    ----------
    data : array_like, `~astropy.units.Quantity`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`, or `~astropy.nddata.NDData`
        The 2D array on which to perform photometry. ``data`` should be
        background-subtracted.  Units can be used during the photometry,
        either provided with the data (e.g. `~astropy.units.Quantity` or
        `~astropy.nddata.NDData` inputs) or the ``unit`` keyword.  If
        ``data`` is an `~astropy.io.fits.ImageHDU` or
        `~astropy.io.fits.HDUList`, the unit is determined from the
        ``'BUNIT'`` header keyword.  `~astropy.io.fits.ImageHDU` or
        `~astropy.io.fits.HDUList` inputs were deprecated in v0.7.  If
        ``data`` is a `~astropy.units.Quantity` array, then ``error``
        (if input) must also be a `~astropy.units.Quantity` array with
        the same units.  See the Notes section below for more
        information about `~astropy.nddata.NDData` input.

    apertures : `~photutils.Aperture` or list of `~photutils.Aperture`
        The aperture(s) to use for the photometry.  If ``apertures`` is
        a list of `~photutils.Aperture` then they all must have the same
        position(s).

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.  If a
        `~astropy.units.Quantity` array, then ``data`` must also be a
        `~astropy.units.Quantity` array with the same units.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    method : {'exact', 'center', 'subpixel'}, optional
        The method used to determine the overlap of the aperture on the
        pixel grid.  Not all options are available for all aperture
        types.  Note that the more precise methods are generally slower.
        The following methods are available:

            * ``'exact'`` (default):
                The the exact fractional overlap of the aperture and
                each pixel is calculated.  The returned mask will
                contain values between 0 and 1.

            * ``'center'``:
                A pixel is considered to be entirely in or out of the
                aperture depending on whether its center is in or out of
                the aperture.  The returned mask will contain values
                only of 0 (out) and 1 (in).

            * ``'subpixel'``:
                A pixel is divided into subpixels (see the ``subpixels``
                keyword), each of which are considered to be entirely in
                or out of the aperture depending on whether its center
                is in or out of the aperture.  If ``subpixels=1``, this
                method is equivalent to ``'center'``.  The returned mask
                will contain values between 0 and 1.

    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor in
        each dimension.  That is, each pixel is divided into ``subpixels
        ** 2`` subpixels.

    unit : `~astropy.units.UnitBase` object or str, optional
        Deprecated in v0.7.
        An object that represents the unit associated with the input
        ``data`` and ``error`` arrays.  Must be a
        `~astropy.units.UnitBase` object or a string parseable by the
        :mod:`~astropy.units` package.  If ``data`` or ``error`` already
        have a different unit, the input ``unit`` will not be used and a
        warning will be raised.  If ``data`` is an
        `~astropy.io.fits.ImageHDU` or `~astropy.io.fits.HDUList`,
        ``unit`` will override the ``'BUNIT'`` header keyword.  This
        keyword should be used sparingly (it exists to support the input
        of `~astropy.nddata.NDData` objects).  Instead one should input
        the ``data`` (and optional ``error``) as
        `~astropy.units.Quantity` objects.

    wcs : `~astropy.wcs.WCS`, optional
        The WCS transformation to use if the input ``apertures`` is a
        `SkyAperture` object.  If ``data`` is an
        `~astropy.io.fits.ImageHDU` or `~astropy.io.fits.HDUList`,
        ``wcs`` overrides any WCS transformation present in the header.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of the photometry with the following columns:

            * ``'id'``:
              The source ID.

            * ``'xcenter'``, ``'ycenter'``:
              The ``x`` and ``y`` pixel coordinates of the input
              aperture center(s).

            * ``'sky_center'``:
              The sky coordinates of the input aperture center(s).
              Returned only if the input ``apertures`` is a
              `SkyAperture` object.

            * ``'aperture_sum'``:
              The sum of the values within the aperture.

            * ``'aperture_sum_err'``:
              The corresponding uncertainty in the ``'aperture_sum'``
              values.  Returned only if the input ``error`` is not
              `None`.

        The table metadata includes the Astropy and Photutils version
        numbers and the `aperture_photometry` calling arguments.

    Notes
    -----
    If the input ``data`` is a `~astropy.nddata.NDData` instance, then
    the ``error``, ``mask``, ``unit``, and ``wcs`` keyword inputs are
    ignored.  Instead, these values should be defined as attributes in
    the `~astropy.nddata.NDData` object.  In the case of ``error``, it
    must be defined in the ``uncertainty`` attribute with a
    `~astropy.nddata.StdDevUncertainty` instance.
    """

    if isinstance(data, NDData):
        nddata_attr = {'error': error, 'mask': mask, 'unit': unit, 'wcs': wcs}
        for key, value in nddata_attr.items():
            if value is not None:
                warnings.warn('The {0!r} keyword is be ignored.  Its value '
                              'is obtained from the input NDData object.'
                              .format(key), AstropyUserWarning)

        mask = data.mask
        wcs = data.wcs

        if isinstance(data.uncertainty, StdDevUncertainty):
            if data.uncertainty.unit is None:
                error = data.uncertainty.array
            else:
                error = data.uncertainty.array * data.uncertainty.unit

        if data.unit is not None:
            data = u.Quantity(data.data, unit=data.unit)
        else:
            data = data.data

        return aperture_photometry(data, apertures, error=error, mask=mask,
                                   method=method, subpixels=subpixels,
                                   wcs=wcs)

    # handle FITS HDU input data
    data, bunit, fits_wcs = _handle_hdu_input(data)
    # NOTE: input unit overrides bunit
    if unit is None:
        unit = bunit
    # NOTE: input wcs overrides FITS WCS
    if not wcs:
        wcs = fits_wcs

    # validate inputs
    data, error = _validate_inputs(data, error)

    # handle data, error, and unit inputs
    # output data and error are ndarray without units
    data, error, unit = _handle_units(data, error, unit)

    # compute variance and apply input mask
    data, variance = _prepare_photometry_data(data, error, mask)

    single_aperture = False
    if isinstance(apertures, Aperture):
        single_aperture = True
        apertures = (apertures,)

    # convert sky to pixel apertures
    skyaper = False
    if isinstance(apertures[0], SkyAperture):
        if wcs is None:
            raise ValueError('A WCS transform must be defined by the input '
                             'data or the wcs keyword when using a '
                             'SkyAperture object.')

        # used to include SkyCoord position in the output table
        skyaper = True
        skycoord_pos = apertures[0].positions

        apertures = [aper.to_pixel(wcs) for aper in apertures]

    # compare positions in pixels to avoid comparing SkyCoord objects
    positions = apertures[0].positions
    for aper in apertures[1:]:
        if not np.array_equal(aper.positions, positions):
            raise ValueError('Input apertures must all have identical '
                             'positions.')

    # define output table meta data
    meta = OrderedDict()
    meta['name'] = 'Aperture photometry results'
    meta['version'] = get_version_info()
    calling_args = "method='{0}', subpixels={1}".format(method, subpixels)
    meta['aperture_photometry_args'] = calling_args

    tbl = QTable(meta=meta)

    positions = np.atleast_2d(apertures[0].positions)
    tbl['id'] = np.arange(positions.shape[0], dtype=int) + 1

    xypos_pixel = np.transpose(positions) * u.pixel
    tbl['xcenter'] = xypos_pixel[0]
    tbl['ycenter'] = xypos_pixel[1]

    if skyaper:
        if skycoord_pos.isscalar:
            # create length-1 SkyCoord array
            tbl['sky_center'] = skycoord_pos.reshape((-1,))
        else:
            tbl['sky_center'] = skycoord_pos

    sum_key_main = 'aperture_sum'
    sum_err_key_main = 'aperture_sum_err'
    for i, aper in enumerate(apertures):
        aper_sum, aper_sum_err = aper._do_photometry(data, variance,
                                                     method=method,
                                                     subpixels=subpixels,
                                                     unit=unit)

        sum_key = sum_key_main
        sum_err_key = sum_err_key_main
        if not single_aperture:
            sum_key += '_{}'.format(i)
            sum_err_key += '_{}'.format(i)

        tbl[sum_key] = aper_sum
        if error is not None:
            tbl[sum_err_key] = aper_sum_err

    return tbl
