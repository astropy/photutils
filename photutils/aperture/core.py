# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the base aperture classes.
"""

import abc
import copy
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils import deprecated
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs.utils import wcs_to_celestial_frame

from .bounding_box import BoundingBox
from ._photometry_utils import (_handle_units, _prepare_photometry_data,
                                _validate_inputs)
from ..utils._wcs_helpers import (_pixel_to_world,
                                  _pixel_scale_angle_at_skycoord,
                                  _world_to_pixel)

__all__ = ['Aperture', 'SkyAperture', 'PixelAperture']


class Aperture(metaclass=abc.ABCMeta):
    """
    Abstract base class for all apertures.
    """

    _shape_params = ()
    positions = np.array(())
    theta = None

    def __len__(self):
        if self.isscalar:
            raise TypeError('Scalar {0!r} object has no len()'
                            .format(self.__class__.__name__))
        return self.shape[0]

    def __getitem__(self, index):
        kwargs = dict()
        for param in self._shape_params:
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
        cls_info = [self._positions_str(prefix)]
        for param in self._shape_params:
            cls_info.append('{0}={1}'.format(param, getattr(self, param)))
        cls_info = ', '.join(cls_info)

        return '{0}{1})>'.format(prefix, cls_info)

    def __str__(self):
        prefix = 'positions'
        cls_info = [
            ('Aperture', self.__class__.__name__),
            (prefix, self._positions_str(prefix + ': '))]
        if self._shape_params is not None:
            for param in self._shape_params:
                cls_info.append((param, getattr(self, param)))
        fmt = ['{0}: {1}'.format(key, val) for key, val in cls_info]

        return '\n'.join(fmt)

    @property
    def shape(self):
        """
        The shape of the instance.
        """

        if isinstance(self.positions, SkyCoord):
            return self.positions.shape
        else:
            return self.positions.shape[:-1]

    @property
    def isscalar(self):
        """
        Whether the instance is scalar (i.e., a single position).
        """

        return self.shape == ()


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
    def _xy_extents(self):
        """
        The (x, y) extents of the aperture measured from the center
        position.

        In other words, the (x, y) extents are half of the aperture
        minimal bounding box size in each dimension.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')

    @property
    @deprecated('0.7', alternative='bbox')
    def bounding_boxes(self):
        """
        The minimal bounding box for the aperture.
        """

        return self.bbox

    @property
    def bbox(self):
        """
        The minimal bounding box for the aperture.

        If the aperture is scalar then a single
        `~photutils.aperture.BoundingBox` is returned, otherwise a list
        of `~photutils.aperture.BoundingBox` is returned.
        """

        positions = np.atleast_2d(self.positions)
        x_delta, y_delta = self._xy_extents
        xmin = positions[:, 0] - x_delta
        xmax = positions[:, 0] + x_delta
        ymin = positions[:, 1] - y_delta
        ymax = positions[:, 1] + y_delta

        bboxes = [BoundingBox.from_float(x0, x1, y0, y1)
                  for x0, x1, y0, y1 in zip(xmin, xmax, ymin, ymax)]

        if self.isscalar:
            return bboxes[0]
        else:
            return bboxes

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
                                  np.atleast_1d(self.bbox)):
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
        mask : `~photutils.aperture.ApertureMask` or list of `~photutils.aperture.ApertureMask`
            A mask for the aperture.  If the aperture is scalar then a
            single `~photutils.aperture.ApertureMask` is returned,
            otherwise a list of `~photutils.aperture.ApertureMask` is
            returned.
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

    def do_photometry(self, data, error=None, mask=None, method='exact',
                      subpixels=5):
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
        data, error, unit = _handle_units(data, error)

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

    def _define_patch_params(self, origin=(0, 0), **kwargs):
        """
        Define the aperture patch position and set any default
        matplotlib patch keywords (e.g., ``fill=False``).

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

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
        xy_positions[:, 0] -= origin[0]
        xy_positions[:, 1] -= origin[1]

        patch_params = self._default_patch_properties
        patch_params.update(kwargs)

        return xy_positions, patch_params

    @abc.abstractmethod
    def _to_patch(self, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

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

    def plot(self, axes=None, origin=(0, 0), **kwargs):
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

        kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.
        """

        import matplotlib.pyplot as plt

        if axes is None:
            axes = plt.gca()

        patches = self._to_patch(origin=origin, **kwargs)
        if self.isscalar:
            patches = (patches,)

        for patch in patches:
            axes.add_patch(patch)

    def _to_sky_params(self, wcs):
        """
        Convert the pixel aperture parameters to those for a sky
        aperture.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        sky_params : `dict`
            A dictionary of parameters for an equivalent sky aperture.
        """

        sky_params = {}
        xpos, ypos = np.transpose(self.positions)
        sky_params['positions'] = _pixel_to_world(xpos, ypos, wcs)

        # The aperture object must have a single value for each shape
        # parameter so we must use a single pixel scale for all positions.
        # Here, we define the scale at the WCS CRVAL position.
        crval = SkyCoord(*wcs.wcs.crval, frame=wcs_to_celestial_frame(wcs),
                         unit=wcs.wcs.cunit)
        pixscale, angle = _pixel_scale_angle_at_skycoord(crval, wcs)

        shape_params = list(self._shape_params)

        theta_key = 'theta'
        if theta_key in shape_params:
            sky_params[theta_key] = (self.theta * u.rad) - angle.to(u.rad)
            shape_params.remove(theta_key)

        for shape_param in shape_params:
            value = getattr(self, shape_param)
            sky_params[shape_param] = (value * u.pix * pixscale).to(u.arcsec)

        return sky_params

    @abc.abstractmethod
    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyAperture` object defined in
        celestial coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

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

    def _to_pixel_params(self, wcs):
        """
        Convert the sky aperture parameters to those for a pixel
        aperture.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        pixel_params : `dict`
            A dictionary of parameters for an equivalent pixel aperture.
        """

        pixel_params = {}

        xpos, ypos = _world_to_pixel(self.positions, wcs)
        pixel_params['positions'] = np.array([xpos, ypos]).transpose()

        # The aperture object must have a single value for each shape
        # parameter so we must use a single pixel scale for all positions.
        # Here, we define the scale at the WCS CRVAL position.
        crval = SkyCoord(*wcs.wcs.crval, frame=wcs_to_celestial_frame(wcs),
                         unit=wcs.wcs.cunit)
        pixscale, angle = _pixel_scale_angle_at_skycoord(crval, wcs)

        shape_params = list(self._shape_params)

        theta_key = 'theta'
        if theta_key in shape_params:
            pixel_params[theta_key] = (self.theta + angle).to(u.radian).value
            shape_params.remove(theta_key)

        for shape_param in shape_params:
            value = getattr(self, shape_param)
            if value.unit.physical_type == 'angle':
                pixel_params[shape_param] = ((value / pixscale)
                                             .to(u.pixel).value)
            else:
                pixel_params[shape_param] = value.value

        return pixel_params

    @abc.abstractmethod
    def to_pixel(self, wcs):
        """
        Convert the aperture to a `PixelAperture` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `PixelAperture` object
            A `PixelAperture` object.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')
