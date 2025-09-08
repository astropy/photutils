# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define the base aperture classes.
"""

import abc
import inspect
import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils import lazyproperty

from photutils.aperture.bounding_box import BoundingBox
from photutils.utils._wcs_helpers import _pixel_scale_angle_at_skycoord

__all__ = ['Aperture', 'PixelAperture', 'SkyAperture']


class Aperture(metaclass=abc.ABCMeta):
    """
    Abstract base class for all apertures.
    """

    _params = ()

    def __len__(self):
        if self.isscalar:
            msg = f'A scalar {self.__class__.__name__!r} object has no len()'
            raise TypeError(msg)
        return self.shape[0]

    def __getitem__(self, index):
        if self.isscalar:
            msg = (f'A scalar {self.__class__.__name__!r} object cannot be '
                   'indexed')
            raise TypeError(msg)

        kwargs = {}
        for param in self._params:
            if param == 'positions':
                # slice the positions array
                kwargs[param] = getattr(self, param)[index]
            else:
                kwargs[param] = getattr(self, param)
        return self.__class__(**kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def _positions_str(self, prefix=None):
        if isinstance(self, PixelAperture):
            return np.array2string(self.positions, separator=', ',
                                   prefix=prefix)

        if isinstance(self, SkyAperture):
            return repr(self.positions)

        msg = 'Aperture must be a subclass of PixelAperture or SkyAperture'
        raise TypeError(msg)

    def __repr__(self):
        prefix = f'{self.__class__.__name__}'
        cls_info = []
        for param in self._params:
            if param == 'positions':
                cls_info.append(self._positions_str(prefix))
            else:
                cls_info.append(f'{param}={getattr(self, param)}')
        cls_info = ', '.join(cls_info)
        return f'<{prefix}({cls_info})>'

    def __str__(self):
        cls_info = [('Aperture', self.__class__.__name__)]
        for param in self._params:
            if param == 'positions':
                prefix = 'positions'
                cls_info.append((prefix, self._positions_str(prefix + ': ')))
            else:
                cls_info.append((param, getattr(self, param)))
        fmt = [f'{key}: {val}' for key, val in cls_info]
        return '\n'.join(fmt)

    def __eq__(self, other):
        """
        Equality operator for `Aperture`.

        All Aperture properties are compared for strict equality except
        for Quantity parameters, which allow for different units if they
        are directly convertible.
        """
        if not isinstance(other, self.__class__):
            return False

        self_params = list(self._params)
        other_params = list(other._params)

        # check that both have identical parameters
        if self_params != other_params:
            return False

        # now check the parameter values
        # Note that Quantity comparisons allow for different units if they
        # are directly convertible (e.g., 1.0 * u.deg == 60.0 * u.arcmin)
        try:
            for param in self_params:
                # np.any is used for SkyCoord array comparisons
                if np.any(getattr(self, param) != getattr(other, param)):
                    return False
        except TypeError:
            # TypeError is raised from SkyCoord comparison when they do
            # not have equivalent frames. Here return False instead of
            # the TypeError.
            return False

        return True

    def __ne__(self, other):
        """
        Inequality operator for `Aperture`.
        """
        return not self == other

    @property
    def _lazyproperties(self):
        """
        A list of all class lazyproperties (even in superclasses).
        """
        def islazyproperty(obj):
            return isinstance(obj, lazyproperty)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

    def copy(self):
        """
        Make an deep copy of this object.

        Returns
        -------
        result : `Aperture`
            A deep copy of the Aperture object.
        """
        params_copy = {}
        for param in list(self._params):
            params_copy[param] = deepcopy(getattr(self, param))
        return self.__class__(**params_copy)

    @abc.abstractmethod
    def positions(self):
        """
        The aperture positions, as an array of (x, y) coordinates or a
        `~astropy.coordinates.SkyCoord`.
        """
        msg = 'Needs to be implemented in a subclass'
        raise NotImplementedError(msg)

    @lazyproperty
    def shape(self):
        """
        The shape of the instance.
        """
        if isinstance(self.positions, SkyCoord):
            return self.positions.shape

        return self.positions.shape[:-1]

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (i.e., a single position).
        """
        return self.shape == ()


class PixelAperture(Aperture):
    """
    Abstract base class for apertures defined in pixel coordinates.
    """

    @lazyproperty
    def _default_patch_properties(self):
        """
        A dictionary of default matplotlib.patches.Patch properties.
        """
        mpl_params = {}

        # matplotlib.patches.Patch default is ``fill=True``
        mpl_params['fill'] = False

        return mpl_params

    @staticmethod
    def _translate_mask_mode(mode, subpixels, rectangle=False):
        if mode not in ('center', 'subpixel', 'exact'):
            msg = f'Invalid mask mode: {mode}'
            raise ValueError(msg)

        if rectangle and mode == 'exact':
            mode = 'subpixel'
            subpixels = 32

        if ((mode == 'subpixel')
                and (not isinstance(subpixels, int) or subpixels <= 0)):
            msg = 'subpixels must be a strictly positive integer'
            raise ValueError(msg)

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
        msg = 'Needs to be implemented in a subclass'
        raise NotImplementedError(msg)

    @lazyproperty
    def _positions(self):
        """
        The aperture positions, always as a 2D ndarray.
        """
        return np.atleast_2d(self.positions)

    @lazyproperty
    def _bbox(self):
        """
        The minimal bounding box for the aperture, always as a list of
        `~photutils.aperture.BoundingBox` instances.
        """
        x_delta, y_delta = self._xy_extents
        xmin = self._positions[:, 0] - x_delta
        xmax = self._positions[:, 0] + x_delta
        ymin = self._positions[:, 1] - y_delta
        ymax = self._positions[:, 1] + y_delta

        return [BoundingBox.from_float(x0, x1, y0, y1)
                for x0, x1, y0, y1 in zip(xmin, xmax, ymin, ymax, strict=True)]

    @lazyproperty
    def bbox(self):
        """
        The minimal bounding box for the aperture.

        If the aperture is scalar then a single
        `~photutils.aperture.BoundingBox` is returned, otherwise a list
        of `~photutils.aperture.BoundingBox` is returned.
        """
        if self.isscalar:
            return self._bbox[0]

        return self._bbox

    @lazyproperty
    def _centered_edges(self):
        """
        A list of ``(xmin, xmax, ymin, ymax)`` tuples, one for each
        position, of the pixel edges after recentering the aperture at
        the origin.

        These pixel edges are used by the low-level `photutils.geometry`
        functions.
        """
        edges = []
        for position, bbox in zip(self._positions, self._bbox, strict=True):
            xmin = bbox.ixmin - 0.5 - position[0]
            xmax = bbox.ixmax - 0.5 - position[0]
            ymin = bbox.iymin - 0.5 - position[1]
            ymax = bbox.iymax - 0.5 - position[1]
            edges.append((xmin, xmax, ymin, ymax))

        return edges

    @property
    @abc.abstractmethod
    def area(self):
        """
        The exact geometric area of the aperture shape.

        Use the `area_overlap` method to return the area of overlap
        between the data and the aperture, taking into account the
        aperture mask method, masked data pixels (``mask`` keyword), and
        partial/no overlap of the aperture with the data.

        Returns
        -------
        area : float
            The aperture area.

        See Also
        --------
        area_overlap
        """
        msg = 'Needs to be implemented in a subclass'
        raise NotImplementedError(msg)

    def area_overlap(self, data, *, mask=None, method='exact', subpixels=5):
        """
        Return the area of overlap between the data and the aperture.

        This method takes into account the aperture mask method, masked
        data pixels (``mask`` keyword), and partial/no overlap of the
        aperture with the data. In other words, it returns the area that
        used to compute the aperture sum (assuming identical inputs).

        Use the `area` method to calculate the exact analytical area of
        the aperture shape.

        Parameters
        ----------
        data : array_like or `~astropy.units.Quantity`
            A 2D array.

        mask : array_like (bool), optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked data are excluded from the area overlap.

        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture
            on the pixel grid. Not all options are available for all
            aperture types. Note that the more precise methods are
            generally slower. The following methods are available:

            * ``'exact'`` (default):
              The exact fractional overlap of the aperture and each
              pixel is calculated. The aperture weights will contain
              values between 0 and 1.

            * ``'center'``:
              A pixel is considered to be entirely in or out of the
              aperture depending on whether its center is in or out of
              the aperture. The aperture weights will contain values
              only of 0 (out) and 1 (in).

            * ``'subpixel'``:
              A pixel is divided into subpixels (see the ``subpixels``
              keyword), each of which are considered to be entirely in
              or out of the aperture depending on whether its center is
              in or out of the aperture. If ``subpixels=1``, this method
              is equivalent to ``'center'``. The aperture weights will
              contain values between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this
            factor in each dimension. That is, each pixel is divided
            into ``subpixels**2`` subpixels. This keyword is ignored
            unless ``method='subpixel'``.

        Returns
        -------
        areas : float or array_like
            The area (in pixels**2) of overlap between the data and the
            aperture.

        See Also
        --------
        area
        """
        apermasks = self.to_mask(method=method, subpixels=subpixels)
        if self.isscalar:
            apermasks = (apermasks,)

        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != data.shape:
                msg = 'mask and data must have the same shape'
                raise ValueError(msg)

        areas = []
        for apermask in apermasks:
            slc_large, slc_small = apermask.get_overlap_slices(data.shape)

            # if the aperture does not overlap the data return np.nan
            if slc_large is None:
                area = np.nan
            else:
                aper_weights = apermask.data[slc_small]
                if mask is not None:
                    aper_weights[mask[slc_large]] = 0.0
                area = np.sum(aper_weights)

            areas.append(area)

        areas = np.array(areas)
        if self.isscalar:
            return areas[0]

        return areas

    @abc.abstractmethod
    def to_mask(self, method='exact', subpixels=5):
        """
        Return a mask for the aperture.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture
            on the pixel grid. Not all options are available for all
            aperture types. Note that the more precise methods are
            generally slower. The following methods are available:

            * ``'exact'`` (default):
              The exact fractional overlap of the aperture and each
              pixel is calculated. The aperture weights will contain
              values between 0 and 1.

            * ``'center'``:
              A pixel is considered to be entirely in or out of the
              aperture depending on whether its center is in or out of
              the aperture. The aperture weights will contain values
              only of 0 (out) and 1 (in).

            * ``'subpixel'``:
              A pixel is divided into subpixels (see the ``subpixels``
              keyword), each of which are considered to be entirely in
              or out of the aperture depending on whether its center is
              in or out of the aperture. If ``subpixels=1``, this method
              is equivalent to ``'center'``. The aperture weights will
              contain values between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this
            factor in each dimension. That is, each pixel is divided
            into ``subpixels**2`` subpixels. This keyword is ignored
            unless ``method='subpixel'``.

        Returns
        -------
        mask : `~photutils.aperture.ApertureMask` or list of \
                `~photutils.aperture.ApertureMask`
            A mask for the aperture. If the aperture is scalar then
            a single `~photutils.aperture.ApertureMask` is returned,
            otherwise a list of `~photutils.aperture.ApertureMask` is
            returned.
        """
        msg = 'Needs to be implemented in a subclass'
        raise NotImplementedError(msg)

    def do_photometry(self, data, error=None, mask=None, method='exact',
                      subpixels=5):
        """
        Perform aperture photometry on the input data.

        Parameters
        ----------
        data : array_like or `~astropy.units.Quantity` instance
            The 2D array on which to perform photometry. ``data`` should
            be background subtracted.

        error : array_like or `~astropy.units.Quantity`, optional
            The pixel-wise Gaussian 1-sigma errors of the input
            ``data``. ``error`` is assumed to include *all* sources
            of error, including the Poisson error of the sources (see
            `~photutils.utils.calc_total_error`) . ``error`` must have
            the same shape as the input ``data``.

        mask : array_like (bool), optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked data are excluded from all calculations.

        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture
            on the pixel grid. Not all options are available for all
            aperture types. Note that the more precise methods are
            generally slower. The following methods are available:

            * ``'exact'`` (default):
              The exact fractional overlap of the aperture and each
              pixel is calculated. The aperture weights will contain
              values between 0 and 1.

            * ``'center'``:
              A pixel is considered to be entirely in or out of the
              aperture depending on whether its center is in or out of
              the aperture. The aperture weights will contain values
              only of 0 (out) and 1 (in).

            * ``'subpixel'``:
              A pixel is divided into subpixels (see the ``subpixels``
              keyword), each of which are considered to be entirely in
              or out of the aperture depending on whether its center is
              in or out of the aperture. If ``subpixels=1``, this method
              is equivalent to ``'center'``. The aperture weights will
              contain values between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this
            factor in each dimension. That is, each pixel is divided
            into ``subpixels**2`` subpixels. This keyword is ignored
            unless ``method='subpixel'``.

        Returns
        -------
        aperture_sums : `~numpy.ndarray` or `~astropy.units.Quantity`
            The sums within each aperture.

        aperture_sum_errs : `~numpy.ndarray` or `~astropy.units.Quantity`
            The errors on the sums within each aperture.

        Notes
        -----
        `RectangularAperture` and `RectangularAnnulus` photometry with
        the "exact" method uses a subpixel approximation by subdividing
        each data pixel by a factor of 1024 (``subpixels = 32``). For
        rectangular aperture widths and heights in the range from
        2 to 100 pixels, this subpixel approximation gives results
        typically within 0.001 percent or better of the exact value.
        The differences can be larger for smaller apertures (e.g.,
        aperture sizes of one pixel or smaller). For such small sizes,
        it is recommend to set ``method='subpixel'`` with a larger
        ``subpixels`` size.
        """
        data = np.asanyarray(data)
        if data.ndim != 2:
            msg = 'data must be a 2D array'
            raise ValueError(msg)

        if error is not None:
            error = np.asanyarray(error)
            if error.shape != data.shape:
                msg = 'error and data must have the same shape'
                raise ValueError(msg)

        # check Quantity inputs
        unit = {getattr(arr, 'unit', None) for arr in (data, error)
                if arr is not None}
        if len(unit) > 1:
            msg = ('If data or error has units, then they both must have '
                   'the same units')
            raise ValueError(msg)

        # strip data and error units for performance
        unit = unit.pop()
        if unit is not None:
            unit = data.unit
            data = data.value

            if error is not None:
                error = error.value

        apermasks = self.to_mask(method=method, subpixels=subpixels)
        if self.isscalar:
            apermasks = (apermasks,)

        aperture_sums = []
        aperture_sum_errs = []
        for apermask in apermasks:
            (slc_large,
             aper_weights,
             pixel_mask) = apermask._get_overlap_cutouts(data.shape, mask=mask)

            # no overlap of the aperture with the data
            if slc_large is None:
                aperture_sums.append(np.nan)
                aperture_sum_errs.append(np.nan)
                continue

            with warnings.catch_warnings():
                # ignore multiplication with non-finite data values
                warnings.simplefilter('ignore', RuntimeWarning)

                values = (data[slc_large] * aper_weights)[pixel_mask]
                aperture_sums.append(values.sum())

                if error is not None:
                    variance = (error[slc_large]**2 * aper_weights)[pixel_mask]
                    aperture_sum_errs.append(np.sqrt(variance.sum()))

        aperture_sums = np.array(aperture_sums)
        aperture_sum_errs = np.array(aperture_sum_errs)

        # apply units
        if unit is not None:
            aperture_sums <<= unit
            aperture_sum_errs <<= unit

        return aperture_sums, aperture_sum_errs

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

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        xy_positions : `~numpy.ndarray`
            The aperture patch positions.

        patch_params : dict
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.
        """
        xy_positions = deepcopy(self._positions)
        xy_positions[:, 0] -= origin[0]
        xy_positions[:, 1] -= origin[1]

        patch_params = self._default_patch_properties.copy()
        patch_params.update(kwargs)

        return xy_positions, patch_params

    @abc.abstractmethod
    def _to_patch(self, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.Patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.Patch` or list of \
                `~matplotlib.patches.Patch`
            A patch for the aperture. If the aperture is scalar then a
            single `~matplotlib.patches.Patch` is returned, otherwise a
            list of `~matplotlib.patches.Patch` is returned.
        """
        msg = 'Needs to be implemented in a subclass'
        raise NotImplementedError(msg)

    def plot(self, ax=None, origin=(0, 0), **kwargs):
        """
        Plot the aperture on a matplotlib `~matplotlib.axes.Axes`
        instance.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : list of `~matplotlib.patches.Patch`
            A list of matplotlib patches for the plotted aperture. The
            patches can be used, for example, when adding a plot legend.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        patches = self._to_patch(origin=origin, **kwargs)
        if self.isscalar:
            patches = (patches,)

        for patch in patches:
            ax.add_patch(patch)

        return patches

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
        sky_params : dict
            A dictionary of parameters for an equivalent sky aperture.
        """
        sky_params = {}
        xpos, ypos = np.transpose(self.positions)
        sky_params['positions'] = skypos = wcs.pixel_to_world(xpos, ypos)

        # Aperture objects require scalar shape parameters (e.g.,
        # radius, a, b, theta, etc.), therefore we must calculate the
        # pixel scale and angle at only a single sky position, which
        # we take as the first aperture position. For apertures with
        # multiple positions used with a WCS that contains distortions
        # (e.g., a spatially-dependent pixel scale), this may lead to
        # unexpected results (e.g., results that are dependent of the
        # order of the positions). There is no good way to fix this with
        # the current Aperture API allowing multiple positions.
        if not self.isscalar:
            skypos = skypos[0]
        _, pixscale, angle = _pixel_scale_angle_at_skycoord(skypos, wcs)

        for param in self._params:
            value = getattr(self, param)
            if param == 'positions':
                continue

            if param == 'theta':
                # photutils aperture sky angles are defined as the PA of
                # the semimajor axis (i.e., relative to the WCS latitude
                # axis). region sky angles are defined relative to the WCS
                # longitude axis.
                value = value - angle.to(u.rad)
            else:
                value = (value * u.pix * pixscale).to(u.arcsec)

            sky_params[param] = value

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
        msg = 'Needs to be implemented in a subclass'
        raise NotImplementedError(msg)


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
        pixel_params : dict
            A dictionary of parameters for an equivalent pixel aperture.
        """
        pixel_params = {}

        xpos, ypos = wcs.world_to_pixel(self.positions)
        pixel_params['positions'] = np.transpose((xpos, ypos))

        # Aperture objects require scalar shape parameters (e.g.,
        # radius, a, b, theta, etc.), therefore we must calculate the
        # pixel scale and angle at only a single sky position, which
        # we take as the first aperture position. For apertures with
        # multiple positions used with a WCS that contains distortions
        # (e.g., a spatially-dependent pixel scale), this may lead to
        # unexpected results (e.g., results that are dependent of the
        # order of the positions). There is no good way to fix this with
        # the current Aperture API allowing multiple positions.
        skypos = self.positions if self.isscalar else self.positions[0]
        _, pixscale, angle = _pixel_scale_angle_at_skycoord(skypos, wcs)

        for param in self._params:
            value = getattr(self, param)
            if param == 'positions':
                continue

            if param == 'theta':
                # photutils aperture sky angles are defined as the PA of
                # the semimajor axis (i.e., relative to the WCS latitude
                # axis). region sky angles are defined relative to the WCS
                # longitude axis.
                value = (value + angle).to(u.radian)
            elif value.unit.physical_type == 'angle':
                value = (value / pixscale).to(u.pixel).value
            else:
                value = value.value

            pixel_params[param] = value

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
        msg = 'Needs to be implemented in a subclass'
        raise NotImplementedError(msg)


def _aperture_metadata(aperture, index=''):
    """
    Return a dictionary of aperture metadata.

    Parameters
    ----------
    aperture : `Aperture`
        An aperture object.

    index : str, optional
        A string that will be prepended to each metadata key.

    Returns
    -------
    meta : dict
        A dictionary of aperture metadata
    """
    params = aperture._params
    meta = {}
    for param in params:
        if param != 'positions':
            meta[f'aperture{index}'] = aperture.__class__.__name__
            meta[f'aperture{index}_{param}'] = getattr(aperture, param)
    return meta
