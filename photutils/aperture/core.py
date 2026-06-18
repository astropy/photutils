# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Base aperture classes.
"""

import abc
import inspect
import re
import textwrap
import warnings
from copy import deepcopy

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils import lazyproperty

from photutils.aperture._batch_photometry import batch_aperture_sums
from photutils.aperture.bounding_box import BoundingBox
from photutils.aperture.mask import ApertureMask
from photutils.utils._deprecation import deprecated_positional_kwargs

__all__ = ['Aperture', 'PixelAperture', 'SkyAperture']


# Canonical descriptions of the ``method`` and ``subpixels`` parameters
# shared by many aperture and profile docstrings. The text is stored
# without leading indentation and is re-indented to match the
# placeholder by ``_update_method_subpixels_docstring``.
_METHOD_SUBPIXELS_DOC = """\
method : {'exact', 'center', 'subpixel'}, optional
    The method used to determine the pixel weights (the fraction
    of the pixel area covered by the aperture):

    * ``'exact'`` (default):
      Calculates the exact geometric overlap area. Weights are
      continuous in the range [0, 1].
    * ``'center'``:
      Binary weighting based on the pixel center. Weights are
      either 0 or 1. A pixel is included only if its center lies
      strictly inside the aperture; pixel centers lying exactly
      on the aperture boundary are excluded (weight 0).
    * ``'subpixel'``:
      Approximates the overlap by averaging binary samples on a
      subgrid. The number of samples is set by the ``subpixels``
      parameter. Weights are discrete in the range [0, 1]. A
      subpixel is included only if its center lies strictly
      inside the aperture; subpixel centers lying exactly on the
      aperture boundary are excluded (weight 0).

subpixels : int, optional
    The subsampling factor per axis used when
    ``method='subpixel'``. Each pixel is divided into a grid of
    ``subpixels**2`` subpixels to approximate the overlap. This
    parameter is ignored for other methods."""

_METHOD_SUBPIXELS_PLACEHOLDER = re.compile(
    r'^([ \t]*)<method_subpixels_descriptions>[ \t]*$', re.MULTILINE)


def _update_method_subpixels_docstring(obj):
    """
    Decorator to insert the standard ``method`` and ``subpixels``
    parameter descriptions into a docstring.

    The descriptions replace a ``<method_subpixels_descriptions>``
    placeholder, which must appear on its own line. The placeholder's
    indentation is applied to the inserted text, so the same source
    descriptions can be used at any docstring indentation level.

    Parameters
    ----------
    obj : function or type
        The function or class whose docstring will be updated.

    Returns
    -------
    obj : function or type
        The input ``obj`` with its ``__doc__`` updated in place.
    """
    docstring = obj.__doc__
    if docstring is None:
        return obj

    def replace(match):
        indent = match.group(1)
        return textwrap.indent(_METHOD_SUBPIXELS_DOC, indent)

    obj.__doc__ = _METHOD_SUBPIXELS_PLACEHOLDER.sub(replace, docstring)
    return obj


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
                # Slice the positions array
                kwargs[param] = getattr(self, param)[index]
            else:
                kwargs[param] = getattr(self, param)
        return self.__class__(**kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def _positions_str(self, *, prefix=None):
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
                cls_info.append(self._positions_str(prefix=prefix))
            else:
                cls_info.append(f'{param}={getattr(self, param)}')
        cls_info = ', '.join(cls_info)
        return f'<{prefix}({cls_info})>'

    def __str__(self):
        cls_info = [('Aperture', self.__class__.__name__)]
        for param in self._params:
            if param == 'positions':
                prefix = 'positions'
                cls_info.append((prefix,
                                 self._positions_str(prefix=prefix + ': ')))
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

        # Check that both have identical parameters
        if self_params != other_params:
            return False

        # Now check the parameter values.
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
        Make a deep copy of this object.

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
    def _translate_mask_method(method, subpixels):
        """
        Translate the mask method and subpixels parameters to the values
        used by the low-level `photutils.geometry` functions.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}
            The mask method.

        subpixels : int
            The number of subpixels for the 'subpixel' method.

        Returns
        -------
        use_exact : int
            Whether to use exact method (1) or not (0).

        subpixels : int
            The number of subpixels for subpixel method.
        """
        if method not in ('center', 'subpixel', 'exact'):
            msg = f'Invalid mask method: {method}'
            raise ValueError(msg)

        if ((method == 'subpixel')
                and (not isinstance(subpixels, int) or subpixels <= 0)):
            msg = 'subpixels must be a strictly positive integer'
            raise ValueError(msg)

        if method == 'center':
            use_exact = 0
            subpixels = 1
        elif method == 'subpixel':
            use_exact = 0
        elif method == 'exact':
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

    @_update_method_subpixels_docstring
    def area_overlap(self, data, *, mask=None, method='exact', subpixels=5):
        # numpydoc ignore: PR01,PR02,PR04,PR07
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

        <method_subpixels_descriptions>

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

            # If the aperture does not overlap the data, return np.nan
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

    @_update_method_subpixels_docstring
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def to_mask(self, method='exact', subpixels=5):
        """
        Return a mask for the aperture.

        Parameters
        ----------
        <method_subpixels_descriptions>

        Returns
        -------
        mask : `~photutils.aperture.ApertureMask` or list of \
                `~photutils.aperture.ApertureMask`
            A mask for the aperture. If the aperture is scalar then
            a single `~photutils.aperture.ApertureMask` is returned,
            otherwise a list of `~photutils.aperture.ApertureMask` is
            returned.
        """
        use_exact, subpixels = self._translate_mask_method(method, subpixels)

        masks = []
        for bbox, edges in zip(self._bbox, self._centered_edges, strict=True):
            ny, nx = bbox.shape
            overlap = self._compute_overlap(
                edges, nx, ny, use_exact, subpixels)
            masks.append(ApertureMask(overlap, bbox))

        if self.isscalar:
            return masks[0]

        return masks

    @abc.abstractmethod
    def _compute_overlap(self, edges, nx, ny, use_exact, subpixels):
        """
        Compute the overlap of the aperture for a single position.

        Parameters
        ----------
        edges : tuple of float
            The ``(xmin, xmax, ymin, ymax)`` pixel edges centered at
            the origin.

        nx, ny : int
            The number of pixels in x and y.

        use_exact : int
            Whether to use exact method (1) or not (0).

        subpixels : int
            The number of subpixels for subpixel method.

        Returns
        -------
        overlap : 2D `~numpy.ndarray`
            The overlap array.
        """

    def _do_mask_photometry(self, data, *, error, mask, method, subpixels):
        """
        Perform aperture photometry using per-source aperture masks.

        This is the fallback code path for apertures or inputs that are
        not supported by the batch Cython driver.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The 2D array on which to perform photometry, with any units
            already stripped.

        error, mask, method, subpixels
            See `do_photometry`. Any units must already be stripped from
            ``error``.

        Returns
        -------
        aperture_sums, aperture_sum_errs : `~numpy.ndarray`
            The aperture sums and errors.
        """
        apermasks = self.to_mask(method=method, subpixels=subpixels)
        if self.isscalar:
            apermasks = (apermasks,)

        aperture_sums = []
        aperture_sum_errs = []
        with warnings.catch_warnings():
            # Ignore multiplication with non-finite data values
            warnings.simplefilter('ignore', RuntimeWarning)

            for apermask in apermasks:
                (slc_large,
                 aper_weights,
                 pixel_mask) = apermask._get_overlap_cutouts(data.shape,
                                                             mask=mask)

                # No overlap of the aperture with the data
                if slc_large is None:
                    aperture_sums.append(np.nan)
                    aperture_sum_errs.append(np.nan)
                    continue

                values = (data[slc_large] * aper_weights)[pixel_mask]
                aperture_sums.append(values.sum())

                if error is not None:
                    variance = (error[slc_large]**2
                                * aper_weights)[pixel_mask]
                    aperture_sum_errs.append(np.sqrt(variance.sum()))

        return np.array(aperture_sums), np.array(aperture_sum_errs)

    def _batch_shape_params(self):
        """
        The aperture shape code and parameters for the batch Cython
        photometry driver.

        Returns
        -------
        spec : tuple or `None`
            A ``(shape_code, params)`` tuple, where
            ``shape_code`` is one of the shape codes defined in
            `photutils.aperture._batch_photometry` and ``params``
            is a tuple of the aperture shape parameters expected by
            `~photutils.aperture._batch_photometry.batch_aperture_sums`
            for that shape. `None` is returned if batch photometry is
            not supported for this aperture, in which case the slower
            mask-based code path is used.

        Notes
        -----
        The batch driver is used only if this hook is defined in the
        aperture instance's own class (see `_do_batch_photometry`),
        so subclasses must define this method (e.g., by calling
        ``super()``) to opt in to the batch code path.
        """
        return

    def _do_batch_photometry(self, data, *, error, mask, method, subpixels):
        """
        Perform aperture photometry using the batch Cython driver.

        The batch driver computes results identical to the mask-based
        code path, but without creating per-source mask arrays or making
        per-source Python calls.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The 2D array on which to perform photometry, with any units
            already stripped.

        error, mask, method, subpixels
            See `do_photometry`. Any units must already be stripped from
            ``error``.

        Returns
        -------
        result : tuple of `~numpy.ndarray` or `None`
            A ``(aperture_sums, aperture_sum_errs)`` tuple of float64
            arrays, or `None` if the batch driver does not support this
            aperture or these inputs (in which case the caller should
            use the mask-based code path).
        """
        # Use the batch driver only if the aperture's own class defines
        # the _batch_shape_params hook. Subclasses that do not define
        # it may override other behavior (e.g., to_mask) that the batch
        # driver would not honor, so they use the mask-based code path.
        if '_batch_shape_params' not in type(self).__dict__:
            return None

        spec = self._batch_shape_params()
        if spec is None:
            return None

        def _supported(arr):
            return (type(arr) is np.ndarray and arr.dtype.kind in 'fiub'
                    and arr.dtype.itemsize <= 8)

        if not _supported(data) or (error is not None
                                    and not _supported(error)):
            return None

        if mask is not None:
            if (not isinstance(mask, np.ndarray) or mask.dtype != bool
                    or mask.shape != data.shape):
                return None
            mask = np.ascontiguousarray(mask, dtype=np.uint8)

        use_exact, subpixels = self._translate_mask_method(method, subpixels)

        shape_code, params = spec
        if error is not None:
            error = np.ascontiguousarray(error, dtype=np.float64)
        ext_x, ext_y = self._xy_extents

        sums, errs, overlap = batch_aperture_sums(
            np.ascontiguousarray(data, dtype=np.float64), error, mask,
            np.ascontiguousarray(self._positions, dtype=np.float64),
            shape_code, np.array(params, dtype=np.float64),
            float(ext_x), float(ext_y), use_exact, subpixels)

        if error is None:
            # Match the mask-based path, which collects one NaN per
            # non-overlapping source when error is not input.
            errs = np.full(np.count_nonzero(~overlap), np.nan)

        return sums, errs

    @_update_method_subpixels_docstring
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def do_photometry(self, data, error=None, mask=None, method='exact',
                      subpixels=5):
        # numpydoc ignore: PR01,PR02,PR04,PR07
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
            `~photutils.utils.calc_total_error`). ``error`` must have
            the same shape as the input ``data``.

        mask : array_like (bool), optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked data are excluded from all calculations.

        <method_subpixels_descriptions>

        Returns
        -------
        aperture_sums : `~numpy.ndarray` or `~astropy.units.Quantity`
            The sum within each aperture. The values are always
            float64, regardless of the input ``data`` dtype.

        aperture_sum_errs : `~numpy.ndarray` or `~astropy.units.Quantity`
            The errors on the sum within each aperture. The values are
            always float64, regardless of the input ``error`` dtype.
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

        # Check Quantity inputs
        unit = {getattr(arr, 'unit', None) for arr in (data, error)
                if arr is not None}
        if len(unit) > 1:
            msg = ('If data or error has units, then they both must have '
                   'the same units')
            raise ValueError(msg)

        # Strip data and error units for performance
        unit = unit.pop()
        if unit is not None:
            unit = data.unit
            data = data.value

            if error is not None:
                error = error.value

        result = self._do_batch_photometry(data, error=error, mask=mask,
                                           method=method,
                                           subpixels=subpixels)

        if result is not None:
            aperture_sums, aperture_sum_errs = result
        else:
            aperture_sums, aperture_sum_errs = self._do_mask_photometry(
                data, error=error, mask=mask, method=method,
                subpixels=subpixels)

        # Apply units
        if unit is not None:
            aperture_sums <<= unit
            aperture_sum_errs <<= unit

        return aperture_sums, aperture_sum_errs

    @staticmethod
    def _make_annulus_path(patch_inner, patch_outer):
        """
        Define a matplotlib annulus path from two patches.

        This preserves the cubic Bézier curves (CURVE4) of the aperture
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

    def _define_patch_params(self, *, origin=(0, 0), **kwargs):
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
    def _to_patch(self, *, origin=(0, 0), **kwargs):
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

    @deprecated_positional_kwargs(since='3.0', until='4.0')
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


class SkyAperture(Aperture):
    """
    Abstract base class for all apertures defined in celestial
    coordinates.
    """

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


def _aperture_metadata(aperture, *, index=''):
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
    meta[f'aperture{index}'] = aperture.__class__.__name__
    for param in params:
        if param != 'positions':
            meta[f'aperture{index}_{param}'] = getattr(aperture, param)
    return meta
