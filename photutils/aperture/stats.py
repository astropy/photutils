# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for calculating the properties of sources
defined by an Aperture.
"""

import functools
import inspect
import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, StdDevUncertainty
from astropy.stats import (SigmaClip, biweight_location, biweight_midvariance,
                           mad_std)
from astropy.table import QTable
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import Aperture, SkyAperture
from photutils.utils._misc import _get_meta
from photutils.utils._moments import _moments, _moments_central
from photutils.utils._quantity_helpers import process_quantities

__all__ = ['ApertureStats']


# default table columns for `to_table()` output
DEFAULT_COLUMNS = ['id', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'sum', 'sum_err', 'sum_aper_area', 'center_aper_area',
                   'min', 'max', 'mean', 'median', 'mode', 'std',
                   'mad_std', 'var', 'biweight_location',
                   'biweight_midvariance', 'fwhm', 'semimajor_sigma',
                   'semiminor_sigma', 'orientation', 'eccentricity']


def as_scalar(method):
    """
    Return a scalar value from a method if the class is scalar.
    """

    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        result = method(*args, **kwargs)
        try:
            return (result[0] if args[0].isscalar and len(result) == 1
                    else result)
        except TypeError:  # if result has no len
            return result

    return _decorator


class ApertureStats:
    """
    Class to create a catalog of statistics for pixels within an
    aperture.

    Note that this class returns the statistics of the input
    ``data`` values within the aperture. It does not convert data
    in surface brightness units to flux or counts. Conversion from
    surface-brightness units should be performed before using this
    function.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`, `~astropy.units.Quantity`, `~astropy.nddata.NDData`
        The 2D array from which to calculate the source properties.
        For accurate source properties, ``data`` should be
        background-subtracted. Non-finite ``data`` values (NaN and inf)
        are automatically masked.

    aperture : `~photutils.aperture.Aperture`
        The aperture to apply to the data. The aperture object
        may contain more than one position. If ``aperture`` is a
        `~photutils.aperture.SkyAperture` object, then a WCS must be
        input using the ``wcs`` keyword.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The total error array corresponding to the input ``data``
        array. ``error`` is assumed to include *all* sources of
        error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) . ``error`` must have
        the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity` array then ``error`` must be a
        `~astropy.units.Quantity` array (and vice versa) with identical
        units. Non-finite ``error`` values (NaN and +/- inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array. Such pixels can
        be masked using the ``mask`` keyword.

    mask : 2D `~numpy.ndarray` (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations. Non-finite
        values (NaN and inf) in the input ``data`` are automatically
        masked.

    wcs : WCS object or `None`, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). ``wcs`` is required if the
        input ``aperture`` is a `~photutils.aperture.SkyAperture`. If
        `None`, then all sky-based properties will be set to `None`.

    sigma_clip : `None` or `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will
        be performed.

    sum_method : {'exact', 'center', 'subpixel'}, optional
        The method used to determine the overlap of the aperture on
        the pixel grid. This method is used only for calculating the
        ``sum``, ``sum_error``, ``sum_aper_area``, ``data_sumcutout``,
        and ``error_sumcutout`` properties. All other properties use the
        "center" aperture mask method. Not all options are available for
        all aperture types. The following methods are available:

          * ``'exact'`` (default):
            The the exact fractional overlap of the aperture and each
            pixel is calculated. The aperture weights will contain
            values between 0 and 1.

          * ``'center'``:
            A pixel is considered to be entirely in or out of the
            aperture depending on whether its center is in or out of the
            aperture. The aperture weights will contain values only of 0
            (out) and 1 (in).

          * ``'subpixel'``:
            A pixel is divided into subpixels (see the ``subpixels``
            keyword), each of which are considered to be entirely in
            or out of the aperture depending on whether its center is
            in or out of the aperture. If ``subpixels=1``, this method
            is equivalent to ``'center'``. The aperture weights will
            contain values between 0 and 1.

    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor
        in each dimension. That is, each pixel is divided into
        ``subpixels**2`` subpixels. This keyword is ignored unless
        ``sum_method='subpixel'``.

    local_bkg : float, `~numpy.ndarray`,  `~astropy.units.Quantity`, or `None`
        The *per-pixel* local background values to subtract from the
        data before performing measurements. If input as any array,
        the order of ``local_bkg`` values corresponds to the order
        of the input ``aperture`` positions. ``local_bkg`` must have
        the same length as the the input ``aperture`` or must be a
        scalar value, which will be broadcast to all apertures. If
        `None`, then no local background subtraction is performed. If
        the input ``data`` has units, then ``local_bkg`` must be a
        `~astropy.units.Quantity` with the same units.

    Notes
    -----
    ``data`` should be background-subtracted for accurate source
    properties. In addition to global background subtraction, local
    background subtraction can be performed using the ``local_bkg``
    keyword values.

    Most source properties are calculated using the "center"
    aperture-mask method, which gives aperture weights of 0 or 1. This
    avoids the need to compute weighted statistics --- the ``data``
    pixel values are directly used.

    The input ``sum_method`` and ``subpixels`` keywords are used
    to determine the aperture-mask method when calculating the
    sum-related properties: ``sum``, ``sum_error``, ``sum_aper_area``,
    ``data_sumcutout``, and ``error_sumcutout``. The default is
    ``sum_method='exact'``, which produces exact aperture-weighted
    photometry.

    .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/

    Examples
    --------
    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils.aperture import CircularAperture, ApertureStats

    >>> data = make_4gaussians_image()
    >>> aper = CircularAperture((150, 25), 8)
    >>> aperstats = ApertureStats(data, aper)
    >>> print(aperstats.xcentroid)  # doctest: +FLOAT_CMP
    149.98737072209013
    >>> print(aperstats.ycentroid)  # doctest: +FLOAT_CMP
    24.99729176183652
    >>> print(aperstats.centroid)  # doctest: +FLOAT_CMP
    [149.98737072  24.99729176]

    >>> print(aperstats.mean, aperstats.median, aperstats.std) #  doctest: +FLOAT_CMP
    46.861845146453526 33.743501730319 38.25291812758177

    >>> print(aperstats.sum)  # doctest: +FLOAT_CMP
    9118.129697119366

    >>> print(aperstats.sum_aper_area) # doctest: +FLOAT_CMP
    201.0619298297468 pix2

    >>> # more than one aperture position
    >>> aper2 = CircularAperture(((150, 25), (90, 60)), 10)
    >>> aperstats2 = ApertureStats(data, aper2)
    >>> print(aperstats2.xcentroid)  # doctest: +FLOAT_CMP
    [149.97230436  90.00833613]
    >>> print(aperstats2.sum)  # doctest: +FLOAT_CMP
    [ 9863.56195844 36629.52906175]
    """

    def __init__(self, data, aperture, *, error=None, mask=None, wcs=None,
                 sigma_clip=None, sum_method='exact', subpixels=5,
                 local_bkg=None):

        if isinstance(data, NDData):
            data, error, mask, wcs = self._unpack_nddata(data, error, mask,
                                                         wcs)

        (data, error, local_bkg), unit = process_quantities(
            (data, error, local_bkg), ('data', 'error', 'local_bkg'))
        self._data = self._validate_array(data, 'data', shape=False)
        self._data_unit = unit
        self.aperture = self._validate_aperture(aperture)

        if isinstance(aperture, SkyAperture):
            if wcs is None:
                raise ValueError('A wcs is required when using a SkyAperture')

        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._wcs = wcs

        if sigma_clip is not None and not isinstance(sigma_clip, SigmaClip):
            raise TypeError('sigma_clip must be a SigmaClip instance')
        self.sigma_clip = sigma_clip

        self.sum_method = sum_method
        self.subpixels = subpixels

        self._local_bkg = np.zeros(self.n_apertures)  # no local bkg
        if local_bkg is not None:
            local_bkg = np.atleast_1d(local_bkg)
            if local_bkg.ndim != 1:
                raise ValueError('local_bkg must be a 1D array')

            n_local_bkg = len(local_bkg)
            if n_local_bkg != 1 and n_local_bkg != self.n_apertures:
                raise ValueError('local_bkg must be scalar or have the same '
                                 'length as the input aperture')
            local_bkg = np.broadcast_to(local_bkg, self.n_apertures)

            if np.any(~np.isfinite(local_bkg)):
                raise ValueError('local_bkg must not contain any non-finite '
                                 '(e.g., inf or NaN) values')
            self._local_bkg = local_bkg  # always an iterable

        self._ids = np.arange(self.n_apertures) + 1
        self.default_columns = DEFAULT_COLUMNS
        self.meta = _get_meta()

    @staticmethod
    def _unpack_nddata(data, error, mask, wcs):
        nddata_attr = {'error': error, 'mask': mask, 'wcs': wcs}
        for key, value in nddata_attr.items():
            if value is not None:
                warnings.warn(f'The {key!r} keyword is be ignored. Its '
                              'value is obtained from the input NDData '
                              'object.', AstropyUserWarning)

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

        return data, error, mask, wcs

    @staticmethod
    def _validate_aperture(aperture):
        if not isinstance(aperture, Aperture):
            raise TypeError('aperture must be an Aperture object')
        return aperture

    def _validate_array(self, array, name, ndim=2, shape=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != ndim:
                raise ValueError(f'{name} must be a {ndim}D array.')
            if shape and array.shape != self._data.shape:
                raise ValueError(f'data and {name} must have the same shape.')
        return array

    @property
    def _lazyproperties(self):
        """
        A list of all class lazyproperties (even in superclasses).
        """

        def islazyproperty(obj):
            return isinstance(obj, lazyproperty)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

    @property
    def properties(self):
        """
        A sorted list of built-in source properties.
        """
        lazyproperties = [name for name in self._lazyproperties if not
                          name.startswith('_')]
        lazyproperties.sort()
        return lazyproperties

    def __getitem__(self, index):
        if self.isscalar:
            raise TypeError(f'A scalar {self.__class__.__name__!r} object '
                            'cannot be indexed')

        newcls = object.__new__(self.__class__)

        # attributes defined in __init__ that are copied directly to the
        # new class
        init_attr = ('_data', '_data_unit', '_error', '_mask', '_wcs',
                     'sigma_clip', 'sum_method', 'subpixels',
                     'default_columns', 'meta')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # need to slice _aperture and _ids;
        # aperture determines isscalar (needed below)
        attrs = ('aperture', '_ids')
        for attr in attrs:
            setattr(newcls, attr, getattr(self, attr)[index])

        # slice evaluated lazyproperty objects
        keys = set(self.__dict__.keys()) & set(self._lazyproperties)
        keys.add('_local_bkg')  # iterable defined in __init__
        for key in keys:
            value = self.__dict__[key]

            # do not insert attributes that are always scalar (e.g.,
            # isscalar, n_apertures), i.e., not an array/list for each
            # source
            if np.isscalar(value):
                continue

            try:
                # keep most _<attrs> as length-1 iterables
                if (newcls.isscalar and key.startswith('_')
                        and key != '_pixel_aperture'):
                    if isinstance(value, np.ndarray):
                        val = value[:, np.newaxis][index]
                    else:
                        val = [value[index]]
                else:
                    val = value[index]
            except TypeError:
                # apply fancy indices (e.g., array/list or bool
                # mask) to lists
                # see https://numpy.org/doc/stable/release/1.20.0-notes.html
                # #arraylike-objects-which-do-not-define-len-and-getitem
                arr = np.empty(len(value), dtype=object)
                arr[:] = list(value)
                val = arr[index].tolist()

            newcls.__dict__[key] = val
        return newcls

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'Length: {self.n_apertures}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.isscalar:
            raise TypeError(f'Scalar {self.__class__.__name__!r} object has '
                            'no len()')
        return self.n_apertures

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single aperture position).
        """
        return self._pixel_aperture.isscalar

    def copy(self):
        """
        Return a deep copy of this object.
        """
        return deepcopy(self)

    @lazyproperty
    def _null_object(self):
        """
        Return `None` values.
        """
        return np.array([None] * self.n_apertures)

    @lazyproperty
    def _null_value(self):
        """
        Return np.nan values.
        """
        values = np.empty(self.n_apertures)
        values.fill(np.nan)
        return values

    @property
    @as_scalar
    def id(self):
        """
        The aperture identification number(s).
        """
        return self._ids

    @property
    def ids(self):
        """
        The aperture identification number(s), always as an iterable
        `~numpy.ndarray`.
        """
        _ids = self._ids
        if self.isscalar:
            _ids = np.array((_ids,))
        return _ids

    def get_id(self, id_num):
        """
        Return a new `ApertureStats` object for the input ID number
        only.

        Parameters
        ----------
        id_num : int
            The aperture ID number.

        Returns
        -------
        result : `ApertureStats`
            A new `ApertureStats` object containing only the source with
            the input ID number.
        """
        return self.get_ids(id_num)

    def get_ids(self, id_nums):
        """
        Return a new `ApertureStats` object for the input ID numbers
        only.

        Parameters
        ----------
        id_nums : list, tuple, or `~numpy.ndarray` of int
            The aperture ID number(s).

        Returns
        -------
        result : `ApertureStats`
            A new `ApertureStats` object containing only the sources with
            the input ID numbers.
        """
        for id_num in np.atleast_1d(id_nums):
            if id_num not in self.ids:
                raise ValueError(f'{id_num} is not a valid source ID number')

        sorter = np.argsort(self.id)
        indices = sorter[np.searchsorted(self.id, id_nums, sorter=sorter)]
        return self[indices]

    def to_table(self, columns=None):
        """
        Create a `~astropy.table.QTable` of source properties.

        Parameters
        ----------
        columns : str, list of str, `None`, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`. The allowed column names are any of
            the `ApertureStats` properties. If ``columns`` is `None`,
            then a default list of scalar-valued properties (as defined
            by the ``default_columns`` attribute) will be used.

        Returns
        -------
        table : `~astropy.table.QTable`
            A table of sources properties with one row per source.
        """
        if columns is None:
            table_columns = self.default_columns
        else:
            table_columns = np.atleast_1d(columns)

        tbl = QTable()
        tbl.meta.update(self.meta)  # keep tbl.meta type

        for column in table_columns:
            values = getattr(self, column)

            # column assignment requires an object with a length
            if self.isscalar:
                values = (values,)

            tbl[column] = values
        return tbl

    @lazyproperty
    def n_apertures(self):
        """
        The number of positions in the input aperture.
        """
        if self.isscalar:
            return 1
        return len(self._pixel_aperture)

    @lazyproperty
    def _pixel_aperture(self):
        """
        The input aperture as a PixelAperture.
        """
        if isinstance(self.aperture, SkyAperture):
            return self.aperture.to_pixel(self._wcs)
        return self.aperture

    @lazyproperty
    def _aperture_masks_center(self):
        """
        The aperture masks (`ApertureMask`) generated with the 'center'
        method, always as an iterable.
        """
        aperture_masks = self._pixel_aperture.to_mask(method='center')
        if self.isscalar:
            aperture_masks = (aperture_masks,)
        return aperture_masks

    @lazyproperty
    def _aperture_masks(self):
        """
        The aperture masks (`ApertureMask`) generated with the
        ``sum_method`` method, always as an iterable.
        """
        aperture_masks = self._pixel_aperture.to_mask(method=self.sum_method,
                                                      subpixels=self.subpixels)
        if self.isscalar:
            aperture_masks = (aperture_masks,)
        return aperture_masks

    @lazyproperty
    def _overlap_slices(self):
        """
        The aperture mask overlap slices with the data, always as an
        iterable.

        The overlap slices are the same for all aperture mask methods.
        """
        overlap_slices = []
        for apermask in self._aperture_masks_center:
            (slc_large, slc_small) = apermask.get_overlap_slices(
                self._data.shape)
            overlap_slices.append((slc_large, slc_small))
        return overlap_slices

    @lazyproperty
    def _data_cutouts(self):
        """
        The local-background-subtracted unmasked data cutouts using the
        aperture bounding box, always as a iterable.
        """
        cutouts = []
        for (slices, local_bkg) in zip(self._overlap_slices,
                                       self._local_bkg):
            if slices[0] is None:
                cutout = None  # no aperture overlap with the data
            else:
                # copy is needed to preserve input data because masks are
                # applied to these cutouts later
                cutout = (self._data[slices[0]].astype(float, copy=True)
                          - local_bkg)
            cutouts.append(cutout)
        return cutouts

    def _make_aperture_cutouts(self, aperture_masks):
        """
        Make aperture-weighted cutouts for the data and variance, and
        cutouts for the total mask and aperture mask weights.

        Parameters
        ----------
        aperture_masks : list of `ApertureMask`
            A list of `ApertureMask` objects.

        Returns
        -------
        data, variance, mask, weights : list of `~numpy.ndarray`
            A list of cutout arrays for the data, variance, mask and weight
            arrays for each source (aperture position).
        """
        data_cutouts = []
        variance_cutouts = []
        mask_cutouts = []
        weight_cutouts = []
        overlaps = []

        for (data_cutout, apermask, slices) in zip(
                self._data_cutouts, aperture_masks, self._overlap_slices):

            slc_large, slc_small = slices
            if slc_large is None:  # aperture does not overlap the data
                overlap = False
                data_cutout = np.array([np.nan])
                variance_cutout = np.array([np.nan])
                mask_cutout = np.array([False])
                weight_cutout = np.array([np.nan])
            else:
                # create a mask of non-finite ``data`` values combined
                # with the input ``mask`` array.
                data_mask = ~np.isfinite(data_cutout)
                if self._mask is not None:
                    data_mask |= self._mask[slc_large]

                overlap = True
                aperweight_cutout = apermask.data[slc_small]
                weight_cutout = aperweight_cutout * ~data_mask

                # apply the aperture mask; for "exact" and "subpixel"
                # this is an expanded boolean mask using the aperture
                # mask zero values
                mask_cutout = (aperweight_cutout == 0) | data_mask

                data_cutout = data_cutout.copy()
                if self.sigma_clip is None:
                    # data_cutout will have zeros where mask_cutout is True
                    data_cutout *= ~mask_cutout
                else:
                    # to input a mask, SigmaClip needs a MaskedArray
                    data_cutout_ma = np.ma.masked_array(data_cutout,
                                                        mask=mask_cutout)
                    data_sigclip = self.sigma_clip(data_cutout_ma)

                    # define a mask of only the sigma-clipped pixels
                    sigclip_mask = data_sigclip.mask & ~mask_cutout
                    weight_cutout *= ~sigclip_mask

                    mask_cutout = data_sigclip.mask
                    data_cutout = data_sigclip.filled(0.0)

                # need to apply the aperture weights
                data_cutout *= aperweight_cutout

                if self._error is None:
                    variance_cutout = None
                else:
                    # apply the exact weights and total mask;
                    # error_cutout will have zeros where mask_cutout is True
                    variance = self._error[slc_large]**2
                    variance_cutout = (variance * aperweight_cutout
                                       * ~mask_cutout)

            data_cutouts.append(data_cutout)
            variance_cutouts.append(variance_cutout)
            mask_cutouts.append(mask_cutout)
            weight_cutouts.append(weight_cutout)
            overlaps.append(overlap)

        # use zip (instead of np.transpose) because these may contain
        # arrays that have different shapes
        return list(zip(data_cutouts, variance_cutouts, mask_cutouts,
                        weight_cutouts, overlaps))

    @lazyproperty
    def _aperture_cutouts_center(self):
        """
        Aperture-weighted cutouts for the data, variance, total mask, and
        aperture weights using the "center" aperture mask method.
        """
        return self._make_aperture_cutouts(self._aperture_masks_center)

    @lazyproperty
    def _aperture_cutouts(self):
        """
        Aperture-weighted cutouts for the data, variance, total mask, and
        aperture weights using the input ``sum_method`` aperture mask
        method.
        """
        return self._make_aperture_cutouts(self._aperture_masks)

    @lazyproperty
    def _mask_cutout_center(self):
        """
        Boolean mask cutouts representing the total mask.

        The total mask is combination of the input ``mask``, non-finite
        ``data`` values, the cutout aperture mask using the "center"
        method, and the sigma-clip mask.
        """
        return list(zip(*self._aperture_cutouts_center))[2]

    @lazyproperty
    def _mask_cutout(self):
        """
        Boolean mask cutouts representing the total mask.

        The total mask is combination of the input ``mask``,
        non-finite ``data`` values, the cutout aperture mask using the
        ``sum_method`` method, and the sigma-clip mask.
        """
        return list(zip(*self._aperture_cutouts))[2]

    def _make_masked_array_center(self, array):
        """
        Return a list of cutout masked arrays using the ``_mask_cutout``
        mask.

        Units are not applied.
        """
        return [np.ma.masked_array(arr, mask=mask)
                for arr, mask in zip(array, self._mask_cutout_center)]

    def _make_masked_array(self, array):
        """
        Return a list of cutout masked arrays using the
        ``_mask_sumcutout`` mask.

        Units are not applied.
        """
        return [np.ma.masked_array(arr, mask=mask)
                for arr, mask in zip(array, self._mask_cutout)]

    @lazyproperty
    @as_scalar
    def data_cutout(self):
        """
        A 2D aperture-weighted cutout from the data using the aperture
        mask with the "center" method as a `~numpy.ma.MaskedArray`.

        The cutout does not have units due to current limitations of
        masked quantity arrays.

        The mask is `True` for pixels from the input ``mask``,
        non-finite ``data`` values (NaN and inf), sigma-clipped pixels
        within the aperture, and pixels where the aperture mask has zero
        weight.
        """
        return self._make_masked_array_center(
            list(zip(*self._aperture_cutouts_center))[0])

    @lazyproperty
    @as_scalar
    def data_sumcutout(self):
        """
        A 2D aperture-weighted cutout from the data using the
        aperture mask with the input ``sum_method`` method as a
        `~numpy.ma.MaskedArray`.

        The cutout does not have units due to current limitations of
        masked quantity arrays.

        The mask is `True` for pixels from the input ``mask``,
        non-finite ``data`` values (NaN and inf), sigma-clipped pixels
        within the aperture, and pixels where the aperture mask has zero
        weight.
        """
        return self._make_masked_array(list(zip(*self._aperture_cutouts))[0])

    @lazyproperty
    def _variance_cutout_center(self):
        """
        A 2D aperture-weighted variance cutout using the aperture mask
        with the input "center" method as a `~numpy.ma.MaskedArray`.

        The cutout does not have units due to current limitations of
        masked quantity arrays.

        The mask is `True` for pixels from the input ``mask``,
        non-finite ``data`` values (NaN and inf), sigma-clipped pixels
        within the aperture, and pixels where the aperture mask has zero
        weight.
        """
        if self._error is None:
            return self._null_object
        return self._make_masked_array_center(
            list(zip(*self._aperture_cutouts_center))[1])

    @lazyproperty
    def _variance_cutout(self):
        """
        A 2D aperture-weighted variance cutout using the
        aperture mask with the input ``sum_method`` method as a
        `~numpy.ma.MaskedArray`.

        The cutout does not have units due to current limitations of
        masked quantity arrays.

        The mask is `True` for pixels from the input ``mask``,
        non-finite ``data`` values (NaN and inf), sigma-clipped pixels
        within the aperture, and pixels where the aperture mask has zero
        weight.
        """
        if self._error is None:
            return self._null_object
        return self._make_masked_array(list(zip(*self._aperture_cutouts))[1])

    @lazyproperty
    @as_scalar
    def error_sumcutout(self):
        """
        A 2D aperture-weighted error cutout using the aperture mask with
        the input ``sum_method`` method as a `~numpy.ma.MaskedArray`.

        The cutout does not have units due to current limitations of
        masked quantity arrays.

        The mask is `True` for pixels from the input ``mask``,
        non-finite ``data`` values (NaN and inf), sigma-clipped pixels
        within the aperture, and pixels where the aperture mask has zero
        weight.
        """
        if self._error is None:
            return self._null_object
        return [np.sqrt(var) for var in self._variance_cutout]

    @lazyproperty
    def _weight_cutout_center(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the aperture mask
        weights array using the aperture bounding box.

        The aperture mask weights are for the "center" method.

        The mask is `True` for pixels outside of the aperture mask,
        pixels from the input ``mask``, non-finite ``data`` values (NaN
        and inf), and sigma-clipped pixels.
        """
        return self._make_masked_array_center(
            list(zip(*self._aperture_cutouts_center))[3])

    @lazyproperty
    def _weight_cutout(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the aperture mask
        weights array using the aperture bounding box.

        The aperture mask weights are for the ``sum_method`` method.

        The mask is `True` for pixels outside of the aperture mask,
        pixels from the input ``mask``, non-finite ``data`` values (NaN
        and inf), and sigma-clipped pixels.
        """
        return self._make_masked_array(list(zip(*self._aperture_cutouts))[3])

    @lazyproperty
    def _moment_data_cutout(self):
        """
        A list of 2D `~numpy.ndarray` cutouts from the data.

        Masked pixels are set to zero in these arrays (zeros do not
        contribute to the image moments). The aperture mask weights are
        for the "center" method.

        These arrays are used to derive moment-based properties.
        """
        data = deepcopy(self.data_cutout)  # self.data_cutout is a list
        if self.isscalar:
            data = (data,)

        cutouts = []
        for arr in data:
            if arr.size == 1 and np.isnan(arr[0]):  # no aperture overlap
                arr_ = np.empty((2, 2))
                arr_.fill(np.nan)
            else:
                arr_ = arr.data
                arr_[arr.mask] = 0.0
            cutouts.append(arr_)

        return cutouts

    @lazyproperty
    def _all_masked(self):
        """
        True if all pixels within the aperture are masked.
        """
        return np.array([np.all(mask) for mask in self._mask_cutout_center])

    @lazyproperty
    def _overlap(self):
        """
        True if there is no overlap of the aperture with the data.
        """
        return list(zip(*self._aperture_cutouts_center))[4]

    def _get_values(self, array):
        """
        Get a 1D array of unmasked aperture-weighted values from the
        input array.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        if self.isscalar:
            array = (array,)
        return [arr.compressed() if len(arr.compressed()) > 0
                else np.array([np.nan]) for arr in array]

    @lazyproperty
    def _data_values_center(self):
        """
        A 1D array of unmasked aperture-weighted data values using the
        "center" method.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        return self._get_values(self.data_cutout)

    @lazyproperty
    @as_scalar
    def moments(self):
        """
        Spatial moments up to 3rd order of the source.
        """
        return np.array([_moments(arr, order=3) for arr in
                         self._moment_data_cutout])

    @lazyproperty
    @as_scalar
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """
        cutout_centroid = self.cutout_centroid
        if self.isscalar:
            cutout_centroid = cutout_centroid[np.newaxis, :]
        return np.array([_moments_central(arr, center=(xcen_, ycen_), order=3)
                         for arr, xcen_, ycen_ in
                         zip(self._moment_data_cutout, cutout_centroid[:, 0],
                             cutout_centroid[:, 1])])

    @lazyproperty
    @as_scalar
    def cutout_centroid(self):
        """
        The ``(x, y)`` coordinate, relative to the cutout data, of
        the centroid within the aperture.

        The centroid is computed as the center of mass of the unmasked
        pixels within the aperture.
        """
        moments = self.moments
        if self.isscalar:
            moments = moments[np.newaxis, :]

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ycentroid = moments[:, 1, 0] / moments[:, 0, 0]
            xcentroid = moments[:, 0, 1] / moments[:, 0, 0]
        return np.transpose((xcentroid, ycentroid))

    @lazyproperty
    @as_scalar
    def centroid(self):
        """
        The ``(x, y)`` coordinate of the centroid.

        The centroid is computed as the center of mass of the unmasked
        pixels within the aperture.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid + origin

    @lazyproperty
    def _xcentroid(self):
        """
        The ``x`` coordinate of the centroid, always as an iterable.
        """
        xcentroid = np.transpose(self.centroid)[0]
        if self.isscalar:
            xcentroid = (xcentroid,)
        return xcentroid

    @lazyproperty
    @as_scalar
    def xcentroid(self):
        """
        The ``x`` coordinate of the centroid.

        The centroid is computed as the center of mass of the unmasked
        pixels within the aperture.
        """
        return self._xcentroid

    @lazyproperty
    def _ycentroid(self):
        """
        The ``y`` coordinate of the centroid, always as an iterable.
        """
        ycentroid = np.transpose(self.centroid)[1]
        if self.isscalar:
            ycentroid = (ycentroid,)
        return ycentroid

    @lazyproperty
    @as_scalar
    def ycentroid(self):
        """
        The ``y`` coordinate of the centroid.

        The centroid is computed as the center of mass of the unmasked
        pixels within the aperture.
        """
        return self._ycentroid

    @lazyproperty
    @as_scalar
    def sky_centroid(self):
        """
        The sky coordinate of the centroid of the unmasked pixels within
        the aperture, returned as a `~astropy.coordinates.SkyCoord`
        object.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self._wcs is None:
            return self._null_object
        return self._wcs.pixel_to_world(self.xcentroid, self.ycentroid)

    @lazyproperty
    @as_scalar
    def sky_centroid_icrs(self):
        """
        The sky coordinate in the International Celestial
        Reference System (ICRS) frame of the centroid of the
        unmasked pixels within the aperture, returned as a
        `~astropy.coordinates.SkyCoord` object.

        `None` if ``wcs`` is not input.
        """
        if self._wcs is None:
            return self._null_object
        return self.sky_centroid.icrs

    @lazyproperty
    def _bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the aperture, always as
        an iterable.
        """
        apertures = self._pixel_aperture
        if self.isscalar:
            apertures = (apertures,)
        return [aperture.bbox for aperture in apertures]

    @lazyproperty
    @as_scalar
    def bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the aperture.

        Note that the aperture bounding box is calculated using the
        exact size of the aperture, which may be slightly larger than
        the aperture mask calculated using the "center" mode.
        """
        return self._bbox

    @lazyproperty
    @as_scalar
    def _bbox_bounds(self):
        """
        The bounding box x/y minimum and maximum bounds.
        """
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        return np.array([(bbox_.ixmin, bbox_.ixmax - 1,
                          bbox_.iymin, bbox_.iymax - 1)
                         for bbox_ in bbox])

    @lazyproperty
    @as_scalar
    def bbox_xmin(self):
        """
        The minimum ``x`` pixel index of the bounding box.
        """
        return np.transpose(self._bbox_bounds)[0]

    @lazyproperty
    @as_scalar
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel index of the bounding box.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.transpose(self._bbox_bounds)[1]

    @lazyproperty
    @as_scalar
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel index of the bounding box.
        """
        return np.transpose(self._bbox_bounds)[2]

    @lazyproperty
    @as_scalar
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel index of the bounding box.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.transpose(self._bbox_bounds)[3]

    def _calculate_stats(self, stat_func, unit=None):
        """
        Apply the input ``stat_func`` to the 1D array of unmasked data
        values in the aperture.

        Units are applied if the input ``data`` has units.

        Parameters
        ----------
        stat_func : callable
            The callable to apply to the 1D `~numpy.ndarray` of unmasked
            data values.

        unit : `None` or `astropy.unit.Unit`, optional
            The unit to apply to the output data. This is used only
            if the input ``data`` has units. If `None` then the input
            ``data`` unit will be used.
        """
        result = np.array([stat_func(arr) for arr in self._data_values_center])
        if unit is None:
            unit = self._data_unit
        if unit is not None:
            result <<= unit
        return result

    @lazyproperty
    @as_scalar
    def center_aper_area(self):
        """
        The total area of the unmasked pixels within the aperture using
        the "center" aperture mask method.
        """
        areas = np.array([np.sum(weight.filled(0.0))
                          for weight in self._weight_cutout_center])
        areas[self._all_masked] = np.nan
        return areas << (u.pix**2)

    @lazyproperty
    @as_scalar
    def sum_aper_area(self):
        """
        The total area of the unmasked pixels within the aperture using
        the input ``sum_method`` aperture mask method.
        """
        areas = np.array([np.sum(weight.filled(0.0))
                          for weight in self._weight_cutout])
        areas[self._all_masked] = np.nan
        return areas << (u.pix**2)

    @lazyproperty
    @as_scalar
    def sum(self):
        r"""
        The sum of the unmasked ``data`` values within the aperture.

        .. math:: F = \sum_{i \in A} I_i

        where :math:`F` is ``sum``, :math:`I_i` is the
        background-subtracted ``data``, and :math:`A` are the unmasked
        pixels in the aperture.

        Non-finite pixel values (NaN and inf) are excluded
        (automatically masked).
        """
        if self.sum_method == 'center':
            return self._calculate_stats(np.sum)

        data_values = self._get_values(self.data_sumcutout)
        result = np.array([np.sum(arr) for arr in data_values])
        if self._data_unit is not None:
            result <<= self._data_unit
        return result

    @lazyproperty
    @as_scalar
    def sum_err(self):
        r"""
        The uncertainty of `sum` , propagated from the input ``error``
        array.

        ``sum_err`` is the quadrature sum of the total errors over the
        unmasked pixels within the aperture:

        .. math:: \Delta F = \sqrt{\sum_{i \in A}
                  \sigma_{\mathrm{tot}, i}^2}

        where :math:`\Delta F` is the `sum`, :math:`\sigma_{\mathrm{tot,
        i}}` are the pixel-wise total errors (``error``), and :math:`A`
        are the unmasked pixels in the aperture.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and inf) that are automatically
        masked, are also masked in the error array.
        """
        if self._error is None:
            err = self._null_value

        else:
            if self.sum_method == 'center':
                variance = self._variance_cutout_center
            else:
                variance = self._variance_cutout

            var_values = [arr.compressed() if len(arr.compressed()) > 0
                          else np.array([np.nan]) for arr in variance]
            err = np.sqrt([np.sum(arr) for arr in var_values])

        if self._data_unit is not None:
            err <<= self._data_unit
        return err

    @lazyproperty
    @as_scalar
    def min(self):
        """
        The minimum of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.min)

    @lazyproperty
    @as_scalar
    def max(self):
        """
        The maximum of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.max)

    @lazyproperty
    @as_scalar
    def mean(self):
        """
        The mean of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.mean)

    @lazyproperty
    @as_scalar
    def median(self):
        """
        The median of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.median)

    @lazyproperty
    @as_scalar
    def mode(self):
        """
        The mode of the unmasked pixel values within the aperture.

        The mode is estimated as ``(3 * median) - (2 * mean)``.
        """
        return 3.0 * self.median - 2.0 * self.mean

    @lazyproperty
    @as_scalar
    def std(self):
        """
        The standard deviation of the unmasked pixel values within the
        aperture.
        """
        return self._calculate_stats(np.std)

    @lazyproperty
    @as_scalar
    def mad_std(self):
        r"""
        The standard deviation calculated using
        the `median absolute deviation (MAD)
        <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

        The standard deviation estimator is given by:

        .. math::

            \sigma \approx \frac{\textrm{MAD}}{\Phi^{-1}(3/4)}
                \approx 1.4826 \ \textrm{MAD}

        where :math:`\Phi^{-1}(P)` is the normal inverse cumulative
        distribution function evaluated at probability :math:`P = 3/4`.
        """
        return self._calculate_stats(mad_std)

    @lazyproperty
    @as_scalar
    def var(self):
        """
        The variance of the unmasked pixel values within the aperture.
        """
        unit = self._data_unit
        if unit is not None:
            unit **= 2
        return self._calculate_stats(np.var, unit=unit)

    @lazyproperty
    @as_scalar
    def biweight_location(self):
        """
        The biweight location of the unmasked pixel values within the
        aperture.

        See `astropy.stats.biweight_location`.
        """
        return self._calculate_stats(biweight_location)

    @lazyproperty
    @as_scalar
    def biweight_midvariance(self):
        """
        The biweight midvariance of the unmasked pixel values within the
        aperture.

        See `astropy.stats.biweight_midvariance`
        """
        unit = self._data_unit
        if unit is not None:
            unit **= 2
        return self._calculate_stats(biweight_midvariance, unit=unit)

    @lazyproperty
    @as_scalar
    def inertia_tensor(self):
        """
        The inertia tensor of the source for the rotation around its
        center of mass.
        """
        moments = self.moments_central
        if self.isscalar:
            moments = moments[np.newaxis, :]
        mu_02 = moments[:, 0, 2]
        mu_11 = -moments[:, 1, 1]
        mu_20 = moments[:, 2, 0]
        tensor = np.array([mu_02, mu_11, mu_11, mu_20]).swapaxes(0, 1)
        return tensor.reshape((tensor.shape[0], 2, 2)) * u.pix**2

    @lazyproperty
    def _covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source, always as an iterable.
        """
        moments = self.moments_central
        if self.isscalar:
            moments = moments[np.newaxis, :]
        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu_norm = moments / moments[:, 0, 0][:, np.newaxis, np.newaxis]

        covar = np.array([mu_norm[:, 0, 2], mu_norm[:, 1, 1],
                          mu_norm[:, 1, 1], mu_norm[:, 2, 0]]).swapaxes(0, 1)
        covar = covar.reshape((covar.shape[0], 2, 2))

        # Modify the covariance matrix in the case of "infinitely" thin
        # detections. This follows SourceExtractor's prescription of
        # incrementally increasing the diagonal elements by 1/12.
        delta = 1.0 / 12
        delta2 = delta**2
        # ignore RuntimeWarning from NaN values in covar
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            covar_det = np.linalg.det(covar)
            idx = np.where(covar_det < delta2)[0]
            while idx.size > 0:  # pragma: no cover
                covar[idx, 0, 0] += delta
                covar[idx, 1, 1] += delta
                covar_det = np.linalg.det(covar)
                idx = np.where(covar_det < delta2)[0]
        return covar

    @lazyproperty
    @as_scalar
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source.
        """
        return self._covariance * (u.pix**2)

    @lazyproperty
    @as_scalar
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """
        eigvals = np.empty((self.n_apertures, 2))
        eigvals.fill(np.nan)
        # np.linalg.eivals requires finite input values
        idx = np.unique(np.where(np.isfinite(self._covariance))[0])
        eigvals[idx] = np.linalg.eigvals(self._covariance[idx])

        # check for negative variance
        # (just in case covariance matrix is not positive (semi)definite)
        idx2 = np.unique(np.where(eigvals < 0)[0])  # pragma: no cover
        eigvals[idx2] = (np.nan, np.nan)  # pragma: no cover

        # sort each eigenvalue pair in descending order
        eigvals.sort(axis=1)
        eigvals = np.fliplr(eigvals)

        return eigvals * u.pix**2

    @lazyproperty
    @as_scalar
    def semimajor_sigma(self):
        """
        The 1-sigma standard deviation along the semimajor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """
        eigvals = self.covariance_eigvals
        if self.isscalar:
            eigvals = eigvals[np.newaxis, :]
        # this matches SourceExtractor's A parameter
        return np.sqrt(eigvals[:, 0])

    @lazyproperty
    @as_scalar
    def semiminor_sigma(self):
        """
        The 1-sigma standard deviation along the semiminor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """
        eigvals = self.covariance_eigvals
        if self.isscalar:
            eigvals = eigvals[np.newaxis, :]
        # this matches SourceExtractor's B parameter
        return np.sqrt(eigvals[:, 1])

    @lazyproperty
    @as_scalar
    def fwhm(self):
        r"""
        The circularized full width at half maximum (FWHM) of the 2D
        Gaussian function that has the same second-order central moments
        as the source.

        .. math::

           \mathrm{FWHM} & = 2 \sqrt{2 \ln(2)} \sqrt{0.5 (a^2 + b^2)}
           \\
                          & = 2 \sqrt{\ln(2) \ (a^2 + b^2)}

        where :math:`a` and :math:`b` are the 1-sigma lengths of the
        semimajor (`semimajor_sigma`) and semiminor (`semiminor_sigma`)
        axes, respectively.
        """
        return 2.0 * np.sqrt(np.log(2.0) * (self.semimajor_sigma**2
                                            + self.semiminor_sigma**2))

    @lazyproperty
    @as_scalar
    def orientation(self):
        """
        The angle between the ``x`` axis and the major axis of the 2D
        Gaussian function that has the same second-order moments as the
        source.

        The angle increases in the counter-clockwise direction.
        """
        covar = self._covariance
        orient_radians = 0.5 * np.arctan2(2.0 * covar[:, 0, 1],
                                          (covar[:, 0, 0] - covar[:, 1, 1]))
        return orient_radians * 180.0 / np.pi * u.deg

    @lazyproperty
    @as_scalar
    def eccentricity(self):
        r"""
        The eccentricity of the 2D Gaussian function that has the same
        second-order moments as the source.

        The eccentricity is the fraction of the distance along the
        semimajor axis at which the focus lies.

        .. math:: e = \sqrt{1 - \frac{b^2}{a^2}}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        semimajor_var, semiminor_var = np.transpose(self.covariance_eigvals)
        return np.sqrt(1.0 - (semiminor_var / semimajor_var))

    @lazyproperty
    @as_scalar
    def elongation(self):
        r"""
        The ratio of the lengths of the semimajor and semiminor axes.

        .. math:: \mathrm{elongation} = \frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self.semimajor_sigma / self.semiminor_sigma

    @lazyproperty
    @as_scalar
    def ellipticity(self):
        r"""
        1.0 minus the ratio of the lengths of the semimajor and
        semiminor axes (or 1.0 minus the `elongation`).

        .. math:: \mathrm{ellipticity} = 1 - \frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return 1.0 - (self.semiminor_sigma / self.semimajor_sigma)

    @lazyproperty
    @as_scalar
    def covar_sigx2(self):
        r"""
        The ``(0, 0)`` element of the `covariance` matrix, representing
        :math:`\sigma_x^2`, in units of pixel**2.
        """
        return self._covariance[:, 0, 0] * u.pix**2

    @lazyproperty
    @as_scalar
    def covar_sigy2(self):
        r"""
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\sigma_y^2`, in units of pixel**2.
        """
        return self._covariance[:, 1, 1] * u.pix**2

    @lazyproperty
    @as_scalar
    def covar_sigxy(self):
        r"""
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\sigma_x \sigma_y`, in units of
        pixel**2.
        """
        return self._covariance[:, 0, 1] * u.pix**2

    @lazyproperty
    @as_scalar
    def cxx(self):
        r"""
        `SourceExtractor`_'s CXX ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
                cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \approx 3`.
        """
        return ((np.cos(self.orientation) / self.semimajor_sigma)**2
                + (np.sin(self.orientation) / self.semiminor_sigma)**2)

    @lazyproperty
    @as_scalar
    def cyy(self):
        r"""
        `SourceExtractor`_'s CYY ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
                cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \approx 3`.
        """
        return ((np.sin(self.orientation) / self.semimajor_sigma)**2
                + (np.cos(self.orientation) / self.semiminor_sigma)**2)

    @lazyproperty
    @as_scalar
    def cxy(self):
        r"""
        `SourceExtractor`_'s CXY ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
                cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \approx 3`.
        """
        return (2.0 * np.cos(self.orientation) * np.sin(self.orientation)
                * ((1.0 / self.semimajor_sigma**2)
                   - (1.0 / self.semiminor_sigma**2)))

    @lazyproperty
    @as_scalar
    def gini(self):
        r"""
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        unmasked pixel values within the aperture.

        The Gini coefficient is calculated using the prescription from
        `Lotz et al. 2004
        <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
        as:

        .. math::
            G = \frac{1}{\left | \bar{x} \right | n (n - 1)}
            \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

        where :math:`\bar{x}` is the mean over pixel values :math:`x_i`
        within the aperture.

        The Gini coefficient is a way of measuring the inequality in a
        given set of values. In the context of galaxy morphology, it
        measures how the light of a galaxy image is distributed among
        its pixels. A Gini coefficient value of 0 corresponds to a
        galaxy image with the light evenly distributed over all pixels
        while a Gini coefficient value of 1 represents a galaxy image
        with all its light concentrated in just one pixel.
        """
        gini = []
        for arr in self._data_values_center:
            if np.all(np.isnan(arr)):
                gini.append(np.nan)
                continue
            npix = np.size(arr)
            normalization = np.abs(np.mean(arr)) * npix * (npix - 1)
            kernel = ((2.0 * np.arange(1, npix + 1) - npix - 1)
                      * np.abs(np.sort(arr)))
            gini.append(np.sum(kernel) / normalization)
        return np.array(gini)
