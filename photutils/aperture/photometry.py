# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for performing aperture photometry.
"""

import warnings

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture._segmentation import process_segmentation_inputs
from photutils.aperture.converters import region_to_aperture
from photutils.aperture.core import (Aperture, SkyAperture, _aperture_metadata,
                                     _update_method_subpixels_docstring)
from photutils.aperture.flags import decode_aperture_flags
from photutils.utils._deprecation import (create_empty_deprecated_qtable,
                                          deprecated_positional_kwargs)
from photutils.utils._misc import _get_meta
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._repr import make_repr

__all__ = ['AperturePhotometry', 'aperture_photometry']


# Remove in 4.0
_DEPRECATED_COLUMNS: dict = {
    'xcenter': 'x_center',
    'ycenter': 'y_center',
}


@_update_method_subpixels_docstring
class AperturePhotometry:
    # numpydoc ignore: PR01,PR02,PR04,PR07
    """
    Class to perform aperture photometry on 2D data.

    This class sums the (weighted) input ``data`` values within the
    given aperture(s) and provides the aperture fluxes, uncertainties,
    unmasked overlap areas, and bitwise quality flags as lazily-computed
    attributes. Use `to_table` to obtain the results as an
    `~astropy.table.QTable`.

    Note that this class returns the sum of the (weighted) input
    ``data`` values within the aperture. It does not convert data in
    surface brightness units to flux or counts. Conversion from
    surface-brightness units should be performed before using this
    class.

    Parameters
    ----------
    data : array_like, `~astropy.units.Quantity`, `~astropy.nddata.NDData`
        The 2D array on which to perform photometry. ``data`` should be
        background-subtracted. If ``data`` is a `~astropy.units.Quantity`
        array, then ``error`` (if input) must also be a
        `~astropy.units.Quantity` array with the same units. Non-finite
        ``data`` values (NaN and inf) are automatically masked. See the
        Notes section below for more information about
        `~astropy.nddata.NDData` input.

    apertures : `~photutils.aperture.Aperture`, supported `regions.Region`, \
        list of `~photutils.aperture.Aperture` or `regions.Region`
        The aperture(s) to use for the photometry. If ``apertures`` is
        a list of `~photutils.aperture.Aperture` or `regions.Region`,
        then they all must have the same position(s). If ``apertures``
        contains a `~photutils.aperture.SkyAperture` or `~regions.SkyRegion`
        object, then a WCS must be input using the ``wcs`` keyword.
        Region objects are converted to aperture objects.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input
        ``data``. ``error`` is assumed to include *all* sources
        of error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`). ``error`` must have the
        same shape as the input ``data``. If a `~astropy.units.Quantity`
        array, then ``data`` must also be a `~astropy.units.Quantity`
        array with the same units.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations. Non-finite
        values (NaN and inf) in the input ``data`` are automatically
        masked.

    wcs : WCS object, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). If provided, the `sky_center`
        attribute and the ``'sky_center'`` column of `to_table` will
        contain the sky coordinates of the input aperture center(s).
        This keyword is required if the input ``apertures`` contains a
        `SkyAperture` or `~regions.SkyRegion`.

    <method_subpixels_descriptions>

    <segmentation_descriptions>

    See Also
    --------
    photutils.aperture.ApertureStats : Per-source statistics (e.g.,
        centroid, min, max, median, standard deviation, and
        morphological properties) of the pixels within an aperture.

    Notes
    -----
    `~regions.Region` objects are converted to `Aperture` objects using
    the :func:`region_to_aperture` function.

    If the input ``data`` is a `~astropy.nddata.NDData` instance,
    then the ``error``, ``mask``, and ``wcs`` keyword inputs are
    ignored. Instead, these values should be defined as attributes in
    the `~astropy.nddata.NDData` object. In the case of ``error``,
    it must be defined in the ``uncertainty`` attribute with a
    `~astropy.nddata.StdDevUncertainty` instance.

    The measured `flux`, `flux_err`, `area`, and `flags` attributes are
    1D arrays with one element per aperture position when a single
    aperture is input. If a list of apertures is input, they are 2D
    arrays with shape ``(n_positions, n_apertures)``.

    Non-finite ``data`` values (NaN and inf) are automatically masked.
    Such pixels are excluded from the `flux`, `flux_err`, and `area`
    calculations and are indicated by the ``non_finite_data`` quality
    flag (see `~photutils.aperture.decode_aperture_flags`).

    This class is immutable after initialization (its cached attributes
    use compute-once `~astropy.utils.decorators.lazyproperty` caching),
    so a single instance can be safely shared across threads.

    Examples
    --------
    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils.aperture import AperturePhotometry, CircularAperture
    >>> data = make_4gaussians_image()
    >>> error = 0.1 * np.ones(data.shape)  # simple background-only error
    >>> aper = CircularAperture([(25, 40), (90, 60), (150, 25)], r=8)
    >>> phot = AperturePhotometry(data, aper, error=error)
    >>> print(phot.flux)
    [ 5853.59627292 28440.27461471  9286.70920641]
    >>> print(phot.flux_err)
    [1.41796308 1.41796308 1.41796308]
    >>> phot.to_table(columns=['id', 'flux', 'flux_err'])
    <QTable length=3>
    id         flux             flux_err
    int64      float64            float64
    ----- ------------------ -----------------
        1  5853.596272924398 1.417963080724414
        2 28440.274614708058 1.417963080724414
        3  9286.709206410273 1.417963080724414
    """

    _repr_params = ('method', 'subpixels', 'mask_method')

    def __init__(self, data, apertures, *, error=None, mask=None, wcs=None,
                 method='exact', subpixels=5, segmentation_image=None,
                 labels=None, mask_method='none'):

        if isinstance(data, NDData):
            data, error, mask, wcs = self._unpack_nddata(data, error, mask,
                                                         wcs)

        (data, error), unit = process_quantities(
            (data, error), ('data', 'error'))
        self._data = self._validate_array(data, 'data', shape=False)
        self._data_unit = unit
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._wcs = wcs
        self.method = method
        self.subpixels = subpixels
        self.mask_method = mask_method

        single_aperture = False
        if not isinstance(apertures, (list, tuple, np.ndarray)):
            single_aperture = True
            apertures = (apertures,)
        self._single_aperture = single_aperture

        # Create table metadata using the input apertures, not the
        # converted ones
        aper_meta = {}
        for i, aperture in enumerate(apertures):
            idx = '' if single_aperture else i
            aper_meta.update(_aperture_metadata(aperture, index=idx))

        # Convert regions to apertures if necessary
        apertures = [region_to_aperture(aper)
                     if not isinstance(aper, Aperture) else aper
                     for aper in apertures]

        # Convert sky to pixel apertures
        self._skyaper = False
        self._sky_positions = None
        if isinstance(apertures[0], SkyAperture):
            if wcs is None:
                msg = ('A WCS transform must be defined by the input data '
                       'or the wcs keyword when using a SkyAperture object.')
                raise ValueError(msg)
            self._skyaper = True
            self._sky_positions = apertures[0].positions
            apertures = [aper.to_pixel(wcs) for aper in apertures]

        # Compare positions in pixels to avoid comparing SkyCoord objects
        positions = apertures[0].positions
        for aper in apertures[1:]:
            if not np.array_equal(aper.positions, positions):
                msg = 'Input apertures must all have identical positions'
                raise ValueError(msg)
        self._pixel_apertures = apertures

        # Validate the segmentation-masking inputs and resolve the
        # per-aperture source labels once
        self.segmentation_image = segmentation_image
        self.labels = labels
        seg_positions = np.atleast_2d(apertures[0].positions)
        (self._segmentation,
         self._seg_labels) = process_segmentation_inputs(
            segmentation_image, labels, mask_method, seg_positions,
            self._data.shape)

        # Define output table metadata
        self.meta = _get_meta()
        calling_args = f"method='{method}', subpixels={subpixels}"
        self.meta['aperture_photometry_args'] = calling_args
        self.meta.update(aper_meta)

        default_columns = ['id', 'x_center', 'y_center']
        if self._wcs is not None or self._skyaper:
            default_columns.append('sky_center')
        default_columns += ['flux', 'flux_err', 'area', 'flags']
        self.default_columns = default_columns

    @staticmethod
    def _unpack_nddata(data, error, mask, wcs):
        nddata_attr = {'error': error, 'mask': mask, 'wcs': wcs}
        for key, value in nddata_attr.items():
            if value is not None:
                msg = (f'The {key!r} keyword is ignored. Its value '
                       'is obtained from the input NDData object.')
                warnings.warn(msg, AstropyUserWarning)

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

    def _validate_array(self, array, name, *, ndim=2, shape=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != ndim:
                msg = f'{name} must be a {ndim}D array'
                raise ValueError(msg)
            if shape and array.shape != self._data.shape:
                msg = f'data and {name} must have the same shape'
                raise ValueError(msg)
        return array

    def __repr__(self):
        return make_repr(self, self._repr_params)

    def __str__(self):
        return make_repr(self, self._repr_params, long=True)

    @lazyproperty
    def _photometry_results(self):
        """
        The per-aperture `~photutils.aperture.ApertureResults`, one for
        each input aperture.
        """
        return [aper.photometry(
            self._data, error=self._error, mask=self._mask,
            method=self.method, subpixels=self.subpixels,
            segmentation_image=self._segmentation,
            labels=self._seg_labels, mask_method=self.mask_method,
            _mask_nonfinite=True)
            for aper in self._pixel_apertures]

    @lazyproperty
    def _positions(self):
        """
        The aperture positions in pixels, always as a 2D array.
        """
        return np.atleast_2d(self._pixel_apertures[0].positions)

    @lazyproperty
    def n_positions(self):
        """
        The number of aperture positions.
        """
        return self._positions.shape[0]

    @lazyproperty
    def id(self):
        """
        The aperture identification number(s).
        """
        return np.arange(self.n_positions) + 1

    @lazyproperty
    def x_center(self):
        """
        The ``x`` pixel coordinate(s) of the aperture center(s).
        """
        return self._positions[:, 0]

    @lazyproperty
    def y_center(self):
        """
        The ``y`` pixel coordinate(s) of the aperture center(s).
        """
        return self._positions[:, 1]

    @lazyproperty
    def sky_center(self):
        """
        The sky coordinates of the aperture center(s), or `None` if no
        ``wcs`` was input.
        """
        if self._skyaper:
            pos = self._sky_positions
            if pos.isscalar:
                # Return a length-1 SkyCoord array
                return pos.reshape((-1,))
            return pos
        if self._wcs is not None:
            return self._wcs.pixel_to_world(*np.transpose(self._positions))
        return None

    def _stack(self, attr):
        """
        Stack a per-aperture result attribute into a 1D array (single
        aperture) or a 2D ``(n_positions, n_apertures)`` array (list of
        apertures).
        """
        values = [getattr(result, attr)
                  for result in self._photometry_results]
        if self._single_aperture:
            return values[0]
        return np.stack(values, axis=1)

    @lazyproperty
    def flux(self):
        """
        The sum of the (weighted) values within the aperture(s).

        The values are always float64, regardless of the input ``data``
        dtype (a `~astropy.units.Quantity` if ``data`` has units).
        """
        values = self._stack('aperture_sum')
        if self._data_unit is not None:
            values <<= self._data_unit
        return values

    @lazyproperty
    def flux_err(self):
        """
        The uncertainty in the `flux` values.

        The values are always float64, regardless of the input ``error``
        dtype (a `~astropy.units.Quantity` if ``data`` has units). If the
        input ``error`` is `None`, this is filled with NaN values.
        """
        values = self._stack('aperture_sum_err')
        if self._data_unit is not None:
            values <<= self._data_unit
        return values

    @lazyproperty
    def area(self):
        """
        The total unmasked overlap area of the aperture(s) (in
        ``pix**2``).

        This takes into account the aperture mask method, masked
        data pixels (``mask`` keyword), segmentation masking, and
        partial/no overlap of the aperture with the data. The value is
        NaN where an aperture does not overlap the data.
        """
        values = [result.area for result in self._photometry_results]
        if self._single_aperture:
            return values[0]
        stacked = np.stack([value.value for value in values], axis=1)
        return u.Quantity(stacked, u.pix**2)

    @lazyproperty
    def flags(self):
        """
        The bitwise quality flags for the aperture(s).

        See `~photutils.aperture.decode_aperture_flags` for decoding
        flag values.
        """
        return self._stack('flags')

    @deprecated_positional_kwargs(since='3.1', until='4.0')
    def to_table(self, *, columns=None):
        """
        Create a `~astropy.table.QTable` of the aperture photometry
        results.

        Parameters
        ----------
        columns : str, list of str, `None`, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`. The allowed column names are
            ``'id'``, ``'x_center'``, ``'y_center'``, ``'sky_center'``,
            ``'flux'``, ``'flux_err'``, ``'area'``, and ``'flags'``. If
            ``columns`` is `None`, then a default list of columns will
            be used (the ``default_columns`` attribute). If a list of
            apertures was input, then the ``'flux'``, ``'flux_err'``,
            ``'area'``, and ``'flags'`` columns will have a ``'_i'``
            suffix (e.g., ``'flux_0'``), where ``i`` is the index of the
            aperture in the input list.

        Returns
        -------
        table : `~astropy.table.QTable`
            A table of the aperture photometry results with one row per
            aperture position.
        """
        if columns is None:
            table_columns = self.default_columns
        elif isinstance(columns, str):
            table_columns = [columns]
        else:
            table_columns = columns

        tbl = QTable()
        tbl.meta.update(self.meta)

        per_aperture = ('flux', 'flux_err', 'area', 'flags')
        for column in table_columns:
            if column in per_aperture and not self._single_aperture:
                values = getattr(self, column)
                for i in range(len(self._pixel_apertures)):
                    name = f'{column}_{i}'
                    tbl[name] = values[:, i]
            else:
                tbl[column] = getattr(self, column)
        return tbl

    def decode_flags(self, *, return_bit_values=False):
        """
        Decode the aperture quality flags into individual components.

        This is a convenience method that calls
        `~photutils.aperture.decode_aperture_flags` with the `flags`
        attribute.

        Parameters
        ----------
        return_bit_values : bool, optional
            If `True`, return the decoded bit flags (integers) instead
            of the flag names (strings).

        Returns
        -------
        decoded : list of list of str or list of list of int
            A list of the active flag names (or bit values) for each
            aperture.

        See Also
        --------
        photutils.aperture.decode_aperture_flags
        """
        return decode_aperture_flags(np.atleast_1d(self.flags),
                                     return_bit_values=return_bit_values)


@_update_method_subpixels_docstring
@deprecated_positional_kwargs(since='3.0', until='4.0')
def aperture_photometry(data, apertures, error=None, mask=None,
                        method='exact', subpixels=5, wcs=None, *,
                        segmentation_image=None, labels=None,
                        mask_method='none'):
    # numpydoc ignore: PR01,PR02,PR04,PR07
    """
    Perform aperture photometry on the input data by summing the flux
    within the given aperture(s).

    Note that this function returns the sum of the (weighted) input
    ``data`` values within the aperture. It does not convert data
    in surface brightness units to flux or counts. Conversion from
    surface-brightness units should be performed before using this
    function.

    Parameters
    ----------
    data : array_like, `~astropy.units.Quantity`, `~astropy.nddata.NDData`
        The 2D array on which to perform photometry. ``data``
        should be background-subtracted. If ``data`` is a
        `~astropy.units.Quantity` array, then ``error`` (if input)
        must also be a `~astropy.units.Quantity` array with the same
        units. See the Notes section below for more information about
        `~astropy.nddata.NDData` input.

    apertures : `~photutils.aperture.Aperture`, supported `regions.Region`, \
        list of `~photutils.aperture.Aperture` or `regions.Region`
        The aperture(s) to use for the photometry. If ``apertures`` is
        a list of `~photutils.aperture.Aperture` or `regions.Region`,
        then they all must have the same position(s). If
        ``apertures`` contains a `~photutils.aperture.SkyAperture` or
        `~regions.SkyRegion` object, then a WCS must be input using
        the ``wcs`` keyword. Region objects are converted to aperture
        objects.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input
        ``data``. ``error`` is assumed to include *all* sources
        of error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`). ``error`` must have the
        same shape as the input ``data``. If a `~astropy.units.Quantity`
        array, then ``data`` must also be a `~astropy.units.Quantity`
        array with the same units.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    <method_subpixels_descriptions>

    wcs : WCS object, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). If provided, the output
        table will include a ``'sky_center'`` column with the sky
        coordinates of the input aperture center(s). This keyword is
        required if the input ``apertures`` contains a `SkyAperture` or
        `~regions.SkyRegion`.

    <segmentation_descriptions>

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of the photometry with the following columns:

        * ``'id'``:
          The source ID.

        * ``'x_center'``, ``'y_center'``:
          The ``x`` and ``y`` pixel coordinates of the input aperture
          center(s).

        * ``'sky_center'``:
          The sky coordinates of the input aperture center(s). Returned
          if a ``wcs`` is input.

        * ``'aperture_sum'``:
          The sum of the values within the aperture(s). The values
          are always float64, regardless of the input ``data`` dtype
          (a `~astropy.units.Quantity` with float64 values if ``data``
          has units).

        * ``'aperture_sum_err'``:
          The corresponding uncertainty in the ``'aperture_sum'``
          values (always float64). If the input ``error`` is `None`,
          this column is filled with NaN values.

        * ``'area'``:
          The total unmasked overlap area of the aperture(s)
          (in ``pix**2``), taking into account the aperture
          mask method, masked data pixels (``mask`` keyword),
          segmentation masking, and partial/no overlap of
          the aperture with the data. This is equivalent to
          :meth:`~photutils.aperture.PixelAperture.area_overlap`
          computed with the same inputs. The value is NaN where an
          aperture does not overlap the data.

        * ``'flags'``:
          The bitwise quality flags for the aperture(s). See
          :func:`~photutils.aperture.decode_aperture_flags` for decoding
          flag values. The flags are:

          <flag_descriptions>

        If multiple apertures are input, the ``'aperture_sum'``,
        ``'aperture_sum_err'``, ``'area'``, and ``'flags'`` columns will
        have a ``'_i'`` suffix (e.g., ``'aperture_sum_0'``), where ``i``
        is the index of the aperture in the input list.

        The table metadata includes the Astropy and Photutils version
        numbers and the `aperture_photometry` calling arguments.

    Notes
    -----
    `~regions.Region` objects are converted to `Aperture` objects using
    the :func:`region_to_aperture` function.

    If the input ``data`` is a `~astropy.nddata.NDData` instance,
    then the ``error``, ``mask``, and ``wcs`` keyword inputs are
    ignored. Instead, these values should be defined as attributes in
    the `~astropy.nddata.NDData` object. In the case of ``error``,
    it must be defined in the ``uncertainty`` attribute with a
    `~astropy.nddata.StdDevUncertainty` instance.
    """
    if isinstance(data, NDData):
        nddata_attr = {'error': error, 'mask': mask, 'wcs': wcs}
        for key, value in nddata_attr.items():
            if value is not None:
                msg = (f'The {key!r} keyword is ignored. Its value '
                       'is obtained from the input NDData object.')
                warnings.warn(msg, AstropyUserWarning)

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
                                   wcs=wcs,
                                   segmentation_image=segmentation_image,
                                   labels=labels,
                                   mask_method=mask_method)

    single_aperture = False
    if not isinstance(apertures, (list, tuple, np.ndarray)):
        single_aperture = True
        apertures = (apertures,)

    # Create table metadata using the input apertures, not the converted
    # ones
    aper_meta = {}
    for i, aperture in enumerate(apertures):
        i = '' if single_aperture else i
        aper_meta.update(_aperture_metadata(aperture, index=i))

    # Convert regions to apertures if necessary
    apertures = [region_to_aperture(aper)
                 if not isinstance(aper, Aperture) else aper
                 for aper in apertures]

    # Convert sky to pixel apertures
    skyaper = False
    if isinstance(apertures[0], SkyAperture):
        if wcs is None:
            msg = ('A WCS transform must be defined by the input data or '
                   'the wcs keyword when using a SkyAperture object.')
            raise ValueError(msg)

        # Include SkyCoord position in the output table
        skyaper = True
        skycoord_pos = apertures[0].positions

        apertures = [aper.to_pixel(wcs) for aper in apertures]

    # Compare positions in pixels to avoid comparing SkyCoord objects
    positions = apertures[0].positions
    for aper in apertures[1:]:
        if not np.array_equal(aper.positions, positions):
            msg = 'Input apertures must all have identical positions'
            raise ValueError(msg)

    # Validate the segmentation-masking inputs and resolve the
    # per-aperture source labels once (the resolved labels are passed
    # explicitly to PixelAperture.photometry to avoid repeated
    # auto-lookups and warnings).
    segmentation, labels = process_segmentation_inputs(
        segmentation_image, labels, mask_method,
        np.atleast_2d(positions), np.shape(data))

    # Define output table meta data
    meta = _get_meta()
    calling_args = f"method='{method}', subpixels={subpixels}"
    meta['aperture_photometry_args'] = calling_args
    meta.update(aper_meta)

    # Replace with QTable in 4.0
    tbl = create_empty_deprecated_qtable(
        _DEPRECATED_COLUMNS, since='3.0', until='4.0')

    tbl.meta.update(meta)  # keep tbl.meta type

    positions = np.atleast_2d(apertures[0].positions)
    tbl['id'] = np.arange(positions.shape[0], dtype=int) + 1

    xypos_pixel = np.transpose(positions)
    tbl['x_center'] = xypos_pixel[0]
    tbl['y_center'] = xypos_pixel[1]

    if skyaper:
        if skycoord_pos.isscalar:
            # Create length-1 SkyCoord array
            tbl['sky_center'] = skycoord_pos.reshape((-1,))
        else:
            tbl['sky_center'] = skycoord_pos

    if wcs is not None and not skyaper:
        tbl['sky_center'] = wcs.pixel_to_world(*np.transpose(positions))

    sum_key_main = 'aperture_sum'
    sum_err_key_main = 'aperture_sum_err'
    area_key_main = 'area'
    flags_key_main = 'flags'
    for i, aper in enumerate(apertures):
        result = aper.photometry(
            data, error=error, mask=mask, method=method, subpixels=subpixels,
            segmentation_image=segmentation, labels=labels,
            mask_method=mask_method)

        sum_key = sum_key_main
        sum_err_key = sum_err_key_main
        area_key = area_key_main
        flags_key = flags_key_main
        if not single_aperture:
            sum_key += f'_{i}'
            sum_err_key += f'_{i}'
            area_key += f'_{i}'
            flags_key += f'_{i}'

        tbl[sum_key] = result.aperture_sum
        tbl[sum_err_key] = result.aperture_sum_err
        tbl[area_key] = result.area
        tbl[flags_key] = result.flags
        tbl[flags_key].info.description = (
            'Aperture quality flags; see decode_aperture_flags')

    return tbl
