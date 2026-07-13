# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for performing aperture photometry.
"""

import warnings

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, StdDevUncertainty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture._segmentation import process_segmentation_inputs
from photutils.aperture.converters import region_to_aperture
from photutils.aperture.core import (Aperture, SkyAperture, _aperture_metadata,
                                     _update_method_subpixels_docstring)
from photutils.utils._deprecation import (create_empty_deprecated_qtable,
                                          deprecated_positional_kwargs)
from photutils.utils._misc import _get_meta

__all__ = ['aperture_photometry']


# Remove in 4.0
_DEPRECATED_COLUMNS: dict = {
    'xcenter': 'x_center',
    'ycenter': 'y_center',
}


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
    # explicitly to do_photometry to avoid repeated auto-lookups and
    # warnings)
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
    for i, aper in enumerate(apertures):
        aper_sum, aper_sum_err = aper.do_photometry(
            data, error=error, mask=mask, method=method, subpixels=subpixels,
            segmentation_image=segmentation, labels=labels,
            mask_method=mask_method)

        sum_key = sum_key_main
        sum_err_key = sum_err_key_main
        if not single_aperture:
            sum_key += f'_{i}'
            sum_err_key += f'_{i}'

        tbl[sum_key] = aper_sum
        tbl[sum_err_key] = aper_sum_err

    return tbl
