# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines tools to perform aperture photometry.
"""

import warnings
from collections import OrderedDict

import numpy as np
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable
import astropy.units as u
from astropy.utils.decorators import deprecated_renamed_argument
from astropy.utils.exceptions import AstropyUserWarning

from .core import Aperture, SkyAperture
from ._photometry_utils import (_handle_hdu_input, _handle_units,
                                _prepare_photometry_data, _validate_inputs)
from ..utils.misc import _get_version_info

__all__ = ['aperture_photometry']


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

    wcs : WCS object, optional
        A world coordinate system (WCS) transformation that supports the
        `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).  Used only if the input
        ``apertures`` contains a `SkyAperture` object.  If ``data`` is
        an `~astropy.io.fits.ImageHDU` or `~astropy.io.fits.HDUList`,
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
    meta['version'] = _get_version_info()
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
