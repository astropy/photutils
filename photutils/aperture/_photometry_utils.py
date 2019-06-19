# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains tools to validate and handle photometry inputs.
"""

import warnings

import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from astropy.wcs import WCS


def _handle_hdu_input(data):  # pragma: no cover
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
