# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines tools to perform aperture photometry.
"""

import warnings

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture.core import Aperture, SkyAperture
from photutils.utils._misc import _get_meta

__all__ = ['aperture_photometry']


def aperture_photometry(data, apertures, error=None, mask=None,
                        method='exact', subpixels=5, wcs=None):
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
        The 2D array on which to perform photometry. ``data`` should be
        background-subtracted.  If ``data`` is a
        `~astropy.units.Quantity` array, then ``error`` (if input) must
        also be a `~astropy.units.Quantity` array with the same units.
        See the Notes section below for more information about
        `~astropy.nddata.NDData` input.

    apertures : `~photutils.aperture.Aperture` or list of `~photutils.aperture.Aperture`
        The aperture(s) to use for the photometry.  If ``apertures`` is
        a list of `~photutils.aperture.Aperture` then they all must have
        the same position(s).

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
                each pixel is calculated. The aperture weights will
                contain values between 0 and 1.

            * ``'center'``:
                A pixel is considered to be entirely in or out of the
                aperture depending on whether its center is in or out of
                the aperture. The aperture weights will contain values
                only of 0 (out) and 1 (in).

            * ``'subpixel'``:
                A pixel is divided into subpixels (see the ``subpixels``
                keyword), each of which are considered to be entirely in
                or out of the aperture depending on whether its center
                is in or out of the aperture. If ``subpixels=1``, this
                method is equivalent to ``'center'``. The aperture
                weights will contain values between 0 and 1.

    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor
        in each dimension. That is, each pixel is divided into
        ``subpixels**2`` subpixels. This keyword is ignored unless
        ``method='subpixel'``.

    wcs : WCS object, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). Used only if the input
        ``apertures`` contains a `SkyAperture` object.

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
    `RectangularAperture` and `RectangularAnnulus` photometry with the
    "exact" method uses a subpixel approximation by subdividing each
    data pixel by a factor of 1024 (``subpixels = 32``). For rectangular
    aperture widths and heights in the range from 2 to 100 pixels, this
    subpixel approximation gives results typically within 0.001 percent
    or better of the exact value. The differences can be larger for
    smaller apertures (e.g., aperture sizes of one pixel or smaller).
    For such small sizes, it is recommend to set ``method='subpixel'``
    with a larger ``subpixels`` size.

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
                warnings.warn(f'The {key!r} keyword is be ignored. Its value '
                              'is obtained from the input NDData object.',
                              AstropyUserWarning)

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
    meta = _get_meta()
    calling_args = f"method='{method}', subpixels={subpixels}"
    meta['aperture_photometry_args'] = calling_args

    tbl = QTable()
    tbl.meta.update(meta)  # keep tbl.meta type

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
        aper_sum, aper_sum_err = aper.do_photometry(data, error=error,
                                                    mask=mask, method=method,
                                                    subpixels=subpixels)

        sum_key = sum_key_main
        sum_err_key = sum_err_key_main
        if not single_aperture:
            sum_key += f'_{i}'
            sum_err_key += f'_{i}'

        tbl[sum_key] = aper_sum
        if error is not None:
            tbl[sum_err_key] = aper_sum_err

    return tbl
