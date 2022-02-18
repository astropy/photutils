# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines tools to perform aperture photometry.
"""
import warnings

import numpy as np
from astropy.nddata import NDData, StdDevUncertainty
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import Aperture, CircularAperture, SkyCircularAperture, SkyAperture
from photutils.aperture._photometry_utils import (_handle_units, _prepare_photometry_data,
                                _validate_inputs)

__all__ = ['fwhm_cog']



def fwhm_cog(data, aperture, uncertainty=None, mask=None,
             aperture_method='center', subpixels=5, wcs=None):
    """
    Compute the full width at half max of a source using the "curve of growth"
    method.

    Parameters
    ----------
    data : array_like, `~astropy.units.Quantity`, `~astropy.nddata.NDData`
        The 2D array on which to perform photometry. ``data`` should be
        background-subtracted.  If ``data`` is a
        `~astropy.units.Quantity` array, then ``uncertainty`` (if input) must
        also be a `~astropy.units.Quantity` array with the same units.
        See the Notes section below for more information about
        `~astropy.nddata.NDData` input.

    aperture : `~photutils.aperture.Aperture` or a tupople
        The aperture to use for calculating the FWHM. For convenience, this can
        also be a length-3 tuple giving the (xcen, ycen, radius), or a length-2
        tuple (skycoord, radius) which are equivalent to passing in
        `~photutils.aperture.CircularAperture` or
        `~photutils.aperture.SkyCircularAperture`.  Note that if passing an
        aperture in it must be a single scalar position.

    uncertainty : array_like or `~astropy.units.Quantity` or bool, optional
        The pixel-wise Gaussian 1-sigma uncertainty of the input ``data``.
        These are used to weight pixels using inverse-variance weighting.  If a
        `~astropy.units.Quantity` array, then ``data`` must also be a
        `~astropy.units.Quantity` array with the same units.
        If False, the weighting step will be skipped even if uncertainties are
        present from a passed-in `NDData` object

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    aperture_method : {'exact', 'center', 'subpixel'}, optional
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

    wcs : WCS object, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). Used only if the input
        ``aperture`` is expressed in sky coordinates.

    Returns
    -------
    result
    """
    # this parsing is from aperture_photometry. TODO: break out into a utility function shared between these functions
    if isinstance(data, NDData):
        nddata_attr = {'uncertainty': uncertainty, 'mask': mask, 'wcs': wcs}
        for key, value in nddata_attr.items():
            if value is not None:
                warnings.warn(f'The {key!r} keyword will be ignored. Its value '
                              'is obtained from the input NDData object.',
                              AstropyUserWarning)

        mask = data.mask
        wcs = data.wcs

        if isinstance(data.uncertainty, StdDevUncertainty):
            if data.uncertainty.unit is None:
                uncertainty = data.uncertainty.array
            else:
                uncertainty = data.uncertainty.array * data.uncertainty.unit

        if data.unit is not None:
            data = u.Quantity(data.data, unit=data.unit)
        else:
            data = data.data

        return fwhm_cog(data, aperture, uncertainty=uncertainty, mask=mask, wcs=wcs, aperture_method=aperture_method, subpixels=subpixels)

    # validate inputs
    data, uncertainty = _validate_inputs(data, uncertainty)

    # handle data, error, and unit inputs
    # output data and error are ndarray without units
    data, uncertainty, unit = _handle_units(data, uncertainty)

    # compute variance and apply input mask
    data, variance = _prepare_photometry_data(data, uncertainty, mask)

    # now the fwhm-specific bits
    if not isinstance(aperture, Aperture):
        try:
            nap = len(aperture)
        except TypeError:
            raise TypeError('aperture is not an Aperture object or a sequence')

        if nap == 3:
            aperture = CircularAperture(aperture[:2], aperture[2])
            #TODO: allow a unitful radius that uses the platescale at the given location as a best-guess
        elif nap == 2:
            aperture = SkyCircularAperture(*aperture)
        else:
            raise ValueError('aperture is not length 2 or 3 nor an Aperture')

    if not aperture.isscalar:
        raise ValueError('aperture has a list of positions, not a single position')

    if isinstance(aperture, SkyAperture):
        pixel_aperture = aperture.to_pixel(wcs)
        assert pixel_aperture is not aperture, 'These should always be different now'  #TODO: can remove this if someone else is certain this is
    else:
        pixel_aperture = aperture

    apermask = aperture.to_mask(method=aperture_method, subpixels=subpixels)
    masked_values = apermask.get_values(data)

    # coordinates masking cannot be "exact" because that's fractional, but subpixel still makes sense
    coomask = aperture.to_mask(method='center' if aperture_method=='exact' else aperture_method,
                                 subpixels=subpixels)
    ygrid, xgrid = np.meshgrid(*[np.arange(sh) for sh in data.shape], copy=False)
    xs = coomask.get_values(xgrid.T)
    ys = coomask.get_values(ygrid.T)

    if aperture is pixel_aperture:
        # a pixel-space COG
        dx = xs - pixel_aperture.positions[0]
        dy = ys - pixel_aperture.positions[1]
        px_distance = np.hypot(dx, dy)
        dsortidx = np.argsort(px_distance)
        dsort = px_distance[dsortidx]
    else:
        # an on-sky COG
        skycoords = wcs.to_world(xs, ys)
        sky_distance = aperture.positions.separation(skycoords)
        dsortidx = np.argsort(sky_distance)
        dsort = sky_distance[dsortidx]

    curve_of_growth = np.cumsum(masked_values[dsortidx])
    hwhm = np.interp(curve_of_growth[-1]/2., curve_of_growth, dsort)  # interp works correctly for unitful `fp`!

    return hwhm*2, dsort, curve_of_growth