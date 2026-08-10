# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Helpers shared by the aperture photometry and statistics classes.

The `~photutils.aperture.AperturePhotometry` and
`~photutils.aperture.ApertureStats` classes accept the same ``data``,
``error``, ``mask``, and ``wcs`` inputs and present their per-source
results with the same scalar-collapse convention. Both also feed the
same batch Cython drivers. The shared logic lives here so that the two
code paths cannot silently diverge.
"""

import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import StdDevUncertainty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture._segmentation import SEG_METHOD_CODES

__all__ = []


# Types of the public per-source attributes that are collapsed to a
# scalar for a scalar instance (see ``collapse_scalar_value``).
SCALAR_COLLAPSE_TYPES = (np.ndarray, SkyCoord, list, tuple)


def unpack_nddata(data, error, mask, wcs):
    """
    Extract the data, error, mask, and WCS from a
    `~astropy.nddata.NDData` object.

    A warning is emitted for each of the ``error``, ``mask``, and
    ``wcs`` keywords that was also input directly, because the
    `~astropy.nddata.NDData` attributes take precedence.

    Parameters
    ----------
    data : `~astropy.nddata.NDData`
        The NDData object.

    error, mask, wcs : object
        The values input directly by the user. These are ignored and
        are used only to emit the warnings.

    Returns
    -------
    data : `~numpy.ndarray` or `~astropy.units.Quantity`
        The NDData data array, as a Quantity if it has units.

    error : `~numpy.ndarray`, `~astropy.units.Quantity`, or `None`
        The standard-deviation uncertainty array, or `None` if the
        NDData uncertainty is not a
        `~astropy.nddata.StdDevUncertainty`.

    mask : `~numpy.ndarray` or `None`
        The NDData mask.

    wcs : WCS object or `None`
        The NDData WCS.
    """
    nddata_attr = {'error': error, 'mask': mask, 'wcs': wcs}
    for key, value in nddata_attr.items():
        if value is not None:
            msg = (f'The {key!r} keyword is ignored. Its value '
                   'is obtained from the input NDData object.')
            warnings.warn(msg, AstropyUserWarning)

    mask = data.mask
    wcs = data.wcs

    error = None
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


def validate_array(array, name, *, ndim=2, shape=None):
    """
    Validate the dimensionality and shape of an input array.

    Parameters
    ----------
    array : array_like or `None`
        The array to validate. `None` (and, for a mask,
        `numpy.ma.nomask`) is returned unchanged.

    name : str
        The name of the array, used in the error messages.

    ndim : int, optional
        The required number of dimensions.

    shape : tuple of int or `None`, optional
        The required shape. If `None`, the shape is not checked.

    Returns
    -------
    array : `~numpy.ndarray` or `None`
        The input array as an ndarray, or `None`.
    """
    if name == 'mask' and array is np.ma.nomask:
        array = None
    if array is not None:
        array = np.asanyarray(array)
        if array.ndim != ndim:
            msg = f'{name} must be a {ndim}D array'
            raise ValueError(msg)
        if shape is not None and array.shape != shape:
            msg = f'data and {name} must have the same shape'
            raise ValueError(msg)
    return array


def collapse_scalar_value(value):
    """
    Collapse the leading length-1 (source) axis of a per-source value.

    Parameters
    ----------
    value : object
        The per-source value.

    Returns
    -------
    value : object
        The first element of ``value`` if it has a leading length-1
        axis, otherwise ``value`` unchanged.
    """
    try:
        if len(value) == 1:
            return value[0]
    except TypeError:  # value has no len
        pass
    return value


def validate_mask_method(method, subpixels, *, method_name='mask method'):
    """
    Validate the aperture-mask method and subpixels keywords.

    Parameters
    ----------
    method : {'exact', 'center', 'subpixel'}
        The aperture-mask method.

    subpixels : int
        The subsampling factor per axis used when
        ``method='subpixel'``. It is not validated for the other
        methods, which ignore it.

    method_name : str, optional
        The name of the method keyword, used in the error message.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``'exact'``, ``'center'``, or
        ``'subpixel'``, or if ``subpixels`` is not a strictly positive
        integer when ``method='subpixel'``.
    """
    if method not in ('center', 'subpixel', 'exact'):
        msg = f'Invalid {method_name}: {method!r}'
        raise ValueError(msg)

    # bool is a subclass of int, so it is rejected explicitly to
    # prevent subpixels=True from being silently used as 1
    if ((method == 'subpixel')
            and (isinstance(subpixels, bool)
                 or not isinstance(subpixels, int) or subpixels <= 0)):
        msg = 'subpixels must be a strictly positive integer'
        raise ValueError(msg)


def batch_array_supported(array):
    """
    Whether an array has a dtype supported by the batch Cython drivers.

    Parameters
    ----------
    array : object
        The array to test.

    Returns
    -------
    supported : bool
        `True` if the array is a plain `~numpy.ndarray` with a numeric
        or boolean dtype of at most 8 bytes.
    """
    return (type(array) is np.ndarray and array.dtype.kind in 'fiub'
            and array.dtype.itemsize <= 8)


def batch_inputs_supported(data, error, mask):
    """
    Whether the data, error, and mask arrays are supported by the batch
    Cython drivers.

    Parameters
    ----------
    data : object
        The data array.

    error : object or `None`
        The error array.

    mask : object or `None`
        The boolean mask array.

    Returns
    -------
    supported : bool
        `True` if all of the input arrays are supported, in which case
        the batch code path can be used. Otherwise `False`, and the
        caller must use the mask-based code path.
    """
    if not batch_array_supported(data):
        return False
    if error is not None and not batch_array_supported(error):
        return False
    return not (mask is not None
                and (not isinstance(mask, np.ndarray)
                     or mask.dtype != bool
                     or mask.shape != data.shape))


def batch_mask_plane(data, mask, *, mask_nonfinite):
    """
    Build the uint8 mask plane used by the batch Cython kernels.

    Bit 1 (value 1) marks input-masked pixels and bit 2 (value 2) marks
    non-finite ``data`` pixels; any nonzero value excludes the pixel.
    Folding the non-finite pixels into the plane lets the caller exclude
    them from the sum, area, and valid-pixel count while still flagging
    them as ``non_finite_data`` rather than ``masked_pixels``.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The data array.

    mask : `~numpy.ndarray` (bool) or `None`
        The input mask.

    mask_nonfinite : bool
        Whether to fold the non-finite ``data`` values into the plane.
        When `False`, the non-finite pixels are left in the data so
        that they corrupt the sum (the 3.0.0 behavior, used by the
        legacy `~photutils.aperture.aperture_photometry` function).

    Returns
    -------
    plane : `~numpy.ndarray` (uint8) or `None`
        The C-contiguous mask plane, or `None` if no pixels are
        excluded.
    """
    plane = None
    if mask is not None:
        plane = mask.astype(np.uint8)
    if mask_nonfinite and data.dtype.kind == 'f':
        nonfinite = ~np.isfinite(data)
        if nonfinite.any():
            if plane is None:
                plane = np.zeros(data.shape, dtype=np.uint8)
                plane[nonfinite] = 2
            else:
                plane[nonfinite & (plane == 0)] = 2

    if plane is None:
        return None
    return np.ascontiguousarray(plane)


def batch_segmentation_arrays(segmentation, labels, mask_method):
    """
    Build the segmentation inputs for the batch Cython kernels.

    Parameters
    ----------
    segmentation : `~numpy.ndarray` or `None`
        The validated segmentation array.

    labels : `~numpy.ndarray` or `None`
        The per-aperture source labels.

    mask_method : {'none', 'mask', 'source_only', 'correct'}
        The segmentation masking method.

    Returns
    -------
    segmentation, labels : `~numpy.ndarray` (intp) or `None`
        The C-contiguous segmentation array and source labels, or
        `None` if no segmentation masking is performed.

    method_code : int
        The segmentation method code (0 disables masking).
    """
    if segmentation is None or mask_method == 'none':
        return None, None, 0

    return (np.ascontiguousarray(segmentation, dtype=np.intp),
            np.ascontiguousarray(labels, dtype=np.intp),
            SEG_METHOD_CODES[mask_method])
