# Licensed under a 3-clause BSD style license - see LICENSE.rst

import collections

import numpy as np
from astropy.nddata.utils import overlap_slices
from astropy.utils import deprecated


__all__ = ['cutout_footprint']


@deprecated('0.5')
def cutout_footprint(data, position, box_size=3, footprint=None, mask=None,
                     error=None):
    """
    Cut out a region from data (and optional mask and error) centered at
    specified (x, y) position.

    The size of the region is specified via the ``box_size`` or
    ``footprint`` keywords.  The output mask for the cutout region
    represents the combination of the input mask and footprint mask.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    position : 2 tuple
        The ``(x, y)`` pixel coordinate of the center of the region.

    box_size : scalar or tuple, optional
        The size of the region to cutout from ``data``.  If ``box_size``
        is a scalar then a square box of size ``box_size`` will be used.
        If ``box_size`` has two elements, they should be in ``(ny, nx)``
        order.  Either ``box_size`` or ``footprint`` must be defined.
        If they are both defined, then ``footprint`` overrides
        ``box_size``.

    footprint : `~numpy.ndarray` of bools, optional
        A boolean array where `True` values describe the local footprint
        region.  ``box_size=(n, m)`` is equivalent to
        ``footprint=np.ones((n, m))``.  Either ``box_size`` or
        ``footprint`` must be defined.  If they are both defined, then
        ``footprint`` overrides ``box_size``.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    Returns
    -------
    region_data : `~numpy.ndarray`
        The ``data`` cutout.

    region_mask : `~numpy.ndarray`
        The ``mask`` cutout.

    region_error : `~numpy.ndarray`
        The ``error`` cutout.

    slices : tuple of slices
        Slices in each dimension of the ``data`` array used to define
        the cutout region.
    """

    if len(position) != 2:
        raise ValueError('position must have a length of 2')

    if footprint is None:
        if box_size is None:
            raise ValueError('box_size or footprint must be defined.')
        if not isinstance(box_size, collections.Iterable):
            shape = (box_size, box_size)
        else:
            if len(box_size) != 2:
                raise ValueError('box_size must have a length of 2')
            shape = box_size
        footprint = np.ones(shape, dtype=bool)
    else:
        footprint = np.asanyarray(footprint, dtype=bool)

    slices_large, slices_small = overlap_slices(data.shape, footprint.shape,
                                                position[::-1])
    region_data = data[slices_large]

    if error is not None:
        region_error = error[slices_large]
    else:
        region_error = None

    if mask is not None:
        region_mask = mask[slices_large]
    else:
        region_mask = np.zeros_like(region_data, dtype=bool)
    footprint_mask = ~footprint
    footprint_mask = footprint_mask[slices_small]    # trim if necessary
    region_mask = np.logical_or(region_mask, footprint_mask)

    return region_data, region_mask, region_error, slices_large
