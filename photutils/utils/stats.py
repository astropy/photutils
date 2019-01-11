# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np


__all__ = ['std_blocksum']


def _mesh_values(data, box_size):
    """
    Extract all the data values in boxes of size ``box_size``.

    Values from incomplete boxes, either because of the image edges or
    masked pixels, are not returned.

    Parameters
    ----------
    data : 2D `~numpy.ma.MaskedArray`
        The input masked array.

    box_size : int
        The box size.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        A 2D array containing the data values in the boxes (along the x
        axis).
    """

    data = np.ma.asanyarray(data)

    ny, nx = data.shape
    nyboxes = ny // box_size
    nxboxes = nx // box_size

    # include only complete boxes
    ny_crop = nyboxes * box_size
    nx_crop = nxboxes * box_size
    data = data[0:ny_crop, 0:nx_crop]

    # a reshaped 2D masked array with mesh data along the x axis
    data = np.ma.swapaxes(data.reshape(
        nyboxes, box_size, nxboxes, box_size), 1, 2).reshape(
            nyboxes * nxboxes, box_size * box_size)

    # include only boxes without any masked pixels
    idx = np.where(np.ma.count_masked(data, axis=1) == 0)

    return data[idx]


def std_blocksum(data, block_sizes, mask=None):
    """
    Calculate the standard deviation of block-summed data values at
    sizes of ``block_sizes``.

    Values from incomplete blocks, either because of the image edges or
    masked pixels, are not included.

    Parameters
    ----------
    data : array-like
        The 2D array to block sum.

    block_sizes : int, array-like of int
        An array of integer (square) block sizes.

    mask : array-like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Blocks that contain *any* masked data are excluded from
        calculations.

    Returns
    -------
    result : `~numpy.ndarray`
        An array of the standard deviations of the block-summed array
        for the input ``block_sizes``.
    """

    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    stds = []
    block_sizes = np.atleast_1d(block_sizes)
    for block_size in block_sizes:
        mesh_values = _mesh_values(data, block_size)
        block_sums = np.sum(mesh_values, axis=1)
        stds.append(np.std(block_sums))

    return np.array(stds)
