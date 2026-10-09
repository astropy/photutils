# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for computing the overlap of a small array with a large array.
"""

from astropy.nddata import overlap_slices as _astropy_overlap_slices


def overlap_slices(large_array_shape, small_array_shape, position, *,
                   mode='partial'):
    """
    Get slices for the overlapping part of a small and a large array.

    This is a thin wrapper around `astropy.nddata.overlap_slices` that
    converts ``small_array_shape`` to a tuple. The astropy function
    compares ``small_array_shape`` to a tuple, which raises a
    `ValueError` for a `~numpy.ndarray` shape (e.g., as returned by
    `~photutils.utils._parameters.as_pair`) when the small array ends
    exactly at the lower edge of the large array. With a tuple, the
    expected `~astropy.nddata.NoOverlapError` is raised instead.

    Parameters
    ----------
    large_array_shape : tuple of int
        The shape of the large array.

    small_array_shape : array_like of int
        The shape of the small array.

    position : tuple of float
        The position of the small array's center with respect to the
        large array, in the same axis order as the shapes.

    mode : {'partial', 'trim', 'strict'}, optional
        The overlap mode. See `astropy.nddata.overlap_slices`.

    Returns
    -------
    slices_large : tuple of slice
        A tuple of slice objects for each axis of the large array, such
        that ``large_array[slices_large]`` extracts the region of the
        large array that overlaps with the small array.

    slices_small : tuple of slice
        A tuple of slice objects for each axis of the small array, such
        that ``small_array[slices_small]`` extracts the region that is
        inside the large array.
    """
    return _astropy_overlap_slices(large_array_shape,
                                   tuple(small_array_shape), position,
                                   mode=mode)
