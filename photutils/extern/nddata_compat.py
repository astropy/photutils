# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains functions from astropy.nddata that are used to maintain
backwards-compatiblity with older versions of astropy
"""

import inspect

import numpy as np


def extract_array(*args, **kwargs):
    """
    Wrapper for astropy.nddata.utils.extract_array that reproduces v1.1
    behavior even if an older Astropy is installed.
    """

    from astropy.nddata.utils import extract_array

    # fill_value keyword is not in v1.0.x
    if 'fill_value' in inspect.getfullargspec(extract_array)[0]:
        return extract_array(*args, **kwargs)
    else:
        return _extract_array_astropy1p1(*args, **kwargs)


# Everything below is taken directly from Astropy v1.1.1 to allow the above
# function to work like v1.1 when older astropys are installed
def _extract_array_astropy1p1(array_large, shape, position, mode='partial',
                              fill_value=np.nan, return_position=False):
    """
    Extract a smaller array of the given shape and position from a
    larger array.


    Parameters
    ----------
    array_large : `~numpy.ndarray`
        The array from which to extract the small array.
    shape : tuple or int
        The shape of the extracted array (for 1D arrays, this can be an
        `int`).  See the ``mode`` keyword for additional details.
    position : tuple of numbers or number
        The position of the small array's center with respect to the
        large array.  The pixel coordinates should be in the same order
        as the array shape.  Integer positions are at the pixel centers
        (for 1D arrays, this can be a number).
    mode : {'partial', 'trim', 'strict'}, optional
        The mode used for extracting the small array.  For the
        ``'partial'`` and ``'trim'`` modes, a partial overlap of the
        small array and the large array is sufficient.  For the
        ``'strict'`` mode, the small array has to be fully contained
        within the large array, otherwise an
        `~astropy.nddata.utils.PartialOverlapError` is raised.   In all
        modes, non-overlapping arrays will raise a
        `~astropy.nddata.utils.NoOverlapError`.  In ``'partial'`` mode,
        positions in the small array that do not overlap with the large
        array will be filled with ``fill_value``.  In ``'trim'`` mode
        only the overlapping elements are returned, thus the resulting
        small array may be smaller than the requested ``shape``.
    fill_value : number, optional
        If ``mode='partial'``, the value to fill pixels in the extracted
        small array that do not overlap with the input ``array_large``.
        ``fill_value`` must have the same ``dtype`` as the
        ``array_large`` array.
    return_position : boolean, optional
        If `True`, return the coordinates of ``position`` in the
        coordinate system of the returned array.

    Returns
    -------
    array_small : `~numpy.ndarray`
        The extracted array.
    new_position : tuple
        If ``return_position`` is true, this tuple will contain the
        coordinates of the input ``position`` in the coordinate system
        of ``array_small``. Note that for partially overlapping arrays,
        ``new_position`` might actually be outside of the
        ``array_small``; ``array_small[new_position]`` might give wrong
        results if any element in ``new_position`` is negative.

    Examples
    --------
    We consider a large array with the shape 11x10, from which we extract
    a small array of shape 3x5:

    >>> import numpy as np
    >>> from astropy.nddata.utils import extract_array
    >>> large_array = np.arange(110).reshape((11, 10))
    >>> extract_array(large_array, (3, 5), (7, 7))
    array([[65, 66, 67, 68, 69],
           [75, 76, 77, 78, 79],
           [85, 86, 87, 88, 89]])
    """

    if np.isscalar(shape):
        shape = (shape, )
    if np.isscalar(position):
        position = (position, )

    if mode not in ['partial', 'trim', 'strict']:
        raise ValueError("Valid modes are 'partial', 'trim', and 'strict'.")
    large_slices, small_slices = _overlap_slices_astropy1p1(
        array_large.shape, shape, position, mode=mode)
    extracted_array = array_large[large_slices]
    if return_position:
        new_position = [i - s.start for i, s in zip(position, large_slices)]
    # Extracting on the edges is presumably a rare case, so treat special here
    if (extracted_array.shape != shape) and (mode == 'partial'):
        extracted_array = np.zeros(shape, dtype=array_large.dtype)
        extracted_array[:] = fill_value
        extracted_array[small_slices] = array_large[large_slices]
        if return_position:
            new_position = [i + s.start for i, s in zip(new_position,
                                                        small_slices)]
    if return_position:
        return extracted_array, tuple(new_position)
    else:
        return extracted_array


def _overlap_slices_astropy1p1(large_array_shape, small_array_shape, position,
                               mode='partial'):
    """
    Get slices for the overlapping part of a small and a large array.

    Given a certain position of the center of the small array, with
    respect to the large array, tuples of slices are returned which can be
    used to extract, add or subtract the small array at the given
    position. This function takes care of the correct behavior at the
    boundaries, where the small array is cut of appropriately.
    Integer positions are at the pixel centers.

    Parameters
    ----------
    large_array_shape : tuple or int
        The shape of the large array (for 1D arrays, this can be an
        `int`).
    small_array_shape : tuple or int
        The shape of the small array (for 1D arrays, this can be an
        `int`).  See the ``mode`` keyword for additional details.
    position : tuple of numbers or number
        The position of the small array's center with respect to the
        large array.  The pixel coordinates should be in the same order
        as the array shape.  Integer positions are at the pixel centers.
        For any axis where ``small_array_shape`` is even, the position
        is rounded up, e.g. extracting two elements with a center of
        ``1`` will define the extracted region as ``[0, 1]``.
    mode : {'partial', 'trim', 'strict'}, optional
        In ``'partial'`` mode, a partial overlap of the small and the
        large array is sufficient.  The ``'trim'`` mode is similar to
        the ``'partial'`` mode, but ``slices_small`` will be adjusted to
        return only the overlapping elements.  In the ``'strict'`` mode,
        the small array has to be fully contained in the large array,
        otherwise an `~astropy.nddata.utils.PartialOverlapError` is
        raised.  In all modes, non-overlapping arrays will raise a
        `~astropy.nddata.utils.NoOverlapError`.

    Returns
    -------
    slices_large : tuple of slices
        A tuple of slice objects for each axis of the large array, such
        that ``large_array[slices_large]`` extracts the region of the
        large array that overlaps with the small array.
    slices_small : slice
        A tuple of slice objects for each axis of the small array, such
        that ``small_array[slices_small]`` extracts the region that is
        inside the large array.
    """

    if mode not in ['partial', 'trim', 'strict']:
        raise ValueError('Mode can be only "partial", "trim", or "strict".')
    if np.isscalar(small_array_shape):
        small_array_shape = (small_array_shape, )
    if np.isscalar(large_array_shape):
        large_array_shape = (large_array_shape, )
    if np.isscalar(position):
        position = (position, )

    if len(small_array_shape) != len(large_array_shape):
        raise ValueError('"large_array_shape" and "small_array_shape" must '
                         'have the same number of dimensions.')

    if len(small_array_shape) != len(position):
        raise ValueError('"position" must have the same number of dimensions '
                         'as "small_array_shape".')
    # Get edge coordinates
    edges_min = [_round(pos + 0.5 - small_shape / 2. + _offset(small_shape))
                 for (pos, small_shape) in zip(position, small_array_shape)]
    edges_max = [_round(pos + 0.5 + small_shape / 2. + _offset(small_shape))
                 for (pos, small_shape) in zip(position, small_array_shape)]

    for e_max in edges_max:
        if e_max <= 0:
            raise NoOverlapError('Arrays do not overlap.')
    for e_min, large_shape in zip(edges_min, large_array_shape):
        if e_min >= large_shape:
            raise NoOverlapError('Arrays do not overlap.')

    if mode == 'strict':
        for e_min in edges_min:
            if e_min < 0:
                raise PartialOverlapError('Arrays overlap only partially.')
        for e_max, large_shape in zip(edges_max, large_array_shape):
            if e_max >= large_shape:
                raise PartialOverlapError('Arrays overlap only partially.')

    # Set up slices
    slices_large = tuple(slice(max(0, edge_min), min(large_shape, edge_max))
                         for (edge_min, edge_max, large_shape) in
                         zip(edges_min, edges_max, large_array_shape))
    if mode == 'trim':
        slices_small = tuple(slice(0, slc.stop - slc.start)
                             for slc in slices_large)
    else:
        slices_small = tuple(slice(max(0, -edge_min),
                                   min(large_shape - edge_min,
                                       edge_max - edge_min))
                             for (edge_min, edge_max, large_shape) in
                             zip(edges_min, edges_max, large_array_shape))

    return slices_large, slices_small


def _round(a):
    '''Always round up.

    ``np.round`` cannot be used here, because it rounds .5 to the nearest
    even number.
    '''
    return int(np.floor(a + 0.5))


def _offset(a):
    '''Offset by 0.5 for an even array.

    For an array with an odd number of elements, the center is
    symmetric, e.g. for 3 elements, it's center +/-1 elements, but for
    four elements it's center -2 / +1
    This function introduces that offset.
    '''
    if np.mod(a, 2) == 0:
        return -0.5
    else:
        return 0.


class NoOverlapError(ValueError):
    '''Raised when determining the overlap of non-overlapping arrays.'''
    pass


class PartialOverlapError(ValueError):
    '''Raised when arrays only partially overlap.'''
    pass
