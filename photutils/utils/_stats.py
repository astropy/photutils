# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines nan-ignoring statistical functions, using bottleneck
for performance if available.
"""

import astropy.units as u
import numpy as np

from photutils.utils._optional_deps import HAS_BOTTLENECK

if HAS_BOTTLENECK:
    import bottleneck as bn


def move_tuple_axes_first(array, axis):
    """
    Bottleneck can only take integer axis, not tuple, so this function
    takes all the axes to be operated on and combines them into the
    first dimension of the array so that we can then use axis=0.
    """
    # Figure out how many axes we are operating over
    naxis = len(axis)

    # Add remaining axes to the axis tuple
    axis += tuple(i for i in range(array.ndim) if i not in axis)

    # The new position of each axis is just in order
    destination = tuple(range(array.ndim))

    # Reorder the array so that the axes being operated on are at the
    # beginning
    array_new = np.moveaxis(array, axis, destination)

    # Collapse the dimensions being operated on into a single dimension
    # so that we can then use axis=0 with the bottleneck functions
    array_new = array_new.reshape((-1,) + array_new.shape[naxis:])

    return array_new


def nanmean(array, axis=None):
    """
    A nanmean function that uses bottleneck if available.
    """
    if HAS_BOTTLENECK:
        if isinstance(axis, tuple):
            array = move_tuple_axes_first(array, axis=axis)
            axis = 0

        if isinstance(array, u.Quantity):
            return array.__array_wrap__(bn.nanmean(array, axis=axis))
        else:
            return bn.nanmean(array, axis=axis)
    else:
        return np.nanmean(array, axis=axis)


def nanmedian(array, axis=None):
    """
    A nanmedian function that uses bottleneck if available.
    """
    if HAS_BOTTLENECK:
        if isinstance(axis, tuple):
            array = move_tuple_axes_first(array, axis=axis)
            axis = 0

        if isinstance(array, u.Quantity):
            return array.__array_wrap__(bn.nanmedian(array, axis=axis))
        else:
            return bn.nanmedian(array, axis=axis)
    else:
        return np.nanmedian(array, axis=axis)


def nanstd(array, axis=None, ddof=0):
    """
    A nanstd function that uses bottleneck if available.
    """
    if HAS_BOTTLENECK:
        if isinstance(axis, tuple):
            array = move_tuple_axes_first(array, axis=axis)
            axis = 0

        if isinstance(array, u.Quantity):
            return array.__array_wrap__(bn.nanstd(array, axis=axis, ddof=ddof))
        else:
            return bn.nanstd(array, axis=axis, ddof=ddof)
    else:
        return np.nanstd(array, axis=axis, ddof=ddof)
