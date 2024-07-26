# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines nan-ignoring statistical functions, using bottleneck
for performance if available.
"""

from functools import partial

import numpy as np
from astropy.units import Quantity

from photutils.utils._optional_deps import HAS_BOTTLENECK

if HAS_BOTTLENECK:
    import bottleneck as bn

    def move_tuple_axes_first(array, axis):
        """
        Move the axes in a tuple to the beginning of the array.

        Bottleneck can only take integer axis, not tuple, so this function
        takes all the axes to be operated on and combines them into the
        first dimension of the array so that we can then use axis=0.

        Parameters
        ----------
        array : `~numpy.ndarray`
            The input array.

        axis : tuple of int
            The axes on which to operate.

        Returns
        -------
        array_new : `~numpy.ndarray`
            Array with the axes being operated on moved into the first
            dimension.
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

    def apply_bottleneck(function, array, axis=None, **kwargs):
        """
        Wrap a bottleneck function to handle tuple axis.

        This function also takes care to ensure the output is of the
        expected type, i.e., a quantity, numpy array, or numpy scalar.

        Parameters
        ----------
        function : callable
            The bottleneck function to apply.

        array : `~numpy.ndarray`
            The array on which to operate.

        axis : int or tuple of int, optional
            The axis or axes on which to operate.

        **kwargs : dict, optional
            Additional keyword arguments to pass to the bottleneck
            function.

        Returns
        -------
        result : `~numpy.ndarray` or float
            The result of the bottleneck function when called with the
            ``array``, ``axis``, and ``kwargs``.
        """
        if isinstance(axis, tuple):
            array = move_tuple_axes_first(array, axis=axis)
            axis = 0

        result = function(array, axis=axis, **kwargs)
        if isinstance(array, Quantity):
            return array.__array_wrap__(result)
        elif isinstance(result, float):
            # For compatibility with numpy, always return a numpy scalar
            return np.float64(result)
        else:
            return result

    nanmean = partial(apply_bottleneck, bn.nanmean)
    nanmedian = partial(apply_bottleneck, bn.nanmedian)
    nanstd = partial(apply_bottleneck, bn.nanstd)

else:
    nanmean = np.nanmean
    nanmedian = np.nanmedian
    nanstd = np.nanstd
