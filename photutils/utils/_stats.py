# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Nan-ignoring statistical functions, using bottleneck for performance if
available.

When bottleneck is installed, it is used only for float64 arrays. For
other dtypes (e.g., float32), NumPy is used instead to work around known
accuracy issues in bottleneck (see bottleneck issues #379 and #462, and
astropy issues #17185 and #11492).
"""

from functools import partial

import numpy as np
from astropy.units import Quantity

from photutils.utils._optional_deps import HAS_BOTTLENECK

_STAT_NAMES = (
    'nansum', 'nanmin', 'nanmax', 'nanmean', 'nanmedian', 'nanstd', 'nanvar',
)

if HAS_BOTTLENECK:
    import bottleneck as bn

    def _move_tuple_axes_last(array, axis):
        """
        Move the specified axes of a NumPy array to the last positions
        and combine them.

        Bottleneck can only take integer axis, not tuple, so this
        function takes all the axes to be operated on and combines them
        into the last dimension of the array so that we can then use
        axis=-1.

        Parameters
        ----------
        array : `~numpy.ndarray`
            The input array.

        axis : tuple of int
            The axes on which to move and combine.

        Returns
        -------
        array_new : `~numpy.ndarray`
            Array with the axes being operated on moved into the last
            dimension.
        """
        other_axes = tuple(i for i in range(array.ndim) if i not in axis)

        # Move the specified axes to the last positions
        array_new = np.transpose(array, other_axes + axis)

        # Reshape the array by combining the moved axes
        return array_new.reshape((*array_new.shape[:len(other_axes)], -1))

    def _apply_bottleneck(function, array, axis=None, **kwargs):
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
            array = _move_tuple_axes_last(array, axis=axis)
            axis = -1

        result = function(array, axis=axis, **kwargs)
        if isinstance(array, Quantity):
            if function == bn.nanvar:
                result <<= array.unit ** 2
            else:
                result = array.__array_wrap__(result)
            return result

        if isinstance(result, float):
            # For compatibility with numpy, always return a numpy scalar.
            return np.float64(result)

        return result

    bn_funcs = {
        name: partial(_apply_bottleneck, getattr(bn, name))
        for name in _STAT_NAMES
    }
    np_funcs = {name: getattr(np, name) for name in _STAT_NAMES}

    def _dtype_dispatch(func_name):
        # Dispatch to bottleneck or numpy depending on the input array
        # dtype. This is done to workaround known accuracy bugs in
        # bottleneck affecting float32 calculations.
        # See https://github.com/pydata/bottleneck/issues/379
        # See https://github.com/pydata/bottleneck/issues/462
        # See https://github.com/astropy/astropy/issues/17185
        # See https://github.com/astropy/astropy/issues/11492
        def wrapped(*args, **kwargs):
            if args[0].dtype.str[1:] == 'f8':
                return bn_funcs[func_name](*args, **kwargs)
            return np_funcs[func_name](*args, **kwargs)

        return wrapped

    (nansum, nanmin, nanmax, nanmean, nanmedian, nanstd, nanvar) = (
        _dtype_dispatch(name) for name in _STAT_NAMES
    )

else:
    (nansum, nanmin, nanmax, nanmean, nanmedian, nanstd, nanvar) = (
        getattr(np, name) for name in _STAT_NAMES
    )
