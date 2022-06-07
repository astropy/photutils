# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides parameter validation tools.
"""

import numpy as np


def as_pair(name, value, lower_bound=None, upper_bound=None, check_odd=False):
    """
    Define a pair of integer values as a 1D array.

    Parameters
    ----------
    name : str
        The name of the parameter, which is used in error messages.

    value : int or int array_like
        The input value.

    lower_bound : int or int array_like, optional
        A tuple defining the allowed lower bound of the value. The first
        element is the bound and the second element indicates whether
        the bound is exclusive (0) or inclusive (1).

    upper_bound : (2,) int tuple, optional
        A tuple defining the allowed upper bounds of the value along
        each axis. For each axis, if ``value`` is larger than the bound,
        it is reset to the bound. ``upper_bound`` is typically set to an
        image shape.

    check_odd : bool, optional
        Whether to raise a `ValueError` if the values are not odd along
        both axes.

    Returns
    -------
    result : (2,) `~numpy.ndarray`
        The pair as a 1D array of two integers.

    Examples
    --------
    >>> from photutils.utils._parameters import as_pair

    >>> as_pair('myparam', 4)
    array([4, 4])

    >>> as_pair('myparam', (3, 4))
    array([3, 4])

    >>> as_pair('myparam', 0, lower_bound=(0, 0))
    array([0, 0])
    """
    value = np.atleast_1d(value)

    if np.any(~np.isfinite(value)):
        raise ValueError(f'{name} must be a finite value')

    if len(value) == 1:
        value = np.array((value[0], value[0]))
    if len(value) != 2:
        raise ValueError(f'{name} must have 1 or 2 elements')
    if value.ndim != 1:
        raise ValueError(f'{name} must be 1D')
    if value.dtype.kind != 'i':
        raise ValueError(f'{name} must have integer values')
    if check_odd and np.all(value % 2) != 1:
        raise ValueError(f'{name} must have an odd value for both axes')

    if lower_bound is not None:
        if len(lower_bound) != 2:
            raise ValueError('lower_bound must contain only 2 elements')
        bound, inclusive = lower_bound
        if inclusive == 1:
            oper = '>'
            mask = value <= bound
        else:
            oper = '>='
            mask = value < bound
        if np.any(mask):
            raise ValueError(f'{name} must be {oper} {bound}')

    if upper_bound is not None:
        # if value is larger than upper_bound, set to upper_bound;
        # upper_bound is typically set to an image shape
        value = np.array((min(value[0], upper_bound[0]),
                          min(value[1], upper_bound[1])))

    return value
