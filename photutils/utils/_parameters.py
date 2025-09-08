# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for parameter validation.
"""

import numpy as np
from astropy.stats import SigmaClip


class SigmaClipSentinelDefault:
    """
    A sentinel object to indicate the default value for sigma_clip.
    """

    def __init__(self, sigma=3.0, maxiters=10):
        """
        Initialize the sentinel with default SigmaClip parameters.
        """
        self.sigma = sigma
        self.maxiters = maxiters

    def __repr__(self):
        return (f'<default: SigmaClip(sigma={self.sigma}, '
                f'maxiters={self.maxiters})>')


def create_default_sigmaclip(sigma=3.0, maxiters=10):
    """
    Return a new, default SigmaClip instance.
    """
    return SigmaClip(sigma=sigma, maxiters=maxiters)


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
        msg = f'{name} must be a finite value'
        raise ValueError(msg)

    if len(value) == 1:
        value = np.array((value[0], value[0]))
    if len(value) != 2:
        msg = f'{name} must have 1 or 2 elements'
        raise ValueError(msg)
    if value.ndim != 1:
        msg = f'{name} must be 1D'
        raise ValueError(msg)
    if value.dtype.kind != 'i':
        msg = f'{name} must have integer values'
        raise ValueError(msg)
    if check_odd and np.all(value % 2) != 1:
        msg = f'{name} must have an odd value for both axes'
        raise ValueError(msg)

    if lower_bound is not None:
        if len(lower_bound) != 2:
            msg = 'lower_bound must contain only 2 elements'
            raise ValueError(msg)
        bound, inclusive = lower_bound
        if inclusive == 1:
            oper = '>'
            mask = value <= bound
        else:
            oper = '>='
            mask = value < bound
        if np.any(mask):
            msg = f'{name} must be {oper} {bound}'
            raise ValueError(msg)

    if upper_bound is not None:
        # if value is larger than upper_bound, set to upper_bound;
        # upper_bound is typically set to an image shape
        value = np.array((min(value[0], upper_bound[0]),
                          min(value[1], upper_bound[1])))

    return value
