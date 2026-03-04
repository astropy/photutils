# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for parameter validation.
"""

import inspect
import warnings
from functools import wraps

import numpy as np
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyDeprecationWarning


class SigmaClipSentinelDefault:
    """
    A sentinel object to indicate the default value for sigma_clip.

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations for the clipping limit.

    maxiters : int, optional
        The maximum number of sigma-clipping iterations.
    """

    def __init__(self, sigma=3.0, maxiters=10):
        self.sigma = sigma
        self.maxiters = maxiters

    def __repr__(self):
        return (f'<default: SigmaClip(sigma={self.sigma}, '
                f'maxiters={self.maxiters})>')


def create_default_sigmaclip(sigma=3.0, maxiters=10):
    """
    Return a new, default SigmaClip instance.

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations for the clipping limit.

    maxiters : int, optional
        The maximum number of sigma-clipping iterations.

    Returns
    -------
    result : `~astropy.stats.SigmaClip`
        A new `~astropy.stats.SigmaClip` instance.
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

    lower_bound : tuple of 2 int, optional
        A tuple defining the allowed lower bound of the value. The first
        element is the bound; the second is 0 for exclusive or 1 for
        inclusive (e.g. (0, 1) means value must be >= 0).

    upper_bound : tuple of 2 int, optional
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

    >>> as_pair('myparam', 0, lower_bound=(0, 1))
    array([0, 0])
    """
    value = np.atleast_1d(value)

    if value.ndim != 1:
        msg = f'{name} must be 1D'
        raise ValueError(msg)

    if np.any(~np.isfinite(value)):
        msg = f'{name} must be a finite value'
        raise ValueError(msg)

    if len(value) not in (1, 2):
        msg = f'{name} must have 1 or 2 elements'
        raise ValueError(msg)
    if len(value) == 1:
        value = np.array((value[0], value[0]))

    if value.dtype.kind != 'i':
        msg = f'{name} must have integer values'
        raise ValueError(msg)
    if check_odd and np.any(value % 2 != 1):
        msg = f'{name} must have an odd value for both axes'
        raise ValueError(msg)

    if lower_bound is not None:
        if len(lower_bound) != 2:
            msg = 'lower_bound must contain only 2 elements'
            raise ValueError(msg)
        bound, inclusive = lower_bound
        if inclusive:
            oper = '>='
            mask = value < bound
        else:
            oper = '>'
            mask = value <= bound
        if np.any(mask):
            msg = f'{name} must be {oper} {bound}'
            raise ValueError(msg)

    if upper_bound is not None:
        if len(upper_bound) != 2:
            msg = 'upper_bound must contain only 2 elements'
            raise ValueError(msg)
        # If value is larger than upper_bound, set to upper_bound;
        # upper_bound is typically set to an image shape
        value = np.array((min(value[0], upper_bound[0]),
                          min(value[1], upper_bound[1])))

    return value


def warn_positional_kwargs(since, *, until=None):
    """
    Decorator to warn when optional arguments are passed positionally.

    Parameters that have no default value (i.e., required parameters)
    are allowed positionally. Parameters with default values (i.e.,
    optional parameters) will trigger a deprecation warning if passed
    positionally.

    Parameters
    ----------
    since : str or int
        The version in which passing optional arguments positionally is
        deprecated.

    until : str or int, optional
        The version in which passing optional arguments positionally
        will be removed. If `None`, the removal version is not mentioned
        in the warning message.

    Returns
    -------
    decorator : function
        A decorator function that can be applied to any function to warn
        about positional arguments.
    """
    def decorator(func):  # numpydoc ignore=GL08
        since_str = str(since)
        until_str = str(until) if until is not None else None
        sig = inspect.signature(func)
        n_positional = 0
        param_names = []
        for name, param in sig.parameters.items():
            param_names.append(name)
            if (param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                               inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and param.default is inspect.Parameter.empty):
                n_positional += 1

        @wraps(func)
        def wrapper(*args, **kwargs):  # numpydoc ignore=GL08
            if len(args) > n_positional:
                extra_names = param_names[n_positional:len(args)]
                quoted = [f"'{name}'" for name in extra_names]
                if len(quoted) == 1:
                    params_str = quoted[0]
                    pronoun = 'it'
                    kwarg_noun = 'a keyword argument'
                elif len(quoted) == 2:
                    params_str = f'{quoted[0]} and {quoted[1]}'
                    pronoun = 'them'
                    kwarg_noun = 'keyword arguments'
                else:
                    params_str = (', '.join(quoted[:-1])
                                  + f', and {quoted[-1]}')
                    pronoun = 'them'
                    kwarg_noun = 'keyword arguments'
                examples_str = ', '.join(f'{name}=...' for name in extra_names)
                remove_str = 'a future version'
                if until_str is not None:
                    remove_str = f'version {until_str}'
                msg = (f'Passing {params_str} positionally to '
                       f"'{func.__name__}' is deprecated as of version "
                       f'{since_str} and will be removed in {remove_str}. '
                       f'Pass {pronoun} as {kwarg_noun} instead '
                       f'(e.g., {examples_str}).')
                warnings.warn(msg, AstropyDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator
