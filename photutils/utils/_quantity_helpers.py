# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides Quantity helper tools.
"""

import astropy.units as u
import numpy as np


def process_quantities(values, names):
    """
    Check and remove units of input values.

    If any of the input values have units then they all must have
    units and the units must be the same.

    The returned values are the input values with units removed and the
    unit.

    Parameters
    ----------
    values : list of scalar, `~numpy.ndarray`, or `~astropy.units.Quantity`
        A list of values.

    names : list of str
        A list of names corresponding to the input ``values``.

    Returns
    -------
    values : list of scalar or `~numpy.ndarray`
        A list of values, where units have been removed.

    unit : `~astropy.unit.Unit`
        The common unit for the input values. `None` will be returned if
        all the input values do not have units.

    Raises
    ------
    ValueError
        If the input values do not all have the same units.
    """
    if len(values) != len(names):
        raise ValueError('The number of values must match the number of '
                         'names.')

    all_units = {name: getattr(arr, 'unit', None)
                 for arr, name in zip(values, names, strict=True)
                 if arr is not None}
    unit = set(all_units.values())

    if len(unit) > 1:
        values = list(all_units.keys())
        msg = [f'The inputs {values} must all have the same units:']
        indent = ' ' * 4
        for key, value in all_units.items():
            if value is None:
                msg.append(f'{indent}{key} does not have units')
            else:
                msg.append(f'{indent}{key} has units of {value}')
        msg = '\n'.join(msg)
        raise ValueError(msg)

    # extract the unit and remove it from the return values
    unit = unit.pop()
    if unit is not None:
        values = [val.value if val is not None else val for val in values]

    return values, unit


def isscalar(value):
    """
    Check if a value is a scalar.

    This works for both `~astropy.units.Quantity` and scalars.

    `numpy.isscalar` always returns False for `~astropy.units.Quantity`
    objects.

    Parameters
    ----------
    value : `~astropy.units.Quantity`, scalar, or array_like
        The value to check.

    Returns
    -------
    isscalar : bool
        `True` if the value is a scalar, `False` otherwise.
    """
    if isinstance(value, u.Quantity):
        return value.isscalar

    return np.isscalar(value)
