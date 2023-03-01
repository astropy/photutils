# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides Quantity helper tools.
"""


def process_quantities(arrays, names):
    """
    Check units of input arrays.

    If any of the input arrays have units then they all must have
    units and the units must be the same.

    Unitless `~numpy.ndarray` objects are returned along with the array
    unit.

    Parameters
    ----------
    arrays : list of `~numpy.ndarray` or `~astropy.units.Quantity`
        A list of arrays.

    names : list of str
        A list of names corresponding to the input ``arrays``.

    Returns
    -------
    arrays : list of `~numpy.ndarray`
        A list of numpy arrays, where units have been removed.

    unit : `~astropy.unit.Unit`
        The data unit. `None` will be returned if the input arrays do
        not have units.

    Raises
    ------
    ValueError
        If the input arrays do not all have the same units.
    """
    unit = {getattr(arr, 'unit', None) for arr in arrays if arr is not None}
    if len(unit) > 1:
        if len(names) == 2:
            str_names = f'{names[0]} or {names[1]}'
        elif len(names) > 2:
            names1 = ', '.join(names[:-1])
            str_names = f'{names1}, or {names[-1]}'
        raise ValueError(f'If {str_names} has units, then they must have the '
                         'same units.')
    unit = unit.pop()
    if unit is not None:
        arrays = [arr.value if arr is not None else arr for arr in arrays]
    return arrays, unit
