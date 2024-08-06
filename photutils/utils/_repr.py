# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for class __repr__ and __str__ strings.
"""


def make_repr(instance, params, ellipsis=(), long=False):
    """
    Generate a __repr__ string for a class instance.

    Parameters
    ----------
    instance : object
        The class instance.

    params : list of str
        List of parameter names to include in the repr.

    ellipsis : list of str
        List of parameter names to replace with '...' if not None.

    long : bool
        Whether to include the module name in the class name.

    Returns
    -------
    repr_str : str
        The generated __repr__ string.
    """
    cls_name = f'{instance.__class__.__name__}'
    if long:
        cls_name = f'{instance.__class__.__module__}.{cls_name}'

    cls_info = []
    for param in params:
        value = getattr(instance, param)
        if param in ellipsis and value is not None:
            value = '...'
        cls_info.append((param, value))

    if long:
        delim = ': '
        join_str = '\n'
    else:
        delim = '='
        join_str = ', '

    fmt = [f'{key}{delim}{val!r}' for key, val in cls_info]
    fmt = f'{join_str}'.join(fmt)

    if long:
        return f'<{cls_name}>\n{fmt}'

    return f'{cls_name}({fmt})'
