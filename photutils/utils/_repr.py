# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for class __repr__ and __str__ strings.
"""


def make_repr(instance, params, *, overrides=None, long=False):
    """
    Generate a __repr__ string for a class instance.

    Parameters
    ----------
    instance : object
        The class instance.

    params : str or list of str
        List of parameter names to include in the repr. The order of
        returned parameters is the same as the order of ``params``.

    overrides : `None` or dict, optional
        Dictionary of parameter names and values to override the
        instance's attributes. This is useful for cases where the
        instance's attributes are not stored long-term (e.g.,
        Background2D). The keys of ``overrides`` must also be in
        ``params``, which determines the order of the returned
        parameters.

    long : bool, optional
        Whether to use the "long" format typically used by __str__.

    Returns
    -------
    repr_str : str
        The generated __repr__ string.
    """
    cls_name = f'{instance.__class__.__name__}'
    if long:
        cls_name = f'{instance.__class__.__module__}.{cls_name}'

    if isinstance(params, str):
        params = [params]

    if (overrides is not None
            and not set(overrides.keys()).issubset(set(params))):
        msg = 'The overrides keys must be a subset of the params list.'
        raise ValueError(msg)

    cls_info = []
    for param in params:
        if overrides is not None and param in overrides:
            # overrides may contain input parameters that are not
            # stored long-term in the instance (e.g., Background2D)
            if param in instance.__dict__ and instance.__dict__[param] is None:
                value = None
            else:
                value = overrides[param]
        elif param in instance.__dict__:
            value = instance.__dict__[param]
        else:
            msg = f'Parameter {param!r} not found in instance or overrides'
            raise ValueError(msg)

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
