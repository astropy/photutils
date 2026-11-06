# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Helpers shared by the photutils ASDF converters.
"""


def optional_params(node, *names):
    """
    Return the optional parameters present in a YAML tree as a keyword
    dictionary.

    A parameter that is absent from ``node`` is omitted from the result
    instead of being returned as `None`, so that the class supplies its
    own default. This is required because the defaults are not always
    `None` and may differ between related classes (e.g., the ``theta``
    default differs between the pixel and sky apertures).

    Parameters
    ----------
    node : dict
        The YAML tree of a tagged object.

    *names : str
        The names of the optional parameters.

    Returns
    -------
    params : dict
        The subset of ``names`` that is present in ``node``, mapped to
        the corresponding values.
    """
    return {name: node[name] for name in names if name in node}


def add_parameter_bounds(node, model):
    """
    Store every parameter bound of a model in its YAML tree.

    The ``asdf-astropy`` transform converter omits a bound whose limits
    are all zero or `None` (e.g., a nonnegative bound of ``(0,
    None)``), so such a bound would be lost on a round-trip.

    Parameters
    ----------
    node : dict
        The YAML tree of the model, which is modified in place.

    model : `astropy.modeling.Model`
        The model being serialized.
    """
    bounds = {name: bound for name, bound in model.bounds.items()
              if any(limit is not None for limit in bound)}
    if bounds:
        node['bounds'] = bounds
