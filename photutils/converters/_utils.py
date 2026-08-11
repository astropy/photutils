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
