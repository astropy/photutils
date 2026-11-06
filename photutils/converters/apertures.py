# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Deprecated location of the ASDF aperture converters.

.. deprecated:: 3.2
    The ``photutils.converters.apertures`` module is deprecated and will
    be removed in version 4.0. Use
    ``photutils.asdf.converters.apertures`` instead.
"""

import warnings

from photutils.asdf.converters import apertures as _apertures
from photutils.utils.exceptions import PhotutilsDeprecationWarning

__all__ = list(_apertures.__all__)

_deprecation_msg = ('photutils.converters.apertures is deprecated (since '
                    'version 3.2) and will be removed in version 4.0. Use '
                    'photutils.asdf.converters.apertures instead.')


def __getattr__(name):
    if name in __all__:
        warnings.warn(_deprecation_msg, PhotutilsDeprecationWarning,
                      stacklevel=2)
        return getattr(_apertures, name)

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)


def __dir__():
    return __all__
