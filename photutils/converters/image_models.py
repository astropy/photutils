# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Deprecated location of the ASDF image PSF model converters.

.. deprecated:: 3.2
    The ``photutils.converters.image_models`` module is deprecated and
    will be removed in version 4.0. Use
    ``photutils.asdf.converters.image_models`` instead.
"""

import warnings

from photutils.asdf.converters import image_models as _image_models
from photutils.utils.exceptions import PhotutilsDeprecationWarning

__all__ = list(_image_models.__all__)

_deprecation_msg = ('photutils.converters.image_models is deprecated (since '
                    'version 3.2) and will be removed in version 4.0. Use '
                    'photutils.asdf.converters.image_models instead.')


def __getattr__(name):
    if name in __all__:
        warnings.warn(_deprecation_msg, PhotutilsDeprecationWarning,
                      stacklevel=2)
        return getattr(_image_models, name)

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)


def __dir__():
    return __all__
