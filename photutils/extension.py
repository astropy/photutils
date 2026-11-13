# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Deprecated location of the ASDF extension.

.. deprecated:: 3.2
    The ``photutils.extension`` module is deprecated and will be removed
    in version 4.0. Use ``photutils.asdf.extension`` instead.
"""

import warnings

from photutils.utils.exceptions import PhotutilsDeprecationWarning

# The names are resolved on access by __getattr__ below
_MOVED_NAMES = (
    'PHOTUTILS_APERTURE_CONVERTERS',
    'PHOTUTILS_CONVERTERS',
    'PHOTUTILS_MANIFEST_URIS',
    'PHOTUTILS_PSF_CONVERTERS',
    'get_extensions',
    'get_resource_mappings',
)

__all__ = list(_MOVED_NAMES)

_deprecation_msg = ('photutils.extension is deprecated (since version 3.2) '
                    'and will be removed in version 4.0. Use '
                    'photutils.asdf.extension instead.')


def __getattr__(name):
    if name in __all__:
        warnings.warn(_deprecation_msg, PhotutilsDeprecationWarning,
                      stacklevel=2)

        # The extension imports asdf, which is an optional dependency
        from photutils.asdf import extension

        return getattr(extension, name)

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)


def __dir__():
    return __all__
