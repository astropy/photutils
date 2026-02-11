# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Deprecated subpackage. Use ``photutils.psf_matching`` instead.
"""

import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

import photutils.psf_matching as _psf_matching
from photutils.psf_matching.fourier import __all__ as _fourier_all
from photutils.psf_matching.windows import __all__ as _windows_all

__all__ = list(_fourier_all) + list(_windows_all)

_deprecation_msg = ('photutils.psf.matching is deprecated (since version '
                    '3.0) and will be removed in a future version. Use '
                    'photutils.psf_matching instead. Please update your '
                    'imports accordingly.')


def __getattr__(name):
    if name in __all__:
        warnings.warn(_deprecation_msg, AstropyDeprecationWarning,
                      stacklevel=2)
        return getattr(_psf_matching, name)
    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)


def __dir__():
    return __all__
