# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ASDF converters.
"""

_ASDF_ASTROPY_INSTALLED = True

try:
    import asdf_astropy  # noqa: F401 -- needed to register the converters
except ImportError:
    _ASDF_ASTROPY_INSTALLED = False

if _ASDF_ASTROPY_INSTALLED:
    from .apertures import CircularApertureConverter
    from .functional_models import AiryDiskPSFConverter

__all__ = [
    'AiryDiskPSFConverter',
    'CircularApertureConverter',
]
