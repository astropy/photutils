# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ASDF converters for photutils objects.

``CircularApertureConverter`` requires only ``asdf``.  ``AiryDiskPSFConverter``
requires ``asdf-astropy`` for full functionality; it is always registered but
raises a clear ``ImportError`` when ``asdf-astropy`` is not installed.
"""

from .apertures import CircularApertureConverter  # noqa: F401
from .functional_models import AiryDiskPSFConverter  # noqa: F401

try:
    import asdf_astropy  # noqa: F401 -- only used to detect availability
except ImportError:
    _ASDF_ASTROPY_INSTALLED = False
else:
    _ASDF_ASTROPY_INSTALLED = True

__all__ = ['AiryDiskPSFConverter', 'CircularApertureConverter']
