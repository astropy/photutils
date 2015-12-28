# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
An Astropy affiliated package for photometry.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    from .aperture_core import *
    from .aperture_funcs import *
    from .background import *
    from .detection import *
    from .segmentation import *
    from .psf import *
