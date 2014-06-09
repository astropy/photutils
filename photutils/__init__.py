# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Astropy affiliated package for image photometry utilities.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

from .aperture import *
from .psf import *
from .detection.core import *
from .detection.findstars import *
