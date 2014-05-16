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
from .aperture import *
from .detection.core import *
from .detection.findstars import *

# TODO: discuss if this should be imported into the top-level namespace:
# from .detection.morphology import *
# from .utils.scale_img import *
