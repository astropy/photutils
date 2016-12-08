# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Photutils is an Astropy affiliated package to provide tools for
detecting and performing photometry of astronomical sources.  It also
has tools for background estimation, PSF matching, centroiding, and
morphological measurements.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    from .aperture import *
    from .background import *
    from .centroids import *
    from .detection import *
    from .morphology import *
    from .psf import *
    from .segmentation import *
