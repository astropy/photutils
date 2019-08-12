# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Photutils is an Astropy affiliated package to provide tools for
detecting and performing photometry of astronomical sources.  It also
has tools for background estimation, ePSF building, PSF matching,
centroiding, and morphological measurements.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:  # noqa
    from .aperture import *  # noqa
    from .background import *  # noqa
    from .centroids import *  # noqa
    from .detection import *  # noqa
    from .morphology import *  # noqa
    from .psf import *  # noqa
    from .segmentation import *  # noqa
