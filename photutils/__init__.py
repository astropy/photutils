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

from .aperture import *  # noqa
from .background import *  # noqa
from .centroids import *  # noqa
from .detection import *  # noqa
from .morphology import *  # noqa
from .psf import *  # noqa
from .segmentation import *  # noqa


# Set the bibtex entry to the article referenced in CITATION.rst.
def _get_bibtex():
    import os
    citation_file = os.path.join(os.path.dirname(__file__), 'CITATION.rst')

    with open(citation_file, 'r') as citation:
        refs = citation.read().split('@software')[1:]
        if len(refs) == 0:
            return ''
        bibtexreference = "@software{0}".format(refs[0])
    return bibtexreference


__citation__ = __bibtex__ = _get_bibtex()

del _astropy_init, _get_bibtex  # noqa
