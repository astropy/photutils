# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Photutils is an Astropy affiliated package to provide tools for
detecting and performing photometry of astronomical sources.  It also
has tools for background estimation, ePSF building, PSF matching,
centroiding, and morphological measurements.
"""

import warnings

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa
# ----------------------------------------------------------------------------

from .aperture import *  # noqa
from .background import *  # noqa
from .detection import *  # noqa
from .psf import *  # noqa
from .segmentation import *  # noqa

# deprecations
from . import centroids
from . import morphology

__depr__ = {}
__depr__[centroids] = ('centroid_com', 'centroid_quadratic',
                       'centroid_sources', 'centroid_epsf',
                       'centroid_1dg', 'gaussian1d_moments', 'centroid_2dg')
__depr__[morphology] = ('data_properties', 'gini')

__depr_mesg__ = ('`photutils.{attr}` is a deprecated alias for '
                 '`{module}.{attr}`. Instead, please use `from {module} '
                 'import {attr}` to silence this warning.')

__depr_attrs__ = {}
for k, vals in __depr__.items():
    for val in vals:
        __depr_attrs__[val] = (getattr(k, val),
                               __depr_mesg__.format(module=k.__name__,
                                                    attr=val))
del k, val, vals


def __getattr__(attr):
    if attr in __depr_attrs__:
        obj, message = __depr_attrs__[attr]
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return obj
    raise AttributeError('module {!r} has no attribute {!r}'
                         .format(__name__, attr))


# Set the bibtex entry to the article referenced in CITATION.rst.
def _get_bibtex():
    import os
    citation_file = os.path.join(os.path.dirname(__file__), 'CITATION.rst')

    with open(citation_file, 'r') as citation:
        refs = citation.read().split('@software')[1:]
        if len(refs) == 0:
            return ''
        bibtexreference = f"@software{refs[0]}"
    return bibtexreference


__citation__ = __bibtex__ = _get_bibtex()

del _astropy_init, _get_bibtex  # noqa
