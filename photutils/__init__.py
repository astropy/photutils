# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Photutils is an Astropy affiliated package to provide tools for
detecting and performing photometry of astronomical sources.

It also has tools for background estimation, ePSF building, PSF
matching, radial profiles, centroiding, and morphological measurements.
"""

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''


# Set the bibtex entry to the article referenced in CITATION.rst.
def _get_bibtex():
    import os
    citation_file = os.path.join(os.path.dirname(__file__), 'CITATION.rst')

    with open(citation_file) as citation:
        refs = citation.read().split('@software')[1:]
        if len(refs) == 0:
            return ''
        return f'@software{refs[0]}'


__citation__ = __bibtex__ = _get_bibtex()

del _get_bibtex
