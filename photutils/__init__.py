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
from ._astropy_init import *  # noqa: F401, F403

from . import aperture
from . import background
from . import detection
from . import psf
from . import segmentation

# deprecations
__depr__ = {}

__depr__[aperture] = ('BoundingBox', 'CircularMaskMixin',
                      'CircularAperture', 'CircularAnnulus',
                      'SkyCircularAperture', 'SkyCircularAnnulus', 'Aperture',
                      'SkyAperture', 'PixelAperture', 'EllipticalMaskMixin',
                      'EllipticalAperture', 'EllipticalAnnulus',
                      'SkyEllipticalAperture', 'SkyEllipticalAnnulus',
                      'ApertureMask', 'aperture_photometry',
                      'RectangularMaskMixin', 'RectangularAperture',
                      'RectangularAnnulus', 'SkyRectangularAperture',
                      'SkyRectangularAnnulus', 'ApertureStats')

__depr__[background] = ('Background2D', 'BackgroundBase', 'BackgroundRMSBase',
                        'MeanBackground', 'MedianBackground',
                        'ModeEstimatorBackground', 'MMMBackground',
                        'SExtractorBackground', 'BiweightLocationBackground',
                        'StdBackgroundRMS', 'MADStdBackgroundRMS',
                        'BiweightScaleBackgroundRMS', 'BkgZoomInterpolator',
                        'BkgIDWInterpolator')

__depr__[detection] = ('StarFinderBase', 'DAOStarFinder', 'IRAFStarFinder',
                       'find_peaks', 'StarFinder')

__depr__[psf] = ('EPSFFitter', 'EPSFBuilder', 'EPSFStar', 'EPSFStars',
                 'LinkedEPSFStar', 'extract_stars', 'DAOGroup', 'DBSCANGroup',
                 'GroupStarsBase', 'NonNormalizable', 'FittableImageModel',
                 'EPSFModel', 'GriddedPSFModel', 'IntegratedGaussianPRF',
                 'PRFAdapter', 'BasicPSFPhotometry',
                 'IterativelySubtractedPSFPhotometry', 'DAOPhotPSFPhotometry',
                 'prepare_psf_model',
                 'get_grouped_psf_model', 'subtract_psf',
                 'resize_psf', 'create_matching_kernel',
                 'SplitCosineBellWindow', 'HanningWindow', 'TukeyWindow',
                 'CosineBellWindow', 'TopHatWindow')

__depr__[segmentation] = ('SourceCatalog', 'SegmentationImage', 'Segment',
                          'deblend_sources', 'detect_threshold',
                          'detect_sources', 'SourceFinder',
                          'make_2dgaussian_kernel')

__depr_mesg__ = ('`photutils.{attr}` is a deprecated alias for '
                 '`{module}.{attr}` and will be removed in the future. '
                 'Instead, please use `from {module} import {attr}` to '
                 'silence this warning.')

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
    raise AttributeError(f'module {__name__!r} has no attribute {attr!r}')


# Set the bibtex entry to the article referenced in CITATION.rst.
def _get_bibtex():
    import os
    citation_file = os.path.join(os.path.dirname(__file__), 'CITATION.rst')

    with open(citation_file) as citation:
        refs = citation.read().split('@software')[1:]
        if len(refs) == 0:
            return ''
        bibtexreference = f'@software{refs[0]}'
    return bibtexreference


__citation__ = __bibtex__ = _get_bibtex()

del _get_bibtex
