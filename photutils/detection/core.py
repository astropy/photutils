# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Temporary module to hold deprecations."""

import warnings

from astropy.utils.decorators import AstropyDeprecationWarning

from .peakfinder import find_peaks as _find_peaks  # noqa
from ..segmentation import detect_threshold as _detect_threshold  # noqa


deprecated = {'detect_threshold': 'photutils.segmentation.detect',
              'find_peaks': 'photutils.detection.peakfinder',
              }


def __getattr__(name):
    if name in deprecated.keys():
        warnings.warn(f'{name} was moved to the {deprecated[name]} module. '
                      'Please update your import statement.',
                      AstropyDeprecationWarning)
        return globals()[f'_{name}']
    raise AttributeError(f'module {__name__} has no attribute {name}')
