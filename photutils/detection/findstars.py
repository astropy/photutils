# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Temporary module to hold deprecations."""

import warnings

from astropy.utils.decorators import AstropyDeprecationWarning

from .base import StarFinderBase as _StarFinderBase  # noqa
from .daofinder import DAOStarFinder as _DAOStarFinder  # noqa
from .irafstarfinder import IRAFStarFinder as _IRAFStarFinder  # noqa

from .daofinder import _DAOFindProperties as __DAOFindProperties  # noqa
from .irafstarfinder import _IRAFStarFindProperties as \
    __IRAFStarFindProperties  # noqa
from ._utils import _StarFinderKernel as __StarFinderKernel  # noqa
from ._utils import _StarCutout as __StarCutout  # noqa
from ._utils import _find_stars as __find_stars  # noqa


deprecated = {'StarFinderBase': 'photutils.detection.base',
              'DAOStarFinder': 'photutils.detection.daofinder',
              'IRAFStarFinder': 'photutils.detection.irafstarfinder',
              '_DAOFindProperties': 'photutils.detection.daofinder',
              '_IRAFStarFindProperties': 'photutils.detection.irafstarfinder',
              '_StarFinderKernel': 'photutils.detection._utils',
              '_StarCutout': 'photutils.detection._utils',
              '_find_stars': 'photutils.detection._utils',
              }


def __getattr__(name):
    if name in deprecated.keys():
        warnings.warn(f'{name} was moved to the {deprecated[name]} module. '
                      'Please update your import statement.',
                      AstropyDeprecationWarning)
        return globals()[f'_{name}']
    raise AttributeError(f'module {__name__} has no attribute {name}')
