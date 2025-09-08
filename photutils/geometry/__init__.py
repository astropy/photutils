# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Subpackage providing low-level geometry functions used by aperture
photometry to calculate the overlap of aperture shapes with a pixel
grid.

These functions are not intended to be used directly by users, but
are used by the higher-level `photutils.aperture` tools.
"""

from .circular_overlap import *  # noqa: F401, F403
from .elliptical_overlap import *  # noqa: F401, F403
from .rectangular_overlap import *  # noqa: F401, F403
