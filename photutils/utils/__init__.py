# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions that will ultimately be merged into `astropy.utils`

from .scale_img import *
try:
    # not guaranteed to be available at setup time
    from .sampling import *
except ImportError:
    if not _ASTROPY_SETUP_:
        raise
