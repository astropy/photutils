# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions that will ultimately be merged into `astropy.utils`

from .scale_img import *
from .array_utils import *
try:
    # not guaranteed to be available at setup time
    from .sampling import *
except ImportError:
    if not _ASTROPY_SETUP_:
        raise

__all__ = ['find_imgcuts', 'img_stats', 'rescale_img', 'scale_linear',
           'scale_sqrt', 'scale_power', 'scale_log', 'scale_asinh',
           'downsample', 'upsample', 'extract_array_2d', 'add_array_2d',
           'subpixel_indices', 'fix_prf_nan']
