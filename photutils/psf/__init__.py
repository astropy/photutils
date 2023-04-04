# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This subpackage contains tools to perform point-spread-function (PSF)
photometry.
"""

from . import epsf, epsf_stars, groupstars, models, photometry, utils
from .epsf import *  # noqa: F401, F403
from .epsf_stars import *  # noqa: F401, F403
from .groupstars import *  # noqa: F401, F403
from .matching import *  # noqa: F401, F403
from .models import *  # noqa: F401, F403
from .photometry import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403

# exclude matching from this list to avoid sphinx warnings
__all__ = []
__all__.extend(epsf.__all__)
__all__.extend(epsf_stars.__all__)
__all__.extend(groupstars.__all__)
__all__.extend(models.__all__)
__all__.extend(photometry.__all__)
__all__.extend(utils.__all__)
