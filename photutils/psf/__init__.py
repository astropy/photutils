# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This subpackage contains tools to perform point-spread-function (PSF)
photometry.
"""

from . import (epsf, epsf_stars, griddedpsfmodel, groupers, groupstars,
               models, photometry, photometry_depr, utils)
from .epsf import *  # noqa: F401, F403
from .epsf_stars import *  # noqa: F401, F403
from .griddedpsfmodel import *  # noqa: F401, F403
from .groupers import *  # noqa: F401, F403
from .groupstars import *  # noqa: F401, F403
from .matching import *  # noqa: F401, F403
from .models import *  # noqa: F401, F403
from .photometry import *  # noqa: F401, F403
from .photometry_depr import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403

# exclude matching from this list to avoid sphinx warnings
__all__ = []
__all__.extend(epsf.__all__)
__all__.extend(epsf_stars.__all__)
__all__.extend(griddedpsfmodel.__all__)
__all__.extend(groupers.__all__)
__all__.extend(groupstars.__all__)
__all__.extend(models.__all__)
__all__.extend(photometry.__all__)
__all__.extend(photometry_depr.__all__)
__all__.extend(utils.__all__)
