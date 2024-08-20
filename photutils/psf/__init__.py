# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This subpackage contains tools to perform point-spread-function (PSF)
photometry.
"""

from . import (epsf, epsf_stars, functional_models, gridded_models, groupers,
               image_models, photometry, utils)
from .epsf import *  # noqa: F401, F403
from .epsf_stars import *  # noqa: F401, F403
from .functional_models import *  # noqa: F401, F403
from .gridded_models import *  # noqa: F401, F403
from .groupers import *  # noqa: F401, F403
from .image_models import *  # noqa: F401, F403
from .matching import *  # noqa: F401, F403
from .photometry import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403

# exclude matching from this list to avoid sphinx warnings
__all__ = []
__all__ += epsf.__all__
__all__ += epsf_stars.__all__
__all__ += functional_models.__all__
__all__ += gridded_models.__all__
__all__ += groupers.__all__
__all__ += image_models.__all__
__all__ += photometry.__all__
__all__ += utils.__all__
