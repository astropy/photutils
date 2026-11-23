# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Subpackage containing tools for fitting elliptical isophotes to galaxy
images.
"""

from .ellipse import *  # noqa: F403
from .fitter import *  # noqa: F403
from .geometry import *  # noqa: F403
from .harmonics import *  # noqa: F403
from .integrator import *  # noqa: F403
from .isophote import *  # noqa: F403
from .model import *  # noqa: F403
from .sample import *  # noqa: F403
from .integrator import _get_deprecated_constant


def __getattr__(name):
    # The integration mode constants were deprecated in version 3.2.
    # They are not in the ``__all__`` of the integrator module, so the
    # wildcard import above does not provide them.
    return _get_deprecated_constant(name, __name__)
