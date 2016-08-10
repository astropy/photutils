# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..windows import (HanningWindow, TukeyWindow, CosineBellWindow,
                       SplitCosineBellWindow, TopHatWindow)

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def test_create_matching_psf():
    win = HanningWindow()
    data = win((5, 5))
    ref = [0., 0.19715007, 0.5, 0.19715007, 0.]
    assert_allclose(data[1, :], ref)

