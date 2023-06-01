# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the windows module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.psf.matching.windows import (CosineBellWindow, HanningWindow,
                                            SplitCosineBellWindow,
                                            TopHatWindow, TukeyWindow)
from photutils.utils._optional_deps import HAS_SCIPY


def test_hanning():
    win = HanningWindow()
    data = win((5, 5))
    ref = [0.0, 0.19715007, 0.5, 0.19715007, 0.0]
    assert_allclose(data[1, :], ref)


def test_hanning_numpy():
    """Test Hanning window against 1D numpy version."""

    size = 101
    cen = (size - 1) // 2
    shape = (size, size)
    win = HanningWindow()
    data = win(shape)
    ref1d = np.hanning(shape[0])
    assert_allclose(data[cen, :], ref1d)


def test_tukey():
    win = TukeyWindow(0.5)
    data = win((5, 5))
    ref = [0.0, 0.63312767, 1.0, 0.63312767, 0.0]
    assert_allclose(data[1, :], ref)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_tukey_scipy():
    """Test Tukey window against 1D scipy version."""

    from scipy.signal.windows import tukey

    size = 101
    cen = (size - 1) // 2
    shape = (size, size)
    alpha = 0.4
    win = TukeyWindow(alpha=alpha)
    data = win(shape)
    ref1d = tukey(shape[0], alpha=alpha)
    assert_allclose(data[cen, :], ref1d)


def test_cosine_bell():
    win = CosineBellWindow(alpha=0.8)
    data = win((7, 7))
    ref = [0.0, 0.0, 0.19715007, 0.5, 0.19715007, 0.0, 0.0]
    assert_allclose(data[2, :], ref)


def test_split_cosine_bell():
    win = SplitCosineBellWindow(alpha=0.8, beta=0.2)
    data = win((5, 5))
    ref = [0.0, 0.3454915, 1.0, 0.3454915, 0.0]
    assert_allclose(data[2, :], ref)


def test_tophat():
    win = TopHatWindow(beta=0.5)
    data = win((5, 5))
    ref = [0.0, 1.0, 1.0, 1.0, 0.0]
    assert_allclose(data[2, :], ref)


def test_invalid_shape():
    with pytest.raises(ValueError):
        win = HanningWindow()
        win((5,))
