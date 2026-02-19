# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the windows module.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose
from scipy.signal.windows import tukey

from photutils.psf_matching.windows import (CosineBellWindow, HanningWindow,
                                            SplitCosineBellWindow,
                                            TopHatWindow, TukeyWindow)


def test_hanning():
    """
    Test Hanning window with basic array values.
    """
    window = HanningWindow()
    data = window((5, 5))
    ref = [0.0, 0.19715007, 0.5, 0.19715007, 0.0]
    assert_allclose(data[1, :], ref)


def test_hanning_numpy():
    """
    Test Hanning window against 1D numpy version.
    """
    size = 101
    cen = (size - 1) // 2
    shape = (size, size)
    window = HanningWindow()
    data = window(shape)
    ref1d = np.hanning(shape[0])
    assert_allclose(data[cen, :], ref1d)


def test_tukey():
    """
    Test Tukey window with basic array values.
    """
    window = TukeyWindow(0.5)
    data = window((5, 5))
    ref = [0.0, 0.63312767, 1.0, 0.63312767, 0.0]
    assert_allclose(data[1, :], ref)


def test_tukey_scipy():
    """
    Test Tukey window against 1D scipy version.
    """
    size = 101
    cen = (size - 1) // 2
    shape = (size, size)
    alpha = 0.4
    window = TukeyWindow(alpha=alpha)
    data = window(shape)
    ref1d = tukey(shape[0], alpha=alpha)
    assert_allclose(data[cen, :], ref1d)


def test_cosine_bell():
    """
    Test cosine bell window with basic array values.
    """
    window = CosineBellWindow(alpha=0.8)
    data = window((7, 7))
    ref = [0.0, 0.011467736745367552, 0.36162762260757253,
           0.6294095225512605, 0.36162762260757253,
           0.011467736745367552, 0.0]
    assert_allclose(data[2, :], ref)


def test_split_cosine_bell():
    """
    Test split cosine bell window with basic array values.
    """
    window = SplitCosineBellWindow(alpha=0.8, beta=0.2)
    data = window((5, 5))
    ref = [0.0, 0.6913417161825449, 1.0, 0.6913417161825449, 0.0]
    assert_allclose(data[2, :], ref)


def test_split_cosine_bell_invalid_inputs():
    """
    Test that invalid alpha and beta values raise ValueError.
    """
    match = 'alpha must be between 0.0 and 1.0'
    with pytest.raises(ValueError, match=match):
        SplitCosineBellWindow(alpha=-0.1, beta=0.2)
    with pytest.raises(ValueError, match=match):
        SplitCosineBellWindow(alpha=1.1, beta=0.2)

    match = 'beta must be between 0.0 and 1.0'
    with pytest.raises(ValueError, match=match):
        SplitCosineBellWindow(alpha=0.8, beta=-0.1)
    with pytest.raises(ValueError, match=match):
        SplitCosineBellWindow(alpha=0.8, beta=1.1)


def test_split_cosine_bell_alpha_plus_beta_gt_one():
    """
    Test that alpha + beta > 1.0 warns about taper clipping.
    """
    match = 'alpha.*beta.*>.*1.0'
    with pytest.warns(AstropyUserWarning, match=match):
        SplitCosineBellWindow(alpha=0.8, beta=0.5)


def test_tophat():
    """
    Test top hat window with basic array values.
    """
    window = TopHatWindow(beta=0.5)
    data = window((5, 5))
    ref = [0.0, 1.0, 1.0, 1.0, 0.0]
    assert_allclose(data[2, :], ref)


def test_invalid_shape():
    """
    Test that invalid shape raises ValueError.
    """
    window = HanningWindow()
    match = 'shape must have only 2 elements'
    with pytest.raises(ValueError, match=match):
        window((5,))


def test_asymmetric_shape():
    """
    Test window with asymmetric shape.
    """
    shape = (51, 25)
    window = HanningWindow()
    data = window(shape)
    assert data.shape == shape
    assert_allclose(data[25, 12], 1.0)


def test_repr_and_str():
    """
    Test __repr__ and __str__ for all window classes.
    """
    window = SplitCosineBellWindow(alpha=0.4, beta=0.3)
    assert repr(window) == 'SplitCosineBellWindow(alpha=0.4, beta=0.3)'
    assert str(window) == repr(window)

    window = HanningWindow()
    assert repr(window) == 'HanningWindow()'
    assert str(window) == repr(window)

    window = TukeyWindow(alpha=0.5)
    assert repr(window) == 'TukeyWindow(alpha=0.5)'
    assert str(window) == repr(window)

    window = CosineBellWindow(alpha=0.3)
    assert repr(window) == 'CosineBellWindow(alpha=0.3)'
    assert str(window) == repr(window)

    window = TopHatWindow(beta=0.4)
    assert repr(window) == 'TopHatWindow(beta=0.4)'
    assert str(window) == repr(window)
