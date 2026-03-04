# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the parameters module.
"""

import warnings

import numpy as np
import pytest
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_equal

from photutils.utils._parameters import (SigmaClipSentinelDefault, as_pair,
                                         create_default_sigmaclip,
                                         warn_positional_kwargs)


def test_as_pair():
    """
    Test as_pair with various inputs and validation options.
    """
    assert_equal(as_pair('myparam', 4), (4, 4))

    assert_equal(as_pair('myparam', (3, 4)), (3, 4))

    assert_equal(as_pair('myparam', 0), (0, 0))

    match = 'must be > 0'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', 0, lower_bound=(0, 0))

    match = 'must be a finite value'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', (1, np.nan))

    with pytest.raises(ValueError, match=match):
        as_pair('myparam', (1, np.inf))

    # Test check_odd=True success cases
    assert_equal(as_pair('myparam', (3, 5), check_odd=True), (3, 5))
    assert_equal(as_pair('myparam', 3, check_odd=True), (3, 3))

    match = 'must have an odd value for both axes'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', (3, 4), check_odd=True)

    with pytest.raises(ValueError, match=match):
        as_pair('myparam', 4, check_odd=True)

    # Test len(value) != 2 (e.g., 3 elements)
    match = 'must have 1 or 2 elements'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', (1, 2, 3))

    # Test value.ndim != 1 (e.g., 2D input with 2 elements)
    match = 'must be 1D'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', np.array([[1, 2]]))

    # Test non-integer dtype
    match = 'must have integer values'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', 1.5)

    # Test lower_bound with wrong length
    match = 'lower_bound must contain only 2 elements'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', 1, lower_bound=(0,))

    # Test upper_bound clipping
    result = as_pair('myparam', (10, 20), upper_bound=(5, 15))
    assert_equal(result, (5, 15))

    # Test inclusive lower_bound (>= bound)
    result = as_pair('myparam', 0, lower_bound=(0, 1))
    assert_equal(result, (0, 0))


def test_sigmaclip_sentinel_repr():
    """
    Test SigmaClipSentinelDefault __repr__ output.
    """
    sentinel = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)
    result = repr(sentinel)
    assert 'SigmaClip' in result
    assert '3.0' in result
    assert '10' in result


def test_create_default_sigmaclip():
    """
    Test that create_default_sigmaclip returns a SigmaClip with the
    expected default parameters.
    """
    sc = create_default_sigmaclip()
    assert isinstance(sc, SigmaClip)
    assert sc.sigma == 3.0
    assert sc.maxiters == 10


def test_create_default_sigmaclip_custom():
    """
    Test that create_default_sigmaclip respects custom parameters.
    """
    sc = create_default_sigmaclip(sigma=2.5, maxiters=5)
    assert isinstance(sc, SigmaClip)
    assert sc.sigma == 2.5
    assert sc.maxiters == 5


@warn_positional_kwargs(1, '1.0', '2.0')
def _example_func(a, b=10, c=20):
    """
    Example function for testing warn_positional_kwargs.
    """
    return a + b + c


class TestWarnPositionalKwargs:
    """
    Tests for the warn_positional_kwargs decorator.
    """

    def test_no_warning_at_limit(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = _example_func(1)
        assert result == 31

    def test_no_warning_keyword_only(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = _example_func(1, b=5, c=3)
        assert result == 9

    def test_warns_when_exceeded(self):
        match = '_example_func'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            result = _example_func(1, 2)
        assert result == 23

    def test_warning_message_versions(self):
        with pytest.warns(AstropyDeprecationWarning) as record:
            _example_func(1, 2, 3)
        msg = str(record[0].message)
        assert '1.0' in msg
        assert '2.0' in msg

    def test_return_value_preserved(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert _example_func(5, 3, 2) == 10
        assert _example_func(5) == 35

    def test_preserves_metadata(self):
        assert _example_func.__name__ == '_example_func'
        assert 'Example function' in _example_func.__doc__

    def test_zero_positional(self):
        @warn_positional_kwargs(0, '1.5', '2.5')
        def _no_pos(x=0):
            return x

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = _no_pos(x=42)
        assert result == 42

        with pytest.warns(AstropyDeprecationWarning):
            result = _no_pos(42)
        assert result == 42

    def test_negative_n_positional_raises(self):
        match = 'n_positional must be >= 0'
        with pytest.raises(ValueError, match=match):
            warn_positional_kwargs(-1, '1.0', '2.0')
