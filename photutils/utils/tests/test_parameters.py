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


class TestAsPairBasic:
    """
    Tests for as_pair scalar/tuple broadcasting and basic validation.
    """

    def test_scalar_broadcast(self):
        assert_equal(as_pair('p', 4), (4, 4))

    def test_tuple_passthrough(self):
        assert_equal(as_pair('p', (3, 4)), (3, 4))

    def test_scalar_zero(self):
        assert_equal(as_pair('p', 0), (0, 0))

    def test_too_many_elements(self):
        match = 'must have 1 or 2 elements'
        with pytest.raises(ValueError, match=match):
            as_pair('p', (1, 2, 3))

    def test_2d_input(self):
        match = 'must be 1D'
        with pytest.raises(ValueError, match=match):
            as_pair('p', np.array([[1, 2]]))

    def test_non_integer_dtype(self):
        match = 'must have integer values'
        with pytest.raises(ValueError, match=match):
            as_pair('p', 1.5)

    @pytest.mark.parametrize('value', [(1, np.nan), (1, np.inf)])
    def test_non_finite(self, value):
        match = 'must be a finite value'
        with pytest.raises(ValueError, match=match):
            as_pair('p', value)


class TestAsPairCheckOdd:
    """
    Tests for the check_odd parameter.
    """

    def test_odd_tuple(self):
        assert_equal(as_pair('p', (3, 5), check_odd=True), (3, 5))

    def test_odd_scalar(self):
        assert_equal(as_pair('p', 3, check_odd=True), (3, 3))

    @pytest.mark.parametrize('value', [(3, 4), 4])
    def test_even_raises(self, value):
        match = 'must have an odd value for both axes'
        with pytest.raises(ValueError, match=match):
            as_pair('p', value, check_odd=True)


class TestAsPairLowerBound:
    """
    Tests for lower_bound validation.
    """

    def test_exclusive_lower_bound(self):
        match = 'must be > 0'
        with pytest.raises(ValueError, match=match):
            as_pair('p', 0, lower_bound=(0, 0))

    def test_inclusive_lower_bound(self):
        result = as_pair('p', 0, lower_bound=(0, 1))
        assert_equal(result, (0, 0))

    def test_inclusive_lower_bound_violation(self):
        match = r'must be >= 1'
        with pytest.raises(ValueError, match=match):
            as_pair('p', 0, lower_bound=(1, 1))

    def test_lower_bound_wrong_length(self):
        match = 'lower_bound must contain only 2 elements'
        with pytest.raises(ValueError, match=match):
            as_pair('p', 1, lower_bound=(0,))


class TestAsPairUpperBound:
    """
    Tests for upper_bound validation and clamping.
    """

    def test_upper_bound_clipping(self):
        result = as_pair('p', (10, 20), upper_bound=(5, 15))
        assert_equal(result, (5, 15))

    def test_upper_bound_no_clipping(self):
        result = as_pair('p', (3, 4), upper_bound=(10, 10))
        assert_equal(result, (3, 4))

    def test_upper_bound_wrong_length(self):
        match = 'upper_bound must contain only 2 elements'
        with pytest.raises(ValueError, match=match):
            as_pair('p', (3, 4), upper_bound=(5,))


class TestSigmaClipSentinelDefault:
    """
    Tests for SigmaClipSentinelDefault.
    """

    def test_repr(self):
        sentinel = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)
        result = repr(sentinel)
        assert 'SigmaClip' in result
        assert '3.0' in result
        assert '10' in result

    def test_custom_params(self):
        sentinel = SigmaClipSentinelDefault(sigma=2.0, maxiters=5)
        assert sentinel.sigma == 2.0
        assert sentinel.maxiters == 5


class TestCreateDefaultSigmaClip:
    """
    Tests for create_default_sigmaclip.
    """

    def test_defaults(self):
        sc = create_default_sigmaclip()
        assert isinstance(sc, SigmaClip)
        assert sc.sigma == 3.0
        assert sc.maxiters == 10

    def test_custom(self):
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
