# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for deprecation warnings when optional arguments are passed
positionally.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.morphology import data_properties, gini


class TestDataPropertiesPositionalKwargs:
    """
    Test data_properties warns for positional optional args.
    """

    def setup_method(self):
        self.data = np.random.default_rng(0).random((10, 10))

    def test_positional_warns(self):
        mask = np.zeros((10, 10), dtype=bool)
        match = 'data_properties'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            data_properties(self.data, mask)

    def test_keyword_no_warning(self):
        mask = np.zeros((10, 10), dtype=bool)
        data_properties(self.data, mask=mask)


class TestGiniPositionalKwargs:
    """
    Test gini warns for positional optional args.
    """

    def test_positional_warns(self):
        data = np.arange(100, dtype=float)
        mask = np.zeros(100, dtype=bool)
        match = 'gini'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            gini(data, mask)

    def test_keyword_no_warning(self):
        data = np.arange(100, dtype=float)
        mask = np.zeros(100, dtype=bool)
        gini(data, mask=mask)
