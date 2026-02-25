# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the convolution module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import Gaussian2DKernel
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.datasets import make_100gaussians_image
from photutils.utils._convolution import _filter_data


class TestFilterData:
    def setup_class(self):
        self.data = make_100gaussians_image()
        self.kernel = Gaussian2DKernel(3.0, x_size=3, y_size=3)

    def test_filter_data(self):
        """
        Test _filter_data with Kernel2D and array kernels.
        """
        filt_data1 = _filter_data(self.data, self.kernel)
        filt_data2 = _filter_data(self.data, self.kernel.array)
        assert_allclose(filt_data1, filt_data2)

    def test_filter_data_units(self):
        """
        Test _filter_data with Quantity input data.
        """
        unit = u.electron
        filt_data = _filter_data(self.data * unit, self.kernel)
        assert isinstance(filt_data, u.Quantity)
        assert filt_data.unit == unit

    def test_filter_data_types(self):
        """
        Test that output is a float array for integer input data.
        """
        filt_data = _filter_data(self.data.astype(int),
                                 self.kernel.array.astype(int))
        assert filt_data.dtype == float

        filt_data = _filter_data(self.data.astype(int),
                                 self.kernel.array.astype(float))
        assert filt_data.dtype == float

        filt_data = _filter_data(self.data.astype(float),
                                 self.kernel.array.astype(int))
        assert filt_data.dtype == float

        filt_data = _filter_data(self.data.astype(float),
                                 self.kernel.array.astype(float))
        assert filt_data.dtype == float

    def test_filter_data_kernel_none(self):
        """
        Test _filter_data with kernel=None.
        """
        kernel = None
        filt_data = _filter_data(self.data, kernel)
        assert_allclose(filt_data, self.data)

    def test_filter_data_unnormalized_kernel(self):
        """
        Test that a warning is issued for an unnormalized kernel.
        """
        kernel = np.ones((3, 3))  # sums to 9, not normalized
        match = 'The kernel is not normalized'
        with pytest.warns(AstropyUserWarning, match=match):
            _filter_data(self.data, kernel, check_normalization=True)
