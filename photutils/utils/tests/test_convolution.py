# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the convolution module.
"""

import astropy.units as u
import pytest
from astropy.convolution import Gaussian2DKernel
from numpy.testing import assert_allclose

from photutils.datasets import make_100gaussians_image
from photutils.utils._convolution import _filter_data
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestFilterData:
    def setup_class(self):
        self.data = make_100gaussians_image()
        self.kernel = Gaussian2DKernel(3.0, x_size=3, y_size=3)

    def test_filter_data(self):
        filt_data1 = _filter_data(self.data, self.kernel)
        filt_data2 = _filter_data(self.data, self.kernel.array)
        assert_allclose(filt_data1, filt_data2)

    def test_filter_data_units(self):
        unit = u.electron
        filt_data = _filter_data(self.data * unit, self.kernel)
        assert isinstance(filt_data, u.Quantity)
        assert filt_data.unit == unit

    def test_filter_data_types(self):
        """
        Test to ensure output is a float array for integer input data.
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
        Test for kernel=None.
        """
        kernel = None
        filt_data = _filter_data(self.data, kernel)
        assert_allclose(filt_data, self.data)
