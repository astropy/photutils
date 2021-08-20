# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the convolution module.
"""

from astropy.convolution import Gaussian2DKernel
from astropy.tests.helper import catch_warnings
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose
import pytest

from .._convolution import _filter_data
from .._optional_deps import HAS_SCIPY  # noqa
from ...datasets import make_100gaussians_image


@pytest.mark.skipif('not HAS_SCIPY')
class TestFilterData:
    def setup_class(self):
        self.data = make_100gaussians_image()
        self.kernel = Gaussian2DKernel(3., x_size=3, y_size=3)

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

    def test_filter_data_check_normalization(self):
        """
        Test kernel normalization check.
        """

        with catch_warnings(AstropyUserWarning) as w:
            _filter_data(self.data, self.kernel, check_normalization=True)
            assert len(w) == 1
