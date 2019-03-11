# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy.convolution import Gaussian2DKernel
from astropy.tests.helper import catch_warnings
from astropy.utils.exceptions import AstropyUserWarning

from ..convolution import filter_data
from ...datasets import make_100gaussians_image

try:
    import scipy  # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestFilterData:
    def setup_class(self):
        self.data = make_100gaussians_image()
        self.kernel = Gaussian2DKernel(3., x_size=3, y_size=3)

    def test_filter_data(self):
        fdata1 = filter_data(self.data, self.kernel)
        fdata2 = filter_data(self.data, self.kernel.array)
        assert_allclose(fdata1, fdata2)

    def test_filter_data_types(self):
        """
        Test to ensure output is a float array for integer input data.
        """

        fdata = filter_data(self.data.astype(int),
                            self.kernel.array.astype(int))
        assert fdata.dtype == np.float64

        fdata = filter_data(self.data.astype(int),
                            self.kernel.array.astype(float))
        assert fdata.dtype == np.float64

        fdata = filter_data(self.data.astype(float),
                            self.kernel.array.astype(int))
        assert fdata.dtype == np.float64

        fdata = filter_data(self.data.astype(float),
                            self.kernel.array.astype(float))
        assert fdata.dtype == np.float64

    def test_filter_data_kernel_none(self):
        """
        Test for kernel=None.
        """

        kernel = None
        fdata = filter_data(self.data, kernel)
        assert_allclose(fdata, self.data)

    def test_filter_data_check_normalization(self):
        """
        Test kernel normalization check.
        """

        with catch_warnings(AstropyUserWarning) as w:
            filter_data(self.data, self.kernel, check_normalization=True)
            assert len(w) == 1
