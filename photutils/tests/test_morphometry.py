# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
from ..morphometry import gini
import numpy as np


def test_gini():
	"""
    Test Gini coefficient measurement
    """
	#test equally distributate case
	data = np.ones((100, 100))
	assert gini(data) == 0
	#test extremely concentrate case
	data = np.zeros((100, 100))
	data[50][50] = 1
	assert gini(data) == 1