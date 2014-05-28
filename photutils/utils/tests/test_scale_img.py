# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from .. import scale_img


def test_find_imgcuts():
    data = np.arange(101)

    mincut, maxcut = scale_img.find_imgcuts(data, min_cut=10)
    assert_equal([mincut, maxcut], [10, 100])

    mincut, maxcut = scale_img.find_imgcuts(data, max_cut=90)
    assert_equal([mincut, maxcut], [0, 90])

    mincut, maxcut = scale_img.find_imgcuts(data, min_cut=10, max_cut=90)
    assert_equal([mincut, maxcut], [10, 90])

    mincut, maxcut = scale_img.find_imgcuts(data, min_percent=20)
    assert_equal([mincut, maxcut], [20, 100])

    mincut, maxcut = scale_img.find_imgcuts(data, max_percent=80)
    assert_equal([mincut, maxcut], [0, 80])

    mincut, maxcut = scale_img.find_imgcuts(data, min_percent=20,
                                            max_percent=80)
    assert_equal([mincut, maxcut], [20, 80])

    mincut, maxcut = scale_img.find_imgcuts(data, percent=90)
    assert_equal([mincut, maxcut], [5, 95])

    mincut, maxcut = scale_img.find_imgcuts(data, min_percent=20,
                                            max_percent=80, percent=90)
    assert_equal([mincut, maxcut], [20, 80])

    mincut, maxcut = scale_img.find_imgcuts(data, min_cut=10, min_percent=20)
    assert_equal([mincut, maxcut], [10, 100])

    mincut, maxcut = scale_img.find_imgcuts(data, max_cut=90, max_percent=80)
    assert_equal([mincut, maxcut], [0, 90])
