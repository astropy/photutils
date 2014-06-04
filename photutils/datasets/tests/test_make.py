# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .. import make


def test_make_example_image():
    shape = (100, 100)
    image = make.make_example_image(shape)
    assert image.shape == shape
