# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Make example datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling.models import Gaussian2D

__all__ = ['make_example_image']


def make_example_image(shape=(100, 100), psf_width=5):
    """Make example image.
    
    Randomly distribute Gaussian sources.
    
    TODO: this is just an example to discuss the `photutils.datasets` sub-package.
    TODO: move the example code from ``photutils/detection/test/test_findstars.py`` here.

    """
    y, x = np.mgrid[:shape[1], :shape[0]]
    model = Gaussian2D(amplitude=42, x_mean=30, y_mean=70,
                       x_stddev=psf_width, y_stddev=psf_width)
    image = model(x, y)

    return image
