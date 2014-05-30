# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Make example datasets.

We could put more useful functions here ... e.g. tiny subsets of the functionality here: 
* http://www.astromatic.net/software/stuff
* http://www.astromatic.net/software/skymaker
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
#from astropy.modeling.models import Gaussian2D

__all__ = ['make_example_image']


def make_example_image(shape=(1000, 1000), n_sources=42, psf_width=3):
    """Make example image.
    
    Randomly distribute Gaussian sources.
    
    TODO: this is just an example to discuss the `photutils.datasets` sub-package.
    TODO: move the example code from ``photutils/detection/test/test_findstars.py`` here.

    """
    # TODO: implement me
    return np.random.random(shape)
