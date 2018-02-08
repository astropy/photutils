# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
PSF-building tools.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


__all__ = ['PSFBuilder']


class PSFBuilder(object):

    def __init__(self, model, fitter=None, recenterer=None, max_iters=10):
        self.model = model
        self.fitter = fitter
        self.recenterer = recenterer
        self.max_iters = max_iters

    def __call__(self, data, stars_table):
        raise NotImplementedError
