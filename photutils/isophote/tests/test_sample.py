# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .make_test_data import make_test_image
from ..integrator import MEDIAN, MEAN, BI_LINEAR, NEAREST_NEIGHBOR
from ..isophote import Isophote
from ..sample import Sample


DATA = make_test_image(background=100., i0=0., noise=10., random_state=123)


def _doit(integrmode, amin, amax):
    sample = Sample(DATA, 50., astep=0.2, integrmode=integrmode)
    sample.update()
    iso = Isophote(sample, 0, True, 0)

    assert iso.pix_stddev < amax
    assert iso.pix_stddev > amin


def test_scatter():
    '''
    Checks that the pixel standard deviation can be reliably estimated
    from the rms scatter and the sector area.

    The test data is just a flat image with noise. No galaxy. We define
    the noise rms and then compare how close the pixel std dev estimated
    at extraction matches this input noise.
    '''
    _doit(NEAREST_NEIGHBOR, 7., 15.)
    _doit(BI_LINEAR,        7., 15.)
    _doit(MEAN,             7., 15.)
    _doit(MEDIAN,           6., 15.) # the median is not so good at estimating rms


def test_coordinates():
    sample = Sample(DATA, 50.)
    sample.update()

    x, y = sample.coordinates()

    array = np.array([])
    assert type(x) == type(array)
    assert type(y) == type(array)


def test_sclip():
    sample = Sample(DATA, 50., nclip=3)
    sample.update()

    x, y = sample.coordinates()

    array = np.array([])
    assert type(x) == type(array)
    assert type(y) == type(array)
