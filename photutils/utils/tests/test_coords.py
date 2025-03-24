# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _coords module.
"""

import pytest
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._coords import make_random_xycoords


@pytest.mark.parametrize('min_sep', [0.0, 5.7, 10.0, 17.4])
def test_make_random_xycoords(min_sep):
    xmin, xmax = 97, 903
    ymin, ymax = 0, 501
    ncoords = 100
    xycoords = make_random_xycoords(ncoords, (xmin, xmax), (ymin, ymax),
                                    min_separation=min_sep, seed=0)

    assert xycoords.shape == (ncoords, 2)
    assert xycoords[:, 0].min() >= xmin
    assert xycoords[:, 0].max() <= xmax
    assert xycoords[:, 1].min() >= ymin
    assert xycoords[:, 1].max() <= ymax

    # check that the minimum separation is met
    if min_sep > 0:
        dists2 = ((xycoords[:, None, 0] - xycoords[None, :, 0])**2
                  + (xycoords[:, None, 1] - xycoords[None, :, 1])**2)
        dists2 = dists2[dists2 > 0]
        assert dists2.min() >= min_sep**2


def test_make_random_xycoords_crowded():
    match = 'coordinates within the given shape and minimum separation'
    with pytest.warns(AstropyUserWarning, match=match):
        make_random_xycoords(50, (0, 50), (10, 30), min_separation=10, seed=0)
