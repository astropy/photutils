# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the tools that make the simulated test images.
"""

import pytest
from astropy.io import fits
from numpy.testing import assert_array_equal

from photutils.isophote.tests.make_test_data import (make_fits_test_image,
                                                     make_test_image)


@pytest.mark.parametrize(('x0', 'y0'), [(30, 60), (None, 60), (30, None)])
def test_make_test_image_center(x0, y0):
    """
    Test that the image has no spike at its center pixel, which is
    replaced with an interpolated value, for an image that is not
    square and a center off its diagonal.
    """
    nx = 120
    ny = 90
    data = make_test_image(nx=nx, ny=ny, x0=x0, y0=y0, seed=0)
    assert data.shape == (ny, nx)
    xcen = nx // 2 if x0 is None else x0
    ycen = ny // 2 if y0 is None else y0
    neighbors = [data[ycen, xcen - 1], data[ycen, xcen + 1],
                 data[ycen - 1, xcen], data[ycen + 1, xcen]]
    assert data.max() == max(neighbors)
    assert min(neighbors) < data[ycen, xcen] < max(neighbors)


def test_make_fits_test_image(tmp_path):
    name = str(tmp_path / 'synth')
    kwargs = {'nx': 64, 'ny': 48, 'x0': 20, 'y0': 30, 'seed': 0}
    make_fits_test_image(name, **kwargs)
    with fits.open(f'{name}.fits') as hdulist:
        assert_array_equal(hdulist[0].data, make_test_image(**kwargs))
        assert hdulist[0].header['X0'] == 20
        assert hdulist[0].header['Y0'] == 30
