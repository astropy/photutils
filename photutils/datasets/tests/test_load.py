# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the load module.
"""

import pytest

from photutils.datasets import get_path, load


def test_get_path():
    fn = '4gaussians_params.ecsv'
    path = get_path(fn, location='local')
    assert fn in path

    match = 'Invalid location:'
    with pytest.raises(ValueError, match=match):
        get_path('filename', location='invalid')


@pytest.mark.remote_data
def test_load_star_image():
    hdu = load.load_star_image()
    assert len(hdu.header) == 106
    assert hdu.data.shape == (1059, 1059)
