# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the load module.
"""

import pytest

from photutils.datasets import get_path, load


def test_get_path():
    with pytest.raises(ValueError):
        get_path('filename', location='invalid')


@pytest.mark.remote_data
def test_load_star_image():
    hdu = load.load_star_image()
    assert len(hdu.header) == 106
    assert hdu.data.shape == (1059, 1059)
