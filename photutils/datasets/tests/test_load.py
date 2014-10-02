# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest, remote_data
from .. import load, get_path


def test_get_path():
    with pytest.raises(ValueError):
        get_path('filename', location='invalid')


def test_load_fermi_image():
    hdu = load.load_fermi_image()
    assert len(hdu.header) == 81
    assert hdu.data.shape == (201, 401)


@remote_data
def test_load_star_image():
    hdu = load.load_star_image()
    assert len(hdu.header) == 104
    assert hdu.data.shape == (1059, 1059)
