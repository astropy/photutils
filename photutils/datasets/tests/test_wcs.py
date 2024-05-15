# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the wcs module.
"""

import pytest
from numpy.testing import assert_allclose

from photutils.datasets import make_gwcs, make_wcs
from photutils.utils._optional_deps import HAS_GWCS


def test_make_wcs():
    shape = (100, 200)
    wcs = make_wcs(shape)

    assert wcs.pixel_shape == shape
    assert wcs.wcs.radesys == 'ICRS'

    wcs = make_wcs(shape, galactic=True)
    assert wcs.wcs.ctype[0] == 'GLON-CAR'
    assert wcs.wcs.ctype[1] == 'GLAT-CAR'


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
def test_make_gwcs():
    shape = (100, 200)

    wcs = make_gwcs(shape)
    assert wcs.pixel_n_dim == 2
    assert wcs.available_frames == ['detector', 'icrs']
    assert wcs.output_frame.name == 'icrs'
    assert wcs.output_frame.axes_names == ('lon', 'lat')

    wcs = make_gwcs(shape, galactic=True)
    assert wcs.pixel_n_dim == 2
    assert wcs.available_frames == ['detector', 'galactic']
    assert wcs.output_frame.name == 'galactic'
    assert wcs.output_frame.axes_names == ('lon', 'lat')


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
def test_make_wcs_compare():
    shape = (200, 300)
    wcs = make_wcs(shape)
    gwcs_obj = make_gwcs(shape)
    sc1 = wcs.pixel_to_world((50, 75), (50, 100))
    sc2 = gwcs_obj.pixel_to_world((50, 75), (50, 100))

    assert_allclose(sc1.ra, sc2.ra)
    assert_allclose(sc1.dec, sc2.dec)
