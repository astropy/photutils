# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the wcs module.
"""

import pytest
from numpy.testing import assert_allclose

from photutils.datasets import make_gwcs, make_wcs
from photutils.utils._optional_deps import HAS_GWCS


def test_make_wcs():
    """
    Test the basic functionality of make_wcs.
    """
    shape = (100, 200)
    wcs = make_wcs(shape)

    assert wcs.pixel_shape == (200, 100)  # (nx, ny)
    assert wcs.array_shape == shape
    assert wcs.wcs.radesys == 'ICRS'

    wcs = make_wcs(shape, galactic=True)
    assert wcs.wcs.ctype[0] == 'GLON-TAN'
    assert wcs.wcs.ctype[1] == 'GLAT-TAN'


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
def test_make_gwcs():
    """
    Test the basic functionality of make_gwcs.
    """
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
@pytest.mark.parametrize('galactic', [False, True])
def test_make_wcs_compare(galactic):
    """
    Test that make_wcs and make_gwcs return equivalent WCS
    transformations.

    The image is large so that the corners are far enough from the
    reference pixel to distinguish between projections.
    """
    shape = (4000, 6000)
    wcs = make_wcs(shape, galactic=galactic)
    gwcs_obj = make_gwcs(shape, galactic=galactic)
    xpos = (0, 5999, 0, 5999, 3000)
    ypos = (0, 0, 3999, 3999, 2000)
    sc1 = wcs.pixel_to_world(xpos, ypos)
    sc2 = gwcs_obj.pixel_to_world(xpos, ypos)

    assert_allclose(sc1.separation(sc2).arcsec, 0, atol=1e-6)
