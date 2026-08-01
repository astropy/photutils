# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils aperture converters.
"""

import asdf
import pytest
from numpy.testing import assert_array_equal

from photutils.converters import _ASDF_ASTROPY_INSTALLED


@pytest.fixture
def aperobj(request):
    """
    A pytest fixture that returns an aperture object and the
    list of parameters to test.
    """
    return request.getfixturevalue(request.param)


aper_params = pytest.mark.parametrize('aperobj', [
    'circular_aperture_single_pos',
    'circular_aperture_multi_pos',
    'circular_annulus_single_pos',
    'circular_annulus_single_pos_tuple',
    'circular_annulus_multi_pos',
    'elliptical_aperture',
    'elliptical_annulus',
    'sky_circular_annulus',
    'sky_circular_aperture',
    'sky_elliptical_annulus',
    'polygon_aperture',
    'polygon_aperture_vertices',
    'sky_polygon_aperture',
    'rectangular_aperture',
    'rectangular_annulus',
    'sky_rectangular_aperture',
    'sky_rectangular_annulus',
], indirect=True)


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@aper_params
def test_aperture_converters(tmp_path, aperobj):
    """
    Test that the aperture converters can round-trip an aperture object.
    """
    aperture, pars = aperobj

    with asdf.AsdfFile() as af:
        af['aper'] = aperture
        af.write_to(tmp_path / 'aperture.asdf')

        with asdf.open(tmp_path / 'aperture.asdf') as af:
            aperture2 = af['aper']
            for parameter in pars:
                assert_array_equal(getattr(aperture, parameter),
                                   getattr(aperture2, parameter))
