# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils aperture converters.
"""

import asdf
import pytest
from numpy.testing import assert_array_equal

from photutils.converters import _ASDF_ASTROPY_INSTALLED
from photutils.converters.tests import examples
from photutils.extension import PHOTUTILS_APERTURE_CONVERTERS


@pytest.fixture
def aperobj(request):
    """
    A pytest fixture that returns an aperture object and the
    list of parameters to test.
    """
    return request.getfixturevalue(request.param)


# Examples that take no rotation angle
APERTURE_EXAMPLE_NAMES = [
    'circular_aperture_single_pos',
    'circular_aperture_multi_pos',
    'circular_annulus_single_pos',
    'circular_annulus_single_pos_tuple',
    'circular_annulus_multi_pos',
    'polygon_aperture',
    'polygon_aperture_vertices',
    'sky_circular_annulus',
    'sky_circular_aperture',
    'sky_elliptical_annulus',
    'sky_elliptical_aperture',
    'sky_polygon_aperture',
    'sky_rectangular_aperture',
    'sky_rectangular_annulus',
]

# Examples that take a rotation angle
THETA_EXAMPLE_NAMES = [
    'elliptical_annulus',
    'elliptical_aperture',
    'rectangular_annulus',
    'rectangular_aperture',
]

aper_params = pytest.mark.parametrize('aperobj', APERTURE_EXAMPLE_NAMES,
                                      indirect=True)
aper_theta = pytest.mark.parametrize('aperobj', THETA_EXAMPLE_NAMES,
                                     indirect=True)


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@aper_theta
@pytest.mark.usefixtures('theta')
def test_aperture_converters_theta(tmp_path, aperobj):
    """
    Test that the aperture converters can round-trip an aperture
    rotated at an angle ``theta``.
    """
    _run(tmp_path, aperobj)


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@aper_params
def test_aperture_converters(tmp_path, aperobj):
    """
    Test that the aperture converters can round-trip an aperture object.
    """
    _run(tmp_path, aperobj)


def _run(tmp_path, aper):
    """
    Run the comparison test.
    """
    aperture, pars = aper
    with asdf.AsdfFile() as af:
        af['aper'] = aperture
        af.write_to(tmp_path / 'aperture.asdf')

        with asdf.open(tmp_path / 'aperture.asdf') as af:
            aperture2 = af['aper']
            for parameter in pars:
                assert_array_equal(getattr(aperture, parameter),
                                   getattr(aperture2, parameter))


@pytest.mark.parametrize('name', APERTURE_EXAMPLE_NAMES)
def test_aperture_fixture_matches_example(request, name):
    """
    Test that each fixture returns the example of the same name.

    The fixtures wrap the example functions, so a fixture wired to the
    wrong function would silently drop an aperture from the round-trip
    test.
    """
    expected, expected_params = getattr(examples, name)()
    aper, params = request.getfixturevalue(name)
    assert type(aper) is type(expected)
    assert params == expected_params


@pytest.mark.parametrize('name', THETA_EXAMPLE_NAMES)
def test_aperture_theta_fixture_matches_example(request, name, theta):
    """
    Test that each rotated-aperture fixture returns the example of the
    same name, using the rotation angle of the ``theta`` fixture.
    """
    expected, expected_params = getattr(examples, name)(theta)
    aper, params = request.getfixturevalue(name)
    assert type(aper) is type(expected)
    assert aper.theta == expected.theta
    assert params == expected_params


def test_aperture_examples_cover_all_converters():
    """
    Test that the round-trip test exercises every type handled by an
    aperture converter.
    """
    covered = {type(getattr(examples, name)()[0]).__name__
               for name in APERTURE_EXAMPLE_NAMES}
    covered |= {type(getattr(examples, name)(0.0)[0]).__name__
                for name in THETA_EXAMPLE_NAMES}
    handled = {name.rsplit('.', 1)[-1]
               for converter in PHOTUTILS_APERTURE_CONVERTERS
               for name in converter.types}
    assert covered == handled
