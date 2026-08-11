# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils aperture converters.
"""

import asdf
import pytest
from astropy import units as u
from astropy.coordinates import Angle
from numpy.testing import assert_array_equal

from photutils.converters import _ASDF_ASTROPY_INSTALLED
from photutils.converters.tests import examples
from photutils.extension import PHOTUTILS_APERTURE_CONVERTERS

# Examples that take no rotation angle
APERTURE_EXAMPLES = [
    examples.circular_aperture_single_pos,
    examples.circular_aperture_multi_pos,
    examples.circular_annulus_single_pos,
    examples.circular_annulus_single_pos_tuple,
    examples.circular_annulus_multi_pos,
    examples.polygon_aperture,
    examples.polygon_aperture_vertices,
    examples.sky_circular_annulus,
    examples.sky_circular_aperture,
    examples.sky_elliptical_annulus,
    examples.sky_elliptical_aperture,
    examples.sky_polygon_aperture,
    examples.sky_rectangular_aperture,
    examples.sky_rectangular_annulus,
]

# Examples that take a rotation angle
THETA_EXAMPLES = [
    examples.elliptical_annulus,
    examples.elliptical_aperture,
    examples.rectangular_annulus,
    examples.rectangular_aperture,
]

# Rotation angles as a float, a Quantity, and an Angle
THETAS = [0.23, 0.5 * u.deg, Angle(80, 'deg')]


def _example_id(example):
    """
    Return the test ID of an example function.
    """
    return example.__name__


def _check_roundtrip(tmp_path, aperture, params):
    """
    Test that an aperture is unchanged by a write and read.
    """
    filename = tmp_path / 'aperture.asdf'
    with asdf.AsdfFile() as af:
        af['aper'] = aperture
        af.write_to(filename)

    with asdf.open(filename) as af:
        aperture2 = af['aper']
        for param in params:
            assert_array_equal(getattr(aperture, param),
                               getattr(aperture2, param))


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('example', APERTURE_EXAMPLES, ids=_example_id)
def test_aperture_converters(tmp_path, example):
    """
    Test that the aperture converters can round-trip an aperture object.
    """
    _check_roundtrip(tmp_path, *example())


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('theta', THETAS, ids=['float', 'quantity', 'angle'])
@pytest.mark.parametrize('example', THETA_EXAMPLES, ids=_example_id)
def test_aperture_converters_theta(tmp_path, example, theta):
    """
    Test that the aperture converters can round-trip an aperture
    rotated at an angle ``theta``.
    """
    _check_roundtrip(tmp_path, *example(theta))


def test_aperture_examples_cover_all_converters():
    """
    Test that the round-trip tests exercise every type handled by an
    aperture converter.
    """
    covered = {type(example()[0]).__name__ for example in APERTURE_EXAMPLES}
    covered |= {type(example(0.0)[0]).__name__ for example in THETA_EXAMPLES}
    handled = {name.rsplit('.', 1)[-1]
               for converter in PHOTUTILS_APERTURE_CONVERTERS
               for name in converter.types}
    assert covered == handled
