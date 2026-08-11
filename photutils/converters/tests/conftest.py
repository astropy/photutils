# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Fixtures for the photutils ASDF converters.

Object fixtures return a photutils object and parameter names for
round-trip comparisons. The ``theta`` fixture supplies rotation-angle
values for parametrized aperture fixtures.
"""

import pytest
from astropy import units as u
from astropy.coordinates import Angle

from photutils.converters.tests import examples


@pytest.fixture
def airy_disk_units():
    return examples.airy_disk_units()


@pytest.fixture
def airy_disk():
    return examples.airy_disk()


@pytest.fixture
def circular_gaussian_prf_units():
    return examples.circular_gaussian_prf_units()


@pytest.fixture
def circular_gaussian_prf():
    return examples.circular_gaussian_prf()


@pytest.fixture
def circular_gaussian_sigma_prf_units():
    return examples.circular_gaussian_sigma_prf_units()


@pytest.fixture
def circular_gaussian_sigma_prf():
    return examples.circular_gaussian_sigma_prf()


@pytest.fixture
def circular_gaussian_psf_units():
    return examples.circular_gaussian_psf_units()


@pytest.fixture
def circular_gaussian_psf():
    return examples.circular_gaussian_psf()


@pytest.fixture
def gaussian_prf_units():
    return examples.gaussian_prf_units()


@pytest.fixture
def gaussian_prf():
    return examples.gaussian_prf()


@pytest.fixture
def gaussian_psf_units():
    return examples.gaussian_psf_units()


@pytest.fixture
def gaussian_psf():
    return examples.gaussian_psf()


@pytest.fixture
def moffat_psf_units():
    return examples.moffat_psf_units()


@pytest.fixture
def moffat_psf():
    return examples.moffat_psf()


@pytest.fixture
def image_psf():
    return examples.image_psf()


@pytest.fixture
def gridded_psf():
    return examples.gridded_psf()


@pytest.fixture
def stdpsf_single_detector():
    return examples.stdpsf_single_detector()


@pytest.fixture
def circular_aperture_single_pos():
    return examples.circular_aperture_single_pos()


@pytest.fixture
def circular_aperture_multi_pos():
    return examples.circular_aperture_multi_pos()


@pytest.fixture
def circular_annulus_single_pos():
    return examples.circular_annulus_single_pos()


@pytest.fixture
def circular_annulus_single_pos_tuple():
    return examples.circular_annulus_single_pos_tuple()


@pytest.fixture
def circular_annulus_multi_pos():
    return examples.circular_annulus_multi_pos()


@pytest.fixture
def sky_circular_annulus():
    return examples.sky_circular_annulus()


@pytest.fixture
def sky_circular_aperture():
    return examples.sky_circular_aperture()


@pytest.fixture(params=[0.23, 0.5 * u.deg, Angle(80, 'deg')])
def theta(request):
    """
    Fixture that returns a rotation angle value.

    The fixture is parametrized with three different types of rotation
    angles: a float, a Quantity, and an Angle.
    """
    return request.param


@pytest.fixture
def elliptical_aperture(theta):
    return examples.elliptical_aperture(theta)


@pytest.fixture
def sky_elliptical_aperture():
    return examples.sky_elliptical_aperture()


@pytest.fixture
def elliptical_annulus(theta):
    return examples.elliptical_annulus(theta)


@pytest.fixture
def sky_elliptical_annulus():
    return examples.sky_elliptical_annulus()


@pytest.fixture
def polygon_aperture():
    return examples.polygon_aperture()


@pytest.fixture
def polygon_aperture_vertices():
    return examples.polygon_aperture_vertices()


@pytest.fixture
def sky_polygon_aperture():
    return examples.sky_polygon_aperture()


@pytest.fixture
def rectangular_aperture(theta):
    return examples.rectangular_aperture(theta)


@pytest.fixture
def sky_rectangular_aperture():
    return examples.sky_rectangular_aperture()


@pytest.fixture
def rectangular_annulus(theta):
    return examples.rectangular_annulus(theta)


@pytest.fixture
def sky_rectangular_annulus():
    return examples.sky_rectangular_annulus()
