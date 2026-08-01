# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Fixtures for the photutils PSF converters.

Each fixture returns a tuple of ``(psf_instance, parameter_names)``,
where ``parameter_names`` is the list of attributes to compare in
round-trip tests.
"""
import pytest

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
    return examples.circular_gaussian_prf_units()


@pytest.fixture
def circular_gaussian_sigma_prf():
    return examples.circular_gaussian_prf()


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


@pytest.fixture
def elliptical_aperture():
    return examples.elliptical_aperture()


@pytest.fixture
def elliptical_annulus():
    return examples.elliptical_annulus()


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
def rectangular_aperture():
    return examples.rectangular_aperture()


@pytest.fixture
def sky_rectangular_aperture():
    return examples.sky_rectangular_aperture()


@pytest.fixture
def rectangular_annulus():
    return examples.rectangular_annulus()


@pytest.fixture
def sky_rectangular_annulus():
    return examples.sky_rectangular_annulus()
