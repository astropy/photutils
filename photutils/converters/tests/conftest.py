# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the photutils PSF converters.
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
