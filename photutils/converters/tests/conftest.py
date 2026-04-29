# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Fixtures for the photutils PSF converters.
"""
import pytest

from photutils.converters.tests import examples


@pytest.fixture
def airy_disk_units():
    return examples.airy_disk_units()


@pytest.fixture
def airy_disk():
    return examples.airy_disk()
