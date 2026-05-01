# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Fixtures for the photutils PSF converters.

Each fixture returns a tuple of ``(psf_instance, parameter_names)``,
where ``parameter_names`` is the list of attributes to compare in
round-trip tests.
"""

import pytest
from astropy import units as u

from photutils.psf import AiryDiskPSF

AIRY_DISK_PARAMETERS = ['flux', 'x_0', 'y_0', 'radius', 'bbox_factor']


@pytest.fixture(params=['no_units', 'units'], ids=['no_units', 'units'])
def airy_disk_psf(request):
    """
    Return an AiryDiskPSF instance and its parameter names.

    Parametrized to cover both unit-bearing and unit-less inputs.
    """
    if request.param == 'units':
        psf = AiryDiskPSF(flux=1 * u.Jy, x_0=0 * u.arcsec,
                          y_0=0 * u.arcsec, radius=1 * u.arcsec,
                          bbox_factor=2)
    else:
        psf = AiryDiskPSF(flux=2, x_0=1, y_0=1, radius=2, bbox_factor=3)

    return psf, AIRY_DISK_PARAMETERS
