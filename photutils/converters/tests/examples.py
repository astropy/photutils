# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the photutils PSF converters.
All examples return a PSF instance and the list of parameters to test.
"""
from astropy import units as u

from photutils.psf import AiryDiskPSF

parameters = {
    'AiryDiskPSF': ['flux', 'x_0', 'y_0', 'radius', 'bbox_factor'],
}


def airy_disk_units():
    """Return an AiryDiskPSF with units."""
    return (AiryDiskPSF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                        radius=1 * u.arcsec, bbox_factor=2),
            parameters['AiryDiskPSF'])


def airy_disk():
    """Return an AiryDiskPSF without units."""
    return (AiryDiskPSF(flux=2, x_0=1, y_0=1, radius=2, bbox_factor=3),
            parameters['AiryDiskPSF'])
