# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the photutils PSF converters.
"""
import asdf
import pytest
from astropy import units as u

from photutils.converters import ASDF_ASTROPY_INSTALLED
from photutils.psf import AiryDiskPSF

psfs = [
    AiryDiskPSF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                radius=1 * u.arcsec, bbox_factor=2),
    AiryDiskPSF(flux=2, x_0=1, y_0=1, radius=2, bbox_factor=3),
]


@pytest.mark.skipif(not ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_psf_converters(tmp_path):
    """
    Test that the PSF converters can round-trip a PSF object.
    """
    for psf in psfs:
        with asdf.AsdfFile() as af:
            af['psf'] = psf
            af.write_to(tmp_path / 'psf.asdf')

        with asdf.open(tmp_path / 'psf.asdf') as af:
            psf2 = af['psf']

            assert psf.flux == psf2.flux
            assert psf.x_0 == psf2.x_0
            assert psf.y_0 == psf2.y_0
            assert psf.radius == psf2.radius
            assert psf.bbox_factor == psf2.bbox_factor
