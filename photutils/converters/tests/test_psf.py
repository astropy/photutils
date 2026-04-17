# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the photutils PSF converters.
"""
import asdf
from astropy import units as u

from photutils.psf import (
    AiryDiskPSF,
    CircularGaussianPRF,
    CircularGaussianPSF,
    CircularGaussianSigmaPRF,
    GaussianPRF,
    GaussianPSF,
    MoffatPSF,
)


psfs = {
    'AiryDiskPSF': [
        AiryDiskPSF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                    radius=1 * u.arcsec, bbox_factor=2),
        AiryDiskPSF(flux=2, x_0=1, y_0=1, radius=2, bbox_factor=3),
    ],
    'CircularGaussianPRF': [
        CircularGaussianPRF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                            fwhm=1 * u.arcsec, bbox_factor=2),
        CircularGaussianPRF(flux=2, x_0=1, y_0=1, fwhm=2, bbox_factor=3),
    ],
    'CircularGaussianPSF': [
        CircularGaussianPSF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                            fwhm=1 * u.arcsec, bbox_factor=2),
        CircularGaussianPSF(flux=2, x_0=1, y_0=1, fwhm=2, bbox_factor=3),
    ],
    'CircularGaussianSigmaPRF': [
        CircularGaussianSigmaPRF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                            sigma=1 * u.arcsec, bbox_factor=2),
        CircularGaussianSigmaPRF(flux=2, x_0=1, y_0=1, sigma=2, bbox_factor=3),
    ],
    'GaussianPRF': [
        GaussianPRF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                            x_fwhm=1 * u.arcsec, y_fwhm=1 * u.arcsec, theta=0 * u.deg, bbox_factor=2),
        GaussianPRF(flux=2, x_0=1, y_0=1, x_fwhm=2, y_fwhm=2, theta=0, bbox_factor=3),
    ],
    'GaussianPSF': [
        GaussianPSF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                            x_fwhm=1 * u.arcsec, y_fwhm=1 * u.arcsec, theta=0 * u.deg, bbox_factor=2),
        GaussianPSF(flux=2, x_0=1, y_0=1, x_fwhm=2, y_fwhm=2, theta=0, bbox_factor=3),
    ],
    'MoffatPSF': [
        MoffatPSF(flux=71.4, x_0=24.3, y_0=25.2, alpha=5.1, beta=3.2),
        MoffatPSF(flux=71.4 * u.Jy, x_0=24.3 * u.pix, y_0=25.2 * u.pix,
                  alpha=5.1 * u.pix, beta=3.2),
      ]
}


parameters = {
    'AiryDiskPSF': ['flux', 'x_0', 'y_0', 'radius', 'bbox_factor'],
    'CircularGaussianPRF': ['flux', 'x_0', 'y_0', 'fwhm', 'bbox_factor'],
    'CircularGaussianPSF': ['flux', 'x_0', 'y_0', 'fwhm', 'bbox_factor'],
    'CircularGaussianSigmaPRF': ['flux', 'x_0', 'y_0', 'sigma', 'bbox_factor'],
    'GaussianPRF': ['flux', 'x_0', 'y_0', 'x_fwhm', 'y_fwhm', 'theta', 'bbox_factor'],
    'GaussianPSF': ['flux', 'x_0', 'y_0', 'x_fwhm', 'y_fwhm', 'theta', 'bbox_factor'],
    'MoffatPSF': ['flux', 'x_0', 'y_0', 'alpha', 'beta', 'bbox_factor'],
}


def test_psf_converters(tmp_path):
    """
    Test that the PSF converters can round-trip a PSF object.
    """
    for psf, instances in psfs.items():
        for instance in instances:
            with asdf.AsdfFile() as af:
                af['psf'] = instance
                af.write_to(tmp_path / 'psf.asdf')
            
            with asdf.open(tmp_path / 'psf.asdf') as af:
                psf2 = af['psf']

            for parameter in parameters[psf]:
                assert getattr(instance, parameter) == getattr(psf2, parameter)
