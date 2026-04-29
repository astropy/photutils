# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the photutils PSF converters.
All examples return a PSF instance and the list of parameters to test.
"""
import numpy as np
from astropy import units as u
from astropy.nddata import NDData

from photutils.psf import (AiryDiskPSF, CircularGaussianPRF,
                           CircularGaussianPSF, GaussianPRF, GaussianPSF,
                           GriddedPSFModel, ImagePSF, MoffatPSF)

parameters = {
    'AiryDiskPSF': ['flux', 'x_0', 'y_0', 'radius', 'bbox_factor'],
    'CircularGaussianPRF': ['flux', 'x_0', 'y_0', 'fwhm', 'bbox_factor'],
    'CircularGaussianPSF': ['flux', 'x_0', 'y_0', 'fwhm', 'bbox_factor'],
    'CircularGaussianSigmaPRF': ['flux', 'x_0', 'y_0', 'sigma',
                                 'bbox_factor'],
    'GaussianPRF': ['flux', 'x_0', 'y_0', 'x_fwhm', 'y_fwhm', 'theta',
                    'bbox_factor'],
    'GaussianPSF': ['flux', 'x_0', 'y_0', 'x_fwhm', 'y_fwhm', 'theta',
                    'bbox_factor'],
    'MoffatPSF': ['flux', 'x_0', 'y_0', 'alpha', 'beta', 'bbox_factor'],
    'ImagePSF': ['data', 'flux', 'x_0', 'y_0', 'oversampling',
                 'fill_value', 'origin'],
    'GriddedPSF': ['data', 'flux', 'x_0', 'y_0', 'oversampling',
                   'fill_value', 'grid_xypos'],
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


def circular_gaussian_prf_units():
    """Return a CircularGaussianPRF with units."""
    return (CircularGaussianPRF(flux=1 * u.Jy,
                                x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                                fwhm=1 * u.arcsec, bbox_factor=2),
            parameters['CircularGaussianPRF'])


def circular_gaussian_prf():
    """Return a CircularGaussianPRF without units."""
    return (CircularGaussianPRF(flux=2, x_0=1, y_0=1, fwhm=2, bbox_factor=3),
            parameters['CircularGaussianPRF'])


def circular_gaussian_psf_units():
    """""Return a CircularGaussianPSF with units."""
    return (CircularGaussianPSF(flux=1 * u.Jy,
                                x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                                fwhm=1 * u.arcsec, bbox_factor=2),
            parameters['CircularGaussianPSF'])


def circular_gaussian_psf():
    """Return a CircularGaussianPSF without units."""
    return (CircularGaussianPSF(flux=2, x_0=1, y_0=1, fwhm=2,
                                bbox_factor=3),
            parameters['CircularGaussianPSF'])


def gaussian_prf_units():
    """Return a GaussianPRF with units."""
    return (GaussianPRF(flux=1 * u.Jy,
                        x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                        x_fwhm=1 * u.arcsec, y_fwhm=1 * u.arcsec,
                        theta=0 * u.deg, bbox_factor=2),
            parameters['GaussianPRF'])


def gaussian_prf():
    """Return a GaussianPRF without units."""
    return (GaussianPRF(flux=2, x_0=1, y_0=1, x_fwhm=2, y_fwhm=2,
                        theta=0,
                        bbox_factor=3),
            parameters['GaussianPRF'])


def gaussian_psf_units():
    """Return a GaussianPSF with units."""
    return (GaussianPSF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                        x_fwhm=1 * u.arcsec, y_fwhm=1 * u.arcsec,
                        theta=0 * u.deg, bbox_factor=2),
            parameters['GaussianPSF'])


def gaussian_psf():
    """Return a GaussianPSF without units."""
    return (GaussianPSF(flux=2, x_0=1, y_0=1,
                        x_fwhm=2, y_fwhm=2,
                        theta=0, bbox_factor=3),
            parameters['GaussianPSF'])


def moffat_psf_units():
    """Return a MoffatPSF with units."""
    return (MoffatPSF(flux=71.4 * u.Jy,
                      x_0=24.3 * u.pix, y_0=25.2 * u.pix,
                      alpha=5.1 * u.pix, beta=3.2),
            parameters['MoffatPSF'])


def moffat_psf():
    """Return a MoffatPSF without units."""
    return (MoffatPSF(flux=71.4, x_0=24.3, y_0=25.2, alpha=5.1, beta=3.2),
            parameters['MoffatPSF'])


def image_psf():
    """Return an ImagePSF without units."""
    return (ImagePSF(data=np.arange(36).reshape((6, 6)),
                     flux=6,
                     x_0=0.5, y_0=0.5,
                     oversampling=1,
                     fill_value=0,
                     origin=(0, 0)),
            parameters['ImagePSF'])


def gridded_psf():
    """Return a GriddedPSFModel without units."""
    nd_data = NDData(data=np.arange(180).reshape((5, 6, 6)),
                     meta={'oversampling': 1,
                           'grid_xypos': [(2, 2),
                                          (2, 65),
                                          (65, 2),
                                          (65, 2),
                                          (65, 65)],
                           },
                     )
    return (GriddedPSFModel(nddata=nd_data, flux=6,
                            x_0=0.5, y_0=0.5,
                            fill_value=np.nan),
            parameters['GriddedPSF'])
