# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Examples for testing the photutils ASDF converters.

Each example returns a tuple containing a photutils object and the list
of parameter names to compare in round-trip tests.
"""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.nddata import NDData

from photutils import aperture
from photutils.psf import (AiryDiskPSF, CircularGaussianPRF,
                           CircularGaussianPSF, CircularGaussianSigmaPRF,
                           GaussianPRF, GaussianPSF, GriddedPSFModel, ImagePSF,
                           MoffatPSF, STDPSFGrid)

# The directory holding the STDPSF test files. STDPSFGrid accepts only
# a string filename, so the paths are converted where they are used.
PSF_DATA_DIR = Path(__file__).resolve().parents[2] / 'psf' / 'tests' / 'data'

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
    'GriddedPSFModel': ['data', 'flux', 'x_0', 'y_0', 'oversampling',
                        'fill_value', 'grid_xypos'],
    'STDPSFGrid': ['data', 'grid_xypos', 'oversampling'],
    'CircularAperture': ['positions', 'r'],
    'CircularAnnulus': ['positions', 'r_in', 'r_out'],
    'EllipticalAperture': ['positions', 'a', 'b', 'theta'],
    'EllipticalAnnulus': ['positions', 'a_in', 'a_out',
                          'b_in', 'b_out', 'theta'],
    'PolygonAperture': ['positions', 'vertex_offsets'],
    'RectangularAperture': ['positions', 'w', 'h', 'theta'],
    'RectangularAnnulus': ['positions', 'w_in', 'w_out',
                           'h_in', 'h_out', 'theta'],
}


def airy_disk_units():
    """
    Return an AiryDiskPSF with units.
    """
    return (AiryDiskPSF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                        radius=1 * u.arcsec, bbox_factor=2),
            parameters['AiryDiskPSF'])


def airy_disk():
    """
    Return an AiryDiskPSF without units.
    """
    return (AiryDiskPSF(flux=2, x_0=1, y_0=1, radius=2, bbox_factor=3),
            parameters['AiryDiskPSF'])


def circular_gaussian_prf_units():
    """
    Return a CircularGaussianPRF with units.
    """
    return (CircularGaussianPRF(flux=1 * u.Jy,
                                x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                                fwhm=1 * u.arcsec, bbox_factor=2),
            parameters['CircularGaussianPRF'])


def circular_gaussian_prf():
    """
    Return a CircularGaussianPRF without units.
    """
    return (CircularGaussianPRF(flux=2, x_0=1, y_0=1, fwhm=2, bbox_factor=3),
            parameters['CircularGaussianPRF'])


def circular_gaussian_sigma_prf_units():
    """
    Return a CircularGaussianPRF with units.
    """
    return (CircularGaussianSigmaPRF(flux=71.4 * u.Jy,
                                     x_0=24.3 * u.pix, y_0=25.2 * u.pix,
                                     sigma=5.1 * u.pix),
            parameters['CircularGaussianSigmaPRF'])


def circular_gaussian_sigma_prf():
    """
    Return a CircularGaussianPRF without units.
    """
    return (CircularGaussianSigmaPRF(flux=71.4, x_0=24.3, y_0=25.2,
                                     sigma=5.1),
            parameters['CircularGaussianSigmaPRF'])


def circular_gaussian_psf_units():
    """
    Return a CircularGaussianPSF with units.
    """
    return (CircularGaussianPSF(flux=1 * u.Jy,
                                x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                                fwhm=1 * u.arcsec, bbox_factor=2),
            parameters['CircularGaussianPSF'])


def circular_gaussian_psf():
    """
    Return a CircularGaussianPSF without units.
    """
    return (CircularGaussianPSF(flux=2, x_0=1, y_0=1, fwhm=2,
                                bbox_factor=3),
            parameters['CircularGaussianPSF'])


def gaussian_prf_units():
    """
    Return a GaussianPRF with units.
    """
    return (GaussianPRF(flux=1 * u.Jy,
                        x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                        x_fwhm=1 * u.arcsec, y_fwhm=1 * u.arcsec,
                        theta=0 * u.deg, bbox_factor=2),
            parameters['GaussianPRF'])


def gaussian_prf():
    """
    Return a GaussianPRF without units.
    """
    return (GaussianPRF(flux=2, x_0=1, y_0=1, x_fwhm=2, y_fwhm=2,
                        theta=0,
                        bbox_factor=3),
            parameters['GaussianPRF'])


def gaussian_psf_units():
    """
    Return a GaussianPSF with units.
    """
    return (GaussianPSF(flux=1 * u.Jy, x_0=0 * u.arcsec, y_0=0 * u.arcsec,
                        x_fwhm=1 * u.arcsec, y_fwhm=1 * u.arcsec,
                        theta=0 * u.deg, bbox_factor=2),
            parameters['GaussianPSF'])


def gaussian_psf():
    """
    Return a GaussianPSF without units.
    """
    return (GaussianPSF(flux=2, x_0=1, y_0=1,
                        x_fwhm=2, y_fwhm=2,
                        theta=0, bbox_factor=3),
            parameters['GaussianPSF'])


def moffat_psf_units():
    """
    Return a MoffatPSF with units.
    """
    return (MoffatPSF(flux=71.4 * u.Jy,
                      x_0=24.3 * u.pix, y_0=25.2 * u.pix,
                      alpha=5.1 * u.pix, beta=3.2),
            parameters['MoffatPSF'])


def moffat_psf():
    """
    Return a MoffatPSF without units.
    """
    return (MoffatPSF(flux=71.4, x_0=24.3, y_0=25.2, alpha=5.1, beta=3.2),
            parameters['MoffatPSF'])


def image_psf():
    """
    Return an ImagePSF without units.
    """
    return (ImagePSF(data=np.arange(36).reshape((6, 6)),
                     flux=6,
                     x_0=0.5, y_0=0.5,
                     oversampling=1,
                     fill_value=0,
                     origin=(0, 0)),
            parameters['ImagePSF'])


def gridded_psf():
    """
    Return a GriddedPSFModel without units.
    """
    nd_data = NDData(data=np.arange(144).reshape((4, 6, 6)),
                     meta={'oversampling': 1,
                           'grid_xypos': [(2, 2),
                                          (65, 2),
                                          (2, 65),
                                          (65, 65)],
                           },
                     )
    return (GriddedPSFModel(nddata=nd_data, flux=6,
                            x_0=0.5, y_0=0.5,
                            fill_value=np.nan),
            parameters['GriddedPSFModel'])


def stdpsf_single_detector():
    """
    Return a STDPSFGrid object read from a FITS STDPSF file.
    """
    psfgrid = STDPSFGrid(str(PSF_DATA_DIR / 'STDPSF_NRCA1_F150W_mock.fits'))
    return psfgrid, parameters['STDPSFGrid']


def circular_aperture_single_pos():
    """
    Circular Aperture with a single position.
    """
    return (aperture.CircularAperture(positions=(5, 6), r=7),
            parameters['CircularAperture'])


def circular_aperture_multi_pos():
    """
    Circular aperture with multiple positions.
    """
    return (aperture.CircularAperture(positions=[(1, 2), (3, 4)], r=5),
            parameters['CircularAperture'])


def circular_annulus_single_pos():
    """
    Circular annulus aperture with a single position and theta=0.
    """
    return (aperture.CircularAnnulus([10.0, 20.0], 3.0, 5.0),
            parameters['CircularAnnulus'])


def circular_annulus_single_pos_tuple():
    """
    Circular annulus aperture with a single position as a tuple.
    """
    return (aperture.CircularAnnulus((10.0, 20.0), 3.0, 5.0),
            parameters['CircularAnnulus'])


def circular_annulus_multi_pos():
    """
    Circular annulus aperture with multiple positions.
    """
    return (aperture.CircularAnnulus([(10, 20), (30, 40)], 3, 5),
            parameters['CircularAnnulus'])


def sky_circular_annulus():
    """
    Circular annulus aperture on the sky.
    """
    sky = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    return (aperture.SkyCircularAnnulus(sky, 3.0 * u.arcsec, 5.0 * u.arcsec),
            parameters['CircularAnnulus'])


def sky_circular_aperture():
    """
    Circular aperture on the sky.
    """
    sky = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    return (aperture.SkyCircularAperture(sky, 3.0 * u.arcsec),
            parameters['CircularAperture'])


def elliptical_aperture(theta):
    """
    Elliptical aperture in pix coordinates with several values of theta.

    Parameters
    ----------
    theta : float, `~astropy.units.Quantity`, `~astropy.coordinates.Angle`
        Rotation angle.
    """
    return (aperture.EllipticalAperture(positions=(5, 6),
                                        a=7, b=3, theta=theta),
            parameters['EllipticalAperture'])


def sky_elliptical_aperture():
    """
    Elliptical aperture on the sky.
    """
    sky = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')

    return (aperture.SkyEllipticalAperture(sky,
                                           a=3 * u.arcsec,
                                           b=5 * u.arcsec,
                                           theta=0.5 * u.deg),
            parameters['EllipticalAperture'])


def elliptical_annulus(theta):
    """
    Elliptical annulus aperture in pixel space.

    Parameters
    ----------
    theta : float, `~astropy.units.Quantity`, `~astropy.coordinates.Angle`
        Rotation angle.
    """
    return (aperture.EllipticalAnnulus(positions=(5, 6), a_in=3, a_out=5,
                                       b_out=5, theta=theta),
            parameters['EllipticalAnnulus'])


def sky_elliptical_annulus():
    """
    Elliptical annulus aperture on the sky.
    """
    sky = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    return (aperture.SkyEllipticalAnnulus(sky,
            a_in=3 * u.arcsec,
            a_out=5 * u.arcsec,
            b_out=5 * u.arcsec,
            theta=0.5 * u.deg),
            parameters['EllipticalAnnulus'])


def polygon_aperture():
    """
    Polygon aperture on pixel space.
    """
    theta = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
    offsets = np.column_stack([5.0 * np.cos(theta),
                               5.0 * np.sin(theta)])
    return (aperture.PolygonAperture((10, 20), offsets),
            parameters['PolygonAperture'])


def polygon_aperture_vertices():
    """
    Polygon aperture in pixel space from vertices.
    """
    verts = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]
    return (aperture.PolygonAperture.from_vertices(verts),
            parameters['PolygonAperture'])


def sky_polygon_aperture():
    """
    Polygon aperture on the sky.
    """
    sky = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    r = 1.0
    offsets = np.column_stack([r * np.cos(theta),
                               r * np.sin(theta)]) * u.arcsec
    return (aperture.SkyPolygonAperture(sky, offsets),
            parameters['PolygonAperture'])


def rectangular_aperture(theta):
    """
    Rectangular aperture in pixel space.

    Parameters
    ----------
    theta : float, `~astropy.units.Quantity`, `~astropy.coordinates.Angle`
        Rotation angle.
    """
    return (aperture.RectangularAperture((10, 20), 5, 3, theta=theta),
            parameters['RectangularAperture'])


def sky_rectangular_aperture():
    """
    Rectangular aperture on the sky.
    """
    positions = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    return (aperture.SkyRectangularAperture(positions,
                                            1.0 * u.arcsec,
                                            0.5 * u.arcsec),
            parameters['RectangularAperture'])


def rectangular_annulus(theta):
    """
    Rectangular annulus aperture in pixel space.

    Parameters
    ----------
    theta : float, `~astropy.units.Quantity`, `~astropy.coordinates.Angle`
        Rotation angle.
    """
    return (aperture.RectangularAnnulus((10, 20), w_in=3, w_out=5,
                                        h_in=2, h_out=4, theta=theta),
            parameters['RectangularAnnulus'])


def sky_rectangular_annulus():
    """
    Rectangular annulus aperture on the sky.
    """
    positions = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    theta = Angle(80, 'deg')
    return (aperture.SkyRectangularAnnulus(positions,
                                           w_in=3 * u.arcsec,
                                           w_out=5 * u.arcsec,
                                           h_in=2 * u.arcsec,
                                           h_out=4 * u.arcsec,
                                           theta=theta),
            parameters['RectangularAnnulus'])
