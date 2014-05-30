# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Load example datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename, download_file

__all__ = ['load_fermi_image', 'load_star_image']


def _get_url(filename):
    # TODO: for now we use a GitHub repo to store large files
    return 'https://github.com/cdeil/photutils-datasets/blob/master/{0}?raw=true'.format(filename)


def load_fermi_image():
    """Load Fermi counts image for the Galactic center region.
    
    An example of a small file bundled in the ``photutils`` repo.
    
    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        Image HDU
    """
    filename = get_pkg_data_filename('data/fermi_counts.fits.gz')
    hdu = fits.open(filename)[1]

    return hdu


def load_star_image():
    """Load optical image with stars.
    
    An example of a large file from the external ``photutils-datasets`` repo.

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        Image HDU
    """
    #URL = 'http://irsa.ipac.caltech.edu/data/SPITZER/GLIMPSE/images/I/1.2_mosaics_v2.0/GLON_10-30/GLM_01800+0000_mosaic_I2.fits'
    url = _get_url('M6707HH.fits.gz')
    filename = download_file(url, cache=True)
    hdu = fits.open(filename)[0]

    return hdu
