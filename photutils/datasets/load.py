# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Load example datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename, download_file

__all__ = ['get_path',
           'load_spitzer_image',
           'load_spitzer_catalog',
           'load_fermi_image',
           'load_star_image'
           ]


def get_path(filename, location='local'):
    """Get path (location on your disk) for a given file.

    Parameters
    ----------
    filename : str
        File name in the local or remote data folder
    location : {'local', 'remote'}
        File location.
        ``'local'`` means bundled with ``photutils``.
        ``'remote'`` means a server or the Astropy cache on your machine.

    Returns
    -------
    path : str
        Path (location on your disk) of the file.

    Examples
    --------
    >>> from astropy.io import fits
    >>> from photutils import datasets
    >>> hdu_list = fits.open(datasets.get_path('fermi_counts.fits.gz'))
    """
    if location == 'local':
        path = get_pkg_data_filename('data/' + filename)
    elif location == 'remote':    # pragma: no cover
        url = ('https://github.com/astropy/photutils-datasets/blob/master/'
               'data/{0}?raw=true'.format(filename))
        path = download_file(url, cache=True)
    else:
        raise ValueError('Invalid location: {0}'.format(location))

    return path


def load_spitzer_image():    # pragma: no cover
    """
    Load 4.5 micron Spitzer image.

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        The 4.5 micron image

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        hdu = datasets.load_spitzer_image()
        plt.imshow(hdu.data, origin='lower', vmax=50)
    """
    path = get_path('spitzer_example_image.fits', location='remote')
    hdu = fits.open(path)[0]

    return hdu


def load_spitzer_catalog():    # pragma: no cover
    """
    Load 4.5 micron Spitzer catalog.

    This is the catalog corresponding to the image returned by
    `~photutils.datasets.load_spitzer_image`.

    Returns
    -------
    catalog : `~astropy.table.Table`
        The catalog of sources

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        catalog = datasets.load_spitzer_catalog()
        plt.scatter(catalog['l'], catalog['b'])
        plt.xlim(18.39, 18.05)
        plt.ylim(0.13, 0.30)
    """
    path = get_path('spitzer_example_catalog.xml', location='remote')
    table = Table.read(path)

    return table


def load_fermi_image():
    """Load Fermi counts image for the Galactic center region.

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        Image HDU

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        hdu = datasets.load_fermi_image()
        plt.imshow(hdu.data, origin='lower', vmax=10)
    """
    path = get_path('fermi_counts.fits.gz', location='local')
    hdu = fits.open(path)[1]

    return hdu


def load_star_image():    # pragma: no cover
    """Load an optical image with stars.

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        Image HDU

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        hdu = datasets.load_star_image()
        plt.imshow(hdu.data, origin='lower', cmap='gray')
    """
    path = get_path('M6707HH.fits.gz', location='remote')
    hdu = fits.open(path)[0]

    return hdu
