# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Load example datasets.
"""

from urllib.error import HTTPError, URLError

from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename, download_file


__all__ = ['get_path', 'load_spitzer_image', 'load_spitzer_catalog',
           'load_irac_psf', 'load_fermi_image', 'load_star_image',
           'load_simulated_hst_star_image']


def get_path(filename, location='local', cache=True, show_progress=False):
    """
    Get path (location on your disk) for a given file.

    Parameters
    ----------
    filename : str
        File name in the local or remote data folder.
    location : {'local', 'remote', 'photutils-datasets'}
        File location.  ``'local'`` means bundled with ``photutils``.
        ``'remote'`` means the astropy data server (or the
        photutils-datasets repo as a backup) or the Astropy cache on
        your machine. ``'photutils-datasets'`` means the
        photutils-datasets repo or the Astropy cache on your machine.
    cache : bool, optional
        Whether to cache the contents of remote URLs.  Default is
        `True`.
    show_progress : bool, optional
        Whether to display a progress bar during the download (default
        is `False`).

    Returns
    -------
    path : str
        Path (location on your disk) of the file.

    Examples
    --------
    >>> from astropy.io import fits
    >>> from photutils import datasets
    >>> hdulist = fits.open(datasets.get_path('fermi_counts.fits.gz'))
    """

    datasets_url = ('https://github.com/astropy/photutils-datasets/raw/'
                    'master/data/{0}'.format(filename))

    if location == 'local':
        path = get_pkg_data_filename('data/' + filename)
    elif location == 'remote':    # pragma: no cover
        try:
            url = 'https://data.astropy.org/photometry/{0}'.format(filename)
            path = download_file(url, cache=cache,
                                 show_progress=show_progress)
        except (URLError, HTTPError):   # timeout or not found
            path = download_file(datasets_url, cache=cache,
                                 show_progress=show_progress)
    elif location == 'photutils-datasets':    # pragma: no cover
            path = download_file(datasets_url, cache=cache,
                                 show_progress=show_progress)
    else:
        raise ValueError('Invalid location: {0}'.format(location))

    return path


def load_spitzer_image(show_progress=False):    # pragma: no cover
    """
    Load a 4.5 micron Spitzer image.

    The catalog for this image is returned by
    :func:`load_spitzer_catalog`.

    Parameters
    ----------
    show_progress : bool, optional
        Whether to display a progress bar during the download (default
        is `False`).

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        The 4.5 micron Spitzer image in a FITS image HDU.

    See Also
    --------
    load_spitzer_catalog

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        hdu = datasets.load_spitzer_image()
        plt.imshow(hdu.data, origin='lower', vmax=50)
    """

    path = get_path('spitzer_example_image.fits', location='remote',
                    show_progress=show_progress)
    hdu = fits.open(path)[0]

    return hdu


def load_spitzer_catalog(show_progress=False):    # pragma: no cover
    """
    Load a 4.5 micron Spitzer catalog.

    The image from which this catalog was derived is returned by
    :func:`load_spitzer_image`.

    Parameters
    ----------
    show_progress : bool, optional
        Whether to display a progress bar during the download (default
        is `False`).

    Returns
    -------
    catalog : `~astropy.table.Table`
        The catalog of sources.

    See Also
    --------
    load_spitzer_image

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        catalog = datasets.load_spitzer_catalog()
        plt.scatter(catalog['l'], catalog['b'])
        plt.xlabel('Galactic l')
        plt.ylabel('Galactic b')
        plt.xlim(18.39, 18.05)
        plt.ylim(0.13, 0.30)
    """

    path = get_path('spitzer_example_catalog.xml', location='remote',
                    show_progress=show_progress)
    table = Table.read(path)

    return table


def load_irac_psf(channel, show_progress=False):    # pragma: no cover
    """
    Load a Spitzer IRAC PSF image.

    Parameters
    ----------
    channel : int (1-4)
        The IRAC channel number:

          * Channel 1:  3.6 microns
          * Channel 2:  4.5 microns
          * Channel 3:  5.8 microns
          * Channel 4:  8.0 microns

    show_progress : bool, optional
        Whether to display a progress bar during the download (default
        is `False`).

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        The IRAC PSF in a FITS image HDU.

    Examples
    --------
    .. plot::
        :include-source:

        from astropy.visualization import LogStretch, ImageNormalize
        from photutils.datasets import load_irac_psf
        hdu1 = load_irac_psf(1)
        hdu2 = load_irac_psf(2)
        hdu3 = load_irac_psf(3)
        hdu4 = load_irac_psf(4)

        norm = ImageNormalize(hdu1.data, stretch=LogStretch())

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(hdu1.data, origin='lower', interpolation='nearest',
                   norm=norm)
        ax1.set_title('IRAC Ch1 PSF')
        ax2.imshow(hdu2.data, origin='lower', interpolation='nearest',
                   norm=norm)
        ax2.set_title('IRAC Ch2 PSF')
        ax3.imshow(hdu3.data, origin='lower', interpolation='nearest',
                   norm=norm)
        ax3.set_title('IRAC Ch3 PSF')
        ax4.imshow(hdu4.data, origin='lower', interpolation='nearest',
                   norm=norm)
        ax4.set_title('IRAC Ch4 PSF')
        plt.tight_layout()
        plt.show()
    """

    channel = int(channel)
    if channel < 1 or channel > 4:
        raise ValueError('channel must be 1, 2, 3, or 4')

    fn = 'irac_ch{0}_flight.fits'.format(channel)
    path = get_path(fn, location='remote', show_progress=show_progress)
    hdu = fits.open(path)[0]

    return hdu


def load_fermi_image(show_progress=False):
    """
    Load a Fermi counts image for the Galactic center region.

    Parameters
    ----------
    show_progress : bool, optional
        Whether to display a progress bar during the download (default
        is `False`).

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        A FITS image HDU.

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        hdu = datasets.load_fermi_image()
        plt.imshow(hdu.data, vmax=10, origin='lower', interpolation='nearest')
    """

    path = get_path('fermi_counts.fits.gz', location='local',
                    show_progress=show_progress)
    hdu = fits.open(path)[1]

    return hdu


def load_star_image(show_progress=False):    # pragma: no cover
    """
    Load an optical image of stars.

    This is an image of M67 from photographic data obtained as part of
    the National Geographic Society - Palomar Observatory Sky Survey
    (NGS-POSS).  The image was digitized from the POSS-I Red plates as
    part of the Digitized Sky Survey produced at the Space Telescope
    Science Institute.

    Parameters
    ----------
    show_progress : bool, optional
        Whether to display a progress bar during the download (default
        is `False`).

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        The M67 image in a FITS image HDU.

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        hdu = datasets.load_star_image()
        plt.imshow(hdu.data, origin='lower', interpolation='nearest')
    """

    path = get_path('M6707HH.fits', location='remote',
                    show_progress=show_progress)
    hdu = fits.open(path)[0]

    return hdu


def load_simulated_hst_star_image(show_progress=False):  # pragma: no cover
    """
    Load a simulated HST WFC3/IR F160W image of stars.

    The simulated image does not contain any background or noise.

    Parameters
    ----------
    show_progress : bool, optional
        Whether to display a progress bar during the download (default
        is `False`).

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        A FITS image HDU containing the simulated HST star image.

    Examples
    --------
    .. plot::
        :include-source:

        from photutils import datasets
        hdu = datasets.load_simulated_hst_star_image()
        plt.imshow(hdu.data, origin='lower', interpolation='nearest')
    """

    path = get_path('hst_wfc3ir_f160w_simulated_starfield.fits',
                    location='photutils-datasets',
                    show_progress=show_progress)
    hdu = fits.open(path)[0]

    return hdu
