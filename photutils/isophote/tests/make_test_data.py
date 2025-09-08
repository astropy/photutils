# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provide tools for making a simulated image for tests.
"""

import numpy as np
from astropy.io import fits

from photutils.datasets import make_noise_image
from photutils.isophote.geometry import EllipseGeometry


def make_test_image(nx=512, ny=512, x0=None, y0=None,
                    background=100.0, noise=1.0e-6, i0=100.0, sma=40.0,
                    eps=0.2, pa=0.0, seed=None):
    """
    Make a simulated image for testing the isophote subpackage.

    Parameters
    ----------
    nx, ny : int, optional
        The image size.

    x0, y0 : int, optional
        The center position. If `None`, the default is the image center.

    background : float, optional
        The constant background level to add to the image values.

    noise : float, optional
        The standard deviation of the Gaussian noise to add to the image.

    i0 : float, optional
        The surface brightness over the reference elliptical isophote.

    sma : float, optional
        The semi-major axis length of the reference elliptical isophote.

    eps : float, optional
        The ellipticity of the reference isophote.

    pa : float, optional
        The position angle of the reference isophote.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The resulting simulated image.
    """
    if x0 is None or y0 is None:
        xcen = nx / 2
        ycen = ny / 2
    else:
        xcen = x0
        ycen = y0

    g = EllipseGeometry(xcen, ycen, sma, eps, pa, 0.1, linear_growth=False)

    y, x = np.mgrid[0:ny, 0:nx]
    radius, angle = g.to_polar(x, y)
    e_radius = g.radius(angle)
    tmp_image = radius / e_radius

    image = i0 * np.exp(-7.669 * (tmp_image**0.25 - 1.0)) + background

    # central pixel is messed up; replace it with interpolated value
    image[int(xcen), int(ycen)] = (image[int(xcen - 1), int(ycen)]
                                   + image[int(xcen + 1), int(ycen)]
                                   + image[int(xcen), int(ycen - 1)]
                                   + image[int(xcen), int(ycen + 1)]) / 4.0

    image += make_noise_image(image.shape, distribution='gaussian', mean=0.0,
                              stddev=noise, seed=seed)

    return image


def make_fits_test_image(name, nx=512, ny=512, x0=None, y0=None,
                         background=100.0, noise=1.0e-6, i0=100.0, sma=40.0,
                         eps=0.2, pa=0.0, seed=None):
    """
    Make a simulated image and write it to a FITS file.

    Used for testing the isophot subpackage.

    Parameters
    ----------
    name : str
        The name of the output FITS file.

    nx, ny : int, optional
        The image size.

    x0, y0 : int, optional
        The center position. If `None`, the default is the image center.

    background : float, optional
        The constant background level to add to the image values.

    noise : float, optional
        The standard deviation of the Gaussian noise to add to the image.

    i0 : float, optional
        The surface brightness over the reference elliptical isophote.

    sma : float, optional
        The semi-major axis length of the reference elliptical isophote.

    eps : float, optional
        The ellipticity of the reference isophote.

    pa : float, optional
        The position angle of the reference isophote.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled
        from the OS.

    Examples
    --------
    import numpy as np
    pa = np.pi / 4.
    make_fits_test_image('synth_lowsnr.fits', noise=40., pa=pa,
                         seed=0)
    make_fits_test_image('synth_highsnr.fits', noise=1.e-12, pa=pa,
                         seed=0)
    make_fits_test_image('synth.fits', pa=pa, seed=0)
    """
    if not name.endswith('.fits'):
        name += '.fits'

    array = make_test_image(nx, ny, x0, y0, background, noise, i0, sma, eps,
                            pa, seed=seed)

    hdu = fits.PrimaryHDU(array)
    hdulist = fits.HDUList([hdu])

    header = hdulist[0].header
    header['X0'] = (x0, 'x position of galaxy center')
    header['Y0'] = (y0, 'y position of galaxy center')
    header['BACK'] = (background, 'constant background value')
    header['NOISE'] = (noise, 'standard deviation of noise')
    header['I0'] = (i0, 'reference pixel value')
    header['SMA'] = (sma, 'reference semi major axis')
    header['EPS'] = (eps, 'ellipticity')
    header['PA'] = (pa, 'position ange')

    hdulist.writeto(name)
