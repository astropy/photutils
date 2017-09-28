# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import numpy as np
from astropy.io import fits

from ...datasets import make_noise_image
from ..geometry import Geometry, DEFAULT_EPS


DEFAULT_SIZE = 512
DEFAULT_POS = int(DEFAULT_SIZE / 2)
DEFAULT_PA = 0.


def make_test_image(nx=DEFAULT_SIZE, ny=DEFAULT_SIZE, x0=None, y0=None,
                    background=100., noise=1.e-6, i0=100., sma=40.,
                    eps=DEFAULT_EPS, pa=DEFAULT_PA, random_state=None):
    """
    Make a simulated image for testing the isophot subpackage.

    Parameters
    ----------
    nx, ny : int, optional
        The image size.
    x0, y0 : int, optional
        The center position.  If `None`, the default is the image center.
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
    random_state : int or `~numpy.random.RandomState`, optional
        Pseudo-random number generator state used for random sampling.
        Separate function calls with the same ``random_state`` will
        generate the identical noise image.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The resulting simulated image.
    """

    image = np.zeros((ny, nx)) + background

    x1 = x0
    y1 = y0
    if not x0 or not y0:
        x1 = nx / 2
        y1 = ny / 2

    g = Geometry(x1, y1, sma, eps, pa, 0.1, False)

    for j in range(ny):
        for i in range(nx):
            radius, angle = g.to_polar(i, j)
            e_radius = g.radius(angle)
            value = (lambda x: i0 * math.exp(-7.669 * (x**0.25 - 1.)))(radius / e_radius)
            image[j, i] += value

    # central pixel is messed up; replace it with interpolated value
    image[int(x1), int(y1)] = (image[int(x1 - 1), int(y1)] +
                               image[int(x1 + 1), int(y1)] +
                               image[int(x1), int(y1 - 1)] +
                               image[int(x1), int(y1 + 1)]) / 4.

    image += make_noise_image(image.shape, type='gaussian', mean=0.,
                              stddev=noise, random_state=random_state)

    return image


def make_fits_test_image(name, nx=512, ny=512, x0=None, y0=None,
                         background=100., noise=1.e-6, i0=100., sma=40.,
                         eps=0.2, pa=0., random_state=None):
    """
    Make a simulated image and write it to a FITS file.

    Used for testing the isophot subpackage.

    Examples
    --------
    import numpy as np
    pa = np.pi / 4.
    make_fits_test_image('synth_lowsnr.fits', noise=40., pa=pa,
                         random_state=123)
    make_fits_test_image('synth_highsnr.fits', noise=1.e-12, pa=pa,
                         random_state=123)
    make_fits_test_image('synth.fits', pa=pa, random_state=123)
    """

    if not name.endswith('.fits'):
        name += '.fits'

    array = make_test_image(nx, ny, x0, y0, background, noise, i0, sma, eps,
                            pa)

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
