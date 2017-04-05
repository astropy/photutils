import math

import numpy as np

import astropy.io.fits as fits

from .geometry import Geometry, DEFAULT_EPS

DEFAULT_SIZE = 512
DEFAULT_POS = int(DEFAULT_SIZE / 2)
DEFAULT_PA = 0.


def build(nx=DEFAULT_SIZE, ny=DEFAULT_SIZE, x0=None, y0=None, background=100., noise=1.E-6, i0=100., sma=40., eps=DEFAULT_EPS, pa=DEFAULT_PA):
    '''
    Builds artificial image for testing purposes

    :param nx: int
        image size
    :param ny: int
        image size
    :param x0: int
        position of center. If None, default to image center.
    :param y0: int
        position of center. If None, default to image center.
    :param background: float
        constant background level to be add to all pixels in the image
    :param noise: float
        standard deviation of the Gaussian noise to be added to all pixels in the image
    :param i0: float
        surface brightness over reference elliptical isophote
    :param sma: float
        semi-major axis length of reference elliptical isophote.
    :param eps: float
        ellipticity of reference isophote
    :param pa: float
        position angle of reference isophote
    :return: 2-d numpy array
        resulting image
    '''
    image = np.zeros((ny, nx)) + background

    x1 = x0
    y1 = y0
    if not x0 or not y0:
        x1 = nx/2
        y1 = ny/2

    g = Geometry(x1, y1, sma, eps, pa, 0.1, False)

    for j in range(ny):
        for i in range(nx):
            radius, angle = g.to_polar(i, j)
            e_radius = g.radius(angle)
            value = (lambda x: i0 * math.exp(-7.669 * ((x) ** 0.25 - 1.)))(radius / e_radius)
            image[j,i] += value

    # central pixel is messed up; replace it with interpolated value
    image[int(x1), int(y1)] = (image[int(x1-1), int(y1)]   + image[int(x1+1), int(y1)] +
                               image[int(x1),   int(y1-1)] + image[int(x1),   int(y1+1)]) / 4.

    image += np.random.normal(0., noise, image.shape)

    return image


def build_image(name, nx=512, ny=512, x0=None, y0=None, background=100., noise=1.E-6, i0=100., sma=40., eps=0.2, pa=0.):

    if not name.endswith('.fits'):
        name += '.fits'

    array = build(nx, ny, x0, y0, background, noise, i0, sma, eps, pa)

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


if __name__ == '__main__' :
    # build_image('../test/data/synth_lowsnr', noise=40., pa=np.pi/4)
    build_image('../test/data/synth_highsnr', noise=1.E-12, pa=np.pi/4)
    # build_image('../test/data/synth', pa=np.pi/4)



