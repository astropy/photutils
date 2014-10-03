# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import os.path as op
import itertools
import numpy as np
from astropy.table import Table
from numpy.testing import assert_allclose
from ..findstars import daofind, irafstarfind

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


VALS1 = [0.5, 1.0, 2.0, 5.0, 10.0]
VALS2 = [1.3, 1.5, 2.0, 5.0, 10.0]
THRESHOLDS = [3.0, 5.0, 10.0]


def _rotxy(x, y, ang):
    """
    Rotate an (x, y) coordinate counterclockwise by ``ang`` degrees.

    Parameters
    ----------
    x, y : float
        (x, y) coordinate pair.

    ang : float
        The counterclockwise rotation angle in degrees.

    Returns
    -------
    x, y : float
        Rotated (x, y) coordinate pair.
    """

    ang *= np.pi / 180.0
    xrot = x*np.cos(ang) - y*np.sin(ang)
    yrot = x*np.sin(ang) + y*np.cos(ang)
    return xrot, yrot


def _subimg_bbox(img, subimage, xc, yc):
    """
    Find the x/y bounding-box pixel coordinates in ``img`` needed to
    add ``subimage``, centered at ``(xc, yc)``, to ``img``.  Returns
    ``None`` if the ``subimage`` would extend past the ``img``
    boundary.
    """

    ys, xs = subimage.shape
    y, x = img.shape
    y0 = int(yc - (ys - 1) / 2.0)
    y1 = y0 + ys
    x0 = int(xc - (xs - 1) / 2.0)
    x1 = x0 + xs
    if (x0 >= 0) and (y0 >= 0) and (x1 < x) and (y1 < y):
        return (x0, x1, y0, y1)
    else:
        return None


def _addsubimg(img, subimage, bbox):
    """
    Add a ``subimage`` to ``img`` at the specified ``bbox`` bounding
    box.
    """

    x0, x1, y0, y1 = bbox
    img[y0:y1, x0:x1] += subimage
    return img


def _gaussian2d(shape, xcen, ycen, xfwhm, yfwhm, rot=0.0, normalize=True):
    """
    Generate a 2D Gaussian function.

    Parameters
    ----------
    shape :
        The shape of the output 2D image.  Note that ``shape`` will
        be forced to be odd.

    xcen, ycen : float
        The (x, y) center of the 2D Gaussian funciton.

    xfwhm, yfwhm : float
        The x/y FWHM of the 2D Gaussian function.

    rot : float, optional
        The counterclockwise rotation angle (in degrees) of the 2D
        Gaussian "x-axis".

    normalize : boolean, optional
        Set to normalize the total of the Gaussian function to 1.0.
        Default is ``True``.

    Returns
    -------
    data : array-like
        A 2D array of given ``shape`` (``shape`` will be made odd).
    """

    sigtofwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    xsigma = xfwhm / sigtofwhm
    ysigma = yfwhm / sigtofwhm
    shape = [(n - 1) if (n % 2 == 0) else n for n in shape]   # make odd
    y, x = np.indices(shape)
    xx = x - xcen
    yy = y - ycen
    if rot:
        xx, yy = _rotxy(xx, yy, rot)
    g2d = np.exp(-((xx**2 / (2.0 * xsigma**2)) + (yy**2 / (2.0 * ysigma**2))))
    if normalize:
        g2d /= g2d.sum()
    return g2d


class SetupData(object):
    def _setup(self, nobj=200, xsize=501, ysize=501):
        img = np.zeros((ysize, xsize))
        nround = int(nobj * 0.1)
        prng = np.random.RandomState(1234567890)
        xcens = prng.uniform(0, xsize, nobj)
        ycens = prng.uniform(0, ysize, nobj)
        xfwhms = prng.uniform(0.5, 10.0, nobj)
        yfwhms = prng.uniform(0.5, 10.0, nobj)
        yfwhms[0:nround] = xfwhms[0:nround]
        rots = prng.uniform(0, 360, nobj)
        peaks = prng.normal(100, 20, nobj)
        for (xcen, ycen, xfwhm, yfwhm, rot, peak) in zip(xcens, ycens,
                                                         xfwhms, yfwhms,
                                                         rots, peaks):
            kysize = np.int(yfwhm * 4.0) + 1
            kxsize = np.int(xfwhm * 4.0) + 1
            kycen, kxcen = kysize // 2, kxsize // 2
            g = _gaussian2d((kysize, kxsize), kxcen, kycen, xfwhm,
                            yfwhm, rot=rot) * peak
            bbox = _subimg_bbox(img, g, xcen, ycen)
            if bbox:
                img = _addsubimg(img, g, bbox)
        noiseimg = prng.randn(ysize, xsize)
        self.img = img + 0.4*noiseimg


class TestDAOFind(SetupData):
    @pytest.mark.parametrize(('fwhm', 'sigma_radius'),
                             list(itertools.product(VALS1, VALS1)))
    @pytest.mark.skipif('not HAS_SCIPY')
    def test_ellrotobj(self, fwhm, sigma_radius):
        self._setup()
        threshold = 5.0
        tbl = daofind(self.img, threshold, fwhm, sigma_radius=sigma_radius)
        datafn = ('daofind_test_fwhm{0:04.1f}_thresh{1:04.1f}_'
                  'sigrad{2:04.1f}.txt'.format(fwhm, threshold, sigma_radius))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t = Table.read(datafn, format='ascii')
        assert_allclose(np.array(tbl).astype(np.float),
                        np.array(t).astype(np.float))

    @pytest.mark.parametrize(('threshold'), THRESHOLDS)
    @pytest.mark.skipif('not HAS_SCIPY')
    def test_ellrotobj_threshold(self, threshold):
        self._setup()
        fwhm = 3.0
        sigma_radius = 1.5
        tbl = daofind(self.img, threshold, fwhm, sigma_radius=sigma_radius)
        datafn = ('daofind_test_fwhm{0:04.1f}_thresh{1:04.1f}_'
                  'sigrad{2:04.1f}.txt'.format(fwhm, threshold, sigma_radius))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t = Table.read(datafn, format='ascii')
        assert_allclose(np.array(tbl).astype(np.float),
                        np.array(t).astype(np.float))


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestIRAFStarFind(SetupData):
    @pytest.mark.parametrize(('fwhm', 'sigma_radius'),
                             list(itertools.product(VALS1, VALS2)))
    def test_isf_ellrotobj(self, fwhm, sigma_radius):
        self._setup()
        threshold = 5.0
        tbl = irafstarfind(self.img, threshold, fwhm,
                           sigma_radius=sigma_radius)
        datafn = ('irafstarfind_test_fwhm{0:04.1f}_thresh{1:04.1f}_'
                  'sigrad{2:04.1f}.txt'.format(fwhm, threshold, sigma_radius))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t = Table.read(datafn, format='ascii')
        assert_allclose(np.array(tbl).astype(np.float),
                        np.array(t).astype(np.float))

    @pytest.mark.parametrize(('threshold'), THRESHOLDS)
    def test_isf_ellrotobj_threshold(self, threshold):
        self._setup()
        fwhm = 3.0
        sigma_radius = 1.5
        tbl = irafstarfind(self.img, threshold, fwhm,
                           sigma_radius=sigma_radius)
        datafn = ('irafstarfind_test_fwhm{0:04.1f}_thresh{1:04.1f}_'
                  'sigrad{2:04.1f}.txt'.format(fwhm, threshold, sigma_radius))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t = Table.read(datafn, format='ascii')
        assert_allclose(np.array(tbl).astype(np.float),
                        np.array(t).astype(np.float))
