# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The tests in this file test the accuracy of the photometric results.
Here we test directly with aperture objects since we are checking the
algorithms in aperture_photometry, not in the wrappers.
"""

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_array_less)

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
import astropy.units as u
from astropy.utils import minversion
from astropy.wcs.utils import pixel_to_skycoord

from ..core import aperture_photometry
from ..circle import (CircularAperture, CircularAnnulus, SkyCircularAperture,
                      SkyCircularAnnulus)
from ..ellipse import (EllipticalAperture, EllipticalAnnulus,
                       SkyEllipticalAperture, SkyEllipticalAnnulus)
from ..rectangle import (RectangularAperture, RectangularAnnulus,
                         SkyRectangularAperture, SkyRectangularAnnulus)
from ...datasets import (get_path, make_4gaussians_image, make_wcs,
                         make_imagehdu)

try:
    import matplotlib    # noqa
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

NUMPY_LT_1_14 = not minversion('numpy', '1.14dev')

APERTURE_CL = [CircularAperture,
               CircularAnnulus,
               EllipticalAperture,
               EllipticalAnnulus,
               RectangularAperture,
               RectangularAnnulus]

TEST_APERTURES = list(zip(APERTURE_CL, ((3.,), (3., 5.),
                                        (3., 5., 1.), (3., 5., 4., 1.),
                                        (5, 8, np.pi / 4),
                                        (8, 12, 8, np.pi / 8))))


@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_outside_array(aperture_class, params):
    data = np.ones((10, 10), dtype=np.float)
    aperture = aperture_class((-60, 60), *params)
    fluxtable = aperture_photometry(data, aperture)
    # aperture is fully outside array:
    assert np.isnan(fluxtable['aperture_sum'])


@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_inside_array_simple(aperture_class, params):
    data = np.ones((40, 40), dtype=np.float)
    aperture = aperture_class((20., 20.), *params)
    table1 = aperture_photometry(data, aperture, method='center', subpixels=10)
    table2 = aperture_photometry(data, aperture, method='subpixel',
                                 subpixels=10)
    table3 = aperture_photometry(data, aperture, method='exact', subpixels=10)
    true_flux = aperture.area()

    if not isinstance(aperture, (RectangularAperture, RectangularAnnulus)):
        assert_allclose(table3['aperture_sum'], true_flux)
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)
    assert table1['aperture_sum'] < table3['aperture_sum']


@pytest.mark.skipif('not HAS_MATPLOTLIB')
@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_aperture_plots(aperture_class, params):
    # This test should run without any errors, and there is no return
    # value.
    # TODO: check the content of the plot
    aperture = aperture_class((20., 20.), *params)
    aperture.plot()


def test_aperture_pixel_positions():
    pos1 = (10, 20)
    pos2 = u.Quantity((10, 20), unit=u.pixel)
    pos3 = ((10, 20, 30), (10, 20, 30))
    pos3_pairs = ((10, 10), (20, 20), (30, 30))

    r = 3
    ap1 = CircularAperture(pos1, r)
    ap2 = CircularAperture(pos2, r)
    ap3 = CircularAperture(pos3, r)

    assert_allclose(np.atleast_2d(pos1), ap1.positions)
    assert_allclose(np.atleast_2d(pos2.value), ap2.positions)
    assert_allclose(pos3_pairs, ap3.positions)


class BaseTestAperturePhotometry:

    def test_scalar_error(self):
        # Scalar error
        error = 1.
        if not hasattr(self, 'mask'):
            mask = None
            true_error = np.sqrt(self.area)
        else:
            mask = self.mask
            # 1 masked pixel
            true_error = np.sqrt(self.area - 1)

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, error=error)
        table2 = aperture_photometry(self.data,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, error=error)
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, error=error)

        if not isinstance(self.aperture, (RectangularAperture,
                                          RectangularAnnulus)):
            assert_allclose(table3['aperture_sum'], self.true_flux)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum'] < table3['aperture_sum'])

        if not isinstance(self.aperture, (RectangularAperture,
                                          RectangularAnnulus)):
            assert_allclose(table3['aperture_sum_err'], true_error)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum_err'] < table3['aperture_sum_err'])

    def test_array_error(self):
        # Array error
        error = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
            true_error = np.sqrt(self.area)
        else:
            mask = self.mask
            # 1 masked pixel
            true_error = np.sqrt(self.area - 1)

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, error=error)
        table2 = aperture_photometry(self.data,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, error=error)
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, error=error)

        if not isinstance(self.aperture, (RectangularAperture,
                                          RectangularAnnulus)):
            assert_allclose(table3['aperture_sum'], self.true_flux)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum'] < table3['aperture_sum'])

        if not isinstance(self.aperture, (RectangularAperture,
                                          RectangularAnnulus)):
            assert_allclose(table3['aperture_sum_err'], true_error)
            assert_allclose(table2['aperture_sum_err'],
                            table3['aperture_sum_err'], atol=0.1)
        assert np.all(table1['aperture_sum_err'] < table3['aperture_sum_err'])


class TestCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        r = 10.
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.true_flux = self.area


class TestCircularArray(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = ((20., 20.), (25., 25.))
        r = 10.
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.area = np.array((self.area, ) * 2)
        self.true_flux = self.area


class TestCircularAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        r_in = 8.
        r_out = 10.
        self.aperture = CircularAnnulus(position, r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.true_flux = self.area


class TestCircularAnnulusArray(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = ((20., 20.), (25., 25.))
        r_in = 8.
        r_out = 10.
        self.aperture = CircularAnnulus(position, r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.area = np.array((self.area, ) * 2)
        self.true_flux = self.area


class TestElliptical(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        a = 10.
        b = 5.
        theta = -np.pi / 4.
        self.aperture = EllipticalAperture(position, a, b, theta)
        self.area = np.pi * a * b
        self.true_flux = self.area


class TestEllipticalAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        a_in = 5.
        a_out = 8.
        b_out = 5.
        theta = -np.pi / 4.
        self.aperture = EllipticalAnnulus(position, a_in, a_out, b_out, theta)
        self.area = (np.pi * (a_out * b_out) -
                     np.pi * (a_in * b_out * a_in / a_out))
        self.true_flux = self.area


class TestRectangularAperture(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        h = 5.
        w = 8.
        theta = np.pi / 4.
        self.aperture = RectangularAperture(position, w, h, theta)
        self.area = h * w
        self.true_flux = self.area


class TestRectangularAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        h_out = 8.
        w_in = 8.
        w_out = 12.
        h_in = w_in * h_out / w_out
        theta = np.pi / 8.
        self.aperture = RectangularAnnulus(position, w_in, w_out, h_out, theta)
        self.area = h_out * w_out - h_in * w_in
        self.true_flux = self.area


class TestMaskedSkipCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[20, 20] = True
        position = (20., 20.)
        r = 10.
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.true_flux = self.area - 1


class BaseTestDifferentData:

    def test_basic_circular_aperture_photometry(self):
        aperture = CircularAperture(self.position, self.radius)
        table = aperture_photometry(self.data, aperture,
                                    method='exact', unit='adu')

        assert_allclose(table['aperture_sum'].value, self.true_flux)
        assert table['aperture_sum'].unit, self.fluxunit

        assert np.all(table['xcenter'].value ==
                      np.transpose(self.position)[0])
        assert np.all(table['ycenter'].value ==
                      np.transpose(self.position)[1])


class TestInputPrimaryHDU(BaseTestDifferentData):

    def setup_class(self):
        data = np.ones((40, 40), dtype=np.float)
        self.data = fits.ImageHDU(data=data)
        self.data.header['BUNIT'] = 'adu'
        self.radius = 3
        self.position = (20, 20)
        self.true_flux = np.pi * self.radius * self.radius
        self.fluxunit = u.adu


class TestInputHDUList(BaseTestDifferentData):

    def setup_class(self):
        data0 = np.ones((40, 40), dtype=np.float)
        data1 = np.empty((40, 40), dtype=np.float)
        data1.fill(2)
        self.data = fits.HDUList([fits.ImageHDU(data=data0),
                                  fits.ImageHDU(data=data1)])
        self.radius = 3
        self.position = (20, 20)
        # It should stop at the first extension
        self.true_flux = np.pi * self.radius * self.radius


class TestInputHDUDifferentBUNIT(BaseTestDifferentData):

    def setup_class(self):
        data = np.ones((40, 40), dtype=np.float)
        self.data = fits.ImageHDU(data=data)
        self.data.header['BUNIT'] = 'Jy'
        self.radius = 3
        self.position = (20, 20)
        self.true_flux = np.pi * self.radius * self.radius
        self.fluxunit = u.adu


class TestInputNDData(BaseTestDifferentData):

    def setup_class(self):
        data = np.ones((40, 40), dtype=np.float)
        self.data = NDData(data, unit=u.adu)
        self.radius = 3
        self.position = [(20, 20), (30, 30)]
        self.true_flux = np.pi * self.radius * self.radius
        self.fluxunit = u.adu


@pytest.mark.remote_data
def test_wcs_based_photometry_to_catalogue():
    pathcat = get_path('spitzer_example_catalog.xml', location='remote')
    pathhdu = get_path('spitzer_example_image.fits', location='remote')
    hdu = fits.open(pathhdu)
    scale = hdu[0].header['PIXSCAL1']

    catalog = Table.read(pathcat)

    pos_skycoord = SkyCoord(catalog['l'], catalog['b'], frame='galactic')

    photometry_skycoord = aperture_photometry(
        hdu, SkyCircularAperture(pos_skycoord, 4 * u.arcsec))

    photometry_skycoord_pix = aperture_photometry(
        hdu, SkyCircularAperture(pos_skycoord, 4. / scale * u.pixel))

    assert_allclose(photometry_skycoord['aperture_sum'],
                    photometry_skycoord_pix['aperture_sum'])

    # Photometric unit conversion is needed to match the catalogue
    factor = (1.2 * u.arcsec) ** 2 / u.pixel
    converted_aperture_sum = (photometry_skycoord['aperture_sum'] *
                              factor).to(u.mJy / u.pixel)

    fluxes_catalog = catalog['f4_5'].filled()

    # There shouldn't be large outliers, but some differences is OK, as
    # fluxes_catalog is based on PSF photometry, etc.
    assert_allclose(fluxes_catalog, converted_aperture_sum.value, rtol=1e0)

    assert(np.mean(np.fabs(((fluxes_catalog - converted_aperture_sum.value) /
                            fluxes_catalog))) < 0.1)

    # close the file
    hdu.close()


def test_wcs_based_photometry():
    data = make_4gaussians_image()
    wcs = make_wcs(data.shape)
    hdu = make_imagehdu(data, wcs=wcs)

    # hard wired positions in make_4gaussian_image
    pos_orig_pixel = u.Quantity(([160., 25., 150., 90.],
                                 [70., 40., 25., 60.]), unit=u.pixel)

    pos_skycoord = pixel_to_skycoord(pos_orig_pixel[0], pos_orig_pixel[1], wcs)

    pos_skycoord_s = pos_skycoord[2]

    photometry_skycoord_circ = aperture_photometry(
        hdu, SkyCircularAperture(pos_skycoord, 3 * u.arcsec))
    photometry_skycoord_circ_2 = aperture_photometry(
        hdu, SkyCircularAperture(pos_skycoord, 2 * u.arcsec))
    photometry_skycoord_circ_s = aperture_photometry(
        hdu, SkyCircularAperture(pos_skycoord_s, 3 * u.arcsec))

    assert_allclose(photometry_skycoord_circ['aperture_sum'][2],
                    photometry_skycoord_circ_s['aperture_sum'])

    photometry_skycoord_circ_ann = aperture_photometry(
        hdu, SkyCircularAnnulus(pos_skycoord, 2 * u.arcsec, 3 * u.arcsec))
    photometry_skycoord_circ_ann_s = aperture_photometry(
        hdu, SkyCircularAnnulus(pos_skycoord_s, 2 * u.arcsec, 3 * u.arcsec))

    assert_allclose(photometry_skycoord_circ_ann['aperture_sum'][2],
                    photometry_skycoord_circ_ann_s['aperture_sum'])

    assert_allclose(photometry_skycoord_circ_ann['aperture_sum'],
                    photometry_skycoord_circ['aperture_sum'] -
                    photometry_skycoord_circ_2['aperture_sum'])

    photometry_skycoord_ell = aperture_photometry(
        hdu, SkyEllipticalAperture(pos_skycoord, 3 * u.arcsec,
                                   3.0001 * u.arcsec, 45 * u.arcsec))
    photometry_skycoord_ell_2 = aperture_photometry(
        hdu, SkyEllipticalAperture(pos_skycoord, 2 * u.arcsec,
                                   2.0001 * u.arcsec, 45 * u.arcsec))
    photometry_skycoord_ell_s = aperture_photometry(
        hdu, SkyEllipticalAperture(pos_skycoord_s, 3 * u.arcsec,
                                   3.0001 * u.arcsec, 45 * u.arcsec))
    photometry_skycoord_ell_ann = aperture_photometry(
        hdu, SkyEllipticalAnnulus(pos_skycoord, 2 * u.arcsec, 3 * u.arcsec,
                                  3.0001 * u.arcsec, 45 * u.arcsec))
    photometry_skycoord_ell_ann_s = aperture_photometry(
        hdu, SkyEllipticalAnnulus(pos_skycoord_s, 2 * u.arcsec, 3 * u.arcsec,
                                  3.0001 * u.arcsec, 45 * u.arcsec))

    assert_allclose(photometry_skycoord_ell['aperture_sum'][2],
                    photometry_skycoord_ell_s['aperture_sum'])

    assert_allclose(photometry_skycoord_ell_ann['aperture_sum'][2],
                    photometry_skycoord_ell_ann_s['aperture_sum'])

    assert_allclose(photometry_skycoord_ell['aperture_sum'],
                    photometry_skycoord_circ['aperture_sum'], rtol=5e-3)

    assert_allclose(photometry_skycoord_ell_ann['aperture_sum'],
                    photometry_skycoord_ell['aperture_sum'] -
                    photometry_skycoord_ell_2['aperture_sum'], rtol=1e-4)

    photometry_skycoord_rec = aperture_photometry(
        hdu, SkyRectangularAperture(pos_skycoord,
                                    6 * u.arcsec, 6 * u.arcsec,
                                    0 * u.arcsec),
        method='subpixel', subpixels=20)
    photometry_skycoord_rec_4 = aperture_photometry(
        hdu, SkyRectangularAperture(pos_skycoord,
                                    4 * u.arcsec, 4 * u.arcsec,
                                    0 * u.arcsec),
        method='subpixel', subpixels=20)
    photometry_skycoord_rec_s = aperture_photometry(
        hdu, SkyRectangularAperture(pos_skycoord_s,
                                    6 * u.arcsec, 6 * u.arcsec,
                                    0 * u.arcsec),
        method='subpixel', subpixels=20)
    photometry_skycoord_rec_ann = aperture_photometry(
        hdu, SkyRectangularAnnulus(pos_skycoord, 4 * u.arcsec, 6 * u.arcsec,
                                   6 * u.arcsec, 0 * u.arcsec),
        method='subpixel', subpixels=20)
    photometry_skycoord_rec_ann_s = aperture_photometry(
        hdu, SkyRectangularAnnulus(pos_skycoord_s, 4 * u.arcsec, 6 * u.arcsec,
                                   6 * u.arcsec, 0 * u.arcsec),
        method='subpixel', subpixels=20)

    assert_allclose(photometry_skycoord_rec['aperture_sum'][2],
                    photometry_skycoord_rec_s['aperture_sum'])

    assert np.all(photometry_skycoord_rec['aperture_sum'] >
                  photometry_skycoord_circ['aperture_sum'])

    assert_allclose(photometry_skycoord_rec_ann['aperture_sum'][2],
                    photometry_skycoord_rec_ann_s['aperture_sum'])

    assert_allclose(photometry_skycoord_rec_ann['aperture_sum'],
                    photometry_skycoord_rec['aperture_sum'] -
                    photometry_skycoord_rec_4['aperture_sum'], rtol=1e-4)


def test_basic_circular_aperture_photometry_unit():
    data1 = np.ones((40, 40), dtype=np.float)
    data2 = u.Quantity(data1, unit=u.adu)

    radius = 3
    position = (20, 20)
    true_flux = np.pi * radius * radius
    unit = u.adu

    table1 = aperture_photometry(data1, CircularAperture(position, radius),
                                 unit=unit)
    table2 = aperture_photometry(data2, CircularAperture(position, radius))

    assert_allclose(table1['aperture_sum'].value, true_flux)
    assert_allclose(table2['aperture_sum'].value, true_flux)
    assert table1['aperture_sum'].unit == unit
    assert table2['aperture_sum'].unit == data2.unit == unit


def test_aperture_photometry_with_error_units():
    """Test aperture_photometry when error has units (see #176)."""

    data1 = np.ones((40, 40), dtype=np.float)
    data2 = u.Quantity(data1, unit=u.adu)
    error = u.Quantity(data1, unit=u.adu)
    radius = 3
    true_flux = np.pi * radius * radius
    unit = u.adu
    position = (20, 20)
    table1 = aperture_photometry(data2, CircularAperture(position, radius),
                                 error=error)
    assert_allclose(table1['aperture_sum'].value, true_flux)
    assert_allclose(table1['aperture_sum_err'].value, np.sqrt(true_flux))
    assert table1['aperture_sum'].unit == unit
    assert table1['aperture_sum_err'].unit == unit


def test_aperture_photometry_inputs_with_mask():
    """
    Test that aperture_photometry does not modify the input
    data or error array when a mask is input.
    """

    data = np.ones((5, 5))
    aperture = CircularAperture((2, 2), 2.)
    mask = np.zeros_like(data, dtype=bool)
    data[2, 2] = 100.   # bad pixel
    mask[2, 2] = True
    error = np.sqrt(data)
    data_in = data.copy()
    error_in = error.copy()
    t1 = aperture_photometry(data, aperture, error=error, mask=mask)
    assert_array_equal(data, data_in)
    assert_array_equal(error, error_in)
    assert_allclose(t1['aperture_sum'][0], 11.5663706144)
    t2 = aperture_photometry(data, aperture)
    assert_allclose(t2['aperture_sum'][0], 111.566370614)


TEST_ELLIPSE_EXACT_APERTURES = [(3.469906, 3.923861394, 3.),
                                (0.3834415188257778, 0.3834415188257778, 0.3)]


@pytest.mark.parametrize('x,y,r', TEST_ELLIPSE_EXACT_APERTURES)
def test_ellipse_exact_grid(x, y, r):
    """
    Test elliptical exact aperture photometry on a grid of pixel positions.

    This is a regression test for the bug discovered in this issue:
    https://github.com/astropy/photutils/issues/198
    """

    data = np.ones((10, 10))

    aperture = EllipticalAperture((x, y), r, r, 0.)
    t = aperture_photometry(data, aperture, method='exact')
    actual = t['aperture_sum'][0] / (np.pi * r ** 2)
    assert_allclose(actual, 1)


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nan_inf_mask(value):
    """Test that nans and infs are properly masked [267]."""

    data = np.ones((9, 9))
    mask = np.zeros_like(data, dtype=bool)
    data[4, 4] = value
    mask[4, 4] = True
    radius = 2.
    aper = CircularAperture((4, 4), radius)
    tbl = aperture_photometry(data, aper, mask=mask)
    desired = (np.pi * radius**2) - 1
    assert_allclose(tbl['aperture_sum'], desired)


def test_aperture_partial_overlap():
    data = np.ones((20, 20))
    error = np.ones((20, 20))
    xypos = [(10, 10), (0, 0), (0, 19), (19, 0), (19, 19)]
    r = 5.
    aper = CircularAperture(xypos, r=r)
    tbl = aperture_photometry(data, aper, error=error)
    assert_allclose(tbl['aperture_sum'][0], np.pi * r ** 2)
    assert_array_less(tbl['aperture_sum'][1:], np.pi * r ** 2)

    unit = u.MJy / u.sr
    tbl = aperture_photometry(data * unit, aper, error=error * unit)
    assert_allclose(tbl['aperture_sum'][0].value, np.pi * r ** 2)
    assert_array_less(tbl['aperture_sum'][1:].value, np.pi * r ** 2)
    assert_array_less(tbl['aperture_sum_err'][1:].value, np.pi * r ** 2)
    assert tbl['aperture_sum'].unit == unit
    assert tbl['aperture_sum_err'].unit == unit


def test_pixel_aperture_repr():
    aper = CircularAperture((10, 20), r=3.0)
    a_repr = '<CircularAperture([[10, 20]], r=3.0)>'
    a_str = 'Aperture: CircularAperture\npositions: [[10, 20]]\nr: 3.0'
    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = CircularAnnulus((10, 20), r_in=3.0, r_out=5.0)
    a_repr = '<CircularAnnulus([[10, 20]], r_in=3.0, r_out=5.0)>'
    a_str = ('Aperture: CircularAnnulus\npositions: [[10, 20]]\nr_in: 3.0\n'
             'r_out: 5.0')
    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = EllipticalAperture((10, 20), a=5.0, b=3.0, theta=15.0)
    a_repr = '<EllipticalAperture([[10, 20]], a=5.0, b=3.0, theta=15.0)>'
    a_str = ('Aperture: EllipticalAperture\npositions: [[10, 20]]\n'
             'a: 5.0\nb: 3.0\ntheta: 15.0')
    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = EllipticalAnnulus((10, 20), a_in=4.0, a_out=8.0, b_out=4.0,
                             theta=15.0)
    a_repr = ('<EllipticalAnnulus([[10, 20]], a_in=4.0, a_out=8.0, b_out='
              '4.0, theta=15.0)>')
    a_str = ('Aperture: EllipticalAnnulus\npositions: [[10, 20]]\na_in: '
             '4.0\na_out: 8.0\nb_out: 4.0\ntheta: 15.0')
    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = RectangularAperture((10, 20), w=5.0, h=3.0, theta=15.0)
    a_repr = '<RectangularAperture([[10, 20]], w=5.0, h=3.0, theta=15.0)>'
    a_str = ('Aperture: RectangularAperture\npositions: [[10, 20]]\n'
             'w: 5.0\nh: 3.0\ntheta: 15.0')
    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = RectangularAnnulus((10, 20), w_in=4.0, w_out=8.0, h_out=4.0,
                              theta=15.0)
    a_repr = ('<RectangularAnnulus([[10, 20]], w_in=4.0, w_out=8.0, '
              'h_out=4.0, theta=15.0)>')
    a_str = ('Aperture: RectangularAnnulus\npositions: [[10, 20]]\n'
             'w_in: 4.0\nw_out: 8.0\nh_out: 4.0\ntheta: 15.0')
    assert repr(aper) == a_repr
    assert str(aper) == a_str


def test_sky_aperture_repr():
    s = SkyCoord([1, 2], [3, 4], unit='deg')

    aper = SkyCircularAperture(s, r=3*u.pix)
    if NUMPY_LT_1_14:
        a_repr = ('<SkyCircularAperture(<SkyCoord (ICRS): (ra, dec) in deg\n'
                  '    [( 1.,  3.), ( 2.,  4.)]>, r=3.0 pix)>')
        a_str = ('Aperture: SkyCircularAperture\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [( 1.,  3.), ( 2.,  4.)]>\n'
                 'r: 3.0 pix')
    else:
        a_repr = ('<SkyCircularAperture(<SkyCoord (ICRS): (ra, dec) in deg\n'
                  '    [(1., 3.), (2., 4.)]>, r=3.0 pix)>')
        a_str = ('Aperture: SkyCircularAperture\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
                 'r: 3.0 pix')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyCircularAnnulus(s, r_in=3.*u.pix, r_out=5*u.pix)
    if NUMPY_LT_1_14:
        a_repr = ('<SkyCircularAnnulus(<SkyCoord (ICRS): (ra, dec) in deg\n'
                  '    [( 1.,  3.), ( 2.,  4.)]>, r_in=3.0 pix, '
                  'r_out=5.0 pix)>')
        a_str = ('Aperture: SkyCircularAnnulus\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [( 1.,  3.), ( 2.,  4.)]>\n'
                 'r_in: 3.0 pix\nr_out: 5.0 pix')
    else:
        a_repr = ('<SkyCircularAnnulus(<SkyCoord (ICRS): (ra, dec) in deg\n'
                  '    [(1., 3.), (2., 4.)]>, r_in=3.0 pix, r_out=5.0 pix)>')
        a_str = ('Aperture: SkyCircularAnnulus\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
                 'r_in: 3.0 pix\nr_out: 5.0 pix')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyEllipticalAperture(s, a=3*u.pix, b=5*u.pix, theta=15*u.deg)
    if NUMPY_LT_1_14:
        a_repr = ('<SkyEllipticalAperture(<SkyCoord (ICRS): (ra, dec) in '
                  'deg\n    [( 1.,  3.), ( 2.,  4.)]>, a=3.0 pix, b=5.0 pix,'
                  ' theta=15.0 deg)>')
        a_str = ('Aperture: SkyEllipticalAperture\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [( 1.,  3.), ( 2.,  4.)]>\n'
                 'a: 3.0 pix\nb: 5.0 pix\ntheta: 15.0 deg')
    else:
        a_repr = ('<SkyEllipticalAperture(<SkyCoord (ICRS): (ra, dec) in '
                  'deg\n    [(1., 3.), (2., 4.)]>, a=3.0 pix, b=5.0 pix,'
                  ' theta=15.0 deg)>')
        a_str = ('Aperture: SkyEllipticalAperture\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
                 'a: 3.0 pix\nb: 5.0 pix\ntheta: 15.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyEllipticalAnnulus(s, a_in=3*u.pix, a_out=5*u.pix, b_out=3*u.pix,
                                theta=15*u.deg)
    if NUMPY_LT_1_14:
        a_repr = ('<SkyEllipticalAnnulus(<SkyCoord (ICRS): (ra, dec) in '
                  'deg\n    [( 1.,  3.), ( 2.,  4.)]>, a_in=3.0 pix, '
                  'a_out=5.0 pix, b_out=3.0 pix, theta=15.0 deg)>')
        a_str = ('Aperture: SkyEllipticalAnnulus\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [( 1.,  3.), ( 2.,  4.)]>\n'
                 'a_in: 3.0 pix\na_out: 5.0 pix\nb_out: 3.0 pix\n'
                 'theta: 15.0 deg')
    else:
        a_repr = ('<SkyEllipticalAnnulus(<SkyCoord (ICRS): (ra, dec) in '
                  'deg\n    [(1., 3.), (2., 4.)]>, a_in=3.0 pix, '
                  'a_out=5.0 pix, b_out=3.0 pix, theta=15.0 deg)>')
        a_str = ('Aperture: SkyEllipticalAnnulus\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
                 'a_in: 3.0 pix\na_out: 5.0 pix\nb_out: 3.0 pix\n'
                 'theta: 15.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyRectangularAperture(s, w=3*u.pix, h=5*u.pix, theta=15*u.deg)
    if NUMPY_LT_1_14:
        a_repr = ('<SkyRectangularAperture(<SkyCoord (ICRS): (ra, dec) in '
                  'deg\n    [( 1.,  3.), ( 2.,  4.)]>, w=3.0 pix, h=5.0 pix'
                  ', theta=15.0 deg)>')
        a_str = ('Aperture: SkyRectangularAperture\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [( 1.,  3.), ( 2.,  4.)]>\n'
                 'w: 3.0 pix\nh: 5.0 pix\ntheta: 15.0 deg')
    else:
        a_repr = ('<SkyRectangularAperture(<SkyCoord (ICRS): (ra, dec) in '
                  'deg\n    [(1., 3.), (2., 4.)]>, w=3.0 pix, h=5.0 pix'
                  ', theta=15.0 deg)>')
        a_str = ('Aperture: SkyRectangularAperture\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
                 'w: 3.0 pix\nh: 5.0 pix\ntheta: 15.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyRectangularAnnulus(s, w_in=3*u.pix, w_out=3.4*u.pix,
                                 h_out=5*u.pix, theta=15*u.deg)
    if NUMPY_LT_1_14:
        a_repr = ('<SkyRectangularAnnulus(<SkyCoord (ICRS): (ra, dec) in deg'
                  '\n    [( 1.,  3.), ( 2.,  4.)]>, w_in=3.0 pix, '
                  'w_out=3.4 pix, h_out=5.0 pix, theta=15.0 deg)>')
        a_str = ('Aperture: SkyRectangularAnnulus\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [( 1.,  3.), ( 2.,  4.)]>\n'
                 'w_in: 3.0 pix\nw_out: 3.4 pix\nh_out: 5.0 pix\n'
                 'theta: 15.0 deg')
    else:
        a_repr = ('<SkyRectangularAnnulus(<SkyCoord (ICRS): (ra, dec) in deg'
                  '\n    [(1., 3.), (2., 4.)]>, w_in=3.0 pix, '
                  'w_out=3.4 pix, h_out=5.0 pix, theta=15.0 deg)>')
        a_str = ('Aperture: SkyRectangularAnnulus\npositions: <SkyCoord '
                 '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
                 'w_in: 3.0 pix\nw_out: 3.4 pix\nh_out: 5.0 pix\n'
                 'theta: 15.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str


def test_rectangular_bbox():
    # odd sizes
    width = 7
    height = 3
    a = RectangularAperture((50, 50), w=width, h=height, theta=0)
    assert a.bounding_boxes[0].shape == (height, width)

    a = RectangularAperture((50.5, 50.5), w=width, h=height, theta=0)
    assert a.bounding_boxes[0].shape == (height + 1, width + 1)

    a = RectangularAperture((50, 50), w=width, h=height, theta=90.*np.pi/180.)
    assert a.bounding_boxes[0].shape == (width, height)

    # even sizes
    width = 8
    height = 4
    a = RectangularAperture((50, 50), w=width, h=height, theta=0)
    assert a.bounding_boxes[0].shape == (height + 1, width + 1)

    a = RectangularAperture((50.5, 50.5), w=width, h=height, theta=0)
    assert a.bounding_boxes[0].shape == (height, width)

    a = RectangularAperture((50.5, 50.5), w=width, h=height,
                            theta=90.*np.pi/180.)
    assert a.bounding_boxes[0].shape == (width, height)


def test_elliptical_bbox():
    # integer axes
    a = 7
    b = 3
    ap = EllipticalAperture((50, 50), a=a, b=b, theta=0)
    assert ap.bounding_boxes[0].shape == (2*b + 1, 2*a + 1)

    ap = EllipticalAperture((50.5, 50.5), a=a, b=b, theta=0)
    assert ap.bounding_boxes[0].shape == (2*b, 2*a)

    ap = EllipticalAperture((50, 50), a=a, b=b, theta=90.*np.pi/180.)
    assert ap.bounding_boxes[0].shape == (2*a + 1, 2*b + 1)

    # fractional axes
    a = 7.5
    b = 4.5
    ap = EllipticalAperture((50, 50), a=a, b=b, theta=0)
    assert ap.bounding_boxes[0].shape == (2*b, 2*a)

    ap = EllipticalAperture((50.5, 50.5), a=a, b=b, theta=0)
    assert ap.bounding_boxes[0].shape == (2*b + 1, 2*a + 1)

    ap = EllipticalAperture((50, 50), a=a, b=b, theta=90.*np.pi/180.)
    assert ap.bounding_boxes[0].shape == (2*a, 2*b)


def test_to_sky_pixel():
    data = make_4gaussians_image()
    wcs = make_wcs(data.shape)

    ap = CircularAperture(((12.3, 15.7), (48.19, 98.14)), r=3.14)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.r, ap2.r)

    ap = CircularAnnulus(((12.3, 15.7), (48.19, 98.14)), r_in=3.14,
                         r_out=5.32)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.r_in, ap2.r_in)
    assert_allclose(ap.r_out, ap2.r_out)

    ap = EllipticalAperture(((12.3, 15.7), (48.19, 98.14)), a=3.14, b=5.32,
                            theta=103.*np.pi/180.)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.a, ap2.a)
    assert_allclose(ap.b, ap2.b)
    assert_allclose(ap.theta, ap2.theta)

    ap = EllipticalAnnulus(((12.3, 15.7), (48.19, 98.14)), a_in=3.14,
                           a_out=15.32, b_out=4.89, theta=103.*np.pi/180.)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.a_in, ap2.a_in)
    assert_allclose(ap.a_out, ap2.a_out)
    assert_allclose(ap.b_out, ap2.b_out)
    assert_allclose(ap.theta, ap2.theta)

    ap = RectangularAperture(((12.3, 15.7), (48.19, 98.14)), w=3.14, h=5.32,
                             theta=103.*np.pi/180.)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.w, ap2.w)
    assert_allclose(ap.h, ap2.h)
    assert_allclose(ap.theta, ap2.theta)

    ap = RectangularAnnulus(((12.3, 15.7), (48.19, 98.14)), w_in=3.14,
                            w_out=15.32, h_out=4.89, theta=103.*np.pi/180.)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.w_in, ap2.w_in)
    assert_allclose(ap.w_out, ap2.w_out)
    assert_allclose(ap.h_out, ap2.h_out)
    assert_allclose(ap.theta, ap2.theta)


def test_position_units():
    """Regression test for unit check."""
    pos = (10, 10) * u.pix
    pos = np.sqrt(pos**2)
    ap = CircularAperture(pos, r=3.)
    assert_allclose(ap.positions, np.array([[10, 10]]))


def test_radius_units():
    """Regression test for unit check."""
    pos = SkyCoord(10, 10, unit='deg')
    r = 3.*u.pix
    r = np.sqrt(r**2)
    ap = SkyCircularAperture(pos, r=r)
    assert ap.r.value == 3.0
    assert ap.r.unit == u.pix


def test_scalar_aperture():
    """
    Regression test to check that length-1 aperture list appends a "_0"
    on the column names to be consistent with list inputs.
    """

    data = np.ones((20, 20), dtype=np.float)

    ap = CircularAperture((10, 10), r=3.)
    colnames1 = aperture_photometry(data, ap, error=data).colnames
    assert (colnames1 == ['id', 'xcenter', 'ycenter', 'aperture_sum',
                          'aperture_sum_err'])

    colnames2 = aperture_photometry(data, [ap], error=data).colnames
    assert (colnames2 == ['id', 'xcenter', 'ycenter', 'aperture_sum_0',
                          'aperture_sum_err_0'])

    colnames3 = aperture_photometry(data, [ap, ap], error=data).colnames
    assert (colnames3 == ['id', 'xcenter', 'ycenter', 'aperture_sum_0',
                          'aperture_sum_err_0', 'aperture_sum_1',
                          'aperture_sum_err_1'])
