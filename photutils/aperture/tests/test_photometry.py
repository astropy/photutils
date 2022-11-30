# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photometry module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import Table
from astropy.wcs import WCS
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_array_less)

from photutils.aperture.circle import (CircularAnnulus, CircularAperture,
                                       SkyCircularAnnulus, SkyCircularAperture)
from photutils.aperture.ellipse import (EllipticalAnnulus, EllipticalAperture,
                                        SkyEllipticalAnnulus,
                                        SkyEllipticalAperture)
from photutils.aperture.photometry import aperture_photometry
from photutils.aperture.rectangle import (RectangularAnnulus,
                                          RectangularAperture,
                                          SkyRectangularAnnulus,
                                          SkyRectangularAperture)
from photutils.datasets import (get_path, make_4gaussians_image, make_gwcs,
                                make_wcs)
from photutils.utils._optional_deps import HAS_GWCS, HAS_MATPLOTLIB

APERTURE_CL = [CircularAperture,
               CircularAnnulus,
               EllipticalAperture,
               EllipticalAnnulus,
               RectangularAperture,
               RectangularAnnulus]

TEST_APERTURES = list(zip(APERTURE_CL, ((3.0,),
                                        (3.0, 5.0),
                                        (3.0, 5.0, 1.0),
                                        (3.0, 5.0, 4.0, 12.0 / 5.0, 1.0),
                                        (5, 8, np.pi / 4),
                                        (8, 12, 8, 16.0 / 3.0, np.pi / 8))))


@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_outside_array(aperture_class, params):
    data = np.ones((10, 10), dtype=float)
    aperture = aperture_class((-60, 60), *params)
    fluxtable = aperture_photometry(data, aperture)
    # aperture is fully outside array:
    assert np.isnan(fluxtable['aperture_sum'])


@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_inside_array_simple(aperture_class, params):
    data = np.ones((40, 40), dtype=float)
    aperture = aperture_class((20.0, 20.0), *params)
    table1 = aperture_photometry(data, aperture, method='center',
                                 subpixels=10)
    table2 = aperture_photometry(data, aperture, method='subpixel',
                                 subpixels=10)
    table3 = aperture_photometry(data, aperture, method='exact', subpixels=10)
    true_flux = aperture.area
    assert table1['aperture_sum'] < table3['aperture_sum']

    if not isinstance(aperture, (RectangularAperture, RectangularAnnulus)):
        assert_allclose(table3['aperture_sum'], true_flux)
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_aperture_plots(aperture_class, params):
    # This test should run without any errors, and there is no return
    # value.
    # TODO: check the content of the plot
    aperture = aperture_class((20.0, 20.0), *params)
    aperture.plot()


def test_aperture_pixel_positions():
    pos1 = (10, 20)
    pos2 = [(10, 20)]
    r = 3
    ap1 = CircularAperture(pos1, r)
    ap2 = CircularAperture(pos2, r)
    assert not np.array_equal(ap1.positions, ap2.positions)


class BaseTestAperturePhotometry:
    def test_array_error(self):
        # Array error
        error = np.ones(self.data.shape, dtype=float)
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
        self.data = np.ones((40, 40), dtype=float)
        position = (20.0, 20.0)
        r = 10.0
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.true_flux = self.area


class TestCircularArray(BaseTestAperturePhotometry):
    def setup_class(self):
        self.data = np.ones((40, 40), dtype=float)
        position = ((20.0, 20.0), (25.0, 25.0))
        r = 10.0
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.area = np.array((self.area,) * 2)
        self.true_flux = self.area


class TestCircularAnnulus(BaseTestAperturePhotometry):
    def setup_class(self):
        self.data = np.ones((40, 40), dtype=float)
        position = (20.0, 20.0)
        r_in = 8.0
        r_out = 10.0
        self.aperture = CircularAnnulus(position, r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.true_flux = self.area


class TestCircularAnnulusArray(BaseTestAperturePhotometry):
    def setup_class(self):
        self.data = np.ones((40, 40), dtype=float)
        position = ((20.0, 20.0), (25.0, 25.0))
        r_in = 8.0
        r_out = 10.0
        self.aperture = CircularAnnulus(position, r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.area = np.array((self.area,) * 2)
        self.true_flux = self.area


class TestElliptical(BaseTestAperturePhotometry):
    def setup_class(self):
        self.data = np.ones((40, 40), dtype=float)
        position = (20.0, 20.0)
        a = 10.0
        b = 5.0
        theta = -np.pi / 4.0
        self.aperture = EllipticalAperture(position, a, b, theta=theta)
        self.area = np.pi * a * b
        self.true_flux = self.area


class TestEllipticalAnnulus(BaseTestAperturePhotometry):
    def setup_class(self):
        self.data = np.ones((40, 40), dtype=float)
        position = (20.0, 20.0)
        a_in = 5.0
        a_out = 8.0
        b_out = 5.0
        theta = -np.pi / 4.0
        self.aperture = EllipticalAnnulus(position, a_in, a_out, b_out,
                                          theta=theta)
        self.area = (np.pi * (a_out * b_out)
                     - np.pi * (a_in * b_out * a_in / a_out))
        self.true_flux = self.area


class TestRectangularAperture(BaseTestAperturePhotometry):
    def setup_class(self):
        self.data = np.ones((40, 40), dtype=float)
        position = (20.0, 20.0)
        h = 5.0
        w = 8.0
        theta = np.pi / 4.0
        self.aperture = RectangularAperture(position, w, h, theta=theta)
        self.area = h * w
        self.true_flux = self.area


class TestRectangularAnnulus(BaseTestAperturePhotometry):
    def setup_class(self):
        self.data = np.ones((40, 40), dtype=float)
        position = (20.0, 20.0)
        h_out = 8.0
        w_in = 8.0
        w_out = 12.0
        h_in = w_in * h_out / w_out
        theta = np.pi / 8.0
        self.aperture = RectangularAnnulus(position, w_in, w_out, h_out,
                                           theta=theta)
        self.area = h_out * w_out - h_in * w_in
        self.true_flux = self.area


class TestMaskedSkipCircular(BaseTestAperturePhotometry):
    def setup_class(self):
        self.data = np.ones((40, 40), dtype=float)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[20, 20] = True
        position = (20.0, 20.0)
        r = 10.0
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.true_flux = self.area - 1


class BaseTestDifferentData:
    def test_basic_circular_aperture_photometry(self):
        aperture = CircularAperture(self.position, self.radius)
        table = aperture_photometry(self.data, aperture,
                                    method='exact')

        assert_allclose(table['aperture_sum'].value, self.true_flux)
        assert table['aperture_sum'].unit, self.fluxunit

        assert np.all(table['xcenter'].value
                      == np.transpose(self.position)[0])
        assert np.all(table['ycenter'].value
                      == np.transpose(self.position)[1])


class TestInputNDData(BaseTestDifferentData):
    def setup_class(self):
        data = np.ones((40, 40), dtype=float)
        self.data = NDData(data, unit=u.adu)
        self.radius = 3
        self.position = [(20, 20), (30, 30)]
        self.true_flux = np.pi * self.radius * self.radius
        self.fluxunit = u.adu


@pytest.mark.remote_data
def test_wcs_based_photometry_to_catalog():
    pathcat = get_path('spitzer_example_catalog.xml', location='remote')
    pathhdu = get_path('spitzer_example_image.fits', location='remote')
    hdu = fits.open(pathhdu)
    data = u.Quantity(hdu[0].data, unit=hdu[0].header['BUNIT'])
    wcs = WCS(hdu[0].header)

    catalog = Table.read(pathcat)

    pos_skycoord = SkyCoord(catalog['l'], catalog['b'], frame='galactic')

    photometry_skycoord = aperture_photometry(
        data, SkyCircularAperture(pos_skycoord, 4 * u.arcsec), wcs=wcs)

    # Photometric unit conversion is needed to match the catalog
    factor = (1.2 * u.arcsec) ** 2 / u.pixel
    converted_aperture_sum = (photometry_skycoord['aperture_sum']
                              * factor).to(u.mJy / u.pixel)

    fluxes_catalog = catalog['f4_5'].filled()

    # There shouldn't be large outliers, but some differences is OK, as
    # fluxes_catalog is based on PSF photometry, etc.
    assert_allclose(fluxes_catalog, converted_aperture_sum.value, rtol=1e0)

    assert np.mean(np.fabs((fluxes_catalog - converted_aperture_sum.value)
                           / fluxes_catalog)) < 0.1

    # close the file
    hdu.close()


def test_wcs_based_photometry():
    data = make_4gaussians_image()
    wcs = make_wcs(data.shape)

    # hard wired positions in make_4gaussian_image
    pos_orig_pixel = u.Quantity(([160.0, 25.0, 150.0, 90.0],
                                 [70.0, 40.0, 25.0, 60.0]), unit=u.pixel)

    pos_skycoord = wcs.pixel_to_world(pos_orig_pixel[0], pos_orig_pixel[1])
    pos_skycoord_s = pos_skycoord[2]

    photometry_skycoord_circ = aperture_photometry(
        data, SkyCircularAperture(pos_skycoord, 3 * u.arcsec), wcs=wcs)
    photometry_skycoord_circ_2 = aperture_photometry(
        data, SkyCircularAperture(pos_skycoord, 2 * u.arcsec), wcs=wcs)
    photometry_skycoord_circ_s = aperture_photometry(
        data, SkyCircularAperture(pos_skycoord_s, 3 * u.arcsec), wcs=wcs)

    assert_allclose(photometry_skycoord_circ['aperture_sum'][2],
                    photometry_skycoord_circ_s['aperture_sum'])

    photometry_skycoord_circ_ann = aperture_photometry(
        data, SkyCircularAnnulus(pos_skycoord, 2 * u.arcsec, 3 * u.arcsec),
        wcs=wcs)
    photometry_skycoord_circ_ann_s = aperture_photometry(
        data, SkyCircularAnnulus(pos_skycoord_s, 2 * u.arcsec, 3 * u.arcsec),
        wcs=wcs)

    assert_allclose(photometry_skycoord_circ_ann['aperture_sum'][2],
                    photometry_skycoord_circ_ann_s['aperture_sum'])

    assert_allclose(photometry_skycoord_circ_ann['aperture_sum'],
                    photometry_skycoord_circ['aperture_sum']
                    - photometry_skycoord_circ_2['aperture_sum'])

    photometry_skycoord_ell = aperture_photometry(
        data, SkyEllipticalAperture(pos_skycoord, 3 * u.arcsec,
                                    3.0001 * u.arcsec, theta=45 * u.arcsec),
        wcs=wcs)
    photometry_skycoord_ell_2 = aperture_photometry(
        data, SkyEllipticalAperture(pos_skycoord, 2 * u.arcsec,
                                    2.0001 * u.arcsec, theta=45 * u.arcsec),
        wcs=wcs)
    photometry_skycoord_ell_s = aperture_photometry(
        data, SkyEllipticalAperture(pos_skycoord_s, 3 * u.arcsec,
                                    3.0001 * u.arcsec, theta=45 * u.arcsec),
        wcs=wcs)
    photometry_skycoord_ell_ann = aperture_photometry(
        data, SkyEllipticalAnnulus(pos_skycoord, 2 * u.arcsec, 3 * u.arcsec,
                                   3.0001 * u.arcsec, theta=45 * u.arcsec),
        wcs=wcs)
    photometry_skycoord_ell_ann_s = aperture_photometry(
        data, SkyEllipticalAnnulus(pos_skycoord_s, 2 * u.arcsec, 3 * u.arcsec,
                                   3.0001 * u.arcsec, theta=45 * u.arcsec),
        wcs=wcs)

    assert_allclose(photometry_skycoord_ell['aperture_sum'][2],
                    photometry_skycoord_ell_s['aperture_sum'])

    assert_allclose(photometry_skycoord_ell_ann['aperture_sum'][2],
                    photometry_skycoord_ell_ann_s['aperture_sum'])

    assert_allclose(photometry_skycoord_ell['aperture_sum'],
                    photometry_skycoord_circ['aperture_sum'], rtol=5e-3)

    assert_allclose(photometry_skycoord_ell_ann['aperture_sum'],
                    photometry_skycoord_ell['aperture_sum']
                    - photometry_skycoord_ell_2['aperture_sum'], rtol=1e-4)

    photometry_skycoord_rec = aperture_photometry(
        data, SkyRectangularAperture(pos_skycoord,
                                     6 * u.arcsec, 6 * u.arcsec,
                                     0 * u.arcsec),
        method='subpixel', subpixels=20, wcs=wcs)
    photometry_skycoord_rec_4 = aperture_photometry(
        data, SkyRectangularAperture(pos_skycoord,
                                     4 * u.arcsec, 4 * u.arcsec,
                                     0 * u.arcsec),
        method='subpixel', subpixels=20, wcs=wcs)
    photometry_skycoord_rec_s = aperture_photometry(
        data, SkyRectangularAperture(pos_skycoord_s,
                                     6 * u.arcsec, 6 * u.arcsec,
                                     0 * u.arcsec),
        method='subpixel', subpixels=20, wcs=wcs)
    photometry_skycoord_rec_ann = aperture_photometry(
        data, SkyRectangularAnnulus(pos_skycoord, 4 * u.arcsec, 6 * u.arcsec,
                                    6 * u.arcsec, theta=0 * u.arcsec),
        method='subpixel', subpixels=20, wcs=wcs)
    photometry_skycoord_rec_ann_s = aperture_photometry(
        data, SkyRectangularAnnulus(pos_skycoord_s, 4 * u.arcsec,
                                    6 * u.arcsec, 6 * u.arcsec,
                                    theta=0 * u.arcsec),
        method='subpixel', subpixels=20, wcs=wcs)

    assert_allclose(photometry_skycoord_rec['aperture_sum'][2],
                    photometry_skycoord_rec_s['aperture_sum'])

    assert np.all(photometry_skycoord_rec['aperture_sum']
                  > photometry_skycoord_circ['aperture_sum'])

    assert_allclose(photometry_skycoord_rec_ann['aperture_sum'][2],
                    photometry_skycoord_rec_ann_s['aperture_sum'])

    assert_allclose(photometry_skycoord_rec_ann['aperture_sum'],
                    photometry_skycoord_rec['aperture_sum']
                    - photometry_skycoord_rec_4['aperture_sum'], rtol=1e-4)


def test_basic_circular_aperture_photometry_unit():
    radius = 3
    true_flux = np.pi * radius * radius
    aper = CircularAperture((12, 12), radius)

    data1 = np.ones((25, 25), dtype=float)
    table1 = aperture_photometry(data1, aper)
    assert_allclose(table1['aperture_sum'], true_flux)

    unit = u.adu
    data2 = u.Quantity(data1 * unit)
    table2 = aperture_photometry(data2, aper)
    assert_allclose(table2['aperture_sum'].value, true_flux)
    assert table2['aperture_sum'].unit == data2.unit == unit

    error1 = np.ones((25, 25))
    with pytest.raises(ValueError):
        # data has unit, but error does not
        aperture_photometry(data2, aper, error=error1)

    error2 = u.Quantity(error1 * u.Jy)
    with pytest.raises(ValueError):
        # data and error have different units
        aperture_photometry(data2, aper, error=error2)


def test_aperture_photometry_with_error_units():
    """Test aperture_photometry when error has units (see #176)."""

    data1 = np.ones((40, 40), dtype=float)
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
    aperture = CircularAperture((2, 2), 2.0)
    mask = np.zeros_like(data, dtype=bool)
    data[2, 2] = 100.0  # bad pixel
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


TEST_ELLIPSE_EXACT_APERTURES = [(3.469906, 3.923861394, 3.0),
                                (0.3834415188257778, 0.3834415188257778, 0.3)]


@pytest.mark.parametrize('x,y,r', TEST_ELLIPSE_EXACT_APERTURES)
def test_ellipse_exact_grid(x, y, r):
    """
    Test elliptical exact aperture photometry on a grid of pixel positions.

    This is a regression test for the bug discovered in this issue:
    https://github.com/astropy/photutils/issues/198
    """

    data = np.ones((10, 10))

    aperture = EllipticalAperture((x, y), r, r, 0.0)
    t = aperture_photometry(data, aperture, method='exact')
    actual = t['aperture_sum'][0] / (np.pi * r**2)
    assert_allclose(actual, 1)


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nan_inf_mask(value):
    """Test that nans and infs are properly masked [267]."""

    data = np.ones((9, 9))
    mask = np.zeros_like(data, dtype=bool)
    data[4, 4] = value
    mask[4, 4] = True
    radius = 2.0
    aper = CircularAperture((4, 4), radius)
    tbl = aperture_photometry(data, aper, mask=mask)
    desired = (np.pi * radius**2) - 1
    assert_allclose(tbl['aperture_sum'], desired)


def test_aperture_partial_overlap():
    data = np.ones((20, 20))
    error = np.ones((20, 20))
    xypos = [(10, 10), (0, 0), (0, 19), (19, 0), (19, 19)]
    r = 5.0
    aper = CircularAperture(xypos, r=r)
    tbl = aperture_photometry(data, aper, error=error)
    assert_allclose(tbl['aperture_sum'][0], np.pi * r**2)
    assert_array_less(tbl['aperture_sum'][1:], np.pi * r**2)

    unit = u.MJy / u.sr
    tbl = aperture_photometry(data * unit, aper, error=error * unit)
    assert_allclose(tbl['aperture_sum'][0].value, np.pi * r**2)
    assert_array_less(tbl['aperture_sum'][1:].value, np.pi * r**2)
    assert_array_less(tbl['aperture_sum_err'][1:].value, np.pi * r**2)
    assert tbl['aperture_sum'].unit == unit
    assert tbl['aperture_sum_err'].unit == unit


def test_pixel_aperture_repr():
    aper = CircularAperture((10, 20), r=3.0)
    assert '<CircularAperture(' in repr(aper)
    assert 'Aperture: CircularAperture' in str(aper)

    aper = CircularAnnulus((10, 20), r_in=3.0, r_out=5.0)
    assert '<CircularAnnulus(' in repr(aper)
    assert 'Aperture: CircularAnnulus' in str(aper)

    aper = EllipticalAperture((10, 20), a=5.0, b=3.0, theta=15.0)
    assert '<EllipticalAperture(' in repr(aper)
    assert 'Aperture: EllipticalAperture' in str(aper)

    aper = EllipticalAnnulus((10, 20), a_in=4.0, a_out=8.0, b_out=4.0,
                             theta=15.0)
    assert '<EllipticalAnnulus(' in repr(aper)
    assert 'Aperture: EllipticalAnnulus' in str(aper)

    aper = RectangularAperture((10, 20), w=5.0, h=3.0, theta=15.0)
    assert '<RectangularAperture(' in repr(aper)
    assert 'Aperture: RectangularAperture' in str(aper)

    aper = RectangularAnnulus((10, 20), w_in=4.0, w_out=8.0, h_out=4.0,
                              theta=15.0)
    assert '<RectangularAnnulus(' in repr(aper)
    assert 'Aperture: RectangularAnnulus' in str(aper)


def test_sky_aperture_repr():
    s = SkyCoord([1, 2], [3, 4], unit='deg')

    aper = SkyCircularAperture(s, r=3 * u.deg)
    a_repr = ('<SkyCircularAperture(<SkyCoord (ICRS): (ra, dec) in deg\n'
              '    [(1., 3.), (2., 4.)]>, r=3.0 deg)>')
    a_str = ('Aperture: SkyCircularAperture\npositions: <SkyCoord '
             '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
             'r: 3.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyCircularAnnulus(s, r_in=3.0 * u.deg, r_out=5 * u.deg)
    a_repr = ('<SkyCircularAnnulus(<SkyCoord (ICRS): (ra, dec) in deg\n'
              '    [(1., 3.), (2., 4.)]>, r_in=3.0 deg, r_out=5.0 deg)>')
    a_str = ('Aperture: SkyCircularAnnulus\npositions: <SkyCoord '
             '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
             'r_in: 3.0 deg\nr_out: 5.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyEllipticalAperture(s, a=3 * u.deg, b=5 * u.deg, theta=15 * u.deg)
    a_repr = ('<SkyEllipticalAperture(<SkyCoord (ICRS): (ra, dec) in '
              'deg\n    [(1., 3.), (2., 4.)]>, a=3.0 deg, b=5.0 deg, '
              'theta=15.0 deg)>')
    a_str = ('Aperture: SkyEllipticalAperture\npositions: <SkyCoord '
             '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
             'a: 3.0 deg\nb: 5.0 deg\ntheta: 15.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyEllipticalAnnulus(s, a_in=3 * u.deg, a_out=5 * u.deg,
                                b_out=3 * u.deg, theta=15 * u.deg)
    a_repr = ('<SkyEllipticalAnnulus(<SkyCoord (ICRS): (ra, dec) in '
              'deg\n    [(1., 3.), (2., 4.)]>, a_in=3.0 deg, '
              'a_out=5.0 deg, b_in=1.8 deg, b_out=3.0 deg, '
              'theta=15.0 deg)>')
    a_str = ('Aperture: SkyEllipticalAnnulus\npositions: <SkyCoord '
             '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
             'a_in: 3.0 deg\na_out: 5.0 deg\nb_in: 1.8 deg\n'
             'b_out: 3.0 deg\ntheta: 15.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyRectangularAperture(s, w=3 * u.deg, h=5 * u.deg,
                                  theta=15 * u.deg)
    a_repr = ('<SkyRectangularAperture(<SkyCoord (ICRS): (ra, dec) in '
              'deg\n    [(1., 3.), (2., 4.)]>, w=3.0 deg, h=5.0 deg'
              ', theta=15.0 deg)>')
    a_str = ('Aperture: SkyRectangularAperture\npositions: <SkyCoord '
             '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
             'w: 3.0 deg\nh: 5.0 deg\ntheta: 15.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str

    aper = SkyRectangularAnnulus(s, w_in=5 * u.deg, w_out=10 * u.deg,
                                 h_out=6 * u.deg, theta=15 * u.deg)
    a_repr = ('<SkyRectangularAnnulus(<SkyCoord (ICRS): (ra, dec) in deg'
              '\n    [(1., 3.), (2., 4.)]>, w_in=5.0 deg, '
              'w_out=10.0 deg, h_in=3.0 deg, h_out=6.0 deg, '
              'theta=15.0 deg)>')
    a_str = ('Aperture: SkyRectangularAnnulus\npositions: <SkyCoord '
             '(ICRS): (ra, dec) in deg\n    [(1., 3.), (2., 4.)]>\n'
             'w_in: 5.0 deg\nw_out: 10.0 deg\nh_in: 3.0 deg\n'
             'h_out: 6.0 deg\ntheta: 15.0 deg')

    assert repr(aper) == a_repr
    assert str(aper) == a_str


def test_rectangular_bbox():
    # test odd sizes
    width = 7
    height = 3
    a = RectangularAperture((50, 50), w=width, h=height, theta=0)
    assert a.bbox.shape == (height, width)

    a = RectangularAperture((50.5, 50.5), w=width, h=height, theta=0)
    assert a.bbox.shape == (height + 1, width + 1)

    a = RectangularAperture((50, 50), w=width, h=height,
                            theta=90.0 * np.pi / 180.0)
    assert a.bbox.shape == (width, height)

    # test even sizes
    width = 8
    height = 4
    a = RectangularAperture((50, 50), w=width, h=height, theta=0)
    assert a.bbox.shape == (height + 1, width + 1)

    a = RectangularAperture((50.5, 50.5), w=width, h=height, theta=0)
    assert a.bbox.shape == (height, width)

    a = RectangularAperture((50.5, 50.5), w=width, h=height,
                            theta=90.0 * np.pi / 180.0)
    assert a.bbox.shape == (width, height)


def test_elliptical_bbox():
    # integer axes
    a = 7
    b = 3
    ap = EllipticalAperture((50, 50), a=a, b=b, theta=0)
    assert ap.bbox.shape == (2 * b + 1, 2 * a + 1)

    ap = EllipticalAperture((50.5, 50.5), a=a, b=b, theta=0)
    assert ap.bbox.shape == (2 * b, 2 * a)

    ap = EllipticalAperture((50, 50), a=a, b=b, theta=90.0 * np.pi / 180.0)
    assert ap.bbox.shape == (2 * a + 1, 2 * b + 1)

    # fractional axes
    a = 7.5
    b = 4.5
    ap = EllipticalAperture((50, 50), a=a, b=b, theta=0)
    assert ap.bbox.shape == (2 * b, 2 * a)

    ap = EllipticalAperture((50.5, 50.5), a=a, b=b, theta=0)
    assert ap.bbox.shape == (2 * b + 1, 2 * a + 1)

    ap = EllipticalAperture((50, 50), a=a, b=b, theta=90.0 * np.pi / 180.0)
    assert ap.bbox.shape == (2 * a, 2 * b)


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
@pytest.mark.parametrize('wcs_type', ('wcs', 'gwcs'))
def test_to_sky_pixel(wcs_type):
    data = make_4gaussians_image()

    if wcs_type == 'wcs':
        wcs = make_wcs(data.shape)
    elif wcs_type == 'gwcs':
        wcs = make_gwcs(data.shape)

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
                            theta=103.0 * np.pi / 180.0)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.a, ap2.a)
    assert_allclose(ap.b, ap2.b)
    assert_allclose(ap.theta, ap2.theta)

    ap = EllipticalAnnulus(((12.3, 15.7), (48.19, 98.14)), a_in=3.14,
                           a_out=15.32, b_out=4.89,
                           theta=103.0 * np.pi / 180.0)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.a_in, ap2.a_in)
    assert_allclose(ap.a_out, ap2.a_out)
    assert_allclose(ap.b_out, ap2.b_out)
    assert_allclose(ap.theta, ap2.theta)

    ap = RectangularAperture(((12.3, 15.7), (48.19, 98.14)), w=3.14, h=5.32,
                             theta=103.0 * np.pi / 180.0)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.w, ap2.w)
    assert_allclose(ap.h, ap2.h)
    assert_allclose(ap.theta, ap2.theta)

    ap = RectangularAnnulus(((12.3, 15.7), (48.19, 98.14)), w_in=3.14,
                            w_out=15.32, h_out=4.89,
                            theta=103.0 * np.pi / 180.0)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.w_in, ap2.w_in)
    assert_allclose(ap.w_out, ap2.w_out)
    assert_allclose(ap.h_out, ap2.h_out)
    assert_allclose(ap.theta, ap2.theta)


def test_scalar_aperture():
    """
    Regression test to check that length-1 aperture list appends a "_0"
    on the column names to be consistent with list inputs.
    """

    data = np.ones((20, 20), dtype=float)

    ap = CircularAperture((10, 10), r=3.0)
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


def test_nan_in_bbox():
    """
    Regression test that non-finite data values outside of the aperture
    mask but within the bounding box do not affect the photometry.
    """

    data1 = np.ones((101, 101))
    data2 = data1.copy()
    data1[33, 33] = np.nan
    data1[67, 67] = np.inf
    data1[33, 67] = -np.inf
    data1[22, 22] = np.nan
    data1[22, 23] = np.inf
    error = data1.copy()

    aper1 = CircularAperture((50, 50), r=20.0)
    aper2 = CircularAperture((5, 5), r=20.0)

    tbl1 = aperture_photometry(data1, aper1, error=error)
    tbl2 = aperture_photometry(data2, aper1, error=error)
    assert_allclose(tbl1['aperture_sum'], tbl2['aperture_sum'])
    assert_allclose(tbl1['aperture_sum_err'], tbl2['aperture_sum_err'])

    tbl3 = aperture_photometry(data1, aper2, error=error)
    tbl4 = aperture_photometry(data2, aper2, error=error)
    assert_allclose(tbl3['aperture_sum'], tbl4['aperture_sum'])
    assert_allclose(tbl3['aperture_sum_err'], tbl4['aperture_sum_err'])


def test_scalar_skycoord():
    """
    Regression test to check that scalar SkyCoords are added to the table
    as a length-1 SkyCoord array.
    """

    data = make_4gaussians_image()
    wcs = make_wcs(data.shape)
    skycoord = wcs.pixel_to_world(90, 60)
    aper = SkyCircularAperture(skycoord, r=0.1 * u.arcsec)
    tbl = aperture_photometry(data, aper, wcs=wcs)
    assert isinstance(tbl['sky_center'], SkyCoord)


def test_nddata_input():
    data = np.arange(400).reshape((20, 20))
    error = np.sqrt(data)
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:13, 8:13] = True
    unit = 'adu'
    wcs = make_wcs(data.shape)
    skycoord = wcs.pixel_to_world(10, 10)
    aper = SkyCircularAperture(skycoord, r=0.7 * u.arcsec)

    tbl1 = aperture_photometry(data * u.adu, aper, error=error * u.adu,
                               mask=mask, wcs=wcs)

    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty, mask=mask, wcs=wcs,
                    unit=unit)
    tbl2 = aperture_photometry(nddata, aper)

    for column in tbl1.columns:
        if column == 'sky_center':  # cannot test SkyCoord equality
            continue
        assert_allclose(tbl1[column], tbl2[column])
