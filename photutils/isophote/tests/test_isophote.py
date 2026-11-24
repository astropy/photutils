# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the isophote module.
"""

import copy
import pickle  # nosec B403
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import MaskedColumn, QTable, Table
from astropy.utils.masked import Masked
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from photutils.aperture import EllipticalAperture
from photutils.datasets.load import _get_path
from photutils.isophote.ellipse import Ellipse, _central_isophote
from photutils.isophote.fitter import _EllipseFitter
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.isophote import (_ISOPHOTE_PROPERTIES, Isophote,
                                         IsophoteList, _make_property)
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning

DEFAULT_FIX = np.array([False, False, False, False])

RENAMED = {'intens': 'intensity', 'int_err': 'intensity_err',
           'eps': 'ellipticity', 'ellip_err': 'ellipticity_err',
           'grad': 'gradient', 'pix_stddev': 'pixel_stddev',
           'sarea': 'sector_area', 'n_flag': 'n_flagged',
           'nflag': 'n_flagged'}


@pytest.mark.remote_data
class TestIsophote:
    def setup_class(self):
        path = _get_path('isophote/M51.fits', location='photutils-datasets',
                         cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()

    def test_fit(self):
        # Low noise image, fitted perfectly by sample
        data = make_test_image(noise=1.0e-10, seed=0)
        sample = EllipseSample(data, 40)
        fitter = _EllipseFitter(sample)
        iso = fitter.fit(maxiters=400)

        assert iso.valid
        assert iso.stop_code in (0, 2)

        # Fitted values
        assert iso.intensity <= 201.0
        assert iso.intensity >= 199.0
        assert iso.intensity_err <= 0.0010
        assert iso.intensity_err >= 0.0009
        assert iso.pixel_stddev <= 0.03
        assert iso.pixel_stddev >= 0.02
        assert abs(iso.gradient) <= 4.25
        assert abs(iso.gradient) >= 4.20

        # Integrals
        assert iso.total_flux_ellipse <= 1.85e6
        assert iso.total_flux_ellipse >= 1.82e6
        assert iso.total_flux_circle <= 2.025e6
        assert iso.total_flux_circle >= 2.022e6

        # Deviations from perfect ellipticity. The amplitudes are NaN
        # if the upper harmonic fit failed.
        assert np.isnan(iso.a3) or abs(iso.a3) <= 0.01
        assert np.isnan(iso.b3) or abs(iso.b3) <= 0.01
        assert np.isnan(iso.a4) or abs(iso.a4) <= 0.01
        assert np.isnan(iso.b4) or abs(iso.b4) <= 0.01

    def test_m51(self):
        sample = EllipseSample(self.data, 21.44)
        fitter = _EllipseFitter(sample)
        iso = fitter.fit()

        assert iso.valid
        assert iso.stop_code in (0, 2)

        # Geometry
        g = iso.sample.geometry
        assert g.x0 >= (257 - 1.5)  # position within 1.5 pixel
        assert g.x0 <= (257 + 1.5)
        assert g.y0 >= (259 - 1.5)
        assert g.y0 <= (259 + 2.0)
        assert g.ellipticity >= (0.19 - 0.05)  # ellipticity within 0.05
        assert g.ellipticity <= (0.19 + 0.05)
        orientation = g.orientation.to_value(u.rad)
        assert orientation >= (0.62 - 0.05)  # orientation within 5 deg
        assert orientation <= (0.62 + 0.05)

        # Fitted values
        assert_allclose(iso.intensity, 686.0, atol=0.1)
        assert_allclose(iso.rms, 82.96, atol=0.01)
        assert_allclose(iso.intensity_err, 7.61, atol=0.01)
        assert_allclose(iso.pixel_stddev, 117.3, atol=0.1)
        assert_allclose(iso.gradient, -35.28, atol=0.1)

        # Integrals
        assert iso.total_flux_ellipse <= 1.20e6
        assert iso.total_flux_ellipse >= 1.19e6
        assert iso.total_flux_circle <= 1.38e6
        assert iso.total_flux_circle >= 1.36e6

        # Deviations from perfect ellipticity. The amplitudes are NaN
        # if the upper harmonic fit failed.
        assert np.isnan(iso.a3) or abs(iso.a3) <= 0.05
        assert np.isnan(iso.b3) or abs(iso.b3) <= 0.05
        assert np.isnan(iso.a4) or abs(iso.a4) <= 0.05
        assert np.isnan(iso.b4) or abs(iso.b4) <= 0.05

    def test_m51_niter(self):
        # Compare with old STSDAS task. In this task, the
        # default for the starting value of SMA is 10. It
        # fits with 20 iterations, which is the minimum number of
        # iterations (2 * minit) for the first isophote.
        sample = EllipseSample(self.data, 10)
        fitter = _EllipseFitter(sample)
        iso = fitter.fit(miniters=20)

        assert iso.valid
        assert iso.stop_code == 0
        assert iso.n_iter == 20


@pytest.mark.parametrize('sma', [-1.0, np.nan])
def test_isophote_invalid_sma(sma):
    """
    Test that an isophote cannot be created from a sample with a
    negative or NaN semimajor axis length.

    The geometry and the sample reject such a length, so it is set on
    the geometry attribute after they are created.
    """
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 10.0)
    sample.geometry.sma = sma
    match = 'sma of the sample must be nonnegative'
    with pytest.raises(ValueError, match=match):
        Isophote(sample, 0, valid=False, stop_code=3)


@pytest.mark.filterwarnings('ignore::photutils.utils.exceptions.'
                            'PhotutilsDeprecationWarning')
def test_isophote_comparisons():
    data = make_test_image(seed=0)
    sma1 = 40.0
    sma2 = 100.0
    k = 5
    sample0 = EllipseSample(data, sma1 + k)
    sample1 = EllipseSample(data, sma1 + k)
    sample2 = EllipseSample(data, sma2 + k)
    sample0._update(fixed_parameters=DEFAULT_FIX)
    sample1._update(fixed_parameters=DEFAULT_FIX)
    sample2._update(fixed_parameters=DEFAULT_FIX)
    iso0 = Isophote(sample0, k, valid=True, stop_code=0)
    iso1 = Isophote(sample1, k, valid=True, stop_code=0)
    iso2 = Isophote(sample2, k, valid=True, stop_code=0)

    assert iso1 < iso2
    assert iso2 > iso1
    assert iso1 <= iso2
    assert iso2 >= iso1
    assert iso1 != iso2
    assert iso0 == iso1

    # Objects without a semimajor axis length cannot be ordered and
    # are never equal to an isophote.
    match = 'not supported between instances'
    for other in (sample1, None, 1.0):
        with pytest.raises(TypeError, match=match):
            assert iso1 < other
        with pytest.raises(TypeError, match=match):
            assert iso1 > other
        with pytest.raises(TypeError, match=match):
            assert iso1 <= other
        with pytest.raises(TypeError, match=match):
            assert iso1 >= other
        assert iso1 != other
        assert not iso1 == other  # noqa: SIM201


def test_fix_geometry():
    """
    Test that a sample re-extracted after ``_fix_geometry`` lies on the
    new ellipse.
    """
    data = make_test_image(seed=0)
    geometry1 = EllipseGeometry(256.0, 256.0, 20.0, 0.1, 0.0 * u.rad)
    geometry2 = EllipseGeometry(257.0, 255.0, 20.0, 0.5, 1.0 * u.rad)
    sample1 = EllipseSample(data, 20.0, geometry=geometry1)
    sample2 = EllipseSample(data, 20.0, geometry=geometry2)
    sample1._update(fixed_parameters=DEFAULT_FIX)
    sample2._update(fixed_parameters=DEFAULT_FIX)
    iso1 = Isophote(sample1, 0, valid=True, stop_code=0)
    iso2 = Isophote(sample2, 0, valid=True, stop_code=0)

    iso1._fix_geometry(iso2)
    sample1._values = None
    sample1._update(fixed_parameters=DEFAULT_FIX)

    angles, radii = sample1.angles, sample1.radii
    assert_allclose(radii, geometry2.radius(angles))
    assert_allclose(sample1.intensities, sample2.intensities)
    assert_allclose(sample1.intensity, sample2.intensity)


def test_isophote_small_flux_box():
    """
    Test the fluxes of an isophote whose enclosing box at the image
    corner is too small to integrate.
    """
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 0.2, x0=0.0, y0=0.0)
    sample._update(fixed_parameters=DEFAULT_FIX)
    isophote = Isophote(sample, 0, valid=True, stop_code=0)
    assert sample.n_valid > 0
    assert isophote.total_flux_ellipse == 0.0
    assert isophote.total_flux_circle == 0.0
    assert isophote.n_pixels_ellipse == 0
    assert isophote.n_pixels_circle == 0


def test_isophote_fluxes_integer_center():
    """
    Test that an integer ellipse center gives the same fluxes as the
    equivalent float center.
    """
    data = np.ones((51, 51))
    npix = []
    for x0 in (25, 25.0):
        geometry = EllipseGeometry(x0, x0, 10.0, 0.3, 0.5 * u.rad)
        sample = EllipseSample(data, 10.0, geometry=geometry)
        sample.update()
        isophote = Isophote(sample, 1, valid=True, stop_code=0)
        npix.append((isophote.n_pixels_circle, isophote.n_pixels_ellipse))
    assert npix[0] == npix[1]
    assert_allclose(npix[1][0], np.pi * 10.0**2, rtol=0.02)


def test_isophote_round_orientation_err():
    """
    Test that the orientation error of a round isophote is zero.
    """
    data = make_test_image(eps=0.0, seed=0)
    sample = EllipseSample(data, 20.0, ellipticity=0.0)
    sample._update(fixed_parameters=DEFAULT_FIX)
    isophote = Isophote(sample, 0, valid=True, stop_code=0)
    assert isophote.orientation_err == 0.0 * u.deg
    assert np.isfinite(isophote.ellipticity_err)


class TestIsophoteList:
    def setup_class(self):
        data = make_test_image(seed=0)
        self.slen = 5
        self.isolist_sma10 = self.build_list(data, sma0=10.0, slen=self.slen)
        self.isolist_sma100 = self.build_list(data, sma0=100.0, slen=self.slen)
        self.isolist_sma200 = self.build_list(data, sma0=200.0, slen=self.slen)
        self.data = data

    @staticmethod
    def build_list(data, sma0, *, slen=5):
        iso_list = []
        for k in range(slen):
            sample = EllipseSample(data, float(k + sma0))
            sample._update(fixed_parameters=DEFAULT_FIX)
            iso_list.append(Isophote(sample, k, valid=True, stop_code=0))
        return IsophoteList(iso_list)

    def test_basic_list(self):
        # Make sure it can be indexed as a list.
        result = self.isolist_sma10[:]
        assert isinstance(result[0], Isophote)

        # Make sure the important arrays contain floats, especially the
        # sma array, which is derived from a property in the Isophote
        # class.
        assert isinstance(result.sma, np.ndarray)
        assert isinstance(result.sma[0], float)
        assert isinstance(result.intensity, np.ndarray)
        assert isinstance(result.intensity[0], float)
        assert isinstance(result.rms, np.ndarray)
        assert isinstance(result.intensity_err, np.ndarray)
        assert isinstance(result.pixel_stddev, np.ndarray)
        assert isinstance(result.gradient, np.ndarray)
        assert isinstance(result.gradient_err, np.ndarray)
        assert isinstance(result.gradient_rel_err, np.ndarray)
        assert isinstance(result.sector_area, np.ndarray)
        assert isinstance(result.n_iter, np.ndarray)
        assert isinstance(result.n_data, np.ndarray)
        assert isinstance(result.n_flagged, np.ndarray)
        assert isinstance(result.valid, np.ndarray)
        assert isinstance(result.stop_code, np.ndarray)
        assert isinstance(result.total_flux_circle, np.ndarray)
        assert isinstance(result.total_flux_ellipse, np.ndarray)
        assert isinstance(result.n_pixels_circle, np.ndarray)
        assert isinstance(result.n_pixels_ellipse, np.ndarray)
        assert isinstance(result.a3, np.ndarray)
        assert isinstance(result.a4, np.ndarray)
        assert isinstance(result.b3, np.ndarray)
        assert isinstance(result.b4, np.ndarray)

        # The counts and stop codes are integers and the validity flags
        # are booleans.
        for name in ('n_iter', 'n_data', 'n_flagged', 'stop_code'):
            assert getattr(result, name).dtype.kind == 'i'
        assert result.valid.dtype == bool

        samples = result.sample
        assert isinstance(samples, list)
        assert isinstance(samples[0], EllipseSample)

        iso = result.get_closest(13.6)
        assert isinstance(iso, Isophote)
        assert_allclose(iso.sma, 14.0, atol=1e-6)

    def test_to_table_columns(self):
        isolist = self.isolist_sma10
        match = "'main' and 'all' must be input by themselves"
        for columns in (['main', 'sma'], ['sma', 'all']):
            with pytest.raises(ValueError, match=match):
                isolist.to_table(columns=columns)

        columns = ['sma', 'ellipticity', 'intensity_err',
                   'ellipticity_err']
        tbl = isolist.to_table(columns=columns)
        assert tbl.colnames == columns
        assert_allclose(tbl['ellipticity'], isolist.ellipticity)
        assert_allclose(tbl['intensity_err'], isolist.intensity_err)

        # A deprecated name gives a column with the new name.
        match = "'eps' attribute was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            tbl = isolist.to_table(columns=['eps'])
        assert tbl.colnames == ['ellipticity']

    def test_to_table_deprecated_column_warns_once(self):
        match = "'niter' attribute was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning,
                          match=match) as record:
            tbl = self.isolist_sma10.to_table(columns=['sma', 'niter'])
        assert len(record) == 1
        assert record[0].filename == __file__
        assert tbl.colnames == ['sma', 'n_iter']

    def test_mutators(self):
        isolist = IsophoteList(list(self.isolist_sma10))
        n_isophotes = len(isolist)
        first = isolist[0]
        last = isolist[-1]

        isolist.append(first)
        assert len(isolist) == n_isophotes + 1
        assert isolist[-1] is first

        isolist.insert(0, last)
        assert len(isolist) == n_isophotes + 2
        assert isolist[0] is last

        isolist[1] = last
        assert isolist[1] is last

        del isolist[0]
        assert len(isolist) == n_isophotes + 1
        assert isolist[0] is last

        # The original list is not modified.
        assert len(self.isolist_sma10) == n_isophotes

    @pytest.mark.parametrize('name', ['intensity_err', 'ellipticity_err',
                                      'orientation_err', 'x0_err', 'y0_err',
                                      'gradient_err', 'gradient_rel_err',
                                      'a3_err', 'b3_err', 'a4_err', 'b4_err'])
    def test_error_properties(self, name):
        isolist = self.isolist_sma10
        values = getattr(isolist, name)
        assert values.shape == (len(isolist),)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0)
        expected = [getattr(iso, name) for iso in isolist]
        if name == 'orientation_err':
            expected = u.Quantity(expected)
        assert_allclose(values, expected)

    def test_deprecated_attribute(self):
        match = "'niter' attribute was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            n_iter = self.isolist_sma10.niter
        assert_allclose(n_iter, self.isolist_sma10.n_iter)

    def test_isophote_deprecated_attribute(self):
        isophote = self.isolist_sma10[2]
        match = "'niter' attribute was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            n_iter = isophote.niter
        assert n_iter == isophote.n_iter

    @pytest.mark.parametrize(('old_name', 'new_name'),
                             [('tflux_e', 'total_flux_ellipse'),
                              ('tflux_c', 'total_flux_circle'),
                              ('npix_e', 'n_pixels_ellipse'),
                              ('npix_c', 'n_pixels_circle')])
    def test_deprecated_flux_attributes(self, old_name, new_name):
        isolist = self.isolist_sma10
        isophote = isolist[2]
        match = f"'{old_name}' attribute was deprecated in version 3.2"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            value = getattr(isophote, old_name)
        assert value == getattr(isophote, new_name)
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            values = getattr(isolist, old_name)
        assert_allclose(values, getattr(isolist, new_name))

        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            tbl = isolist.to_table(columns=[old_name])
        assert tbl.colnames == [new_name]

        tbl = isolist.to_table(columns=isolist.properties)
        assert new_name in tbl.colnames
        assert old_name not in tbl.colnames
        match = f"'{old_name}' was deprecated in version 3.2"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            column = tbl[old_name]
        assert_allclose(column, tbl[new_name])

    def test_deprecated_since_versions(self):
        tbl = self.isolist_sma10.to_table(
            columns=self.isolist_sma10.properties)
        match = "'niter' was deprecated in version 3.0"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            _ = tbl['niter']

    def test_isophote_methods(self):
        isophote = self.isolist_sma10[2]
        assert 'sma' in str(isophote)

        table = isophote.to_table()
        assert len(table) == 1
        assert table['sma'][0] == isophote.sma

        x, y = isophote.sampled_coordinates()
        assert len(x) == len(y) == isophote.n_data
        radius = np.hypot(x - isophote.x0, y - isophote.y0)
        assert_allclose(radius, isophote.sample.radii)

    def test_central_pixel(self):
        # Test the central pixel isophote.
        geometry = EllipseGeometry(257.0, 258.0, 10.0, 0.2, 0.3 * u.rad)
        cenpix = _central_isophote(self.data, geometry)
        assert type(cenpix) is Isophote
        assert cenpix.x0 == cenpix.sample.geometry.x0
        assert cenpix.y0 == cenpix.sample.geometry.y0
        assert cenpix.ellipticity == 0.0
        assert cenpix.orientation == 0.0 * u.deg

    def test_to_aperture(self):
        """
        Test that the aperture of an isophote is the ellipse of its
        geometry, and that the central pixel has no aperture.
        """
        isophote = self.isolist_sma10[-1]
        aperture = isophote.to_aperture()
        assert isinstance(aperture, EllipticalAperture)
        assert_allclose(aperture.positions, (isophote.x0, isophote.y0))
        assert aperture.a == isophote.sma
        assert aperture.b == isophote.sma * (1.0 - isophote.ellipticity)
        assert_allclose(aperture.theta.to_value(u.deg),
                        isophote.orientation.to_value(u.deg))

        geometry = EllipseGeometry(257.0, 258.0, 10.0, 0.2, 0.3 * u.rad)
        cenpix = _central_isophote(self.data, geometry)
        match = 'positive semimajor axis length'
        with pytest.raises(ValueError, match=match):
            cenpix.to_aperture()

    def test_extend(self):
        # The extend method shouldn't return anything, and should modify
        # the first list in place.
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        assert len(inner_list) == self.slen
        assert len(outer_list) == self.slen
        inner_list.extend(outer_list)
        assert len(inner_list) == 2 * self.slen

        # The __iadd__ operator should behave like the extend method.
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        inner_list += outer_list
        assert len(inner_list) == 2 * self.slen

        # The __add__ operator should create a new IsophoteList instance
        # with the result, and should not modify the operands.
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        result = inner_list + outer_list
        assert isinstance(result, IsophoteList)
        assert len(inner_list) == self.slen
        assert len(outer_list) == self.slen
        assert len(result) == 2 * self.slen

    def test_slicing(self):
        iso_list = self.isolist_sma10[:]
        assert len(iso_list) == self.slen
        assert len(iso_list[1:-1]) == self.slen - 2
        assert len(iso_list[2:-2]) == self.slen - 4

    def test_combined(self):
        # Combine extend with slicing.
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        sublist = inner_list[2:-2]
        sublist.extend(outer_list)
        assert len(sublist) == 2 * self.slen - 4

        # Try one more slice.
        even_outer_list = self.isolist_sma200
        sublist.extend(even_outer_list[1:-1])
        assert len(sublist) == 2 * self.slen - 4 + 3

        # Combine __add__ with slicing.
        sublist = inner_list[2:-2]
        result = sublist + outer_list
        assert isinstance(result, IsophoteList)
        assert len(sublist) == self.slen - 4
        assert len(result) == 2 * self.slen - 4

        result = inner_list[2:-2] + outer_list
        assert isinstance(result, IsophoteList)
        assert len(result) == 2 * self.slen - 4

    def test_sort(self):
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        result = outer_list[2:-2] + inner_list

        assert result[-1].sma < result[0].sma
        result.sort()
        assert result[-1].sma > result[0].sma

    def test_to_table(self):
        test_img = make_test_image(nx=55, ny=55, x0=27, y0=27,
                                   background=100.0, noise=1.0e-6, i0=100.0,
                                   sma=10.0, eps=0.2, pa=0.0, seed=1)
        g = EllipseGeometry(27, 27, 5, 0.2, 0 * u.rad)
        ellipse = Ellipse(test_img, geometry=g)
        isolist = ellipse.fit_image(max_sma=27)

        assert len(isolist.properties) >= 30

        tbl = isolist.to_table()
        assert len(tbl.colnames) == 18

        tbl = isolist.to_table(columns=isolist.properties)
        assert len(tbl.colnames) >= 30

        tbl = isolist.to_table(columns=['sma'])
        assert len(tbl.colnames) == 1

        tbl = isolist.to_table(columns=['total_flux_ellipse',
                                        'total_flux_circle',
                                        'n_pixels_ellipse',
                                        'n_pixels_circle'])
        assert len(tbl.colnames) == 4

    @pytest.mark.parametrize('columns', ['invalid', 'append',
                                         ['id', 'n_iter']])
    def test_invalid_column(self, columns):
        test_img = make_test_image(nx=55, ny=55, x0=27, y0=27,
                                   background=100.0, noise=1.0e-6, i0=100.0,
                                   sma=10.0, eps=0.2, pa=0.0, seed=1)
        g = EllipseGeometry(27, 27, 5, 0.2, 0 * u.rad)
        ellipse = Ellipse(test_img, geometry=g)
        isolist = ellipse.fit_image(max_sma=27)
        match = 'Invalid column name'
        with pytest.raises(ValueError, match=match):
            isolist.to_table(columns=columns)

    def test_deprecated_column(self):
        """
        Regression test to ensure that deprecated column names are still
        accepted by ``to_table``, not rejected by the new column-name
        validation.
        """
        test_img = make_test_image(nx=55, ny=55, x0=27, y0=27,
                                   background=100.0, noise=1.0e-6, i0=100.0,
                                   sma=10.0, eps=0.2, pa=0.0, seed=1)
        g = EllipseGeometry(27, 27, 5, 0.2, 0 * u.rad)
        ellipse = Ellipse(test_img, geometry=g)
        isolist = ellipse.fit_image(max_sma=27)
        match = "'grad_error' attribute was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            tbl = isolist.to_table(columns=['grad_error'])
        assert tbl.colnames == ['gradient_err']

    def test_deprecated_grad_rerror_column(self):
        """
        Regression test for the deprecated ``grad_rerror`` column name,
        which was removed by mistake in version 3.1.
        """
        test_img = make_test_image(nx=55, ny=55, x0=27, y0=27,
                                   background=100.0, noise=1.0e-6, i0=100.0,
                                   sma=10.0, eps=0.2, pa=0.0, seed=1)
        g = EllipseGeometry(27, 27, 5, 0.2, 0 * u.rad)
        ellipse = Ellipse(test_img, geometry=g)
        isolist = ellipse.fit_image(max_sma=27)
        tbl = isolist.to_table(columns=isolist.properties)
        assert 'grad_rerror' not in tbl.colnames

        match = "'grad_rerror' was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match) as rec:
            values = tbl['grad_rerror']
        assert len(rec) == 1
        assert_equal(values, tbl['gradient_rel_err'])

        match = "'grad_rerror' column was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match) as rec:
            tbl2 = isolist.to_table(columns=['grad_rerror'])
        assert len(rec) == 1
        assert tbl2.colnames == ['gradient_rel_err']
        assert_equal(tbl2['gradient_rel_err'], isolist.gradient_rel_err)


@pytest.mark.parametrize(('width', 'n_data'),
                         [(0.22, 3), (0.3, 4), (0.33, 5), (0.4, 6)])
def test_isophote_few_points_warnings(width, n_data):
    """
    Test that no warnings are emitted for an isophote with as many or
    slightly more data points than harmonic coefficients.

    The warnings are recorded because the errors raised for them by the
    pytest warning filter are caught in the harmonic computations.
    """
    data = make_test_image(seed=0)
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    angle = np.arctan2(yy - 256, xx - 256)
    mask = ~((angle > 0.0) & (angle < width))
    sample = EllipseSample(np.ma.masked_array(data, mask=mask), 20.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        sample._update(fixed_parameters=DEFAULT_FIX)
        isophote = Isophote(sample, 1, valid=False, stop_code=3)
    assert len(record) == 0
    assert isophote.n_data == n_data

    # The third and fourth harmonic fits have 3 coefficients and the
    # first and second harmonic fit has 5.
    assert np.isnan(isophote.a3) == (n_data <= 3)
    assert (isophote.x0_err == 0.0) == (n_data <= 5)


@pytest.mark.parametrize(('width', 'has_deviations'),
                         [(0.22, False), (0.3, True)])
def test_isophote_few_points(width, has_deviations):
    """
    Test an isophote with too few data points to estimate the residual
    variance of the harmonic fits.
    """
    data = make_test_image(seed=0)
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    angle = np.arctan2(yy - 256, xx - 256)
    mask = ~((angle > 0.0) & (angle < width))
    sample = EllipseSample(np.ma.masked_array(data, mask=mask), 20.0)
    sample._update(fixed_parameters=DEFAULT_FIX)
    isophote = Isophote(sample, 1, valid=False, stop_code=3)
    assert 0 < isophote.n_data <= 5
    assert isophote.x0_err == 0.0
    assert isophote.ellipticity_err == 0.0
    if has_deviations:
        assert isophote.n_data > 3
        assert np.isfinite(isophote.a3)
        assert np.isfinite(isophote.a3_err)
    else:
        assert isophote.n_data <= 3
        assert np.isnan(isophote.a3)
        assert np.isnan(isophote.a4_err)


def test_to_table_dtypes():
    """
    Test that the table columns have a numerical dtype, with NaN for the
    values that are not defined for the central pixel.
    """
    data = make_test_image(nx=200, ny=200, seed=0)
    isolist = Ellipse(data).fit_image(max_sma=20.0)
    assert type(isolist[0]) is Isophote

    for columns in (None, isolist.properties):
        tbl = isolist.to_table(columns=columns)
        for name in tbl.colnames:
            assert tbl[name].dtype.kind in 'fib'
        assert np.isnan(tbl['gradient_err'][0])
        assert np.all(np.isfinite(tbl['gradient_err'][1:]))
    assert tbl['n_data'].dtype.kind == 'i'
    assert tbl['valid'].dtype == bool
    assert np.isnan(tbl['total_flux_ellipse'][0])

    tbl = isolist[0].to_table()
    assert tbl['gradient_rel_err'].dtype.kind == 'f'
    assert np.isnan(tbl['gradient_rel_err'][0])


def test_central_isophote_is_plain_isophote():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    iso = isolist[0]
    assert type(iso) is Isophote
    assert iso.sma == 0.0
    assert iso.geometry is iso.sample.geometry
    assert iso.ellipticity == 0.0
    assert iso.orientation == 0.0 * u.deg
    assert iso.geometry.ellipticity == 0.0
    assert iso.geometry.orientation == 0.0 * u.deg
    assert iso.gradient == 0.0
    assert iso.a3 == 0.0
    assert iso.x0_err == 0.0


def test_isophote_list_properties_match_definitions():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    for name, dtype, _, description in _ISOPHOTE_PROPERTIES:
        values = getattr(isolist, name)
        assert len(values) == len(isolist)
        assert np.asarray(values).dtype == np.dtype(dtype)
        assert description
        assert getattr(IsophoteList, name).__doc__ == description
    assert isolist.properties == [p[0] for p in _ISOPHOTE_PROPERTIES]


def test_isophote_property_unit_must_be_degrees():
    match = "'length' isophote property has the unit m.*'_length_rad'"
    with pytest.raises(ValueError, match=match):
        _make_property('length', float, u.m, 'A length.')


@pytest.mark.parametrize(('old', 'new'), RENAMED.items())
def test_renamed_result_attributes(old, new):
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    since = '3.0' if old == 'nflag' else '3.2'
    match = f"'{old}' attribute was deprecated in version {since}"
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        old_values = getattr(isolist, old)
    assert_array_equal(old_values, getattr(isolist, new))
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        assert getattr(isolist[5], old) == getattr(isolist[5], new)


def test_to_table_columns_equal_attributes():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    tbl = isolist.to_table(columns=isolist.properties)
    assert tbl.colnames == isolist.properties
    with pytest.warns(PhotutilsDeprecationWarning, match='intens_err'):
        assert_array_equal(tbl['intens_err'], isolist.intensity_err)


def test_missing_values_are_nan():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    for iso in isolist:
        for name, *_ in _ISOPHOTE_PROPERTIES:
            assert getattr(iso, name) is not None
    central = isolist[0]
    for name in ('rms', 'pixel_stddev', 'sector_area',
                 'gradient_err', 'gradient_rel_err',
                 'total_flux_ellipse', 'n_pixels_circle'):
        assert np.isnan(getattr(central, name))


def test_isophote_list_copy_and_pickle():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    for clone in (copy.deepcopy(isolist),
                  pickle.loads(  # noqa: S301
                      pickle.dumps(isolist))):  # nosec B301
        assert_array_equal(clone.sma, isolist.sma)
        assert_array_equal(clone.orientation, isolist.orientation)
        assert clone[3].geometry is clone[3].sample.geometry


def test_isophote_orientation_units():
    data = make_test_image(pa=np.pi / 4, eps=0.4, seed=2)
    isolist = Ellipse(data).fit_image()
    assert isolist.orientation.unit == u.deg
    assert isolist.orientation_err.unit == u.deg
    assert isolist[5].orientation.unit == u.deg
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa'"):
        pa = isolist.pa
    assert not isinstance(pa, u.Quantity)
    assert_allclose(pa, isolist.orientation.to_value(u.rad))
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa_err'"):
        assert_allclose(isolist.pa_err,
                        isolist.orientation_err.to_value(u.rad))
    tbl = isolist.to_table()
    assert_array_equal(tbl['orientation'], isolist.orientation)
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa'"):
        assert_array_equal(tbl['pa'], isolist.orientation)


def test_isophote_deprecated_orientation_names():
    data = make_test_image(pa=np.pi / 4, eps=0.4, seed=2)
    isolist = Ellipse(data).fit_image()
    iso = isolist[5]
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa'"):
        pa = iso.pa
    assert pa == iso.geometry._orientation_rad
    assert not isinstance(pa, u.Quantity)
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa_err'"):
        pa_err = iso.pa_err
    assert not isinstance(pa_err, u.Quantity)
    assert_allclose(pa_err, iso.orientation_err.to_value(u.rad))

    match = "'pa_err' attribute was deprecated in version 3.2"
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        tbl = isolist.to_table(columns=['sma', 'pa_err'])
    assert tbl.colnames == ['sma', 'orientation_err']
    assert tbl['orientation_err'].unit == u.deg


# The deprecated Isophote attribute names that were plain instance
# attributes, with their new names and deprecation versions.
SETTABLE_RENAMED = {
    'intens': ('intensity', '3.2'),
    'int_err': ('intensity_err', '3.2'),
    'grad': ('gradient', '3.2'),
    'pix_stddev': ('pixel_stddev', '3.2'),
    'sarea': ('sector_area', '3.2'),
    'n_flag': ('n_flagged', '3.2'),
    'ellip_err': ('ellipticity_err', '3.2'),
    'tflux_e': ('total_flux_ellipse', '3.2'),
    'tflux_c': ('total_flux_circle', '3.2'),
    'npix_e': ('n_pixels_ellipse', '3.2'),
    'npix_c': ('n_pixels_circle', '3.2'),
    'grad_error': ('gradient_err', '3.0'),
    'grad_r_error': ('gradient_rel_err', '3.0'),
    'niter': ('n_iter', '3.0'),
    'ndata': ('n_data', '3.0'),
    'nflag': ('n_flagged', '3.0'),
}


def _make_isophote():
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 20.0)
    sample._update(fixed_parameters=DEFAULT_FIX)
    return Isophote(sample, 0, valid=True, stop_code=0)


@pytest.mark.parametrize(('old', 'new', 'since'),
                         [(old, *value)
                          for old, value in SETTABLE_RENAMED.items()])
def test_isophote_deprecated_attribute_assignment(old, new, since):
    iso = _make_isophote()
    match = f"'{old}' attribute was deprecated in version {since}"
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        setattr(iso, old, 7)
    assert record[0].filename == __file__
    assert getattr(iso, new) == 7
    assert old not in vars(iso)
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        assert getattr(iso, old) == 7
    assert record[0].filename == __file__


def test_isophote_deprecated_assignment_changes_list():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    expected = isolist[5].intensity - 50.0
    with pytest.warns(PhotutilsDeprecationWarning, match="'intens'"):
        isolist[5].intens = expected
    assert isolist.intensity[5] == expected


def test_isophote_deprecated_angle_assignment():
    iso = _make_isophote()
    match = "'pa_err' attribute was deprecated in version 3.2"
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        iso.pa_err = np.pi / 4
    assert record[0].filename == __file__
    assert_allclose(iso.orientation_err.to_value(u.deg), 45.0)
    assert 'pa_err' not in vars(iso)

    for name in ('pa', 'eps'):
        with pytest.raises(AttributeError):
            setattr(iso, name, 0.5)
        assert name not in vars(iso)
    with pytest.warns(PhotutilsDeprecationWarning, match="'eps'") as record:
        assert iso.eps == iso.ellipticity
    assert record[0].filename == __file__
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa'") as record:
        assert iso.pa == iso.geometry._orientation_rad
    assert record[0].filename == __file__


def test_isophote_copy_and_pickle_no_warnings():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    tbl = isolist.to_table(columns=isolist.properties)
    table_isolist = IsophoteList.from_table(tbl)
    for iso in (isolist[5], table_isolist[5]):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            clones = (copy.copy(iso), copy.deepcopy(iso),
                      pickle.loads(  # noqa: S301
                          pickle.dumps(iso)))  # nosec B301
        for clone in clones:
            assert clone.intensity == iso.intensity
            assert clone.n_data == iso.n_data
            assert clone.orientation_err == iso.orientation_err


def test_to_table_default_columns():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    assert isolist.to_table().colnames == isolist.default_columns
    isolist.default_columns = ['sma', 'intensity']
    assert isolist.to_table().colnames == ['sma', 'intensity']
    assert isolist[2:5].default_columns == ['sma', 'intensity']
    tbl = isolist.to_table(columns=isolist.properties)
    assert tbl.colnames == isolist.properties


def test_default_columns_carried_as_copy():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    isolist.default_columns = ['sma', 'intensity']

    sliced = isolist[2:5]
    sliced.default_columns.append('x0')
    assert isolist.default_columns == ['sma', 'intensity']

    added = isolist[:3] + isolist[3:6]
    assert added.default_columns == ['sma', 'intensity']
    added.default_columns.append('y0')
    assert isolist.default_columns == ['sma', 'intensity']

    assert isolist[0].to_table().colnames == list(
        IsophoteList([]).default_columns)


@pytest.mark.parametrize('name', ['main', 'all'])
def test_to_table_deprecated_shortcuts(name):
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    with pytest.warns(PhotutilsDeprecationWarning, match=name) as record:
        tbl = isolist.to_table(columns=name)
    assert len(record) == 1
    assert record[0].filename == __file__
    expected = (isolist.default_columns if name == 'main'
                else isolist.properties)
    assert tbl.colnames == expected


def test_get_names_deprecated():
    isolist = IsophoteList([])
    match = "The 'IsophoteList.get_names' method was deprecated"
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        assert isolist.get_names() == isolist.properties
    assert record[0].filename == __file__


def test_isophote_comparisons_deprecated():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    match = 'omparing isophotes'
    for compare in (lambda a, b: a < b, lambda a, b: a > b,
                    lambda a, b: a <= b, lambda a, b: a >= b,
                    lambda a, b: a == b):
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            compare(isolist[1], isolist[2])


def test_fix_geometry_deprecated():
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 20.0)
    sample._update(fixed_parameters=DEFAULT_FIX)
    iso = Isophote(sample, 0, valid=True, stop_code=0)
    match = "The 'Isophote.fix_geometry' method was deprecated"
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        iso.fix_geometry(iso)


def test_from_table_round_trip():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    tbl = isolist.to_table(columns=isolist.properties)
    result = IsophoteList.from_table(tbl)
    assert len(result) == len(isolist)
    for name in isolist.properties:
        expected = getattr(isolist, name)
        value = getattr(result, name)
        if name in ('orientation', 'orientation_err'):
            assert_allclose(value.to_value(u.deg),
                            expected.to_value(u.deg), rtol=1e-13)
        else:
            assert_array_equal(value, expected)
    assert result.sample == [None] * len(result)


def test_from_table_plain_table_and_copies():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    tbl = Table(isolist.to_table(columns=isolist.properties))
    result = IsophoteList.from_table(tbl)
    assert_array_equal(result.intensity, isolist.intensity)
    assert_array_equal(result.sample, [None] * len(result))
    unpickled = pickle.loads(  # noqa: S301
        pickle.dumps(result))  # nosec B301
    for clone in (copy.deepcopy(result), unpickled):
        assert_array_equal(clone.sma, result.sma)
        assert_array_equal(clone.intensity, result.intensity)
        assert_allclose(clone.orientation.to_value(u.deg),
                        result.orientation.to_value(u.deg))
    iso = copy.deepcopy(result[3])
    assert iso.sample is None
    with pytest.warns(PhotutilsDeprecationWarning, match='intens'):
        assert iso.intens == result[3].intensity


def test_from_table_defaults_and_errors():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    names = ['sma', 'x0', 'y0', 'ellipticity', 'orientation']
    tbl = isolist.to_table(columns=names)
    tbl['extra'] = 1.0
    result = IsophoteList.from_table(tbl, step=0.2)
    assert np.all(np.isnan(result.intensity))
    assert np.all(result.n_data == 0)
    assert np.all(result.stop_code == 0)
    assert np.all(result.valid)
    assert result[3].geometry.step == 0.2
    match = 'has no sample'
    with pytest.raises(ValueError, match=match):
        result[3].sampled_coordinates()

    bad = tbl.copy()
    bad.remove_column('ellipticity')
    with pytest.raises(ValueError, match='ellipticity'):
        IsophoteList.from_table(bad)


def test_from_table_unitless_angle(tmp_path):
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    path = tmp_path / 'isophotes.csv'
    isolist.to_table().write(path)
    tbl = QTable.read(path)
    with pytest.raises(ValueError, match=r"'orientation'.*unit"):
        IsophoteList.from_table(tbl)
    tbl['orientation'].unit = u.deg
    with pytest.raises(ValueError, match=r"'orientation_err'.*unit"):
        IsophoteList.from_table(tbl)
    tbl['orientation_err'].unit = u.deg
    result = IsophoteList.from_table(tbl)
    assert_allclose(result.sma, isolist.sma)

    tbl['orientation'] = tbl['orientation'].value << u.m
    with pytest.raises(ValueError, match=r"'orientation'.*angular"):
        IsophoteList.from_table(tbl)


def test_from_table_old_column_names():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    tbl = QTable(isolist.to_table())
    tbl.rename_column('intensity', 'intens')
    tbl.rename_column('orientation', 'pa')
    with pytest.warns(PhotutilsDeprecationWarning) as record:
        result = IsophoteList.from_table(tbl)
    assert len(record) == 2
    assert_array_equal(result.intensity, isolist.intensity)
    assert_allclose(result.orientation.to_value(u.deg),
                    isolist.orientation.to_value(u.deg))


def test_from_table_then_measure_and_model():
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    isolist = ellipse.fit_image()
    loaded = IsophoteList.from_table(isolist.to_table())
    model = loaded.make_model_image(data.shape)
    assert_allclose(model, isolist.make_model_image(data.shape),
                    rtol=1e-10)
    measured = ellipse.measure_isophotes(loaded)
    assert_allclose(measured.intensity, isolist.intensity, rtol=1e-6)
    assert measured[3].sample is not None


def test_from_table_masked_columns():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    names = ['sma', 'x0', 'y0', 'ellipticity', 'orientation',
             'intensity', 'n_data', 'stop_code', 'valid']
    tbl = Table(isolist.to_table(columns=names))

    # A masked float entry becomes NaN
    intensity = MaskedColumn(tbl['intensity'].copy())
    intensity.data[2] = -999.0
    intensity[2] = np.ma.masked
    tbl['intensity'] = intensity
    result = IsophoteList.from_table(tbl)
    assert np.isnan(result.intensity[2])
    assert np.isfinite(result.intensity[3])

    # A masked entry elsewhere raises
    for name in ('sma', 'orientation', 'n_data', 'stop_code', 'valid'):
        bad = Table(isolist.to_table(columns=names))
        col = MaskedColumn(bad[name].copy())
        col[1] = np.ma.masked
        bad[name] = col
        with pytest.raises(ValueError, match=name):
            IsophoteList.from_table(bad)


def test_from_table_csv_valid_column(tmp_path):
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    tbl = isolist.to_table(columns=['sma', 'x0', 'y0', 'ellipticity',
                                    'orientation', 'valid'])
    tbl['valid'][1] = False
    path = tmp_path / 'isophotes.csv'
    tbl.write(path)
    tbl2 = QTable.read(path)
    tbl2['orientation'].unit = u.deg
    result = IsophoteList.from_table(tbl2)
    assert not result.valid[1]
    assert result.valid[0]

    tbl2['valid'] = ['yes'] * len(tbl2)
    with pytest.raises(ValueError, match='valid'):
        IsophoteList.from_table(tbl2)
    tbl2['valid'] = 1.5
    with pytest.raises(ValueError, match='valid'):
        IsophoteList.from_table(tbl2)


def test_from_table_conflicting_columns():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    tbl = QTable(isolist.to_table())
    tbl['pa'] = tbl['orientation'] + 10 * u.deg
    with pytest.raises(ValueError, match=r"'pa'.*'orientation'"):
        IsophoteList.from_table(tbl)

    tbl = QTable(isolist.to_table())
    tbl['int_err'] = tbl['intensity_err']
    tbl['intens_err'] = 2 * tbl['intensity_err']
    tbl.remove_column('intensity_err')
    with pytest.raises(ValueError, match=r"'int_err'.*'intens_err'"):
        IsophoteList.from_table(tbl)

    tbl = QTable(isolist.to_table())
    tbl['nflag'] = tbl['n_flagged']
    tbl['n_flag'] = tbl['n_flagged']
    tbl.remove_column('n_flagged')
    with pytest.raises(ValueError, match=r"'nflag'.*'n_flag'"):
        IsophoteList.from_table(tbl)

    tbl = QTable(isolist.to_table())
    tbl['grad_r_error'] = tbl['gradient_rel_err']
    tbl['grad_rerror'] = tbl['gradient_rel_err']
    tbl.remove_column('gradient_rel_err')
    with pytest.raises(ValueError, match=r"'grad_r_error'.*'grad_rerror'"):
        IsophoteList.from_table(tbl)


def test_from_table_masked_quantity_angle_column():
    tbl = _make_geometry_table()
    mask = [False, True, False, False]
    tbl['orientation_err'] = Masked(tbl['orientation_err'], mask=mask)
    result = IsophoteList.from_table(tbl)
    assert np.isnan(result.orientation_err[1])
    assert_allclose(result.orientation_err[2:], [1.0, 1.0] * u.deg)

    tbl['orientation'] = Masked(tbl['orientation'], mask=mask)
    with pytest.raises(ValueError, match="'orientation' column has masked"):
        IsophoteList.from_table(tbl)


def test_fit_isophote_with_table_isophote_list():
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    isolist = ellipse.fit_image()
    loaded = IsophoteList.from_table(isolist.to_table())
    iso = ellipse.fit_isophote(30.0, isophote_list=loaded)
    assert iso.sample is not None


@pytest.mark.parametrize('columns', [('sma', 'intensity'), 'sma'])
def test_default_columns_tuple_or_string(columns):
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    isolist.default_columns = columns
    expected = [columns] if isinstance(columns, str) else list(columns)
    assert isolist.to_table().colnames == expected
    sliced = isolist[2:5]
    assert sliced.default_columns == expected
    assert sliced.to_table().colnames == expected
    added = isolist[:3] + isolist[3:6]
    assert added.default_columns == expected


@pytest.mark.parametrize('name', ['main', 'all'])
def test_default_columns_shortcut_names_invalid(name):
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    isolist.default_columns = [name]
    with pytest.raises(ValueError, match='Invalid column name'):
        isolist.to_table()


def test_from_table_coerces_dtypes():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    names = ['sma', 'x0', 'y0', 'ellipticity', 'orientation', 'n_data',
             'n_flagged', 'n_iter', 'stop_code', 'valid',
             'n_pixels_ellipse']
    tbl = QTable(isolist.to_table(columns=names))
    tbl['n_data'] = tbl['n_data'].astype(float)
    tbl['valid'] = tbl['valid'].astype(int)
    tbl['n_pixels_ellipse'] = np.arange(len(tbl))
    result = IsophoteList.from_table(tbl)
    iso = result[3]
    assert type(iso.n_data) is int
    assert iso.n_data == isolist[3].n_data
    assert iso.valid is True
    assert type(iso.n_pixels_ellipse) is float
    assert_array_equal(result.n_data, isolist.n_data)
    assert_array_equal(result.valid, isolist.valid)


@pytest.mark.parametrize('name', ['n_data', 'n_flagged', 'n_iter',
                                  'stop_code'])
@pytest.mark.parametrize('value', [np.nan, 2.5])
def test_from_table_integer_column_invalid(name, value):
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    names = ['sma', 'x0', 'y0', 'ellipticity', 'orientation', name]
    tbl = QTable(isolist.to_table(columns=names))
    tbl[name] = tbl[name].astype(float)
    tbl[name][2] = value
    with pytest.raises(ValueError, match=f"'{name}'"):
        IsophoteList.from_table(tbl)


def test_from_table_valid_integer_invalid():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    names = ['sma', 'x0', 'y0', 'ellipticity', 'orientation', 'valid']
    tbl = QTable(isolist.to_table(columns=names))
    tbl['valid'] = np.full(len(tbl), 2)
    with pytest.raises(ValueError, match="'valid'"):
        IsophoteList.from_table(tbl)


def _make_geometry_table():
    """
    Make a small table with the required from_table columns.
    """
    return QTable({'sma': [0.0, 2.0, 4.0, 8.0],
                   'x0': [50.0, 50.1, 50.2, 50.3],
                   'y0': [40.0, 40.1, 40.2, 40.3],
                   'ellipticity': [0.0, 0.1, 0.2, 0.3],
                   'orientation': [0.0, 10.0, 20.0, 30.0] * u.deg,
                   'orientation_err': [np.nan, 1.0, 1.0, 1.0] * u.deg})


@pytest.mark.parametrize(('name', 'value'), [
    ('sma', -1.0), ('sma', np.nan), ('sma', np.inf),
    ('x0', np.nan), ('x0', -np.inf), ('y0', np.nan), ('y0', np.inf),
    ('ellipticity', -0.1), ('ellipticity', 1.0), ('ellipticity', 1.5),
    ('ellipticity', np.nan), ('ellipticity', np.inf),
    ('orientation', np.nan), ('orientation', np.inf),
    ('orientation_err', np.inf), ('orientation_err', -np.inf),
])
def test_from_table_invalid_values(name, value):
    tbl = _make_geometry_table()
    tbl[name][2] = value * tbl[name].unit if tbl[name].unit else value
    with pytest.raises(ValueError, match=rf"'{name}' column.*row 2\b"):
        IsophoteList.from_table(tbl)


def test_from_table_invalid_value_first_row():
    tbl = _make_geometry_table()
    tbl['sma'][1] = -1.0
    tbl['sma'][3] = -2.0
    with pytest.raises(ValueError, match=r'row 1\b'):
        IsophoteList.from_table(tbl)


def test_from_table_invalid_value_deprecated_name():
    tbl = _make_geometry_table()
    tbl.rename_column('ellipticity', 'eps')
    tbl['eps'][3] = 1.0
    with (pytest.warns(PhotutilsDeprecationWarning, match="'eps'"),
          pytest.raises(ValueError, match=r"'eps' column.*row 3\b")):
        IsophoteList.from_table(tbl)


def test_from_table_valid_edge_values():
    tbl = _make_geometry_table()
    tbl['ellipticity'][3] = 0.999
    tbl['orientation'][3] = -400.0 * u.deg
    result = IsophoteList.from_table(tbl)
    assert result.sma[0] == 0.0
    assert result.ellipticity[3] == 0.999
    assert np.isnan(result.orientation_err[0])


@pytest.mark.parametrize('step', [0.0, -0.1, np.nan, np.inf, None, 'a'])
def test_from_table_invalid_step(step):
    tbl = _make_geometry_table()
    with pytest.raises(ValueError, match='step'):
        IsophoteList.from_table(tbl, step=step)


def test_from_table_numeric_step_string():
    tbl = _make_geometry_table()
    result = IsophoteList.from_table(tbl, step='0.2')
    assert result[1].geometry.step == 0.2


def test_from_table_numeric_text_columns():
    tbl = _make_geometry_table()
    for name in ('sma', 'x0', 'ellipticity', 'orientation'):
        tbl[name] = np.asarray(tbl[name]).astype(str)
    tbl['orientation'].unit = u.deg
    result = IsophoteList.from_table(Table(tbl))
    assert_array_equal(result.sma, [0.0, 2.0, 4.0, 8.0])
    assert_allclose(result.orientation.to_value(u.deg),
                    [0.0, 10.0, 20.0, 30.0])


@pytest.mark.parametrize('name', ['sma', 'orientation', 'intensity'])
def test_from_table_non_numeric_column(name):
    tbl = Table(_make_geometry_table())
    tbl[name] = ['a', 'b', 'c', 'd']
    if name == 'orientation':
        tbl[name].unit = u.deg
    with pytest.raises(ValueError, match=f"'{name}' column must contain"):
        IsophoteList.from_table(tbl)


@pytest.mark.parametrize('name', ['sma', 'x0', 'y0', 'x0_err', 'y0_err',
                                  'ellipticity', 'intensity'])
def test_from_table_plain_column_units(name):
    tbl = Table(_make_geometry_table())
    if name not in tbl.colnames:
        tbl[name] = np.ones(len(tbl))
    values = np.array(tbl[name])
    pixel_names = ('sma', 'x0', 'y0', 'x0_err', 'y0_err')

    tbl[name].unit = u.dimensionless_unscaled
    result = IsophoteList.from_table(tbl)
    assert_array_equal(getattr(result, name), values)

    tbl[name].unit = u.pix
    if name in pixel_names:
        result = IsophoteList.from_table(tbl)
        assert_array_equal(getattr(result, name), values)
    else:
        with pytest.raises(ValueError, match=rf"'{name}'.*pix"):
            IsophoteList.from_table(tbl)

    tbl[name].unit = u.arcsec
    with pytest.raises(ValueError, match=rf"'{name}'.*arcsec"):
        IsophoteList.from_table(tbl)


@pytest.mark.parametrize('table_class', [Table, QTable])
def test_from_table_ecsv_round_trip(tmp_path, table_class):
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    path = tmp_path / 'isophotes.ecsv'
    isolist.to_table(columns=isolist.properties).write(path)
    result = IsophoteList.from_table(table_class.read(path))
    assert_array_equal(result.sma, isolist.sma)
    assert_array_equal(result.intensity, isolist.intensity)
    assert_allclose(result.orientation.to_value(u.deg),
                    isolist.orientation.to_value(u.deg), rtol=1e-13)


@pytest.mark.parametrize('name', ['sma', 'x0', 'y0', 'x0_err', 'y0_err',
                                  'ellipticity', 'intensity'])
def test_from_table_quantity_units(name):
    tbl = _make_geometry_table()
    if name not in tbl.colnames:
        tbl[name] = np.ones(len(tbl))
    values = np.asarray(tbl[name])
    pixel_names = ('sma', 'x0', 'y0', 'x0_err', 'y0_err')

    tbl[name] = values * u.dimensionless_unscaled
    result = IsophoteList.from_table(tbl)
    assert_array_equal(getattr(result, name), values)

    tbl[name] = values * u.pix
    if name in pixel_names:
        result = IsophoteList.from_table(tbl)
        assert_array_equal(getattr(result, name), values)
    else:
        with pytest.raises(ValueError, match=rf"'{name}'.*pix"):
            IsophoteList.from_table(tbl)

    tbl[name] = values * u.arcsec
    with pytest.raises(ValueError, match=rf"'{name}'.*arcsec"):
        IsophoteList.from_table(tbl)


def test_from_table_float_valid_column():
    tbl = _make_geometry_table()
    tbl['valid'] = [1.0, 0.0, 1.0, 1.0]
    result = IsophoteList.from_table(tbl)
    assert_array_equal(result.valid, [True, False, True, True])
    assert result[1].valid is False

    tbl['valid'] = [1.0, 0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="'valid'"):
        IsophoteList.from_table(tbl)
    tbl['valid'] = [1.0, 0.0, np.nan, 1.0]
    with pytest.raises(ValueError, match="'valid'"):
        IsophoteList.from_table(tbl)


def test_from_table_dimensionless_angle_message():
    tbl = _make_geometry_table()
    tbl['orientation'] = tbl['orientation'].value * u.dimensionless_unscaled
    match = ("The 'orientation' column must have an angular unit, not a "
             'dimensionless unit.')
    with pytest.raises(ValueError, match=match):
        IsophoteList.from_table(tbl)


def test_from_table_not_a_table():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    tbl = isolist.to_table()
    match = 'astropy Table or QTable is required'
    for value in ({name: tbl[name] for name in tbl.colnames}, tbl[0]):
        with pytest.raises(TypeError, match=match):
            IsophoteList.from_table(value)


def test_isophote_equality_with_non_isophote():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image(max_sma=20.0)
    iso = isolist[3]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert (iso == None) is False  # noqa: E711
        assert (iso != 'a') is True
        assert iso != 1.0
    with pytest.warns(PhotutilsDeprecationWarning, match='omparing'):
        assert iso == isolist[3]
