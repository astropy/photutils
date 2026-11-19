# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the ellipse module.
"""

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.datasets import make_noise_image
from photutils.datasets.load import _get_path
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning

# Define an off-center position and a tilted sma.
POS = 384
PA = np.deg2rad(10.0)

# Build off-center test data. It's fine to have a single np array to use
# in all tests that need it, but do not use a single instance of
# EllipseGeometry. The code may eventually modify it's contents. The safe
# bet is to build it wherever it's needed. The cost is negligible.
OFFSET_GALAXY = make_test_image(x0=POS, y0=POS, pa=PA, noise=1.0e-12,
                                seed=0)


class TestEllipse:
    def setup_class(self):
        # Centered, tilted galaxy
        self.data = make_test_image(pa=PA, seed=0)

    @pytest.mark.remote_data
    def test_find_center(self):
        path = _get_path('isophote/M51.fits', location='photutils-datasets',
                         cache=True)
        hdu = fits.open(path)
        data = hdu[0].data
        hdu.close()

        geometry = EllipseGeometry(252, 253, 10.0, 0.2, np.pi / 2)
        geometry.find_center(data)
        assert geometry.x0 == 257.0
        assert geometry.y0 == 258.0

    def test_basic(self):
        ellipse = Ellipse(self.data)
        isophote_list = ellipse.fit_image()

        assert isinstance(isophote_list, IsophoteList)
        assert len(isophote_list) > 1
        assert isinstance(isophote_list[0], Isophote)

        # Verify that the list is properly sorted in sem-major axis length
        assert isophote_list[-1] > isophote_list[0]

        # The fit should stop where gradient loses reliability
        assert len(isophote_list) == 69
        assert isophote_list[-1].stop_code == 1

    def test_linear(self):
        ellipse = Ellipse(self.data)
        # The maxsma keyword limits the runtime. The linear spacing checks
        # below do not depend on the outer extent of the fit.
        isophote_list = ellipse.fit_image(linear=True, step=2.0,
                                          maxsma=80.0)

        # Verify that the list is properly sorted in sem-major axis length
        assert isophote_list[-1] > isophote_list[0]

        # Difference in sma between successive isohpotes must be constant
        step = isophote_list[-1].sma - isophote_list[-2].sma
        assert math.isclose((isophote_list[-2].sma - isophote_list[-3].sma),
                            step, rel_tol=0.01)
        assert math.isclose((isophote_list[-3].sma - isophote_list[-4].sma),
                            step, rel_tol=0.01)
        assert math.isclose((isophote_list[2].sma - isophote_list[1].sma),
                            step, rel_tol=0.01)

    def test_fit_image_geometry_unchanged(self):
        """
        Test that the fix_XXX and linear parameters of fit_image apply
        only to that call and do not modify the input geometry.
        """
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, PA)
        ellipse = Ellipse(self.data, geometry=geometry)
        isolist = ellipse.fit_image(maxsma=30.0, step=2.0, linear=True,
                                    fix_center=True)
        assert np.all(isolist.x0 == 256.0)
        assert np.all(isolist.y0 == 256.0)
        assert_allclose(np.diff(isolist.sma[-3:]), 2.0)

        assert not np.any(geometry.fix)
        assert not geometry.linear_growth

        # A second call without these parameters fits the center and
        # uses geometric growth.
        isolist = ellipse.fit_image(maxsma=30.0)
        x0 = isolist.x0[isolist.sma > 0]
        assert np.any(x0 != 256.0)
        assert_allclose(isolist.sma[-1] / isolist.sma[-2], 1.1)

    @pytest.mark.parametrize('maxsma', [None, 200.0])
    def test_maxgerr(self, maxsma):
        """
        Test that the maxgerr criterion is triggered when two
        consecutive isophotes exceed it while the semimajor axis grows
        outwards.

        Without ``maxsma``, the semimajor axis growth then stops.
        With ``maxsma``, the remaining isophotes up to ``maxsma`` are
        extracted in noniterative mode (stop code 4).
        """
        data = make_test_image(noise=5.0, seed=0)
        maxgerr = 0.05
        isolist = Ellipse(data).fit_image(minsma=8.0, maxsma=maxsma,
                                          maxgerr=maxgerr)
        sma = isolist.sma
        stop_code = isolist.stop_code
        fitted = (sma >= 10.0) & (stop_code != 4)
        exceeded = isolist.gradient_rel_err[fitted] > maxgerr
        assert np.all(exceeded[-2:])
        assert not np.any(exceeded[:-2] & exceeded[1:-1])

        # The isophotes that exceed maxgerr are fitted instead of
        # being stopped and replaced by the geometry of the previous
        # isophote (stop code 5)
        assert np.all(stop_code[fitted][-2:] != 5)

        if maxsma is None:
            assert not np.any(stop_code == 4)
        else:
            last_fitted = sma[fitted][-1]
            assert np.all(stop_code[sma > last_fitted] == 4)
            assert sma[-1] < maxsma <= sma[-1] * 1.1

    @pytest.mark.parametrize('kwargs', [{'maxgerr': 0.5},
                                        {'going_inwards': True}])
    def test_fit_isophote_deprecated_keywords(self, kwargs):
        ellipse = Ellipse(self.data)
        match = f"'{next(iter(kwargs))}' was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            ellipse.fit_isophote(40.0, **kwargs)

    def test_round_galaxy_sample_matches_geometry(self):
        """
        Test that the sampled values of the fitted isophotes match their
        geometry for a round galaxy.

        For a round galaxy, the ellipticity corrections can cross zero.
        The ellipse is then flipped to a positive ellipticity, and the
        sample must be extracted from the flipped geometry.
        """
        data = make_test_image(eps=0.0, noise=1.0, seed=0)
        isolist = Ellipse(data).fit_image(maxsma=60.0)
        for isophote in isolist:
            if isophote.sma == 0:
                continue
            angles, radii, _ = isophote.sample.values
            assert_allclose(radii, isophote.sample.geometry.radius(angles),
                            rtol=1e-10)

    def test_first_inward_isophote(self):
        """
        Test that the first inward isophote starts from the geometry of
        the first fitted isophote instead of the last (outermost) one.
        """
        geometry = EllipseGeometry(256.0, 256.0, 10.0, 0.2, PA)
        ellipse = Ellipse(self.data, geometry=geometry)
        isolist = ellipse.fit_image(maxsma=40.0)

        first = isolist.get_closest(10.0)
        assert first.sma == 10.0
        sma = 10.0 / 1.1
        step = 1.0 / 1.1 - 1.0
        expected = ellipse.fit_isophote(sma, step=step,
                                        isophote_list=[first])
        isophote = isolist.get_closest(sma)
        assert isophote.sma == expected.sma
        assert isophote.x0 == expected.x0
        assert isophote.y0 == expected.y0
        assert isophote.eps == expected.eps
        assert isophote.pa == expected.pa

    def test_threshold_deprecated(self):
        geometry = EllipseGeometry(250.0, 260.0, 10.0, 0.2, PA)
        match = "'threshold' was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            ellipse = Ellipse(self.data, geometry=geometry, threshold=0.1)

        match = 'Use EllipseGeometry.find_center instead'
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            ellipse.set_threshold(0.1)

        # The object centerer is not run.
        assert geometry.x0 == 250.0
        assert geometry.y0 == 260.0

    def test_nonfinite_values(self):
        """
        Test that nonfinite values are excluded from the fits, like
        masked pixels.
        """
        isolist = Ellipse(self.data).fit_image(maxsma=60.0)

        data = self.data.copy()
        data[256, 294:296] = np.nan  # on the sma ~38 isophote
        data[250, 240] = np.inf
        match = 'Input image contains nonfinite'
        with pytest.warns(AstropyUserWarning, match=match):
            ellipse = Ellipse(data)
        isolist_nan = ellipse.fit_image(maxsma=60.0)
        assert_allclose(isolist_nan.sma, isolist.sma)
        assert_allclose(isolist_nan.intens, isolist.intens, rtol=1e-4)
        assert np.all(np.isfinite(isolist_nan.tflux_e[1:]))
        assert np.all(np.isfinite(isolist_nan.tflux_c[1:]))

        masked = np.ma.masked_invalid(data)
        isolist_masked = Ellipse(masked).fit_image(maxsma=60.0)
        assert_allclose(isolist_masked.intens, isolist_nan.intens)

    def test_nonfinite_center(self):
        """
        Test a nonfinite central pixel.

        The central pixel intensity is NaN, and the innermost isophotes,
        whose samples all include the central pixel, are not fitted.
        """
        data = self.data.copy()
        data[256, 256] = np.nan
        match = 'Input image contains nonfinite'
        with pytest.warns(AstropyUserWarning, match=match):
            ellipse = Ellipse(data)
        match = 'Too small sample to warrant a fit'
        with pytest.warns(AstropyUserWarning, match=match):
            isolist = ellipse.fit_image(maxsma=20.0)
        assert isolist[0].sma == 0
        assert np.isnan(isolist[0].intens)
        assert isolist[0].n_data == 0
        assert np.all(np.isfinite(isolist.intens[1:]))

    def test_invalid_image(self):
        match = 'image must not be a Quantity array'
        with pytest.raises(TypeError, match=match):
            Ellipse(self.data * u.Jy)

        match = 'image must be a 2D array'
        with pytest.raises(ValueError, match=match):
            Ellipse(self.data[0])

    def test_array_like_image(self):
        ellipse = Ellipse(self.data.tolist())
        assert isinstance(ellipse.image, np.ndarray)
        assert_allclose(ellipse.image, self.data)

    @pytest.mark.parametrize(('sma0', 'geometry_sma'),
                             [(0.0, 10.0), (-1.0, 10.0), (np.nan, 10.0),
                              (None, 0.0)])
    def test_invalid_sma0(self, sma0, geometry_sma):
        geometry = EllipseGeometry(256.0, 256.0, geometry_sma, 0.2, PA)
        ellipse = Ellipse(self.data, geometry=geometry)
        match = 'The starting semimajor axis length'
        with pytest.raises(ValueError, match=match):
            ellipse.fit_image(sma0=sma0)

    def test_zero_maxsma_maxrit(self):
        """
        Test that zero values of maxsma and maxrit are used instead of
        being treated as None.
        """
        ellipse = Ellipse(self.data)

        # No isophotes are fit outwards from sma0.
        isolist = ellipse.fit_image(sma0=10.0, maxsma=0.0)
        assert isolist.sma[-1] == 10.0

        # All isophotes are extracted in noniterative mode.
        isolist = ellipse.fit_image(sma0=10.0, maxsma=20.0, maxrit=0.0)
        assert np.all(isolist.stop_code[isolist.sma > 0] == 4)

    def test_fit_one_ellipse(self):
        ellipse = Ellipse(self.data)
        isophote = ellipse.fit_isophote(40.0)

        assert isinstance(isophote, Isophote)
        assert isophote.valid

    def test_nearest_neighbor_zero_gradient(self):
        """
        Regression test for a crash with the nearest_neighbor
        integration mode.

        At small semimajor axes, nearest-neighbor sampling can extract
        identical pixel sets at two nearby radii, making the local
        gradient exactly zero. The harmonic correctors would then
        divide by zero, producing a nonfinite geometry that crashed
        the integrator with "ValueError: cannot convert float NaN to
        integer".
        """
        g = EllipseGeometry(256.0, 256.0, 20.0, 0.2, 0.0)
        ellipse = Ellipse(self.data, geometry=g)
        isophote_list = ellipse.fit_image(integrmode='nearest_neighbor',
                                          maxsma=30.0)

        assert isinstance(isophote_list, IsophoteList)
        assert len(isophote_list) > 1

    def test_offcenter_fail(self):
        # A first guess ellipse that is centered in the image frame.
        # This should result in failure since the real galaxy image is
        # off-center by a large offset.
        ellipse = Ellipse(OFFSET_GALAXY)

        match = 'No meaningful fit was possible'
        with pytest.warns(AstropyUserWarning, match=match):
            isophote_list = ellipse.fit_image()
        assert len(isophote_list) == 0

    @pytest.mark.parametrize('case', ['masked', 'outside'])
    def test_no_valid_data(self, case):
        """
        Test that fit_image returns an empty list when the first
        isophote has no valid data points.
        """
        if case == 'masked':
            data = np.ma.masked_all(self.data.shape)
            geometry = None
        else:
            data = self.data
            geometry = EllipseGeometry(256.0, 256.0, 2000.0, 0.2, PA)
        ellipse = Ellipse(data, geometry=geometry)
        match1 = 'Too small sample to warrant a fit'
        match2 = 'No meaningful fit was possible'
        with (pytest.warns(AstropyUserWarning, match=match1),
              pytest.warns(AstropyUserWarning, match=match2)):
            isophote_list = ellipse.fit_image()
        assert isinstance(isophote_list, IsophoteList)
        assert len(isophote_list) == 0

    def test_offcenter_fit(self):
        # A first guess ellipse that is roughly centered on the offset
        # galaxy image.
        g = EllipseGeometry(POS + 5, POS + 5, 10.0, eps=0.2, pa=PA, astep=0.1)
        ellipse = Ellipse(OFFSET_GALAXY, geometry=g)
        isophote_list = ellipse.fit_image()

        # The fit should stop when too many potential sample points fall
        # outside the image frame.
        assert len(isophote_list) == 63
        assert isophote_list[-1].stop_code == 1

    def test_offcenter_go_beyond_frame(self):
        # Same as before, but now force the fit to go beyond the image
        # frame limits.
        g = EllipseGeometry(POS + 5, POS + 5, 10.0, eps=0.2, pa=PA, astep=0.1)
        ellipse = Ellipse(OFFSET_GALAXY, geometry=g)
        isophote_list = ellipse.fit_image(maxsma=400.0)

        # The fit should go to maxsma, but with fixed geometry
        assert len(isophote_list) == 71
        assert isophote_list[-1].stop_code == 4

        # Check that no zero-valued intensities were left behind in the
        # sample arrays when sampling outside the image.
        for iso in isophote_list:
            assert not np.any(iso.sample.values[2] == 0)

    def test_ellipse_shape(self):
        """
        Regression test for #670/673.
        """
        ny = 500
        nx = 150
        g = Gaussian2D(100.0, nx / 2.0, ny / 2.0, 20, 12,
                       theta=np.deg2rad(40.0))
        y, x = np.mgrid[0:ny, 0:nx]
        noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                                 stddev=2.0, seed=0)
        data = g(x, y) + noise
        ellipse = Ellipse(data)  # estimates initial center
        isolist = ellipse.fit_image()

        # The isophotes are centered on the galaxy and have its shape.
        # The number of isophotes is not checked because the fits of
        # the noise-dominated outermost isophotes, which set where the
        # fit stops, depend on the platform.
        mask = (isolist.sma > 2) & (isolist.sma < 40)
        assert_allclose(isolist.x0[mask], nx / 2.0, atol=1.0)
        assert_allclose(isolist.y0[mask], ny / 2.0, atol=1.0)
        assert_allclose(np.median(isolist.eps[mask]), 1.0 - 12 / 20,
                        atol=0.02)
        assert_allclose(np.median(isolist.pa[mask]), np.deg2rad(40.0),
                        atol=np.deg2rad(2.0))
