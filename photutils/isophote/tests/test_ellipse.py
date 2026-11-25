# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the ellipse module.
"""

import math
import sys
from concurrent.futures import ThreadPoolExecutor

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

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
# EllipseGeometry. The code may eventually modify its contents. The safe
# bet is to build it wherever it's needed. The cost is negligible.
OFFSET_GALAXY = make_test_image(x0=POS, y0=POS, pa=PA, noise=1.0e-12,
                                seed=0)


def _assert_sma_grid(isolist, *, step=0.1):
    """
    Assert that the isophotes, excluding the central pixel, are on a
    contiguous geometric grid of semimajor axis lengths.
    """
    sma = isolist.sma[isolist.sma > 0]
    assert_allclose(sma[1:] / sma[:-1], 1.0 + step)


def _assert_fflag_stop(isolist, *, fflag=0.7):
    """
    Assert that the outward growth stopped because more than a fraction
    1 - fflag of the sample points of the last isophote were flagged.
    """
    flagged = isolist.n_flagged / (isolist.n_flagged + isolist.n_data)
    assert isolist[-1].stop_code == 1
    assert flagged[-1] > 1.0 - fflag
    assert flagged[-2] <= 1.0 - fflag


def _assert_same_geometry(isophote1, isophote2):
    """
    Assert that two isophotes have the same center, ellipticity, and
    orientation.
    """
    for name in ('x0', 'y0', 'ellipticity', 'orientation'):
        assert getattr(isophote1, name) == getattr(isophote2, name)


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

        geometry = EllipseGeometry(252, 253, 10.0, 0.2, (np.pi / 2) * u.rad)
        geometry.find_center(data)
        assert geometry.x0 == 257.0
        assert geometry.y0 == 258.0

    def test_basic(self):
        ellipse = Ellipse(self.data)
        isophote_list = ellipse.fit_image()

        assert isinstance(isophote_list, IsophoteList)
        assert len(isophote_list) > 1
        assert isinstance(isophote_list[0], Isophote)

        # Verify that the list is properly sorted in semimajor axis length.
        assert isophote_list[-1].sma > isophote_list[0].sma

        # The fit should stop when too many sample points fall outside
        # the image frame.
        _assert_fflag_stop(isophote_list)
        _assert_sma_grid(isophote_list)

    def test_linear(self):
        ellipse = Ellipse(self.data)
        # The max_sma keyword limits the runtime. The linear spacing checks
        # below do not depend on the outer extent of the fit.
        isophote_list = ellipse.fit_image(linear_growth=True, step=2.0,
                                          max_sma=80.0)

        # Verify that the list is properly sorted in semimajor axis length.
        assert isophote_list[-1].sma > isophote_list[0].sma

        # Difference in sma between successive isophotes must be constant.
        step = isophote_list[-1].sma - isophote_list[-2].sma
        assert math.isclose((isophote_list[-2].sma - isophote_list[-3].sma),
                            step, rel_tol=0.01)
        assert math.isclose((isophote_list[-3].sma - isophote_list[-4].sma),
                            step, rel_tol=0.01)
        assert math.isclose((isophote_list[2].sma - isophote_list[1].sma),
                            step, rel_tol=0.01)

    def test_fit_image_geometry_unchanged(self):
        """
        Test that the fix_center, fix_orientation, fix_ellipticity,
        and linear_growth parameters of fit_image apply only to that
        call and do not modify the input geometry.
        """
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, PA * u.rad)
        ellipse = Ellipse(self.data, geometry=geometry)
        isolist = ellipse.fit_image(max_sma=30.0, step=2.0, linear_growth=True,
                                    fix_center=True)
        assert np.all(isolist.x0 == 256.0)
        assert np.all(isolist.y0 == 256.0)
        assert_allclose(np.diff(isolist.sma[-3:]), 2.0)

        assert not np.any(geometry._fix)
        assert not geometry.linear_growth

        # A second call without these parameters fits the center and
        # uses geometric growth.
        isolist = ellipse.fit_image(max_sma=30.0)
        x0 = isolist.x0[isolist.sma > 0]
        assert np.any(x0 != 256.0)
        assert_allclose(isolist.sma[-1] / isolist.sma[-2], 1.1)

    def test_geometry_step(self):
        """
        Test that fit_image and fit_isophote use the step of the input
        geometry unless the step keyword is set.
        """
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, PA * u.rad,
                                   step=3.0, linear_growth=True)
        ellipse = Ellipse(self.data, geometry=geometry)
        isolist = ellipse.fit_image(max_sma=30.0)
        assert_allclose(np.diff(isolist.sma[-3:]), 3.0)
        isophote = ellipse.fit_isophote(20.0)
        assert isophote.sample.geometry.step == 3.0

        isolist = ellipse.fit_image(max_sma=30.0, step=2.0)
        assert_allclose(np.diff(isolist.sma[-3:]), 2.0)
        isophote = ellipse.fit_isophote(20.0, step=2.0)
        assert isophote.sample.geometry.step == 2.0
        assert geometry.step == 3.0

    @pytest.mark.parametrize('max_sma', [None, 200.0])
    def test_max_gradient_rel_err(self, max_sma):
        """
        Test that the max_gradient_rel_err criterion is triggered when
        two consecutive isophotes exceed it while the semimajor axis
        grows outwards.

        Without ``max_sma``, the semimajor axis growth then stops.
        With ``max_sma``, the remaining isophotes up to ``max_sma`` are
        extracted in noniterative mode (stop code 4).
        """
        data = make_test_image(noise=5.0, seed=0)
        max_gradient_rel_err = 0.05
        isolist = Ellipse(data).fit_image(
            min_sma=8.0, max_sma=max_sma,
            max_gradient_rel_err=max_gradient_rel_err)
        sma = isolist.sma
        stop_code = isolist.stop_code
        fitted = (sma >= 10.0) & (stop_code != 4)
        exceeded = isolist.gradient_rel_err[fitted] > max_gradient_rel_err
        assert np.all(exceeded[-2:])
        assert not np.any(exceeded[:-2] & exceeded[1:-1])

        # The isophotes that exceed the limit are fitted instead of
        # being stopped and replaced by the geometry of the previous
        # isophote (stop code 5).
        assert np.all(stop_code[fitted][-2:] != 5)

        if max_sma is None:
            assert not np.any(stop_code == 4)
        else:
            last_fitted = sma[fitted][-1]
            assert np.all(stop_code[sma > last_fitted] == 4)
            assert sma[-1] <= max_sma < sma[-1] * 1.1

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
        isolist = Ellipse(data).fit_image(max_sma=60.0)
        for isophote in isolist:
            if isophote.sma == 0:
                continue
            angles, radii = isophote.sample.angles, isophote.sample.radii
            assert_allclose(radii, isophote.sample.geometry.radius(angles),
                            rtol=1e-10)

    def test_first_inward_isophote(self):
        """
        Test that the first inward isophote starts from the geometry of
        the first fitted isophote instead of the last (outermost) one.
        """
        geometry = EllipseGeometry(256.0, 256.0, 10.0, 0.2, PA * u.rad)
        ellipse = Ellipse(self.data, geometry=geometry)
        isolist = ellipse.fit_image(max_sma=40.0)

        first = isolist.get_closest(10.0)
        assert first.sma == 10.0
        sma = 10.0 / 1.1
        expected = ellipse.fit_isophote(sma, isophote_list=[first])
        isophote = isolist.get_closest(sma)
        assert isophote.sma == expected.sma
        assert isophote.x0 == expected.x0
        assert isophote.y0 == expected.y0
        assert isophote.ellipticity == expected.ellipticity
        assert isophote.orientation == expected.orientation

    def test_first_isophote_miniters(self):
        """
        Test that only the first isophote runs twice the minimum number
        of iterations.
        """
        miniters = 10
        isolist = Ellipse(self.data).fit_image(miniters=miniters, max_sma=20.0)
        first = isolist.get_closest(10.0)
        assert first.sma == 10.0
        assert first.n_iter >= 2 * miniters
        outer = isolist.sma > first.sma
        assert np.all(isolist.n_iter[outer] >= miniters)
        assert np.all(isolist.n_iter[outer] < 2 * miniters)

    def test_failed_outward_isophote_geometry(self):
        """
        Test that an isophote whose outward fit failed is extracted
        with the geometry of the previous isophote.

        A bright ring makes the gradient positive for the isophotes
        just inside of it, so their fits fail.
        """
        yy, xx = np.mgrid[0:201, 0:201]
        galaxy_radius = np.hypot(xx - 100, (yy - 100) / 0.8)
        ring_radius = np.hypot(xx - 100, yy - 100)
        data = (1000.0 * np.exp(-galaxy_radius / 10.0)
                + 100.0 * np.exp(-0.125 * (ring_radius - 40.0)**2))
        geometry = EllipseGeometry(100.0, 100.0, 10.0, 0.2, 0.0 * u.rad)
        isolist = Ellipse(data, geometry=geometry).fit_image()

        # The outward growth stops after two consecutive failed fits.
        assert_array_equal(isolist.stop_code[-3:], [0, 5, 5])
        _assert_same_geometry(isolist[-2], isolist[-3])
        _assert_same_geometry(isolist[-1], isolist[-2])
        first = isolist.get_closest(10.0)
        assert isolist[-3].ellipticity != first.ellipticity

    def test_too_many_flagged_second_isophote(self):
        """
        Test that the outward growth stops at the first isophote with
        too many flagged data points (stop code 1), including the second
        isophote.
        """
        data = make_test_image(nx=200, ny=200, x0=12.0, y0=100.0,
                               noise=0.5, seed=0)
        geometry = EllipseGeometry(12.0, 100.0, 20.0, 0.2, 0.0 * u.deg)
        isolist = Ellipse(data, geometry=geometry).fit_image(min_sma=20.0)
        assert_allclose(isolist.sma, [20.0, 22.0])
        assert_array_equal(isolist.stop_code, [0, 1])

        # The failed isophote was extracted with the geometry of the
        # first one, so its geometry errors are not measured.
        assert np.isfinite(isolist[0].x0_err)
        assert np.isnan(isolist[1].x0_err)
        assert np.isnan(isolist[1].ellipticity_err)

    def test_too_many_flagged_noniterative(self):
        """
        Test that the outward growth in noniterative mode stops at the
        first isophote with too many flagged data points (stop code 1)
        if ``max_sma`` is not set.
        """
        data = make_test_image(nx=100, ny=100, x0=50.0, y0=50.0,
                               noise=0.2, seed=0)
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.2, 0.0 * u.deg)
        ellipse = Ellipse(data, geometry=geometry)
        isolist = ellipse.fit_image(max_iterative_sma=12.0)
        valid_fraction = isolist.n_data / (isolist.n_data
                                           + isolist.n_flagged)
        assert isolist[-1].stop_code == 1
        assert isolist[-1].n_iter == 0
        assert valid_fraction[-1] < 0.7
        assert np.all(valid_fraction[:-1] >= 0.7)
        assert np.all(isolist.stop_code[:-1] != 1)
        assert np.isnan(isolist[-1].ellipticity_err)

        # Measuring along a fixed ellipse is not affected.
        measured = ellipse.measure_isophotes([isolist[-1]])
        assert measured[0].stop_code == 4

    def test_too_many_flagged_converged(self):
        """
        Test that an isophote with too many flagged data points has a
        stop code of 1 even if its fit converged in the same iteration.
        """
        data = make_test_image(noise=1.0, seed=0)
        xx = np.arange(data.shape[1])[np.newaxis, :]
        mask = np.broadcast_to(np.abs(xx - 256) > 20, data.shape)
        geometry = EllipseGeometry(256.0, 256.0, 40.0, 0.2, 0.0 * u.deg)
        ellipse = Ellipse(np.ma.masked_array(data, mask=mask),
                          geometry=geometry)
        isophote = ellipse.fit_isophote(40.0, miniters=1, convergence=10.0)
        assert isophote.n_data < 0.7 * (isophote.n_data + isophote.n_flagged)
        assert isophote.stop_code == 1

    def test_inward_isophote_without_valid_data(self):
        """
        Test that an inward isophote that cannot be fit (stop code 3)
        is skipped and the inward fits continue, as in the STSDAS
        ellipse task.

        The isophotes inside a masked elliptical ring are fitted.
        """
        data = make_test_image(noise=1.0, seed=0)
        yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        radius = np.hypot(xx - 256.0, (yy - 256.0) / 0.8)
        mask = (radius > 6.5) & (radius < 9.5)
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, 0.0 * u.deg)
        ellipse = Ellipse(np.ma.masked_array(data, mask=mask),
                          geometry=geometry)
        match = 'Too small sample to warrant a fit'
        with pytest.warns(AstropyUserWarning, match=match):
            isolist = ellipse.fit_image(max_sma=25.0)

        inner = (isolist.sma > 2.0) & (isolist.sma < 5.0)
        assert np.count_nonzero(inner) > 5
        assert np.all(isolist.stop_code[inner] == 0)

    def test_failed_inward_isophote_geometry(self):
        """
        Test that the isophotes whose inward fits failed are extracted
        with the geometry of the adjacent isophote with the next larger
        semimajor axis length, not that of the first fitted isophote.
        """
        isolist = Ellipse(self.data).fit_image(max_sma=20.0)
        first = isolist.get_closest(10.0)
        failed = np.nonzero(isolist.stop_code == 5)[0]
        assert len(failed) > 0
        for idx in failed:
            assert isolist[idx].sma < first.sma
            _assert_same_geometry(isolist[idx], isolist[idx + 1])

            # The copied geometry was not fitted.
            assert np.isnan(isolist[idx].x0_err)
            assert np.isnan(isolist[idx].orientation_err)

        # At least one failed isophote is next to a fitted isophote
        # whose geometry differs from the first isophote.
        assert any(isolist[idx + 1].stop_code == 0
                   and isolist[idx + 1].ellipticity != first.ellipticity
                   for idx in failed)

    def test_threshold_deprecated(self):
        geometry = EllipseGeometry(250.0, 260.0, 10.0, 0.2, PA * u.rad)
        match = "'threshold' was deprecated"
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            ellipse = Ellipse(self.data, geometry=geometry, threshold=0.1)

        match = "The 'Ellipse.set_threshold' method was deprecated"
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
        isolist = Ellipse(self.data).fit_image(max_sma=60.0)

        data = self.data.copy()
        data[256, 294:296] = np.nan  # on the sma ~38 isophote
        data[250, 240] = np.inf
        match = 'Input image contains nonfinite'
        with pytest.warns(AstropyUserWarning, match=match):
            ellipse = Ellipse(data)
        isolist_nan = ellipse.fit_image(max_sma=60.0)
        assert_allclose(isolist_nan.sma, isolist.sma)
        assert_allclose(isolist_nan.intensity, isolist.intensity, rtol=1e-4)
        assert np.all(np.isfinite(isolist_nan.total_flux_ellipse[1:]))
        assert np.all(np.isfinite(isolist_nan.total_flux_circle[1:]))

        masked = np.ma.masked_invalid(data)
        isolist_masked = Ellipse(masked).fit_image(max_sma=60.0)
        assert_allclose(isolist_masked.intensity, isolist_nan.intensity)

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
            isolist = ellipse.fit_image(max_sma=20.0)
        assert isolist[0].sma == 0
        assert np.isnan(isolist[0].intensity)
        assert isolist[0].n_data == 0
        assert np.all(np.isfinite(isolist.intensity[1:]))

    def test_invalid_image(self):
        match = 'image must not be a Quantity array'
        with pytest.raises(TypeError, match=match):
            Ellipse(self.data * u.Jy)

        match = 'image must be a 2D array'
        with pytest.raises(ValueError, match=match):
            Ellipse(self.data[0])

    def test_array_like_image(self):
        data = make_test_image(nx=100, ny=100, seed=0)
        isophote = Ellipse(data).fit_isophote(20.0)
        isophote_from_list = Ellipse(data.tolist()).fit_isophote(20.0)
        assert isophote_from_list.intensity == isophote.intensity
        assert isophote_from_list.ellipticity == isophote.ellipticity

    @pytest.mark.parametrize(('sma_start', 'geometry_sma'),
                             [(0.0, 10.0), (-1.0, 10.0), (np.nan, 10.0),
                              (None, 0.0)])
    def test_invalid_sma_start(self, sma_start, geometry_sma):
        geometry = EllipseGeometry(256.0, 256.0, geometry_sma, 0.2, PA * u.rad)
        ellipse = Ellipse(self.data, geometry=geometry)
        match = 'The starting semimajor axis length'
        with pytest.raises(ValueError, match=match):
            ellipse.fit_image(sma_start=sma_start)

    def test_zero_max_sma_max_iterative_sma(self):
        """
        Test that zero values of max_sma and max_iterative_sma are used
        instead of being treated as None.
        """
        ellipse = Ellipse(self.data)

        # The starting sma is larger than a max_sma of zero.
        match = 'must be between min_sma and max_sma'
        with pytest.raises(ValueError, match=match):
            ellipse.fit_image(sma_start=10.0, max_sma=0.0)

        # All isophotes are extracted in noniterative mode.
        isolist = ellipse.fit_image(sma_start=10.0, max_sma=20.0,
                                    max_iterative_sma=0.0)
        assert np.all(isolist.stop_code[isolist.sma > 0] == 4)

    @pytest.mark.parametrize('kwargs', [{'min_sma': np.nan},
                                        {'max_sma': np.nan}])
    def test_nan_sma_limits(self, kwargs):
        match = 'min_sma and max_sma must not be NaN'
        with pytest.raises(ValueError, match=match):
            Ellipse(self.data).fit_image(**kwargs)

    def test_min_sma_larger_than_max_sma(self):
        match = 'min_sma must not be larger than max_sma'
        with pytest.raises(ValueError, match=match):
            Ellipse(self.data).fit_image(sma_start=20.0, min_sma=30.0,
                                         max_sma=25.0)

    @pytest.mark.parametrize('kwargs', [
        {'sma_start': 20.0, 'min_sma': 22.0},
        {'sma_start': 20.0, 'max_sma': 18.0},
        {'min_sma': 12.0},
    ])
    def test_sma_start_outside_limits(self, kwargs):
        """
        Test that the starting semimajor axis length, from ``sma_start``
        or from the geometry (10 by default), must be between
        ``min_sma`` and ``max_sma``.
        """
        match = 'must be between min_sma and max_sma'
        with pytest.raises(ValueError, match=match):
            Ellipse(self.data).fit_image(**kwargs)

    def test_fit_image_threads(self):
        """
        Test that concurrent fit_image calls on the same Ellipse
        instance, with different parameters, give the same results as
        serial calls.
        """
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, PA * u.rad)
        ellipse = Ellipse(self.data, geometry=geometry)
        kwargs_list = [{}, {'fix_center': True},
                       {'fix_orientation': True, 'linear_growth': True,
                        'step': 5.0},
                       {'fix_ellipticity': True}]
        expected = [ellipse.fit_image(max_sma=40.0, **kwargs)
                    for kwargs in kwargs_list]

        n_repeats = 2
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(ellipse.fit_image, max_sma=40.0,
                                       **kwargs)
                       for kwargs in kwargs_list * n_repeats]
            results = [future.result() for future in futures]

        for idx, result in enumerate(results):
            isolist = expected[idx % len(kwargs_list)]
            for name in ('sma', 'intensity', 'x0', 'y0', 'ellipticity',
                         'orientation'):
                assert_array_equal(getattr(result, name),
                                   getattr(isolist, name))
        assert not np.any(geometry._fix)
        assert not geometry.linear_growth

    def test_fit_image_threads_shared_sigma_clip(self):
        """
        Test that concurrent fit_image calls that share a SigmaClip
        object give the same results as serial calls.

        Before astropy 8.1, a SigmaClip object stores its clipping
        bounds on itself while it is called.
        """
        images = []
        for idx in range(4):
            data = make_test_image(background=100.0 * idx, seed=idx)
            data[250:256, 290:296] += 2000.0
            images.append(data)
        sigma_clip = SigmaClip(sigma=2.5, maxiters=3)

        def fit(data):
            return Ellipse(data).fit_image(max_sma=30.0,
                                           sigma_clip=sigma_clip)

        expected = [fit(data) for data in images]
        # A short switch interval makes the threads interleave often.
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1.0e-6)
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(fit, images))
        finally:
            sys.setswitchinterval(interval)

        for idx, result in enumerate(results):
            isolist = expected[idx % len(images)]
            assert_array_equal(result.n_flagged, isolist.n_flagged)
            assert_array_equal(result.intensity, isolist.intensity)

    @pytest.mark.parametrize(('kwargs', 'fixed_names'),
                             [({'fix_center': True}, ('x0', 'y0')),
                              ({'fix_orientation': True}, ('orientation',)),
                              ({'fix_ellipticity': True}, ('ellipticity',)),
                              ({'fix_center': True, 'fix_ellipticity': True},
                               ('x0', 'y0', 'ellipticity'))])
    def test_fixed_parameters(self, kwargs, fixed_names):
        """
        Test that the fixed parameters keep their initial values and
        that the other parameters are fitted.
        """
        geometry = EllipseGeometry(256.5, 255.5, 20.0, 0.25,
                                   (PA + 0.1) * u.rad)
        isolist = Ellipse(self.data, geometry=geometry).fit_image(
            max_sma=40.0, **kwargs)
        sma_positive = isolist.sma > 0
        for name in ('x0', 'y0', 'ellipticity', 'orientation'):
            values = getattr(isolist, name)[sma_positive]
            if name in fixed_names:
                assert np.all(values == getattr(geometry, name))
            else:
                assert np.any(values != getattr(geometry, name))

    @pytest.mark.parametrize('use_geometry', [False, True])
    def test_everything_fixed(self, use_geometry):
        kwargs = {'fix_center': True, 'fix_orientation': True,
                  'fix_ellipticity': True}
        if use_geometry:
            geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, PA * u.rad,
                                       fix_center=True, fix_orientation=True,
                                       fix_ellipticity=True)
            ellipse = Ellipse(self.data, geometry=geometry)
            kwargs = {}
        else:
            ellipse = Ellipse(self.data)
        match = 'Everything is fixed'
        with pytest.warns(AstropyUserWarning, match=match):
            isolist = ellipse.fit_image(**kwargs)
        assert len(isolist) == 0

    def test_step_sampling_annulus(self):
        """
        Test that the step of fit_image and fit_isophote defines the
        sampling annulus, which sets the sector areas of the area
        integration modes.
        """
        ellipse = Ellipse(self.data)
        sarea = {}
        for step in (0.1, 0.3):
            isolist = ellipse.fit_image(sma_start=20.0, max_sma=40.0,
                                        min_sma=15.0, step=step,
                                        integration_mode='mean')
            assert np.all([iso.sample.geometry.step == step
                           for iso in isolist])
            sarea[step] = isolist.get_closest(20.0).sector_area
            isophote = ellipse.fit_isophote(20.0, step=step,
                                            integration_mode='mean')
            assert isophote.sample.geometry.step == step
        assert sarea[0.3] > 2 * sarea[0.1]

        isolist = ellipse.fit_image(sma_start=20.0, max_sma=40.0,
                                    min_sma=15.0, step=5.0,
                                    linear_growth=True)
        assert np.all([iso.sample.geometry.step == 5.0 for iso in isolist])

    def test_sampled_on_ellipse(self):
        """
        Test that every fitted isophote, including the diverged ones
        (stop_code 5) whose geometry is replaced by that of a neighbor,
        is sampled along its reported ellipse.
        """
        isophote_list = Ellipse(self.data).fit_image()
        assert np.any(isophote_list.stop_code == 5)
        for iso in isophote_list[1:]:
            angles, radii = iso.sample.angles, iso.sample.radii
            assert_allclose(radii, iso.sample.geometry.radius(angles))

    def test_masked_outer_isophote(self):
        """
        Test that the outward growth stops at an isophote that has no
        valid data points.

        The masked annulus contains the whole isophote at sma=33.8
        (step=0.3), but not the sample used to measure the gradient of
        the previous isophote. This used to repeat the same failed fit
        forever.
        """
        data = make_test_image(eps=0.0, seed=0)
        yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        radius = np.hypot(xx - 256, yy - 256)
        data = np.ma.masked_array(data, mask=(radius > 31) & (radius < 40))
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.05, 0.1 * u.rad)
        ellipse = Ellipse(data, geometry=geometry)
        match = 'Too small sample to warrant a fit'
        with pytest.warns(AstropyUserWarning, match=match):
            isolist = ellipse.fit_image(step=0.3, max_sma=80.0)
        assert_allclose(isolist.sma[-1], 26.0)

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
        g = EllipseGeometry(256.0, 256.0, 20.0, 0.2, 0.0 * u.rad)
        ellipse = Ellipse(self.data, geometry=g)
        isophote_list = ellipse.fit_image(integration_mode='nearest_neighbor',
                                          max_sma=30.0)

        assert isinstance(isophote_list, IsophoteList)
        assert len(isophote_list) > 1

    def test_offcenter_fail(self):
        # A first guess ellipse that is centered in the image frame.
        # This should result in failure since the real galaxy image is
        # off-center by a large offset.
        ellipse = Ellipse(OFFSET_GALAXY)

        match = (r'No meaningful fit was possible\. The first isophote '
                 r'\(sma=10\) failed with stop code 5 because the ellipse '
                 'diverged or no meaningful intensity gradient could be '
                 'measured.*try a slightly different starting semimajor '
                 r'axis length \(sma_start\).*low signal-to-noise ratio.*'
                 r'user_guide/isophote\.html#isophote-low-snr')
        with pytest.warns(AstropyUserWarning, match=match):
            isophote_list = ellipse.fit_image()
        assert len(isophote_list) == 0

    def test_first_isophote_flagged_fail(self):
        """
        Test the warning emitted when too many data points of the first
        isophote are flagged.

        A large part of the starting ellipse is outside of the image.
        """
        geometry = EllipseGeometry(10.0, 256.0, 50.0, 0.2, PA * u.rad)
        ellipse = Ellipse(self.data, geometry=geometry)
        match = (r'No meaningful fit was possible\. The first isophote '
                 r'\(sma=50\) failed with stop code 1 because the fraction '
                 r'of valid data points is smaller than '
                 r'min_valid_fraction \(0\.7\)')
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
            geometry = EllipseGeometry(256.0, 256.0, 2000.0, 0.2, PA * u.rad)
        ellipse = Ellipse(data, geometry=geometry)
        match1 = 'Too small sample to warrant a fit'
        match2 = (r'No meaningful fit was possible\. The first isophote '
                  r'\(sma=\d+\) failed with stop code 3 because it has too '
                  'few valid data points')
        with (pytest.warns(AstropyUserWarning, match=match1),
              pytest.warns(AstropyUserWarning, match=match2)):
            isophote_list = ellipse.fit_image()
        assert isinstance(isophote_list, IsophoteList)
        assert len(isophote_list) == 0

    def test_no_valid_data_noniterative(self):
        """
        Test the warning emitted when the first isophote is extracted
        without being fit and has no valid data points.
        """
        geometry = EllipseGeometry(256.0, 256.0, 2000.0, 0.2, PA * u.rad)
        ellipse = Ellipse(self.data, geometry=geometry)
        match = (r'The first isophote \(sma=2000\) failed with stop code 3 '
                 'because it has too few valid data points')
        with pytest.warns(AstropyUserWarning, match=match):
            isophote_list = ellipse.fit_image(max_iterative_sma=10.0)
        assert len(isophote_list) == 0

    def test_offcenter_fit(self):
        # A first guess ellipse that is roughly centered on the offset
        # galaxy image.
        g = EllipseGeometry(POS + 5, POS + 5, 10.0, ellipticity=0.2,
                            orientation=PA * u.rad, step=0.1)
        ellipse = Ellipse(OFFSET_GALAXY, geometry=g)
        isophote_list = ellipse.fit_image()

        # The fit should stop when too many potential sample points fall
        # outside the image frame.
        _assert_fflag_stop(isophote_list)
        _assert_sma_grid(isophote_list)

    def test_max_center_shift(self):
        # A companion source pulls the centers of the outer isophotes
        # away from the galaxy center.
        yy, xx = np.mgrid[0:512, 0:512]
        companion = Gaussian2D(50.0, 330, 290, 8.0, 8.0)(xx, yy)
        data = self.data + companion

        g = EllipseGeometry(256, 256, 20.0, ellipticity=0.2,
                            orientation=PA * u.rad)
        isolist = Ellipse(data, geometry=g).fit_image()
        shift = np.hypot(isolist.x0 - 256, isolist.y0 - 256)
        assert np.max(shift) > 5.0

        g = EllipseGeometry(256, 256, 20.0, ellipticity=0.2,
                            orientation=PA * u.rad)
        isolist_bound = Ellipse(data, geometry=g).fit_image(
            max_center_shift=1.0)
        shift = np.hypot(isolist_bound.x0 - 256, isolist_bound.y0 - 256)
        assert np.all(shift <= 1.0 + 1.0e-10)

        # Isophotes with the center held on the bound are still fitted,
        # and the fit covers the same isophotes as without the bound.
        at_bound = np.isclose(shift, 1.0)
        assert np.count_nonzero(at_bound) > 1
        assert np.all(isolist_bound.stop_code[at_bound] == 0)
        assert_allclose(isolist_bound.sma, isolist.sma)
        assert np.array_equal(isolist_bound.stop_code, isolist.stop_code)

        # Ellipticity is still fitted with the center on the bound, so
        # it follows a fit with a fixed center.
        g = EllipseGeometry(256, 256, 20.0, ellipticity=0.2,
                            orientation=PA * u.rad)
        isolist_fixed = Ellipse(data, geometry=g).fit_image(fix_center=True)
        assert_allclose(isolist_bound.ellipticity[at_bound],
                        isolist_fixed.ellipticity[at_bound], atol=0.01)

    def test_max_center_shift_fit_isophote(self):
        x0 = POS + 5
        y0 = POS + 5
        g = EllipseGeometry(x0, y0, 40.0, ellipticity=0.2,
                            orientation=PA * u.rad, step=0.1)
        ellipse = Ellipse(OFFSET_GALAXY, geometry=g)
        isophote = ellipse.fit_isophote(40.0, max_center_shift=2.0)

        geom = isophote.sample.geometry
        assert_allclose(np.hypot(geom.x0 - x0, geom.y0 - y0), 2.0)

    def test_offcenter_go_beyond_frame(self):
        # Same as before, but now force the fit to go beyond the image
        # frame limits.
        g = EllipseGeometry(POS + 5, POS + 5, 10.0, ellipticity=0.2,
                            orientation=PA * u.rad, step=0.1)
        ellipse = Ellipse(OFFSET_GALAXY, geometry=g)
        isophote_list = ellipse.fit_image(max_sma=400.0)

        # The fit should go to max_sma, but with fixed geometry
        # (noniterative mode) after too many sample points fall outside
        # the image frame. The isophotes with too many flagged points
        # have a stop code of 1, also in noniterative mode.
        sma = isophote_list.sma
        stop_code = isophote_list.stop_code
        assert sma[-1] < 400.0 <= sma[-1] * 1.1
        idx = np.nonzero(stop_code == 1)[0][0]
        assert np.all(stop_code[idx + 1:] == 1)
        assert np.all(isophote_list.n_iter[idx + 1:] == 0)
        assert np.all(stop_code[:idx + 1] != 4)
        _assert_sma_grid(isophote_list)

        # Check that no zero-valued intensities were left behind in the
        # sample arrays when sampling outside the image.
        for iso in isophote_list:
            assert not np.any(iso.sample.intensities == 0)

    def test_default_geometry(self):
        """
        Test the default geometry used when no geometry is input.
        """
        data = np.ones((50, 80))
        isophote = Ellipse(data).fit_isophote(10.0, noniterate=True)
        assert isophote.x0 == 39.5
        assert isophote.y0 == 24.5
        assert isophote.ellipticity == 0.2
        assert isophote.orientation == 0.0 * u.deg

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
        ellipse = Ellipse(data)  # starts from the image center
        isolist = ellipse.fit_image()

        # The isophotes are centered on the galaxy and have its shape.
        # The number of isophotes is not checked because the fits of
        # the noise-dominated outermost isophotes, which set where the
        # fit stops, depend on the platform.
        mask = (isolist.sma > 2) & (isolist.sma < 40)
        assert_allclose(isolist.x0[mask], nx / 2.0, atol=1.0)
        assert_allclose(isolist.y0[mask], ny / 2.0, atol=1.0)
        assert_allclose(np.median(isolist.ellipticity[mask]), 1.0 - 12 / 20,
                        atol=0.02)
        assert_allclose(np.median(isolist.orientation[mask]), 40.0 * u.deg,
                        atol=2.0 * u.deg)
        _assert_sma_grid(isolist)

    def test_round_starting_ellipse(self):
        """
        Test a starting ellipse with an ellipticity of exactly zero,
        for which the position angle correction diverges.
        """
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.0, 0.0 * u.rad)
        isolist = Ellipse(self.data, geometry=geometry).fit_image(
            max_sma=40.0)
        mask = isolist.sma > 5
        assert np.count_nonzero(mask) > 10
        assert_allclose(isolist.ellipticity[mask], 0.2, atol=0.01)
        assert_allclose(isolist.orientation[mask].to_value(u.rad), PA,
                        atol=0.02)

    def test_fixed_round_ellipse(self):
        """
        Test that a fixed ellipticity of zero is not changed.
        """
        geometry = EllipseGeometry(256.5, 255.5, 20.0, 0.0, 0.3 * u.rad,
                                   fix_ellipticity=True)
        isolist = Ellipse(self.data, geometry=geometry).fit_image(
            max_sma=40.0)
        assert len(isolist) > 10
        assert np.all(isolist.ellipticity == 0.0)
        mask = isolist.sma > 5
        assert np.all(isolist.orientation[mask] == 0.3 * u.rad)
        assert_allclose(isolist.x0[mask], 256.0, atol=0.1)
        assert_allclose(isolist.y0[mask], 256.0, atol=0.1)

    def test_first_inward_sma(self):
        """
        Test that the first inward isophote is not fit if its semimajor
        axis length is below the minimum.
        """
        # A linear step larger than the starting sma
        geometry = EllipseGeometry(256.0, 256.0, 3.0, 0.2, PA * u.rad)
        isolist = Ellipse(self.data, geometry=geometry).fit_image(
            step=5.0, linear_growth=True, max_sma=15.0)
        assert_allclose(isolist.sma, [0.0, 3.0, 8.0, 13.0])

        # The central pixel is at the center of the innermost fitted
        # isophote, not of the last fitted (outermost) one.
        assert isolist[0].x0 == isolist[1].x0
        assert isolist[0].y0 == isolist[1].y0
        assert isolist[0].x0 != isolist[-1].x0

        geometry = EllipseGeometry(256.0, 256.0, 10.0, 0.2, PA * u.rad)
        isolist = Ellipse(self.data, geometry=geometry).fit_image(
            min_sma=9.5, max_sma=12.0)
        assert_allclose(isolist.sma, [10.0, 11.0])

    @pytest.mark.parametrize(('step', 'min_sma', 'max_sma', 'expected'), [
        (1.0, 15.0, 25.0, np.arange(15.0, 26.0)),
        (0.1, 19.5, 20.5, np.linspace(19.5, 20.5, 11)),
        (1.0, 20.0, 20.0, [20.0]),
    ])
    def test_sma_limits_inclusive_linear(self, step, min_sma, max_sma,
                                         expected):
        """
        Test that isophotes are fit at the ``min_sma`` and ``max_sma``
        values when the semimajor axis grid reaches them.

        A step of 0.1 reaches the limits only to within rounding
        errors.
        """
        isolist = Ellipse(self.data).fit_image(
            sma_start=20.0, min_sma=min_sma, max_sma=max_sma, step=step,
            linear_growth=True)
        assert_allclose(isolist.sma, expected)

    def test_sma_limits_inclusive_geometric(self):
        """
        Test that the ``min_sma`` and ``max_sma`` limits are inclusive
        for geometric growth, where the semimajor axis grid reaches
        them only to within rounding errors.
        """
        factor = 1.1**np.arange(-3, 4)
        for scale in (1.0 - 1.0e-13, 1.0, 1.0 + 1.0e-13):
            isolist = Ellipse(self.data).fit_image(
                sma_start=20.0, min_sma=20.0 * factor[0] / scale,
                max_sma=20.0 * factor[-1] * scale, step=0.1)
            assert_allclose(isolist.sma, 20.0 * factor)

    def test_sma_limits_small_minimum(self):
        """
        Test that the smallest fitted semimajor axis length is 0.5
        pixels, which is the lower limit when ``min_sma`` is smaller.
        """
        isolist = Ellipse(self.data).fit_image(
            sma_start=2.0, max_sma=2.0, step=0.5, linear_growth=True)
        assert_allclose(isolist.sma, [0.0, 0.5, 1.0, 1.5, 2.0])

    @pytest.mark.parametrize('step', [0.0, -0.1, np.nan])
    def test_invalid_step(self, step):
        match = 'must be positive'
        ellipse = Ellipse(self.data)
        with pytest.raises(ValueError, match=match):
            ellipse.fit_image(step=step)
        with pytest.raises(ValueError, match=match):
            ellipse.fit_isophote(20.0, step=step)

        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, PA * u.rad,
                                   step=step)
        ellipse = Ellipse(self.data, geometry=geometry)
        with pytest.raises(ValueError, match=match):
            ellipse.fit_image()
        with pytest.raises(ValueError, match=match):
            ellipse.fit_isophote(20.0)

    @pytest.mark.parametrize('sma', [-1.0, np.nan, np.inf])
    @pytest.mark.parametrize('noniterate', [False, True])
    def test_fit_isophote_invalid_sma(self, sma, noniterate):
        """
        Test that a negative or nonfinite semimajor axis length is
        rejected instead of returning the central isophote or failing
        inside the sampling code.
        """
        match = 'sma must be finite and nonnegative'
        ellipse = Ellipse(self.data)
        with pytest.raises(ValueError, match=match):
            ellipse.fit_isophote(sma, noniterate=noniterate)

    def test_fixed_pa_round_galaxy(self):
        """
        Test that a fixed position angle is not rotated by 90 degrees
        when the ellipticity corrections cross zero.
        """
        data = make_test_image(eps=0.0, noise=1.0, seed=0)
        pa = 0.3
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.05, pa * u.rad,
                                   fix_orientation=True)
        isolist = Ellipse(data, geometry=geometry).fit_image(max_sma=60.0)
        mask = isolist.sma > 0
        assert np.count_nonzero(mask) > 40
        assert np.all(isolist.orientation[mask] == pa * u.rad)
        assert np.all(isolist.ellipticity[mask] > 0)
        assert np.median(isolist.ellipticity[isolist.sma > 5]) < 0.06

    def test_max_sma_outside_image(self):
        """
        Test that the outward growth in noniterative mode stops at an
        isophote that lies entirely outside of the image.

        The isophotes extracted beyond the image have too many flagged
        data points (stop code 1).
        """
        data = make_test_image(nx=200, ny=200, pa=PA, noise=1.0, seed=0)
        isolist = Ellipse(data).fit_image(max_sma=400.0)
        assert isolist[-1].stop_code == 1
        assert np.all(isolist.n_data > 0)
        assert np.all(np.isfinite(isolist.intensity))
        assert 150 < isolist.sma[-1] < 220

    def test_fit_isophote_linear(self):
        """
        Test that fit_isophote uses the linear_growth attribute of the
        geometry unless the linear keyword is set.
        """
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, PA * u.rad,
                                   linear_growth=True)
        ellipse = Ellipse(self.data, geometry=geometry)
        isophote = ellipse.fit_isophote(20.0, step=2.0)
        assert isophote.sample.geometry.linear_growth
        assert isophote.sample.geometry.step == 2.0

        isophote = ellipse.fit_isophote(20.0, linear_growth=False)
        assert not isophote.sample.geometry.linear_growth

        assert geometry.linear_growth
        assert geometry.step == 0.1

    def test_linear_numpy_bool(self):
        """
        Test that fit_image and fit_isophote accept a NumPy boolean
        for the linear keyword.
        """
        ellipse = Ellipse(self.data)
        isolist = ellipse.fit_image(sma_start=20.0, min_sma=10.0,
                                    max_sma=40.0, step=5.0,
                                    linear_growth=np.True_)
        assert_allclose(isolist.sma, np.arange(10.0, 41.0, 5.0))
        isophote = ellipse.fit_isophote(20.0, linear_growth=np.True_)
        assert isophote.sample.geometry.linear_growth is True


class TestMeasureIsophotes:
    def setup_class(self):
        self.data = make_test_image(pa=PA, seed=0)
        self.isolist = Ellipse(self.data).fit_image()

    def test_same_image(self):
        """
        Test that measuring the fitted image along its own isophotes
        reproduces the fitted intensities.
        """
        ellipse = Ellipse(self.data)
        isolist = ellipse.measure_isophotes(self.isolist)

        assert isinstance(isolist, IsophoteList)
        assert len(isolist) == len(self.isolist)
        for name in ('sma', 'x0', 'y0', 'ellipticity', 'orientation'):
            assert_allclose(getattr(isolist, name),
                            getattr(self.isolist, name))
        assert_equal(isolist.n_iter, 0)

        # The diverged isophotes (stop_code 5), whose geometry is
        # replaced by that of a neighbor, are included.
        assert np.any(self.isolist.stop_code == 5)
        assert_allclose(isolist.intensity, self.isolist.intensity)

        # The central pixel is a plain Isophote with stop_code 0.
        assert type(isolist[0]) is Isophote
        assert isolist[0].sma == 0.0
        assert_equal(isolist.stop_code[1:], 4)

    def test_scaled_image(self):
        """
        Test measuring a scaled image along the isophotes of a reference
        image.
        """
        isolist1 = Ellipse(self.data).measure_isophotes(self.isolist)
        isolist2 = Ellipse(2.0 * self.data).measure_isophotes(self.isolist)
        assert_allclose(isolist2.intensity, 2.0 * isolist1.intensity)
        assert_allclose(isolist2.ellipticity, self.isolist.ellipticity)
        assert_allclose(isolist2.orientation, self.isolist.orientation)

    def test_input_unchanged(self):
        """
        Test that the input isophotes and geometries are not modified.
        """
        names = ('sma', 'x0', 'y0', 'ellipticity', 'orientation',
                 'intensity')
        values = {name: getattr(self.isolist, name).copy()
                  for name in names}
        fixed = [iso.sample.geometry._fix.copy() for iso in self.isolist]
        ellipse = Ellipse(2.0 * self.data)
        isolist = ellipse.measure_isophotes(self.isolist)

        for name in names:
            assert_equal(getattr(self.isolist, name), values[name])
        for iso, iso_ref, fix in zip(isolist, self.isolist, fixed,
                                     strict=True):
            assert_equal(iso_ref.sample.geometry._fix, fix)
            assert iso.sample.geometry is not iso_ref.sample.geometry

    def test_geometries(self):
        """
        Test that EllipseGeometry inputs give the same result as the
        equivalent Isophote inputs, sorted by semimajor axis length.
        """
        ellipse = Ellipse(self.data)
        isolist1 = ellipse.measure_isophotes(self.isolist[1:])
        geometries = [iso.sample.geometry for iso in self.isolist[1:]]
        isolist2 = ellipse.measure_isophotes(geometries[::-1])

        assert len(isolist2) == len(isolist1)
        for name in ('sma', 'x0', 'y0', 'ellipticity', 'orientation',
                     'intensity'):
            assert_allclose(getattr(isolist2, name),
                            getattr(isolist1, name))

    def test_user_geometries(self):
        """
        Test user-defined geometries, including mixed Isophote and
        EllipseGeometry inputs.
        """
        geometries = [EllipseGeometry(256.0, 256.0, sma, 0.2, PA * u.rad)
                      for sma in (0.0, 10.0, 20.0)]
        ellipse = Ellipse(self.data)
        isolist = ellipse.measure_isophotes([self.isolist[-1],
                                             *geometries])

        assert len(isolist) == 4
        assert_allclose(isolist.sma[:3], [0.0, 10.0, 20.0])
        assert_allclose(isolist.ellipticity[1:3], 0.2)
        assert_allclose(isolist.orientation[1:3].to_value(u.rad), PA)
        assert_allclose(isolist.x0[:3], 256.0)
        assert isolist[-1].sma == self.isolist[-1].sma
        assert np.all(np.diff(isolist.intensity) < 0)

    def test_measure_options(self):
        """
        Test that the sampling options are passed to the samples.
        """
        ellipse = Ellipse(self.data)
        sigma_clip = SigmaClip(sigma=2.5, maxiters=3)
        isolist = ellipse.measure_isophotes(self.isolist[1:5],
                                            integration_mode='median',
                                            sigma_clip=sigma_clip)
        for iso in isolist:
            assert iso.sample.integration_mode == 'median'
            assert iso.sample.sigma_clip.sigma == 2.5
            assert iso.sample.sigma_clip.maxiters == 3

    def test_outside_image(self):
        """
        Test that an ellipse entirely outside the image is kept with a
        NaN intensity.
        """
        geometry = EllipseGeometry(-500.0, -500.0, 20.0, 0.2, PA * u.rad)
        ellipse = Ellipse(self.data)
        isolist = ellipse.measure_isophotes([geometry])

        assert len(isolist) == 1
        assert np.isnan(isolist[0].intensity)
        assert isolist[0].n_data == 0
        assert isolist[0].stop_code == 3

    def test_empty(self):
        ellipse = Ellipse(self.data)
        isolist = ellipse.measure_isophotes([])
        assert isinstance(isolist, IsophoteList)
        assert len(isolist) == 0

    def test_invalid_input(self):
        ellipse = Ellipse(self.data)
        match = 'must contain only Isophote or EllipseGeometry'
        with pytest.raises(TypeError, match=match):
            ellipse.measure_isophotes([self.isolist[1], 10.0])

    def test_invalid_sma(self):
        """
        Test that a geometry with a negative semimajor axis length is
        rejected.

        The geometry rejects such a length, so it is set on the sma
        attribute after the geometry is created.
        """
        geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, PA * u.rad)
        geometry.sma = -1.0
        ellipse = Ellipse(self.data)
        match = 'sma must be finite and nonnegative'
        with pytest.raises(ValueError, match=match):
            ellipse.measure_isophotes([geometry])


# The values differ from the defaults enough to change the fit result.
FIT_IMAGE_RENAMES = [
    ('sma0', 'sma_start', 20.0), ('minsma', 'min_sma', 5.0),
    ('maxsma', 'max_sma', 60.0), ('conver', 'convergence', 0.1),
    ('minit', 'miniters', 5), ('maxit', 'maxiters', 5),
    ('fflag', 'min_valid_fraction', 0.5),
    ('maxgerr', 'max_gradient_rel_err', 0.001),
    ('integrmode', 'integration_mode', 'mean'),
    ('linear', 'linear_growth', True),
    ('maxrit', 'max_iterative_sma', 50.0),
    ('fix_pa', 'fix_orientation', True),
    ('fix_eps', 'fix_ellipticity', True),
]

FIT_ISOPHOTE_RENAMES = [
    ('conver', 'convergence', 1.0e-4), ('minit', 'miniters', 5),
    ('maxit', 'maxiters', 5), ('fflag', 'min_valid_fraction', 0.5),
    ('integrmode', 'integration_mode', 'mean'),
    ('linear', 'linear_growth', True),
    ('maxrit', 'max_iterative_sma', 20.0),
]


def _isophote_list_values(isolist):
    """
    Return the values that tell whether two isophote lists differ.
    """
    return [isolist.sma, isolist.intensity, isolist.ellipticity,
            isolist.orientation.value, isolist.x0, isolist.y0,
            isolist.n_iter, isolist.stop_code]


def _isophote_lists_equal(isolist1, isolist2):
    return len(isolist1) == len(isolist2) and all(
        np.array_equal(values1, values2, equal_nan=True)
        for values1, values2 in zip(_isophote_list_values(isolist1),
                                    _isophote_list_values(isolist2),
                                    strict=True))


def _isophotes_equal(isophote1, isophote2):
    """
    Return whether two isophotes have the same values.
    """
    values1, values2 = (
        np.array([isophote.sma, isophote.intensity, isophote.ellipticity,
                  isophote.orientation.value, isophote.x0, isophote.y0,
                  isophote.n_iter, isophote.stop_code])
        for isophote in (isophote1, isophote2))
    return np.array_equal(values1, values2, equal_nan=True)


@pytest.mark.parametrize(('old', 'new', 'value'), FIT_IMAGE_RENAMES)
def test_fit_image_renamed_keywords(old, new, value):
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    # A linear step of 0.1 pixels would fit thousands of isophotes.
    extra = {'linear_growth': {'step': 5.0}}.get(new, {})
    expected = ellipse.fit_image(**{new: value}, **extra)
    # The value must change the result, so that a shim that dropped
    # the keyword would fail.
    assert not _isophote_lists_equal(expected, ellipse.fit_image(**extra))
    with pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"):
        result = ellipse.fit_image(**{old: value}, **extra)
    assert _isophote_lists_equal(result, expected)
    with (pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"),
          pytest.raises(TypeError)):
        ellipse.fit_image(**{old: value, new: value}, **extra)


@pytest.mark.parametrize(('old', 'new', 'value'), FIT_ISOPHOTE_RENAMES)
def test_fit_isophote_renamed_keywords(old, new, value):
    data = make_test_image(seed=0)
    geometry = EllipseGeometry(256.0, 256.0, 10.0, 0.2, 0.0 * u.deg)
    ellipse = Ellipse(data, geometry=geometry)
    # The minimum valid fraction has an effect only when data points
    # are flagged. Linear growth with the default step narrows the
    # sampling annulus. In the 'bilinear' mode, this only shifts the
    # starting angle of the path slightly. The 'mean' mode falls back to
    # the 'bilinear' mode for the narrow annulus, which changes the
    # result clearly.
    sigma_clip = SigmaClip(sigma=1.0, maxiters=1, cenfunc='mean',
                           stdfunc='std')
    extra = {'min_valid_fraction': {'sigma_clip': sigma_clip},
             'linear_growth': {'integration_mode': 'mean'}}.get(new, {})
    expected = ellipse.fit_isophote(40.0, **{new: value}, **extra)
    # The value must change the result, so that a shim that dropped
    # the keyword would fail.
    default = ellipse.fit_isophote(40.0, **extra)
    assert not _isophotes_equal(expected, default)
    with pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"):
        result = ellipse.fit_isophote(40.0, **{old: value}, **extra)
    assert _isophotes_equal(result, expected)
    with (pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"),
          pytest.raises(TypeError)):
        ellipse.fit_isophote(40.0, **{old: value, new: value}, **extra)


# The deprecated sigma-clipping keywords
LEGACY_CLIP_KWARGS = [{'sclip': 2.5, 'n_clip': 3},
                      {'sclip': 2.5, 'nclip': 3}]


def _legacy_sigma_clip():
    """
    Return the sigma clipping equivalent to ``LEGACY_CLIP_KWARGS``.
    """
    return SigmaClip(sigma=2.5, maxiters=3, cenfunc='mean', stdfunc='std')


@pytest.mark.parametrize('kwargs', LEGACY_CLIP_KWARGS)
def test_fit_image_deprecated_clip_keywords(kwargs):
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    expected = ellipse.fit_image(max_sma=60.0,
                                 sigma_clip=_legacy_sigma_clip())
    # The clipping must change the result, so that a shim that dropped
    # the keywords would fail.
    assert not _isophote_lists_equal(expected,
                                     ellipse.fit_image(max_sma=60.0))
    match = "Use argument 'sigma_clip' instead"
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        result = ellipse.fit_image(max_sma=60.0, **kwargs)
    assert len(record) == 2
    assert _isophote_lists_equal(result, expected)


@pytest.mark.parametrize('kwargs', LEGACY_CLIP_KWARGS)
def test_fit_isophote_deprecated_clip_keywords(kwargs):
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    expected = ellipse.fit_isophote(40.0, sigma_clip=_legacy_sigma_clip())
    assert not _isophotes_equal(expected, ellipse.fit_isophote(40.0))
    match = "Use argument 'sigma_clip' instead"
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        result = ellipse.fit_isophote(40.0, **kwargs)
    assert len(record) == 2
    assert _isophotes_equal(result, expected)


def test_fit_image_positional_clip_values():
    """
    Test that the deprecated sigma-clipping values keep their
    positions in the argument list.
    """
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    expected = ellipse.fit_image(max_sma=60.0,
                                 sigma_clip=_legacy_sigma_clip())
    with pytest.warns(PhotutilsDeprecationWarning) as record:
        result = ellipse.fit_image(None, 0.0, 60.0, 0.1, 0.05, 10, 50, 0.7,
                                   0.5, 2.5, 3)
    messages = sorted(str(item.message) for item in record)
    assert len(messages) == 3
    assert messages[0].startswith("'n_clip' was deprecated")
    assert messages[1].startswith("'sclip' was deprecated")
    assert 'positional' in messages[2]
    assert _isophote_lists_equal(result, expected)


def test_invalid_sigma_clip():
    """
    Test that an invalid sigma_clip raises an error even if no sample is
    made.
    """
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    match = 'sigma_clip must be a SigmaClip instance'
    with pytest.raises(TypeError, match=match):
        ellipse.fit_image(fix_center=True, fix_orientation=True,
                          fix_ellipticity=True, sigma_clip=3.0)
    with pytest.raises(TypeError, match=match):
        ellipse.measure_isophotes([], sigma_clip=3.0)
    with pytest.raises(TypeError, match=match):
        ellipse.fit_isophote(40.0, sigma_clip=3.0)


@pytest.mark.parametrize('method', ['fit_image', 'fit_isophote'])
def test_deprecated_clip_keywords_with_sigma_clip(method):
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    args = (40.0,) if method == 'fit_isophote' else ()
    match = 'sigma_clip cannot be used with the deprecated'
    with (pytest.warns(PhotutilsDeprecationWarning, match="'n_clip'"),
          pytest.raises(ValueError, match=match)):
        getattr(ellipse, method)(*args, sigma_clip=SigmaClip(), n_clip=3)


def test_fit_image_positional_order_unchanged():
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    expected = ellipse.fit_image(sma_start=20.0, min_sma=5.0,
                                 max_sma=60.0)
    with pytest.warns(PhotutilsDeprecationWarning, match='positional'):
        result = ellipse.fit_image(20.0, 5.0, 60.0)
    assert_array_equal(result.sma, expected.sma)


def test_measure_isophotes_keywords_have_no_shim():
    data = make_test_image(seed=0)
    ellipse = Ellipse(data)
    isolist = ellipse.fit_image()
    for old in ('sclip', 'n_clip', 'integrmode'):
        with pytest.raises(TypeError):
            ellipse.measure_isophotes(isolist, **{old: 3.0})
    result = ellipse.measure_isophotes(isolist, sigma_clip=SigmaClip(),
                                       integration_mode='mean')
    assert len(result) == len(isolist)


def test_fit_image_default_geometry_flattened_galaxy():
    """
    Test the default geometry for a galaxy that is twice as flattened.

    The default ellipticity of 0.2 is half of the galaxy ellipticity,
    so the orientation correction of the first isophote overshoots.
    """
    orientation = 40.0
    data = make_test_image(eps=0.4, pa=np.deg2rad(orientation), seed=0)
    isolist = Ellipse(data).fit_image()
    idx = (isolist.sma > 10.0) & (isolist.sma < 100.0)
    assert np.all(isolist.stop_code[idx] == 0)
    assert_allclose(isolist.ellipticity[idx], 0.4, atol=0.01)
    assert_allclose(isolist.orientation[idx].to_value(u.deg), orientation,
                    atol=0.5)
