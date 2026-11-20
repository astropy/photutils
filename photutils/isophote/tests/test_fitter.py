# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the fitter module.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.datasets.load import _get_path
from photutils.isophote.fitter import CentralEllipseFitter, EllipseFitter
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.harmonics import fit_first_and_second_harmonics
from photutils.isophote.integrator import MEAN
from photutils.isophote.isophote import Isophote
from photutils.isophote.sample import CentralEllipseSample, EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning

DATA = make_test_image(seed=0)
DEFAULT_POS = 256

DEFAULT_FIX = np.array([False, False, False, False])


def test_gradient():
    sample = EllipseSample(DATA, 40.0)
    sample.update(fixed_parameters=DEFAULT_FIX)

    assert_allclose(sample.mean, 200.02, atol=0.01)
    assert_allclose(sample.gradient, -4.222, atol=0.001)
    assert_allclose(sample.gradient_err, 0.0003, atol=0.0001)
    assert_allclose(sample.gradient_rel_err, 7.45e-05, atol=1.0e-5)
    assert_allclose(sample.sector_area, 2.00, atol=0.01)


def test_zero_gradient_flat_image():
    """
    Test that the fitter handles a zero gradient without crashing.

    On a perfectly flat image, the nearest-neighbor integrator extracts
    identical pixel values at all radii, so the local gradient is
    exactly zero. Since no parameter corrections can be computed from a
    zero gradient, the fit should stop cleanly instead of dividing by
    zero and crashing on the resulting nonfinite geometry.
    """
    data = np.ones((101, 101))
    sample = EllipseSample(data, 10.0, integrmode='nearest_neighbor')
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isophote.valid
    assert isophote.stop_code == -1
    assert isophote.sample.gradient == 0.0
    assert isophote.intens == 1.0


def test_positive_gradient():
    """
    Test that the fitter does not apply corrections computed from a
    positive gradient.

    The intensity of this image increases outward, so the initial
    ellipse is returned unchanged.
    """
    yy, xx = np.mgrid[0:201, 0:201]
    data = 10.0 + 0.5 * np.hypot(xx - 100, yy - 100)
    sample = EllipseSample(data, 20.0, x0=100.0, y0=100.0, eps=0.2)
    isophote = EllipseFitter(sample).fit()
    assert isophote.stop_code == -1
    assert isophote.grad > 0
    assert isophote.eps == 0.2
    assert isophote.x0 == 100.0
    assert isophote.y0 == 100.0


@pytest.mark.parametrize('ring_radius', [21.0, 23.0])
def test_nonmeaningful_gradient_uses_previous(ring_radius):
    """
    Test that the iterations continue when the gradient first measured
    for a corrected ellipse is not meaningful.

    A bright ring just outside of the fitted ellipse distorts the
    gradient. For a ring radius of 21, the gradient measured for the
    third ellipse is 20 times shallower than the gradient of the
    previous iteration and has a large error. The gradient measured
    at a larger radius is used instead. For a ring radius of 23, the
    gradient measured for the fourth ellipse is positive at both
    radii. The gradient of the previous iteration is then used instead,
    made slightly shallower.
    """
    yy, xx = np.mgrid[0:201, 0:201]
    radius = np.hypot(xx - 100, yy - 100)
    data = (1000.0 * np.exp(-radius / 10.0)
            + 100.0 * np.exp(-0.5 * (radius - ring_radius)**2))
    sample = EllipseSample(data, 20.0, x0=101.0, y0=100.0, eps=0.1)
    isophote = EllipseFitter(sample).fit()
    assert isophote.stop_code != -1
    assert isophote.n_iter > 4
    assert isophote.grad < 0


def test_nonmeaningful_gradient_center_at_bound():
    """
    Test that the gradient of the previous iteration is also used for
    a corrected ellipse whose center was moved onto the
    ``max_center_shift`` boundary.
    """
    yy, xx = np.mgrid[0:201, 0:201]
    radius = np.hypot(xx - 100, yy - 100)
    data = (1000.0 * np.exp(-radius / 10.0)
            + 100.0 * np.exp(-0.5 * (radius - 23.0)**2))
    sample = EllipseSample(data, 20.0, x0=101.0, y0=100.0, eps=0.1)
    isophote = EllipseFitter(sample).fit(max_center_shift=0.5)
    assert isophote.stop_code != -1
    assert isophote.n_iter > 5
    assert isophote.grad < 0
    assert_allclose(isophote.x0, 100.5)


def test_everything_fixed():
    """
    Test that the isophote is extracted with the input geometry,
    without fitting, if all the ellipse parameters are fixed.
    """
    geometry = EllipseGeometry(250.0, 260.0, 40.0, 0.3, 1.0, fix_center=True,
                               fix_pa=True, fix_eps=True)
    sample = EllipseSample(DATA, 40.0, geometry=geometry)
    isophote = EllipseFitter(sample).fit()
    assert isophote.stop_code == 4
    assert isophote.n_iter == 0
    assert (isophote.x0, isophote.y0, isophote.eps, isophote.pa) == (
        250.0, 260.0, 0.3, 1.0)


def test_fit_keeps_sample_settings():
    """
    Test that the sample of the fitted isophote, which is made by the
    parameter correctors, keeps the settings of the input sample.
    """
    sample = EllipseSample(DATA, 40.0, astep=0.2, eps=0.3, sclip=2.5,
                           n_clip=2, integrmode=MEAN)
    isophote = EllipseFitter(sample).fit()
    fitted_sample = isophote.sample
    assert isophote.stop_code == 0
    assert fitted_sample is not sample
    assert fitted_sample.sclip == 2.5
    assert fitted_sample.n_clip == 2
    assert fitted_sample.integrmode == MEAN
    assert fitted_sample.geometry.sma == 40.0
    assert fitted_sample.geometry.astep == 0.2
    assert not fitted_sample.geometry.linear_growth


def test_fitting_raw():
    """
    This test performs a raw (no EllipseFitter), 1-step correction in
    one single ellipse coefficient.
    """
    # Pick first guess ellipse that is off in just one of the parameters
    # (eps).
    sample = EllipseSample(DATA, 40.0, eps=2 * 0.2)
    sample.update(fixed_parameters=DEFAULT_FIX)
    s = sample.extract()

    harmonics = fit_first_and_second_harmonics(s[0], s[2])
    _, a1, b1, a2, b2 = harmonics[0]

    # When eps is off, b2 is the largest (in absolute value).
    assert abs(b2) > abs(a1)
    assert abs(b2) > abs(b1)
    assert abs(b2) > abs(a2)

    correction = (b2 * 2.0 * (1.0 - sample.geometry.eps)
                  / sample.geometry.sma / sample.gradient)
    new_eps = sample.geometry.eps - correction

    # Got closer to test data (eps=0.2)
    assert_allclose(new_eps, 0.21, atol=0.01)


def test_fitting_small_radii():
    sample = EllipseSample(DATA, 2.0)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    assert isophote.n_data == 13


def test_fitting_small_sample():
    """
    Test that a sample with fewer data points than harmonic
    coefficients is not fit.
    """
    data = np.ma.masked_array(DATA, mask=np.ones(DATA.shape, dtype=bool))
    data.mask[255:258, 295:298] = False
    sample = EllipseSample(data, 40.0)
    assert 0 < len(sample.extract()[2]) < 5

    match = 'Too small sample to warrant a fit'
    with pytest.warns(AstropyUserWarning, match=match):
        isophote = EllipseFitter(sample).fit()
    assert not isophote.valid
    assert isophote.stop_code == 3


def test_fitting_failed_harmonic_fit(monkeypatch):
    """
    Test that a failed harmonic fit gives an invalid isophote.
    """
    def fit_harmonics(*args):  # noqa: ARG001
        msg = 'Error in least-squares fit'
        raise RuntimeError(msg)

    monkeypatch.setattr('photutils.isophote.fitter.'
                        'fit_first_and_second_harmonics', fit_harmonics)
    sample = EllipseSample(DATA, 40.0)
    match = 'The harmonic fit failed at SMA 40.0: Error in least-squares'
    with pytest.warns(AstropyUserWarning, match=match):
        isophote = EllipseFitter(sample).fit()
    assert not isophote.valid
    assert isophote.stop_code == 3


def test_fitting_eps():
    # Initial guess is off in the eps parameter.
    sample = EllipseSample(DATA, 40.0, eps=2 * 0.2)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    g = isophote.sample.geometry
    assert_allclose(g.eps, 0.2, atol=0.01)


@pytest.mark.parametrize('galaxy_eps', [0.97, 0.98])
def test_fitting_eps_diverged(galaxy_eps):
    """
    Test that the fit stops when a correction makes the ellipticity
    exceed its maximum allowed value of 0.95.

    The best-fit sample is returned instead of the diverged one.
    """
    data = make_test_image(eps=galaxy_eps, seed=0)
    sample = EllipseSample(data, 40.0, eps=0.9)
    isophote = EllipseFitter(sample).fit()
    assert isophote.stop_code == -1
    assert isophote.eps == 0.9


@pytest.mark.parametrize('kwargs', [{'maxgerr': 0.5},
                                    {'going_inwards': True}])
def test_fit_deprecated_keywords(kwargs):
    sample = EllipseSample(DATA, 40.0)
    match = f"'{next(iter(kwargs))}' was deprecated"
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        EllipseFitter(sample).fit(**kwargs)


def test_fitting_pa():
    data = make_test_image(pa=np.pi / 4, noise=0.01, seed=0)

    # Initial guess is off in the pa parameter.
    sample = EllipseSample(data, 40)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()
    g = isophote.sample.geometry

    assert_allclose(g.pa, np.pi / 4, atol=0.05)


def test_fitting_xy():
    pos = DEFAULT_POS - 5
    data = make_test_image(x0=pos, y0=pos, seed=0)

    # Initial guess is off in the x0 and y0 parameters.
    sample = EllipseSample(data, 40)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()
    g = isophote.sample.geometry

    assert_allclose((g.x0, g.y0), pos, atol=1.0)


def test_fitting_all():
    # Build test image that is off from the defaults assumed by the
    # EllipseSample constructor.
    pos = DEFAULT_POS - 5
    angle = np.pi / 4
    eps = 2 * 0.2
    data = make_test_image(x0=pos, y0=pos, eps=eps, pa=angle, seed=0)
    sma = 60.0

    # Initial guess is off in all parameters. We find that the initial
    # guesses, especially for position angle, must be kinda close to the
    # actual value. 20% off max seems to work in this case of high SNR.
    sample = EllipseSample(data, sma, position_angle=(1.2 * angle))
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isophote.stop_code == 0

    g = isophote.sample.geometry
    assert_allclose((g.x0, g.y0), pos, atol=1.5)  # within 1.5 pixel
    assert_allclose(g.eps, eps, atol=0.01)
    assert_allclose(g.pa, angle, atol=0.05)  # within ~3 deg

    sample_m = EllipseSample(data, sma, position_angle=(1.2 * angle),
                             integrmode=MEAN)
    fitter_m = EllipseFitter(sample_m)
    isophote_m = fitter_m.fit()

    assert isophote_m.stop_code == 0


class TestMaxCenterShift:
    def setup_class(self):
        # The galaxy is offset by (-5, -5) pixels from the default
        # initial guess at (256, 256).
        self.pos = DEFAULT_POS - 5
        self.data = make_test_image(x0=self.pos, y0=self.pos, seed=0)

    def test_center_held_at_bound(self):
        sample = EllipseSample(self.data, 40)
        isophote = EllipseFitter(sample).fit(max_center_shift=2.0)

        g = isophote.sample.geometry
        shift = np.hypot(g.x0 - DEFAULT_POS, g.y0 - DEFAULT_POS)
        assert_allclose(shift, 2.0)

        # The center is held fixed only within this fit
        assert np.array_equal(g.fix, DEFAULT_FIX)

    def test_shift_within_bound(self):
        # A bound larger than the true offset does not change the fit
        sample1 = EllipseSample(self.data, 40)
        isophote1 = EllipseFitter(sample1).fit()
        sample2 = EllipseSample(self.data, 40)
        isophote2 = EllipseFitter(sample2).fit(max_center_shift=20.0)

        g1 = isophote1.sample.geometry
        g2 = isophote2.sample.geometry
        assert g1.x0 == g2.x0
        assert g1.y0 == g2.y0
        assert g1.eps == g2.eps
        assert g1.pa == g2.pa

    def test_reference_center(self):
        reference = (DEFAULT_POS + 4, DEFAULT_POS + 4)
        sample = EllipseSample(self.data, 40)
        isophote = EllipseFitter(sample).fit(max_center_shift=6.0,
                                             reference_center=reference)

        g = isophote.sample.geometry
        shift = np.hypot(g.x0 - reference[0], g.y0 - reference[1])
        assert_allclose(shift, 6.0)

    def test_all_parameters_fixed_at_bound(self):
        # With PA and ellipticity fixed, holding the center at the bound
        # leaves nothing to fit, so the fit stops there.
        sample = EllipseSample(self.data, 40)
        sample.geometry.fix = np.array([False, False, True, True])
        isophote = EllipseFitter(sample).fit(max_center_shift=1.0)

        g = isophote.sample.geometry
        shift = np.hypot(g.x0 - DEFAULT_POS, g.y0 - DEFAULT_POS)
        assert_allclose(shift, 1.0)
        assert isophote.stop_code == 0
        assert np.array_equal(g.fix, [False, False, True, True])

    @pytest.mark.parametrize('value', [-1.0, np.nan])
    def test_invalid(self, value):
        sample = EllipseSample(self.data, 40)
        match = 'max_center_shift must be a non-negative number'
        with pytest.raises(ValueError, match=match):
            EllipseFitter(sample).fit(max_center_shift=value)


@pytest.mark.remote_data
class TestM51:
    def setup_class(self):
        path = _get_path('isophote/M51.fits', location='photutils-datasets',
                         cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()

    def test_m51(self):
        # Here we evaluate the detailed convergence behavior for a
        # particular ellipse where we can see the eps parameter jumping
        # back and forth. We start the fit with initial values taken
        # from previous isophote, as determined by the old code.

        # Sample taken in high SNR region
        sample = EllipseSample(self.data, 21.44, eps=0.18,
                               position_angle=np.deg2rad(36.0))
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert isophote.n_data == 119
        assert_allclose(isophote.intens, 685.4, atol=0.1)

        # Last sample taken by the original code, before turning
        # inwards. The original code ran this fit for the maximum
        # number of iterations and measured an intensity of 153, an
        # ellipticity of 0.015, and 380 data points. The corrected
        # ellipses cycle through three ellipticities that are far
        # apart, so the best one does not depend on rounding. The
        # tolerances below include the values of the original code.
        sample = EllipseSample(self.data, 61.16, x0=253.15, y0=260.15,
                               eps=0.219,
                               position_angle=np.deg2rad(77.5 + 90))
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert isophote.stop_code == 2
        assert isophote.n_iter == 50
        assert_allclose(isophote.n_data, 383, atol=3)
        assert_allclose(isophote.intens, 153.2, atol=0.5)
        assert_allclose(isophote.eps, 0.016, atol=0.005)

    def test_m51_outer(self):
        # Sample taken at the outskirts of the image, so many
        # data points lay outside the image frame. This checks
        # for the presence of gaps in the sample arrays.
        sample = EllipseSample(self.data, 330.0, eps=0.2,
                               position_angle=np.deg2rad(90),
                               integrmode='median')
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert not np.any(isophote.sample.values[2] == 0)

    def test_m51_central(self):
        # This code finds central x and y offset by about 0.1 pixel wrt the
        # spp code. In here we use as input the position computed by this
        # code, thus this test is checking just the extraction algorithm.
        g = EllipseGeometry(257.02, 258.1, 0.0, 0.0, 0.0, astep=0.1,
                            linear_growth=False)
        sample = CentralEllipseSample(self.data, 0.0, geometry=g)
        fitter = CentralEllipseFitter(sample)
        isophote = fitter.fit()

        # The central pixel intensity is about 3% larger than found by
        # the spp code.
        assert isophote.n_data == 1
        assert isophote.intens <= 7560.0
        assert isophote.intens >= 7550.0


def test_fitting_round_start():
    """
    Test a starting ellipse with an ellipticity of exactly zero.

    The position angle correction diverges for a circle, so the starting
    ellipse is made slightly flat before it is sampled.
    """
    data = make_test_image(pa=np.pi / 4, seed=0)
    sample = EllipseSample(data, 40.0, eps=0.0)
    isophote = EllipseFitter(sample).fit()
    assert isophote.stop_code == 0
    assert_allclose(isophote.eps, 0.2, atol=0.01)
    assert_allclose(isophote.pa, np.pi / 4, atol=0.05)


@pytest.mark.parametrize('fix_center', [False, True])
def test_fixed_round_ellipse(fix_center):
    """
    Test that a fixed ellipticity of zero is not changed.

    The position angle of a circle is undefined, so it is not fit.
    """
    geometry = EllipseGeometry(257.0, 255.0, 40.0, 0.0, 0.5, fix_eps=True,
                               fix_center=fix_center)
    sample = EllipseSample(DATA, 40.0, geometry=geometry)
    isophote = EllipseFitter(sample).fit()
    assert isophote.eps == 0.0
    assert isophote.pa == 0.5
    if fix_center:
        assert isophote.stop_code == 4
        assert (isophote.x0, isophote.y0) == (257.0, 255.0)
    else:
        assert isophote.n_iter > 1
        assert_allclose((isophote.x0, isophote.y0), DEFAULT_POS, atol=0.5)


def test_everything_fixed_no_data():
    """
    Test that an isophote extracted without fitting is invalid if it has
    no valid data points.
    """
    geometry = EllipseGeometry(250.0, 260.0, 2000.0, 0.3, 1.0,
                               fix_center=True, fix_pa=True, fix_eps=True)
    sample = EllipseSample(DATA, 2000.0, geometry=geometry)
    isophote = EllipseFitter(sample).fit()
    assert not isophote.valid
    assert isophote.stop_code == 3
    assert isophote.n_data == 0
