# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the fitter module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.datasets.load import _get_path
from photutils.isophote.ellipse import _central_isophote
from photutils.isophote.fitter import EllipseFitter, _EllipseFitter
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.harmonics import _fit_first_and_second_harmonics
from photutils.isophote.isophote import Isophote
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning

DATA = make_test_image(seed=0)
DEFAULT_POS = 256

DEFAULT_FIX = np.array([False, False, False, False])


def test_gradient():
    sample = EllipseSample(DATA, 40.0)
    sample._update(fixed_parameters=DEFAULT_FIX)

    assert_allclose(sample.intensity, 200.02, atol=0.01)
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
    sample = EllipseSample(data, 10.0, integration_mode='nearest_neighbor')
    fitter = _EllipseFitter(sample)
    isophote = fitter.fit()

    assert isophote.valid
    assert isophote.stop_code == 5
    assert isophote.sample.gradient == 0.0
    assert isophote.intensity == 1.0


def test_positive_gradient():
    """
    Test that the fitter does not apply corrections computed from a
    positive gradient.

    The intensity of this image increases outward, so the initial
    ellipse is returned unchanged.
    """
    yy, xx = np.mgrid[0:201, 0:201]
    data = 10.0 + 0.5 * np.hypot(xx - 100, yy - 100)
    sample = EllipseSample(data, 20.0, x0=100.0, y0=100.0, ellipticity=0.2)
    isophote = _EllipseFitter(sample).fit()
    assert isophote.stop_code == 5
    assert isophote.gradient > 0
    assert isophote.ellipticity == 0.2
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
    sample = EllipseSample(data, 20.0, x0=101.0, y0=100.0, ellipticity=0.1)
    isophote = _EllipseFitter(sample).fit()
    assert isophote.stop_code != 5
    assert isophote.n_iter > 4
    assert isophote.gradient < 0


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
    sample = EllipseSample(data, 20.0, x0=101.0, y0=100.0, ellipticity=0.1)
    isophote = _EllipseFitter(sample).fit(max_center_shift=0.5)
    assert isophote.stop_code != 5
    assert isophote.n_iter > 5
    assert isophote.gradient < 0
    assert_allclose(isophote.x0, 100.5)


def test_everything_fixed():
    """
    Test that the isophote is extracted with the input geometry,
    without fitting, if all the ellipse parameters are fixed.
    """
    geometry = EllipseGeometry(250.0, 260.0, 40.0, 0.3, 1.0 * u.rad,
                               fix_center=True, fix_orientation=True,
                               fix_ellipticity=True)
    sample = EllipseSample(DATA, 40.0, geometry=geometry)
    isophote = _EllipseFitter(sample).fit()
    assert isophote.stop_code == 4
    assert isophote.n_iter == 0
    assert (isophote.x0, isophote.y0, isophote.ellipticity,
            isophote.orientation) == (250.0, 260.0, 0.3, 1.0 * u.rad)


def test_fit_keeps_sample_settings():
    """
    Test that the sample of the fitted isophote, which is made by the
    parameter correctors, keeps the settings of the input sample.
    """
    sample = EllipseSample(DATA, 60.0, step=0.2, ellipticity=0.3,
                           clip_sigma=2.5, n_clip=2,
                           integration_mode='mean')
    isophote = _EllipseFitter(sample).fit()
    fitted_sample = isophote.sample
    assert isophote.stop_code == 0
    assert fitted_sample is not sample
    assert fitted_sample.clip_sigma == 2.5
    assert fitted_sample.n_clip == 2
    assert fitted_sample.integration_mode == 'mean'
    assert fitted_sample.geometry.sma == 60.0
    assert fitted_sample.geometry.step == 0.2
    assert not fitted_sample.geometry.linear_growth


def test_fitting_raw():
    """
    This test performs a raw (no fitter), 1-step correction in
    one single ellipse coefficient.
    """
    # Pick first guess ellipse that is off in just one of the parameters
    # (eps).
    sample = EllipseSample(DATA, 40.0, ellipticity=2 * 0.2)
    sample._update(fixed_parameters=DEFAULT_FIX)
    s = sample.extract()

    harmonics = _fit_first_and_second_harmonics(s[0], s[2])
    _, a1, b1, a2, b2 = harmonics[0]

    # When eps is off, b2 is the largest (in absolute value).
    assert abs(b2) > abs(a1)
    assert abs(b2) > abs(b1)
    assert abs(b2) > abs(a2)

    correction = (b2 * 2.0 * (1.0 - sample.geometry.ellipticity)
                  / sample.geometry.sma / sample.gradient)
    new_eps = sample.geometry.ellipticity - correction

    # Got closer to test data (eps=0.2)
    assert_allclose(new_eps, 0.21, atol=0.01)


def test_fitting_small_radii():
    sample = EllipseSample(DATA, 2.0)
    fitter = _EllipseFitter(sample)
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
        isophote = _EllipseFitter(sample).fit()
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
                        '_fit_first_and_second_harmonics', fit_harmonics)
    sample = EllipseSample(DATA, 40.0)
    match = 'The harmonic fit failed at SMA 40.0: Error in least-squares'
    with pytest.warns(AstropyUserWarning, match=match):
        isophote = _EllipseFitter(sample).fit()
    assert not isophote.valid
    assert isophote.stop_code == 3


def test_fitting_eps():
    # Initial guess is off in the eps parameter.
    sample = EllipseSample(DATA, 40.0, ellipticity=2 * 0.2)
    fitter = _EllipseFitter(sample)
    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    g = isophote.sample.geometry
    assert_allclose(g.ellipticity, 0.2, atol=0.01)


@pytest.mark.parametrize('galaxy_eps', [0.97, 0.98])
def test_fitting_eps_diverged(galaxy_eps):
    """
    Test that the fit stops when a correction makes the ellipticity
    exceed its maximum allowed value of 0.95.

    The best-fit sample is returned instead of the diverged one.
    """
    data = make_test_image(eps=galaxy_eps, seed=0)
    sample = EllipseSample(data, 40.0, ellipticity=0.9)
    isophote = _EllipseFitter(sample).fit()
    assert isophote.stop_code == 5
    assert isophote.ellipticity == 0.9


@pytest.mark.parametrize('kwargs', [{'maxgerr': 0.5},
                                    {'going_inwards': True}])
def test_fit_deprecated_keywords(kwargs):
    sample = EllipseSample(DATA, 40.0)
    match = f"'{next(iter(kwargs))}' was deprecated"
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        _EllipseFitter(sample).fit(**kwargs)


def test_fitting_pa():
    data = make_test_image(pa=np.pi / 4, noise=0.01, seed=0)

    # Initial guess is off in the pa parameter.
    sample = EllipseSample(data, 40)
    fitter = _EllipseFitter(sample)
    isophote = fitter.fit()
    g = isophote.sample.geometry

    assert_allclose(g.orientation.to_value(u.rad), np.pi / 4, atol=0.05)


def test_fitting_xy():
    pos = DEFAULT_POS - 5
    data = make_test_image(x0=pos, y0=pos, seed=0)

    # Initial guess is off in the x0 and y0 parameters.
    sample = EllipseSample(data, 40)
    fitter = _EllipseFitter(sample)
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
    sample = EllipseSample(data, sma, orientation=((1.2 * angle) * u.rad))
    fitter = _EllipseFitter(sample)
    isophote = fitter.fit()

    assert isophote.stop_code == 0

    g = isophote.sample.geometry
    assert_allclose((g.x0, g.y0), pos, atol=1.5)  # within 1.5 pixel
    assert_allclose(g.ellipticity, eps, atol=0.01)
    # Within about 3 degrees
    assert_allclose(g.orientation.to_value(u.rad), angle, atol=0.05)

    sample_m = EllipseSample(data, sma, orientation=((1.2 * angle) * u.rad),
                             integration_mode='mean')
    fitter_m = _EllipseFitter(sample_m)
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
        isophote = _EllipseFitter(sample).fit(max_center_shift=2.0)

        g = isophote.sample.geometry
        shift = np.hypot(g.x0 - DEFAULT_POS, g.y0 - DEFAULT_POS)
        assert_allclose(shift, 2.0)

        # The center is held fixed only within this fit
        assert np.array_equal(g._fix, DEFAULT_FIX)

    def test_shift_within_bound(self):
        # A bound larger than the true offset does not change the fit
        sample1 = EllipseSample(self.data, 40)
        isophote1 = _EllipseFitter(sample1).fit()
        sample2 = EllipseSample(self.data, 40)
        isophote2 = _EllipseFitter(sample2).fit(max_center_shift=20.0)

        g1 = isophote1.sample.geometry
        g2 = isophote2.sample.geometry
        assert g1.x0 == g2.x0
        assert g1.y0 == g2.y0
        assert g1.ellipticity == g2.ellipticity
        assert g1.orientation == g2.orientation

    def test_reference_center(self):
        reference = (DEFAULT_POS + 4, DEFAULT_POS + 4)
        sample = EllipseSample(self.data, 40)
        isophote = _EllipseFitter(sample).fit(max_center_shift=6.0,
                                              reference_center=reference)

        g = isophote.sample.geometry
        shift = np.hypot(g.x0 - reference[0], g.y0 - reference[1])
        assert_allclose(shift, 6.0)

    def test_all_parameters_fixed_at_bound(self):
        # With PA and ellipticity fixed, holding the center at the bound
        # leaves nothing to fit, so the fit stops there.
        sample = EllipseSample(self.data, 40)
        sample.geometry._fix = np.array([False, False, True, True])
        isophote = _EllipseFitter(sample).fit(max_center_shift=1.0)

        g = isophote.sample.geometry
        shift = np.hypot(g.x0 - DEFAULT_POS, g.y0 - DEFAULT_POS)
        assert_allclose(shift, 1.0)
        assert isophote.stop_code == 0
        assert np.array_equal(g._fix, [False, False, True, True])

    @pytest.mark.parametrize('value', [-1.0, np.nan])
    def test_invalid(self, value):
        sample = EllipseSample(self.data, 40)
        match = 'max_center_shift must be a non-negative number'
        with pytest.raises(ValueError, match=match):
            _EllipseFitter(sample).fit(max_center_shift=value)


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
        sample = EllipseSample(self.data, 21.44, ellipticity=0.18,
                               orientation=np.deg2rad(36.0) * u.rad)
        fitter = _EllipseFitter(sample)
        isophote = fitter.fit()

        assert isophote.n_data == 119
        assert_allclose(isophote.intensity, 685.4, atol=0.1)

        # Last sample taken by the original code, before turning
        # inwards. The original code ran this fit for the maximum
        # number of iterations and measured an intensity of 153, an
        # ellipticity of 0.015, and 380 data points. The corrected
        # ellipses cycle through three ellipticities that are far
        # apart, so the best one does not depend on rounding. The
        # tolerances below include the values of the original code.
        sample = EllipseSample(self.data, 61.16, x0=253.15, y0=260.15,
                               ellipticity=0.219,
                               orientation=np.deg2rad(77.5 + 90) * u.rad)
        fitter = _EllipseFitter(sample)
        isophote = fitter.fit()

        assert isophote.stop_code == 2
        assert isophote.n_iter == 50
        assert_allclose(isophote.n_data, 383, atol=3)
        assert_allclose(isophote.intensity, 153.2, atol=0.5)
        assert_allclose(isophote.ellipticity, 0.016, atol=0.005)

    def test_m51_outer(self):
        # Sample taken at the outskirts of the image, so many
        # data points lay outside the image frame. This checks
        # for the presence of gaps in the sample arrays.
        sample = EllipseSample(self.data, 330.0, ellipticity=0.2,
                               orientation=np.deg2rad(90) * u.rad,
                               integration_mode='median')
        fitter = _EllipseFitter(sample)
        isophote = fitter.fit()

        assert not np.any(isophote.sample.intensities == 0)

    def test_m51_central(self):
        # This code finds central x and y offset by about 0.1 pixel wrt the
        # spp code. In here we use as input the position computed by this
        # code, thus this test is checking just the extraction algorithm.
        g = EllipseGeometry(257.02, 258.1, 0.0, 0.0, 0.0 * u.rad, step=0.1,
                            linear_growth=False)
        isophote = _central_isophote(self.data, g)

        # The central pixel intensity is about 3% larger than found by
        # the spp code.
        assert isophote.n_data == 1
        assert isophote.intensity <= 7560.0
        assert isophote.intensity >= 7550.0


def test_fitting_round_start():
    """
    Test a starting ellipse with an ellipticity of exactly zero.

    The position angle correction diverges for a circle, so the starting
    ellipse is made slightly flat before it is sampled.
    """
    data = make_test_image(pa=np.pi / 4, seed=0)
    sample = EllipseSample(data, 40.0, ellipticity=0.0)
    isophote = _EllipseFitter(sample).fit()
    assert isophote.stop_code == 0
    assert_allclose(isophote.ellipticity, 0.2, atol=0.01)
    assert_allclose(isophote.orientation.to_value(u.rad), np.pi / 4,
                    atol=0.05)


@pytest.mark.parametrize('offset', [15.0, 20.0, 30.0, 45.0, 60.0])
def test_fitting_orientation_overshoot(offset):
    """
    Test a starting ellipse whose orientation correction overshoots.

    The orientation correction is inversely proportional to the
    ellipticity of the trial ellipse. If that ellipticity is about
    half of the galaxy ellipticity, the correction is about twice as
    large as needed. Without damping, the orientation oscillates
    between two values on either side of the solution, where the
    ellipticity harmonic is small, so the ellipticity is never
    corrected.
    """
    orientation = 40.0
    data = make_test_image(eps=0.4, pa=np.deg2rad(orientation), seed=0)
    sample = EllipseSample(data, 40.0, ellipticity=0.2,
                           orientation=(orientation - offset) * u.deg)
    isophote = _EllipseFitter(sample).fit()
    assert isophote.stop_code == 0
    assert_allclose(isophote.ellipticity, 0.4, atol=0.01)
    assert_allclose(isophote.orientation.to_value(u.deg), orientation,
                    atol=0.5)


@pytest.mark.parametrize('index', [0, 1, 2, 3])
def test_fitting_reversed_correction_damped(monkeypatch, index):
    """
    Test that a correction that reverses the previous correction of
    the same ellipse parameter is halved.

    The harmonic fit is replaced by one where a single harmonic
    alternates in sign with amplitudes of 1, -0.9, and 0.8. The
    amplitudes decrease, so the fit returns the last sampled ellipse.
    The second correction is -0.9 times the first one, or -0.45 times
    if it is halved.
    """
    def make_fit_harmonics():
        amplitudes = iter([1.0, -0.9, 0.8])

        def fit_harmonics(angles, intensities):  # noqa: ARG001
            coeffs = np.zeros(5)
            coeffs[0] = np.mean(intensities)
            coeffs[index + 1] = next(amplitudes)
            return (coeffs,)

        return fit_harmonics

    def fit_parameters(maxiters):
        monkeypatch.setattr('photutils.isophote.fitter.'
                            '_fit_first_and_second_harmonics',
                            make_fit_harmonics())
        sample = EllipseSample(DATA, 40.0)
        isophote = _EllipseFitter(sample).fit(convergence=0.0,
                                              maxiters=maxiters)
        geometry = isophote.sample.geometry
        return np.array([geometry.x0, geometry.y0,
                         geometry.orientation.to_value(u.rad),
                         geometry.ellipticity])

    start = fit_parameters(1)
    one_correction = fit_parameters(2) - start
    two_corrections = fit_parameters(3) - start
    assert np.any(one_correction != 0.0)
    ratio = ((one_correction @ two_corrections)
             / (one_correction @ one_correction))
    assert_allclose(ratio, 0.55, atol=0.02)


@pytest.mark.parametrize('fix_center', [False, True])
def test_fixed_round_ellipse(fix_center):
    """
    Test that a fixed ellipticity of zero is not changed.

    The position angle of a circle is undefined, so it is not fit.
    """
    geometry = EllipseGeometry(257.0, 255.0, 40.0, 0.0, 0.5 * u.rad,
                               fix_ellipticity=True, fix_center=fix_center)
    sample = EllipseSample(DATA, 40.0, geometry=geometry)
    isophote = _EllipseFitter(sample).fit()
    assert isophote.ellipticity == 0.0
    assert isophote.orientation == 0.5 * u.rad
    if fix_center:
        assert isophote.stop_code == 4
        assert (isophote.x0, isophote.y0) == (257.0, 255.0)
    else:
        assert isophote.n_iter > 1
        assert_allclose((isophote.x0, isophote.y0), DEFAULT_POS, atol=0.5)


@pytest.mark.parametrize(('fix_eps', 'fix_center'),
                         [(False, False), (True, False), (True, True)])
def test_fit_does_not_modify_input(fix_eps, fix_center):
    """
    Test that fitting a round starting ellipse does not modify the
    geometry of the input sample.
    """
    geometry = EllipseGeometry(257.0, 255.0, 40.0, 0.0, 0.5 * u.rad,
                               fix_ellipticity=fix_eps,
                               fix_center=fix_center)
    sample = EllipseSample(DATA, 40.0, geometry=geometry)
    fix = sample.geometry._fix.copy()
    _EllipseFitter(sample).fit()
    assert sample.geometry.ellipticity == 0.0
    assert sample.geometry.orientation == 0.5 * u.rad
    assert (sample.geometry.x0, sample.geometry.y0) == (257.0, 255.0)
    assert np.array_equal(sample.geometry._fix, fix)


def test_everything_fixed_no_data():
    """
    Test that an isophote extracted without fitting is invalid if it has
    no valid data points.
    """
    geometry = EllipseGeometry(250.0, 260.0, 2000.0, 0.3, 1.0 * u.rad,
                               fix_center=True, fix_orientation=True,
                               fix_ellipticity=True)
    sample = EllipseSample(DATA, 2000.0, geometry=geometry)
    isophote = _EllipseFitter(sample).fit()
    assert not isophote.valid
    assert isophote.stop_code == 3
    assert isophote.n_data == 0


@pytest.mark.parametrize(('old', 'new', 'value'),
                         [('conver', 'convergence', 0.1),
                          ('minit', 'miniters', 5),
                          ('maxit', 'maxiters', 30),
                          ('fflag', 'min_valid_fraction', 0.5)])
def test_fit_renamed_keywords(old, new, value):
    sample = EllipseSample(DATA, 40.0)
    expected = _EllipseFitter(sample).fit(**{new: value})
    with pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"):
        result = _EllipseFitter(sample).fit(**{old: value})
    assert result.intensity == expected.intensity
    with (pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"),
          pytest.raises(TypeError)):
        _EllipseFitter(sample).fit(**{old: value, new: value})


def test_ellipse_fitter_deprecated():
    sample = EllipseSample(DATA, 40.0)
    match = 'EllipseFitter.*Ellipse.fit_isophote'
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        fitter = EllipseFitter(sample)
    assert fitter.fit().valid
