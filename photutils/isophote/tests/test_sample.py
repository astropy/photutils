# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the sample module.
"""

import copy
import pickle  # nosec B403
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle
from astropy.stats import SigmaClip
from numpy.testing import assert_allclose, assert_array_equal

from photutils.isophote.ellipse import Ellipse
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.isophote import Isophote
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning

DEFAULT_FIX = np.array([False, False, False, False])

DATA = make_test_image(background=100.0, i0=0.0, noise=10.0, seed=0)


def _make_outlier_image():
    """
    Make a galaxy image with a bright spot on the default ellipse with
    a semimajor axis length of 40 pixels.
    """
    data = make_test_image(seed=0)
    data[254:259, 294:299] += 1.0e4
    return data


# The median is not so good at estimating rms.
@pytest.mark.parametrize(('integrmode', 'amin', 'amax'),
                         [('nearest_neighbor', 7.0, 15.0),
                          ('bilinear', 7.0, 15.0),
                          ('mean', 7.0, 15.0),
                          ('median', 6.0, 15.0)])
def test_scatter(integrmode, amin, amax):
    """
    Check that the pixel standard deviation can be reliably estimated
    from the rms scatter and the sector area.

    The test data is just a flat image with noise, no galaxy. We define
    the noise rms and then compare how close the pixel std dev estimated
    at extraction matches this input noise.
    """
    sample = EllipseSample(DATA, 50.0, step=0.2, integration_mode=integrmode)
    sample._update(fixed_parameters=DEFAULT_FIX)
    iso = Isophote(sample, 0, valid=True, stop_code=0)

    assert iso.pixel_stddev < amax
    assert iso.pixel_stddev > amin


def test_coordinates():
    sample = EllipseSample(DATA, 50.0)
    sample._update(fixed_parameters=DEFAULT_FIX)
    x, y = sample.coordinates()

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_sigma_clip_coordinates():
    sample = EllipseSample(DATA, 50.0,
                           sigma_clip=SigmaClip(sigma=3.0, maxiters=3))
    sample._update(fixed_parameters=DEFAULT_FIX)
    x, y = sample.coordinates()

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_geometry_sma_override_resets_cached_attributes():
    geometry = EllipseGeometry(25.0, 25.0, 10.0, 0.3, (np.pi / 4.0) * u.rad,
                               step=0.2)
    sample = EllipseSample(DATA, 20.0, geometry=geometry)
    expected_geometry = EllipseGeometry(25.0, 25.0, 20.0, 0.3,
                                        (np.pi / 4.0) * u.rad, step=0.2)

    assert sample.geometry.sma == 20.0
    assert geometry.sma == 10.0
    np.testing.assert_allclose(sample.geometry._area_factor,
                               expected_geometry._area_factor)
    np.testing.assert_allclose(sample.geometry._sector_angular_width,
                               expected_geometry._sector_angular_width)
    np.testing.assert_allclose(sample.geometry._initial_polar_angle,
                               expected_geometry._initial_polar_angle)
    np.testing.assert_allclose(sample.geometry._initial_polar_radius,
                               expected_geometry._initial_polar_radius)


@pytest.mark.parametrize(('x0', 'y0'), [(0.0, 40.0), (40.0, 0.0),
                                        (0.0, 0.0)])
def test_zero_center(x0, y0):
    """
    Test that a center coordinate of zero is not replaced by the image
    center.
    """
    sample = EllipseSample(DATA, 10.0, x0=x0, y0=y0)
    assert sample.geometry.x0 == x0
    assert sample.geometry.y0 == y0


def test_default_center():
    sample = EllipseSample(DATA, 10.0)
    assert sample.geometry.x0 == DATA.shape[1] / 2
    assert sample.geometry.y0 == DATA.shape[0] / 2


@pytest.mark.parametrize('integrmode', ['nearest_neighbor', 'bilinear', 'mean',
                                        'median'])
@pytest.mark.parametrize('eps', [1.0, 1.5])
def test_invalid_ellipticity(integrmode, eps):
    """
    Test that an ellipse with an ellipticity of 1 or larger, which has
    no elliptical path, cannot be sampled.
    """
    sample = EllipseSample(DATA, 10.0, ellipticity=eps,
                           integration_mode=integrmode)
    match = 'ellipticity must be less than 1'
    with pytest.raises(ValueError, match=match):
        sample.extract()
    with pytest.raises(ValueError, match=match):
        sample.update()

    geometry = EllipseGeometry(256.0, 256.0, 10.0, eps, 0.0 * u.rad)
    ellipse = Ellipse(DATA, geometry=geometry)
    with pytest.raises(ValueError, match=match):
        ellipse.fit_image(integration_mode=integrmode)


@pytest.mark.parametrize('integrmode', ['nearest_neighbor', 'bilinear', 'mean',
                                        'median'])
@pytest.mark.parametrize(('x0', 'y0'), [(np.nan, 256.0), (256.0, np.nan),
                                        (np.inf, 256.0), (256.0, -np.inf)])
def test_nonfinite_center(integrmode, x0, y0):
    """
    Test that an ellipse with a nonfinite center cannot be sampled.
    """
    sample = EllipseSample(DATA, 10.0, x0=x0, y0=y0,
                           integration_mode=integrmode)
    match = 'center of the ellipse must be finite'
    with pytest.raises(ValueError, match=match):
        sample.extract()
    with pytest.raises(ValueError, match=match):
        sample.update()

    geometry = EllipseGeometry(x0, y0, 10.0, 0.2, 0.0 * u.rad)
    ellipse = Ellipse(DATA, geometry=geometry)
    with pytest.raises(ValueError, match=match):
        ellipse.fit_image(integration_mode=integrmode)


@pytest.mark.parametrize('integrmode', ['mean', 'median'])
def test_zero_inner_sma(integrmode):
    """
    Test the area integration modes when the inner bounding ellipse of
    the sectors has zero size.
    """
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 1.0, step=2.0, linear_growth=True,
                           integration_mode=integrmode)
    sample._update(fixed_parameters=DEFAULT_FIX)
    assert sample.n_valid > 0
    assert np.isfinite(sample.intensity)


def test_invalid_image():
    match = 'image must not be a Quantity array'
    with pytest.raises(TypeError, match=match):
        EllipseSample(DATA * u.Jy, 10.0)

    match = 'image must be a 2D array'
    with pytest.raises(ValueError, match=match):
        EllipseSample(DATA[0], 10.0)


def test_invalid_integrmode():
    match = 'integration_mode must be one of'
    with pytest.raises(ValueError, match=match):
        EllipseSample(DATA, 10.0, integration_mode='foo')
    with pytest.raises(ValueError, match=match):
        Ellipse(DATA).fit_image(integration_mode='foo')
    with pytest.raises(ValueError, match=match):
        Ellipse(DATA).fit_isophote(10.0, integration_mode='foo')


def test_array_like_image():
    sample = EllipseSample(DATA.tolist(), 10.0)
    assert isinstance(sample.image, np.ndarray)
    expected = EllipseSample(DATA, 10.0)
    assert_array_equal(sample.extract(), expected.extract())


@pytest.mark.parametrize('integrmode', ['nearest_neighbor', 'bilinear', 'mean',
                                        'median'])
def test_nonfinite_values(integrmode):
    """
    Test that nonfinite values are excluded from the sample, like
    masked pixels.
    """
    # The nonfinite pixels are on the major and minor axes of the
    # elliptical path.
    data = DATA.copy()
    data[254:259, 294:299] = np.nan
    data[286:291, 254:259] = np.inf
    values = EllipseSample(data, 40.0, integration_mode=integrmode).extract()
    assert np.all(np.isfinite(values[2]))

    masked = np.ma.masked_invalid(data)
    expected = EllipseSample(masked, 40.0,
                             integration_mode=integrmode).extract()
    assert_array_equal(values, expected)

    unmasked = EllipseSample(DATA, 40.0,
                             integration_mode=integrmode).extract()
    assert not np.array_equal(values[2], unmasked[2])


@pytest.mark.parametrize(('astep', 'linear_growth'),
                         [(0.3, False), (2.0, True), (5.0, True)])
def test_gradient_radius(astep, linear_growth):
    """
    Test that the local gradient is measured at a fixed relative
    distance from the ellipse, as in the STSDAS ellipse task,
    regardless of the semimajor axis step.

    The gradient samples start at slightly different polar angles for
    different steps, so the gradients are not identical.
    """
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 20.0, step=0.1)
    sample._update(fixed_parameters=DEFAULT_FIX)
    sample_step = EllipseSample(data, 20.0, step=astep,
                                linear_growth=linear_growth)
    sample_step._update(fixed_parameters=DEFAULT_FIX)
    assert_allclose(sample_step.gradient, sample.gradient, rtol=1e-3)


def test_update_keeps_fixed_parameters():
    """
    Test that update without fixed_parameters keeps the fixed parameters
    of the geometry.
    """
    data = make_test_image(seed=0)
    geometry = EllipseGeometry(256.0, 256.0, 20.0, 0.2, 0.0 * u.rad,
                               fix_center=True, fix_ellipticity=True)
    sample = EllipseSample(data, 20.0, geometry=geometry)
    sample.update()
    assert list(sample.geometry._fix) == [True, True, False, True]

    sample._update(fixed_parameters=np.array([False, False, True, False]))
    assert list(sample.geometry._fix) == [False, False, True, False]


@pytest.mark.parametrize('integrmode', ['mean', 'median'])
def test_large_step_sectors(integrmode):
    """
    Test the area integration modes when the step is larger than twice
    the semimajor axis length.

    The sampling annulus then extends to the center, as for a step of
    exactly twice the semimajor axis length.
    """
    data = make_test_image(seed=0)
    n_points = []
    for astep in (4.0, 5.0):
        sample = EllipseSample(data, 2.0, step=astep, linear_growth=True,
                               integration_mode=integrmode)
        sample._update(fixed_parameters=DEFAULT_FIX)
        assert sample.n_valid == sample.n_total
        assert np.isfinite(sample.intensity)
        n_points.append(sample.n_total)
    assert n_points[0] == n_points[1]


@pytest.mark.parametrize(('x0', 'y0'), [(30.0, None), (None, 40.0)])
def test_partial_center(x0, y0):
    """
    Test that only the center coordinate that is not input is set to the
    image center.
    """
    sample = EllipseSample(DATA, 10.0, x0=x0, y0=y0)
    assert sample.geometry.x0 == (DATA.shape[1] / 2 if x0 is None else x0)
    assert sample.geometry.y0 == (DATA.shape[0] / 2 if y0 is None else y0)


def test_sigma_clip_constant_sample():
    """
    Test that sigma clipping keeps a sample without any scatter.
    """
    data = np.full((100, 100), 5.0)
    sample = EllipseSample(data, 10.0, sigma_clip=SigmaClip(maxiters=3),
                           integration_mode='nearest_neighbor')
    sample.extract()
    assert sample.n_total > 50
    assert sample.n_valid == sample.n_total


def test_sigma_clip():
    """
    Test that sigma clipping removes the outlying sample points from
    the angles, radii, and intensities.
    """
    data = _make_outlier_image()
    sample = EllipseSample(data, 40.0)
    intensities = sample.extract()[2]
    assert sample.sigma_clip is None
    assert sample.n_valid == sample.n_total
    assert np.max(intensities) > 5000.0

    sigma_clip = SigmaClip(sigma=3.0, maxiters=3)
    sample = EllipseSample(data, 40.0, sigma_clip=sigma_clip)
    angles, radii, intensities = sample.extract()
    # The sample uses a copy, which can be shared by concurrent fits.
    assert sample.sigma_clip is not sigma_clip
    assert sample.sigma_clip.sigma == 3.0
    assert sample.sigma_clip.maxiters == 3
    assert sample.n_valid < sample.n_total
    assert len(angles) == len(radii) == len(intensities) == sample.n_valid
    assert np.max(intensities) < 5000.0


def test_sigma_clip_invalid():
    match = 'sigma_clip must be a SigmaClip instance'
    with pytest.raises(TypeError, match=match):
        EllipseSample(DATA, 40.0, sigma_clip=3.0)


def test_update_repeated():
    """
    Test that a repeated update does not change the gradient.

    The intensity of this image increases outward, so no meaningful
    gradient can be measured and the gradient is estimated from a
    reference value.
    """
    yy, xx = np.mgrid[0:201, 0:201]
    data = 10.0 + 0.5 * np.hypot(xx - 100, yy - 100)
    sample = EllipseSample(data, 20.0, x0=100.0, y0=100.0)
    sample.update()
    gradient = sample.gradient
    assert gradient > 0
    assert sample.gradient_err is None
    sample.update()
    assert sample.gradient == gradient

    # The gradient is computed again if the sample values change.
    sample.geometry.x0 = 120.0
    sample._values = None
    sample.update()
    assert sample.gradient != gradient


@pytest.mark.parametrize(('profile', 'modulation', 'expected'),
                         [('rising', 0.0, None),
                          ('shallow', 0.0, -1.478),
                          ('shallow', 1.0, None),
                          ('similar', 1.0, -10.92)])
def test_meaningful_gradient(profile, modulation, expected):
    """
    Test which measured gradients are meaningful with respect to the
    reference gradient from the last update.

    The reference is the gradient of a steep galaxy. The sample is then
    moved to a second source. Its gradient is not meaningful if it is
    positive, or if it is more than 3 times shallower than the
    reference and its relative error is larger than 20%. The gradient
    is then 0.8 times the reference and has no error. An azimuthal
    modulation of the second source increases the gradient error
    without changing the gradient.
    """
    yy, xx = np.mgrid[0:201, 0:301]
    galaxy = 1000.0 * np.exp(-np.hypot(xx - 60, yy - 100) / 10.0)
    radius = np.hypot(xx - 220, yy - 100)
    theta = np.arctan2(yy - 100, xx - 220)
    profiles = {'rising': 10.0 + 0.5 * radius,
                'shallow': 100.0 * np.exp(-radius / 40.0),
                'similar': 900.0 * np.exp(-radius / 10.0)}
    source = profiles[profile] * (1.0 + modulation * np.cos(6.0 * theta))
    data = np.where(xx < 140, galaxy, source)

    sample = EllipseSample(data, 20.0, x0=60.0, y0=100.0, ellipticity=0.0)
    sample.update()
    reference = sample.gradient
    assert_allclose(reference, -12.27, atol=0.01)

    sample.geometry.x0 = 220.0
    sample._values = None
    sample.update()
    if expected is None:
        assert_allclose(sample.gradient, 0.8 * reference)
        assert sample.gradient_err is None
        assert sample.gradient_rel_err is None
    else:
        assert_allclose(sample.gradient, expected, rtol=0.01)

        # The modulated source has a large gradient error, so its
        # gradient is meaningful only because it is steep enough.
        if modulation:
            assert sample.gradient_rel_err > 0.2
        else:
            assert sample.gradient_rel_err < 0.2


def test_mixed_infinities():
    """
    Test that neighboring pixels with infinities of opposite sign are
    excluded without numpy warnings.
    """
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 40.0)
    sample.update()
    x, y = sample.coordinates()
    i = int(x[10])
    j = int(y[10])
    data[j, i] = np.inf
    data[j + 1, i] = -np.inf
    data[j, i + 1] = -np.inf
    sample_inf = EllipseSample(data, 40.0)
    sample_inf.update()
    assert sample_inf.n_total == sample.n_total
    assert sample_inf.n_valid < sample.n_valid
    assert np.all(np.isfinite(sample_inf.extract()))


@pytest.mark.parametrize('sma', [1e20, 1e300])
def test_sampling_path_huge_sma(sma):
    """
    Test that the sampling path of a huge ellipse, whose angle step is
    too small to change the polar angle, ends at its first point.
    """
    sample = EllipseSample(DATA, sma)
    values = sample.extract()
    assert values.shape == (3, 0)
    assert sample.n_total == 1
    assert sample.n_valid == 0


@pytest.mark.parametrize('sma', [1.0, 40.0])
def test_sampling_path(sma):
    """
    Test that the sample points lie on the ellipse and are spaced by
    about one pixel, but by at most 0.5 radians.
    """
    data = make_test_image(seed=0)
    sample = EllipseSample(data, sma, ellipticity=0.3)
    angles, radii, _ = sample.extract()
    assert sample.n_valid == sample.n_total
    assert angles[0] == sample.geometry._initial_polar_angle
    assert_allclose(radii, sample.geometry.radius(angles))
    steps = np.minimum(1.0 / radii, 0.5)
    assert_allclose(np.diff(angles), steps[:-1])
    phi_max = 2.0 * np.pi + 0.05
    assert angles[-1] <= phi_max < angles[-1] + steps[-1]


def test_sample_sma_zero():
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 0.0, x0=256.0, y0=256.0)
    sample.update()
    assert sample.n_total == 1
    assert sample.n_valid == 1
    assert sample.extract().shape == (3, 1)
    assert sample.gradient is None


def test_sample_named_arrays():
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 40.0, ellipticity=0.2,
                           orientation=0.0 * u.rad,
                           integration_mode='bilinear')
    assert sample.angles is None
    assert sample.radii is None
    assert sample.intensities is None
    sample.update()
    assert sample.angles.shape == sample.intensities.shape
    assert sample.radii.shape == sample.angles.shape
    assert sample.n_valid == len(sample.intensities)
    assert sample.n_valid <= sample.n_total
    assert_allclose(sample.intensity, np.mean(sample.intensities))
    assert_array_equal(sample.extract(),
                       [sample.angles, sample.radii, sample.intensities])


def test_sample_deprecated_values_setter():
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 40.0)
    sample.update()
    first = sample.intensities
    with pytest.warns(PhotutilsDeprecationWarning, match="'values'"):
        assert_array_equal(sample.values[2], first)
    with pytest.warns(PhotutilsDeprecationWarning, match="'values'"):
        sample.values = None
    assert sample.intensities is None
    sample.extract()
    assert_array_equal(sample.intensities, first)


def test_sample_copy_and_pickle():
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 40.0)
    sample.update()
    for clone in (copy.copy(sample), copy.deepcopy(sample),
                  pickle.loads(  # noqa: S301
                      pickle.dumps(sample))):  # nosec B301
        assert_array_equal(clone.intensities, sample.intensities)
        assert clone.n_valid == sample.n_valid


@pytest.mark.parametrize(('old', 'new', 'value'),
                         [('astep', 'step', 0.2),
                          ('eps', 'ellipticity', 0.3),
                          ('position_angle', 'orientation', 0.5 * u.rad),
                          ('integrmode', 'integration_mode', 'mean')])
def test_sample_deprecated_keywords(old, new, value):
    data = make_test_image(seed=0)
    with pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"):
        old_sample = EllipseSample(data, 40.0, **{old: value})
    new_sample = EllipseSample(data, 40.0, **{new: value})
    assert_array_equal(old_sample.extract(), new_sample.extract())


@pytest.mark.parametrize('kwargs', [{'sclip': 2.0, 'n_clip': 3},
                                    {'sclip': 2.0, 'nclip': 3},
                                    {'n_clip': 3}])
def test_sample_deprecated_clip_keywords(kwargs):
    """
    Test that the deprecated sigma-clipping keywords give the same
    sample as the equivalent ``sigma_clip``.
    """
    data = _make_outlier_image()
    sigma_clip = SigmaClip(sigma=kwargs.get('sclip', 3.0), maxiters=3,
                           cenfunc='mean', stdfunc='std')
    new_sample = EllipseSample(data, 40.0, sigma_clip=sigma_clip)
    match = "Use argument 'sigma_clip' instead"
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        old_sample = EllipseSample(data, 40.0, **kwargs)
    assert len(record) == len(kwargs)
    assert_array_equal(old_sample.extract(), new_sample.extract())
    assert old_sample.n_valid < old_sample.n_total


def test_sample_deprecated_sclip_without_iterations():
    """
    Test that the deprecated ``sclip`` keyword alone does not clip,
    because the number of clipping iterations was zero by default.
    """
    with pytest.warns(PhotutilsDeprecationWarning, match="'sclip'"):
        sample = EllipseSample(DATA, 40.0, sclip=2.0)
    assert sample.sigma_clip is None


def test_sample_deprecated_clip_keywords_with_sigma_clip():
    match = 'sigma_clip cannot be used with the deprecated'
    with (pytest.warns(PhotutilsDeprecationWarning, match="'n_clip'"),
          pytest.raises(ValueError, match=match)):
        EllipseSample(DATA, 40.0, sigma_clip=SigmaClip(), n_clip=3)


class TestOrientationUnits:
    """
    Test that the ``orientation`` of a sample requires units and that a
    number without units is accepted only on the legacy paths.
    """

    @staticmethod
    def _records(func):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            result = func()
        records = [record for record in records
                   if issubclass(record.category,
                                 PhotutilsDeprecationWarning)]
        return records, result

    def test_keyword_quantity(self):
        records, sample = self._records(
            lambda: EllipseSample(DATA, 40.0, orientation=30.0 * u.deg))
        assert records == []
        assert_allclose(sample.geometry._orientation_rad, np.pi / 6)

    def test_keyword_angle(self):
        records, sample = self._records(
            lambda: EllipseSample(DATA, 40.0, orientation=Angle(30.0, 'deg')))
        assert records == []
        assert_allclose(sample.geometry._orientation_rad, np.pi / 6)

    @pytest.mark.parametrize('value', [0.5, 0, np.float64(0.5)],
                             ids=['float', 'int', 'float64'])
    def test_keyword_number(self, value):
        match = 'orientation must be an angular Quantity, e.g., 30 \\* u.deg'
        with pytest.raises(TypeError, match=match):
            EllipseSample(DATA, 40.0, orientation=value)

    def test_omitted(self):
        records, sample = self._records(lambda: EllipseSample(DATA, 40.0))
        assert records == []
        assert sample.geometry._orientation_rad == 0.0
        assert sample.geometry.orientation == 0.0 * u.deg

    def test_position_angle_number(self):
        records, sample = self._records(
            lambda: EllipseSample(DATA, 40.0, position_angle=0.5))
        assert len(records) == 1
        assert "'position_angle' was deprecated" in str(records[0].message)
        assert records[0].filename == __file__
        assert sample.geometry._orientation_rad == 0.5

    def test_position_angle_quantity(self):
        records, sample = self._records(
            lambda: EllipseSample(DATA, 40.0, position_angle=30.0 * u.deg))
        assert len(records) == 1
        assert "'position_angle' was deprecated" in str(records[0].message)
        assert records[0].filename == __file__
        assert_allclose(sample.geometry._orientation_rad, np.pi / 6)

    def test_positional_number(self):
        records, sample = self._records(
            lambda: EllipseSample(DATA, 40.0, 256.0, 256.0, 0.1, 0.2, 0.5))
        assert len(records) == 1
        assert 'positionally' in str(records[0].message)
        assert records[0].filename == __file__
        assert sample.geometry._orientation_rad == 0.5

    def test_copy_and_pickle(self):
        sample = EllipseSample(DATA, 40.0, orientation=0.5 * u.rad)

        def clones():
            return (copy.copy(sample), copy.deepcopy(sample),
                    pickle.loads(  # noqa: S301
                        pickle.dumps(sample)))  # nosec B301

        records, result = self._records(clones)
        assert records == []
        for clone in result:
            assert clone.geometry._orientation_rad == 0.5


@pytest.mark.parametrize(('old', 'new'),
                         [('mean', 'intensity'),
                          ('actual_points', 'n_valid'),
                          ('total_points', 'n_total'),
                          ('integrmode', 'integration_mode')])
def test_sample_deprecated_attributes(old, new):
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 40.0)
    sample.update()
    with pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"):
        assert getattr(sample, old) == getattr(sample, new)
    with pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"):
        setattr(sample, old, getattr(sample, new))


@pytest.mark.parametrize(('name', 'value', 'default'),
                         [('sclip', 2.0, 3.0), ('n_clip', 3, 0),
                          ('nclip', 3, 0)])
def test_sample_deprecated_clip_attributes(name, value, default):
    """
    Test that the deprecated sigma-clipping attributes give the values
    of the ``sigma_clip`` attribute.
    """
    match = f"The '{name}' attribute was deprecated.*Use 'sigma_clip'"
    sample = EllipseSample(DATA, 40.0,
                           sigma_clip=SigmaClip(sigma=2.0, maxiters=3))
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        assert getattr(sample, name) == value

    sample = EllipseSample(DATA, 40.0)
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        assert getattr(sample, name) == default


def test_sample_deprecated_clip_attributes_set():
    """
    Test that setting the deprecated sigma-clipping attributes sets the
    ``sigma_clip`` attribute.
    """
    sample = EllipseSample(DATA, 40.0)
    with pytest.warns(PhotutilsDeprecationWarning, match="'n_clip'"):
        sample.n_clip = 3
    assert sample.sigma_clip.sigma == 3.0
    assert sample.sigma_clip.maxiters == 3

    with pytest.warns(PhotutilsDeprecationWarning, match="'sclip'"):
        sample.sclip = 2.0
    assert sample.sigma_clip.sigma == 2.0
    assert sample.sigma_clip.maxiters == 3

    with pytest.warns(PhotutilsDeprecationWarning, match="'nclip'"):
        sample.nclip = 0
    assert sample.sigma_clip is None


def test_sample_deprecated_clip_attributes_set_sclip_first():
    """
    Test that a ``sclip`` value set without sigma clipping is used when
    ``n_clip`` is set later.
    """
    sample = EllipseSample(DATA, 40.0)
    with pytest.warns(PhotutilsDeprecationWarning, match="'sclip'"):
        sample.sclip = 2.5
    assert sample.sigma_clip is None
    with pytest.warns(PhotutilsDeprecationWarning, match="'sclip'"):
        assert sample.sclip == 2.5

    with pytest.warns(PhotutilsDeprecationWarning, match="'n_clip'"):
        sample.n_clip = 3
    assert sample.sigma_clip.sigma == 2.5
    assert sample.sigma_clip.maxiters == 3


def test_sample_deprecated_clip_attributes_keep_settings():
    """
    Test that setting the deprecated sigma-clipping attributes keeps the
    other settings of the ``sigma_clip`` attribute.
    """
    sigma_clip = SigmaClip(sigma_lower=2.0, sigma_upper=4.0, maxiters=3,
                           cenfunc='median', stdfunc='mad_std')
    sample = EllipseSample(DATA, 40.0, sigma_clip=sigma_clip)
    with pytest.warns(PhotutilsDeprecationWarning, match="'n_clip'"):
        sample.n_clip = 5
    assert sample.sigma_clip.maxiters == 5
    assert sample.sigma_clip.sigma_lower == 2.0
    assert sample.sigma_clip.sigma_upper == 4.0
    assert sample.sigma_clip.cenfunc == 'median'
    assert sample.sigma_clip.stdfunc == 'mad_std'

    # A new sigma value replaces the lower and upper values.
    with pytest.warns(PhotutilsDeprecationWarning, match="'sclip'"):
        sample.sclip = 2.5
    assert sample.sigma_clip.sigma_lower == 2.5
    assert sample.sigma_clip.sigma_upper == 2.5
    assert sample.sigma_clip.maxiters == 5
    assert sample.sigma_clip.cenfunc == 'median'


def test_sample_deprecated_update_fixed_parameters():
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 40.0)
    with pytest.warns(PhotutilsDeprecationWarning,
                      match="'fixed_parameters'"):
        sample.update(fixed_parameters=DEFAULT_FIX)
    assert sample.n_valid > 0
