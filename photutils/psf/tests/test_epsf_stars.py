# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf_stars module.
"""

import warnings
from multiprocessing.reduction import ForkingPickler

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from numpy.testing import assert_allclose, assert_array_equal

from photutils.psf.epsf_stars import (EPSFStar, EPSFStars, LinkedEPSFStar,
                                      _compute_mean_sky_coordinate,
                                      _create_weights_cutout,
                                      _prepare_uncertainty_info, extract_stars)
from photutils.psf.functional_models import CircularGaussianPRF
from photutils.psf.image_models import ImagePSF


@pytest.fixture
def simple_wcs():
    """
    Create a simple WCS for testing.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [25, 25]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.cdelt = [1, 1]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


@pytest.fixture
def simple_data():
    """
    Create simple 50x50 array of ones for testing.
    """
    return np.ones((50, 50))


@pytest.fixture
def simple_nddata(simple_data):
    """
    Create simple NDData object for testing.
    """
    return NDData(simple_data)


@pytest.fixture
def simple_table():
    """
    Create simple table with single star at center.
    """
    return Table({'x': [25], 'y': [25]})


@pytest.fixture
def stars_table():
    """
    Create table with multiple star positions.
    """
    table = Table()
    table['x'] = [15, 15, 35, 35]
    table['y'] = [15, 35, 40, 10]
    return table


@pytest.fixture
def stars_data(stars_table):
    """
    Create image data with stars using CircularGaussianPRF model.
    """
    yy, xx = np.mgrid[0:51, 0:55]
    data = np.zeros(xx.shape)
    model = CircularGaussianPRF(fwhm=3.5)
    for xi, yi in zip(stars_table['x'], stars_table['y'], strict=True):
        data += model.evaluate(xx, yy, 100, xi, yi, 3.5)
    return data


@pytest.fixture
def stars_nddata(stars_data):
    """
    Create NDData object with star data.
    """
    return NDData(data=stars_data)


def test_compute_mean_sky_coordinate():
    """
    Test spherical coordinate averaging.
    """
    delta = 0.5 / 3600.0  # 0.5 arcsec in degrees
    ra = 10.0
    dec = 30.0
    coords = np.array([
        [ra - delta, dec - delta],
        [ra + delta, dec - delta],
        [ra - delta, dec + delta],
        [ra + delta, dec + delta],
    ])
    mean_lon, mean_lat = _compute_mean_sky_coordinate(coords)
    assert_allclose(mean_lon, ra)
    assert_allclose(mean_lat, dec)


def test_compute_mean_sky_coordinate_edge_cases():
    """
    Test mean sky coordinate calculation edge cases.
    """
    # Test coordinates near poles
    coords = np.array([
        [0.0, 89.0],
        [90.0, 89.0],
        [180.0, 89.0],
        [270.0, 89.0],
    ])
    # Mean latitude should be close to 89 - relax tolerance for edge case
    _, mean_lat = _compute_mean_sky_coordinate(coords)
    assert abs(mean_lat - 89.0) < 1.1

    # Test with single coordinate
    single_coord = np.array([[45.0, 30.0]])
    mean_lon, mean_lat = _compute_mean_sky_coordinate(single_coord)
    assert abs(mean_lon - 45.0) < 1e-10
    assert abs(mean_lat - 30.0) < 1e-10


def test_prepare_uncertainty_info():
    """
    Test uncertainty info preparation.
    """
    # Test with no uncertainty
    data = NDData(np.ones((5, 5)))
    info = _prepare_uncertainty_info(data)
    assert info['type'] == 'none'

    # Test with weight-like uncertainty by creating custom uncertainty
    class WeightsUncertainty(StdDevUncertainty):
        @property
        def uncertainty_type(self):
            return 'weights'

    weights = np.ones((5, 5)) * 2
    data.uncertainty = WeightsUncertainty(weights)

    info = _prepare_uncertainty_info(data)
    assert info['type'] == 'weights'
    assert_array_equal(info['array'], weights)


def test_prepare_uncertainty_info_variants():
    """
    Test uncertainty preparation for different uncertainty types.
    """
    # Test standard deviation uncertainty
    data = NDData(np.ones((5, 5)))
    data.uncertainty = StdDevUncertainty(np.ones((5, 5)) * 0.1)

    info = _prepare_uncertainty_info(data)
    assert info['type'] == 'uncertainty'
    assert 'uncertainty' in info


def test_create_weights_cutout():
    """
    Test weights cutout creation.
    """
    # Test with no uncertainty
    info = {'type': 'none'}
    slices = (slice(1, 4), slice(1, 4))  # 3x3 cutout
    mask = None

    weights, has_nonfinite = _create_weights_cutout(info, mask, slices)
    assert weights.shape == (3, 3)
    assert_array_equal(weights, np.ones((3, 3)))
    assert not has_nonfinite

    # Test with mask
    full_mask = np.zeros((5, 5), dtype=bool)
    full_mask[2, 2] = True  # Mask center of cutout

    weights, has_nonfinite = _create_weights_cutout(info, full_mask, slices)
    assert weights[1, 1] == 0.0  # Should be masked
    assert not has_nonfinite


def test_create_weights_cutout_with_uncertainty():
    """
    Test weights cutout creation with uncertainty.
    """
    # Create uncertainty info
    uncertainty = StdDevUncertainty(np.ones((5, 5)) * 0.1)
    info = {
        'type': 'uncertainty',
        'uncertainty': uncertainty,
    }

    slices = (slice(1, 4), slice(1, 4))
    mask = None

    weights, has_nonfinite = _create_weights_cutout(info, mask, slices)
    assert weights.shape == (3, 3)
    # Should be inverse of uncertainty values (1/0.1 = 10)
    assert_allclose(weights, np.ones((3, 3)) * 10)
    assert not has_nonfinite


def test_create_weights_cutout_non_finite_warning():
    """
    Test detection of non-finite weights.
    """
    # Create weights with non-finite values
    bad_weights = np.ones((5, 5))
    bad_weights[2, 2] = np.inf

    info = {
        'type': 'weights',
        'array': bad_weights,
    }

    slices = (slice(1, 4), slice(1, 4))
    mask = None

    # Function should return has_nonfinite=True (warning is now
    # emitted by caller)
    weights, has_nonfinite = _create_weights_cutout(info, mask, slices)
    assert has_nonfinite
    # Non-finite value should be set to zero
    assert weights[1, 1] == 0.0


class TestEPSFStar:
    """
    Tests for EPSFStar class functionality.
    """

    def test_basic_initialization(self):
        """
        Test basic EPSFStar initialization.
        """
        data = np.ones((11, 11))
        star = EPSFStar(data)

        assert star.data.shape == (11, 11)
        assert star.cutout_center is not None
        assert star.weights.shape == data.shape
        assert star.flux > 0

    def test_explicit_flux(self):
        """
        Test EPSFStar initialization with explicit flux value.
        """
        data = np.ones((5, 5))
        explicit_flux = 100.0
        star = EPSFStar(data, flux=explicit_flux)

        # Flux should be the explicitly provided value
        assert star.flux == explicit_flux

    def test_invalid_data_input(self):
        """
        Test EPSFStar initialization with invalid data.
        """
        # Test 1D data
        match = 'Input data must be 2-dimensional'
        with pytest.raises(ValueError, match=match):
            EPSFStar(np.ones(10))

        # Test 3D data
        with pytest.raises(ValueError, match=match):
            EPSFStar(np.ones((5, 5, 5)))

        # Test empty data
        with pytest.raises(ValueError, match='Input data cannot be empty'):
            EPSFStar(np.array([]).reshape(0, 0))

    def test_weights_validation(self):
        """
        Test weight validation in EPSFStar.
        """
        data = np.ones((5, 5))

        # Test mismatched weights shape
        wrong_weights = np.ones((3, 3))
        match = 'Weights shape .* must match data shape'
        with pytest.raises(ValueError, match=match):
            EPSFStar(data, weights=wrong_weights)

        # Test non-finite weights (should generate warning)
        bad_weights = np.ones((5, 5))
        bad_weights[2, 2] = np.inf
        bad_weights[1, 1] = np.nan

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            star = EPSFStar(data, weights=bad_weights)
            assert len(w) == 1
            assert issubclass(w[0].category, AstropyUserWarning)
            assert 'Non-finite weight values' in str(w[0].message)

        # Check that non-finite weights were set to zero
        assert star.weights[2, 2] == 0.0
        assert star.weights[1, 1] == 0.0

    def test_invalid_data_handling(self):
        """
        Test handling of invalid pixel values.
        """
        data = np.ones((5, 5))
        data[1, 1] = np.nan
        data[2, 2] = np.inf
        data[3, 3] = np.nan
        data[4, 4] = np.inf

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            star = EPSFStar(data)
            # Should mask invalid pixels
            assert star.mask[1, 1]
            assert star.mask[2, 2]
            assert star.mask[3, 3]
            assert star.mask[4, 4]
            assert star.weights[1, 1] == 0.0
            assert star.weights[2, 2] == 0.0
            assert star.weights[3, 3] == 0.0
            assert star.weights[4, 4] == 0.0
            # Check that warning was issued about invalid data
            assert len(w) > 0

    def test_cutout_center_validation(self):
        """
        Test cutout_center validation.
        """
        data = np.ones((5, 5))
        star = EPSFStar(data)

        # Test invalid shape
        match = 'cutout_center must have exactly two elements'
        with pytest.raises(ValueError, match=match):
            star.cutout_center = [1, 2, 3]

        # Test non-finite values
        with pytest.raises(ValueError, match='must be finite'):
            star.cutout_center = [np.nan, 2.0]

        # Test bounds warnings (should warn but not raise)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            star.cutout_center = [-1, 2]  # Outside bounds
            assert len(w) >= 1
            # Check that warning mentions coordinates outside bounds
            warning_messages = [str(warning.message) for warning in w]
            assert any('outside the cutout bounds' in msg
                       for msg in warning_messages)

    def test_origin_validation(self):
        """
        Test origin parameter validation.
        """
        data = np.ones((5, 5))

        # Test invalid origin shape
        match = 'Origin must have exactly 2 elements'
        with pytest.raises(ValueError, match=match):
            EPSFStar(data, origin=[1, 2, 3])

        # Test non-finite origin
        match = 'Origin coordinates must be finite'
        with pytest.raises(ValueError, match=match):
            EPSFStar(data, origin=[np.inf, 2])

    def test_estimate_flux_masked_data(self):
        """
        Test flux estimation with masked data.
        """
        data = np.ones((5, 5)) * 10

        # Create weights that mask some pixels
        weights = np.ones((5, 5))
        weights[1:3, 1:3] = 0  # Mask central 2x2 region

        star = EPSFStar(data, weights=weights)

        # Flux should be estimated via interpolation
        assert star.flux > 0
        # Should be close to total flux despite masking
        assert star.flux == pytest.approx(250, rel=0.1)  # 5*5*10 = 250

    def test_data_shape_validation(self):
        """
        Test EPSFStar validation for various data shapes.
        """
        # Test zero-dimension data - this actually triggers "empty"
        # error
        with pytest.raises(ValueError, match='Input data cannot be empty'):
            EPSFStar(np.zeros((0, 5)))

        with pytest.raises(ValueError, match='Input data cannot be empty'):
            EPSFStar(np.zeros((5, 0)))

    def test_flux_estimation_failure(self):
        """
        Test flux estimation behavior with all masked data.
        """
        # Create data with all masked pixels - this should raise
        # ValueError because estimated flux will be NaN
        data = np.ones((5, 5))
        weights = np.zeros((5, 5))  # All masked data

        # This should raise ValueError because flux estimation returns
        # NaN
        match = 'Estimated flux is non-finite or non-positive'
        with pytest.raises(ValueError, match=match):
            EPSFStar(data, weights=weights)

    def test_array_method(self):
        """
        Test the __array__ method.
        """
        data = np.random.default_rng(42).random((5, 5))
        star = EPSFStar(data)

        # Test that __array__ returns the data
        star_array = star.__array__()
        assert_array_equal(star_array, data)

    def test_properties(self):
        """
        Test star properties.
        """
        data = np.ones((7, 9))
        origin = (10, 20)
        star = EPSFStar(data, origin=origin)

        # Test shape property
        assert star.shape == (7, 9)

        # Test center property (different from cutout_center)
        expected_center = star.cutout_center + np.array(origin)
        assert_array_equal(star.center, expected_center)

        # Test slices property
        # Implementation uses (origin_y to origin_y+shape[0],
        # origin_x to origin_x+shape[1])
        expected_slices = (slice(20, 29), slice(10, 17))
        assert star.slices == expected_slices

        # Test bbox property
        bbox = star.bbox
        assert bbox.ixmin == 10
        assert bbox.ixmax == 17
        assert bbox.iymin == 20
        assert bbox.iymax == 29

    def test_flux_estimation_interpolation_fallback(self):
        """
        Test flux estimation with interpolation fallbacks.
        """
        data = np.ones((5, 5)) * 10
        weights = np.ones((5, 5))
        weights[2, 2] = 0  # Mask center pixel

        star = EPSFStar(data, weights=weights)

        # Should estimate flux using interpolation
        # Flux should be close to total despite masked pixel
        assert star.flux == pytest.approx(250, rel=0.1)

    def test_register_epsf(self):
        """
        Test ePSF registration and scaling.
        """
        data = np.ones((11, 11))
        star = EPSFStar(data)

        # Create a simple ePSF model
        epsf_data = np.zeros((5, 5))
        epsf_data[2, 2] = 1  # Central peak
        epsf = ImagePSF(epsf_data)

        # Register the ePSF
        registered = star.register_epsf(epsf)

        assert registered.shape == data.shape
        assert isinstance(registered, np.ndarray)

    def test_private_properties(self):
        """
        Test private properties.
        """
        data = np.random.default_rng(42).random((5, 5))
        weights = np.ones((5, 5))
        weights[1, 1] = 0  # Mask one pixel
        star = EPSFStar(data, weights=weights)

        # Test _xyidx_centered
        x_centered, y_centered = star._xyidx_centered
        assert len(x_centered) == len(y_centered)
        assert len(x_centered) == np.sum(~star.mask)

        # Verify centering is correct
        yidx, xidx = np.indices(data.shape)
        expected_x = xidx[~star.mask].ravel() - star.cutout_center[0]
        expected_y = yidx[~star.mask].ravel() - star.cutout_center[1]
        assert_array_equal(x_centered, expected_x)
        assert_array_equal(y_centered, expected_y)

        # Test normalized data values
        expected_values = data[~star.mask].ravel()
        normalized = star._data_values_normalized
        expected_normalized = expected_values / star.flux
        assert_allclose(normalized, expected_normalized)

    def test_flux_estimation_exception_handling(self):
        """
        Test flux estimation exception handling when estimate_flux returns
        invalid values.
        """
        # Test with data that results in zero flux (non-positive)
        data = np.zeros((3, 3))

        # This should raise ValueError because flux is non-positive
        match = 'Estimated flux is non-finite or non-positive'
        with pytest.raises(ValueError, match=match):
            EPSFStar(data)

    def test_cutout_center_out_of_bounds_y(self):
        """
        Test cutout_center validation for y-coordinate out of bounds.
        """
        data = np.ones((5, 5))
        star = EPSFStar(data)

        # Test y-coordinate outside bounds
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            star.cutout_center = (2.0, -1.0)  # y < 0
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any('y-coordinate' in msg and 'outside' in msg
                       for msg in warning_messages)

        # Test y-coordinate at upper bound
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            star.cutout_center = (2.0, 6.0)  # y >= shape[0]
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any('y-coordinate' in msg and 'outside' in msg
                       for msg in warning_messages)

    def test_empty_data_validation(self):
        """
        Test empty data validation.
        """
        data = np.array([[]])  # Empty 2D array
        with pytest.raises(ValueError, match='Input data cannot be empty'):
            EPSFStar(data)

    def test_residual_image(self):
        """
        Test to ensure ``compute_residual_image`` gives correct
        residuals.
        """
        size = 100
        yy, xx, = np.mgrid[0:size + 1, 0:size + 1] / 4
        gmodel = CircularGaussianPRF().evaluate(xx, yy, 1, 12.5, 12.5, 2.5)
        epsf = ImagePSF(gmodel, oversampling=4)
        _size = 25
        data = np.zeros((_size, _size))
        _yy, _xx, = np.mgrid[0:_size, 0:_size]
        data += epsf.evaluate(x=_xx, y=_yy, flux=16, x_0=12, y_0=12)
        tbl = Table()
        tbl['x'] = [12]
        tbl['y'] = [12]
        stars = extract_stars(NDData(data), tbl, size=23)
        residual = stars[0].compute_residual_image(epsf)
        assert_allclose(np.sum(residual), 0.0)


class TestEPSFStars:
    """
    Tests for EPSFStars collection class functionality.
    """

    def test_initialization_variants(self):
        """
        Test different initialization methods.
        """
        data1 = np.ones((5, 5))
        data2 = np.ones((7, 7))
        star1 = EPSFStar(data1)
        star2 = EPSFStar(data2)

        # Test single star initialization
        stars_single = EPSFStars(star1)
        assert len(stars_single) == 1

        # Test list initialization
        stars_list = EPSFStars([star1, star2])
        assert len(stars_list) == 2

        # Test invalid initialization
        with pytest.raises(TypeError, match='stars_list must be a list'):
            EPSFStars('invalid')

    def test_indexing_operations(self):
        """
        Test indexing and slicing operations.
        """
        stars = [EPSFStar(np.ones((5, 5))) for _ in range(3)]
        stars_obj = EPSFStars(stars)

        # Test getitem
        first = stars_obj[0]
        assert isinstance(first, EPSFStars)
        assert len(first) == 1

        # Test delitem
        del stars_obj[1]
        assert len(stars_obj) == 2

        # Test iteration
        count = 0
        for star in stars_obj:
            count += 1
            assert isinstance(star, EPSFStar)
        assert count == 2

    def test_pickle_operations(self):
        """
        Test pickle state management.
        """
        stars = [EPSFStar(np.ones((5, 5))) for _ in range(2)]
        stars_obj = EPSFStars(stars)

        # Test getstate/setstate
        state = stars_obj.__getstate__()
        new_obj = EPSFStars([])
        new_obj.__setstate__(state)
        assert len(new_obj) == len(stars_obj)

    def test_attribute_access(self):
        """
        Test dynamic attribute access.
        """
        data1 = np.ones((5, 5))
        data2 = np.ones((7, 7)) * 2
        stars = EPSFStars([EPSFStar(data1), EPSFStar(data2)])

        # Test accessing cutout_center attribute
        centers = stars.cutout_center
        assert len(centers) == 2
        assert centers.shape == (2, 2)

        # Test accessing flux attribute
        fluxes = stars.flux
        assert len(fluxes) == 2

        # Test accessing _excluded_from_fit attribute
        excluded = stars._excluded_from_fit
        assert len(excluded) == 2
        assert not any(excluded)  # Should all be False initially

    def test_flat_attributes(self):
        """
        Test flat attribute access methods.
        """
        stars = [EPSFStar(np.ones((5, 5))) for _ in range(2)]
        stars_obj = EPSFStars(stars)

        # Test cutout_center_flat
        centers_flat = stars_obj.cutout_center_flat
        assert centers_flat.shape == (2, 2)

        # Test center_flat
        centers_flat = stars_obj.center_flat
        assert centers_flat.shape == (2, 2)

    def test_star_counting(self):
        """
        Test star counting properties.
        """
        stars = [EPSFStar(np.ones((5, 5))) for _ in range(3)]
        stars_obj = EPSFStars(stars)

        # Test counting properties
        assert stars_obj.n_stars == 3
        assert stars_obj.n_all_stars == 3
        assert stars_obj.n_good_stars == 3

        # Test all_stars and all_good_stars properties
        all_stars = stars_obj.all_stars
        assert len(all_stars) == 3

        good_stars = stars_obj.all_good_stars
        assert len(good_stars) == 3

        # Mark one star as excluded
        stars[1]._excluded_from_fit = True
        assert stars_obj.n_good_stars == 2

    def test_shape_attribute(self):
        """
        Test accessing shape attribute through EPSFStars.
        """
        stars = [EPSFStar(np.ones((5, 5))), EPSFStar(np.ones((7, 9)))]
        stars_obj = EPSFStars(stars)

        # Access individual star shapes through the container
        shapes = stars_obj.shape
        assert len(shapes) == 2
        assert shapes[0] == (5, 5)
        assert shapes[1] == (7, 9)

    def test_pickleable(self):
        """
        Verify that EPSFStars can be successfully pickled/unpickled for
        multiprocessing.
        """
        # This should not fail
        stars = EPSFStars([1])
        ForkingPickler.loads(ForkingPickler.dumps(stars))

    def test_cutout_center_flat_with_linked_stars(self, simple_wcs):
        """
        Test cutout_center_flat property with LinkedEPSFStar objects.
        """
        # Create regular stars
        star1 = EPSFStar(np.ones((5, 5)))
        star2 = EPSFStar(np.ones((7, 7)))

        # Create linked stars
        linked_star1 = EPSFStar(np.ones((6, 6)), wcs_large=simple_wcs)
        linked_star2 = EPSFStar(np.ones((8, 8)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([linked_star1, linked_star2])

        # Create EPSFStars collection with mix of regular and linked stars
        stars = EPSFStars([star1, linked, star2])

        # Test cutout_center_flat property
        centers_flat = stars.cutout_center_flat
        # Should have 4 centers: star1, linked_star1, linked_star2, star2
        assert len(centers_flat) == 4
        assert centers_flat.shape == (4, 2)

    def test_all_stars_with_linked_stars(self, simple_wcs):
        """
        Test all_stars property with LinkedEPSFStar objects.
        """
        # Create regular stars
        star1 = EPSFStar(np.ones((5, 5)))
        star2 = EPSFStar(np.ones((7, 7)))

        # Create linked stars
        linked_star1 = EPSFStar(np.ones((6, 6)), wcs_large=simple_wcs)
        linked_star2 = EPSFStar(np.ones((8, 8)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([linked_star1, linked_star2])

        # Create EPSFStars collection with mix of regular and linked
        # stars
        stars = EPSFStars([star1, linked, star2])

        # Test all_stars property
        all_stars_list = stars.all_stars
        # Should have 4 stars total: star1, linked_star1, linked_star2,
        # star2
        assert len(all_stars_list) == 4

        # Verify they are all EPSFStar instances
        for star in all_stars_list:
            assert isinstance(star, EPSFStar)


class TestLinkedEPSFStar:
    """
    Tests for LinkedEPSFStar functionality.
    """

    def test_initialization_validation(self):
        """
        Test LinkedEPSFStar initialization validation.
        """
        # Test with non-EPSFStar objects
        with pytest.raises(TypeError, match='must contain only EPSFStar'):
            LinkedEPSFStar(['not_a_star', 'also_not_a_star'])

        # Test with EPSFStar without WCS
        star_no_wcs = EPSFStar(np.ones((5, 5)))
        with pytest.raises(ValueError, match='must have a valid wcs_large'):
            LinkedEPSFStar([star_no_wcs])

    def test_constraint_no_good_stars(self, simple_wcs):
        """
        Test constraining centers with no good stars.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star2 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)

        # Mark both as excluded
        star1._excluded_from_fit = True
        star2._excluded_from_fit = True

        linked = LinkedEPSFStar([star1, star2])

        # Should warn about no good stars
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            linked.constrain_centers()
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any('have all been excluded' in msg
                       for msg in warning_messages)

    def test_constraint_single_star(self, simple_wcs):
        """
        Test constraining centers with single star (no-op).
        """
        star = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([star])

        # Should do nothing for single star
        original_center = star.cutout_center.copy()
        linked.constrain_centers()
        assert_array_equal(star.cutout_center, original_center)

    def test_all_excluded_property(self, simple_wcs):
        """
        Test the all_excluded property.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star2 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([star1, star2])

        # Initially, no stars are excluded
        assert not linked.all_excluded

        # Exclude one star
        star1._excluded_from_fit = True
        assert not linked.all_excluded

        # Exclude both stars
        star2._excluded_from_fit = True
        assert linked.all_excluded

    def test_constrain_centers_with_good_stars(self, simple_wcs):
        """
        Test constrain_centers method with good stars.
        """
        # Create multiple stars with different positions (within bounds)
        star1 = EPSFStar(np.ones((7, 7)), wcs_large=simple_wcs,
                         cutout_center=(3.1, 3.1), origin=(20, 20))
        star2 = EPSFStar(np.ones((7, 7)), wcs_large=simple_wcs,
                         cutout_center=(2.9, 2.9), origin=(20, 20))
        star3 = EPSFStar(np.ones((7, 7)), wcs_large=simple_wcs,
                         cutout_center=(3.0, 3.2), origin=(20, 20))

        # Make sure none are excluded
        star1._excluded_from_fit = False
        star2._excluded_from_fit = False
        star3._excluded_from_fit = False

        linked = LinkedEPSFStar([star1, star2, star3])

        # Test constrain_centers (should execute without error)
        linked.constrain_centers()

    def test_constrain_centers_with_some_excluded_stars(self, simple_wcs):
        """
        Test constrain_centers with some excluded stars.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star2 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star3 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)

        # Exclude some stars but not all
        star1._excluded_from_fit = True  # Excluded
        star2._excluded_from_fit = False  # Good
        star3._excluded_from_fit = False  # Good

        linked = LinkedEPSFStar([star1, star2, star3])

        # This should process only the good stars; should not raise
        # warnings since there are good stars
        linked.constrain_centers()

    def test_constrain_all_excluded(self, simple_wcs):
        """
        Test constrain_centers when all stars excluded.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star2 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)

        # Exclude all stars
        star1._excluded_from_fit = True
        star2._excluded_from_fit = True

        linked = LinkedEPSFStar([star1, star2])

        # Should trigger early return and emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            linked.constrain_centers()
            # Should get warning about no good stars
            warning_messages = [str(warning.message) for warning in w]
            has_warning = any('Cannot constrain centers' in msg
                              for msg in warning_messages)
            assert has_warning

    def test_len_getitem_iter(self, simple_wcs):
        """
        Test __len__, __getitem__, and __iter__ methods.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star2 = EPSFStar(np.ones((7, 7)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([star1, star2])

        # Test __len__
        assert len(linked) == 2

        # Test __getitem__
        assert linked[0] is star1
        assert linked[1] is star2

        # Test __iter__
        stars_list = list(linked)
        assert len(stars_list) == 2
        assert stars_list[0] is star1
        assert stars_list[1] is star2

    def test_getattr_delegation(self, simple_wcs):
        """
        Test __getattr__ delegation for various attributes.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star2 = EPSFStar(np.ones((7, 7)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([star1, star2])

        # Test accessing flux attribute (should be array)
        fluxes = linked.flux
        assert len(fluxes) == 2
        assert fluxes[0] == star1.flux
        assert fluxes[1] == star2.flux

        # Test accessing cutout_center (should be array)
        centers = linked.cutout_center
        assert centers.shape == (2, 2)

        # Test accessing center (should be array)
        centers = linked.center
        assert centers.shape == (2, 2)

    def test_getattr_single_star(self, simple_wcs):
        """
        Test __getattr__ with single star (returns scalar not array).
        """
        star = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([star])

        # With single star, should return single value not array
        flux = linked.flux
        assert flux == star.flux
        assert not isinstance(flux, np.ndarray)

    def test_getattr_private_attribute_error(self, simple_wcs):
        """
        Test that accessing non-existent private attributes raises
        error.
        """
        star = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([star])

        # Accessing non-existent private attribute should raise
        match = "'LinkedEPSFStar' object has no attribute"
        with pytest.raises(AttributeError, match=match):
            _ = linked._nonexistent_attribute

    def test_pickle_operations(self, simple_wcs):
        """
        Test __getstate__ and __setstate__ for pickling.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star2 = EPSFStar(np.ones((7, 7)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([star1, star2])

        # Test getstate/setstate
        state = linked.__getstate__()
        new_linked = LinkedEPSFStar([EPSFStar(np.ones((3, 3)),
                                              wcs_large=simple_wcs)])
        new_linked.__setstate__(state)
        assert len(new_linked) == 2

    def test_flat_properties(self, simple_wcs):
        """
        Test cutout_center_flat and center_flat properties.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs,
                         origin=(10, 20))
        star2 = EPSFStar(np.ones((7, 7)), wcs_large=simple_wcs,
                         origin=(30, 40))
        linked = LinkedEPSFStar([star1, star2])

        # Test cutout_center_flat
        centers_flat = linked.cutout_center_flat
        assert centers_flat.shape == (2, 2)
        assert_array_equal(centers_flat[0], star1.cutout_center)
        assert_array_equal(centers_flat[1], star2.cutout_center)

        # Test center_flat
        centers = linked.center_flat
        assert centers.shape == (2, 2)
        assert_array_equal(centers[0], star1.center)
        assert_array_equal(centers[1], star2.center)

    def test_counting_properties(self, simple_wcs):
        """
        Test n_stars, n_all_stars, and n_good_stars properties.
        """
        star1 = EPSFStar(np.ones((5, 5)), wcs_large=simple_wcs)
        star2 = EPSFStar(np.ones((7, 7)), wcs_large=simple_wcs)
        star3 = EPSFStar(np.ones((6, 6)), wcs_large=simple_wcs)
        linked = LinkedEPSFStar([star1, star2, star3])

        # Test n_stars and n_all_stars (should be same for
        # LinkedEPSFStar)
        assert linked.n_stars == 3
        assert linked.n_all_stars == 3

        # Test n_good_stars
        assert linked.n_good_stars == 3

        # Exclude one star
        star2._excluded_from_fit = True
        assert linked.n_good_stars == 2


class TestExtractStars:
    """
    Tests for extract_stars function.
    """

    def test_extract_stars(self, stars_nddata, stars_table):
        """
        Test basic star extraction functionality.
        """
        size = 11
        stars = extract_stars(stars_nddata, stars_table, size=size)
        assert len(stars) == 4
        assert isinstance(stars, EPSFStars)
        assert isinstance(stars[0], EPSFStars)
        assert stars[0].data.shape == (size, size)
        assert stars.n_stars == stars.n_all_stars
        assert stars.n_stars == stars.n_good_stars
        assert stars.center.shape == (len(stars), 2)

    def test_extract_stars_inputs(self, stars_nddata, stars_table):
        """
        Test extract_stars input validation.
        """
        match = 'data must be a single NDData object or list of NDData objects'
        with pytest.raises(TypeError, match=match):
            extract_stars(np.ones(3), stars_table)

        match = 'All catalog elements must be Table objects'
        with pytest.raises(TypeError, match=match):
            extract_stars(stars_nddata, [(1, 1), (2, 2), (3, 3)])

        match = 'number of catalogs must match the number of input images'
        with pytest.raises(ValueError, match=match):
            extract_stars(stars_nddata, [stars_table, stars_table])

        match = 'the catalog must have a "skycoord" column'
        with pytest.raises(ValueError, match=match):
            extract_stars([stars_nddata, stars_nddata], stars_table)

    def test_empty_catalog(self, simple_nddata):
        """
        Test extraction with empty catalog.
        """
        empty_table = Table()
        empty_table['x'] = []
        empty_table['y'] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            stars = extract_stars(simple_nddata, empty_table)
            assert len(stars) == 0
            # Should warn about empty catalog
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any('empty' in msg.lower() for msg in warning_messages)

    def test_stars_outside_image(self, simple_nddata):
        """
        Test extraction with stars outside image bounds.
        """
        table = Table()
        table['x'] = [-10, 100]  # Outside image bounds
        table['y'] = [25, 25]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            stars = extract_stars(simple_nddata, table, size=11)
            assert len(stars) == 0
            # Should warn about excluded stars
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any('not extracted' in msg for msg in warning_messages)

    def test_invalid_input_types(self, simple_nddata):
        """
        Test extraction with invalid input types.
        """
        table = Table()
        table['x'] = [25]
        table['y'] = [25]

        # Test invalid data type
        with pytest.raises(TypeError, match='must be a single NDData object'):
            extract_stars('not_nddata', table)

        # Test invalid catalog type
        with pytest.raises(TypeError, match='must be a single Table object'):
            extract_stars(simple_nddata, 'not_table')

    def test_coordinate_validation(self, simple_nddata):
        """
        Test coordinate system validation.
        """
        table = Table()
        table['x'] = [25]
        table['y'] = [25]

        # Test missing skycoord for multiple images
        with pytest.raises(ValueError, match='must have a "skycoord" column'):
            extract_stars([simple_nddata, simple_nddata], table)

        # Test missing coordinate columns
        bad_table = Table()
        bad_table['flux'] = [100]  # No x, y, or skycoord

        with pytest.raises(ValueError, match='must have either'):
            extract_stars(simple_nddata, bad_table)

    def test_data_validation(self, simple_table):
        """
        Test data input validation.
        """
        # Test invalid data types in list
        with pytest.raises(TypeError, match='All data elements must be'):
            extract_stars(['not_nddata'], simple_table)

        # Test NDData with no data array
        empty_nddata = NDData(np.array([]))  # Provide empty array
        with pytest.raises(ValueError, match='must contain 2D data'):
            extract_stars(empty_nddata, simple_table)

        # Test NDData with wrong dimensions
        nddata_1d = NDData(np.ones(50))
        with pytest.raises(ValueError, match='must contain 2D data'):
            extract_stars(nddata_1d, simple_table)

    def test_catalog_validation(self, simple_nddata):
        """
        Test catalog input validation.
        """
        # Test invalid catalog types in list
        with pytest.raises(TypeError, match='All catalog elements must be'):
            extract_stars(simple_nddata, ['not_table'])

    def test_coordinate_system_validation(self, simple_nddata):
        """
        Test coordinate system validation for complex cases.
        """
        # Test skycoord-only catalog without WCS
        skycoord_table = Table()
        skycoord_table['skycoord'] = [SkyCoord(0, 0, unit='deg')]

        with pytest.raises(ValueError,
                           match='must have a wcs attribute'):
            extract_stars(simple_nddata, skycoord_table)

        # Test multiple catalogs with mismatched count
        table1 = Table({'x': [25], 'y': [25]})
        table2 = Table({'x': [25], 'y': [25]})
        with pytest.raises(ValueError,
                           match='number of catalogs must match'):
            extract_stars(simple_nddata, [table1, table2])

    def test_extract_stars_skycoord_and_wcs(self, simple_data, simple_wcs):
        """
        Test extract_stars with skycoord input and WCS.
        """
        nddata_with_wcs = NDData(simple_data)
        nddata_with_wcs.wcs = simple_wcs

        table = Table()
        table['skycoord'] = [SkyCoord(0, 0, unit='deg')]

        stars = extract_stars(nddata_with_wcs, table, size=(11, 11))

        valid_stars = [s for s in stars.all_stars if s is not None]
        assert len(valid_stars) >= 1

    def test_extract_stars_size_validation_coverage(self, simple_nddata):
        """
        Test size validation paths in extract_stars.
        """
        table = Table({'x': [25], 'y': [25]})

        # Test various size configurations to hit validation paths.
        # This should exercise the as_pair validation.
        stars = extract_stars(simple_nddata, table, size=11)
        assert len(stars) == 1

        # Test tuple size
        stars = extract_stars(simple_nddata, table, size=(11, 13))
        assert len(stars) == 1
        assert stars[0].data.shape == (11, 13)

    def test_extract_stars_coordinate_conversion_paths(self,
                                                       simple_data,
                                                       simple_wcs):
        """
        Test coordinate conversion paths in extract_stars.
        """
        nddata_with_wcs = NDData(simple_data)
        nddata_with_wcs.wcs = simple_wcs

        # Test with both x,y and skycoord present (should prefer x,y)
        table = Table()
        table['x'] = [25.0]
        table['y'] = [25.0]
        table['skycoord'] = [SkyCoord(0, 0, unit='deg')]

        stars = extract_stars(nddata_with_wcs, table, size=11)
        assert len(stars) == 1

    def test_extract_stars_id_handling(self, simple_nddata):
        """
        Test ID handling in extract_stars.
        """
        # Test with explicit IDs
        table = Table()
        table['x'] = [25, 30]
        table['y'] = [25, 30]
        table['id'] = ['star_a', 'star_b']

        stars = extract_stars(simple_nddata, table, size=11)
        assert len(stars) == 2
        assert stars[0].id_label == 'star_a'
        assert stars[1].id_label == 'star_b'

        # Test without IDs (should auto-generate)
        table_no_id = Table()
        table_no_id['x'] = [25, 30]
        table_no_id['y'] = [25, 30]

        stars = extract_stars(simple_nddata, table_no_id, size=11)
        assert len(stars) == 2
        assert stars[0].id_label == 1  # Auto-generated starting from 1
        assert stars[1].id_label == 2

    def test_extract_linked_stars_multiple_images(self, simple_wcs):
        """
        Test extracting linked stars from multiple images with single
        catalog.
        """
        # Create two images with WCS
        data1 = np.ones((50, 50)) * 10
        data2 = np.ones((50, 50)) * 20
        nddata1 = NDData(data1)
        nddata1.wcs = simple_wcs
        nddata2 = NDData(data2)
        nddata2.wcs = simple_wcs

        # Create catalog with skycoord at center of image
        table = Table()
        table['skycoord'] = [SkyCoord(0, 0, unit='deg')]

        # Extract linked stars (suppress warnings to avoid pytest error)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            stars = extract_stars([nddata1, nddata2], table, size=11)

        # Should have 1 linked star containing 2 EPSFStar objects
        assert len(stars) == 1
        assert isinstance(stars._data[0], LinkedEPSFStar)
        assert len(stars._data[0]) == 2

    def test_extract_unlinked_stars_multiple_catalogs(self):
        """
        Test extracting stars with multiple catalogs (no linking).
        """
        # Create two images
        data1 = np.ones((50, 50)) * 10
        data2 = np.ones((50, 50)) * 20
        nddata1 = NDData(data1)
        nddata2 = NDData(data2)

        # Create two catalogs with different stars
        table1 = Table({'x': [25], 'y': [25]})
        table2 = Table({'x': [30], 'y': [30]})

        # Extract stars
        stars = extract_stars([nddata1, nddata2], [table1, table2], size=11)

        # Should have 2 separate (not linked) stars
        assert len(stars) == 2
        assert all(isinstance(s, EPSFStar) for s in stars._data)

    def test_extract_linked_stars_partial_extraction(self, simple_wcs):
        """
        Test linked star extraction where star is valid in one image but
        not another (edge case).
        """
        # Create two images - second one is smaller so star near edge
        # won't be extractable
        data1 = np.ones((50, 50)) * 10
        data2 = np.ones((20, 20)) * 20  # Smaller image
        nddata1 = NDData(data1)
        nddata1.wcs = simple_wcs
        nddata2 = NDData(data2)
        nddata2.wcs = simple_wcs

        # Create catalog with star at position that's valid in first
        # but not second
        table = Table()
        table['skycoord'] = [SkyCoord(0, 0, unit='deg')]  # Center

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            stars = extract_stars([nddata1, nddata2], table, size=11)

        # Should have extracted at least 1 star (from first image)
        # The second image star is outside bounds so only 1 is extracted
        assert len(stars) >= 1

    def test_extract_stars_flux_estimation_failure(self):
        """
        Test that EPSFStar creation failure emits warning.
        """
        # Create data where star cutout will have all zeros (fails flux
        # estimation)
        data = np.zeros((50, 50))
        nddata = NDData(data)

        table = Table({'x': [25], 'y': [25]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            stars = extract_stars(nddata, table, size=11)
            # Should warn about failed EPSFStar creation
            warning_messages = [str(warning.message) for warning in w]
            assert any('Failed to create EPSFStar' in msg
                       for msg in warning_messages)

        # No valid stars should be extracted
        assert len(stars) == 0

    def test_extract_stars_nonfinite_weights_warning(self):
        """
        Test that non-finite weights in uncertainty emit warning.
        """
        data = np.ones((50, 50)) * 100
        nddata = NDData(data)

        # Create uncertainty with non-finite values
        uncertainty = np.ones((50, 50)) * 0.1
        uncertainty[20:30, 20:30] = 0  # Will cause 1/0 = inf in weights
        nddata.uncertainty = StdDevUncertainty(uncertainty)

        table = Table({'x': [25], 'y': [25]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            stars = extract_stars(nddata, table, size=11)
            # Should warn about non-finite weights
            warning_messages = [str(warning.message) for warning in w]
            assert any('non-finite weight values' in msg
                       for msg in warning_messages)

        # Star should still be extracted (non-finite weights set to 0)
        assert len(stars) == 1

    def test_validate_single_catalog_multiple_images_no_wcs(self):
        """
        Test validation error when single catalog with multiple images
        but images lack WCS.
        """
        # Create two images without WCS
        data1 = np.ones((50, 50))
        data2 = np.ones((50, 50))
        nddata1 = NDData(data1)
        nddata2 = NDData(data2)

        # Create catalog with skycoord
        table = Table()
        table['skycoord'] = [SkyCoord(0, 0, unit='deg')]

        # Should raise because images don't have WCS
        with pytest.raises(ValueError, match='must have a wcs attribute'):
            extract_stars([nddata1, nddata2], table, size=11)

    def test_validate_skycoord_only_catalog_no_wcs(self):
        """
        Test validation when catalog has only skycoord but NDData lacks
        WCS.
        """
        # Create NDData without WCS
        nddata = NDData(np.ones((50, 50)))

        # Create catalog with only skycoord (no x, y columns)
        table = Table()
        table['skycoord'] = [SkyCoord(0, 0, unit='deg')]

        # Should raise because NDData does not have WCS
        with pytest.raises(ValueError, match='must have a wcs attribute'):
            extract_stars(nddata, table, size=11)

    def test_validate_multiple_catalogs_skycoord_only_no_wcs(self, simple_wcs):
        """
        Test validation when catalog has only skycoord and some NDData
        objects lack WCS.

        This tests the branch where the corresponding NDData has WCS,
        but another NDData in the list does not have WCS.
        """
        nddata1 = NDData(np.ones((50, 50)))
        # nddata1 intentionally has no WCS
        nddata2 = NDData(np.ones((50, 50)))
        nddata2.wcs = simple_wcs  # Second image has WCS

        # First catalog has x,y (does not need WCS), second has only
        # skycoord
        table1 = Table({'x': [25], 'y': [25]})
        table2 = Table()
        table2['skycoord'] = [SkyCoord(0, 0, unit='deg')]

        # nddata2 has WCS, but nddata1 does not
        with pytest.raises(ValueError,
                           match='each NDData object must have a wcs'):
            extract_stars([nddata1, nddata2], [table1, table2], size=11)
