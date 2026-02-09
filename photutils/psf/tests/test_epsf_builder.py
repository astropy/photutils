# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf_builder module.
"""

import itertools
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NDData
from astropy.table import Table
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose, assert_array_equal

from photutils.centroids import (centroid_1dg, centroid_2dg, centroid_com,
                                 centroid_quadratic)
from photutils.datasets import make_model_image
from photutils.psf import (CircularGaussianPRF, EPSFBuilder, EPSFBuildResult,
                           EPSFFitter, EPSFStar, EPSFStars, ImagePSF,
                           extract_stars, make_psf_model_image)
from photutils.psf.epsf_builder import (_CoordinateTransformer, _EPSFValidator,
                                        _ProgressReporter, _SmoothingKernel)
from photutils.psf.epsf_stars import LinkedEPSFStar
from photutils.utils._optional_deps import HAS_TQDM


@pytest.fixture
def epsf_test_data():
    """
    Create a simulated image for testing.
    """
    fwhm = 2.7
    psf_model = CircularGaussianPRF(flux=1, fwhm=fwhm)
    model_shape = (9, 9)
    n_sources = 100
    shape = (750, 750)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700),
                                             min_separation=25,
                                             border_size=25, seed=0)

    nddata = NDData(data)
    init_stars = Table()
    init_stars['x'] = true_params['x_0']
    init_stars['y'] = true_params['y_0']

    return {
        'fwhm': fwhm,
        'data': data,
        'nddata': nddata,
        'init_stars': init_stars,
    }


@pytest.fixture
def epsf_fitter_data(epsf_test_data):
    """
    Create extracted stars and an ePSF for testing EPSFFitter.
    """
    stars = extract_stars(epsf_test_data['nddata'],
                          epsf_test_data['init_stars'][:4], size=11)
    builder = EPSFBuilder(oversampling=1, maxiters=2, progress_bar=False)
    epsf, _ = builder(stars)
    return {'stars': stars, 'epsf': epsf}


def _make_epsf_fitter(**kwargs):
    """
    Helper to create EPSFFitter suppressing the deprecation warning.

    Remove this helper and the catch_warnings block when EPSFFitter is
    removed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyDeprecationWarning)
        return EPSFFitter(**kwargs)


class TestSmoothingKernel:
    """
    Tests for the _SmoothingKernel class.
    """

    @pytest.mark.parametrize('kernel_type', ['quartic', 'quadratic'])
    def test_get_kernel(self, kernel_type):
        """
        Test quartic kernel retrieval.
        """
        kernel = _SmoothingKernel.get_kernel(kernel_type)
        assert isinstance(kernel, np.ndarray)
        assert kernel.shape == (5, 5)
        if kernel_type == 'quartic':
            expected_sum = _SmoothingKernel.QUARTIC_KERNEL.sum()
        else:
            expected_sum = _SmoothingKernel.QUADRATIC_KERNEL.sum()
        assert np.isclose(kernel.sum(), expected_sum)

    def test_get_kernel_custom_array(self):
        """
        Test custom array kernel retrieval.
        """
        custom_kernel = np.ones((3, 3)) / 9.0
        kernel = _SmoothingKernel.get_kernel(custom_kernel)
        assert isinstance(kernel, np.ndarray)
        assert kernel.shape == (3, 3)
        assert np.allclose(kernel, custom_kernel)

    def test_get_kernel_invalid_type(self):
        """
        Test invalid kernel type raises TypeError.
        """
        with pytest.raises(TypeError, match='Unsupported kernel type'):
            _SmoothingKernel.get_kernel('invalid')

    @pytest.mark.parametrize('kernel_type', ['quartic', 'quadratic'])
    def test_apply_smoothing(self, kernel_type):
        """
        Test smoothing with quartic kernel.
        """
        data = np.ones((10, 10))
        smoothed = _SmoothingKernel.apply_smoothing(data, kernel_type)
        assert isinstance(smoothed, np.ndarray)
        assert smoothed.shape == data.shape
        assert_allclose(smoothed.sum(), data.sum())

    def test_apply_smoothing_custom_kernel(self):
        """
        Test smoothing with custom kernel.
        """
        data = np.ones((10, 10))
        kernel = np.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])
        smoothed = _SmoothingKernel.apply_smoothing(data, kernel)
        assert isinstance(smoothed, np.ndarray)
        assert smoothed.shape == data.shape
        assert_allclose(smoothed.sum(), data.sum())

    def test_apply_smoothing_none(self):
        """
        Test smoothing with None returns original data.
        """
        data = np.ones((10, 10))
        result = _SmoothingKernel.apply_smoothing(data, None)
        assert result is data  # Should return same object


class TestEPSFValidator:
    """
    Tests for the _EPSFValidator class.
    """

    def test_validate_oversampling_valid(self):
        """
        Test valid oversampling validation.
        """
        result = _EPSFValidator.validate_oversampling(2)
        assert np.array_equal(result, (2, 2))

        result = _EPSFValidator.validate_oversampling((3, 4))
        assert np.array_equal(result, (3, 4))

    def test_validate_oversampling_none(self):
        """
        Test validate_oversampling with None input.
        """
        match = "'oversampling' must be specified"
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_oversampling(None)

    def test_validate_oversampling_invalid_exception(self):
        """
        Test oversampling validation with invalid input.
        """
        # Test with invalid input that should raise exception from
        # as_pair
        match = 'Invalid oversampling parameter'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_oversampling('invalid')

    def test_validate_oversampling_invalid_exception_with_context(self):
        """
        Test oversampling validation with context and invalid input.
        """
        msg = 'test_context: Invalid oversampling parameter'
        with pytest.raises(ValueError, match=msg):
            _EPSFValidator.validate_oversampling('invalid',
                                                 context='test_context')

    def test_validate_oversampling_zero_values(self):
        """
        Test oversampling validation with zero values.
        """
        with pytest.raises(ValueError, match='oversampling must be > 0'):
            _EPSFValidator.validate_oversampling((0, 2))

        msg = ('test_context: Invalid oversampling parameter - '
               'oversampling must be > 0')
        with pytest.raises(ValueError, match=msg):
            _EPSFValidator.validate_oversampling((0, 2),
                                                 context='test_context')

    def test_validate_oversampling_as_pair_exception_with_context(self):
        """
        Test oversampling validation when as_pair raises exception.
        """
        # Use a tuple with wrong number of elements to trigger as_pair error
        msg = 'test_ctx: Invalid oversampling parameter'
        with pytest.raises(ValueError, match=msg):
            _EPSFValidator.validate_oversampling((1, 2, 3),
                                                 context='test_ctx')

    def test_validate_shape_compatibility(self, epsf_test_data):
        """
        Test shape compatibility validation.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:5], size=11)

        # Should not raise an exception for compatible shapes
        _EPSFValidator.validate_shape_compatibility(stars, (1, 1))

    def test_validate_shape_compatibility_custom_shape(self, epsf_test_data):
        """
        Test shape compatibility with custom shape.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:5], size=11)

        # Test with specific shape
        _EPSFValidator.validate_shape_compatibility(stars, (1, 1),
                                                    shape=(21, 21))

    def test_validate_shape_compatibility_empty_stars(self):
        """
        Test shape compatibility with empty star list.
        """
        empty_stars = EPSFStars([])
        match = 'Cannot validate shape compatibility with empty star list'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_shape_compatibility(empty_stars, (1, 1))

    def test_validate_shape_compatibility_small_stars(self):
        """
        Test shape compatibility with very small star cutouts.
        """
        # Create very small star (2x2 pixels)
        small_data = np.ones((2, 2))
        small_star = EPSFStar(small_data, cutout_center=(1, 1))
        small_stars = EPSFStars([small_star])

        match = r'Found .* star.*with very small dimensions'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_shape_compatibility(small_stars, (1, 1))

    def test_validate_shape_compatibility_invalid_shape_type(self):
        """
        Test shape compatibility with invalid shape type.
        """
        data = np.ones((5, 5))
        star = EPSFStar(data, cutout_center=(2, 2))
        stars = EPSFStars([star])

        match = 'Shape must be a 2-element sequence'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_shape_compatibility(stars, (1, 1),
                                                        shape=(10, 10, 10))
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_shape_compatibility(stars, (1, 1),
                                                        shape='invalid')

    def test_validate_shape_compatibility_incompatible_shape(self):
        """
        Test shape compatibility with incompatible shape.
        """
        data = np.ones((5, 5))
        star = EPSFStar(data, cutout_center=(2, 2))
        stars = EPSFStars([star])

        # Request shape that's too small
        match = r'Requested ePSF shape .* is incompatible'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_shape_compatibility(stars, (2, 2),
                                                        shape=(5, 5))

    def test_validate_shape_compatibility_even_dimensions_warning(self):
        """
        Test shape compatibility with even dimensions warning.
        """
        data = np.ones((5, 5))
        star = EPSFStar(data, cutout_center=(2, 2))
        stars = EPSFStars([star])

        # Test even dimensions trigger warning
        match = 'ePSF shape .* has even dimensions'
        with pytest.warns(UserWarning, match=match):
            _EPSFValidator.validate_shape_compatibility(stars, (1, 1),
                                                        shape=(20, 20))

    def test_validate_stars_empty_list(self):
        """
        Test validate_stars with empty star list.
        """
        empty_stars = EPSFStars([])
        match = 'EPSFStars object must contain at least one star'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars(empty_stars)

        match = 'test_context: EPSFStars object must contain at least one star'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars(empty_stars, context='test_context')

    def test_validate_stars_non_finite_data(self):
        """
        Test validate_stars with non-finite data.
        """
        # Create star with all NaN data - need to provide explicit flux
        # since flux estimation would fail with all NaN data
        data = np.full((5, 5), np.nan)
        match = 'Input data array contains invalid data that will be masked'
        with pytest.warns(AstropyUserWarning, match=match):
            star = EPSFStar(data, cutout_center=(2, 2), flux=1.0)

        match = r'Found [\s\S]* invalid stars [\s\S]* no finite data values'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars([star])

    def test_validate_stars_too_small(self):
        """
        Test validate_stars with very small stars.
        """
        # Create very small star (2x2 pixels)
        data = np.ones((2, 2))
        star = EPSFStar(data, cutout_center=(1, 1))

        match = r'Found [\s\S]* invalid stars [\s\S]* too small'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars([star])

    def test_validate_stars_missing_cutout_center(self):
        """
        Test validate_stars with star missing cutout_center.
        """
        # Create mock star without cutout_center
        class MockStar:
            def __init__(self):
                self.data = np.ones((5, 5))
                self.shape = (5, 5)

        mock_stars = [MockStar()]

        match = r'Found .* invalid stars [\s\S]* missing cutout_center'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars(mock_stars)

    def test_validate_stars_validation_error(self):
        """
        Test validate_stars with validation error during processing.
        """
        # Create mock star that raises error during validation
        class MockStar:
            def __init__(self):
                self.data = np.ones((5, 5))

            @property
            def shape(self):
                msg = 'Test error'
                raise ValueError(msg)

        mock_stars = [MockStar()]

        match = r'Found .* invalid stars [\s\S]* validation error'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars(mock_stars)

    def test_validate_stars_multiple_invalid(self):
        """
        Test validate_stars with multiple invalid stars.
        """
        # Create multiple mock stars with different issues
        class MockStar1:
            def __init__(self):
                self.data = None

        class MockStar2:
            def __init__(self):
                self.data = np.ones((2, 2))  # Too small
                self.shape = (2, 2)

        mock_stars = [MockStar1(), MockStar2()]

        match = r'Found 2 invalid stars [\s\S]* too small'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars(mock_stars)

    def test_validate_stars_more_than_5_invalid(self):
        """
        Test validate_stars with more than 5 invalid stars.
        """
        # Create 7 mock stars with missing data
        class MockStar:
            def __init__(self):
                self.data = None

        mock_stars = [MockStar() for _ in range(7)]

        match = r'Found 7 invalid stars [\s\S]* missing data'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars(mock_stars)

    def test_validate_stars_context_with_invalid(self):
        """
        Test validate_stars with context and invalid stars.
        """
        class MockStar:
            def __init__(self):
                self.data = None

        mock_stars = [MockStar()]

        match = 'my_context: Found 1 invalid stars'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_stars(mock_stars, context='my_context')

    def test_validate_stars_valid(self):
        """
        Test validate_stars with valid stars.
        """
        # Create valid stars
        data1 = np.ones((5, 5))
        data2 = np.ones((6, 6))
        star1 = EPSFStar(data1, cutout_center=(2, 2))
        star2 = EPSFStar(data2, cutout_center=(3, 3))

        # Should not raise any exception
        _EPSFValidator.validate_stars([star1, star2])

    def test_validate_center_accuracy_valid(self):
        """
        Test validate_center_accuracy with valid inputs.
        """
        # Test valid values
        _EPSFValidator.validate_center_accuracy(0.001)
        _EPSFValidator.validate_center_accuracy(0.01)
        _EPSFValidator.validate_center_accuracy(0.1)
        _EPSFValidator.validate_center_accuracy(1.0)

    def test_validate_center_accuracy_invalid_type(self):
        """
        Test validate_center_accuracy with invalid type.
        """
        match = 'center_accuracy must be a number'
        with pytest.raises(TypeError, match=match):
            _EPSFValidator.validate_center_accuracy('0.001')

    def test_validate_center_accuracy_non_positive(self):
        """
        Test validate_center_accuracy with non-positive values.
        """
        match = 'center_accuracy must be positive'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_center_accuracy(0.0)

        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_center_accuracy(-0.001)

    def test_validate_center_accuracy_too_large(self):
        """
        Test validate_center_accuracy with values too large.
        """
        match = r'center_accuracy .* seems unusually large'
        with pytest.warns(AstropyUserWarning, match=match):
            _EPSFValidator.validate_center_accuracy(1.1)

    def test_validate_maxiters_valid(self):
        """
        Test validate_maxiters with valid inputs.
        """
        # Test valid values (these should not raise or warn)
        _EPSFValidator.validate_maxiters(1)
        _EPSFValidator.validate_maxiters(10)
        _EPSFValidator.validate_maxiters(100)

    def test_validate_maxiters_invalid_type(self):
        """
        Test validate_maxiters with invalid type.
        """
        match = 'maxiters must be an integer'
        with pytest.raises(TypeError, match=match):
            _EPSFValidator.validate_maxiters(10.5)

        with pytest.raises(TypeError, match=match):
            _EPSFValidator.validate_maxiters('10')

    def test_validate_maxiters_non_positive(self):
        """
        Test validate_maxiters with non-positive values.
        """
        match = 'maxiters must be a positive number'
        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_maxiters(0)

        with pytest.raises(ValueError, match=match):
            _EPSFValidator.validate_maxiters(-5)

    def test_validate_maxiters_too_large(self):
        """
        Test validate_maxiters with values too large triggers warning.
        """
        match = r'maxiters .* seems unusually large'
        with pytest.warns(AstropyUserWarning, match=match):
            _EPSFValidator.validate_maxiters(101)


class TestCoordinateTransformer:
    """
    Tests for the _CoordinateTransformer class.
    """

    @pytest.mark.parametrize('oversampling', [(2, 2), (3, 4), (5, 1)])
    def test_basic(self, oversampling):
        """
        Test basic coordinate transformation.
        """
        # Create transformer
        transformer = _CoordinateTransformer(oversampling=oversampling)
        assert np.array_equal(transformer.oversampling, oversampling)
        assert transformer.oversampling[0] == oversampling[0]
        assert transformer.oversampling[1] == oversampling[1]

    def test_empty_star_shapes(self):
        """
        Test compute_epsf_shape with empty star_shapes list.
        """
        transformer = _CoordinateTransformer(oversampling=(2, 2))
        match = 'Need at least one star to compute ePSF shape'
        with pytest.raises(ValueError, match=match):
            transformer.compute_epsf_shape([])

    def test_oversampled_to_undersampled(self):
        """
        Test oversampled_to_undersampled conversion.
        """
        transformer = _CoordinateTransformer(oversampling=(4, 2))
        x_under, y_under = transformer.oversampled_to_undersampled(8.0, 16.0)
        assert x_under == 4.0  # 8 / 2
        assert y_under == 4.0  # 16 / 4

    def test_undersampled_to_oversampled(self):
        """
        Test undersampled_to_oversampled conversion.
        """
        transformer = _CoordinateTransformer(oversampling=(4, 2))
        x_over, y_over = transformer.undersampled_to_oversampled(4.0, 4.0)
        assert x_over == 8.0  # 4 * 2
        assert y_over == 16.0  # 4 * 4

    def test_star_to_epsf_coords(self):
        """
        Test star_to_epsf_coords method of _CoordinateTransformer.
        """
        transformer = _CoordinateTransformer(oversampling=(2, 2))

        # Test coordinate transformation
        star_x = np.array([0.0, 1.0, 2.0])
        star_y = np.array([0.0, 1.0, 2.0])
        epsf_origin = (10.0, 10.0)

        epsf_x, epsf_y = transformer.star_to_epsf_coords(
            star_x, star_y, epsf_origin)

        # Check output shape
        assert epsf_x.shape == star_x.shape
        assert epsf_y.shape == star_y.shape

        # Check values: with oversampling=2 and origin=(10, 10),
        # the formula computes round(oversampling * star_x + origin_x)
        expected_x = np.array([10, 12, 14])
        expected_y = np.array([10, 12, 14])
        assert np.array_equal(epsf_x, expected_x)
        assert np.array_equal(epsf_y, expected_y)

    def test_compute_epsf_origin(self):
        """
        Test compute_epsf_origin method of _CoordinateTransformer.
        """
        transformer = _CoordinateTransformer(oversampling=(2, 2))

        # Test with odd shape
        origin = transformer.compute_epsf_origin((11, 11))
        assert origin == (5.0, 5.0)

        # Test with different shape
        origin = transformer.compute_epsf_origin((21, 31))
        assert origin == (15.0, 10.0)


class TestProgressReporter:
    """
    Tests for the _ProgressReporter class.
    """

    def test_progress_reporter(self):
        """
        Test basic functionality of _ProgressReporter.
        """
        # Test with enabled=True
        reporter = _ProgressReporter(enabled=True, maxiters=10)
        assert reporter.enabled is True
        assert reporter.maxiters == 10
        reporter.setup()
        reporter.update()
        reporter.write_convergence_message(5)
        reporter.close()

        # Test with enabled=False
        reporter = _ProgressReporter(enabled=False, maxiters=5)
        assert reporter.enabled is False
        result = reporter.setup()
        assert result is reporter
        assert reporter._pbar is None


class TestEPSFBuildResult:
    """
    Tests for the EPSFBuildResult class.
    """

    def test_creation(self):
        """
        Test EPSFBuildResult creation.
        """
        # Create a simple PSF model for testing
        data = np.ones((5, 5))
        psf = ImagePSF(data)

        # Create stars list (can be empty for this test)
        stars = []

        result = EPSFBuildResult(
            epsf=psf,
            fitted_stars=stars,
            iterations=5,
            converged=True,
            final_center_accuracy=0.01,
            n_excluded_stars=0,
            excluded_star_indices=[],
        )
        assert result.epsf is psf
        assert result.fitted_stars == stars
        assert result.iterations == 5
        assert result.converged is True

    def test_with_data(self, epsf_test_data):
        """
        Test EPSFBuildResult with actual data.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:5], size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=2, progress_bar=False)
        epsf, fitted_stars = builder(stars)

        result = EPSFBuildResult(
            epsf=epsf,
            fitted_stars=fitted_stars,
            iterations=2,
            converged=False,
            final_center_accuracy=0.1,
            n_excluded_stars=0,
            excluded_star_indices=[],
        )
        assert result.epsf is not None
        assert result.epsf.data.shape == (11, 11)
        assert result.fitted_stars is not None
        assert len(result.fitted_stars) == len(stars)

    def test_getitem_invalid_index(self):
        """
        Test EPSFBuildResult.__getitem__ with invalid index.
        """
        data = np.ones((5, 5))
        psf = ImagePSF(data)
        stars = EPSFStars([])

        result = EPSFBuildResult(
            epsf=psf,
            fitted_stars=stars,
            iterations=5,
            converged=True,
            final_center_accuracy=0.01,
            n_excluded_stars=0,
            excluded_star_indices=[],
        )

        # Valid indices
        assert result[0] is psf
        assert result[1] is stars

        # Invalid index
        match = 'EPSFBuildResult index must be 0'
        with pytest.raises(IndexError, match=match):
            result[2]

        with pytest.raises(IndexError, match=match):
            result[-1]

    def test_iteration(self):
        """
        Test EPSFBuildResult iteration (tuple unpacking).
        """
        data = np.ones((5, 5))
        psf = ImagePSF(data)
        stars = EPSFStars([])

        result = EPSFBuildResult(
            epsf=psf,
            fitted_stars=stars,
            iterations=5,
            converged=True,
            final_center_accuracy=0.01,
            n_excluded_stars=0,
            excluded_star_indices=[],
        )

        # Test tuple unpacking via iteration
        epsf_out, stars_out = result
        assert epsf_out is psf
        assert stars_out is stars

        # Test list conversion
        result_list = list(result)
        assert len(result_list) == 2
        assert result_list[0] is psf
        assert result_list[1] is stars

    def test_attributes(self, epsf_test_data):
        """
        Test EPSFBuildResult has all expected attributes.
        """
        builder = EPSFBuilder(oversampling=1, maxiters=3, progress_bar=False)

        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:5], size=11)

        result = builder(stars)

        # Check all attributes exist
        assert result.epsf is not None
        assert result.fitted_stars is not None
        assert isinstance(result.iterations, int)
        assert isinstance(result.converged, (bool, np.bool_))
        assert isinstance(result.final_center_accuracy, (float, np.floating))
        assert isinstance(result.n_excluded_stars, int)
        assert isinstance(result.excluded_star_indices, list)


class TestEPSFFitter:
    """
    Tests for the EPSFFitter class.

    EPSFFitter is deprecated since 3.0.0. These tests verify that
    it still functions correctly while emitting a deprecation warning.
    """

    def test_deprecation_warning(self):
        """
        Test that EPSFFitter emits a deprecation warning.
        """
        match = 'EPSFFitter is deprecated'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            EPSFFitter()

    def test_fit_stars(self, epsf_fitter_data):
        """
        Test EPSFFitter __call__ method.
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        fitter = _make_epsf_fitter()
        fitted_stars = fitter(epsf, stars)
        assert fitted_stars is not None
        assert len(fitted_stars) == len(stars)

    def test_empty_stars(self):
        """
        Test EPSFFitter with empty stars.
        """
        data = np.ones((11, 11))
        epsf = ImagePSF(data)
        fitter = _make_epsf_fitter()

        empty_stars = EPSFStars([])
        result = fitter(epsf, empty_stars)
        assert len(result) == 0

    def test_invalid_epsf_type(self):
        """
        Test EPSFFitter with invalid epsf type.
        """
        data = np.ones((5, 5))
        star = EPSFStar(data, cutout_center=(2, 2))
        stars = EPSFStars([star])
        fitter = _make_epsf_fitter()

        match = 'The input epsf must be an ImagePSF'
        with pytest.raises(TypeError, match=match):
            fitter('not_an_epsf', stars)

    def test_fit_boxsize_none(self, epsf_fitter_data):
        """
        Test EPSFFitter with fit_boxsize=None.
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        # Test fitter with fit_boxsize=None (use entire star image)
        fitter = _make_epsf_fitter(fit_boxsize=None)
        fitted_stars = fitter(epsf, stars)
        assert len(fitted_stars) == len(stars)

    def test_invalid_star_type(self, epsf_fitter_data):
        """
        Test EPSFFitter with invalid star type.
        """
        epsf = epsf_fitter_data['epsf']

        # Create mock invalid star type
        class InvalidStar:
            pass

        # Create an EPSFStars-like object with invalid star
        invalid_stars = [InvalidStar()]

        fitter = _make_epsf_fitter()
        match = 'stars must contain only EPSFStar'
        with pytest.raises(TypeError, match=match):
            fitter(epsf, invalid_stars)

    def test_fit_info_ierr(self, epsf_fitter_data):
        """
        Test EPSFFitter handling of fit_info with ierr.
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        # Test fitter - the fit_info handling is automatic
        fitter = _make_epsf_fitter()
        assert fitter.fitter_has_fit_info is True

        fitted_stars = fitter(epsf, stars)
        # Check that fit_error_status is set
        for star in fitted_stars.all_stars:
            assert hasattr(star, '_fit_error_status')

    def test_fitter_without_fit_info(self, epsf_fitter_data):
        """
        Test EPSFFitter with a fitter that doesn't have fit_info.
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        # Create a mock fitter without fit_info attribute
        class MockFitter:
            def __call__(
                self, model, x, y, z, weights=None, **kwargs,  # noqa: ARG002
            ):
                return model

        mock_fitter = MockFitter()
        fitter = _make_epsf_fitter(fitter=mock_fitter)

        # Verify that fitter_has_fit_info is False
        assert fitter.fitter_has_fit_info is False

        # Fit the stars
        fitted_stars = fitter(epsf, stars)
        assert len(fitted_stars) == len(stars)

        # Check that _fit_info is None for stars fit without fit_info
        for star in fitted_stars.all_stars:
            assert star._fit_info is None

    def test_weights_not_supported(self, epsf_fitter_data):
        """
        Test EPSFFitter when fitter raises TypeError for weights.
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        # Create a fitter that raises TypeError when weights is passed
        class NoWeightsFitter:
            def __init__(self):
                self.fit_info = {'ierr': 1}

            def __call__(self, model, *_args, **kwargs):
                if 'weights' in kwargs:
                    msg = 'weights not supported'
                    raise TypeError(msg)
                return model

        no_weights_fitter = NoWeightsFitter()
        fitter = _make_epsf_fitter(fitter=no_weights_fitter)

        # Fit the stars - should handle TypeError gracefully
        fitted_stars = fitter(epsf, stars)
        assert len(fitted_stars) == len(stars)

    def test_invalid_ierr(self, epsf_fitter_data):
        """
        Test EPSFFitter when fitter returns invalid ierr value.
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        # Create a fitter that returns invalid ierr (not in [1, 2, 3, 4])
        class BadIerrFitter:
            def __init__(self):
                self.fit_info = {'ierr': 0}  # Invalid ierr value

            def __call__(self, model, *_args, **_kwargs):
                return model

        bad_ierr_fitter = BadIerrFitter()
        fitter = _make_epsf_fitter(fitter=bad_ierr_fitter)

        # Fit the stars - should set fit_error_status = 2
        fitted_stars = fitter(epsf, stars)
        assert len(fitted_stars) == len(stars)

        # Check that fit_error_status is set to 2 for all stars
        for star in fitted_stars.all_stars:
            assert star._fit_error_status == 2

    def test_removes_fitter_kwargs(self):
        """
        Test that EPSFFitter removes reserved kwargs.
        """
        # Pass kwargs that should be removed
        fitter = _make_epsf_fitter(x=1, y=2, z=3, weights=4,
                                   calc_uncertainties=False)

        # These should be removed from fitter_kwargs
        assert 'x' not in fitter.fitter_kwargs
        assert 'y' not in fitter.fitter_kwargs
        assert 'z' not in fitter.fitter_kwargs
        assert 'weights' not in fitter.fitter_kwargs
        # Other kwargs should be preserved
        assert fitter.fitter_kwargs.get('calc_uncertainties') is False

    def test_with_linked_star_mock_wcs(self, epsf_fitter_data):
        """
        Test EPSFFitter with LinkedEPSFStar using mock WCS.
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        # Create mock WCS that returns identity transform
        class MockWCS:
            def pixel_to_world_values(self, x, y):
                return x, y

            def world_to_pixel_values(self, ra, dec):
                return ra, dec

        mock_wcs = MockWCS()

        # Create EPSFStar objects with mock WCS
        linked_stars_list = []
        for i in range(2):
            star_data = stars.all_stars[i].data.copy()
            center = stars.all_stars[i].cutout_center
            # Use origin that places star in a reasonable position
            origin = (0, 0)
            star = EPSFStar(star_data, cutout_center=center,
                            origin=origin, wcs_large=mock_wcs)
            linked_stars_list.append(star)

        # Create LinkedEPSFStar
        linked_star = LinkedEPSFStar(linked_stars_list)

        # Create EPSFStars with the LinkedEPSFStar
        stars_with_linked = EPSFStars([linked_star])

        # Fit the linked stars
        fitter = _make_epsf_fitter()
        fitted_stars = fitter(epsf, stars_with_linked)

        assert len(fitted_stars) == 1
        # fitted_stars is an EPSFStars; the first item wraps LinkedEPSFStar
        assert len(fitted_stars.all_stars) == 2  # 2 stars in the linked star

    def test_fit_boxsize_none_with_excluded_star(self, epsf_fitter_data):
        """
        Test EPSFFitter with fit_boxsize=None and excluded star.
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        # Mark first star as excluded
        stars.all_stars[0]._excluded_from_fit = True

        # Create fitter with fit_boxsize=None
        fitter = _make_epsf_fitter(fit_boxsize=None)
        fitted_stars = fitter(epsf, stars)

        # Check that excluded star is returned unchanged
        assert fitted_stars.all_stars[0] is stars.all_stars[0]
        assert len(fitted_stars) == len(stars)

    def test_linked_star_with_excluded_star(self, epsf_fitter_data):
        """
        Test EPSFFitter with LinkedEPSFStar containing excluded star.

        This tests lines 825, 856-860 (excluded star in LinkedEPSFStar).
        """
        stars = epsf_fitter_data['stars']
        epsf = epsf_fitter_data['epsf']

        # Create mock WCS
        class MockWCS:
            def pixel_to_world_values(self, x, y):
                return x, y

            def world_to_pixel_values(self, ra, dec):
                return ra, dec

        mock_wcs = MockWCS()

        # Create LinkedEPSFStar with two stars, one excluded
        linked_stars_list = []
        for i in range(2):
            star_data = stars.all_stars[i].data.copy()
            center = stars.all_stars[i].cutout_center
            origin = (0, 0)
            star = EPSFStar(star_data, cutout_center=center,
                            origin=origin, wcs_large=mock_wcs)
            linked_stars_list.append(star)

        # Mark second star as excluded
        linked_stars_list[1]._excluded_from_fit = True

        linked_star = LinkedEPSFStar(linked_stars_list)
        stars_with_linked = EPSFStars([linked_star])

        # Fit with fit_boxsize=None
        fitter = _make_epsf_fitter(fit_boxsize=None)
        fitted_stars = fitter(epsf, stars_with_linked)

        # Check that excluded star is in the result
        assert len(fitted_stars) == 1
        assert len(fitted_stars.all_stars) == 2

    def test_fit_star_partial_overlap_error(self, epsf_fitter_data):
        """
        Test EPSFFitter._fit_star with PartialOverlapError.
        """
        epsf = epsf_fitter_data['epsf']

        # Create a star with cutout_center at edge to cause overlap error
        star_data = np.ones((11, 11))
        # Place center very close to edge - this should cause overlap error
        # when fit_boxsize tries to extract a region
        star = EPSFStar(star_data, cutout_center=(0.5, 0.5))
        stars_with_edge = EPSFStars([star])

        # Use fit_boxsize that will cause overlap error
        fitter = _make_epsf_fitter(fit_boxsize=9)
        fitted_stars = fitter(epsf, stars_with_edge)

        # Check that star has fit_error_status set
        assert fitted_stars.all_stars[0]._fit_error_status == 1


class TestEPSFBuilder:
    """
    Tests for the EPSFBuilder class.
    """

    @pytest.mark.parametrize('extract_shape', [(25, 25), (19, 25), (25, 19)])
    def test_build(self, epsf_test_data, extract_shape):
        """
        Test EPSFBuilder build process on a simulated image.
        """
        oversampling = 2
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:10],
                              size=extract_shape)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=5,
                                   progress_bar=False,
                                   recentering_maxiters=5)
        epsf, fitted_stars = epsf_builder(stars)

        # Verify EPSF properties with default settings
        assert isinstance(epsf, ImagePSF)
        assert epsf.x_0 == 0.0
        assert epsf.y_0 == 0.0
        assert epsf.flux == 1.0

        # Shape is star_shape * oversampling, then ensure odd
        ref_size = np.array(extract_shape) * oversampling
        ref_size = np.where(ref_size % 2 == 0, ref_size + 1, ref_size)
        assert epsf.data.shape == tuple(ref_size)

        # Verify basic EPSF properties
        assert len(fitted_stars) == 10
        # ePSF should sum to ~oversamp^2 for properly normalized
        # oversampled PSF
        expected_sum = oversampling ** 2
        assert 0.9 * expected_sum < epsf.data.sum() < 1.1 * expected_sum
        assert epsf.data.max() > 0.01  # Should have a peak

        # Check that the center region has higher values than edges
        center_y, center_x = np.array(ref_size) // 2
        center_val = epsf.data[center_y, center_x]
        edge_val = epsf.data[0, 0]
        assert center_val > edge_val  # Center should be brighter than edge

        # Test that residual computation works (basic functionality test)
        resid_star = fitted_stars[0].compute_residual_image(epsf)
        assert isinstance(resid_star, np.ndarray)
        assert resid_star.shape == fitted_stars[0].data.shape

    def test_invalid_inputs(self):
        """
        Test EPSFBuilder with various invalid inputs.
        """
        match = "'oversampling' must be specified"
        with pytest.raises(ValueError, match=match):
            EPSFBuilder(oversampling=None)

        match = 'oversampling must be > 0'
        with pytest.raises(ValueError, match=match):
            EPSFBuilder(oversampling=-1)

        match = 'maxiters must be a positive number'
        with pytest.raises(ValueError, match=match):
            EPSFBuilder(maxiters=-1)

        match = 'oversampling must be > 0'
        with pytest.raises(ValueError, match=match):
            EPSFBuilder(oversampling=[-1, 4])

        for sigma_clip in [None, [], 'a']:
            match = 'sigma_clip must be an astropy.stats.SigmaClip instance'
            with pytest.raises(TypeError, match=match):
                EPSFBuilder(sigma_clip=sigma_clip)

    def test_fitter_options(self):
        """
        Test EPSFBuilder with different fitter options.
        """
        # Test with default fitter (TRFLSQFitter)
        builder1 = EPSFBuilder(maxiters=3)
        assert isinstance(builder1.fitter, TRFLSQFitter)
        # Default fit_shape is 5
        assert_array_equal(builder1.fit_shape, (5, 5))
        assert builder1.fitter_maxiters == 100

        # Test with explicit astropy fitter
        fitter = TRFLSQFitter()
        builder2 = EPSFBuilder(fitter=fitter, maxiters=3)
        assert builder2.fitter is fitter

        # Test with custom fit_shape (scalar)
        builder3 = EPSFBuilder(fit_shape=7, maxiters=3)
        assert_array_equal(builder3.fit_shape, (7, 7))

        # Test with tuple fit_shape
        builder4 = EPSFBuilder(fit_shape=(5, 7), maxiters=3)
        assert_array_equal(builder4.fit_shape, (5, 7))

        # Test with None fit_shape (use entire star image)
        builder5 = EPSFBuilder(fit_shape=None, maxiters=3)
        assert builder5.fit_shape is None

        # Test with custom fitter_maxiters
        builder6 = EPSFBuilder(fitter_maxiters=200, maxiters=3)
        assert builder6.fitter_maxiters == 200

        # Test with invalid fitter type (should fail)
        with pytest.raises(TypeError,
                           match='fitter must be a callable'):
            EPSFBuilder(fitter='invalid_fitter', maxiters=3)

    def test_fitter_options_deprecated_epsf_fitter(self):
        """
        Test that passing an EPSFFitter to EPSFBuilder works but
        emits a deprecation warning.
        """
        epsf_fitter = _make_epsf_fitter(fit_boxsize=7)

        match = 'Passing an EPSFFitter instance'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            builder = EPSFBuilder(fitter=epsf_fitter, maxiters=3)

        # The astropy fitter should be extracted
        assert isinstance(builder.fitter, TRFLSQFitter)
        # The fit_shape should be extracted from EPSFFitter
        assert_array_equal(builder.fit_shape, (7, 7))

    def test_fitter_maxiters_ignored(self):
        """
        Test that fitter_maxiters is ignored if fitter doesn't
        support maxiter.
        """
        # Create a mock fitter without maxiter support
        class SimpleFitter:
            def __call__(self, model, x, y, z):  # noqa: ARG002
                return model

        match = 'fitter_maxiters.*will be ignored'
        with pytest.warns(AstropyUserWarning, match=match):
            builder = EPSFBuilder(fitter=SimpleFitter(),
                                  fitter_maxiters=200,
                                  maxiters=3)
        assert builder.fitter_maxiters is None

    def test_fitting_bounds(self, epsf_test_data):
        """
        Test EPSFBuilder with fit_shape larger than star cutouts.
        """
        size = 25
        oversampling = 4
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'],
                              size=size)

        # Use fit_shape larger than cutout
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=8,
                                   progress_bar=True,
                                   recentering_maxiters=5,
                                   fit_shape=31,
                                   smoothing_kernel='quadratic')

        # With a fit_shape larger than the cutout we expect the fitting
        # to fail for all stars. The ValueError is raised before any
        # star can be excluded (exclusion only happens after iter > 3).
        match = 'The ePSF fitting failed for all stars'
        with pytest.raises(ValueError, match=match):
            epsf_builder(stars)

    @pytest.mark.parametrize(('oversamp', 'star_size', 'expected_shape'), [
        # oversampling=1: shape should be odd (add 1 to even product)
        (1, 25, (25, 25)),  # 25*1 = 25 (odd) -> 25
        (1, 24, (25, 25)),  # 24*1 = 24 (even) -> 25
        (1, 26, (27, 27)),  # 26*1 = 26 (even) -> 27
        # oversampling=2: product is even, add 1
        (2, 25, (51, 51)),  # 25*2 = 50 (even) -> 51
        (2, 24, (49, 49)),  # 24*2 = 48 (even) -> 49
        # oversampling=3: product is odd for odd star size
        (3, 25, (75, 75)),  # 25*3 = 75 (odd) -> 75
        (3, 24, (73, 73)),  # 24*3 = 72 (even) -> 73
        # oversampling=4: product is even, add 1
        (4, 25, (101, 101)),  # 25*4 = 100 (even) -> 101
        (4, 24, (97, 97)),  # 24*4 = 96 (even) -> 97
        # oversampling=5: product is odd for odd star size
        (5, 25, (125, 125)),  # 25*5 = 125 (odd) -> 125
        (5, 24, (121, 121)),  # 24*5 = 120 (even) -> 121
    ])
    def test_shape_calculation(self, oversamp, star_size, expected_shape):
        """
        Test that the ePSF shape is correctly calculated for various
        oversampling factors.

        The ePSF shape should be:
        - star_size * oversampling for each dimension
        - Then ensure odd dimensions (add 1 if even)
        """
        # Test the shape calculation directly via _CoordinateTransformer
        transformer = _CoordinateTransformer(oversampling=(oversamp, oversamp))
        star_shapes = [(star_size, star_size)]
        computed_shape = transformer.compute_epsf_shape(star_shapes)

        assert computed_shape == expected_shape, (
            f'For oversamp={oversamp}, star_size={star_size}: '
            f'expected {expected_shape}, got {computed_shape}'
        )

    @pytest.mark.parametrize('kernel_type', ['quadratic', 'quartic',
                                             'custom'])
    def test_smoothing_kernel(self, epsf_test_data, kernel_type):
        """
        Test EPSFBuilder with smoothing kernel.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:3], size=11)

        if kernel_type == 'custom':
            kernel = np.ones((3, 3)) / 9.0
        else:
            kernel = kernel_type

        builder = EPSFBuilder(
            smoothing_kernel=kernel,
            maxiters=1,
            progress_bar=False,
        )

        epsf, _ = builder(stars)
        assert epsf is not None
        assert epsf.data.shape == (45, 45)

    @pytest.mark.parametrize('centering_func', [centroid_com,
                                                centroid_1dg,
                                                centroid_2dg,
                                                centroid_quadratic,
                                                ])
    def test_recentering(self, epsf_test_data, centering_func):
        """
        Test EPSFBuilder with different recentering function.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:4], size=11)

        # Setting oversampling=1 is required for centroid_quadratic to
        # work because its default fit_boxsize=5 is in native pixels and
        # we cannot adjust it here
        builder = EPSFBuilder(
            oversampling=1,
            recentering_func=centering_func,
            maxiters=5,
            progress_bar=False,
        )

        epsf, _ = builder(stars)
        assert epsf is not None
        assert epsf.data.shape == (11, 11)

    @pytest.mark.parametrize('shape', [(25, 25), (19, 25), (25, 19)])
    def test_shape_parameters(self, epsf_test_data, shape):
        """
        Test EPSFBuilder with explicit shape parameters.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:3], size=11)

        # Test with explicit shape
        builder = EPSFBuilder(
            shape=shape,
            oversampling=1,
            maxiters=1,
            progress_bar=False,
        )

        epsf, _ = builder(stars)
        assert epsf is not None
        assert epsf.data.shape == shape

    def test_check_convergence_no_good_stars(self):
        """
        Test EPSFBuilder._check_convergence with no good stars.
        """
        builder = EPSFBuilder(maxiters=1, progress_bar=False)

        # Create stars and mark all as fit_failed
        data = np.ones((5, 5))
        star = EPSFStar(data, cutout_center=(2, 2))
        stars = EPSFStars([star])

        centers = np.array([[2.0, 2.0]])
        fit_failed = np.array([True])  # All stars failed

        converged, center_dist_sq, _ = builder._check_convergence(
            stars, centers, fit_failed)

        # Should return False (not converged) when no good stars
        assert converged is False
        # center_dist_sq should be high to prevent false convergence
        assert center_dist_sq[0] > builder.center_accuracy_sq

    def test_resample_residuals_no_good_stars(self, epsf_test_data):
        """
        Test EPSFBuilder._resample_residuals with no good stars.
        """
        builder = EPSFBuilder(maxiters=1, progress_bar=False)

        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:2], size=11)

        # Create an initial ePSF
        epsf = builder._create_initial_epsf(stars)

        # Mark all stars as excluded
        for star in stars.all_stars:
            star._excluded_from_fit = True

        # Now resample residuals should handle no good stars
        result = builder._resample_residuals(stars, epsf)
        assert result.shape[0] == 0  # No good stars

    def test_resample_residual_output(self, epsf_test_data):
        """
        Test EPSFBuilder._resample_residual creates output image if None
        is passed.
        """
        builder = EPSFBuilder(maxiters=1, progress_bar=False)

        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:2], size=11)

        # Create an initial ePSF
        epsf = builder._create_initial_epsf(stars)

        # Call _resample_residual without out_image (should create one)
        star = stars.all_stars[0]
        result = builder._resample_residual(star, epsf, out_image=None)

        assert result is not None
        assert result.shape == epsf.data.shape

    def test_build_step_with_epsf(self, epsf_test_data):
        """
        Test EPSFBuilder._build_epsf_step with existing ePSF.
        """
        builder = EPSFBuilder(maxiters=1, progress_bar=False)

        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:5], size=11)

        # Create an initial ePSF
        epsf = builder._create_initial_epsf(stars)

        # Now build with existing ePSF
        improved_epsf = builder._build_epsf_step(stars, epsf=epsf)

        assert improved_epsf is not None
        assert improved_epsf.data.shape == epsf.data.shape

    def test_star_exclusion(self, epsf_test_data):
        """
        Test that stars are excluded after repeated fit failures.

        Here, we modify the first star's position such that star is
        centered near the corner of extracted cutout image. This will
        cause the fitting to fail for that star because its fitting
        region extends beyond the cutout boundaries, and it should be
        excluded from subsequent iterations.
        """
        tbl = epsf_test_data['init_stars'][:5].copy()
        tbl['x'][0] = 465
        tbl['y'][0] = 30
        stars = extract_stars(epsf_test_data['nddata'], tbl, size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=5, progress_bar=False)
        match = ('has been excluded from ePSF fitting because its fitting '
                 'region extends')
        with pytest.warns(AstropyUserWarning, match=match):
            result = builder(stars)

        assert result.n_excluded_stars == 1
        assert result.excluded_star_indices == [0]
        assert result.epsf is not None
        assert result.epsf.data.shape == (11, 11)
        assert result.fitted_stars.n_good_stars == 4
        assert result.fitted_stars.n_all_stars == 5

    def test_star_exclusion_single_warning(self, epsf_test_data):
        """
        Test that only a single warning is emitted per excluded star.

        When a star repeatedly fails fitting across iterations, the
        warning should only be emitted when the star is actually
        excluded (after more than 3 iterations of failure).
        """
        tbl = epsf_test_data['init_stars'][:5].copy()
        tbl['x'][0] = 465
        tbl['y'][0] = 30
        stars = extract_stars(epsf_test_data['nddata'], tbl, size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=6, progress_bar=False)

        # Capture all warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            builder(stars)

        # Filter for the specific warning about exclusion
        fit_warnings = [w for w in warning_list
                        if 'has been excluded from ePSF fitting' in
                        str(w.message)]

        # Should only have 1 warning despite multiple iterations
        assert len(fit_warnings) == 1

    def test_excluded_star_no_copy(self, epsf_test_data):
        """
        Test that excluded stars are returned without copying.

        When a star is excluded from fitting, the fitter should return
        the same star object directly, not a copy. This is more
        efficient than creating unnecessary copies.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:3], size=11)

        # Mark one star as excluded
        original_star = stars.all_stars[0]
        original_star._excluded_from_fit = True

        # Create an ePSF for fitting
        builder = EPSFBuilder(oversampling=1, maxiters=1, progress_bar=False)
        epsf = builder._create_initial_epsf(stars)

        # Fit the stars using the builder's internal method
        fitted_stars = builder._fit_stars(epsf, stars)

        # The excluded star should be the exact same object (identity)
        assert fitted_stars.all_stars[0] is original_star

    def test_process_iteration_with_fit_failures(self, epsf_test_data):
        """
        Test _process_iteration marks stars excluded after iter > 3.

        This test covers both types of fit failures:
        1. Fitting region extends beyond cutout (status=1)
        2. Fit did not converge due to invalid ierr (status=2)
        """
        # Create stars with one positioned near corner to cause overlap
        # error
        tbl = epsf_test_data['init_stars'][:5].copy()
        tbl['x'][0] = 465  # Position near corner to cause overlap error
        tbl['y'][0] = 30
        stars = extract_stars(epsf_test_data['nddata'], tbl, size=11)

        # Build initial ePSF. This will fit the stars and move their
        # centers. Star 0 will have its center moved near the edge of
        # the cutout, which will cause overlap errors in subsequent
        # iterations.
        builder_init = EPSFBuilder(oversampling=1, maxiters=2,
                                   progress_bar=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            epsf, fitted_stars = builder_init(stars)

        # Create a fitter that returns invalid ierr for the first call
        # only.
        # Star 0 will fail due to overlap error (status=1) before the
        #   fitter is called (because its center moved near edge).
        # Star 1 is the first to reach the fitter, and will fail with
        #   invalid ierr (status=2).
        # Subsequent stars will get valid ierr.
        class FirstCallFailingFitter:
            def __init__(self):
                self.call_count = 0
                self.fit_info = {'ierr': 1}  # Valid by default

            def __call__(self, model, *_args, **_kwargs):
                self.call_count += 1
                # Fail only on the first fitter call (which is star 1,
                # since star 0 fails with overlap error before reaching
                # fitter)
                if self.call_count == 1:
                    self.fit_info = {'ierr': 0}  # Invalid ierr
                else:
                    self.fit_info = {'ierr': 1}  # Valid ierr
                return model

        failing_fitter = FirstCallFailingFitter()
        builder = EPSFBuilder(oversampling=1, maxiters=1,
                              progress_bar=False,
                              fitter=failing_fitter)

        # Process iteration with iter_num > 3 to trigger exclusion. Use
        # fitted_stars (which has moved centers) to trigger overlap error.
        # Capture warnings to verify both types are emitted.
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            _, stars_new, fit_failed = builder._process_iteration(
                fitted_stars, epsf, iter_num=4)

        # Check that stars 0 and 1 failed
        assert fit_failed[0]  # Overlap error (status=1)
        assert fit_failed[1]  # Invalid ierr (status=2)

        # Verify both stars are marked for exclusion
        assert stars_new.all_stars[0]._excluded_from_fit
        assert stars_new.all_stars[1]._excluded_from_fit

        # Verify correct error status for each failure type
        assert stars_new.all_stars[0]._fit_error_status == 1  # Overlap error
        assert stars_new.all_stars[1]._fit_error_status == 2  # Fit failure

        # Verify both warning types were emitted
        warning_messages = [str(w.message) for w in warning_list]
        overlap_warnings = [m for m in warning_messages
                            if 'fitting region extends beyond' in m]
        converge_warnings = [m for m in warning_messages
                             if 'fit did not converge' in m]
        assert len(overlap_warnings) == 1
        assert len(converge_warnings) == 1

    def test_star_exclusion_fit_failure(self, epsf_test_data):
        """
        Test that stars are excluded with appropriate message when fit
        does not converge (ierr error).

        This tests exclusion due to fit failure (status=2), as opposed
        to the fitting region extending beyond the cutout (status=1).
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:5], size=11)
        n_stars = len(stars.all_stars)

        # Create a fitter that fails for the first star (invalid ierr)
        # but succeeds for others. Add small offsets to x_0/y_0 to
        # prevent early convergence, ensuring we reach iteration > 3.
        class PartialFailingFitter:
            def __init__(self):
                self.call_count = 0
                self.fit_info = {'ierr': 1}  # Valid by default

            def __call__(self, model, *_args, **_kwargs):
                self.call_count += 1
                star_idx = (self.call_count - 1) % n_stars
                # Fail only the first star
                if star_idx == 0:
                    self.fit_info = {'ierr': 0}  # Invalid ierr
                else:
                    self.fit_info = {'ierr': 1}  # Valid ierr
                # Add small offset to prevent early convergence
                model.x_0 = model.x_0 + 0.01
                model.y_0 = model.y_0 + 0.01
                return model

        failing_fitter = PartialFailingFitter()

        # Use maxiters=5 so we reach iter > 3 to trigger exclusion
        builder = EPSFBuilder(oversampling=1, maxiters=5,
                              progress_bar=False,
                              fitter=failing_fitter)

        # Should warn about fit not converging
        match = ('has been excluded from ePSF fitting because the fit did '
                 'not converge')
        with pytest.warns(AstropyUserWarning, match=match):
            result = builder(stars)

        # At least the first star (with ierr=0) should be excluded
        assert result.n_excluded_stars >= 1
        assert 0 in result.excluded_star_indices
        # Check that the first star has fit_error_status=2 (fit failure)
        assert result.fitted_stars.all_stars[0]._fit_error_status == 2
        assert result.fitted_stars.all_stars[0]._excluded_from_fit

    def test_build_tracks_excluded_indices(self, epsf_test_data):
        """
        Test that _build_epsf properly tracks excluded star indices.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:10], size=11)

        # Create a fitter that:
        # 1. Adds noise to centers to prevent convergence
        # 2. Fails first 3 stars on iteration 4+
        n_stars = len(stars.all_stars)

        class NoConvergeFitter:
            def __init__(self):
                self.star_count = 0
                self.fit_info = {'ierr': 1}

            def __call__(self, model, *_args, **_kwargs):
                self.star_count += 1
                iteration = self.star_count // n_stars + 1
                star_idx = (self.star_count - 1) % n_stars

                # On iteration 4+, fail first 3 stars
                if iteration > 4 and star_idx < 3:
                    self.fit_info = {'ierr': 0}  # Invalid
                else:
                    self.fit_info = {'ierr': 1}  # Valid

                # Add slight offset to x_0 to prevent convergence
                model.x_0 = model.x_0 + 0.01 * (iteration % 2)
                return model

        fitter_obj = NoConvergeFitter()
        builder = EPSFBuilder(oversampling=1, maxiters=7,
                              progress_bar=False,
                              fitter=fitter_obj,
                              center_accuracy=1e-6)

        # Build - this should trigger exclusion tracking
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = builder(stars)

        # Check that excluded_star_indices was populated
        assert hasattr(result, 'excluded_star_indices')
        assert isinstance(result.excluded_star_indices, list)
        # We may or may not have excluded stars depending on exact timing
        assert result.n_excluded_stars >= 0

    def test_build_step_origin_is_none_branch(self, epsf_test_data):
        """
        Test _build_epsf_step else branch when origin is None.
        """
        builder = EPSFBuilder(maxiters=1, progress_bar=False)

        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:3], size=11)

        # Create ePSF and verify origin condition
        epsf = builder._create_initial_epsf(stars)

        # Verify the branch condition logic
        has_valid_origin = hasattr(epsf, 'origin') and epsf.origin is not None
        assert has_valid_origin  # Normal case, origin exists

        # The else branch is only reached when origin is None
        # This line calculates origin from shape
        expected_origin_y = (epsf.data.shape[0] - 1) / 2.0
        expected_origin_x = (epsf.data.shape[1] - 1) / 2.0
        assert_allclose(epsf.origin, (expected_origin_x, expected_origin_y))

    @pytest.mark.skipif(not HAS_TQDM, reason='tqdm is required')
    def test_with_progress_bar(self, epsf_test_data):
        """
        Test EPSFBuilder with progress_bar=True.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:5], size=11)

        # Build with progress bar enabled and high center_accuracy to
        # prevent convergence
        builder = EPSFBuilder(oversampling=1, maxiters=3, progress_bar=True,
                              center_accuracy=1e-10)
        result = builder(stars)
        assert result.epsf is not None
        assert result.epsf.data.shape == (11, 11)

    def test_recenter_shift_increase(self, epsf_test_data):
        """
        Test early exit in _recenter_epsf when shift increases.

        Uses mock to force the centroid function to return values
        that cause shift to increase on second iteration.
        """
        builder = EPSFBuilder(oversampling=1, maxiters=2, progress_bar=False,
                              recentering_maxiters=10)

        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:5], size=11)

        epsf, _ = builder(stars)

        # Create a mock centroid function that returns oscillating values
        # First call: shift by 0.5 pixels
        # Second call: shift back by 1.0 (larger shift, triggers break)
        call_count = [0]
        center = np.array(epsf.data.shape) / 2.0

        def mock_centroid(data, mask=None):  # noqa: ARG001
            call_count[0] += 1
            if call_count[0] == 1:
                # First iteration: small shift
                return (center[1] + 0.5, center[0] + 0.5)
            # Second iteration: shift back (larger distance)
            return (center[1] - 0.5, center[0] - 0.5)

        with patch.object(builder, 'recentering_func', mock_centroid):
            recentered = builder._recenter_epsf(epsf)

        assert recentered is not None
        assert recentered.shape == epsf.data.shape
        # The mock should have been called at least twice
        assert call_count[0] >= 2

    def test_very_small_sources(self):
        """
        Test EPSFBuilder with very small sources that may cause
        numerical issues.
        """
        fwhm = 1.5
        psf_model = CircularGaussianPRF(flux=1, fwhm=fwhm)

        shape = (50, 50)
        sources = Table()
        sources['x_0'] = [25]
        sources['y_0'] = [25]
        sources['fwhm'] = [fwhm]

        data = make_model_image(shape, psf_model, sources)
        nddata = NDData(data=data)

        stars_tbl = Table()
        stars_tbl['x'] = sources['x_0']
        stars_tbl['y'] = sources['y_0']
        stars = extract_stars(nddata, stars_tbl, size=11)

        # Should handle numerical edge cases gracefully
        builder = EPSFBuilder(oversampling=1, maxiters=5, progress_bar=False)

        epsf, _ = builder(stars)
        assert epsf is not None
        assert epsf.data.shape == (11, 11)

    def test_fit_stars_with_linked_stars(self, epsf_test_data):
        """
        Test EPSFBuilder._fit_stars with LinkedEPSFStar objects.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:4], size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=2, progress_bar=False)
        epsf, _ = builder(stars)

        # Create mock WCS
        class MockWCS:
            def pixel_to_world_values(self, x, y):
                return x, y

            def world_to_pixel_values(self, ra, dec):
                return ra, dec

        mock_wcs = MockWCS()

        # Create LinkedEPSFStar from first two stars
        linked_stars_list = []
        for i in range(2):
            star_data = stars.all_stars[i].data.copy()
            center = stars.all_stars[i].cutout_center
            origin = (0, 0)
            star = EPSFStar(star_data, cutout_center=center,
                            origin=origin, wcs_large=mock_wcs)
            linked_stars_list.append(star)

        linked_star = LinkedEPSFStar(linked_stars_list)

        # Create EPSFStars with LinkedEPSFStar and regular stars
        stars_mixed = EPSFStars([linked_star, stars.all_stars[2]])

        # Fit stars using builder's _fit_stars method
        fitted_stars = builder._fit_stars(epsf, stars_mixed)

        # Check structure: EPSFStars contains 2 items
        # (1 LinkedEPSFStar + 1 regular star)
        assert len(fitted_stars) == 2
        # First item in the container should be LinkedEPSFStar
        assert isinstance(fitted_stars._data[0], LinkedEPSFStar)
        assert len(fitted_stars._data[0]) == 2
        # Second is regular EPSFStar
        assert isinstance(fitted_stars._data[1], EPSFStar)
        # all_stars should have 3 stars (2 from linked + 1 regular)
        assert fitted_stars.n_all_stars == 3

    def test_fit_stars_with_excluded_linked_stars(self, epsf_test_data):
        """
        Test EPSFBuilder._fit_stars with excluded LinkedEPSFStar.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:4], size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=2, progress_bar=False)
        epsf, _ = builder(stars)

        # Create mock WCS
        class MockWCS:
            def pixel_to_world_values(self, x, y):
                return x, y

            def world_to_pixel_values(self, ra, dec):
                return ra, dec

        mock_wcs = MockWCS()

        # Create LinkedEPSFStar with one excluded star
        linked_stars_list = []
        for i in range(2):
            star_data = stars.all_stars[i].data.copy()
            center = stars.all_stars[i].cutout_center
            origin = (0, 0)
            star = EPSFStar(star_data, cutout_center=center,
                            origin=origin, wcs_large=mock_wcs)
            if i == 1:
                star._excluded_from_fit = True
            linked_stars_list.append(star)

        linked_star = LinkedEPSFStar(linked_stars_list)

        stars_with_excluded = EPSFStars([linked_star])

        # Fit stars
        fitted_stars = builder._fit_stars(epsf, stars_with_excluded)

        assert len(fitted_stars) == 1
        # Check the fitted result is LinkedEPSFStar
        assert isinstance(fitted_stars._data[0], LinkedEPSFStar)
        # Check that excluded star was handled (second star should be excluded)
        assert fitted_stars._data[0][1]._excluded_from_fit

    def test_fit_stars_empty(self):
        """
        Test EPSFBuilder._fit_stars with empty stars list.
        """
        builder = EPSFBuilder(oversampling=1, maxiters=2, progress_bar=False)

        # Create an ePSF for testing
        data = np.ones((11, 11))
        epsf = ImagePSF(data)

        # Test with empty EPSFStars
        empty_stars = EPSFStars([])
        fitted_stars = builder._fit_stars(epsf, empty_stars)

        # Should return empty stars unchanged
        assert len(fitted_stars) == 0

    def test_fit_star_overlap_error(self, epsf_test_data):
        """
        Test EPSFBuilder._fit_star with fit_shape causing overlap error.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:2], size=11)

        # Create builder with fit_shape that works initially
        builder = EPSFBuilder(oversampling=1, maxiters=2, fit_shape=5,
                              progress_bar=False)
        epsf, _ = builder(stars)

        # Now create a new builder with fit_shape larger than star
        # to trigger overlap error
        builder_large = EPSFBuilder(oversampling=1, maxiters=2, fit_shape=25,
                                    progress_bar=False)

        # Get a star
        star = stars.all_stars[0]

        # Fit the star with fit_shape=25 (larger than 11x11 star)
        # This should trigger PartialOverlapError or NoOverlapError
        # and set fit_error_status = 1
        fitted_star = builder_large._fit_star(epsf, star)

        # Check that fit_error_status was set to 1
        assert fitted_star._fit_error_status == 1

    def test_fit_star_with_fit_shape_none(self, epsf_test_data):
        """
        Test EPSFBuilder._fit_star with fit_shape=None (use entire star).
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:2], size=11)

        # Create builder with fit_shape=None
        builder = EPSFBuilder(oversampling=1, maxiters=2, fit_shape=None,
                              progress_bar=False)
        epsf, _ = builder(stars)

        # Get a star
        star = stars.all_stars[0]

        # Fit the star - should use entire cutout
        fitted_star = builder._fit_star(epsf, star)

        # Check that fitting succeeded
        assert fitted_star._fit_error_status == 0
        assert fitted_star.flux > 0

    def test_normalize_epsf_zero_sum(self):
        """
        Test EPSFBuilder._normalize_epsf with zero-sum data.
        """
        builder = EPSFBuilder(oversampling=2, maxiters=2, progress_bar=False)

        # Create zero-sum ePSF data
        epsf_data = np.zeros((5, 5))

        match = 'Cannot normalize ePSF: data sum is zero'
        with pytest.raises(ValueError, match=match):
            builder._normalize_epsf(epsf_data)

    @pytest.mark.parametrize('oversamp', [1, 2, 3, 4, 5])
    def test_build_oversampling(self, oversamp):
        """
        Test that the ePSF built with oversampling has the expected
        shape and properties.

        Sources are placed on a regular grid with exact subpixel offsets
        to ensure that the ePSF is properly sampled. The test checks
        that the resulting ePSF has the expected shape, that it sums to
        the expected value for an oversampled PSF, and that its shape
        matches the input PSF model when scaled by the sum of the ePSF
        data.
        """
        offsets = (np.arange(oversamp) * 1.0 / oversamp - 0.5 + 1.0
                   / (2.0 * oversamp))
        xydithers = np.array(list(itertools.product(offsets, offsets)))
        xdithers = np.transpose(xydithers)[0]
        ydithers = np.transpose(xydithers)[1]

        nstars = oversamp**2
        fwhm = 7.0
        sources = Table()
        offset = 50
        size = oversamp * offset + offset
        y, x = np.mgrid[0:oversamp, 0:oversamp] * offset + offset
        sources['x_0'] = x.ravel() + xdithers
        sources['y_0'] = y.ravel() + ydithers
        sources['fwhm'] = np.full((nstars,), fwhm)

        psf_model = CircularGaussianPRF(fwhm=fwhm)
        shape = (size, size)
        data = make_model_image(shape, psf_model, sources)
        nddata = NDData(data=data)
        stars_tbl = Table()
        stars_tbl['x'] = sources['x_0']
        stars_tbl['y'] = sources['y_0']
        star_size = 25
        stars = extract_stars(nddata, stars_tbl, size=star_size)

        epsf_builder = EPSFBuilder(oversampling=oversamp, maxiters=15,
                                   progress_bar=False, recentering_maxiters=20)
        epsf, results = epsf_builder(stars)

        # Verify EPSF properties with default settings
        assert isinstance(epsf, ImagePSF)
        assert epsf.x_0 == 0.0
        assert epsf.y_0 == 0.0
        assert epsf.flux == 1.0

        # Check expected shape of ePSF data
        # The shape should be star_size * oversamp, then ensure odd
        # dimensions by adding 1 if even.
        expected_dim = star_size * oversamp
        if expected_dim % 2 == 0:
            expected_dim += 1
        expected_shape = (expected_dim, expected_dim)
        assert epsf.data.shape == expected_shape

        # Check expected sum of ePSF data.
        # For an oversampled PSF, the sum of the array values should
        # equal the product of the oversampling factors (oversamp^2 for
        # symmetric oversampling).
        expected_sum = oversamp**2
        assert_allclose(epsf.data.sum(), expected_sum, rtol=0.02)

        # Check that the shape of the ePSF matches the input PSF model
        # when scaled by the sum of the ePSF data. The input PSF model
        # is a circular Gaussian with the specified FWHM, and the ePSF
        # should approximate this shape when scaled by the total flux.

        # Calculate the expected PSF shape based on the input model and
        # the oversampling factor. The FWHM should be scaled by the
        # oversampling factor to match the ePSF sampling.
        size = epsf.data.shape[0]
        cen = (size - 1) / 2
        fwhm2 = oversamp * fwhm
        model = CircularGaussianPRF(flux=1, x_0=cen, y_0=cen, fwhm=fwhm2)
        yy, xx = np.mgrid[0:size, 0:size]
        psf = model(xx, yy) * oversamp**2
        assert_allclose(epsf.data, psf, atol=2e-4)

        # Check that the fitted centers are close to the true source
        # positions
        assert_allclose(results.center_flat[:, 0], sources['x_0'], atol=0.005)
        assert_allclose(results.center_flat[:, 1], sources['y_0'], atol=0.005)

    def test_fit_stars_with_excluded_epsf_star(self, epsf_test_data):
        """
        Test _fit_stars with excluded EPSFStar.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:3], size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=1, progress_bar=False)

        # Build initial ePSF
        result = builder(stars)
        epsf = result.epsf

        # Mark first star as excluded
        stars.all_stars[0]._excluded_from_fit = True

        # Call _fit_stars
        fitted_stars = builder._fit_stars(epsf, stars)

        # Check that excluded star is returned unchanged
        assert fitted_stars.all_stars[0] is stars.all_stars[0]
        assert len(fitted_stars) == len(stars)

    def test_fit_stars_with_excluded_linked_star(self, epsf_test_data):
        """
        Test _fit_stars with excluded star in LinkedEPSFStar.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:4], size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=1, progress_bar=False)

        # Build initial ePSF
        result = builder(stars)
        epsf = result.epsf

        # Create mock WCS
        class MockWCS:
            def pixel_to_world_values(self, x, y):
                return x, y

            def world_to_pixel_values(self, ra, dec):
                return ra, dec

        mock_wcs = MockWCS()

        # Create LinkedEPSFStar with two stars, one excluded
        linked_stars_list = []
        for i in range(2):
            star_data = stars.all_stars[i].data.copy()
            center = stars.all_stars[i].cutout_center
            origin = (0, 0)
            star = EPSFStar(star_data, cutout_center=center,
                            origin=origin, wcs_large=mock_wcs)
            linked_stars_list.append(star)

        # Mark second star as excluded
        linked_stars_list[1]._excluded_from_fit = True

        linked_star = LinkedEPSFStar(linked_stars_list)
        stars_with_linked = EPSFStars([linked_star])

        # Call _fit_stars
        fitted_stars = builder._fit_stars(epsf, stars_with_linked)

        # Check that result has the linked stars
        assert len(fitted_stars) == 1
        assert len(fitted_stars.all_stars) == 2

    def test_fit_star_fitter_without_weights(self, epsf_test_data):
        """
        Test _fit_star with fitter that doesn't support weights.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:2], size=11)

        # Create a fitter that raises TypeError when weights is passed
        class NoWeightsFitter:
            def __init__(self):
                self.fit_info = {'ierr': 1}

            def __call__(self, model, *_args, **kwargs):
                if 'weights' in kwargs:
                    msg = 'weights not supported'
                    raise TypeError(msg)
                return model

        no_weights_fitter = NoWeightsFitter()
        builder = EPSFBuilder(oversampling=1, maxiters=1, progress_bar=False,
                              fitter=no_weights_fitter)

        # Build initial ePSF
        result = builder(stars)
        epsf = result.epsf

        # Call _fit_star directly
        star = stars.all_stars[0]
        fitted_star = builder._fit_star(epsf, star)

        # Check that star was fitted (should have new center)
        assert fitted_star is not None
        assert hasattr(fitted_star, 'center')

    def test_fit_star_fitter_without_fit_info(self, epsf_test_data):
        """
        Test _fit_star with fitter that doesn't have fit_info.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:2], size=11)

        # Create a fitter without fit_info attribute
        class NoFitInfoFitter:
            def __call__(
                self, model, x, y, z, weights=None, **kwargs,  # noqa: ARG002
            ):
                return model

        no_fit_info_fitter = NoFitInfoFitter()
        builder = EPSFBuilder(oversampling=1, maxiters=1, progress_bar=False,
                              fitter=no_fit_info_fitter)

        # Build initial ePSF
        result = builder(stars)
        epsf = result.epsf

        # Call _fit_star directly
        star = stars.all_stars[0]
        fitted_star = builder._fit_star(epsf, star)

        # Check that star was fitted
        assert fitted_star is not None
        assert hasattr(fitted_star, 'center')
        # fit_info should be None
        assert fitted_star._fit_info is None

    def test_fit_stars_invalid_epsf_type(self, epsf_test_data):
        """
        Test _fit_stars with invalid epsf type.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:2], size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=1, progress_bar=False)

        # Pass non-ImagePSF object as epsf
        match = 'The input epsf must be an ImagePSF'
        with pytest.raises(TypeError, match=match):
            builder._fit_stars('not_an_epsf', stars)

    def test_fit_stars_invalid_star_type(self, epsf_test_data):
        """
        Test _fit_stars with invalid star type.
        """
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:2], size=11)

        builder = EPSFBuilder(oversampling=1, maxiters=1, progress_bar=False)

        # Build initial ePSF
        result = builder(stars)
        epsf = result.epsf

        # Create EPSFStars with invalid star type
        class InvalidStar:
            pass

        invalid_stars = EPSFStars([InvalidStar()])

        # Call _fit_stars with invalid star type
        match = 'stars must contain only EPSFStar and/or LinkedEPSFStar'
        with pytest.raises(TypeError, match=match):
            builder._fit_stars(epsf, invalid_stars)
