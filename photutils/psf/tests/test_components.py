# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _components module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import LevMarLSQFitter, TRFLSQFitter
from astropy.table import QTable, Table
from numpy.testing import assert_equal

from photutils.detection import DAOStarFinder
from photutils.psf import CircularGaussianPRF
from photutils.psf._components import (PSFDataProcessor, PSFFitter,
                                       PSFResultsAssembler)
from photutils.psf.photometry import _PSFParameterMapper


@pytest.fixture(name='basic_data')
def fixture_basic_data():
    """
    Create basic test data for component testing.
    """
    # Create simple test image
    shape = (25, 25)
    yy, xx = np.mgrid[:shape[0], :shape[1]]

    # Add a simple Gaussian source
    model = CircularGaussianPRF(flux=100, x_0=12, y_0=12, fwhm=2.0)
    data = model(xx, yy)
    rng = np.random.default_rng(0)
    data += rng.normal(0, 1, shape)

    error = np.ones_like(data)
    mask = np.zeros_like(data, dtype=bool)

    return data, error, mask


@pytest.fixture(name='psf_model')
def fixture_psf_model():
    """
    Create a basic PSF model for testing.
    """
    return CircularGaussianPRF(flux=1, fwhm=2.7)


@pytest.fixture(name='param_mapper')
def fixture_param_mapper(psf_model):
    """
    Create a parameter mapper for testing.
    """
    return _PSFParameterMapper(psf_model)


@pytest.fixture(name='init_params')
def fixture_init_params():
    """
    Create initial parameters table for testing.
    """
    return Table({
        'x_init': [12.0, 8.0, 16.0],
        'y_init': [12.0, 8.0, 16.0],
        'flux_init': [100.0, 50.0, 75.0],
    })


class TestPSFDataProcessor:
    """
    Test the PSFDataProcessor class.
    """

    def test_init(self, param_mapper):
        """
        Test PSFDataProcessor initialization.
        """
        fit_shape = (7, 7)
        processor = PSFDataProcessor(
            param_mapper=param_mapper,
            fit_shape=fit_shape,
            finder=None,
            aperture_radius=3.0,
            localbkg_estimator=None,
        )

        assert processor.param_mapper is param_mapper
        assert processor.fit_shape == fit_shape
        assert processor.finder is None
        assert processor.aperture_radius == 3.0
        assert processor.localbkg_estimator is None
        assert processor.data_unit is None
        assert processor.finder_results is None
        assert processor._cached_offsets is None
        assert processor._cache_key is None

    def test_validate_array_valid_input(self, param_mapper):
        """
        Test validate_array with valid inputs.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))

        # Test valid 2D array
        data = np.ones((10, 10))
        result = processor.validate_array(data, 'data')
        assert_equal(result, data)

        # Test None input
        result = processor.validate_array(None, 'mask')
        assert result is None

        # Test np.ma.nomask
        result = processor.validate_array(np.ma.nomask, 'mask')
        assert result is None

    def test_validate_array_invalid_input(self, param_mapper):
        """
        Test validate_array with invalid inputs.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))

        # Test 1D array
        with pytest.raises(ValueError, match='data must be a 2D array'):
            processor.validate_array(np.ones(10), 'data')

        # Test 3D array
        with pytest.raises(ValueError, match='error must be a 2D array'):
            processor.validate_array(np.ones((5, 5, 5)), 'error')

        # Test shape mismatch
        data = np.ones((10, 10))
        error = np.ones((5, 5))
        match_str = 'data and error must have the same shape'
        with pytest.raises(ValueError, match=match_str):
            processor.validate_array(error, 'error', data_shape=data.shape)

    def test_normalize_init_units_no_units(self, param_mapper):
        """
        Test normalize_init_units when neither have units.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))
        processor.data_unit = None

        init_params = Table({'flux_init': [100.0, 200.0]})
        result = processor.normalize_init_units(init_params, 'flux_init')

        assert result is init_params
        assert result['flux_init'].unit is None

    def test_normalize_init_units_both_have_units(self, param_mapper):
        """
        Test normalize_init_units when both have compatible units.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))
        processor.data_unit = u.count

        init_params = QTable({'flux_init': [100.0, 200.0] * u.count})
        result = processor.normalize_init_units(init_params, 'flux_init')

        assert result is init_params
        assert result['flux_init'].unit == u.count

    def test_normalize_init_units_incompatible_units(self, param_mapper):
        """
        Test normalize_init_units with incompatible units.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))
        processor.data_unit = u.count

        init_params = QTable({'flux_init': [100.0, 200.0] * u.meter})
        match_str = 'incompatible with the input data units'
        with pytest.raises(ValueError, match=match_str):
            processor.normalize_init_units(init_params, 'flux_init')

    def test_normalize_init_units_data_has_units_init_does_not(
            self, param_mapper):
        """
        Test normalize_init_units when data has units but init doesn't.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))
        processor.data_unit = u.count

        init_params = Table({'flux_init': [100.0, 200.0]})
        match_str = 'input data has units.*does not have units'
        with pytest.raises(ValueError, match=match_str):
            processor.normalize_init_units(init_params, 'flux_init')

    def test_normalize_init_units_init_has_units_data_does_not(
            self, param_mapper):
        """
        Test normalize_init_units when init has units but data doesn't.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))
        processor.data_unit = None

        init_params = QTable({'flux_init': [100.0, 200.0] * u.count})
        match_str = 'has units.*input data does not have units'
        with pytest.raises(ValueError, match=match_str):
            processor.normalize_init_units(init_params, 'flux_init')

    def test_validate_init_params_valid(self, param_mapper, init_params):
        """
        Test validate_init_params with valid input.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))
        processor.data_unit = None

        result = processor.validate_init_params(init_params)

        assert isinstance(result, Table)
        assert 'x_init' in result.colnames
        assert 'y_init' in result.colnames
        assert 'flux_init' in result.colnames

    def test_validate_init_params_none(self, param_mapper):
        """
        Test validate_init_params with None input.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))

        result = processor.validate_init_params(None)
        assert result is None

    def test_validate_init_params_invalid_type(self, param_mapper):
        """
        Test validate_init_params with invalid type.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))

        match_str = 'init_params must be an astropy Table'
        with pytest.raises(TypeError, match=match_str):
            processor.validate_init_params([1, 2, 3])

    def test_validate_init_params_missing_positions(self, param_mapper):
        """
        Test validate_init_params with missing position columns.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))

        # Missing x column
        init_params = Table({'y_init': [12.0], 'flux_init': [100.0]})
        match_str = 'must contain valid column names.*x and y'
        with pytest.raises(ValueError, match=match_str):
            processor.validate_init_params(init_params)

        # Missing y column
        init_params = Table({'x_init': [12.0], 'flux_init': [100.0]})
        with pytest.raises(ValueError, match=match_str):
            processor.validate_init_params(init_params)

    def test_validate_init_params_nonfinite_local_bkg(self, param_mapper):
        """
        Test validate_init_params with non-finite local_bkg values.
        """
        processor = PSFDataProcessor(param_mapper, (7, 7))
        processor.data_unit = None

        init_params = Table({
            'x_init': [12.0],
            'y_init': [12.0],
            'local_bkg': [np.nan],
        })

        match_str = 'local_bkg column contains non-finite values'
        with pytest.raises(ValueError, match=match_str):
            processor.validate_init_params(init_params)

    def test_get_aper_fluxes(self, param_mapper, basic_data, init_params):
        """
        Test get_aper_fluxes method.
        """
        data, _, mask = basic_data
        processor = PSFDataProcessor(param_mapper, (7, 7), aperture_radius=3.0)

        fluxes = processor.get_aper_fluxes(data, mask, init_params)

        assert isinstance(fluxes, np.ndarray)
        assert len(fluxes) == len(init_params)
        assert np.all(np.isfinite(fluxes))

    def test_find_sources_if_needed_with_init_params(self, param_mapper,
                                                     basic_data, init_params):
        """
        Test find_sources_if_needed when init_params is provided.
        """
        data, _, mask = basic_data
        processor = PSFDataProcessor(param_mapper, (7, 7))

        result = processor.find_sources_if_needed(data, mask, init_params)

        assert result is not None
        assert 'id' in result.colnames
        assert len(result) == len(init_params)

    def test_find_sources_if_needed_no_init_no_finder(self, param_mapper,
                                                      basic_data):
        """
        Test find_sources_if_needed when neither is provided.
        """
        data, _, mask = basic_data
        processor = PSFDataProcessor(param_mapper, (7, 7), finder=None)

        match_str = 'finder must be defined if init_params is not input'
        with pytest.raises(ValueError, match=match_str):
            processor.find_sources_if_needed(data, mask, None)

    def test_find_sources_if_needed_with_finder(self, param_mapper,
                                                basic_data):
        """
        Test find_sources_if_needed with a finder.
        """
        data, _, mask = basic_data
        finder = DAOStarFinder(threshold=5.0, fwhm=2.0)
        processor = PSFDataProcessor(param_mapper, (7, 7), finder=finder)

        result = processor.find_sources_if_needed(data, mask, None)

        # Should find at least one source in our test data
        assert result is not None
        assert len(result) >= 1


class TestPSFFitter:
    """
    Test the PSFFitter class.
    """

    def test_init(self, psf_model, param_mapper):
        """
        Test PSFFitter initialization.
        """
        fitter = PSFFitter(
            psf_model=psf_model,
            param_mapper=param_mapper,
            fitter=None,
            fitter_maxiters=100,
            xy_bounds=None,
            group_warning_threshold=25,
        )

        assert fitter.psf_model is psf_model
        assert fitter.param_mapper is param_mapper
        assert isinstance(fitter.fitter, TRFLSQFitter)  # Default fitter
        assert fitter.fitter_maxiters == 100
        assert fitter.xy_bounds is None
        assert fitter.group_warning_threshold == 25

    def test_init_custom_fitter(self, psf_model, param_mapper):
        """
        Test PSFFitter initialization with custom fitter.
        """
        custom_fitter = LevMarLSQFitter()
        fitter = PSFFitter(
            psf_model=psf_model,
            param_mapper=param_mapper,
            fitter=custom_fitter,
            xy_bounds=(1.0, 1.0),
        )

        assert fitter.fitter is custom_fitter
        assert fitter.xy_bounds == (1.0, 1.0)

    def test_make_psf_model_single_source(self, psf_model, param_mapper):
        """
        Test make_psf_model with a single source.
        """
        fitter = PSFFitter(psf_model, param_mapper)

        sources = Table({
            'id': [1],
            'x_init': [10.0],
            'y_init': [15.0],
            'flux_init': [100.0],
        })

        model = fitter.make_psf_model(sources)

        assert model.name == 1
        assert model.x_0.value == 10.0
        assert model.y_0.value == 15.0
        assert model.flux.value == 100.0

    def test_make_psf_model_multiple_sources(self, psf_model, param_mapper):
        """
        Test make_psf_model with multiple sources.
        """
        fitter = PSFFitter(psf_model, param_mapper)

        sources = Table({
            'id': [1, 2],
            'x_init': [10.0, 20.0],
            'y_init': [15.0, 25.0],
            'flux_init': [100.0, 200.0],
        })

        model = fitter.make_psf_model(sources)

        # Should be a flat model with parameters for each source
        assert hasattr(model, 'flux_0')
        assert hasattr(model, 'x_0_0')
        assert hasattr(model, 'y_0_0')
        assert hasattr(model, 'flux_1')
        assert hasattr(model, 'x_0_1')
        assert hasattr(model, 'y_0_1')

        # Check parameter values
        assert model.flux_0.value == 100.0
        assert model.x_0_0.value == 10.0
        assert model.y_0_0.value == 15.0
        assert model.flux_1.value == 200.0
        assert model.x_0_1.value == 20.0
        assert model.y_0_1.value == 25.0

    def test_make_psf_model_with_xy_bounds(self, psf_model, param_mapper):
        """
        Test make_psf_model with xy bounds.
        """
        fitter = PSFFitter(psf_model, param_mapper, xy_bounds=(2.0, 3.0))

        sources = Table({
            'id': [1],
            'x_init': [10.0],
            'y_init': [15.0],
            'flux_init': [100.0],
        })

        model = fitter.make_psf_model(sources)

        # Check bounds were set
        assert model.x_0.bounds == (8.0, 12.0)  # 10.0 ± 2.0
        assert model.y_0.bounds == (12.0, 18.0)  # 15.0 ± 3.0

    def test_make_psf_model_with_units(self, psf_model, param_mapper):
        """
        Test make_psf_model with quantity columns.
        """
        fitter = PSFFitter(psf_model, param_mapper)

        sources = QTable({
            'id': [1],
            'x_init': [10.0] * u.pixel,
            'y_init': [15.0] * u.pixel,
            'flux_init': [100.0] * u.count,
        })

        model = fitter.make_psf_model(sources)

        # Units should be stripped for fitting
        assert model.x_0.value == 10.0
        assert model.y_0.value == 15.0
        assert model.flux.value == 100.0


class TestPSFResultsAssembler:
    """
    Test the PSFResultsAssembler class.
    """

    def test_init(self, param_mapper):
        """
        Test PSFResultsAssembler initialization.
        """
        fit_shape = (7, 7)
        assembler = PSFResultsAssembler(
            param_mapper=param_mapper,
            fit_shape=fit_shape,
            xy_bounds=(1.0, 1.0),
        )

        assert assembler.param_mapper is param_mapper
        assert assembler.fit_shape == fit_shape
        assert assembler.xy_bounds == (1.0, 1.0)

    def test_get_fit_error_indices_all_converged(self, param_mapper):
        """
        Test get_fit_error_indices when all fits converged.
        """
        assembler = PSFResultsAssembler(param_mapper, (7, 7))

        # Good convergence status codes
        fit_info = [
            {'ierr': 1},  # Converged
            {'status': 2},  # Converged
            {'ierr': 3},  # Converged
        ]

        bad_indices = assembler.get_fit_error_indices(fit_info)

        assert len(bad_indices) == 0

    def test_get_fit_error_indices_some_failed(self, param_mapper):
        """
        Test get_fit_error_indices when some fits failed.
        """
        assembler = PSFResultsAssembler(param_mapper, (7, 7))

        fit_info = [
            {'ierr': 1},  # Converged
            {'ierr': 0},  # Failed
            {'status': 2},  # Converged
            {'status': 0},  # Failed
        ]

        bad_indices = assembler.get_fit_error_indices(fit_info)

        assert_equal(bad_indices, [1, 3])

    def test_param_errors_to_table_no_units(self, param_mapper):
        """
        Test param_errors_to_table without units.
        """
        assembler = PSFResultsAssembler(param_mapper, (7, 7))

        # Assuming 3 fitted parameters (x, y, flux)
        fit_param_errs = np.array([
            [0.1, 0.1, 1.0],  # errors for source 1
            [0.2, 0.2, 2.0],  # errors for source 2
        ])

        table = assembler.param_errors_to_table(fit_param_errs, data_unit=None)

        assert isinstance(table, QTable)
        assert 'x_err' in table.colnames
        assert 'y_err' in table.colnames
        assert 'flux_err' in table.colnames
        assert len(table) == 2

    def test_param_errors_to_table_with_units(self, param_mapper):
        """
        Test param_errors_to_table with units.
        """
        assembler = PSFResultsAssembler(param_mapper, (7, 7))

        fit_param_errs = np.array([
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 2.0],
        ])

        table = assembler.param_errors_to_table(fit_param_errs,
                                                data_unit=u.count)

        assert isinstance(table, QTable)
        assert table['flux_err'].unit == u.count
        # Position errors should not have units
        assert (not hasattr(table['x_err'], 'unit')
                or table['x_err'].unit is None)


class TestComponentIntegration:
    """
    Test integration between components.
    """

    def test_components_work_together(self, psf_model, basic_data,
                                      init_params):
        """
        Test that all components can work together in sequence.
        """
        data, _, mask = basic_data

        # Create components
        param_mapper = _PSFParameterMapper(psf_model)
        processor = PSFDataProcessor(param_mapper, (7, 7), aperture_radius=3.0)
        fitter = PSFFitter(psf_model, param_mapper)
        assembler = PSFResultsAssembler(param_mapper, (7, 7))

        # Process data
        processor.data_unit = None
        validated_params = processor.validate_init_params(init_params)
        sources = processor.find_sources_if_needed(data, mask,
                                                   validated_params)

        assert sources is not None
        assert len(sources) > 0

        # Create PSF model for fitting (test with first source)
        psf_model_fit = fitter.make_psf_model(sources[:1])

        assert psf_model_fit is not None
        assert hasattr(psf_model_fit, 'x_0')
        assert hasattr(psf_model_fit, 'y_0')

        # Test error handling for fit results
        fit_info = [{'ierr': 1}]  # Successful fit
        bad_indices = assembler.get_fit_error_indices(fit_info)
        assert len(bad_indices) == 0

    def test_components_preserve_units(self, psf_model):
        """
        Test that components properly handle astropy units.
        """
        # Create data with units
        data = np.ones((25, 25)) * u.count
        mask = np.zeros((25, 25), dtype=bool)

        init_params = QTable({
            'x_init': [12.0] * u.pixel,
            'y_init': [12.0] * u.pixel,
            'flux_init': [100.0] * u.count,
        })

        # Create components
        param_mapper = _PSFParameterMapper(psf_model)
        processor = PSFDataProcessor(param_mapper, (7, 7))

        # Set data unit and validate
        processor.data_unit = u.count
        validated_params = processor.validate_init_params(init_params)
        sources_with_id = processor.find_sources_if_needed(
            data, mask, validated_params)

        assert sources_with_id['flux_init'].unit == u.count

        # Test that PSF fitting strips units
        fitter = PSFFitter(psf_model, param_mapper)
        psf_model_fit = fitter.make_psf_model(sources_with_id)

        # Model parameters should be unitless for fitting
        assert not hasattr(psf_model_fit.x_0.value, 'unit')
        assert not hasattr(psf_model_fit.y_0.value, 'unit')
        assert not hasattr(psf_model_fit.flux.value, 'unit')
