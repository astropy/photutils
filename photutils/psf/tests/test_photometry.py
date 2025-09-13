# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photometry module.
"""

import tempfile

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import (LevMarLSQFitter, LMLSQFitter,
                                      SimplexLSQFitter, TRFLSQFitter)
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable, Table
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose, assert_equal

from photutils.background import LocalBackground, MMMBackground
from photutils.datasets import make_model_image, make_noise_image
from photutils.detection import DAOStarFinder
from photutils.psf import (CircularGaussianPRF, PSFPhotometry, SourceGrouper,
                           make_psf_model, make_psf_model_image)
from photutils.utils.exceptions import NoDetectionsWarning


@pytest.fixture(name='test_data')
def fixture_test_data():
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    model_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700),
                                             min_separation=10, seed=0)
    sigma = 0.9
    noise = make_noise_image(data.shape, mean=0, stddev=sigma, seed=0)
    data += noise
    error = np.full(data.shape, sigma)

    return data, error, true_params


def make_mock_finder(x_col, y_col, units=False):
    def finder(data, mask=None):  # noqa: ARG001
        source_table = Table()
        x_val = [25.1]
        y_val = [24.9]
        if units:
            x_val *= u.pixel
            y_val *= u.pixel
        source_table[x_col] = x_val
        source_table[y_col] = y_val
        return source_table
    return finder


def test_invalid_inputs():
    model = CircularGaussianPRF(fwhm=1.0)

    match = 'psf_model must be an Astropy Model subclass'
    with pytest.raises(TypeError, match=match):
        _ = PSFPhotometry(1, 3)

    match = 'psf_model must be two-dimensional'
    psf_model = Gaussian1D()
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(psf_model, 3)

    match = 'psf_model must be two-dimensional'
    psf_model = Gaussian1D()
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(psf_model, 3)

    match = 'Invalid PSF model - could not find PSF parameter names'
    psf_model = Gaussian2D()
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(psf_model, 3)

    match = 'fit_shape must have an odd value for both axes'
    for shape in ((0, 0), (4, 3)):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, shape)

    match = 'fit_shape must be >= 1'
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(model, (-1, 1))

    match = 'fit_shape must be a finite value'
    for shape in ((np.nan, 3), (5, np.inf)):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, shape)

    kwargs = {'finder': 1, 'fitter': 1}
    for key, val in kwargs.items():
        match = f"'{key}' must be a callable object"
        with pytest.raises(TypeError, match=match):
            _ = PSFPhotometry(model, 1, **{key: val})

    match = 'localbkg_estimator must be a LocalBackground instance'
    localbkg = MMMBackground()
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(model, 1, localbkg_estimator=localbkg)

    match = 'aperture_radius must be a strictly-positive scalar'
    for radius in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, 1, aperture_radius=radius)

    match = 'grouper must be a SourceGrouper instance'
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(model, (5, 5), grouper=1)

    match = 'data must be a 2D array'
    psfphot = PSFPhotometry(model, (3, 3))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(np.arange(3))

    match = 'data and error must have the same shape'
    data = np.ones((11, 11))
    error = np.ones((3, 3))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, error=error)

    match = 'data and mask must have the same shape'
    data = np.ones((11, 11))
    mask = np.ones((3, 3))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, mask=mask)

    match = 'init_params must be an astropy Table'
    data = np.ones((11, 11))
    with pytest.raises(TypeError, match=match):
        _ = psfphot(data, init_params=1)

    match = ('init_params must contain valid column names for the x and y '
             'source positions')
    tbl = Table()
    tbl['a'] = np.arange(3)
    data = np.ones((11, 11))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, init_params=tbl)

    # test no finder or init_params
    match = 'finder must be defined if init_params is not input'
    psfphot = PSFPhotometry(model, (3, 3), aperture_radius=5)
    data = np.ones((11, 11))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data)

    # data has unmasked non-finite value
    match = 'Input data contains unmasked non-finite values'
    psfphot2 = PSFPhotometry(model, (3, 3), aperture_radius=3)
    init_params = Table()
    init_params['x_init'] = [1, 2]
    init_params['y_init'] = [1, 2]
    data = np.ones((11, 11))
    data[5, 5] = np.nan
    with pytest.warns(AstropyUserWarning, match=match):
        _ = psfphot2(data, init_params=init_params)

    # mask is input, but data has unmasked non-finite value
    match = 'Input data contains unmasked non-finite values'
    data = np.ones((11, 11))
    data[5, 5] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[7, 7] = True
    with pytest.warns(AstropyUserWarning, match=match):
        _ = psfphot2(data, mask=mask, init_params=init_params)

    match = 'init_params local_bkg column contains non-finite values'
    tbl = Table()
    init_params['x_init'] = [1, 2]
    init_params['y_init'] = [1, 2]
    init_params['local_bkg'] = [0.1, np.inf]
    data = np.ones((11, 11))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, init_params=init_params)

    data = np.ones((11, 11))
    tbl = Table()
    tbl['x'] = [1, 2]
    tbl['y'] = [1, 2]

    tbl['group_id'] = [1.1, 2.0]
    match = 'group_id must be an integer array'
    with pytest.raises(TypeError, match=match):
        _ = psfphot(data, init_params=tbl)

    tbl['group_id'] = [1, np.nan]
    match = 'group_id must be finite'
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, init_params=tbl)

    tbl['group_id'] = [0, 1]
    match = 'group_id must contain only positive'
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, init_params=tbl)

    tbl['group_id'] = [-1, 1]
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, init_params=tbl)


def test_psf_photometry(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    resid_data = psfphot.make_residual_image(data, psf_shape=fit_shape)

    assert isinstance(psfphot.finder_results, QTable)
    assert isinstance(phot, QTable)
    assert isinstance(psfphot.results, QTable)
    assert len(phot) == len(sources)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape
    assert phot.colnames[:4] == ['id', 'group_id', 'group_size', 'local_bkg']
    # test that error columns are ordered correctly
    assert phot['x_err'].max() > 0.0062
    assert phot['y_err'].max() > 0.0065
    assert phot['flux_err'].max() > 2.5

    assert isinstance(psfphot.fit_info, list)

    # test that repeated calls reset the results
    phot = psfphot(data, error=error)
    assert len(psfphot.fit_info) == len(phot)

    # test units
    unit = u.Jy
    finderu = DAOStarFinder(6.0 * unit, 2.0)
    psfphotu = PSFPhotometry(psf_model, fit_shape, finder=finderu,
                             aperture_radius=4)
    photu = psfphotu(data * unit, error=error * unit)
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert photu[col].unit == unit
    resid_datau = psfphotu.make_residual_image(data << unit,
                                               psf_shape=fit_shape)
    assert resid_datau.unit == unit
    colnames = ('qfit', 'cfit', 'reduced_chi2')
    for col in colnames:
        assert not isinstance(photu[col], u.Quantity)

    match = 'The fit_params function is deprecated'
    with pytest.warns(AstropyDeprecationWarning, match=match):
        assert isinstance(psfphot.fit_params, Table)


@pytest.mark.parametrize('fit_fwhm', [False, True])
def test_psf_photometry_forced(test_data, fit_fwhm):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    if fit_fwhm:
        psf_model.fwhm.fixed = False
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    resid_data = psfphot.make_residual_image(data, psf_shape=fit_shape)

    assert isinstance(psfphot.finder_results, QTable)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape
    assert phot.colnames[:4] == ['id', 'group_id', 'group_size', 'local_bkg']
    assert_equal(phot['x_init'], phot['x_fit'])

    if fit_fwhm:
        col = 'fwhm'
        suffixes = ('_init', '_fit', '_err')
        colnames = [col + suffix for suffix in suffixes]
        for colname in colnames:
            assert colname in phot.colnames


def test_psf_photometry_nddata(test_data):
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)

    # test NDData input
    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot1 = psfphot(data, error=error)
    phot2 = psfphot(nddata)
    resid_data1 = psfphot.make_residual_image(data, psf_shape=fit_shape)
    resid_data2 = psfphot.make_residual_image(nddata, psf_shape=fit_shape)

    assert np.all(phot1 == phot2)
    assert isinstance(resid_data2, NDData)
    assert resid_data2.data.shape == data.shape
    assert_allclose(resid_data1, resid_data2.data)

    # test NDData input with units
    unit = u.Jy
    finderu = DAOStarFinder(6.0 * unit, 2.0)
    psfphotu = PSFPhotometry(psf_model, fit_shape, finder=finderu,
                             aperture_radius=4)
    photu = psfphotu(data * unit, error=error * unit)
    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty, unit=unit)
    photu = psfphotu(nddata)
    assert photu['flux_init'].unit == unit
    assert photu['flux_fit'].unit == unit
    assert photu['flux_err'].unit == unit
    resid_data3 = psfphotu.make_residual_image(nddata, psf_shape=fit_shape)
    assert resid_data3.unit == unit


def test_psf_photometry_finite_weights(test_data):
    data, _, _ = test_data
    error = np.zeros_like(data)

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    match = 'Error array contains non-positive or non-finite values'
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, error=error)


def test_model_residual_image(test_data):
    data, error, _ = test_data

    data = data + 10
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(16.0, 2.0)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4,
                            localbkg_estimator=localbkg_estimator)
    psfphot(data, error=error)

    psf_shape = (25, 25)
    model1 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=False)
    model2 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=True)
    resid1 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=False)
    resid2 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=True)

    x, y = 0, 100
    assert model1[y, x] < 0.1
    assert model2[y, x] > 9
    assert resid1[y, x] > 9
    assert resid2[y, x] < 0

    x, y = 0, 80
    assert model1[y, x] < 0.1
    assert model2[y, x] > 18
    assert resid1[y, x] > 9
    assert resid2[y, x] < -9


@pytest.mark.parametrize('fit_stddev', [False, True])
def test_psf_photometry_compound_psfmodel(test_data, fit_stddev):
    """
    Test compound models output from ``make_psf_model``.
    """
    data, error, sources = test_data
    x_stddev = y_stddev = 1.2
    psf_func = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=x_stddev,
                          y_stddev=y_stddev)
    psf_model = make_psf_model(psf_func, x_name='x_mean', y_name='y_mean')
    if fit_stddev:
        psf_model.x_stddev_2.fixed = False
        psf_model.y_stddev_2.fixed = False

    fit_shape = (5, 5)
    finder = DAOStarFinder(5.0, 3.0)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)

    if fit_stddev:
        cols = ('x_stddev_2', 'y_stddev_2')
        suffixes = ('_init', '_fit', '_err')
        colnames = [col + suffix for suffix in suffixes for col in cols]
        for colname in colnames:
            assert colname in phot.colnames

    # test model and residual images
    psf_shape = (9, 9)
    model1 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=False)
    resid1 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=False)
    model2 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=True)
    resid2 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=True)
    assert model1.shape == data.shape
    assert model2.shape == data.shape
    assert resid1.shape == data.shape
    assert resid2.shape == data.shape
    assert_equal(data - model1, resid1)
    assert_equal(data - model2, resid2)

    # test with init_params
    init_params = psfphot.results_to_init_params()
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)

    if fit_stddev:
        cols = ('x_stddev_2', 'y_stddev_2')
        suffixes = ('_init', '_fit', '_err')
        colnames = [col + suffix for suffix in suffixes for col in cols]
        for colname in colnames:
            assert colname in phot.colnames

    # test results when fit does not converge (fitter_maxiters=3)
    match = r'One or more fit\(s\) may not have converged.'
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4, fitter_maxiters=3)
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error)
    assert len(phot) == len(sources)


def test_psf_photometry_mask(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    # test np.ma.nomask
    phot = psfphot(data, error=error, mask=None)
    photm = psfphot(data, error=error, mask=np.ma.nomask)
    assert np.all(phot == photm)

    # masked near source at ~(63, 49)
    data_orig = data.copy()
    data = data.copy()
    data[55, 60:70] = np.nan

    match = 'Input data contains unmasked non-finite values'
    with pytest.warns(AstropyUserWarning, match=match):
        phot1 = psfphot(data, error=error, mask=None)
    assert len(phot1) == len(sources)

    mask = ~np.isfinite(data)
    phot2 = psfphot(data, error=error, mask=mask)
    assert np.all(phot1 == phot2)

    # unmasked NaN with mask not None
    match = 'Input data contains unmasked non-finite values'
    mask = ~np.isfinite(data)
    mask[55, 65] = False
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error, mask=mask)
    assert len(phot) == len(sources)

    # mask all True; finder returns no sources
    match = 'No sources were found'
    mask = np.ones(data.shape, dtype=bool)
    with pytest.warns(NoDetectionsWarning, match=match):
        psfphot(data, mask=mask)

    # completely masked source should return NaNs and not raise
    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    mask = np.ones(data.shape, dtype=bool)
    phot_masked = psfphot(data_orig, mask=mask, init_params=init_params)
    assert len(phot_masked) == 1

    colnames = ('x_fit', 'y_fit', 'flux_fit', 'x_err', 'y_err', 'flux_err',
                'qfit', 'cfit', 'reduced_chi2')
    for col in colnames:
        assert np.isnan(phot_masked[col][0])
    assert phot_masked['npixfit'][0] == 0
    assert phot_masked['group_size'][0] == 1
    # new flag 128 for fully masked
    assert (phot_masked['flags'][0] & 128) == 128

    # masked central pixel
    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    mask = np.zeros(data.shape, dtype=bool)
    mask[49, 63] = True
    phot = psfphot(data_orig, mask=mask, init_params=init_params)
    assert len(phot) == 1
    assert np.isnan(phot['cfit'][0])

    # this should not raise a warning because the non-finite pixel was
    # explicitly masked
    psfphot = PSFPhotometry(psf_model, (3, 3), aperture_radius=3)
    data = np.ones((11, 11))
    data[5, 5] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[5, 5] = True
    init_params = Table()
    init_params['x_init'] = [1, 2]
    init_params['y_init'] = [1, 2]
    psfphot(data, mask=mask, init_params=init_params)


def test_psf_photometry_init_params(test_data):
    data, error, _ = test_data
    data = data.copy()

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == 1

    match = 'aperture_radius must be defined if a flux column is not'
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=None)
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, error=error, init_params=init_params)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    init_params['flux'] = 650
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == 1

    init_params['group_id'] = 1
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == 1

    colnames = ('flux', 'local_bkg')
    for col in colnames:
        init_params2 = init_params.copy()
        init_params2.remove_column('flux')
        init_params2[col] = [650 * u.Jy]
        match = 'column has units, but the input data does not have units'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data, error=error, init_params=init_params2)

        init_params2[col] = [650 * u.Jy]
        match = 'column has units that are incompatible with the input data'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data << u.m, init_params=init_params2)

        init_params2[col] = [650]
        match = 'The input data has units, but the init_params'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data << u.Jy, init_params=init_params2)

    colnames = ('x_fit', 'y_fit', 'flux_fit', 'x_err', 'y_err', 'flux_err',
                'qfit', 'cfit', 'reduced_chi2')

    # no-overlap source should return NaNs and not raise; also test
    # too-few-pixels
    init_params = QTable()
    init_params['x'] = [-63]
    init_params['y'] = [-49]
    init_params['flux'] = [100]
    phot_no_overlap = psfphot(data, init_params=init_params)
    assert len(phot_no_overlap) == 1
    for col in colnames:
        assert np.isnan(phot_no_overlap[col][0])
    assert phot_no_overlap['npixfit'][0] == 0
    assert phot_no_overlap['group_size'][0] == 1
    # new flag 64 for no overlap
    assert (phot_no_overlap['flags'][0] & 64) == 64

    # too-few pixels (unmasking only 2 pixels < 3 free params) should
    # give NaNs
    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    mask = np.ones(data.shape, dtype=bool)
    mask[49, 63] = False
    mask[49, 64] = False
    phot_few = psfphot(data, error=error, mask=mask, init_params=init_params)
    assert len(phot_few) == 1
    for col in colnames:
        assert np.isnan(phot_few[col][0])
    assert phot_few['npixfit'][0] == 2
    assert phot_few['group_size'][0] == 1
    # new flag 256 for too few pixels
    assert (phot_few['flags'][0] & 256) == 256

    # check that the first matching column name is used
    init_params = QTable()
    x = 63
    y = 49
    flux = 680
    init_params['x'] = [x]
    init_params['y'] = [y]
    init_params['flux'] = [flux]
    init_params['x_cen'] = [x + 0.1]
    init_params['y_cen'] = [y + 0.1]
    init_params['flux0'] = [flux + 0.1]
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == 1
    assert phot['x_init'][0] == x
    assert phot['y_init'][0] == y
    assert phot['flux_init'][0] == flux


def test_psf_photometry_init_params_units(test_data):
    data, error, _ = test_data
    data2 = data.copy()
    error2 = error.copy()

    unit = u.Jy
    data2 <<= unit
    error2 <<= unit

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4)

    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    init_params['flux'] = [650 * unit]
    init_params['local_bkg'] = [0.001 * unit]
    phot = psfphot(data2, error=error2, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == 1

    for val in (True, False):
        im = psfphot.make_model_image(data2.shape, psf_shape=fit_shape,
                                      include_localbkg=val)
        assert isinstance(im, u.Quantity)
        assert im.unit == unit
        resid = psfphot.make_residual_image(data2, psf_shape=fit_shape,
                                            include_localbkg=val)
        assert isinstance(resid, u.Quantity)
        assert resid.unit == unit

    # test invalid units
    colnames = ('flux', 'local_bkg')
    for col in colnames:
        init_params2 = init_params.copy()
        init_params2.remove_column('flux')
        init_params2.remove_column('local_bkg')
        init_params2[col] = [650 * u.Jy]
        match = 'column has units, but the input data does not have units'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data, error=error, init_params=init_params2)

        init_params2[col] = [650 * u.Jy]
        match = 'column has units that are incompatible with the input data'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data << u.m, init_params=init_params2)

        init_params2[col] = [650]
        match = 'The input data has units, but the init_params'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data << u.Jy, init_params=init_params2)


def test_psf_photometry_init_params_columns(test_data):
    data, error, _ = test_data
    data = data.copy()

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder)

    xy_suffixes = ('_init', 'init', 'centroid', '_centroid', '_peak', '',
                   'cen', '_cen', 'pos', '_pos', '_0', '0')
    flux_cols = ['flux_init', 'flux_0', 'flux0', 'flux', 'source_sum',
                 'segment_flux', 'kron_flux']
    pad = len(xy_suffixes) - len(flux_cols)
    flux_cols += flux_cols[0:pad]  # pad to have same length as xy_suffixes

    xcols = ['x' + i for i in xy_suffixes]
    ycols = ['y' + i for i in xy_suffixes]

    phots = []
    for xcol, ycol, fluxcol in zip(xcols, ycols, flux_cols, strict=True):
        init_params = QTable()
        init_params[xcol] = [42]
        init_params[ycol] = [36]
        init_params[fluxcol] = [680]
        phot = psfphot(data, error=error, init_params=init_params)
        assert isinstance(phot, QTable)
        assert len(phot) == 1
        phots.append(phot)

    for phot in phots[1:]:
        assert_allclose(phot['x_fit'], phots[0]['x_fit'])
        assert_allclose(phot['y_fit'], phots[0]['y_fit'])
        assert_allclose(phot['flux_fit'], phots[0]['flux_fit'])


def test_grouper(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    grouper = SourceGrouper(min_separation=20)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            grouper=grouper, aperture_radius=4)
    phot = psfphot(data, error=error)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)
    assert_equal(phot['group_id'], (1, 2, 3, 4, 5, 5, 5, 6, 6, 7))
    assert_equal(phot['group_size'], (1, 1, 1, 1, 3, 3, 3, 2, 2, 1))


def test_grouper_init_params(test_data):
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    grouper = SourceGrouper(min_separation=20)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            grouper=grouper, aperture_radius=4)
    phot0 = psfphot(data, error=error)

    init_params = QTable()
    init_params['id'] = phot0['id']
    init_params['group_id'] = 1
    init_params['x'] = phot0['x_init']
    init_params['y'] = phot0['y_init']
    init_params['flux'] = phot0['flux_init']
    phot1 = psfphot(data, error=error, init_params=init_params)
    nsources = len(phot1)
    assert isinstance(phot1, QTable)
    assert_equal(phot1['group_id'], np.ones(nsources, dtype=int))
    assert_equal(phot1['group_size'], np.ones(nsources, dtype=int) * nsources)

    # test with grouper=None
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            grouper=None, aperture_radius=4)
    phot2 = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot2, QTable)
    assert_equal(phot1['group_id'], np.ones(nsources, dtype=int))
    assert_equal(phot1['group_size'], np.ones(nsources, dtype=int) * nsources)


def test_large_group_warning():
    psf_model = CircularGaussianPRF(flux=1, fwhm=2)
    grouper = SourceGrouper(min_separation=50)
    model_shape = (5, 5)
    fit_shape = (5, 5)
    n_sources = 50
    shape = (301, 301)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700),
                                             min_separation=10, seed=0)
    match = 'Some groups have more than'
    psfphot = PSFPhotometry(psf_model, fit_shape, grouper=grouper)
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot(data, init_params=true_params)


def test_local_bkg(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    grouper = SourceGrouper(min_separation=20)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    finder = DAOStarFinder(10.0, 2.0)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            grouper=grouper, aperture_radius=4,
                            localbkg_estimator=localbkg_estimator)
    phot = psfphot(data, error=error)
    assert np.count_nonzero(phot['local_bkg']) == len(sources)


def test_fixed_params(test_data):
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    psf_model.flux.fixed = True
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    match = r'`bounds` must contain 2 elements'
    with pytest.raises(ValueError, match=match):
        psfphot(data, error=error)


def test_fixed_params_units(test_data):
    data, error, _ = test_data
    unit = u.nJy

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.x_0.fixed = False
    psf_model.y_0.fixed = False
    psf_model.flux.fixed = True
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0 * unit, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    phot = psfphot(data << unit, error=error << unit)
    assert phot['local_bkg'].unit == unit
    assert phot['flux_init'].unit == unit
    assert phot['flux_fit'].unit == unit
    assert phot['flux_err'].unit == unit


def test_fit_warning(test_data):
    data, _, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.flux.fixed = False
    psf_model.fwhm.bounds = (None, None)
    fit_shape = (5, 5)
    fitter = LMLSQFitter()  # uses "status" instead of "ierr"
    finder = DAOStarFinder(6.0, 2.0)
    # set fitter_maxiters = 1 so that the fit error status is set
    psfphot = PSFPhotometry(psf_model, fit_shape, fitter=fitter,
                            fitter_maxiters=1, finder=finder,
                            aperture_radius=4)

    match = r'One or more fit\(s\) may not have converged.'
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data)

    # check that flag=8 is set for these sources
    assert_equal(phot['flags'][0] & 8, np.ones(len(phot)) * 8)


def test_fitter_no_maxiters_no_metrics(test_data):
    """
    Test with a fitter that does not have a maxiters parameter and does
    not produce a residual array.
    """
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.flux.fixed = False
    fit_shape = (5, 5)
    fitter = SimplexLSQFitter()  # does not produce residual array
    finder = DAOStarFinder(6.0, 2.0)
    match = '"maxiters" will be ignored because it is not accepted by'
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot = PSFPhotometry(psf_model, fit_shape, fitter=fitter,
                                finder=finder, aperture_radius=4)
    phot = psfphot(data, error=error)
    colnames = ('qfit', 'cfit', 'reduced_chi2')
    for col in colnames:
        assert np.all(np.isnan(phot[col]))


def test_xy_bounds(test_data):
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    init_params = QTable()
    init_params['x'] = [65]
    init_params['y'] = [51]
    xy_bounds = (1, 1)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4, xy_bounds=xy_bounds)
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == len(init_params)
    assert_allclose(phot['x_fit'], 64.0)  # at lower bound
    assert_allclose(phot['y_fit'], 50.0)  # at lower bound

    psfphot2 = PSFPhotometry(psf_model, fit_shape, finder=None,
                             aperture_radius=4, xy_bounds=1)
    phot2 = psfphot2(data, error=error, init_params=init_params)
    cols = ('x_fit', 'y_fit', 'flux_fit')
    for col in cols:
        assert np.all(phot[col] == phot2[col])

    xy_bounds = (None, 1)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4, xy_bounds=xy_bounds)
    phot = psfphot(data, error=error, init_params=init_params)
    assert phot['x_fit'] < 64.0
    assert_allclose(phot['y_fit'], 50.0)  # at lower bound

    xy_bounds = (1, None)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4, xy_bounds=xy_bounds,
                            fitter_maxiters=500)
    phot = psfphot(data, error=error, init_params=init_params)
    assert_allclose(phot['x_fit'], 64.0)  # at lower bound
    assert phot['y_fit'] < 50.0

    xy_bounds = (None, None)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4, xy_bounds=xy_bounds)
    init_params['x'] = [63]
    init_params['y'] = [49]
    phot = psfphot(data, error=error, init_params=init_params)
    assert phot['x_fit'] < 63.3
    assert phot['y_fit'] < 48.7
    assert phot['flags'] == 0

    # test invalid inputs
    match = 'xy_bounds must have 1 or 2 elements'
    with pytest.raises(ValueError, match=match):
        PSFPhotometry(psf_model, fit_shape, xy_bounds=(1, 2, 3))
    match = 'xy_bounds must be a 1D array'
    with pytest.raises(ValueError, match=match):
        PSFPhotometry(psf_model, fit_shape, xy_bounds=np.ones((1, 1)))
    match = 'xy_bounds must be strictly positive'
    with pytest.raises(ValueError, match=match):
        PSFPhotometry(psf_model, fit_shape, xy_bounds=(-1, 2))
    match = 'xy_bounds must be finite'
    with pytest.raises(ValueError, match=match):
        PSFPhotometry(psf_model, fit_shape, xy_bounds=(np.nan, 2))


def test_grouper_with_xy_bounds(test_data):
    """
    Test source grouping functionality with xy_bounds applied.
    """
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)

    init_params = QTable()
    init_params['x_init'] = [20.0, 22.0, 25.0]
    init_params['y_init'] = [20.0, 21.0, 23.0]
    init_params['flux_init'] = [1000.0, 800.0, 600.0]

    # Test with grouper and xy_bounds
    grouper = SourceGrouper(min_separation=5)
    xy_bounds = (1.0, 1.5)  # Different bounds for x and y

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            grouper=grouper, xy_bounds=xy_bounds,
                            aperture_radius=4)

    phot = psfphot(data, error=error, init_params=init_params)

    # verify sources were grouped
    assert len(phot) == len(init_params)
    assert len(np.unique(phot['group_id'])) < len(phot)

    # Verify that xy_bounds were applied during fitting
    # The fitted positions should be constrained
    for i, row in enumerate(phot):
        x_init = init_params['x_init'][i]
        y_init = init_params['y_init'][i]
        x_fit = row['x_fit']
        y_fit = row['y_fit']

        # Check that fitted positions are within bounds
        # (allowing some tolerance for fitting convergence)
        assert abs(x_fit - x_init) <= xy_bounds[0] + 0.1
        assert abs(y_fit - y_init) <= xy_bounds[1] + 0.1

    # Test that the flat model creation worked with xy_bounds
    flat_model = psfphot._psf_fitter.make_psf_model(init_params)
    if len(init_params) > 1:
        # For multiple sources, check flat model has correct structure
        assert hasattr(flat_model, 'flux_0')
        assert hasattr(flat_model, 'x_0_0')
        assert hasattr(flat_model, 'y_0_0')
        if len(init_params) > 1:
            assert hasattr(flat_model, 'flux_1')
            assert hasattr(flat_model, 'x_0_1')
            assert hasattr(flat_model, 'y_0_1')


def test_negative_xy():
    sources = Table()
    sources['id'] = np.arange(3) + 1
    sources['flux'] = 1
    sources['x_0'] = [-1.4, 15.2, -0.7]
    sources['y_0'] = [-0.3, -0.4, 18.7]
    sources['sigma'] = 3.1
    shape = (31, 31)
    psf_model = CircularGaussianPRF(flux=1, fwhm=3.1)
    data = make_model_image(shape, psf_model, sources)
    fit_shape = (11, 11)
    psfphot = PSFPhotometry(psf_model, fit_shape, aperture_radius=10)
    phot = psfphot(data, init_params=sources)
    assert_equal(phot['x_init'], sources['x_0'])
    assert_equal(phot['y_init'], sources['y_0'])


def test_init_params_xy_with_units():
    """
    Test that init_params table x/y columns with units are accepted.
    """
    shape = (41, 41)
    psf_model = CircularGaussianPRF(flux=500, fwhm=3.0)
    data = np.zeros(shape)
    init_params = QTable()
    init_params['x'] = [20.0] * u.pixel  # units should be stripped
    init_params['y'] = [20.0] * u.pixel
    init_params['flux'] = [500]
    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape, aperture_radius=None)
    phot = psfphot(data, init_params=init_params)
    assert len(phot) == 1
    assert_equal(phot['x_init'][0], 20.0)
    assert_equal(phot['y_init'][0], 20.0)

    finder = make_mock_finder('x', 'y', units=True)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, init_params=None)
    assert len(phot) == 1
    assert_equal(phot['x_init'][0], 25.1)
    assert_equal(phot['y_init'][0], 24.9)


def test_out_of_bounds_centroids():
    sources = Table()
    sources['id'] = np.arange(8) + 1
    sources['flux'] = 1
    sources['x_0'] = [-1.4, 34.5, 14.2, -0.7, 34.5, 14.2, 51.3, 52.0]
    sources['y_0'] = [13, -0.2, -1.6, 40, 51.1, 50.9, 12.2, 42.3]
    sources['sigma'] = 3.1

    shape = (51, 51)
    psf_model = CircularGaussianPRF(flux=1, fwhm=3.1)
    data = make_model_image(shape, psf_model, sources)
    fit_shape = (11, 11)
    psfphot = PSFPhotometry(psf_model, fit_shape, aperture_radius=10)

    phot = psfphot(data, init_params=sources)

    # at least one of the best-fit centroids should be
    # out of the bounds of the dataset, producing a
    # masked value in the `cfit` column:
    assert np.any(np.isnan(phot['cfit']))


def test_make_psf_model():
    normalize = False
    sigma = 3.0
    amplitude = 1.0 / (2 * np.pi * sigma**2)
    xcen = ycen = 0.0
    psf0 = Gaussian2D(amplitude, xcen, ycen, sigma, sigma)
    psf1 = make_psf_model(psf0, x_name='x_mean', y_name='y_mean',
                          normalize=normalize)
    psf2 = make_psf_model(psf0, normalize=normalize)
    psf3 = make_psf_model(psf0, x_name='x_mean', normalize=normalize)
    psf4 = make_psf_model(psf0, y_name='y_mean', normalize=normalize)

    yy, xx = np.mgrid[0:101, 0:101]
    psf = psf1.copy()
    xval = 48
    yval = 52
    flux = 14.51
    psf.x_mean_2 = xval
    psf.y_mean_2 = yval
    data = psf(xx, yy) * flux

    fit_shape = 7
    init_params = Table([[46.1], [57.3], [7.1]],
                        names=['x_0', 'y_0', 'flux_0'])
    phot1 = PSFPhotometry(psf1, fit_shape, aperture_radius=None)
    tbl1 = phot1(data, init_params=init_params)

    phot2 = PSFPhotometry(psf2, fit_shape, aperture_radius=None)
    tbl2 = phot2(data, init_params=init_params)

    phot3 = PSFPhotometry(psf3, fit_shape, aperture_radius=None)
    tbl3 = phot3(data, init_params=init_params)

    phot4 = PSFPhotometry(psf4, fit_shape, aperture_radius=None)
    tbl4 = phot4(data, init_params=init_params)

    assert_allclose((tbl1['x_fit'][0], tbl1['y_fit'][0],
                     tbl1['flux_fit'][0]), (xval, yval, flux))
    assert_allclose((tbl2['x_fit'][0], tbl2['y_fit'][0],
                     tbl2['flux_fit'][0]), (xval, yval, flux))
    assert_allclose((tbl3['x_fit'][0], tbl3['y_fit'][0],
                     tbl3['flux_fit'][0]), (xval, yval, flux))
    assert_allclose((tbl4['x_fit'][0], tbl4['y_fit'][0],
                     tbl4['flux_fit'][0]), (xval, yval, flux))


FINDER_COLUMN_NAMES = [
    ('x', 'y'),
    ('x_init', 'y_init'),
    ('xcentroid', 'ycentroid'),
    ('x_centroid', 'y_centroid'),
    ('xpos', 'ypos'),
    ('x_peak', 'y_peak'),
    ('xcen', 'ycen'),
    ('x_fit', 'y_fit'),
    ('x_invalid', 'y_invalid'),
]


@pytest.mark.parametrize(('x_col', 'y_col'), FINDER_COLUMN_NAMES)
def test_finder_column_names(x_col, y_col):
    """
    Test that PSFPhotometry works correctly with a finder that returns
    different column names for source positions.
    """
    finder = make_mock_finder(x_col, y_col)

    sources = Table()
    sources['id'] = [1]
    sources['flux'] = 7.0
    sources['x_0'] = 25.0
    sources['y_0'] = 25.0
    shape = (31, 31)
    psf_model = CircularGaussianPRF(flux=1.0, fwhm=3.1)
    data = make_model_image(shape, psf_model, sources)

    fit_shape = (9, 9)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=10)

    # invalid column names should raise an error
    if x_col == 'x_invalid' or y_col == 'y_invalid':
        match = 'must contain columns for x and y coordinates'
        with pytest.raises(ValueError, match=match):
            psfphot(data)
        return

    phot_tbl = psfphot(data)

    assert len(phot_tbl) == 1
    assert_allclose(phot_tbl['x_init'][0], 25.1)
    assert_allclose(phot_tbl['y_init'][0], 24.9)
    assert_allclose(phot_tbl['x_fit'][0], 25.0, atol=1e-6)
    assert_allclose(phot_tbl['y_fit'][0], 25.0, atol=1e-6)
    assert_allclose(phot_tbl['flux_fit'][0], 7.0, rtol=1e-6)


def test_repr():
    psf_model = CircularGaussianPRF(flux=1.0, fwhm=3.1)
    fit_shape = (9, 9)
    finder = DAOStarFinder(6.0, 2.0)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=10)
    cls_repr = repr(psfphot)
    assert cls_repr.startswith(f'{psfphot.__class__.__name__}(')


def test_group_warning_threshold(test_data):
    data, error, sources = test_data
    sources['group_id'] = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4, group_warning_threshold=6)
    match = 'Some groups have more than 6 sources'
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error, init_params=sources)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4, group_warning_threshold=7)
    phot = psfphot(data, error=error, init_params=sources)
    assert len(phot) == 10


def test_flag2_boundaries():
    shape = (35, 21)
    psf_model = CircularGaussianPRF(fwhm=3.0)
    init_params = QTable()
    init_params['x_0'] = [-0.4, 20.4, -1.0, 21.0, 5.0, 5.0, 15.0, 15.0]
    init_params['y_0'] = [10.0, 10.0, 20.0, 20.0, -0.4, 34.4, -1.0, 35.0]
    init_params['flux'] = 500
    data = make_model_image(shape, psf_model, init_params)

    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape)
    phot = psfphot(data, init_params=init_params)
    assert len(phot) == 8
    assert_equal(phot['flags'][[2, 3, 6, 7]], [3, 3, 3, 3])
    assert_equal(phot['flags'][[0, 1, 4, 5]], [1, 1, 1, 1])


def test_flag64_no_overlap():
    """
    Test flag=64 for source with no overlap (completely outside).
    """
    shape = (21, 21)
    psf_model = CircularGaussianPRF(fwhm=3.0)
    data = np.zeros(shape)
    init_params = QTable()
    # place source completely outside (beyond + side)
    init_params['x_0'] = [100.0]
    init_params['y_0'] = [100.0]
    init_params['flux'] = [500.0]
    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape)
    phot = psfphot(data, init_params=init_params)
    assert len(phot) == 1
    # Expect bits include 64 (no overlap). Others (2,1,16,32) may also
    # be present. Only assert 64 is set.
    assert (phot['flags'][0] & 64) == 64
    assert phot['npixfit'][0] == 0


def test_flag128_fully_masked():
    """
    Test flag=128 for fully masked source region.
    """
    shape = (25, 25)
    psf_model = CircularGaussianPRF(fwhm=3.0)
    data = np.zeros(shape)
    init_params = QTable()
    init_params['x_0'] = [12.0]
    init_params['y_0'] = [12.0]
    init_params['flux'] = [500.0]
    mask = np.ones(shape, dtype=bool)  # fully masked image
    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape)
    phot = psfphot(data, init_params=init_params, mask=mask)
    assert len(phot) == 1
    assert phot['npixfit'][0] == 0
    assert (phot['flags'][0] & 128) == 128


def test_flag256_too_few_pixels():
    """
    Test flag=256 for too few unmasked pixels to perform a fit.
    """
    shape = (25, 25)
    psf_model = CircularGaussianPRF(fwhm=3.0)
    data = np.zeros(shape)
    init_params = QTable()
    init_params['x_0'] = [12.0]
    init_params['y_0'] = [12.0]
    init_params['flux'] = [500.0]
    mask = np.ones(shape, dtype=bool)
    # Unmask only a single pixel inside the fit box (fewer than params)
    mask[12, 12] = False
    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape)
    phot = psfphot(data, init_params=init_params, mask=mask)
    assert len(phot) == 1
    assert phot['npixfit'][0] == 1
    # Ensure 256 bit set (too few pixels); not fully masked (no 128).
    assert (phot['flags'][0] & 256) == 256


def test_flag16_missing_covariance():
    """
    Test flag=16 when fitter does not provide a covariance matrix.
    """
    shape = (21, 21)
    psf_model = CircularGaussianPRF(fwhm=2.5)
    data = np.zeros(shape)
    init_params = QTable()
    init_params['x_0'] = [10.0, 20.0]
    init_params['y_0'] = [10.0, 20.0]
    init_params['flux'] = [500.0, 500.0]
    init_params['group_id'] = [1, 1]

    # mock fitter that does not return a covariance matrix
    def mock_fitter(model, *args, **kwargs):  # noqa: ARG001
        mock_fitter.fit_info = {'status': 1}
        return model

    mock_fitter.fit_info = {}
    fit_shape = (5, 5)

    match = r'"maxiters" will be ignored because it is not accepted'
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot = PSFPhotometry(psf_model, fit_shape, fitter=mock_fitter)

    phot = psfphot(data, init_params=init_params)
    assert len(phot) == 2
    cols = ('x_err', 'y_err', 'flux_err')
    for col in cols:
        assert col in phot.colnames
        assert np.all(np.isnan(phot[col]))
    assert (phot['flags'][0] & 16) == 16


def test_flag32_parameter_at_bounds():
    """
    Test flag=32 when fitted x/y are exactly at imposed bounds.
    """
    shape = (21, 21)
    psf_model = CircularGaussianPRF(fwhm=2.5)
    data = np.zeros(shape)
    data[11, 11] = 1000.0
    init_params = QTable()
    init_params['x_0'] = [10.0]
    init_params['y_0'] = [10.0]
    init_params['flux'] = [500.0]

    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape, xy_bounds=0.1)
    phot = psfphot(data, init_params=init_params)
    assert len(phot) == 1
    assert (phot['flags'][0] & 32) == 32


def test_psf_photometry_methods(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    match = 'The fit_params function is deprecated'
    with pytest.warns(AstropyDeprecationWarning, match=match):
        assert psfphot.fit_params is None

    match = 'No results available. Please run the PSFPhotometry'
    with pytest.raises(ValueError, match=match):
        psfphot.make_model_image(data.shape)
    with pytest.raises(ValueError, match=match):
        psfphot.make_residual_image(data.shape)

    assert psfphot.results_to_init_params() is None
    assert psfphot.results_to_model_params() is None

    phot = psfphot(data, error=error)
    assert isinstance(phot, QTable)

    resid_data = psfphot.make_residual_image(data, psf_shape=fit_shape)
    assert isinstance(resid_data, np.ndarray)

    assert isinstance(psfphot.fit_info, list)

    match = 'The fit_params function is deprecated'
    with pytest.warns(AstropyDeprecationWarning, match=match):
        assert isinstance(psfphot.fit_params, Table)


@pytest.mark.parametrize('units', [False, True])
def test_invalid_sources(test_data, units):
    data, error, sources = test_data
    if units:
        unit = u.nJy
        data = data << unit
        error = error << unit
    init_params = sources.copy()

    # one item in group is invalid
    init_params['x_0'][0] = 1000
    init_params['y_0'][0] = 1000
    init_params['x_0'][5] = 1000

    # entire group is invalid
    init_params['x_0'][-2] = 1000
    init_params['x_0'][-1] = 1000

    if units:
        init_params['flux'] *= unit

    init_params['group_id'] = [1, 2, 1, 2, 2, 3, 2, 3, 4, 4]
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error, init_params=init_params)

    assert len(phot) == len(init_params)
    assert_equal(phot['group_id'], init_params['group_id'])
    assert_equal(phot['group_size'], [2, 4, 2, 4, 4, 2, 4, 2, 2, 2])

    cols = ('x_fit', 'y_fit', 'flux_fit', 'x_err', 'y_err', 'flux_err',
            'qfit', 'cfit', 'reduced_chi2')
    for col in cols:
        assert np.all(np.isnan(phot[col][[0, 5, -2, -1]]))

    resid = psfphot.make_residual_image(data, psf_shape=fit_shape)
    assert isinstance(resid, np.ndarray)
    assert resid.shape == data.shape
    if units:
        assert isinstance(resid, u.Quantity)
        assert resid.unit == unit

    init_params = psfphot.results_to_init_params()
    assert isinstance(init_params, QTable)
    assert len(init_params) == 6  # 6 valid sources
    assert_equal(init_params['id'], np.arange(1, 7))

    model_params = psfphot.results_to_model_params()
    assert isinstance(model_params, QTable)
    assert len(model_params) == 6
    assert_equal(model_params['id'], np.arange(1, 7))


def test_psf_photometry_table_serialization(test_data):
    """
    Test that photometry results table can be written to file.
    """
    data, error, _ = test_data

    # Create PSFPhotometry with various components to test metadata
    # serialization
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    grouper = SourceGrouper(min_separation=2.0)
    localbkg_estimator = LocalBackground(5, 10)
    fitter = TRFLSQFitter()

    psfphot = PSFPhotometry(
        psf_model, fit_shape,
        finder=finder,
        grouper=grouper,
        localbkg_estimator=localbkg_estimator,
        fitter=fitter,
        aperture_radius=4,
    )

    # Perform photometry
    results = psfphot(data, error=error)

    # Test that we have results
    assert isinstance(results, QTable)
    assert len(results) > 0

    # Test that metadata contains repr strings for class objects
    meta = results.meta
    assert 'psf_model' in meta
    assert 'finder' in meta
    assert 'grouper' in meta
    assert 'localbkg_estimator' in meta
    assert 'fitter' in meta

    # Verify these are string representations, not objects
    assert isinstance(meta['psf_model'], str)
    assert isinstance(meta['finder'], str)
    assert isinstance(meta['grouper'], str)
    assert isinstance(meta['localbkg_estimator'], str)
    assert isinstance(meta['fitter'], str)

    # Verify the repr strings contain expected content
    assert 'CircularGaussianPRF' in meta['psf_model']
    assert 'DAOStarFinder' in meta['finder']
    assert 'SourceGrouper' in meta['grouper']
    assert 'LocalBackground' in meta['localbkg_estimator']
    assert 'TRFLSQFitter' in meta['fitter']

    # Test file writing - this should not raise any errors
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ecsv',
                                     delete=False) as tmp:
        # Write table to ECSV format
        results.write(tmp.name, format='ascii.ecsv', overwrite=True)

        # Read it back to verify it's readable
        read_table = Table.read(tmp.name, format='ascii.ecsv')

        # Basic checks that the table was written and read correctly
        assert len(read_table) == len(results)
        assert set(read_table.colnames) == set(results.colnames)

        # Check that metadata was preserved
        read_meta = read_table.meta
        assert 'psf_model' in read_meta
        assert 'finder' in read_meta
        assert isinstance(read_meta['psf_model'], str)
        assert isinstance(read_meta['finder'], str)


def test_psf_photometry_invalid_coordinates():
    """
    Test PSF photometry with invalid coordinates.
    """
    yy, xx = np.mgrid[:50, :50]

    psf_model = CircularGaussianPRF(x_0=25, y_0=25, flux=120, fwhm=2.7)
    data = psf_model(xx, yy)
    psfphot = PSFPhotometry(psf_model, (5, 5), aperture_radius=3)

    n_sources = 3
    init_params = Table()
    init_params['id'] = np.arange(1, n_sources + 1)
    init_params['x_init'] = [25.0, -5.0, 55.0]
    init_params['y_init'] = [25.0, 25.0, 25.0]
    init_params['flux_init'] = 100.0
    init_params['group_id'] = 1

    results = psfphot(data, init_params=init_params)
    assert len(results) == n_sources
    assert_equal(results['group_size'], [3, 3, 3])
    cols = ('x_fit', 'y_fit', 'flux_fit', 'x_err', 'y_err', 'flux_err',
            'qfit', 'cfit', 'reduced_chi2')
    for col in cols:
        assert np.all(np.isnan(results[col][1:]))


def test_should_skip_source_coverage():
    """
    Test the _should_skip_source method directly to ensure coverage of
    specific boundary conditions.
    """
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psfphot = PSFPhotometry(psf_model, (5, 5), aperture_radius=3)

    data_shape = (50, 50)

    # Test outside bounds - clearly beyond fit region
    row_data = {
        psfphot._param_mapper.init_colnames['x']: -5.0,
        psfphot._param_mapper.init_colnames['y']: 25.0,
    }
    row = Table([row_data])[0]  # Create a table row
    should_skip, reason = psfphot._should_skip_source(row, data_shape)
    assert should_skip is True
    assert reason == 'no_overlap'

    # Test outside bounds - coordinates well beyond data dimensions
    row_data = {
        psfphot._param_mapper.init_colnames['x']: 25.0,
        psfphot._param_mapper.init_colnames['y']: 60.0,
    }
    row = Table([row_data])[0]
    should_skip, reason = psfphot._should_skip_source(row, data_shape)
    assert should_skip is True
    assert reason == 'no_overlap'

    # Test non-finite coordinates - NaN (this will bypass bounds check)
    row_data = {
        psfphot._param_mapper.init_colnames['x']: np.nan,
        psfphot._param_mapper.init_colnames['y']: 25.0,
    }
    row = Table([row_data])[0]
    should_skip, reason = psfphot._should_skip_source(row, data_shape)
    assert should_skip is True
    assert reason == 'invalid_position'

    # Test non-finite coordinates - NaN in y coordinate
    row_data = {
        psfphot._param_mapper.init_colnames['x']: 25.0,
        psfphot._param_mapper.init_colnames['y']: np.nan,
    }
    row = Table([row_data])[0]
    should_skip, reason = psfphot._should_skip_source(row, data_shape)
    assert should_skip is True
    assert reason == 'invalid_position'

    # Test valid coordinates
    row_data = {
        psfphot._param_mapper.init_colnames['x']: 25.0,
        psfphot._param_mapper.init_colnames['y']: 25.0,
    }
    row = Table([row_data])[0]
    should_skip, reason = psfphot._should_skip_source(row, data_shape)
    assert should_skip is False
    assert reason is None


def test_get_source_cutout_data_no_overlap():
    """
    Test the _get_source_cutout_data method with NoOverlapError
    exception.
    """
    shape = (10, 10)
    data = np.zeros(shape)
    psf_model = CircularGaussianPRF(fwhm=2.0)
    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape)

    y_offsets, x_offsets = psfphot._get_fit_offsets()

    # Create a source that will definitely cause NoOverlapError
    # Place it far outside the data bounds (-100, -100) to trigger
    # the exception in overlap_slices and test lines 1183-1192
    init_params = QTable()
    init_params['x_init'] = [-100.0]
    init_params['y_init'] = [-100.0]
    init_params['flux_init'] = [1000.0]
    init_params['local_bkg'] = [0.0]
    init_params['id'] = [1]

    row = init_params[0]

    # Call the method that should trigger NoOverlapError
    result = psfphot._get_source_cutout_data(row, data, None,
                                             y_offsets, x_offsets)

    # Verify the expected result for NoOverlapError exception handling
    assert result['valid'] is False
    assert result['reason'] == 'no_overlap'
    assert result['xx'] is None
    assert result['yy'] is None
    assert result['cutout'] is None
    assert result['npix'] == 0
    assert np.isnan(result['cen_index'])


def test_flags_with_invalid_and_nonfinite_sources():
    """
    Test flag computation with invalid and non-finite sources.

    This test creates scenarios with invalid sources and sources that
    end up with non-finite fitted positions, triggering the continue
    statements in the flag computation.
    """
    shape = (30, 30)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    psf_model = CircularGaussianPRF(x_0=15, y_0=15, fwhm=2.0, flux=100)
    data = psf_model(xx, yy)

    # Use xy_bounds to trigger bound checking code paths
    psf_model = CircularGaussianPRF(fwhm=2.0)
    fit_shape = (7, 7)
    psfphot = PSFPhotometry(psf_model, fit_shape, xy_bounds=2.0)

    # Create init_params with mix of valid and invalid sources
    init_params = QTable()
    init_params['x_init'] = [15.0,    # Valid source
                             -100.0,  # Invalid - outside bounds
                             15.0]    # Valid source
    init_params['y_init'] = [15.0,    # Valid source
                             -100.0,  # Invalid - outside bounds
                             15.0]    # Valid source
    init_params['flux_init'] = [100.0, 100.0, 100.0]

    # Run photometry - should handle mix of valid/invalid sources
    results = psfphot(data, init_params=init_params)

    # Should return results for all sources
    assert len(results) == 3

    # Check that flags were computed appropriately
    assert 'flags' in results.colnames

    # First and third sources should have valid fits
    assert np.isfinite(results['x_fit'][0])
    assert np.isfinite(results['x_fit'][2])

    # Second source should be flagged as invalid (no overlap)
    assert (results['flags'][1] & 64) == 64  # No overlap flag


def test_levmar_fitter_with_fvec_residuals():
    """
    Test LevMarLSQFitter to exercise the 'fvec' residual key path.
    """
    shape = (25, 25)
    psf_model = CircularGaussianPRF(flux=100, fwhm=2.5)
    data, _ = make_psf_model_image(shape, psf_model, n_sources=1,
                                   model_shape=(7, 7),
                                   flux=(100, 100), seed=0)

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.5)

    # Use LevMarLSQFitter which produces 'fvec' in fit_info
    fitter = LevMarLSQFitter()

    init_params = Table()
    init_params['x_init'] = [12.0]
    init_params['y_init'] = [12.0]
    init_params['flux_init'] = [100.0]

    psfphot = PSFPhotometry(psf_model, fit_shape=(7, 7), fitter=fitter)

    # Run photometry - should use 'fvec' residual key
    results = psfphot(data, init_params=init_params)

    # Verify the fit completed successfully
    assert len(results) == 1
    assert np.isfinite(results['x_fit'][0])
    assert np.isfinite(results['y_fit'][0])
    assert np.isfinite(results['flux_fit'][0])

    # Verify that 'fvec' was found in fit_info
    assert 'fvec' in psfphot.fitter.fit_info


def _compare_lists_with_arrays(list1, list2):
    if len(list1) != len(list2):
        return False
    for item1, item2 in zip(list1, list2, strict=True):
        if isinstance(item1, dict) and isinstance(item2, dict):
            if item1.keys() != item2.keys():
                return False
            for key in item1:
                if isinstance(item1[key], np.ndarray):
                    if not np.array_equal(item1[key], item2[key],
                                          equal_nan=True):
                        return False
                elif item1[key] != item2[key]:
                    return False
        elif item1 != item2:
            return False
    return True


@pytest.mark.parametrize('reorder', ['reversed', 'permutate'])
@pytest.mark.parametrize('with_groups', [True, False])
@pytest.mark.parametrize('nonconsec_groups', [True, False])
@pytest.mark.parametrize('nonconsec_ids', [True, False])
def test_init_params_id_order(test_data, reorder, with_groups,
                              nonconsec_groups, nonconsec_ids):
    data, error, sources = test_data
    init_params = sources.copy()

    nsrc = len(sources)
    rng = np.random.default_rng(seed=0)
    init_params['id'] = np.arange(1, nsrc + 1)
    if with_groups:
        # same groupings, but different group ids
        if nonconsec_groups:
            group_ids = [11, 20, 11, 20, 20, 39, 20, 39, 44, 44]
        else:
            group_ids = [1, 2, 1, 2, 2, 3, 2, 3, 4, 4]
        init_params['group_id'] = group_ids
    init_params['local_bkg'] = rng.normal(size=nsrc)
    init_params['x_0'][0] = 1000
    init_params['y_0'][0] = 1000
    init_params['x_0'][5] = 1000
    init_params['x_0'][-2] = 1000
    init_params['x_0'][-1] = 1000

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    psfphot1 = PSFPhotometry(psf_model, fit_shape)
    phot1 = psfphot1(data, error=error, init_params=init_params)

    # reorder init_params
    init_params2 = init_params.copy()
    if nonconsec_ids:
        # non-consecutive random ids
        # monotonically increasing so final results order should be same
        steps = rng.integers(1, 51, size=nsrc - 1)
        init_params2['id'] = np.concatenate(([1], 1 + np.cumsum(steps)))
    if reorder == 'reversed':
        init_params2 = init_params2[::-1]
    elif reorder == 'permutate':
        init_params2 = init_params2[rng.permutation(nsrc)]

    psfphot2 = PSFPhotometry(psf_model, fit_shape)
    phot2 = psfphot2(data, error=error, init_params=init_params2)

    if not nonconsec_ids:
        assert_equal(phot1['id'], phot2['id'])

    if with_groups:
        # without group_id, group_id gets set to id
        assert_equal(phot1['group_id'], phot2['group_id'])

    assert np.all([np.allclose(phot1[col], phot2[col], equal_nan=True)
                   for col in phot1.colnames if col not in ('id', 'group_id')])

    # Compare fit_info with special handling for numpy arrays
    _compare_lists_with_arrays(psfphot1.fit_info, psfphot2.fit_info)


def test_reduced_chi2_metric():
    """
    Test the reduced chi-squared metric calculation.
    """
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    model_shape = (9, 9)
    n_sources = 3
    shape = (51, 51)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700),
                                             min_separation=10, seed=0)

    sigma = 0.9
    noise = make_noise_image(data.shape, mean=0, stddev=sigma, seed=0)
    data += noise
    error = np.full(data.shape, sigma)

    # Test with error array
    psfphot = PSFPhotometry(psf_model, (5, 5), aperture_radius=4)
    results = psfphot(data, error=error, init_params=true_params)

    assert 'reduced_chi2' in results.colnames
    valid_fits = results['flags'] == 0
    assert np.all(np.isfinite(results['reduced_chi2'][valid_fits]))
    assert np.all(results['reduced_chi2'][valid_fits] > 0)
    assert not isinstance(results['reduced_chi2'], u.Quantity)

    # Test without error array
    results_no_error = psfphot(data, init_params=true_params)
    assert np.all(np.isnan(results_no_error['reduced_chi2']))


def test_qfit_cfit_with_different_errors(test_data):
    """
    Test qfit and cfit with different error values.
    """
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    # Test without errors
    phot_no_error = psfphot(data)

    # Test without providing error array
    phot = psfphot(data, error=error)

    # Test with small errors
    error_small = np.full(data.shape, 0.1)
    phot_small_error = psfphot(data, error=error_small)

    # Test with large errors
    error_large = np.full(data.shape, 10.0)
    phot_large_error = psfphot(data, error=error_large)

    assert np.all(phot['qfit'] >= 0)
    assert np.all(phot_no_error['qfit'] >= 0)
    assert np.all(phot_small_error['qfit'] >= 0)
    assert np.all(phot_large_error['qfit'] >= 0)

    assert_allclose(phot['qfit'], phot_no_error['qfit'])
    assert_allclose(phot['cfit'], phot_no_error['cfit'])
    assert_allclose(phot_small_error['qfit'], phot_no_error['qfit'])
    assert_allclose(phot_small_error['cfit'], phot_no_error['cfit'])
    assert_allclose(phot_large_error['qfit'], phot_no_error['qfit'])
    assert_allclose(phot_large_error['cfit'], phot_no_error['cfit'])
