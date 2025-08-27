# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the iterative photometry module.
"""


import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable, Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.background import LocalBackground, MMMBackground
from photutils.datasets import make_model_image, make_noise_image
from photutils.detection import DAOStarFinder
from photutils.psf import (CircularGaussianPRF, IterativePSFPhotometry,
                           SourceGrouper, make_psf_model, make_psf_model_image)
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
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.abs(noise)

    return data, error, true_params


def make_mock_finder(x_col, y_col):
    def finder(data, mask=None):  # noqa: ARG001
        source_table = Table()
        source_table[x_col] = [25.1]
        source_table[y_col] = [24.9]
        return source_table
    return finder


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


@pytest.mark.parametrize('mode', ['new', 'all'])
def test_iterative_psf_photometry_compound(mode):
    x_stddev = y_stddev = 1.7
    psf_func = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=x_stddev,
                          y_stddev=y_stddev)
    psf_model = make_psf_model(psf_func, x_name='x_mean', y_name='y_mean')
    psf_model.x_stddev_2.fixed = False
    psf_model.y_stddev_2.fixed = False

    other_params = {psf_model.flux_name: (500, 700)}

    model_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             **other_params,
                                             min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.abs(noise)

    init_params = QTable()
    init_params['x'] = [54, 29, 80]
    init_params['y'] = [8, 26, 29]
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 3.0)
    grouper = SourceGrouper(min_separation=2)

    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     grouper=grouper, aperture_radius=4,
                                     sub_shape=fit_shape,
                                     mode=mode, maxiters=2)
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == len(true_params)

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
    init_params = psfphot.fit_results[-1].results_to_init_params()
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == len(true_params)

    cols = ('x_stddev_2', 'y_stddev_2')
    suffixes = ('_init', '_fit', '_err')
    colnames = [col + suffix for suffix in suffixes for col in cols]
    for colname in colnames:
        assert colname in phot.colnames


def test_iterative_psf_photometry_mode_new(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    finder = DAOStarFinder(10.0, 2.0)

    init_params = QTable()
    init_params['x'] = [54, 29, 80]
    init_params['y'] = [8, 26, 29]
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     mode='new',
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)
    phot = psfphot(data, error=error, init_params=init_params)
    cols = ['id', 'group_id', 'group_size', 'iter_detected', 'local_bkg']
    assert phot.colnames[:5] == cols
    assert len(psfphot.fit_results) == 2

    assert 'iter_detected' in phot.colnames
    assert len(phot) == len(sources)

    resid_data = psfphot.make_residual_image(data, psf_shape=fit_shape)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape

    nddata = NDData(data)
    resid_nddata = psfphot.make_residual_image(nddata, psf_shape=fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert resid_nddata.data.shape == data.shape

    # test that repeated calls reset the results
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(psfphot.fit_results) == 2

    # test NDData without units
    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty)
    phot0 = psfphot(nddata, init_params=init_params)
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert_allclose(phot0[col], phot[col])
    resid_nddata = psfphot.make_residual_image(nddata, psf_shape=fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert_equal(resid_nddata.data, resid_data)

    # test with units and mode='new'
    unit = u.Jy
    finder_units = DAOStarFinder(10.0 * unit, 2.0)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape,
                                     finder=finder_units, mode='new',
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)

    phot2 = psfphot(data << unit, error=error << unit, init_params=init_params)
    assert phot2['flux_fit'].unit == unit
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert phot2[col].unit == unit
        assert_allclose(phot2[col].value, phot[col])

    # test NDData with units
    uncertainty = StdDevUncertainty(error << unit)
    nddata = NDData(data << unit, uncertainty=uncertainty)
    phot3 = psfphot(nddata, init_params=init_params)
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert phot3[col].unit == unit
        assert_allclose(phot3[col].value, phot2[col].value)
    resid_nddata = psfphot.make_residual_image(nddata, psf_shape=fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert resid_nddata.unit == unit

    # test return None if no stars are found on first iteration
    finder = DAOStarFinder(1000.0, 2.0)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     mode='new',
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)
    match = 'No sources were found'
    with pytest.warns(NoDetectionsWarning, match=match):
        phot = psfphot(data, error=error)
    assert phot is None


def test_iterative_psf_photometry_mode_all():
    sources = QTable()
    sources['x_0'] = [50, 45, 55, 27, 22, 77, 82]
    sources['y_0'] = [50, 52, 48, 27, 30, 77, 79]
    sources['flux'] = [1000, 100, 50, 1000, 100, 1000, 100]

    shape = (101, 101)
    psf_model = CircularGaussianPRF(flux=500, fwhm=9.4)
    psf_shape = (41, 41)
    data = make_model_image(shape, psf_model, sources, model_shape=psf_shape)

    fit_shape = (5, 5)
    finder = DAOStarFinder(0.2, 6.0)
    sub_shape = psf_shape
    grouper = SourceGrouper(10)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     grouper=grouper, aperture_radius=4,
                                     sub_shape=sub_shape, mode='all',
                                     maxiters=3)
    phot = psfphot(data)
    cols = ['id', 'group_id', 'group_size', 'iter_detected', 'local_bkg']
    assert phot.colnames[:5] == cols

    assert len(phot) == 7
    assert_equal(phot['group_id'], [1, 2, 3, 1, 2, 2, 3])
    assert_equal(phot['iter_detected'], [1, 1, 1, 2, 2, 2, 2])
    assert_allclose(phot['flux_fit'], [1000, 1000, 1000, 100, 50, 100, 100])

    resid = psfphot.make_residual_image(data, psf_shape=sub_shape)
    assert_allclose(resid, 0, atol=1e-6)

    match = 'mode must be "new" or "all"'
    with pytest.raises(ValueError, match=match):
        psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                         grouper=grouper, aperture_radius=4,
                                         sub_shape=sub_shape, mode='invalid')

    match = 'grouper must be input for the "all" mode'
    with pytest.raises(ValueError, match=match):
        psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                         grouper=None, aperture_radius=4,
                                         sub_shape=sub_shape, mode='all')

    # test with units and mode='all'
    unit = u.Jy
    finderu = DAOStarFinder(0.2 * unit, 6.0)
    psfphotu = IterativePSFPhotometry(psf_model, fit_shape, finder=finderu,
                                      grouper=grouper, aperture_radius=4,
                                      sub_shape=sub_shape, mode='all',
                                      maxiters=3)
    phot2 = psfphotu(data << unit)
    assert len(phot2) == 7
    assert_equal(phot2['group_id'], [1, 2, 3, 1, 2, 2, 3])
    assert_equal(phot2['iter_detected'], [1, 1, 1, 2, 2, 2, 2])
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert phot2[col].unit == unit
        assert_allclose(phot2[col].value, phot[col])

    # test NDData with units
    nddata = NDData(data * unit)
    phot3 = psfphotu(nddata)
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert phot3[col].unit == unit
        assert_allclose(phot3[col].value, phot[col])
    resid_nddata = psfphotu.make_residual_image(nddata, psf_shape=fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert resid_nddata.unit == unit


def test_iterative_methods(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(10.0, 2.0)

    init_params = QTable()
    init_params['x'] = [54, 29, 80]
    init_params['y'] = [8, 26, 29]
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     mode='new',
                                     aperture_radius=4)

    match = 'No results available. Please run the IterativePSFPhotometry'
    with pytest.raises(ValueError, match=match):
        psfphot.make_model_image(data.shape)
    with pytest.raises(ValueError, match=match):
        psfphot.make_residual_image(data)

    phot = psfphot(data, error=error, init_params=init_params)
    cols = ['id', 'group_id', 'group_size', 'iter_detected', 'local_bkg']
    assert phot.colnames[:5] == cols
    assert len(psfphot.fit_results) == 2

    init_params = psfphot.results_to_init_params()
    assert isinstance(init_params, QTable)
    assert len(init_params) == len(sources)

    model_params = psfphot.results_to_model_params()
    assert isinstance(model_params, QTable)
    assert len(model_params) == len(sources)


def test_iterative_psf_photometry_overlap():
    """
    Regression test for #1769.

    A ValueError should not be raised for no overlap.
    """
    fwhm = 3.5
    psf_model = CircularGaussianPRF(flux=1, fwhm=fwhm)
    data, _ = make_psf_model_image((150, 150), psf_model, n_sources=300,
                                   model_shape=(11, 11), flux=(50, 100),
                                   min_separation=1, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=0.01, seed=0)
    data += noise
    error = np.abs(noise)
    slc = (slice(0, 50), slice(0, 50))
    data = data[slc]
    error = error[slc]

    daofinder = DAOStarFinder(threshold=0.5, fwhm=fwhm)
    grouper = SourceGrouper(min_separation=1.3 * fwhm)
    fitter = TRFLSQFitter()
    fit_shape = (5, 5)
    sub_shape = fit_shape
    psfphot = IterativePSFPhotometry(psf_model, fit_shape=fit_shape,
                                     finder=daofinder, mode='all',
                                     grouper=grouper, maxiters=2,
                                     sub_shape=sub_shape,
                                     aperture_radius=3, fitter=fitter)
    match = r'One or more .* may not have converged'
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error)
    assert len(phot) == 38


def test_iterative_psf_photometry_subshape():
    """
    A ValueError should not be raised if sub_shape=None and the model
    does not have a bounding box.
    """
    fwhm = 3.5
    psf_model = CircularGaussianPRF(flux=1, fwhm=fwhm)
    data, _ = make_psf_model_image((150, 150), psf_model, n_sources=30,
                                   model_shape=(11, 11), flux=(50, 100),
                                   min_separation=1, seed=0)

    daofinder = DAOStarFinder(threshold=0.5, fwhm=fwhm)
    grouper = SourceGrouper(min_separation=1.3 * fwhm)
    fitter = TRFLSQFitter()
    fit_shape = (5, 5)
    sub_shape = None
    psf_model.bounding_box = None
    psfphot = IterativePSFPhotometry(psf_model, fit_shape=fit_shape,
                                     finder=daofinder, mode='all',
                                     grouper=grouper, maxiters=2,
                                     sub_shape=sub_shape,
                                     aperture_radius=3, fitter=fitter)
    match = r'model_shape must be specified .* does not have a bounding_box'
    with pytest.raises(ValueError, match=match):
        psfphot(data)


def test_iterative_psf_photometry_inputs():
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(10.0, 2.0)

    match = 'finder cannot be None for IterativePSFPhotometry'
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=None,
                                   aperture_radius=4)

    match = 'aperture_radius cannot be None for IterativePSFPhotometry'
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                   aperture_radius=None)

    match = 'maxiters must be a strictly-positive scalar'
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                   aperture_radius=4, maxiters=-1)
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                   aperture_radius=4, maxiters=[1, 2])

    match = 'maxiters must be an integer'
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                   aperture_radius=4, maxiters=3.14)


@pytest.mark.parametrize(('x_col', 'y_col'), FINDER_COLUMN_NAMES)
def test_iterative_finder_column_names(x_col, y_col):
    """
    Test that IterativePSFPhotometry works correctly with a finder that
    returns different column names for source positions.
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
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     aperture_radius=10, maxiters=3)

    # invalid column names should raise an error
    if x_col == 'x_invalid' or y_col == 'y_invalid':
        match = 'must contain columns for x and y coordinates'
        with pytest.raises(ValueError, match=match):
            psfphot(data, init_params=sources)
        return

    phot_tbl = psfphot(data)

    assert_allclose(phot_tbl['x_init'][0], 25.1)
    assert_allclose(phot_tbl['y_init'][0], 24.9)
    assert_allclose(phot_tbl['x_fit'][0], 25.0, atol=1e-6)
    assert_allclose(phot_tbl['y_fit'][0], 25.0, atol=1e-6)
    assert_allclose(phot_tbl['flux_fit'][0], 7.0, rtol=1e-6)


def test_repr():
    psf_model = CircularGaussianPRF(flux=1.0, fwhm=3.1)
    fit_shape = (9, 9)
    finder = DAOStarFinder(6.0, 2.0)

    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     aperture_radius=10)
    cls_repr = repr(psfphot)
    assert cls_repr.startswith(f'{psfphot.__class__.__name__}(')


def test_move_column():
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     aperture_radius=4, maxiters=1)
    tbl = QTable()
    tbl['a'] = [1, 2, 3]
    tbl['b'] = [4, 5, 6]
    tbl['c'] = [7, 8, 9]

    tbl1 = psfphot._move_column(tbl, 'a', 'c')
    assert tbl1.colnames == ['b', 'c', 'a']
    tbl2 = psfphot._move_column(tbl, 'd', 'b')
    assert tbl2.colnames == ['a', 'b', 'c']
    tbl3 = psfphot._move_column(tbl, 'b', 'b')
    assert tbl3.colnames == ['a', 'b', 'c']
