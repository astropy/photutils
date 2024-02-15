# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photometry module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import LMLSQFitter, SimplexLSQFitter
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NDData, StdDevUncertainty, overlap_slices
from astropy.table import QTable, Table
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose, assert_equal

from photutils.background import LocalBackground, MMMBackground
from photutils.datasets import (make_gaussian_prf_sources_image,
                                make_noise_image, make_test_psf_data)
from photutils.detection import DAOStarFinder
from photutils.psf import (IntegratedGaussianPRF, IterativePSFPhotometry,
                           PSFPhotometry, SourceGrouper, make_psf_model)
from photutils.psf.photometry_depr import DAOGroup
from photutils.utils._optional_deps import HAS_SCIPY
from photutils.utils.exceptions import NoDetectionsWarning


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_inputs():
    model = IntegratedGaussianPRF(sigma=1.0)

    match = 'psf_model must be an Astropy Model subclass'
    with pytest.raises(TypeError, match=match):
        _ = PSFPhotometry(1, 3)

    match = 'fit_shape must have an odd value for both axes'
    for shape in ((0, 0), (4, 3)):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, shape)

    match = 'fit_shape must be > 0'
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(model, (-1, 1))

    match = 'fit_shape must be a finite value'
    for shape in ((np.nan, 3), (5, np.inf)):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, shape)

    kwargs = {'grouper': 1, 'finder': 1, 'fitter': 1}
    for key, val in kwargs.items():
        match = f"'{key}' must be a callable object"
        with pytest.raises(TypeError, match=match):
            _ = PSFPhotometry(model, 1, **{key: val})

    match = 'Invalid grouper class. Please use SourceGrouper.'
    with pytest.raises(ValueError, match=match):
        with pytest.warns(AstropyDeprecationWarning):
            _ = PSFPhotometry(model, 1, grouper=DAOGroup(1))

    match = 'localbkg_estimator must be a LocalBackground instance'
    with pytest.raises(ValueError, match=match):
        localbkg = MMMBackground()
        _ = PSFPhotometry(model, 1, localbkg_estimator=localbkg)

    match = 'aperture_radius must be a strictly-positive scalar'
    for radius in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, 1, aperture_radius=radius)

    match = 'data must be a 2D array'
    psfphot = PSFPhotometry(model, (3, 3))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(np.arange(3))

    match = 'data and mask must have the same shape.'
    with pytest.raises(ValueError, match=match):
        data = np.ones((11, 11))
        mask = np.ones((3, 3))
        _ = psfphot(data, mask=mask)

    match = 'init_params must be an astropy Table'
    with pytest.raises(TypeError, match=match):
        data = np.ones((11, 11))
        _ = psfphot(data, init_params=1)

    match = ('init_param must contain valid column names for the x and y '
             'source positions')
    with pytest.raises(ValueError, match=match):
        tbl = Table()
        tbl['a'] = np.arange(3)
        data = np.ones((11, 11))
        _ = psfphot(data, init_params=tbl)

    # test no finder or init_params
    match = 'finder must be defined if init_params is not input'
    psfphot = PSFPhotometry(model, (3, 3), aperture_radius=5)
    with pytest.raises(ValueError, match=match):
        data = np.ones((11, 11))
        _ = psfphot(data)

    # data has unmasked non-finite value
    match = 'Input data contains unmasked non-finite values'
    psfphot2 = PSFPhotometry(model, (3, 3), aperture_radius=3)
    init_params = Table()
    init_params['x_init'] = [1, 2]
    init_params['y_init'] = [1, 2]
    with pytest.warns(AstropyUserWarning, match=match):
        data = np.ones((11, 11))
        data[5, 5] = np.nan
        _ = psfphot2(data, init_params=init_params)

    # mask is input, but data has unmasked non-finite value
    match = 'Input data contains unmasked non-finite values'
    with pytest.warns(AstropyUserWarning, match=match):
        data = np.ones((11, 11))
        data[5, 5] = np.nan
        mask = np.zeros(data.shape, dtype=bool)
        mask[7, 7] = True
        _ = psfphot2(data, mask=mask, init_params=init_params)

    # this should not raise a warning because the non-finite pixel was
    # explicitly masked
    data = np.ones((11, 11))
    data[5, 5] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[5, 5] = True
    _ = psfphot2(data, mask=mask, init_params=init_params)


@pytest.fixture(name='test_data')
def fixture_test_data():
    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    psf_shape = (9, 9)
    nsources = 10
    shape = (101, 101)
    data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                           nsources, flux_range=(500, 700),
                                           min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.abs(noise)

    return data, error, true_params


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_photometry(test_data):
    data, error, sources = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    resid_data = psfphot.make_residual_image(data, fit_shape)

    assert isinstance(psfphot.finder_results, QTable)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape

    keys = ('local_bkg', 'init_params', 'fit_infos', 'fit_param_errs',
            'fit_error_indices', 'npixfit', 'nmodels', 'psfcenter_indices',
            'fit_residuals')
    for key in keys:
        assert key in psfphot.fit_results

    unit = u.Jy
    photu = psfphot(data * unit, error=error * unit)
    assert photu['flux_init'].unit == unit
    assert photu['flux_fit'].unit == unit
    assert photu['flux_err'].unit == unit
    resid_datau = psfphot.make_residual_image(data * u.Jy, fit_shape)
    assert resid_datau.unit == unit

    photm = psfphot(data, error=error, mask=np.ma.nomask)
    assert np.all(phot == photm)

    # test NDData input
    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty)
    psfphot2 = PSFPhotometry(psf_model, fit_shape, finder=finder,
                             aperture_radius=4)
    phot2 = psfphot2(nddata)
    resid_data2 = psfphot2.make_residual_image(nddata, fit_shape)

    assert np.all(phot == phot2)
    assert isinstance(resid_data2, NDData)
    assert resid_data2.data.shape == data.shape
    assert_allclose(resid_data, resid_data2.data)

    # test NDData input with units
    unit = u.Jy
    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty, unit=unit)
    psfphotu = PSFPhotometry(psf_model, fit_shape, finder=finder,
                             aperture_radius=4)
    photu = psfphotu(nddata)
    assert photu['flux_init'].unit == unit
    assert photu['flux_fit'].unit == unit
    assert photu['flux_err'].unit == unit
    resid_data3 = psfphotu.make_residual_image(nddata, fit_shape)
    assert resid_data3.unit == unit


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_model_residual_image(test_data):
    data, error, _ = test_data

    data = data + 10
    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(16.0, 2.0)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4,
                            localbkg_estimator=localbkg_estimator)
    psfphot(data, error=error)

    psf_shape = (25, 25)
    model1 = psfphot.make_model_image(data.shape, psf_shape,
                                      include_localbkg=False)
    model2 = psfphot.make_model_image(data.shape, psf_shape,
                                      include_localbkg=True)
    resid1 = psfphot.make_residual_image(data, psf_shape,
                                         include_localbkg=False)
    resid2 = psfphot.make_residual_image(data, psf_shape,
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_photometry_compound(test_data):
    """
    Test compound models output from ``make_psf_model``.
    """
    data, error, sources = test_data
    x_stddev = y_stddev = 1.8
    psf_func = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=x_stddev,
                          y_stddev=y_stddev)
    psf_model = make_psf_model(psf_func, x_name='x_mean', y_name='y_mean')

    fit_shape = (5, 5)
    finder = DAOStarFinder(5.0, 3.0)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)

    # test results when fit does not converge (fitter_maxiters=10)
    match = r'One or more fit\(s\) may not have converged.'
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                                aperture_radius=4, fitter_maxiters=10)
        phot = psfphot(data, error=error)
        columns1 = ['x_err', 'y_err', 'flux_err']
        for column in columns1:
            assert np.all(np.isnan(phot[column]))

    # allow other parameters to vary
    psf_model.x_stddev_2.fixed = False
    psf_model.y_stddev_2.fixed = False
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4, fitter_maxiters=400)
    phot = psfphot(data, error=error)
    columns2 = ['x_stddev_2_init', 'y_stddev_2_init', 'x_stddev_2_fit',
                'y_stddev_2_fit', 'x_stddev_2_err', 'y_stddev_2_err']
    for column in columns2:
        assert column in phot.colnames

    # test results when fit does not converge (fitter_maxiters=10)
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                                aperture_radius=4, fitter_maxiters=10)
        phot = psfphot(data, error=error)
        for column in columns1 + ['x_stddev_2_err', 'y_stddev_2_err']:
            assert np.all(np.isnan(phot[column]))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_photometry_mask(test_data):
    data, error, sources = test_data
    data_orig = data.copy()
    data = data.copy()
    data[50, 40:50] = np.nan

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    match = 'Input data contains unmasked non-finite values'
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error, mask=None)
        assert len(phot) == len(sources)

    # unmasked NaN with mask not None
    match = 'Input data contains unmasked non-finite values'
    with pytest.warns(AstropyUserWarning, match=match):
        mask = ~np.isfinite(data)
        mask[50, 40] = False
        phot = psfphot(data, error=error, mask=mask)
        assert len(phot) == len(sources)

    mask = ~np.isfinite(data)
    phot2 = psfphot(data, error=error, mask=mask)
    assert np.all(phot == phot2)

    # mask all True; finder returns no sources
    with pytest.warns(NoDetectionsWarning):
        mask = np.ones(data.shape, dtype=bool)
        _ = psfphot(data, mask=mask)

    # completely masked source
    match = ('is completely masked. Remove the source from init_params '
             'or correct the input mask')
    with pytest.raises(ValueError, match=match):
        init_params = QTable()
        init_params['x'] = [42]
        init_params['y'] = [36]
        mask = np.ones(data.shape, dtype=bool)
        _ = psfphot(data, mask=mask, init_params=init_params)

    # completely masked source
    match = ('The number of data points is less than the number of fit '
             'parameters.')
    with pytest.raises(ValueError, match=match):
        init_params = QTable()
        init_params['x'] = [42]
        init_params['y'] = [36]
        mask = np.zeros(data.shape, dtype=bool)
        mask[35:37, :] = True
        mask[37, 42:44] = True
        psfphot = PSFPhotometry(psf_model, (3, 3), finder=finder,
                                aperture_radius=4)
        _ = psfphot(data_orig, mask=mask, init_params=init_params)

    # masked central pixel
    init_params = QTable()
    init_params['x'] = [42]
    init_params['y'] = [36]
    mask = np.zeros(data.shape, dtype=bool)
    mask[36, 42] = True
    phot = psfphot(data_orig, mask=mask, init_params=init_params)
    assert len(phot) == 1


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_photometry_init_params(test_data):
    data, error, _ = test_data
    data = data.copy()

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    init_params = QTable()
    init_params['x'] = [42]
    init_params['y'] = [36]
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == 1

    match = 'aperture_radius must be defined if init_params is not input'
    with pytest.raises(ValueError, match=match):
        psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                                aperture_radius=None)
        _ = psfphot(data, error=error, init_params=init_params)

    init_params['flux'] = 650
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == 1

    init_params['group_id'] = 1
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == 1

    init_params['flux'] = [650 * u.Jy]
    match = ('init_params flux column has units, but the input data does '
             'not have units')
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, error=error, init_params=init_params)

    init_params['flux'] = [650 * u.Jy]
    match = ('init_params flux column has units that are incompatible with '
             'the input data units')
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data << u.m, init_params=init_params)

    init_params['flux'] = [650]
    match = ('The input data has units, but the init_params flux column '
             'does not have units')
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data << u.Jy, init_params=init_params)

    init_params = QTable()
    init_params['x'] = [-42]
    init_params['y'] = [-36]
    init_params['flux'] = [100]
    match = 'does not overlap with the input data'
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, init_params=init_params)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_photometry_init_params_columns(test_data):
    data, error, _ = test_data
    data = data.copy()

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder)

    xy_suffixes = ('_init', 'init', 'centroid', '_centroid', '_peak', '',
                   'cen', '_cen', 'pos', '_pos', '_0', '0')
    flux_cols = ['flux_init', 'flux_0', 'flux0', 'flux', 'source_sum',
                 'segment_flux', 'kron_flux']
    flux_cols += flux_cols[0:4]  # pad to have same length as xy_suffixes

    xcols = ['x' + i for i in xy_suffixes]
    ycols = ['y' + i for i in xy_suffixes]

    phots = []
    for xcol, ycol, fluxcol in zip(xcols, ycols, flux_cols):
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper(test_data):
    data, error, sources = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_large_group_warning():
    psf_model = IntegratedGaussianPRF(flux=1, sigma=1.0)
    grouper = SourceGrouper(min_separation=50)
    psf_shape = (5, 5)
    fit_shape = (5, 5)
    nsources = 50
    shape = (301, 301)
    data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                           nsources, flux_range=(500, 700),
                                           min_separation=10, seed=0)
    match = 'Some groups have more than'
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot = PSFPhotometry(psf_model, fit_shape, grouper=grouper)
        psfphot(data, init_params=true_params)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_local_bkg(test_data):
    data, error, sources = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_fixed_params(test_data):
    data, error, _ = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    psf_model.flux.fixed = True
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    match = r'One or more fit\(s\) may not have converged.'
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error)
        assert np.all(np.isnan(phot['x_err']))
        assert np.all(np.isnan(phot['y_err']))
        assert np.all(np.isnan(phot['flux_err']))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_fit_warning(test_data):
    data, _, _ = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    psf_model.flux.fixed = False
    fit_shape = (5, 5)
    fitter = LMLSQFitter()  # uses "status" instead of "ierr"
    finder = DAOStarFinder(6.0, 2.0)
    # set fitter_maxiters = 1 so that the fit error status is set
    psfphot = PSFPhotometry(psf_model, fit_shape, fitter=fitter,
                            fitter_maxiters=1, finder=finder,
                            aperture_radius=4)

    match = r'One or more fit\(s\) may not have converged.'
    with pytest.warns(AstropyUserWarning, match=match):
        _ = psfphot(data)
        assert len(psfphot.fit_results['fit_error_indices']) > 0


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_fitter_no_maxiters_no_residuals(test_data):
    data, error, _ = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    psf_model.flux.fixed = False
    fit_shape = (5, 5)
    fitter = SimplexLSQFitter()  # does not produce residual array
    finder = DAOStarFinder(6.0, 2.0)
    match = '"maxiters" will be ignored because it is not accepted by'
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot = PSFPhotometry(psf_model, fit_shape, fitter=fitter,
                                finder=finder, aperture_radius=4)
        phot = psfphot(data, error=error)
        assert np.all(np.isnan(phot['qfit']))
        assert np.all(np.isnan(phot['cfit']))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_iterative_psf_photometry(test_data):
    data, error, sources = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    finder = DAOStarFinder(10.0, 2.0)

    init_params = QTable()
    init_params['x'] = [54, 29, 80]
    init_params['y'] = [8, 26, 29]
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)
    phot = psfphot(data, error=error, init_params=init_params)

    assert 'iter_detected' in phot.colnames
    assert len(phot) == len(sources)

    resid_data = psfphot.make_residual_image(data, fit_shape)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape

    nddata = NDData(data)
    resid_nddata = psfphot.make_residual_image(nddata, fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert resid_nddata.data.shape == data.shape

    unit = u.Jy
    resid_data = psfphot.make_residual_image(data * unit, fit_shape)
    assert resid_data.unit == unit

    nddata = NDData(data * unit)
    resid_nddata = psfphot.make_residual_image(nddata, fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert resid_nddata.unit == unit

    # test return None if no stars are found on first iteration
    finder = DAOStarFinder(1000.0, 2.0)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)
    with pytest.warns(NoDetectionsWarning):
        phot = psfphot(data, error=error)
        assert phot is None


def make_psf_model_image(shape, psf_model, param_table, psf_shape):
    model_params = param_table.colnames
    data = np.zeros(shape, dtype=float)
    for row in param_table:
        for param in model_params:
            setattr(psf_model, param, row[param])
        xcen = row['x_0']
        ycen = row['y_0']
        slc_lg, _ = overlap_slices(shape, psf_shape, (ycen, xcen), mode='trim')
        yy, xx = np.mgrid[slc_lg]
        data[slc_lg] += psf_model(xx, yy)
    return data


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_iterative_psf_photometry_all():
    sources = QTable()
    sources['x_0'] = [50, 45, 55, 27, 22, 77, 82]
    sources['y_0'] = [50, 52, 48, 27, 30, 77, 79]
    sources['flux'] = [1000, 100, 50, 1000, 100, 1000, 100]

    shape = (101, 101)
    psf_model = IntegratedGaussianPRF(flux=500, sigma=4.0)
    psf_shape = (41, 41)
    data = make_psf_model_image(shape, psf_model, sources, psf_shape)

    fit_shape = (5, 5)
    finder = DAOStarFinder(0.2, 6.0)
    sub_shape = psf_shape
    grouper = SourceGrouper(10)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     grouper=grouper, aperture_radius=4,
                                     sub_shape=sub_shape, mode='all',
                                     maxiters=3)
    phot = psfphot(data)

    assert len(phot) == 7
    assert_equal(phot['group_id'], [1, 2, 3, 1, 2, 2, 3])
    assert_equal(phot['iter_detected'], [1, 1, 1, 2, 2, 2, 2])
    assert_allclose(phot['flux_fit'], [1000, 1000, 1000, 100, 50, 100, 100])

    resid = psfphot.make_residual_image(data, sub_shape)
    assert_allclose(resid, 0, atol=1e-6)

    match = 'mode must be "new" or "all".'
    with pytest.raises(ValueError, match=match):
        psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                         grouper=grouper, aperture_radius=4,
                                         sub_shape=sub_shape, mode='invalid')

    match = 'grouper must be input for the "all" mode'
    with pytest.raises(ValueError, match=match):
        psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                         grouper=None, aperture_radius=4,
                                         sub_shape=sub_shape, mode='all')


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_iterative_psf_photometry_inputs():
    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_negative_xy():
    sources = Table()
    sources['id'] = np.arange(3) + 1
    sources['flux'] = 1
    sources['x_0'] = [-1.4, 15.2, -0.7]
    sources['y_0'] = [-0.3, -0.4, 18.7]
    sources['sigma'] = 3.1
    shape = (31, 31)
    data = make_gaussian_prf_sources_image(shape, sources)
    psf_model = IntegratedGaussianPRF(flux=1, sigma=3.1)
    fit_shape = (11, 11)
    psfphot = PSFPhotometry(psf_model, fit_shape,
                            aperture_radius=10)
    phot = psfphot(data, init_params=sources)
    assert_equal(phot['x_init'], sources['x_0'])
    assert_equal(phot['y_init'], sources['y_0'])


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_out_of_bounds_centroids():
    sources = Table()
    sources['id'] = np.arange(8) + 1
    sources['flux'] = 1
    sources['x_0'] = [-1.4, 34.5, 14.2, -0.7, 34.5, 14.2, 51.3, 52.0]
    sources['y_0'] = [13, -0.2, -1.6, 40, 51.1, 50.9, 12.2, 42.3]
    sources['sigma'] = 3.1

    shape = (51, 51)
    data = make_gaussian_prf_sources_image(shape, sources)

    psf_model = IntegratedGaussianPRF(flux=1, sigma=3.1)
    fit_shape = (11, 11)
    psfphot = PSFPhotometry(psf_model, fit_shape,
                            aperture_radius=10)

    phot = psfphot(data, init_params=sources)

    # at least one of the best-fit centroids should be
    # out of the bounds of the dataset, producing a
    # masked value in the `cfit` column:
    assert np.any(np.isnan(phot['cfit']))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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
