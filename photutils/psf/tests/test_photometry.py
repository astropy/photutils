# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photometry module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable, Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.background import LocalBackground, MMMBackground
from photutils.datasets import make_noise_image, make_test_psf_data
from photutils.detection import DAOStarFinder
from photutils.psf import (IntegratedGaussianPRF, IterativePSFPhotometry,
                           PSFPhotometry, SourceGrouper)
from photutils.psf.models import IntegratedGaussianPRF
from photutils.psf.photometry import PSFPhotometry
from photutils.psf.photometry_depr import DAOGroup


def test_inputs():
    model = IntegratedGaussianPRF(sigma=1.0)

    with pytest.raises(TypeError):
        _ = PSFPhotometry(1, 1)

    shapes = ((0, 0), (-1, 1), (np.nan, 3), (5, np.inf), (4, 3))
    for shape in shapes:
        with pytest.raises(ValueError):
            _ = PSFPhotometry(model, shape)

    kwargs = {'grouper': 1, 'finder': 1, 'fitter': 1}
    for key, val in kwargs.items():
        with pytest.raises(TypeError):
            _ = PSFPhotometry(model, 1, **{key: val})

    with pytest.raises(ValueError):
        grouper = DAOGroup(1)
        _ = PSFPhotometry(model, 1, grouper=grouper)

    for radius in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            _ = PSFPhotometry(model, 1, aperture_radius=radius)

    psfphot = PSFPhotometry(model, (3, 3))
    with pytest.raises(ValueError):
        _ = psfphot(np.arange(3))

    with pytest.raises(ValueError):
        data = np.ones((11, 11))
        mask = np.ones((3, 3))
        _ = psfphot(data, mask=mask)

    with pytest.raises(TypeError):
        data = np.ones((11, 11))
        _ = psfphot(data, init_params=1)

    with pytest.raises(ValueError):
        tbl = Table()
        tbl['a'] = np.arange(3)
        data = np.ones((11, 11))
        _ = psfphot(data, init_params=tbl)

    # test no finder or init_params
    psfphot = PSFPhotometry(model, 1)
    with pytest.raises(ValueError):
        data = np.ones((11, 11))
        _ = psfphot(data)

    psfphot2 = PSFPhotometry(model, (3, 3), aperture_radius=3)
    init_params = Table()
    init_params['x_init'] = [1, 2]
    init_params['y_init'] = [1, 2]
    with pytest.warns(AstropyUserWarning):
        data = np.ones((11, 11))
        data[5, 5] = np.nan
        _ = psfphot2(data, init_params=init_params)

    with pytest.warns(AstropyUserWarning):
        data = np.ones((11, 11))
        data[5, 5] = np.nan
        mask = np.zeros(data.shape, dtype=bool)
        mask[7, 7] = True
        _ = psfphot2(data, init_params=init_params)

    # this should not raise a warning because the non-finite pixel was
    # explicitly masked
    data = np.ones((11, 11))
    data[5, 5] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[5, 5] = True
    _ = psfphot2(data, mask=mask, init_params=init_params)


@pytest.fixture(name='test_data')
def fixutre_test_data():
    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    psf_shape = (25, 25)
    nsources = 10
    shape = (101, 101)
    data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                           nsources, flux_range=(500, 700),
                                           min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.abs(noise)

    return data, error, true_params


def test_psf_photometry(test_data):
    data, error, sources = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    resid_data = psfphot.make_residual_image(data, fit_shape)

    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape

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


def test_psf_photometry_mask(test_data):
    data, error, sources = test_data
    data[50, 40:50] = np.nan

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    with pytest.warns(AstropyUserWarning):
        phot = psfphot(data, error=error, mask=None)
        assert len(phot) == len(sources)

    # unmasked NaN with mask not None
    #with pytest.warns(AstropyUserWarning):
    #    mask = ~np.isfinite(data)
    #    mask[50, 40] = False
    #    phot = psfphot(data, error=error, mask=mask)
    #    assert len(phot) == len(sources)

    mask = ~np.isfinite(data)
    phot2 = psfphot(data, error=error, mask=mask)
    assert np.all(phot == phot2)


def test_psf_photometry_init_params(test_data):
    data, error, sources = test_data

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

    init_params['flux'] = 650
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == 1

    init_params['group_id'] = 1
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == 1

#    init_params['flux'] = 650 * u.Jy
#    with pytest.raises(ValueError):
#        _ = psfphot(data, error=error, init_params=init_params)

    init_params['flux'] = 650 * u.Jy
    with pytest.raises(ValueError):
        _ = psfphot(data << u.m, init_params=init_params)

    init_params['flux'] = 650
    with pytest.raises(ValueError):
        _ = psfphot(data << u.Jy, init_params=init_params)


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
    assert_equal(phot['group_id'], (1, 1, 2, 2, 3, 4, 5, 6, 6, 5))
    assert_equal(phot['group_size'], (2, 2, 2, 2, 1, 1, 2, 2, 2, 2))


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


def test_iterative_psf_photometry(test_data):
    data, error, sources = test_data

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    grouper = SourceGrouper(min_separation=20)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    finder = DAOStarFinder(10.0, 2.0)

    init_params = QTable()
    init_params['x'] = [33, 13, 64]
    init_params['y'] = [12, 15, 22]
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)
    phot = psfphot(data, error=error, init_params=init_params)

    assert 'iter_detected' in phot.colnames
    assert len(phot) == len(sources)
