# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photometry module.
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

from photutils.psf.models import IntegratedGaussianPRF
from photutils.psf.photometry import PSFPhotometry


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
