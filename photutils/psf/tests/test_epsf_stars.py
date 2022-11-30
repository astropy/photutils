# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf_stars module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Moffat2D
from astropy.nddata import NDData
from astropy.table import Table
from numpy.testing import assert_allclose

from photutils.psf.epsf_stars import EPSFStars, extract_stars
from photutils.psf.models import EPSFModel, IntegratedGaussianPRF
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestExtractStars:
    def setup_class(self):
        stars_tbl = Table()
        stars_tbl['x'] = [15, 15, 35, 35]
        stars_tbl['y'] = [15, 35, 40, 10]
        self.stars_tbl = stars_tbl

        yy, xx = np.mgrid[0:51, 0:55]
        self.data = np.zeros(xx.shape)
        for (xi, yi) in zip(stars_tbl['x'], stars_tbl['y']):
            m = Moffat2D(100, xi, yi, 3, 3)
            self.data += m(xx, yy)

        self.nddata = NDData(data=self.data)

    def test_extract_stars(self):
        size = 11
        stars = extract_stars(self.nddata, self.stars_tbl, size=size)
        assert len(stars) == 4
        assert isinstance(stars, EPSFStars)
        assert isinstance(stars[0], EPSFStars)
        assert stars[0].data.shape == (size, size)
        assert stars.n_stars == stars.n_all_stars
        assert stars.n_stars == stars.n_good_stars
        assert stars.center.shape == (len(stars), 2)

    def test_extract_stars_inputs(self):
        with pytest.raises(ValueError):
            extract_stars(np.ones(3), self.stars_tbl)

        with pytest.raises(ValueError):
            extract_stars(self.nddata, [(1, 1), (2, 2), (3, 3)])

        with pytest.raises(ValueError):
            extract_stars(self.nddata, [self.stars_tbl, self.stars_tbl])

        with pytest.raises(ValueError):
            extract_stars([self.nddata, self.nddata], self.stars_tbl)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_epsf_star_residual_image():
    """
    Test to ensure ``compute_residual_image`` gives correct residuals.
    """

    size = 100
    yy, xx, = np.mgrid[0:size + 1, 0:size + 1] / 4
    gmodel = IntegratedGaussianPRF().evaluate(xx, yy, 1, 12.5, 12.5, 2.5)
    epsf = EPSFModel(gmodel, oversampling=4, norm_radius=100)
    _size = 25
    data = np.zeros((_size, _size))
    _yy, _xx, = np.mgrid[0:_size, 0:_size]
    data += epsf.evaluate(x=_xx, y=_yy, flux=16, x_0=12, y_0=12)
    tbl = Table()
    tbl['x'] = [12]
    tbl['y'] = [12]
    stars = extract_stars(NDData(data), tbl, size=23)
    residual = stars[0].compute_residual_image(epsf)
    # As current EPSFStar instances cannot accept IntegratedGaussianPRF
    # as input, we have to accept some loss of precision from the
    # conversion to ePSF, and spline fitting (twice), so assert_allclose
    # cannot be more more precise than 0.001 currently.
    assert_allclose(np.sum(residual), 0.0, atol=1.0e-3, rtol=1e-3)


def test_stars_pickleable():
    """
    Verify that EPSFStars can be successfully
    pickled/unpickled for use multiprocessing
    """
    from multiprocessing.reduction import ForkingPickler

    # Doesn't need to actually contain anything useful
    stars = EPSFStars([1])
    # This should not blow up
    ForkingPickler.loads(ForkingPickler.dumps(stars))
