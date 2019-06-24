# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf_stars module.
"""

from astropy.modeling.models import Moffat2D, Gaussian2D
from astropy.nddata import NDData
from astropy.table import Table
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..epsf_stars import extract_stars, EPSFStar, EPSFStars
from ..models import EPSFModel

try:
    import scipy  # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
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
        assert isinstance(stars[0], EPSFStar)
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


def test_epsf_star_residual_image():
    """
    Test to ensure ``compute_residual_image`` gives correct residuals.
    """

    size = 100
    yy, xx, = np.mgrid[0:size, 0:size]
    gmodel = Gaussian2D(100, 50, 50, 10, 10)(xx, yy)
    epsf = EPSFModel(gmodel, oversampling=4)
    data = np.zeros((size, size))
    data += epsf.evaluate(x=xx, y=yy, flux=16, x_0=50, y_0=50)
    tbl = Table()
    tbl['x'] = [50.]
    tbl['y'] = [50.]
    stars = extract_stars(NDData(data), tbl, size=25)
    residual = stars[0].compute_residual_image(epsf)

    assert_allclose(np.sum(residual), 0., atol=1.e-6)
