# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from astropy.modeling.models import Moffat2D
from astropy.nddata import NDData
from astropy.table import Table

from ..epsf_stars import extract_stars, EPSFStar, EPSFStars

try:
    import scipy    # noqa
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
