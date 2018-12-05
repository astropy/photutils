# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from astropy.nddata import NDData
from astropy.table import Table

from ..epsf_stars import extract_stars, EPSFStar, EPSFStars
from ...datasets import load_simulated_hst_star_image
from ...detection import find_peaks

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


hdu = load_simulated_hst_star_image()
NDDATA = NDData(data=hdu.data)

peaks_tbl = find_peaks(hdu.data, threshold=500)
STARS_TBL = Table()
STARS_TBL['x'] = peaks_tbl['x_peak']
STARS_TBL['y'] = peaks_tbl['y_peak']


def test_extract_stars():
    stars = extract_stars(NDDATA, STARS_TBL, size=25)
    assert len(stars) == 403
    assert isinstance(stars, EPSFStars)
    assert isinstance(stars[0], EPSFStar)
    assert stars[0].data.shape == (25, 25)
    assert stars.n_stars == stars.n_all_stars
    assert stars.n_stars == stars.n_good_stars
    assert stars.center.shape == (len(stars), 2)


def test_extract_stars_inputs():
    with pytest.raises(ValueError):
        extract_stars(np.ones(3), STARS_TBL)

    with pytest.raises(ValueError):
        extract_stars(NDDATA, [(1, 1), (2, 2), (3, 3)])

    with pytest.raises(ValueError):
        extract_stars(NDDATA, [STARS_TBL, STARS_TBL])

    with pytest.raises(ValueError):
        extract_stars([NDDATA, NDDATA], STARS_TBL)
