# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.table import Table

from ..epsf import EPSFBuilder, EPSFFitter
from ..epsf_stars import extract_stars, EPSFStar, EPSFStars
from ...centroids import gaussian1d_moments
from ...datasets import make_gaussian_sources_image

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestEPSFBuild:
    def setup_class(self):
        """
        Create a simulated image for testing.
        """

        from scipy.spatial import cKDTree

        shape = (500, 500)

        # define random star positions
        nstars = 50
        from astropy.utils.misc import NumpyRNGContext
        with NumpyRNGContext(12345):    # seed for repeatability
            xx = np.random.uniform(low=0, high=shape[1], size=nstars)
            yy = np.random.uniform(low=0, high=shape[0], size=nstars)

        # enforce a minimum separation
        min_dist = 25
        coords = [(yy[0], xx[0])]
        for xxi, yyi in zip(xx, yy):
            newcoord = [yyi, xxi]
            dist, distidx = cKDTree([newcoord]).query(coords, 1)
            if np.min(dist) > min_dist:
                coords.append(newcoord)
        yy, xx = np.transpose(coords)

        with NumpyRNGContext(12345):    # seed for repeatability
            zz = np.random.uniform(low=0, high=200000., size=len(xx))

        # define a table of model parameters
        self.stddev = 2.
        sources = Table()
        sources['amplitude'] = zz
        sources['x_mean'] = xx
        sources['y_mean'] = yy
        sources['x_stddev'] = np.zeros(len(xx)) + self.stddev
        sources['y_stddev'] = sources['x_stddev']
        sources['theta'] = 0.

        self.data = make_gaussian_sources_image(shape, sources)
        self.nddata = NDData(self.data)

        init_stars = Table()
        init_stars['x'] = xx.astype(int)
        init_stars['y'] = yy.astype(int)
        self.init_stars = init_stars

    def test_extract_stars(self):
        size = 25
        stars = extract_stars(self.nddata, self.init_stars, size=size)

        assert len(stars) == 41
        assert isinstance(stars, EPSFStars)
        assert isinstance(stars[0], EPSFStar)
        assert stars[0].data.shape == (size, size)

    def test_epsf_build(self):
        """
        This is an end-to-end test of EPSFBuilder on a simulated image.
        """

        size = 25
        oversampling = 4.
        stars = extract_stars(self.nddata, self.init_stars, size=size)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=20,
                                   progress_bar=False)
        epsf, fitted_stars = epsf_builder(stars)

        ref_size = (size * oversampling) + 1
        assert epsf.data.shape == (ref_size, ref_size)

        y0 = int((ref_size - 1) / 2)
        z = epsf.data[y0, :]
        ampl, peak, sigma = gaussian1d_moments(z)
        assert_allclose(ampl, 0.002487, rtol=1e-4)
        assert_allclose(peak, y0, rtol=1e-3)
        assert_allclose(sigma, oversampling * self.stddev, rtol=1e-5)

    def test_epsf_build_invalid_fitter(self):
        """
        Test that the input fitter is an EPSFFitter instance.
        """

        with pytest.raises(TypeError):
            EPSFBuilder(fitter=EPSFFitter, maxiters=3)

        with pytest.raises(TypeError):
            EPSFBuilder(fitter=LevMarLSQFitter(), maxiters=3)

        with pytest.raises(TypeError):
            EPSFBuilder(fitter=LevMarLSQFitter, maxiters=3)
