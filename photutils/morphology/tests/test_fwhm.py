from curses import COLOR_GREEN
from math import log, pi

from numpy.testing import assert_allclose

from astropy import table
from ...datasets import make_gaussian_sources_image
from ..fwhm import fwhm_cog


SIGMA_TO_FWHM = 2*(2*log(2))**0.5

def test_fwhm_cog():
    # this produces
    srctab = table.QTable()
    srctab['amplitude'] = [2, 1, .5, .5]
    srctab['x_mean'] = [100, 200, 300, 400.]
    srctab['y_mean'] = [50, 50, 50, 50.]
    srctab['x_stddev'] = srctab['y_stddev'] = [7., 5., 10., 10.]
    srctab['y_stddev'][-1] = 7.#elliptical
    srctab['theta'] = [0.,0,0,0]

    data = make_gaussian_sources_image((101, 500), srctab)

    for row in srctab:
        radius = 50 # might have this vary in a future version of this test?
        npix = pi*radius**2

        fwhm, d, cog = fwhm_cog(data, (row['x_mean'], row['y_mean'], radius),
                                aperture_method='center')

        # make sure the d and cog have the right number of pixels but don't
        # but don't stress out about values
        assert_allclose(len(d), npix, rtol=1e-2)
        assert_allclose(len(cog), npix, rtol=1e-2)

        predicted_fwhm = SIGMA_TO_FWHM * (srctab['x_stddev']*srctab['y_stddev'])**0.5
        assert_allclose(fwhm, predicted_fwhm)
