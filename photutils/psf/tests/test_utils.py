# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the utils module.
"""

import numpy as np
import pytest
from astropy.convolution.utils import discretize_model
from astropy.modeling.models import Gaussian2D
from astropy.table import Table
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_allclose

from photutils.psf.groupstars import DAOGroup
from photutils.psf.models import IntegratedGaussianPRF
from photutils.psf.photometry_depr import BasicPSFPhotometry
from photutils.psf.utils import (get_grouped_psf_model, grid_from_epsfs,
                                 prepare_psf_model, subtract_psf)
from photutils.utils._optional_deps import HAS_SCIPY

PSF_SIZE = 11
GAUSSIAN_WIDTH = 1.0
IMAGE_SIZE = 101

# Position and FLUXES of test sources
INTAB = Table([[50.0, 23, 12, 86], [50.0, 83, 80, 84],
               [np.pi * 10, 3.654, 20.0, 80 / np.sqrt(3)]],
              names=['x_0', 'y_0', 'flux_0'])

# Create test psf
psf_model = Gaussian2D(1.0 / (2 * np.pi * GAUSSIAN_WIDTH ** 2), PSF_SIZE // 2,
                       PSF_SIZE // 2, GAUSSIAN_WIDTH, GAUSSIAN_WIDTH)
test_psf = discretize_model(psf_model, (0, PSF_SIZE), (0, PSF_SIZE),
                            mode='oversample')

# Set up grid for test image
image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

# Add sources to test image
for x, y, flux in INTAB:
    model = Gaussian2D(flux / (2 * np.pi * GAUSSIAN_WIDTH ** 2),
                       x, y, GAUSSIAN_WIDTH, GAUSSIAN_WIDTH)
    image += discretize_model(model, (0, IMAGE_SIZE), (0, IMAGE_SIZE),
                              mode='oversample')


@pytest.fixture(scope="module")
def moffimg():
    """
    This fixture requires scipy so don't call it from non-scipy tests
    """
    from astropy.modeling.models import Moffat2D
    from scipy import integrate

    mof = Moffat2D(alpha=4.8)

    # this is the analytic value needed to get a total flux of 1
    mof.amplitude = (mof.alpha - 1) / (np.pi * mof.gamma**2)

    # first make sure it really is normalized
    assert (1 - integrate.dblquad(mof, -10, 10,
                                  lambda x: -10, lambda x: 10)[0]) < 1e-6

    # now create an "image" of the PSF
    xg, yg = np.meshgrid(*([np.linspace(-2, 2, 100)] * 2))

    return mof, (xg, yg, mof(xg, yg))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_moffat_fitting(moffimg):
    """
    Test that the Moffat to be fit in test_psf_adapter is behaving correctly
    """
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.modeling.models import Moffat2D

    mof, (xg, yg, img) = moffimg

    # a closeish-but-wrong "guessed Moffat"
    guess_moffat = Moffat2D(x_0=.1, y_0=-.05, gamma=1.05,
                            amplitude=mof.amplitude * 1.06, alpha=4.75)

    f = LevMarLSQFitter()

    fit_mof = f(guess_moffat, xg, yg, img)
    assert_allclose(fit_mof.parameters, mof.parameters, rtol=0.01, atol=0.0005)


# we set the tolerances in flux to be 2-3% because the shape paraameters of
# the guessed version are known to be wrong.
@pytest.mark.parametrize("prepkwargs,tols", [
                         (dict(xname='x_0', yname='y_0', fluxname=None,
                               renormalize_psf=True), (1e-3, 0.02)),
                         (dict(xname=None, yname=None, fluxname=None,
                               renormalize_psf=True), (1e-3, 0.02)),
                         (dict(xname=None, yname=None, fluxname=None,
                               renormalize_psf=False), (1e-3, 0.03)),
                         (dict(xname='x_0', yname='y_0', fluxname='amplitude',
                               renormalize_psf=False), (1e-3, None)),
                         ])
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_prepare_psf_model(moffimg, prepkwargs, tols):
    """
    Test that prepare_psf_model behaves as expected for fitting (don't worry
    about full-on psf photometry for now)
    """

    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.modeling.models import Moffat2D

    mof, (xg, yg, img) = moffimg
    f = LevMarLSQFitter()

    # a close-but-wrong "guessed Moffat"
    guess_moffat = Moffat2D(x_0=.1, y_0=-.05, gamma=1.01,
                            amplitude=mof.amplitude * 1.01, alpha=4.79)
    if prepkwargs['renormalize_psf']:
        # definitely very wrong, so this ensures the re-normalization
        # stuff works
        guess_moffat.amplitude = 5.0

    if prepkwargs['xname'] is None:
        guess_moffat.x_0 = 0
    if prepkwargs['yname'] is None:
        guess_moffat.y_0 = 0

    psfmod = prepare_psf_model(guess_moffat, **prepkwargs)
    xytol, fluxtol = tols

    fit_psfmod = f(psfmod, xg, yg, img)

    if xytol is not None:
        assert np.abs(getattr(fit_psfmod, fit_psfmod.xname)) < xytol
        assert np.abs(getattr(fit_psfmod, fit_psfmod.yname)) < xytol
    if fluxtol is not None:
        assert np.abs(1 - getattr(fit_psfmod, fit_psfmod.fluxname)) < fluxtol

    # ensure the amplitude and shape parameters did *not* change
    assert fit_psfmod.psfmodel.gamma == guess_moffat.gamma
    assert fit_psfmod.psfmodel.alpha == guess_moffat.alpha
    if prepkwargs['fluxname'] is None:
        assert fit_psfmod.psfmodel.amplitude == guess_moffat.amplitude


@pytest.mark.filterwarnings('ignore:aperture_radius is None and could not '
                            'be determined by psf_model')
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_prepare_psf_model_offset():
    """
    Regression test to ensure the offset is in the correct direction.
    """
    with pytest.warns(AstropyDeprecationWarning):
        norm = False
        sigma = 3.0
        amplitude = 1.0 / (2 * np.pi * sigma**2)
        xcen = ycen = 0.0
        psf0 = Gaussian2D(amplitude, xcen, ycen, sigma, sigma)
        psf1 = prepare_psf_model(psf0, xname='x_mean', yname='y_mean',
                                 renormalize_psf=norm)
        psf2 = prepare_psf_model(psf0, renormalize_psf=norm)
        psf3 = prepare_psf_model(psf0, xname='x_mean', renormalize_psf=norm)
        psf4 = prepare_psf_model(psf0, yname='y_mean', renormalize_psf=norm)

        yy, xx = np.mgrid[0:101, 0:101]
        psf = psf1.copy()
        xval = 48
        yval = 52
        flux = 14.51
        psf.x_mean_2 = xval
        psf.y_mean_2 = yval
        data = psf(xx, yy) * flux

        group_maker = DAOGroup(2)
        bkg_estimator = None
        fitshape = 7
        init_guesses = Table([[46.1], [57.3], [7.1]],
                             names=['x_0', 'y_0', 'flux_0'])

        phot1 = BasicPSFPhotometry(group_maker=group_maker,
                                   bkg_estimator=bkg_estimator,
                                   fitshape=fitshape, psf_model=psf1)
        tbl1 = phot1(image=data, init_guesses=init_guesses)

        phot2 = BasicPSFPhotometry(group_maker=group_maker,
                                   bkg_estimator=bkg_estimator,
                                   fitshape=fitshape, psf_model=psf2)
        tbl2 = phot2(image=data, init_guesses=init_guesses)

        phot3 = BasicPSFPhotometry(group_maker=group_maker,
                                   bkg_estimator=bkg_estimator,
                                   fitshape=fitshape, psf_model=psf3)
        tbl3 = phot3(image=data, init_guesses=init_guesses)

        phot4 = BasicPSFPhotometry(group_maker=group_maker,
                                   bkg_estimator=bkg_estimator,
                                   fitshape=fitshape, psf_model=psf4)
        tbl4 = phot4(image=data, init_guesses=init_guesses)

        assert_allclose((tbl1['x_fit'][0], tbl1['y_fit'][0],
                         tbl1['flux_fit'][0]), (xval, yval, flux))
        assert_allclose((tbl2['x_fit'][0], tbl2['y_fit'][0],
                         tbl2['flux_fit'][0]), (xval, yval, flux))
        assert_allclose((tbl3['x_fit'][0], tbl3['y_fit'][0],
                         tbl3['flux_fit'][0]), (xval, yval, flux))
        assert_allclose((tbl4['x_fit'][0], tbl4['y_fit'][0],
                         tbl4['flux_fit'][0]), (xval, yval, flux))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_get_grouped_psf_model():
    igp = IntegratedGaussianPRF(sigma=1.2)
    tab = Table(names=['x_0', 'y_0', 'flux_0'],
                data=[[1, 2], [3, 4], [0.5, 1]])
    pars_to_set = {'x_0': 'x_0', 'y_0': 'y_0', 'flux_0': 'flux'}

    with pytest.warns(AstropyDeprecationWarning):
        gpsf = get_grouped_psf_model(igp, tab, pars_to_set)

    assert gpsf.x_0_0 == 1
    assert gpsf.y_0_1 == 4
    assert gpsf.flux_0 == 0.5
    assert gpsf.flux_1 == 1
    assert gpsf.sigma_0 == gpsf.sigma_1 == 1.2


@pytest.fixture(params=[0, 1, 2])
def prf_model(request):
    # use this instead of pytest.mark.parameterize as we use scipy and
    # it still calls that even if not HAS_SCIPY is set...
    prfs = [IntegratedGaussianPRF(sigma=1.2),
            Gaussian2D(x_stddev=2),
            prepare_psf_model(Gaussian2D(x_stddev=2), renormalize_psf=False)]
    return prfs[request.param]


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_get_grouped_psf_model_submodel_names(prf_model):
    """Verify that submodel tagging works"""
    tab = Table(names=['x_0', 'y_0', 'flux_0'],
                data=[[1, 2], [3, 4], [0.5, 1]])
    pars_to_set = {'x_0': 'x_0', 'y_0': 'y_0', 'flux_0': 'flux'}

    with pytest.warns(AstropyDeprecationWarning):
        gpsf = get_grouped_psf_model(prf_model, tab, pars_to_set)
        # There should be two submodels one named 0 and one named 1
        assert len([submodel for submodel in gpsf.traverse_postorder()
                    if submodel.name == 0]) == 1
        assert len([submodel for submodel in gpsf.traverse_postorder()
                    if submodel.name == 1]) == 1


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_subtract_psf():
    """Test subtract_psf."""
    with pytest.warns(AstropyDeprecationWarning):
        psf = IntegratedGaussianPRF(sigma=1.0)
        posflux = INTAB.copy()
        for n in posflux.colnames:
            posflux.rename_column(n, n.split('_')[0] + '_fit')
        residuals = subtract_psf(image, psf, posflux)
        assert np.max(np.abs(residuals)) < 0.0052


class TestGridFromEPSFs:
    """Tests for `photutils.psf.utils.grid_from_epsfs`."""

    def setup_class(self, cutout_size=25):
        # make a set of 4 EPSF models

        self.cutout_size = cutout_size

        # make simulated image
        hdu = datasets.load_simulated_hst_star_image()
        data = hdu.data

        # break up the image into four quadrants
        q1 = data[0:500, 0:500]
        q2 = data[0:500, 500:1000]
        q3 = data[500:1000, 0:500]
        q4 = data[500:1000, 500:1000]

        # select some starts from each quadrant to use to build the epsf
        quad_stars = {'q1': {'data': q1, 'fiducial': (0., 0.), 'epsf': None},
                      'q2': {'data': q2, 'fiducial': (1000., 1000.), 'epsf': None},
                      'q3': {'data': q3, 'fiducial': (1000., 0.), 'epsf': None},
                      'q4': {'data': q4, 'fiducial': (0., 1000.), 'epsf': None}}

        for q in ['q1', 'q2', 'q3', 'q4']:
            quad_data = quad_stars[q]['data']
            peaks_tbl = find_peaks(quad_data, threshold=500.)

            # filter out sources near edge
            size = cutout_size
            hsize = (size - 1) / 2
            x = peaks_tbl['x_peak']
            y = peaks_tbl['y_peak']
            mask = ((x > hsize) & (x < (quad_data.shape[1] - 1 - hsize))
                    & (y > hsize) & (y < (quad_data.shape[0] - 1 - hsize)))

            stars_tbl = Table()
            stars_tbl['x'] = peaks_tbl['x_peak'][mask]
            stars_tbl['y'] = peaks_tbl['y_peak'][mask]

            stars = extract_stars(NDData(quad_data), stars_tbl,
                                  size=cutout_size)

            epsf_builder = EPSFBuilder(oversampling=4, maxiters=3,
                                       progress_bar=False)
            epsf, fitted_stars = epsf_builder(stars)

            # set x_0, y_0 to fiducial point
            epsf.y_0 = quad_stars[q]['fiducial'][0]
            epsf.x_0 = quad_stars[q]['fiducial'][1]

            quad_stars[q]['epsf'] = epsf

        self.epsfs = [quad_stars[x]['epsf'] for x in quad_stars]
        self.fiducials = [quad_stars[x]['fiducial'] for x in quad_stars]

    def test_basic_test_grid_from_epsfs(self):

        psf_grid = grid_from_epsfs(self.epsfs)

        assert np.all(psf_grid.oversampling == self.epsfs[0].oversampling)
        assert psf_grid.data.shape == (4, psf_grid.oversampling * 25 + 1,
                                       psf_grid.oversampling * 25 + 1)

    def test_fiducials(self):
        """Test both options for setting PSF locations"""

        # default option x_0 and y_0s on input EPSFs
        psf_grid = grid_from_epsfs(self.epsfs)

        assert psf_grid.meta['grid_xypos'] == [(0.0, 0.0), (1000.0, 1000.0),
                                               (0.0, 1000.0), (1000.0, 0.0)]

        # or pass in a list
        fiducials = [(250.0, 250.0), (750.0, 750.0),
                     (250.0, 750.0), (750.0, 250.0)]

        psf_grid = grid_from_epsfs(self.epsfs, fiducials=fiducials)
        assert psf_grid.meta['grid_xypos'] == fiducials

    def test_meta(self):
        """Test the option for setting 'meta'"""

        keys = ['grid_xypos', 'oversampling', 'fill_value']

        # when 'meta' isn't provided, there should be just three keys
        psf_grid = grid_from_epsfs(self.epsfs)
        assert list(psf_grid.meta.keys()) == keys

        # when meta is provided, those new keys should exist and anything
        # in the list above should be overwritten
        meta = {'grid_xypos': 0.0, 'oversampling': 0.0,
                'fill_value': -999, 'extra_key': 'extra'}
        psf_grid = grid_from_epsfs(self.epsfs, meta=meta)
        assert list(psf_grid.meta.keys()) == keys + ['extra_key']
        assert psf_grid.meta['grid_xypos'].sort() == self.fiducials.sort()
        assert psf_grid.meta['oversampling'] == 4
        assert psf_grid.meta['fill_value'] == 0.0
