# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Table
from photutils import psf




def test_psf_fit_overlap():
    gmodi_star = psf.IntegratedGaussianPRF(sigma=3)
    psf_guess = psf.IntegratedGaussianPRF(flux=1, sigma=3)
    psf_guess.flux.fixed = psf_guess.x_0.fixed = psf_guess.y_0.fixed = False
    psf_guess.x_0.sigma = True
    imgshape = (256, 256)
    fitshape = (8,8)
    targetflux = 100
    im = np.zeros(imgshape)
    
    xs = np.array([127, 129])
    ys = np.array([128, 128])
    fluxes = np.ones((2,)) * targetflux
    
    intab = Table(names=['flux_0', 'x_0', 'y_0'], data=[fluxes, xs, ys])

    for row in intab:
        del gmodi_star.bounding_box
        gmodi_star.x_0 = row['x_0']
        gmodi_star.y_0 = row['y_0']
        gmodi_star.flux = row['flux_0']
        gmodi_star.render(im)
        

    
    outtabi = psf.psf_photometry(im, intab, psf_guess, fitshape)
    
    assert(abs(outtabi['flux_fit'][0] - targetflux) < 10)
    assert(abs(outtabi['flux_fit'][1] - targetflux) < 10)

