# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from .. import PRFAdapter
from astropy.tests.helper import pytest
from astropy.modeling.models import Moffat2D


try:
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def normalize_moffat(mof):
    # this is the analytic value needed to get a total flux of 1
    mof = mof.copy()
    mof.amplitude = (mof.alpha-1)/(np.pi*mof.gamma**2)
    return mof


@pytest.mark.parametrize("adapterkwargs", [
    dict(xname='x_0', yname='y_0', fluxname=None, renormalize_psf=False),
    dict(xname=None, yname=None, fluxname=None, renormalize_psf=False),
    dict(xname='x_0', yname='y_0', fluxname='amplitude',
         renormalize_psf=False)])
@pytest.mark.skipif('not HAS_SCIPY')
def test_create_eval_prfadapter(adapterkwargs):
    mof = Moffat2D(gamma=1, alpha=4.8)
    prf = PRFAdapter(mof, **adapterkwargs)

    # make sure these can be set without anything freaking out
    prf.x_0 = 0.5
    prf.y_0 = -0.5
    prf.flux = 1.2

    prf(0, 0)  # just make sure it runs at all


@pytest.mark.parametrize("adapterkwargs", [
    dict(xname='x_0', yname='y_0', fluxname=None, renormalize_psf=True),
    dict(xname='x_0', yname='y_0', fluxname=None, renormalize_psf=False),
    dict(xname=None, yname=None, fluxname=None, renormalize_psf=False)
    ])
@pytest.mark.skipif('not HAS_SCIPY')
def test_prfadapter_integrates(adapterkwargs):
    from scipy.integrate import dblquad

    mof = Moffat2D(gamma=1.5, alpha=4.8)
    if not adapterkwargs['renormalize_psf']:
        mof = normalize_moffat(mof)
    prf1 = PRFAdapter(mof, **adapterkwargs)

    # first check that the PRF over a central grid ends up summing to the
    # integrand over the whole PSF
    xg, yg = np.meshgrid(*([(-1, 0, 1)]*2))
    evalmod = prf1(xg, yg)

    if adapterkwargs['renormalize_psf']:
        mof = normalize_moffat(mof)

    integrando, itol = dblquad(mof, -1.5, 1.5, lambda x: -1.5, lambda x: 1.5)
    assert_allclose(np.sum(evalmod), integrando, atol=itol * 10)


@pytest.mark.parametrize("adapterkwargs", [
    dict(xname='x_0', yname='y_0', fluxname=None, renormalize_psf=False),
    dict(xname=None, yname=None, fluxname=None, renormalize_psf=False)])
@pytest.mark.skipif('not HAS_SCIPY')
def test_prfadapter_sizematch(adapterkwargs):
    from scipy.integrate import dblquad

    mof1 = normalize_moffat(Moffat2D(gamma=1, alpha=4.8))
    prf1 = PRFAdapter(mof1, **adapterkwargs)

    # now try integrating over differently-sampled PRFs
    # and check that they match
    mof2 = normalize_moffat(Moffat2D(gamma=2, alpha=4.8))
    prf2 = PRFAdapter(mof2, **adapterkwargs)

    xg1, yg1 = np.meshgrid(*([(-0.5, 0.5)]*2))
    xg2, yg2 = np.meshgrid(*([(-1.5, -0.5, 0.5, 1.5)]*2))

    eval11 = prf1(xg1, yg1)
    eval22 = prf2(xg2, yg2)

    integrand, itol = dblquad(mof1, -2, 2, lambda x: -2, lambda x: 2)
    # it's a bit of a guess that the above itol is appropriate, but it should
    # be a similar ballpark
    assert_allclose(np.sum(eval11), np.sum(eval22), atol=itol*100)
