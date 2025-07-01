# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Despite being cast as a unit test, this code implements regression
testing of the Ellipse algorithm, against results obtained by the
stsdas$analysis/isophote task 'ellipse'.

The stsdas task was run on test images and results were stored in
tables. The code here runs the Ellipse algorithm on the same images,
producing a list of Isophote instances. The contents of this list then
get compared with the contents of the corresponding table.

Some quantities are compared in assert statements. These were designed
to be executed only when the synth_highsnr.fits image is used as input.
That way, we are mainly checking numerical differences that originate
in the algorithms themselves, and not caused by noise. The quantities
compared this way are:

* mean intensity: less than 1% diff. for sma > 3 pixels, 5% otherwise
* ellipticity: less than 1% diff. for sma > 3 pixels, 20% otherwise
* position angle: less than 1 deg. diff. for sma > 3 pixels, 20 deg.
  otherwise
* X and Y position: less than 0.2 pixel diff.

For the M51 image we have mostly good agreement with the SPP code
in most of the parameters (mean isophotal intensity agrees within a
fraction of 1% mostly), but every now and then the ellipticity and
position angle of the semi-major axis may differ by a large amount
from what the SPP code measures. The code also stops prematurely wrt
the larger sma values measured by the SPP code. This is caused by a
difference in the way the gradient relative error is measured in each
case, and suggests that the SPP code may have a bug.

The not-so-good behavior observed in the case of the M51 image is to
be expected though. This image is exactly the type of galaxy image for
which the algorithm *wasn't* designed for. It has an almost negligible
smooth ellipsoidal component, and a lot of lumpy spiral structure that
causes the radial gradient computation to go berserk. On top of that,
the ellipticity is small (roundish isophotes) throughout the image,
causing large relative errors and instability in the fitting algorithm.

For now, we can only check the bilinear integration mode. The mean and
median modes cannot be checked since the original 'ellipse' task has
a bug that causes the creation of erroneous output tables. A partial
comparison could be made if we write new code that reads the standard
output of 'ellipse' instead, captured from screen, and use it as
reference for the regression.
"""

import math
import os.path as op

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from photutils.datasets import get_path
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.integrator import BILINEAR


# @pytest.mark.parametrize('name', ['M51', 'synth', 'synth_lowsnr',
#                                   'synth_highsnr'])
@pytest.mark.parametrize('name', ['synth_highsnr'])
@pytest.mark.remote_data
def test_regression(name):
    """
    NOTE: The original code in SPP won't create the right table
    for the MEAN integration moder, so use the screen output
    at synth_table_mean.txt to compare results visually with
    synth_table_mean.fits.
    """
    integrmode = BILINEAR
    verbose = False
    filename = f'{name}_table.fits'
    path = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
    table = Table.read(path)

    nrows = len(table['SMA'])
    path = get_path(f'isophote/{name}.fits',
                    location='photutils-datasets', cache=True)
    hdu = fits.open(path)
    data = hdu[0].data
    hdu.close()

    ellipse = Ellipse(data)
    isophote_list = ellipse.fit_image()

    ttype = []
    tsma = []
    tintens = []
    tint_err = []
    tpix_stddev = []
    trms = []
    tellip = []
    tpa = []
    tx0 = []
    ty0 = []
    trerr = []
    tndata = []
    tnflag = []
    tniter = []
    tstop = []
    for row in range(nrows):
        try:
            iso = isophote_list[row]
        except IndexError:
            # skip non-existent rows in isophote list, if that's the case.
            break

        # data from Isophote
        sma_i = iso.sample.geometry.sma
        intens_i = iso.intens
        int_err_i = iso.int_err if iso.int_err else 0.0
        pix_stddev_i = iso.pix_stddev if iso.pix_stddev else 0.0
        rms_i = iso.rms if iso.rms else 0.0
        ellip_i = iso.sample.geometry.eps if iso.sample.geometry.eps else 0.0
        pa_i = iso.sample.geometry.pa if iso.sample.geometry.pa else 0.0
        x0_i = iso.sample.geometry.x0
        y0_i = iso.sample.geometry.y0
        rerr_i = (iso.sample.gradient_relative_error
                  if iso.sample.gradient_relative_error else 0.0)
        ndata_i = iso.ndata
        nflag_i = iso.nflag
        niter_i = iso.niter
        stop_i = iso.stop_code

        # convert to old code reference system
        pa_i = (pa_i - np.pi / 2) / np.pi * 180.0
        x0_i += 1
        y0_i += 1

        # ref data from table
        sma_t = table['SMA'][row]
        intens_t = table['INTENS'][row]
        int_err_t = table['INT_ERR'][row]
        pix_stddev_t = table['PIX_VAR'][row]
        rms_t = table['RMS'][row]
        ellip_t = table['ELLIP'][row]
        pa_t = table['PA'][row]
        x0_t = table['X0'][row]
        y0_t = table['Y0'][row]
        rerr_t = table['GRAD_R_ERR'][row]
        ndata_t = table['NDATA'][row]
        nflag_t = table['NFLAG'][row]
        niter_t = table['NITER'][row] if table['NITER'][row] else 0
        stop_t = table['STOP'][row] if table['STOP'][row] else -1

        # relative differences
        sma_d = (sma_i - sma_t) / sma_t * 100.0 if sma_t > 0.0 else 0.0
        intens_d = (intens_i - intens_t) / intens_t * 100.0
        int_err_d = ((int_err_i - int_err_t) / int_err_t * 100.0
                     if int_err_t > 0.0 else 0.0)
        pix_stddev_d = ((pix_stddev_i - pix_stddev_t) / pix_stddev_t * 100.0
                        if pix_stddev_t > 0.0 else 0.0)
        rms_d = (rms_i - rms_t) / rms_t * 100.0 if rms_t > 0.0 else 0.0
        ellip_d = (ellip_i - ellip_t) / ellip_t * 100.0
        pa_d = pa_i - pa_t  # diff in angle is absolute
        x0_d = x0_i - x0_t  # diff in position is absolute
        y0_d = y0_i - y0_t
        rerr_d = rerr_i - rerr_t  # diff in relative error is absolute
        ndata_d = (ndata_i - ndata_t) / ndata_t * 100.0
        nflag_d = 0
        niter_d = 0
        stop_d = 0 if stop_i == stop_t else -1

        if verbose:
            ttype.extend(('data', 'ref', 'diff'))
            tsma.extend((sma_i, sma_t, sma_d))
            tintens.extend((intens_i, intens_t, intens_d))
            tint_err.extend((int_err_i, int_err_t, int_err_d))
            tpix_stddev.extend((pix_stddev_i, pix_stddev_t, pix_stddev_d))
            trms.extend((rms_i, rms_t, rms_d))
            tellip.extend((ellip_i, ellip_t, ellip_d))
            tpa.extend((pa_i, pa_t, pa_d))
            tx0.extend((x0_i, x0_t, x0_d))
            ty0.extend((y0_i, y0_t, y0_d))
            trerr.extend((rerr_i, rerr_t, rerr_d))
            tndata.extend((ndata_i, ndata_t, ndata_d))
            tnflag.extend((nflag_i, nflag_t, nflag_d))
            tniter.extend((niter_i, niter_t, niter_d))
            tstop.extend((stop_i, stop_t, stop_d))

        if name == 'synth_highsnr' and integrmode == BILINEAR:
            assert abs(x0_d) <= 0.21
            assert abs(y0_d) <= 0.21

            if sma_i > 3.0:
                assert abs(intens_d) <= 1.0
            else:
                assert abs(intens_d) <= 5.0

            # prevent "converting a masked element to nan" warning
            if ellip_d is np.ma.masked:
                continue

            if not math.isnan(ellip_d):
                if sma_i > 3.0:
                    assert abs(ellip_d) <= 1.0  # 1%
                else:
                    assert abs(ellip_d) <= 20.0  # 20%
            if not math.isnan(pa_d):
                if sma_i > 3.0:
                    assert abs(pa_d) <= 1.0  # 1 deg.
                else:
                    assert abs(pa_d) <= 20.0  # 20 deg.

    if verbose:
        tbl = Table()
        tbl['type'] = ttype
        tbl['sma'] = tsma
        tbl['intens'] = tintens
        tbl['int_err'] = tint_err
        tbl['pix_stddev'] = tpix_stddev
        tbl['rms'] = trms
        tbl['ellip'] = tellip
        tbl['pa'] = tpa
        tbl['x0'] = tx0
        tbl['y0'] = ty0
        tbl['rerr'] = trerr
        tbl['ndata'] = tndata
        tbl['nflag'] = tnflag
        tbl['niter'] = tniter
        tbl['stop'] = tstop
        tbl.write('test_regression.ecsv', overwrite=True)
