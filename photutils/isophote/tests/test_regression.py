# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Despite being cast as a unit test, this code implements regression
testing of the Ellipse algorithm, against results obtained by the
stsdas$analysis/isophote task 'ellipse'.

The stsdas task was run on test images and results were stored in
tables.  The code here runs the Ellipse algorithm on the same images,
producing a list of Isophote instances. The contents of this list then
get compared with the contents of the corresponding table.

Some quantities are compared in assert statements. These were designed
to be executed only when the synth_highsnr.fits image is used as input.
That way, we are mainly checking numerical differences that originate in
the algorithms themselves, and not caused by noise. The quantities
compared this way are:

  - mean intensity: less than 1% diff. for sma > 3 pixels, 5% otherwise
  - ellipticity: less than 1% diff. for sma > 3 pixels, 20% otherwise
  - position angle: less than 1 deg. diff. for sma > 3 pixels, 20 deg.
    otherwise
  - X and Y position: less than 0.2 pixel diff.

For the M51 image we have mostly good agreement with the SPP code in
most of the parameters (mean isophotal intensity agrees within a
fraction of 1% mostly), but every now and then the ellipticity and
position angle of the semi-major axis may differ by a large amount from
what the SPP code measures.  The code also stops prematurely wrt the
larger sma values measured by the SPP code. This is caused by a
difference in the way the gradient relative error is measured in each
case, and suggests that the SPP code may have a bug.

The not-so-good behavior observed in the case of the M51 image is to be
expected though. This image is exactly the type of galaxy image for
which the algorithm *wasn't* designed for. It has an almost negligible
smooth ellipsoidal component, and a lot of lumpy spiral structure that
causes the radial gradient computation to go berserk. On top of that,
the ellipticity is small (roundish isophotes) throughout the image,
causing large relative errors and instability in the fitting algorithm.

For now, we can only check the bilinear integration mode. The mean and
median modes cannot be checked since the original 'ellipse' task has a
bug that causes the creation of erroneous output tables. A partial
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
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
# @pytest.mark.parametrize('name', ['M51', 'synth', 'synth_lowsnr',
#                                   'synth_highsnr'])
@pytest.mark.parametrize('name', ['synth_highsnr'])
@pytest.mark.remote_data
def test_regression(name, integrmode=BILINEAR, verbose=False):
    """
    NOTE:  The original code in SPP won't create the right table for the MEAN
    integration moder, so use the screen output at synth_table_mean.txt to
    compare results visually with synth_table_mean.fits.
    """

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
    # isophote_list = ellipse.fit_image(sclip=2.0, nclip=3)

    fmt = ('%5.2f  %6.1f    %8.3f %8.3f %8.3f        %9.5f  %6.2f   '
           '%6.2f %6.2f   %5.2f   %4d  %3d  %3d  %2d')

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
            print('* data ' + fmt % (sma_i, intens_i, int_err_i, pix_stddev_i,
                                     rms_i, ellip_i, pa_i, x0_i, y0_i, rerr_i,
                                     ndata_i, nflag_i, niter_i, stop_i))
            print('  ref  ' + fmt % (sma_t, intens_t, int_err_t, pix_stddev_t,
                                     rms_t, ellip_t, pa_t, x0_t, y0_t, rerr_t,
                                     ndata_t, nflag_t, niter_t, stop_t))
            print('  diff ' + fmt % (sma_d, intens_d, int_err_d, pix_stddev_d,
                                     rms_d, ellip_d, pa_d, x0_d, y0_d, rerr_d,
                                     ndata_d, nflag_d, niter_d, stop_d))
            print()

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
