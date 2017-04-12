from __future__ import (absolute_import, division, print_function, unicode_literals)

import math

import numpy as np
from astropy.io import fits

from astropy.table import Table

from photutils.isophote.ellipse import Ellipse
from photutils.isophote.integrator import BI_LINEAR, MEAN
from photutils.isophote.tests.test_data import TEST_DATA, TEST_DATA_REGRESSION

verb = False

'''
Despite being cast as a unit test, this code implements in fact
regression testing of the Ellipse algorithm, against results obtained by
the stsdas$analysis/isophote task 'ellipse'.

The stsdas task was run on test images and results were stored in tables.
The code in here runs the Ellipse algorithm on the same images, producing
a list of Isophote instances. The contents of this list then get compared
with the contents of the corresponding table.

Some quantities are compared in assert statements. These were designed to be
executed only when the synth_highsnr.fits image is used as input. That way,
we are mainly checking numerical differences that originate in the algorithms
themselves, and not caused by noise. The quantities compared this way are:

  - mean intensity: less than 1% diff. for sma > 3 pixels, 5% otherwise
  - ellipticity: less than 1% diff. for sma > 3 pixels, 20% otherwise
  - position angle: less than 1 deg. diff. for sma > 3 pixels, 20 deg. otherwise
  - X and Y position: less than 0.2 pixel diff.

For the M51 image we have mostly good agreement with the spp code in most
of the parameters (mean isophotal intensity agrees within a fraction of 1%
mostly), but every now and then the ellipticity and position angle of the
semi-major axis may differ by a large amount from what the spp code measures.
The code also stops prematurely wrt the larger sma values measured by the spp
code. This is caused by a difference in the way the gradient relative error is
measured in each case, and suggests that the spp code may have a bug.

The not-so-good behavior observed in the case of the M51 image is to be expected
though. This image is exactly the type of galaxy image for which the algorithm
*wasn't* designed for. It has an almost negligible smooth ellipsoidal component,
and a lot of lumpy spiral structure that causes the radial gradient computation
to go berserk. On top of that, the ellipticity is small (roundish isophotes)
throughout the image, causing large relative errors and instability in the fitting
algorithm.

For now, we can only check the bi-linear integration mode. The mean and median
modes cannot be checked since the original 'ellipse' task has a bug that causes
the creation of erroneous output tables. A partial comparison could be made if we
write new code that reads the standard output of 'ellipse' instead, captured from
screen, and use it as reference for the regression.
'''

def test_regression():

    integrmode = BI_LINEAR
    # integrmode = MEAN # see comment below for the MEAN mode

    # _do_regression("M51")
    # _do_regression("synth")
    # _do_regression("synth_lowsnr")

    # use this for nightly testing (no printouts)
    _do_regression("synth_highsnr", integrmode, verbose=verb)


def _do_regression(name, integrmode, verbose=True):

    table = Table.read(TEST_DATA_REGRESSION + name + '_table.fits')
    # Original code in spp won't create the right table for the 'mean'.
    # integration mode. Use the screen output at synth_table_mean.txt to
    # compare results visually.
    #
    # table = Table.read(DATA + name + '_table_mean.fits')

    nrows = len(table['SMA'])
    # print(table.columns)

    image = fits.open(TEST_DATA + name + ".fits")
    test_data = image[0].data
    ellipse = Ellipse(test_data, verbose=verbose)
    isophote_list = ellipse.fit_image(verbose=verbose)
    # isophote_list = ellipse.fit_image(integrmode=self.integrmode, sclip=2., nclip=3)

    format = "%5.2f  %6.1f    %8.3f %8.3f %8.3f        %9.5f  %6.2f   %6.2f %6.2f   %5.2f   %4d  %3d  %3d  %2d"

    for row in range(nrows):
        try:
            iso = isophote_list[row]
        except IndexError:
            # skip non-existent rows in isophote list, if that's the case.
            break

        # data from Isophote
        sma_i = iso.sample.geometry.sma
        intens_i = iso.intens
        int_err_i = iso.int_err if iso.int_err else 0.
        pix_stddev_i = iso.pix_stddev if iso.pix_stddev else 0.
        rms_i = iso.rms if iso.rms else 0.
        ellip_i = iso.sample.geometry.eps if iso.sample.geometry.eps else 0.
        pa_i = iso.sample.geometry.pa if iso.sample.geometry.pa else 0.
        x0_i = iso.sample.geometry.x0
        y0_i = iso.sample.geometry.y0
        rerr_i = iso.sample.gradient_relative_error if iso.sample.gradient_relative_error else 0.
        ndata_i = iso.ndata
        nflag_i = iso.nflag
        niter_i = iso.niter
        stop_i = iso.stop_code

        # convert to old code reference system
        pa_i = (pa_i - np.pi/2) / np.pi * 180.
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
        sma_d = (sma_i - sma_t) / sma_t * 100. if sma_t > 0. else 0.
        intens_d = (intens_i - intens_t) / intens_t * 100.
        int_err_d = (int_err_i - int_err_t) / int_err_t * 100. if int_err_t > 0. else 0.
        pix_stddev_d = (pix_stddev_i - pix_stddev_t) / pix_stddev_t * 100. if pix_stddev_t > 0. else 0.
        rms_d = (rms_i - rms_t) / rms_t * 100. if rms_t > 0. else 0.
        ellip_d = (ellip_i - ellip_t) / ellip_t * 100.
        pa_d = pa_i - pa_t  # diff in angle is absolute
        x0_d = x0_i - x0_t  # diff in position is absolute
        y0_d = y0_i - y0_t
        rerr_d = rerr_i - rerr_t  # diff in relative error is absolute
        ndata_d = (ndata_i - ndata_t) / ndata_t * 100.
        nflag_d = 0
        niter_d = 0
        stop_d = 0 if stop_i == stop_t else -1

        if verbose:
            print("* data "+format % (sma_i, intens_i, int_err_i, pix_stddev_i, rms_i, ellip_i, pa_i, x0_i, y0_i, rerr_i, ndata_i, nflag_i, niter_i, stop_i))
            print("  ref  "+format % (sma_t, intens_t, int_err_t, pix_stddev_t, rms_t, ellip_t, pa_t, x0_t, y0_t, rerr_t, ndata_t, nflag_t, niter_t, stop_t))
            print("  diff "+format % (sma_d, intens_d, int_err_d, pix_stddev_d, rms_d, ellip_d, pa_d, x0_d, y0_d, rerr_d, ndata_d, nflag_d, niter_d, stop_d))
            print()

        if name == "synth_highsnr" and integrmode == BI_LINEAR:
            assert abs(x0_d) <= 0.21
            assert abs(y0_d) <= 0.21

            if sma_i > 3.:
                assert abs(intens_d) <= 1.
            else:
                assert abs(intens_d) <= 5.

            if not math.isnan(ellip_d):
                if sma_i > 3.:
                    assert abs(ellip_d) <= 1.   #  1%
                else:
                    assert abs(ellip_d) <= 20.  #  20%
            if not math.isnan(pa_d):
                if sma_i > 3.:
                    assert abs(pa_d) <= 1.      #  1 deg.
                else:
                    assert abs(pa_d) <= 20.     #  20 deg.




