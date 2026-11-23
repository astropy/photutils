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
position angle of the semimajor axis may differ by a large amount from
what the SPP code measures. For the noisy images (M51 and
synth_lowsnr), the fit extends by one or more isophotes beyond the
SPP code. The ``maxgerr`` criterion stops the outward growth after two
consecutive isophotes exceed it (soft stopping), while the SPP code
default is to stop after the first one. That accounts for one
isophote. The outermost isophotes of these images are also dominated
by noise, so the semimajor axis length where the growth stops changes
with rounding differences.

The not-so-good behavior observed in the case of the M51 image is to
be expected though. This image is exactly the type of galaxy image for
which the algorithm was not designed. It has an almost negligible smooth
ellipsoidal component, and a lot of lumpy spiral structure that causes
the radial gradient computation to go berserk. On top of that, the
ellipticity is small (roundish isophotes) throughout the image, causing
large relative errors and instability in the fitting algorithm.

For now, we can only check the bilinear integration mode. The mean and
median modes cannot be checked since the original 'ellipse' task has
a bug that causes the creation of erroneous output tables. A partial
comparison could be made if we write new code that reads the standard
output of 'ellipse' instead, captured from screen, and use it as
reference for the regression.
"""

import math
import os.path as op

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from photutils.datasets.load import _get_path
from photutils.isophote.ellipse import Ellipse


@pytest.mark.parametrize('name', ['synth_highsnr'])
@pytest.mark.remote_data
def test_regression(name):
    """
    NOTE: The original code in SPP won't create the right table
    for the MEAN integration mode, so use the screen output
    at synth_table_mean.txt to compare results visually with
    synth_table_mean.fits.
    """
    integrmode = 'bilinear'
    verbose = False
    filename = f'{name}_table.fits'
    path = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
    table = Table.read(path)

    nrows = len(table['SMA'])
    path = _get_path(f'isophote/{name}.fits',
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
            # Skip nonexistent rows in isophote list, if that's the case.
            break

        # Data from Isophote
        sma_i = iso.sample.geometry.sma
        intens_i = iso.intensity
        int_err_i = iso.intensity_err or 0.0
        pix_stddev_i = iso.pixel_stddev or 0.0
        rms_i = iso.rms or 0.0
        ellip_i = iso.sample.geometry.ellipticity or 0.0
        pa_i = iso.sample.geometry.orientation.to_value(u.rad)
        x0_i = iso.sample.geometry.x0
        y0_i = iso.sample.geometry.y0
        rerr_i = (iso.sample.gradient_rel_err or 0.0)
        ndata_i = iso.n_data
        nflag_i = iso.n_flagged
        niter_i = iso.n_iter
        stop_i = iso.stop_code

        # Convert to old code reference system.
        pa_i = np.rad2deg(pa_i - np.pi / 2)
        x0_i += 1
        y0_i += 1

        # Ref data from table
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

        niter_t = table['NITER'][row] or 0
        stop_t = table['STOP'][row] or -1

        # Relative differences
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

        if name == 'synth_highsnr' and integrmode == 'bilinear':
            assert abs(x0_d) <= 0.21
            assert abs(y0_d) <= 0.21

            if sma_i > 3.0:
                assert abs(intens_d) <= 1.0
            else:
                assert abs(intens_d) <= 5.0

            # Prevent "converting a masked element to nan" warning.
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


@pytest.mark.parametrize(('name', 'min_extra', 'max_extra', 'noisy'),
                         [('synth_highsnr', -2, 2, False),
                          ('synth', -2, 2, False),
                          ('synth_lowsnr', -8, 20, True),
                          ('M51', -8, 20, True)])
@pytest.mark.remote_data
def test_regression_summary(name, min_extra, max_extra, noisy):
    """
    Compare the fitted isophotes with the results of the SPP code for
    all the test images.

    Every isophote of the synthetic images with a high signal-to-noise
    ratio must agree with the SPP results. For the noisy images, only
    the median differences are limited.
    """
    path = op.join(op.dirname(op.abspath(__file__)), 'data',
                   f'{name}_table.fits')
    table = Table.read(path)
    path = _get_path(f'isophote/{name}.fits',
                     location='photutils-datasets', cache=True)
    with fits.open(path) as hdu:
        data = hdu[0].data
    isolist = Ellipse(data).fit_image()

    # The fit must extend about as far as the SPP results. In the
    # low-S/N and M51 images, the outermost isophotes are dominated by
    # noise. The semimajor axis length where their outward growth
    # stops changes by many isophotes, in both directions, with
    # rounding differences between platforms. For these images, the
    # number of isophotes is only loosely limited.
    assert min_extra <= len(isolist) - len(table) <= max_extra

    # Match the isophotes by semimajor axis length. The table values
    # have a limited precision. Only the isophotes at sma > 3 pixels
    # are compared.
    table_sma = np.array(table['SMA'])
    index = np.argmin(np.abs(isolist.sma[:, np.newaxis] - table_sma),
                      axis=0)
    matched = np.isclose(isolist.sma[index], table_sma, rtol=1e-3)
    assert np.count_nonzero(matched) >= min(len(isolist), len(table)) - 2
    mask = matched & (table_sma > 3)
    index = index[mask]
    ref = {colname: np.asarray(table[colname][mask], dtype=float)
           for colname in ('INTENS', 'ELLIP', 'PA', 'X0', 'Y0')}

    intens_diff = isolist.intensity[index] / ref['INTENS'] - 1.0
    eps_diff = isolist.ellipticity[index] - ref['ELLIP']
    # The SPP position angle is measured from the y axis. The
    # difference is wrapped into the range [-90, 90) degrees.
    pa_ref = ref['PA'] + 90.0
    pa_diff = ((isolist.orientation[index].to_value(u.deg) - pa_ref + 90.0)
               % 180.0 - 90.0)
    center_diff = np.hypot(isolist.x0[index] + 1 - ref['X0'],
                           isolist.y0[index] + 1 - ref['Y0'])

    # The limits are for the relative intensity, the ellipticity, the
    # position angle in degrees, and the center in pixels.
    if noisy:
        statistic = np.median
        limits = (0.005, 0.01, 1.0, 0.1)
    else:
        statistic = np.max
        limits = (0.001, 0.001, 0.2, 0.01)
    assert statistic(np.abs(intens_diff)) < limits[0]
    assert statistic(np.abs(eps_diff)) < limits[1]
    assert statistic(np.abs(pa_diff)) < limits[2]
    assert statistic(center_diff) < limits[3]
