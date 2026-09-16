#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Monte Carlo verification of the SourceCatalog error properties.

This is a validation script rather than a timing benchmark. It checks
every reported error of ``SourceCatalog`` against the scatter of the
measured quantity over many noise realizations.

Two checks are run:

* the covariance transport alone: pixel positions are drawn from a
  known pixel covariance, converted with the high-level WCS interface,
  and their East and North scatter about the center is compared with
  the errors that ``SourceCatalog`` transports to the sky with the
  local WCS Jacobian, for several WCS types. The same transport serves
  the isophotal, windowed, and quadratic sky centroid errors.

* end to end: a Gaussian source is simulated many times with Gaussian
  noise and measured with ``SourceCatalog`` given an ``error`` array
  and a sheared, anisotropic WCS. The scatter of the measured values
  is compared with the mean reported error for the isophotal,
  windowed, and quadratic centroids in pixel and sky coordinates, and
  for the segment and Kron fluxes.

The end-to-end check is sensitive to the handling of negative pixels.
``SourceCatalog`` zeroes negative data values within a segment before
computing moments, which makes the isophotal centroid a clipped
estimator when the segment extends into wings below the noise. The
reported error, a linear propagation, is then a few percent high. Use
a footprint whose edge is well above the noise (the defaults) to see
agreement, or ``--radius 8`` with the default amplitude to see the
effect.

The quadratic centroid error is a delta-method linearization of the
3x3 peak fit evaluated at the noisy fit coefficients. It agrees at
the default amplitude but is several percent high when the peak is
only a few tens of sigma above the noise (``--amplitude 40``),
whatever the segment size.

The Kron flux scatter also includes the variation of the Kron aperture
itself, whose center, shape, and radius are measured from each
realization. The reported error propagates the ``error`` array over
the aperture of that realization only. At the default signal-to-noise
ratio, the aperture varies little and the ratio stays close to one.

Run ``python benchmarks/mc_catalog_err.py --help`` to see the
available options.
"""

import argparse
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from bench_helpers import print_environment

from photutils.datasets import make_gwcs
from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.utils._optional_deps import HAS_GWCS

ARCSEC_PER_RAD = 3600.0 * np.degrees(1)

# The pixel error covariance used for the transport check (sigma_x =
# 0.2 px, sigma_y = 0.3 px, correlated) and the position at which it
# is transported
PIX_COV = np.array([[0.04, 0.01], [0.01, 0.09]])
PIX_XY = np.array([12.3, 27.6])

# The centroid kinds checked end to end, as (attribute suffix, label)
# pairs. Each kind has x/y pixel values and errors and a sky position
# with East/North errors.
CENTROID_KINDS = (('centroid', 'isophotal'),
                  ('centroid_win', 'windowed'),
                  ('centroid_quad', 'quadratic'))

# The (value, error) flux property pairs checked end to end
FLUX_KINDS = (('segment_flux', 'segment_flux_err'),
              ('kron_flux', 'kron_flux_err'))


def make_tan_wcs(cd_arcsec, *, ctype=('RA---TAN', 'DEC--TAN'),
                 crval=(150.0, 30.0), crpix=(20.5, 20.5)):
    """
    Return a TAN WCS with the given CD matrix in arcsec per pixel.

    Parameters
    ----------
    cd_arcsec : array_like
        The 2x2 CD matrix in arcsec per pixel.

    ctype : tuple of str, optional
        The CTYPE values.

    crval : tuple of float, optional
        The CRVAL values in degrees, in the CTYPE order.

    crpix : tuple of float, optional
        The CRPIX values.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        The WCS.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = list(ctype)
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.crval = list(crval)
    wcs.wcs.cd = np.asarray(cd_arcsec, dtype=float) / 3600.0
    return wcs


def make_sip_wcs(shape):
    """
    Return a TAN-SIP WCS with a strong distortion.

    Parameters
    ----------
    shape : tuple of int
        The ``(ny, nx)`` image shape. CRPIX is at the image center.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        The distorted WCS.
    """
    wcs = make_tan_wcs([[-0.1, 0.0], [0.0, 0.1]],
                       crpix=(shape[1] / 2 + 0.5, shape[0] / 2 + 0.5))
    wcs.wcs.ctype = ['RA---TAN-SIP', 'DEC--TAN-SIP']
    header = wcs.to_header(relax=True)
    header['A_ORDER'] = header['B_ORDER'] = 2
    header['A_2_0'] = 2e-4
    header['A_1_1'] = 1e-4
    header['B_0_2'] = 2e-4
    header['B_1_1'] = -1e-4
    return WCS(header)


def make_wcs_cases(shape):
    """
    Return the (name, wcs) pairs for the transport check.

    Parameters
    ----------
    shape : tuple of int
        The ``(ny, nx)`` image shape.

    Returns
    -------
    result : list of (str, wcs) tuples
        The WCS name and instance pairs.
    """
    cos30 = np.cos(np.pi / 6)
    sin30 = np.sin(np.pi / 6)
    cases = [
        ('rotated TAN', make_tan_wcs(
            0.25 * np.array([[-cos30, sin30], [sin30, cos30]]))),
        ('sheared', make_tan_wcs([[-0.30, 0.08], [0.05, 0.20]])),
        ('swapped axes', make_tan_wcs([[0.05, 0.20], [-0.30, 0.08]],
                                      ctype=('DEC--TAN', 'RA---TAN'),
                                      crval=(30.0, 150.0))),
        ('SIP', make_sip_wcs(shape)),
        ('near pole', make_tan_wcs([[-0.25, 0.0], [0.0, 0.25]],
                                   crval=(10.0, 89.9))),
    ]
    if HAS_GWCS:
        cases.append(('gwcs', make_gwcs(shape)))
    return cases


def tangent_offsets(center, coords):
    """
    Return the East and North offsets of coordinates from a center.

    The offsets are great-circle quantities in arcsec, independent of
    the WCS helpers under test.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The reference position.

    coords : `~astropy.coordinates.SkyCoord`
        The positions.

    Returns
    -------
    east, north : `~numpy.ndarray`
        The offsets in arcsec.
    """
    sep = center.separation(coords).rad
    pa = center.position_angle(coords).rad
    return (sep * np.sin(pa) * ARCSEC_PER_RAD,
            sep * np.cos(pa) * ARCSEC_PER_RAD)


def check_transport(*, n_draw=400_000, seed=0):
    """
    Compare the transported sky errors with a Monte Carlo scatter.

    Parameters
    ----------
    n_draw : int, optional
        The number of pixel positions drawn per WCS.

    seed : int, optional
        The random number generator seed.
    """
    shape = (41, 41)
    data = np.zeros(shape)
    data[19:22, 19:22] = 10.0
    segm = SegmentationImage((data > 0).astype(int))
    rng = np.random.default_rng(seed)

    print('\n== Covariance transport (reported vs Monte Carlo, arcsec) ==')
    print(f'{"WCS":>14}{"East":>10}{"MC":>10}{"ratio":>8}'
          f'{"North":>10}{"MC":>10}{"ratio":>8}')
    precision = 1.0 / np.sqrt(2.0 * n_draw)
    worst = 0.0
    for name, wcs in make_wcs_cases(shape):
        cat = SourceCatalog(data, segm, wcs=wcs)
        err = cat._sky_err_from_cov(PIX_COV[np.newaxis], PIX_XY[np.newaxis])
        err = err.to_value(u.arcsec)[0]
        draws = rng.multivariate_normal(PIX_XY, PIX_COV, size=n_draw)
        center = wcs.pixel_to_world(*PIX_XY)
        coords = wcs.pixel_to_world(draws[:, 0], draws[:, 1])
        east, north = tangent_offsets(center, coords)
        mc = np.array([east.std(ddof=1), north.std(ddof=1)])
        ratio = mc / err
        worst = max(worst, np.abs(ratio - 1.0).max())
        print(f'{name:>14}{err[0]:10.5f}{mc[0]:10.5f}{ratio[0]:8.4f}'
              f'{err[1]:10.5f}{mc[1]:10.5f}{ratio[1]:8.4f}')
    print(f'Monte Carlo precision of a standard deviation: {precision:.4f}. '
          f'Worst |ratio - 1|: {worst:.4f} ({worst / precision:.1f} sigma).')


def measure(cat):
    """
    Return the measured values and reported errors of one catalog.

    Parameters
    ----------
    cat : `~photutils.segmentation.SourceCatalog`
        A single-source catalog built with ``error`` and ``wcs``.

    Returns
    -------
    row : list of float
        For each centroid kind, the ``x`` and ``y`` positions, the RA
        and Dec in degrees, the ``x`` and ``y`` errors, and the East
        and North errors in arcsec. These are followed by the value and
        error of each flux kind.
    """
    row = []
    for kind, _ in CENTROID_KINDS:
        skycoord = getattr(cat, f'sky_{kind}')[0]
        row.extend((getattr(cat, f'x_{kind}')[0],
                    getattr(cat, f'y_{kind}')[0],
                    skycoord.ra.deg, skycoord.dec.deg,
                    getattr(cat, f'x_{kind}_err')[0],
                    getattr(cat, f'y_{kind}_err')[0],
                    getattr(cat, f'sky_{kind}_ra_err')[0].to_value(u.arcsec),
                    getattr(cat, f'sky_{kind}_dec_err')[0].to_value(u.arcsec)))
    for value, err in FLUX_KINDS:
        row.extend((getattr(cat, value)[0], getattr(cat, err)[0]))
    return row


def centroid_rows(label, block):
    """
    Return the comparison rows of one centroid kind.

    Parameters
    ----------
    label : str
        The centroid kind label.

    block : 2D `~numpy.ndarray`
        The eight measurement columns of the kind (see `measure`), one
        row per realization.

    Returns
    -------
    rows : list of (str, float, float) tuples
        The quantity label, the Monte Carlo scatter, and the mean
        reported error.
    """
    xcen, ycen, ra, dec, xerr, yerr, raerr, decerr = block.T
    center = SkyCoord(ra.mean() * u.deg, dec.mean() * u.deg)
    coords = SkyCoord(ra * u.deg, dec * u.deg)
    east, north = tangent_offsets(center, coords)
    return [(f'x {label} (pix)', xcen.std(ddof=1), xerr.mean()),
            (f'y {label} (pix)', ycen.std(ddof=1), yerr.mean()),
            (f'East {label} (arcsec)', east.std(ddof=1), raerr.mean()),
            (f'North {label} (arcsec)', north.std(ddof=1), decerr.mean())]


def check_end_to_end(*, amplitude=400.0, radius=7.0, n_real=4000, seed=0):
    """
    Compare every reported error with the scatter of the measured
    values over noise realizations.

    Parameters
    ----------
    amplitude : float, optional
        The peak of the Gaussian source in units of the noise sigma.

    radius : float, optional
        The radius of the fixed circular segment in pixels.

    n_real : int, optional
        The number of noise realizations.

    seed : int, optional
        The random number generator seed.
    """
    wcs = make_tan_wcs([[-0.30, 0.08], [0.05, 0.20]])
    shape = (41, 41)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    x_true, y_true = 20.3, 19.6
    sigma_src = 2.5
    r2 = (xx - x_true) ** 2 + (yy - y_true) ** 2
    model = amplitude * np.exp(-r2 / (2.0 * sigma_src**2))
    footprint = r2 <= radius**2
    segm = SegmentationImage(footprint.astype(int))
    error = np.ones(shape)
    edge = amplitude * np.exp(-radius**2 / (2.0 * sigma_src**2))

    rng = np.random.default_rng(seed)
    n_cols = 8 * len(CENTROID_KINDS) + 2 * len(FLUX_KINDS)
    results = np.empty((n_real, n_cols))
    n_negative = 0
    for i in range(n_real):
        data = model + rng.normal(0.0, 1.0, shape)
        n_negative += np.count_nonzero(data[footprint] < 0)
        cat = SourceCatalog(data, segm, error=error, wcs=wcs)
        results[i] = measure(cat)

    # Realizations with a non-finite value (e.g., a windowed or
    # quadratic centroid that fell back or failed) would corrupt the
    # scatter, so they are excluded and counted
    finite = np.all(np.isfinite(results), axis=1)
    n_dropped = n_real - np.count_nonzero(finite)
    results = results[finite]

    flux = model[footprint].sum()
    snr = flux / np.sqrt(footprint.sum())

    print('\n== End to end (SourceCatalog on noisy images) ==')
    print(f'Gaussian sigma {sigma_src} px, segment radius {radius:g} px '
          f'({footprint.sum()} px), model at the segment edge '
          f'{edge:.1f} sigma, total S/N {snr:.0f}, {n_real} realizations, '
          f'{n_negative / n_real:.1f} negative pixels per realization, '
          f'{n_dropped} realizations with non-finite values dropped.')
    print(f'{"quantity":>26}{"MC scatter":>12}{"reported":>12}{"ratio":>8}')
    rows = []
    for i, (_, label) in enumerate(CENTROID_KINDS):
        rows.extend(centroid_rows(label, results[:, 8 * i:8 * (i + 1)]))
    offset = 8 * len(CENTROID_KINDS)
    for i, (value, _) in enumerate(FLUX_KINDS):
        measured = results[:, offset + 2 * i]
        reported = results[:, offset + 2 * i + 1]
        rows.append((value, measured.std(ddof=1), reported.mean()))
    for label, mc, reported in rows:
        print(f'{label:>26}{mc:12.5f}{reported:12.5f}{mc / reported:8.4f}')
    precision = 1.0 / np.sqrt(2.0 * len(results))
    print(f'Monte Carlo precision of a standard deviation: {precision:.4f}.')


def main():
    """
    Run the SourceCatalog error checks.
    """
    parser = argparse.ArgumentParser(
        description='Monte Carlo verification of the SourceCatalog '
                    'error properties.')
    parser.add_argument('--n-draw', type=int, default=400_000,
                        help='number of pixel positions drawn per WCS '
                             'in the transport check '
                             '(default: %(default)s)')
    parser.add_argument('--amplitude', type=float, default=400.0,
                        help='peak of the Gaussian source in units of '
                             'the noise sigma (default: %(default)s)')
    parser.add_argument('--radius', type=float, default=7.0,
                        help='radius of the fixed circular segment in '
                             'pixels (default: %(default)s)')
    parser.add_argument('--n-real', type=int, default=4000,
                        help='number of noise realizations in the end '
                             'to end check (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'transport', 'end-to-end'],
                        help='which check to run (default: %(default)s)')
    args = parser.parse_args()

    print_environment()
    warnings.simplefilter('ignore')

    if args.which in ('all', 'transport'):
        check_transport(n_draw=args.n_draw, seed=args.seed)
    if args.which in ('all', 'end-to-end'):
        check_end_to_end(amplitude=args.amplitude, radius=args.radius,
                         n_real=args.n_real, seed=args.seed)


if __name__ == '__main__':
    main()
