# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing aperture photometry on 2-D arrays."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import math
import warnings
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning

__all__ = []


def find_fluxvar(data, fraction, error, flux, gain, imin, imax, jmin, jmax, pixelwise_error):

    if pixelwise_error:

        subvariance = error[jmin:jmax,
                            imin:imax] ** 2

        if gain is not None:
            subvariance += (data[jmin:jmax, imin:imax] /
                            gain[jmin:jmax, imin:imax])

        # Make sure variance is > 0
        fluxvar = np.maximum(np.sum(subvariance * fraction), 0)

    else:

        local_error = error[int((jmin + jmax) / 2 + 0.5),
                            int((imin + imax) / 2 + 0.5)]

        fluxvar = np.maximum(local_error ** 2 * np.sum(fraction), 0)

        if gain is not None:
            local_gain = gain[int((jmin + jmax) / 2 + 0.5),
                              int((imin + imax) / 2 + 0.5)]
            fluxvar += flux / local_gain

    return fluxvar


def do_circular_photometry(data, positions, extents, radius,
                           error, gain, pixelwise_error, method, subpixels, r_in=None):


    ood_filter = extents['ood_filter']
    extent = extents['pixel_extent']
    phot_extent = extents['phot_extent']

    flux = u.Quantity(np.zeros(len(positions), dtype=np.float), unit=data.unit)

    if error is not None:
        fluxvar = u.Quantity(np.zeros(len(positions), dtype=np.float),
                             unit=error.unit ** 2)

    # TODO: flag these objects
    if np.sum(ood_filter):
        flux[ood_filter] = np.nan
        warnings.warn("The aperture at position {0} does not have any "
                      "overlap with the data"
                      .format(positions[ood_filter]),
                      AstropyUserWarning)
        if np.sum(ood_filter) == len(positions):
            return (flux, )

    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent

    if method == 'center':
        use_exact = 0
        subpixels = 1
    elif method == 'subpixel':
        use_exact = 0
    else:
        use_exact = 1
        subpixels = 1

    from .geometry import circular_overlap_grid

    for i in range(len(flux)):

        if not np.isnan(flux[i]):

            fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                             y_pmin[i], y_pmax[i],
                                             x_max[i] - x_min[i],
                                             y_max[i] - y_min[i],
                                             radius, use_exact, subpixels)

            if r_in is not None:
                fraction -= circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                  y_pmin[i], y_pmax[i],
                                                  x_max[i] - x_min[i],
                                                  y_max[i] - y_min[i],
                                                  r_in, use_exact, subpixels)

            flux[i] = np.sum(data[y_min[i]:y_max[i],
                                  x_min[i]:x_max[i]] * fraction)

            if error is not None:

                fluxvar[i] = find_fluxvar(data, fraction, error, flux[i], gain,
                                          x_min[i], x_max[i], y_min[i], y_max[i],
                                          pixelwise_error)

    if error is None:
        return (flux, )
    else:
        return (flux, np.sqrt(fluxvar))


def do_elliptical_photometry(data, positions, extents, a, b, theta,
                             error, gain, pixelwise_error, method, subpixels, a_in=None):

    ood_filter = extents['ood_filter']
    extent = extents['pixel_extent']
    phot_extent = extents['phot_extent']

    flux = u.Quantity(np.zeros(len(positions), dtype=np.float), unit=data.unit)

    if error is not None:
        fluxvar = u.Quantity(np.zeros(len(positions), dtype=np.float),
                             unit=error.unit ** 2)

    # TODO: flag these objects
    if np.sum(ood_filter):
        flux[ood_filter] = np.nan
        warnings.warn("The aperture at position {0} does not have any "
                      "overlap with the data"
                      .format(positions[ood_filter]),
                      AstropyUserWarning)
        if np.sum(ood_filter) == len(positions):
            return (flux, )

    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent

    if method == 'center':
        use_exact = 0
        subpixels = 1
    elif method == 'subpixel':
        use_exact = 0
    else:
        use_exact = 1
        subpixels = 1

    from .geometry import elliptical_overlap_grid

    for i in range(len(flux)):

        if not np.isnan(flux[i]):

            fraction = elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                               y_pmin[i], y_pmax[i],
                                               x_max[i] - x_min[i],
                                               y_max[i] - y_min[i],
                                               a, b, theta, use_exact, subpixels)

            if a_in is not None:
                b_in = a_in * b / a
                fraction -= elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                                    y_pmin[i], y_pmax[i],
                                                    x_max[i] - x_min[i],
                                                    y_max[i] - y_min[i],
                                                    a_in, b_in, theta, use_exact, subpixels)

            flux[i] = np.sum(data[y_min[i]:y_max[i],
                                  x_min[i]:x_max[i]] * fraction)

            if error is not None:
                fluxvar[i] = find_fluxvar(data, fraction, error, flux[i], gain,
                                          x_min[i], x_max[i], y_min[i], y_max[i],
                                          pixelwise_error)

    if error is None:
        return (flux, )
    else:
        return (flux, np.sqrt(fluxvar))


def do_rectangular_photometry(data, positions, extents, w, h, theta,
                              error, gain, pixelwise_error, method, subpixels,
                              reduce='sum'):

    ood_filter = extents['ood_filter']
    extent = extents['pixel_extent']
    phot_extent = extents['phot_extent']

    flux = u.Quantity(np.zeros(len(positions), dtype=np.float), unit=data.unit)

    if error is not None:
        fluxvar = u.Quantity(np.zeros(len(positions), dtype=np.float),
                             unit=error.unit ** 2)

    # TODO: flag these objects
    if np.sum(ood_filter):
        flux[ood_filter] = np.nan
        warnings.warn("The aperture at position {0} does not have any "
                      "overlap with the data"
                      .format(positions[ood_filter]),
                      AstropyUserWarning)
        if np.sum(ood_filter) == len(positions):
            return (flux, )

    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent

    if method in ('center', 'subpixel'):
        if method == 'center':
            method = 'subpixel'
            subpixels = 1

        from .extern.imageutils import downsample

        for i in range(len(flux)):
            if not np.isnan(flux[i]):
                x_size = ((x_pmax[i] - x_pmin[i]) /
                          (data[:, x_min[i]:x_max[i]].shape[1] * subpixels))
                y_size = ((y_pmax[i] - y_pmin[i]) /
                          (data[y_min[i]:y_max[i], :].shape[0] * subpixels))

                x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                      x_pmax[i], x_size)
                y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                      y_pmax[i], y_size)

                xx, yy = np.meshgrid(x_centers, y_centers)

                newx = (xx * math.cos(theta) + yy * math.sin(theta))
                newy = (yy * math.cos(theta) - xx * math.sin(theta))

                halfw = w / 2
                halfh = h / 2
                in_aper = (((-halfw < newx) & (newx < halfw) &
                            (-halfh < newy) & (newy < halfh)).astype(float)
                           / subpixels ** 2)

                fraction = downsample(in_aper, subpixels)
                flux[i] = np.sum(data[y_min[i]:y_max[i],
                                      x_min[i]:x_max[i]] * fraction)
                if error is not None:
                    fluxvar[i] = find_fluxvar(data, fraction, error,
                                              flux[i], gain,
                                              x_min[i], x_max[i],
                                              y_min[i], y_max[i],
                                              pixelwise_error)

    if error is None:
        return (flux, )
    else:
        return (flux, np.sqrt(fluxvar))


def do_annulus_photometry(data, positions, mode, extents,
                          inner_params, outer_params,
                          error=None, gain=None, pixelwise_error=True,
                          method='exact', subpixels=5):

    if mode == 'circular':
        if error is None:
            flux_outer = do_circular_photometry(data, positions, extents,
                                                *outer_params, error=error,
                                                pixelwise_error=pixelwise_error,
                                                method=method, gain=gain,
                                                subpixels=subpixels)
            flux_inner = do_circular_photometry(data, positions, extents,
                                                *inner_params, error=error,
                                                pixelwise_error=pixelwise_error,
                                                method=method, gain=gain,
                                                subpixels=subpixels)
        else:
            flux_outer, fluxerr_o = do_circular_photometry(data, positions,
                                                           extents,
                                                           *outer_params,
                                                           error=error,
                                                           gain=gain,
                                                           pixelwise_error=pixelwise_error,
                                                           method=method,
                                                           subpixels=subpixels)
            flux_inner, fluxerr_i = do_circular_photometry(data, positions,
                                                           extents,
                                                           *inner_params,
                                                           error=error,
                                                           gain=gain,
                                                           pixelwise_error=pixelwise_error,
                                                           method=method,
                                                           subpixels=subpixels)

    elif mode == 'elliptical':
        if error is None:
            flux_inner = do_elliptical_photometry(data, positions, extents,
                                                  *inner_params, error=error,
                                                  pixelwise_error=pixelwise_error,
                                                  method=method, gain=gain,
                                                  subpixels=subpixels)
            flux_outer = do_elliptical_photometry(data, positions, extents,
                                                  *outer_params, error=error,
                                                  pixelwise_error=pixelwise_error,
                                                  method=method, gain=gain,
                                                  subpixels=subpixels)
        else:
            flux_inner, fluxerr_i = do_elliptical_photometry(data, positions,
                                                             extents,
                                                             *inner_params,
                                                             error=error,
                                                             gain=gain,
                                                             pixelwise_error=pixelwise_error,
                                                             method=method,
                                                             subpixels=subpixels)
            flux_outer, fluxerr_o = do_elliptical_photometry(data, positions,
                                                             extents,
                                                             *outer_params,
                                                             error=error,
                                                             gain=gain,
                                                             pixelwise_error=pixelwise_error,
                                                             method=method,
                                                             subpixels=subpixels)

    elif mode == 'rectangular':
        if error is None:
            flux_inner = do_rectangular_photometry(data, positions, extents,
                                                   *inner_params, error=error,
                                                   pixelwise_error=pixelwise_error,
                                                   method=method, gain=gain,
                                                   subpixels=subpixels)
            flux_outer = do_rectangular_photometry(data, positions, extents,
                                                   *outer_params, error=error,
                                                   pixelwise_error=pixelwise_error,
                                                   method=method, gain=gain,
                                                   subpixels=subpixels)
        else:
            flux_inner, fluxerr_i = do_rectangular_photometry(data, positions,
                                                              extents,
                                                              *inner_params,
                                                              error=error,
                                                              gain=gain,
                                                              pixelwise_error=pixelwise_error,
                                                              method=method,
                                                              subpixels=subpixels)
            flux_outer, fluxerr_o = do_rectangular_photometry(data, positions,
                                                              extents,
                                                              *outer_params,
                                                              error=error,
                                                              gain=gain,
                                                              pixelwise_error=pixelwise_error,
                                                              method=method,
                                                              subpixels=subpixels)

    else:
        raise ValueError('{0} mode is not supported for annular photometry'
                         .format(mode))

    if error is None:
        flux = flux_outer[0] - flux_inner[0]
        return (flux, )
    else:
        flux = flux_outer - flux_inner
        fluxvar = np.maximum((fluxerr_o ** 2 - fluxerr_i ** 2), 0)
        return (flux, np.sqrt(fluxvar))
