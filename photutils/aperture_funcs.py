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


def do_circular_photometry(data, positions, extents, radius,
                           error, gain, pixelwise_error, method, subpixels):

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
        for i in range(len(flux)):
            x_size = ((x_pmax[i] - x_pmin[i]) /
                      data[:, x_min[i]:x_max[i]].shape[1])
            y_size = ((y_pmax[i] - y_pmin[i]) /
                      data[y_min[i]:y_max[i], :].shape[0])

            x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                  x_pmax[i], x_size)
            y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                  y_pmax[i], y_size)
            xx, yy = np.meshgrid(x_centers, y_centers)
            if not np.isnan(flux[i]):
                if error is None:
                    flux[i] = np.sum(data[y_min[i]:y_max[i],
                                          x_min[i]:x_max[i]] *
                                     (xx * xx + yy * yy < radius * radius))
                else:
                    fraction = (xx * xx + yy * yy < radius * radius)
                    flux[i] = np.sum(data[y_min[i]:y_max[i],
                                          x_min[i]:x_max[i]] * fraction)
                    if pixelwise_error:
                        subvariance = error[y_min[i]:y_max[i],
                                            x_min[i]:x_max[i]] ** 2
                        if gain is not None:
                            subvariance += (data[y_min[i]:y_max[i],
                                                 x_min[i]:x_max[i]] /
                                            gain[y_min[i]:y_max[i],
                                                 x_min[i]:x_max[i]])
                        # Make sure variance is > 0
                        fluxvar[i] = np.max(np.sum(subvariance * fraction), 0)
                    else:
                        local_error = error[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                            int((x_min[i] + x_max[i]) / 2 + 0.5)]
                        fluxvar[i] = np.max(local_error ** 2 *
                                            np.sum(fraction), 0)
                        if gain is not None:
                            local_gain = gain[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                              int((x_min[i] + x_max[i]) / 2 + 0.5)]
                            fluxvar[i] += flux[i] / local_gain

    elif method == 'subpixel':
        from .geometry import circular_overlap_grid
        for i in range(len(flux)):
            if not np.isnan(flux[i]):
                if error is None:
                    flux[i] = np.sum(data[y_min[i]:y_max[i],
                                          x_min[i]:x_max[i]] *
                                     circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                           y_pmin[i], y_pmax[i],
                                                           x_max[i] - x_min[i],
                                                           y_max[i] - y_min[i],
                                                           radius, 0,
                                                           subpixels))

                else:
                    fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                     y_pmin[i], y_pmax[i],
                                                     x_max[i] - x_min[i],
                                                     y_max[i] - y_min[i],
                                                     radius, 0, subpixels)
                    flux[i] = np.sum(data[y_min[i]:y_max[i],
                                          x_min[i]:x_max[i]] * fraction)

                    if pixelwise_error:
                        subvariance = error[y_min[i]:y_max[i],
                                            x_min[i]:x_max[i]] ** 2
                        if gain is not None:
                            subvariance += (data[y_min[i]:y_max[i],
                                                 x_min[i]:x_max[i]] /
                                            gain[y_min[i]:y_max[i],
                                                 x_min[i]:x_max[i]])
                        # Make sure variance is > 0
                        fluxvar[i] = np.max(np.sum(subvariance * fraction), 0)
                    else:
                        local_error = error[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                            int((x_min[i] + x_max[i]) / 2 + 0.5)]
                        fluxvar[i] = np.max(local_error ** 2 *
                                            np.sum(fraction), 0)
                        if gain is not None:
                            local_gain = gain[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                              int((x_min[i] + x_max[i]) / 2 + 0.5)]
                            fluxvar[i] += flux[i] / local_gain

    elif method == 'exact':
        from .geometry import circular_overlap_grid
        for i in range(len(flux)):
            if not np.isnan(flux[i]):
                if error is None:
                    flux[i] = np.sum(data[y_min[i]:y_max[i], x_min[i]:x_max[i]] *
                                     circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                           y_pmin[i], y_pmax[i],
                                                           x_max[i] - x_min[i],
                                                           y_max[i] - y_min[i],
                                                           radius, 1, 1))
                else:
                    fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                     y_pmin[i], y_pmax[i],
                                                     x_max[i] - x_min[i],
                                                     y_max[i] - y_min[i],
                                                     radius, 1, 1)
                    flux[i] = np.sum(data[y_min[i]:y_max[i],
                                          x_min[i]:x_max[i]] * fraction)

                    if pixelwise_error:
                        subvariance = error[y_min[i]:y_max[i],
                                            x_min[i]:x_max[i]] ** 2
                        if gain is not None:
                            subvariance += (data[y_min[i]:y_max[i],
                                                 x_min[i]:x_max[i]] /
                                            gain[y_min[i]:y_max[i],
                                                 x_min[i]:x_max[i]])
                        # Make sure variance is > 0
                        fluxvar[i] = np.max(np.sum(subvariance * fraction), 0)
                    else:
                        local_error = error[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                            int((x_min[i] + x_max[i]) / 2 + 0.5)]
                        fluxvar[i] = np.max(local_error ** 2 *
                                            np.sum(fraction), 0)
                        if gain is not None:
                            local_gain = gain[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                              int((x_min[i] + x_max[i]) / 2 + 0.5)]
                            fluxvar[i] += flux[i] / local_gain

    if error is None:
        return (flux, )
    else:
        return (flux, np.sqrt(fluxvar))


def do_elliptical_photometry(data, positions, extents, a, b, theta,
                             error, gain, pixelwise_error, method, subpixels):

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

    if method == 'center' or method == 'subpixel':
        if method == 'center': subpixels = 1
        if method == 'subpixel': from imageutils import downsample

        for i in range(len(flux)):
            x_size = ((x_pmax[i] - x_pmin[i]) /
                      (data[:, x_min[i]:x_max[i]].shape[1] * subpixels))
            y_size = ((y_pmax[i] - y_pmin[i]) /
                      (data[y_min[i]:y_max[i], :].shape[0] * subpixels))

            x_centers = np.arange(x_pmin[i] + x_size / 2.,
                                  x_pmax[i], x_size)
            y_centers = np.arange(y_pmin[i] + y_size / 2.,
                                  y_pmax[i], y_size)
            xx, yy = np.meshgrid(x_centers, y_centers)
            numerator1 = (xx * math.cos(theta) + yy * math.sin(theta))
            numerator2 = (yy * math.cos(theta) - xx * math.sin(theta))

            in_aper = ((((numerator1 / a) ** 2 +
                         (numerator2 / b) ** 2) < 1.).astype(float)
                       / subpixels ** 2)

            if method == 'center':
                if not np.isnan(flux[i]):
                    flux[i] = np.sum(data[y_min[i]:y_max[i],
                                          x_min[i]:x_max[i]] * in_aper)
                    if error is not None:
                        if pixelwise_error:
                            subvariance = error[y_min[i]:y_max[i],
                                                x_min[i]:x_max[i]] ** 2
                            if gain is not None:
                                subvariance += (data[y_min[i]:y_max[i],
                                                     x_min[i]:x_max[i]] /
                                                gain[y_min[i]:y_max[i],
                                                     x_min[i]:x_max[i]])
                            # Make sure variance is > 0
                            fluxvar[i] = np.max(np.sum(subvariance *
                                                       in_aper), 0)
                        else:
                            local_error = error[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                                int((x_min[i] + x_max[i]) / 2 + 0.5)]
                            fluxvar[i] = np.max(local_error ** 2 *
                                                np.sum(in_aper), 0)
                            if gain is not None:
                                local_gain = gain[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                                  int((x_min[i] + x_max[i]) / 2 + 0.5)]
                                fluxvar[i] += flux[i] / local_gain

            else:
                if not np.isnan(flux[i]):
                    if error is None:
                        flux[i] = np.sum(data[y_min[i]:y_max[i],
                                              x_min[i]:x_max[i]] *
                                         downsample(in_aper, subpixels))
                    else:
                        fraction = downsample(in_aper, subpixels)
                        flux[i] = np.sum(data[y_min[i]:y_max[i],
                                              x_min[i]:x_max[i]] * fraction)

                        if pixelwise_error:
                            subvariance = error[y_min[i]:y_max[i],
                                                x_min[i]:x_max[i]] ** 2
                            if gain is not None:
                                subvariance += (data[y_min[i]:y_max[i],
                                                     x_min[i]:x_max[i]] /
                                                gain[y_min[i]:y_max[i],
                                                     x_min[i]:x_max[i]])
                            # Make sure variance is > 0
                            fluxvar[i] = np.max(np.sum(subvariance *
                                                       fraction), 0)
                        else:
                            local_error = error[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                                int((x_min[i] + x_max[i]) / 2 + 0.5)]
                            fluxvar[i] = np.max(local_error ** 2 *
                                                np.sum(fraction), 0)
                            if gain is not None:
                                local_gain = gain[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                                  int((x_min[i] + x_max[i]) / 2 + 0.5)]
                                fluxvar[i] += flux[i] / local_gain

    elif method == 'exact':
        from .geometry import elliptical_overlap_grid
        for i in range(len(flux)):
            x_edges = np.linspace(x_pmin[i], x_pmax[i],
                                  data[:, x_min[i]:x_max[i]].shape[1] + 1)
            y_edges = np.linspace(y_pmin[i], y_pmax[i],
                                  data[y_min[i]:y_max[i], :].shape[0] + 1)
            if flux[i] is not np.nan:
                if error is None:
                    flux[i] = np.sum(data[y_min[i]:y_max[i],
                                          x_min[i]:x_max[i]] *
                                     elliptical_overlap_grid(x_edges, y_edges,
                                                             a, b, theta))
                else:
                    fraction = elliptical_overlap_grid(x_edges, y_edges,
                                                       a, b, theta)
                    flux[i] = np.sum(data[y_min[i]:y_max[i],
                                          x_min[i]:x_max[i]] * fraction)

                    if pixelwise_error:
                        subvariance = error[y_min[i]:y_max[i],
                                            x_min[i]:x_max[i]] ** 2
                        if gain is not None:
                            subvariance += (data[y_min[i]:y_max[i],
                                                 x_min[i]:x_max[i]] /
                                            gain[y_min[i]:y_max[i],
                                                 x_min[i]:x_max[i]])
                        # Make sure variance is > 0
                        fluxvar[i] = np.max(np.sum(subvariance * fraction), 0)
                    else:
                        local_error = error[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                            int((x_min[i] + x_max[i]) / 2 + 0.5)]
                        fluxvar[i] = np.max(local_error ** 2 *
                                            np.sum(fraction), 0)
                        if gain is not None:
                            local_gain = gain[int((y_min[i] + y_max[i]) / 2 + 0.5),
                                              int((x_min[i] + x_max[i]) / 2 + 0.5)]
                            fluxvar[i] += flux[i] / local_gain

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
            fluxvar = np.max((fluxerr_o ** 2 - fluxerr_i ** 2), 0)

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
            fluxvar = np.max((fluxerr_o ** 2 - fluxerr_i ** 2), 0)

    else:
        raise ValueError('{0} mode is not supported for annular photometry'
                         '{1}'.format(mode))

    if error is None:
        flux = flux_outer[0] - flux_inner[0]
        return (flux, )
    else:
        flux = flux_outer - flux_inner
        return (flux, np.sqrt(fluxvar))

