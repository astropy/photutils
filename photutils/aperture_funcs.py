# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for performing aperture photometry on 2-D arrays."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import warnings
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning


__all__ = []


def get_phot_extents(data, positions, extents):
    """
    Get the photometry extents and check if the apertures is fully out of data.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.

    Returns
    -------
    extents : dict
        The ``extents`` dictionary contains 3 elements:

        * ``'ood_filter'``
            A boolean array with `True` elements where the aperture is
            falling out of the data region.
        * ``'pixel_extent'``
            x_min, x_max, y_min, y_max : Refined extent of apertures with
            data coverage.
        * ``'phot_extent'``
            x_pmin, x_pmax, y_pmin, y_pmax: Extent centered to the 0, 0
            positions as required by the `~photutils.geometry` functions.
    """

    # Check if an aperture is fully out of data
    ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                               extents[:, 1] <= 0)
    np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                  out=ood_filter)
    np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

    # TODO check whether it makes sense to have negative pixel
    # coordinate, one could imagine a stackes image where the reference
    # was a bit offset from some of the images? Or in those cases just
    # give Skycoord to the Aperture and it should deal with the
    # conversion for the actual case?
    x_min = np.maximum(extents[:, 0], 0)
    x_max = np.minimum(extents[:, 1], data.shape[1])
    y_min = np.maximum(extents[:, 2], 0)
    y_max = np.minimum(extents[:, 3], data.shape[0])

    x_pmin = x_min - positions[:, 0] - 0.5
    x_pmax = x_max - positions[:, 0] - 0.5
    y_pmin = y_min - positions[:, 1] - 0.5
    y_pmax = y_max - positions[:, 1] - 0.5

    # TODO: check whether any pixel is nan in data[y_min[i]:y_max[i],
    # x_min[i]:x_max[i])), if yes return something valid rather than nan

    pixel_extent = [x_min, x_max, y_min, y_max]
    phot_extent = [x_pmin, x_pmax, y_pmin, y_pmax]

    return ood_filter, pixel_extent, phot_extent


def find_fluxvar(data, fraction, error, flux, effective_gain, imin, imax,
                 jmin, jmax, pixelwise_error):

    if isinstance(error, u.Quantity):
        zero_variance = 0 * error.unit**2
    else:
        zero_variance = 0

    if pixelwise_error:

        subvariance = error[jmin:jmax,
                            imin:imax] ** 2

        if effective_gain is not None:
            subvariance += (data[jmin:jmax, imin:imax] /
                            effective_gain[jmin:jmax, imin:imax])

        # Make sure variance is > 0
        fluxvar = np.maximum(np.sum(subvariance * fraction), zero_variance)

    else:

        local_error = error[int((jmin + jmax) / 2 + 0.5),
                            int((imin + imax) / 2 + 0.5)]

        fluxvar = np.maximum(local_error ** 2 * np.sum(fraction),
                             zero_variance)

        if effective_gain is not None:
            local_effective_gain = effective_gain[
                int((jmin + jmax) / 2 + 0.5), int((imin + imax) / 2 + 0.5)]
            fluxvar += flux / local_effective_gain

    return fluxvar


def do_circular_photometry(data, positions, radius, error, effective_gain,
                           pixelwise_error, method, subpixels, r_in=None):

    extents = np.zeros((len(positions), 4), dtype=int)

    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5

    ood_filter, extent, phot_extent = get_phot_extents(data, positions,
                                                       extents)

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

                fluxvar[i] = find_fluxvar(data, fraction, error, flux[i],
                                          effective_gain, x_min[i], x_max[i],
                                          y_min[i], y_max[i], pixelwise_error)

    if error is None:
        return (flux, )
    else:
        return (flux, np.sqrt(fluxvar))


def do_elliptical_photometry(data, positions, a, b, theta, error,
                             effective_gain, pixelwise_error, method,
                             subpixels, a_in=None):

    extents = np.zeros((len(positions), 4), dtype=int)

    # TODO: we can be more efficient in terms of bounding box
    radius = max(a, b)

    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5

    ood_filter, extent, phot_extent = get_phot_extents(data, positions,
                                                       extents)

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
                                               a, b, theta, use_exact,
                                               subpixels)

            if a_in is not None:
                b_in = a_in * b / a
                fraction -= elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                                    y_pmin[i], y_pmax[i],
                                                    x_max[i] - x_min[i],
                                                    y_max[i] - y_min[i],
                                                    a_in, b_in, theta,
                                                    use_exact, subpixels)

            flux[i] = np.sum(data[y_min[i]:y_max[i],
                                  x_min[i]:x_max[i]] * fraction)

            if error is not None:
                fluxvar[i] = find_fluxvar(data, fraction, error, flux[i],
                                          effective_gain, x_min[i], x_max[i],
                                          y_min[i], y_max[i], pixelwise_error)

    if error is None:
        return (flux, )
    else:
        return (flux, np.sqrt(fluxvar))


def do_rectangular_photometry(data, positions, w, h, theta, error,
                              effective_gain, pixelwise_error, method,
                              subpixels, reduce='sum', w_in=None):

    extents = np.zeros((len(positions), 4), dtype=int)

    # TODO: this is an overestimate by up to sqrt(2) unless theta = 45 deg
    radius = max(h, w) * (2 ** -0.5)

    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5

    ood_filter, extent, phot_extent = get_phot_extents(data, positions,
                                                       extents)

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

        from .geometry import rectangular_overlap_grid

        for i in range(len(flux)):
            if not np.isnan(flux[i]):

                fraction = rectangular_overlap_grid(x_pmin[i], x_pmax[i],
                                                    y_pmin[i], y_pmax[i],
                                                    x_max[i] - x_min[i],
                                                    y_max[i] - y_min[i],
                                                    w, h, theta, 0, subpixels)
                if w_in is not None:
                    h_in = w_in * h / w
                    fraction -= rectangular_overlap_grid(x_pmin[i], x_pmax[i],
                                                         y_pmin[i], y_pmax[i],
                                                         x_max[i] - x_min[i],
                                                         y_max[i] - y_min[i],
                                                         w_in, h_in, theta,
                                                         0, subpixels)

                flux[i] = np.sum(data[y_min[i]:y_max[i],
                                      x_min[i]:x_max[i]] * fraction)
                if error is not None:
                    fluxvar[i] = find_fluxvar(data, fraction, error,
                                              flux[i], effective_gain,
                                              x_min[i], x_max[i],
                                              y_min[i], y_max[i],
                                              pixelwise_error)

    if error is None:
        return (flux, )
    else:
        return (flux, np.sqrt(fluxvar))
