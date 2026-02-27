# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for generating random (x, y) coordinates.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from scipy.spatial import KDTree


def apply_separation(xycoords, min_separation):
    """
    Apply a minimum separation to a set of (x, y) coordinates.

    Coordinates that are closer than the minimum separation are removed.

    Parameters
    ----------
    xycoords : `~numpy.ndarray`
        The (x, y) coordinates with shape ``(N, 2)``.

    min_separation : float
        The minimum separation in pixels between coordinates.

    Returns
    -------
    xycoords : `~numpy.ndarray`
        The (x, y) coordinates with shape ``(N, 2)`` after excluding
        points closer than the minimum separation.
    """
    tree = KDTree(xycoords)
    pairs = tree.query_pairs(min_separation, output_type='ndarray')

    if len(pairs) == 0:
        return xycoords

    n = xycoords.shape[0]
    keep = np.ones(n, dtype=bool)

    # Group pairs by first index for vectorized neighbor removal. Each
    # pair has i < j (guaranteed by KDTree). Process groups in ascending
    # first-index order (greedy independent set algorithm): for each
    # kept point, discard all its higher-index neighbors.
    sorted_idx = pairs[:, 0].argsort(kind='stable')
    pairs_sorted = pairs[sorted_idx]
    unique_i, group_start = np.unique(pairs_sorted[:, 0],
                                      return_index=True)
    group_end = np.append(group_start[1:], len(pairs_sorted))

    for k in range(len(unique_i)):
        i = unique_i[k]
        if keep[i]:
            js = pairs_sorted[group_start[k]:group_end[k], 1]
            keep[js] = False

    return xycoords[keep]


def make_random_xycoords(size, x_range, y_range, min_separation=0.0,
                         seed=None, oversample=10):
    """
    Make random (x, y) coordinates.

    Parameters
    ----------
    size : int
        The number of coordinates to generate.

    x_range : tuple
        The range of x values (min, max).

    y_range : tuple
        The range of y values (min, max).

    min_separation : float, optional
        The minimum separation in pixels between coordinates.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    oversample : int, optional
        The oversampling factor used when ``min_separation`` > 0 to
        generate extra candidate coordinates before filtering by
        separation. Higher values increase the chance of producing the
        requested number of coordinates in crowded conditions, at the
        cost of speed. The default is 10.

    Returns
    -------
    xycoords : `~numpy.ndarray`
        The (x, y) random coordinates with shape ``(size, 2)``. When
        ``min_separation`` > 0, fewer than ``size`` coordinates may be
        returned if the range and separation cannot be satisfied. A
        warning is issued in that case.
    """
    if size == 0:
        return np.empty((0, 2))

    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        msg = 'x_range and y_range must be (min, max) with min < max.'
        raise ValueError(msg)

    ncoords = size
    if min_separation > 0:
        # Scale the number of random coordinates to account for
        # some being discarded due to min_separation
        ncoords *= oversample

    rng = np.random.default_rng(seed)
    xc = rng.uniform(x_range[0], x_range[1], ncoords)
    yc = rng.uniform(y_range[0], y_range[1], ncoords)
    xycoords = np.transpose(np.array((xc, yc)))

    xycoords = apply_separation(xycoords, min_separation)
    xycoords = xycoords[:size]

    if len(xycoords) < size:
        warnings.warn(f'Unable to produce {size!r} coordinates within the '
                      'given shape and minimum separation. Only '
                      f'{len(xycoords)!r} coordinates were generated.',
                      AstropyUserWarning)

    return xycoords
