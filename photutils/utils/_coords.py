# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for generating random (x, y) coordinates.
"""

import warnings
from collections import defaultdict

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

    # create a dictionary of nearest neighbors (within min_separation)
    nn = {}
    nn = defaultdict(set)
    for i, j in pairs:
        nn[i].add(j)
        nn[j].add(i)

    keep_idx = []
    discard_idx = set()
    for idx in range(xycoords.shape[0]):
        if idx not in discard_idx:
            keep_idx.append(idx)
            # remove nearest neighbors from the output
            discard_idx.update(nn.get(idx, set()))

    return xycoords[keep_idx]


def make_random_xycoords(size, x_range, y_range, min_separation=0.0,
                         seed=None):
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

    Returns
    -------
    xycoords : `~numpy.ndarray`
        The (x, y) random coordinates with shape ``(size, 2)``.
    """
    ncoords = size
    if min_separation > 0:
        # scale the number of random coordinates to account for
        # some being discarded due to min_separation
        ncoords *= 10

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
