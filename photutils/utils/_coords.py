# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for generating random (x, y) coordinates.
"""
import warnings
from collections import defaultdict

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning


def apply_separation(xycoords, min_separation):
    from scipy.spatial import KDTree

    tree = KDTree(xycoords)
    pairs = tree.query_pairs(min_separation, output_type='ndarray')

    # create a dictionary of nearest neighbors (within min_separation)
    nn = {}
    nn = defaultdict(set)
    for p1, p2 in xycoords[pairs]:
        nn[tuple(p1)].add(tuple(p2))
        nn[tuple(p2)].add(tuple(p1))

    keep = []
    discard = set()
    for xycoord in xycoords:
        xycoord = tuple(xycoord)
        if xycoord not in discard:
            keep.append(xycoord)
            # remove nearest neighbors from the output
            neighbors = nn.get(xycoord, None)
            if neighbors is not None:
                discard.update(neighbors)

    return np.array(keep)


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
    if min_separation > 0:
        # scale the number of random coordinates to account for
        # some being discarded due to min_separation
        ncoords = size * 10

    rng = np.random.default_rng(seed)
    xc = rng.uniform(x_range[0], x_range[1], ncoords)
    yc = rng.uniform(y_range[0], y_range[1], ncoords)
    xycoords = np.transpose(np.array((xc, yc)))

    xycoords = apply_separation(xycoords, min_separation)
    xycoords = xycoords[:size]

    if len(xycoords) < size:
        warnings.warn(f'Unable to produce {size!r} coordinates.',
                      AstropyUserWarning)

    return xycoords
