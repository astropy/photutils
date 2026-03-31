# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for generating matplotlib colormaps.

This module requires matplotlib to be installed.
"""

import numpy as np

from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)

__all__ = ['make_random_cmap']


@deprecated_renamed_argument('ncolors', 'n_colors', '3.0', until='4.0')
@deprecated_positional_kwargs(since='3.0', until='4.0')
def make_random_cmap(n_colors=256, seed=None):
    """
    Make a matplotlib colormap consisting of (random) muted colors.

    A random colormap is very useful for plotting segmentation images.

    Parameters
    ----------
    n_colors : int, optional
        The number of colors in the colormap. The default is 256. Must
        be at least 1.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.
        Separate function calls with the same ``seed`` will generate the
        same colormap.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
        The matplotlib colormap with random colors in RGBA format.
    """
    if n_colors < 1:
        msg = 'n_colors must be at least 1'
        raise ValueError(msg)

    from matplotlib import colors

    rng = np.random.default_rng(seed)
    hue = rng.uniform(low=0.0, high=1.0, size=n_colors)
    sat = rng.uniform(low=0.2, high=0.7, size=n_colors)
    val = rng.uniform(low=0.5, high=1.0, size=n_colors)
    hsv = np.dstack((hue, sat, val))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    return colors.ListedColormap(colors.to_rgba_array(rgb))
