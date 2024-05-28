# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for making simulated example images for
documentation examples and tests.
"""

import numpy as np
from astropy.modeling import models
from astropy.table import QTable

from photutils.datasets import make_model_image

__all__ = ['make_4gaussians_image', 'make_100gaussians_image']


def make_4gaussians_image(noise=True):
    """
    Make an example image containing four 2D Gaussians plus a constant
    background.

    The background has a mean of 5.

    If ``noise`` is `True`, then Gaussian noise with a mean of 0 and a
    standard deviation of 5 is added to the output image.

    Parameters
    ----------
    noise : bool, optional
        Whether to include noise in the output image (default is
        `True`).

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing four 2D Gaussian sources.

    See Also
    --------
    make_100gaussians_image

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.datasets import make_4gaussians_image

        image = make_4gaussians_image()
        plt.imshow(image, origin='lower', interpolation='nearest')
    """
    params = QTable()
    params['amplitude'] = [50, 70, 150, 210]
    params['x_mean'] = [160, 25, 150, 90]
    params['y_mean'] = [70, 40, 25, 60]
    params['x_stddev'] = [15.2, 5.1, 3.0, 8.1]
    params['y_stddev'] = [2.6, 2.5, 3.0, 4.7]
    params['theta'] = np.radians(np.array([145.0, 20.0, 0.0, 60.0]))

    shape = (100, 200)
    model = models.Gaussian2D()
    data = make_model_image(shape, model, params, xname='x_mean', yname='y_mean')

    data += 5.0  # background

    if noise:
        rng = np.random.default_rng(seed=0)
        data += rng.normal(loc=0.0, scale=5.0, size=shape)

    return data


def make_100gaussians_image(noise=True):
    """
    Make an example image containing 100 2D Gaussians plus a constant
    background.

    The background has a mean of 5.

    If ``noise`` is `True`, then Gaussian noise with a mean of 0 and a
    standard deviation of 2 is added to the output image.

    Parameters
    ----------
    noise : bool, optional
        Whether to include noise in the output image (default is
        `True`).

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing 100 2D Gaussian sources.

    See Also
    --------
    make_4gaussians_image

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.datasets import make_100gaussians_image

        image = make_100gaussians_image()
        plt.imshow(image, origin='lower', interpolation='nearest')
    """
    n_sources = 100
    flux_range = [500, 1000]
    xmean_range = [0, 500]
    ymean_range = [0, 300]
    xstddev_range = [1, 5]
    ystddev_range = [1, 5]
    params = {'flux': flux_range,
              'x_mean': xmean_range,
              'y_mean': ymean_range,
              'x_stddev': xstddev_range,
              'y_stddev': ystddev_range,
              'theta': [0, 2 * np.pi]}

    rng = np.random.RandomState(12345)
    sources = QTable()
    for param_name, (lower, upper) in params.items():
        # Generate a column for every item in param_ranges, even if it
        # is not in the model (e.g., flux). However, such columns will
        # be ignored when rendering the image.
        sources[param_name] = rng.uniform(lower, upper, n_sources)
    xstd = sources['x_stddev']
    ystd = sources['y_stddev']
    sources['amplitude'] = sources['flux'] / (2.0 * np.pi * xstd * ystd)

    shape = (300, 500)
    model = models.Gaussian2D()
    data = make_model_image(shape, model, sources, bbox_factor=6.0, xname='x_mean',
                            yname='y_mean')
    data += 5.0  # background

    if noise:
        rng = np.random.RandomState(12345)
        data += rng.normal(loc=0.0, scale=2.0, size=shape)

    return data
