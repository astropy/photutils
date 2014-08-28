# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Make example datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from ..psf import GaussianPSF
from astropy.table import Table
from astropy.modeling.models import Gaussian2D


__all__ = ['make_gaussian_sources', 'make_random_gaussians']


def make_gaussian_sources(image_shape, source_table, noise_stddev=None,
                          noise_lambda=None, seed=None):
    """
    Make an image containing 2D Gaussian sources with optional Gaussian
    or Poisson noise.

    Parameters
    ----------
    image_shape : 2-tuple of int
        Shape of the output 2D image.

    source_table : `astropy.table.Table`
        Table of parameters for the Gaussian sources.  Each row of the
        table corresponds to a Gaussian source whose parameters are
        defined by the column names, which must match the
        `astropy.modeling.functional_models.Gaussian2D` parameter names.

    noise_stddev : float, optional
        The standard deviation of the Gaussian noise to add to the
        output image.  The default is `None`, meaning no Gaussian noise
        will be added.  ``noise_stddev`` and ``noise_lambda`` should not
        both be set.

    noise_lambda : positive float, optional
        The expectation value of the Poisson noise to add to the output
        image.  The default is `None`, meaning no Poisson noise will be
        added.  ``noise_stddev`` and ``noise_lambda`` should not both be
        set.

    seed : int, or array_like, optional
        Random seed initializing the pseudo-random number generator used
        to generate the noise image.  ``seed`` can be an integer or an
        array (or other sequence) of integers of any length.  Separate
        function calls with the same noise parameters and ``seed`` will
        generate the identical noise image.  If ``seed`` is `None`, then
        a new random noise image will be generated each time.

    Returns
    -------
    image : `numpy.ndarray`
        Image containing 2D Gaussian sources and optional noise.

    Examples
    --------

    .. plot::
        :include-source:

        # make a table of Gaussian sources
        from astropy.table import Table
        table = Table()
        table['amplitude'] = [50, 70, 150, 210]
        table['x_mean'] = [160, 25, 150, 90]
        table['y_mean'] = [70, 40, 25, 60]
        table['x_stddev'] = [15.2, 5.1, 3., 8.1]
        table['y_stddev'] = [2.6, 2.5, 3., 4.7]
        table['theta'] = np.array([145., 20., 0., 60.]) * np.pi / 180.

        # make an image of the sources without noise, with Gaussian
        # noise, and with Poisson noise
        from photutils.datasets import make_gaussian_sources
        shape = (100, 200)
        image1 = make_gaussian_sources(shape, table)
        image2 = make_gaussian_sources(shape, table, noise_stddev=5.)
        image3 = make_gaussian_sources(shape, table, noise_lambda=5.)

        # plot the images
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        ax1.imshow(image1, origin='lower', interpolation='nearest')
        ax2.imshow(image2, origin='lower', interpolation='nearest')
        ax3.imshow(image3, origin='lower', interpolation='nearest')
    """

    image = np.zeros(image_shape, dtype=np.float64)
    y, x = np.indices(image_shape)

    for source in source_table:
        model = Gaussian2D(amplitude=source['amplitude'],
                           x_mean=source['x_mean'], y_mean=source['y_mean'],
                           x_stddev=source['x_stddev'],
                           y_stddev=source['y_stddev'], theta=source['theta'])
        image += model(x, y)

    if seed:
        prng = np.random.RandomState(seed)
    else:
        prng = np.random
    if noise_stddev is not None:
        image += prng.normal(loc=0.0, scale=noise_stddev, size=image_shape)
    if noise_lambda is not None:
        image += prng.poisson(lam=noise_lambda, size=image_shape)
    return image


def make_random_gaussians(image_shape, n_sources, amplitude_range,
                          xstddev_range, ystddev_range, noise_stddev=None,
                          noise_lambda=None, seed=None, output_table=False):
    """
    Make an image containing random 2D Gaussian sources, whose
    parameters are drawn from a uniform distribution, with optional
    Gaussian or Poisson noise.

    Parameters
    ----------
    image_shape : 2-tuple of int
        Shape of the output 2D image.

    n_sources : float
        The number of random Gaussian sources to add to the image.

    amplitude_range : array-like
        The lower and upper boundaries input as ``(lower, upper)`` over
        which draw source amplitudes from a uniform distribution.

    xstddev_range : array-like
        The lower and upper boundaries input as ``(lower, upper)`` over
        which draw source x_stddev from a uniform distribution.

    ystddev_range : array-like
        The lower and upper boundaries input as ``(lower, upper)`` over
        which draw source y_stddev from a uniform distribution.

    noise_stddev : float, optional
        The standard deviation of the Gaussian noise to add to the
        output image.  The default is `None`, meaning no Gaussian noise
        will be added.  ``noise_stddev`` and ``noise_lambda`` should not
        both be set.

    noise_lambda : positive float, optional
        The expectation value of the Poisson noise to add to the output
        image.  The default is `None`, meaning no Poisson noise will be
        added.  ``noise_stddev`` and ``noise_lambda`` should not both be
        set.

    seed : int, or array_like, optional
        Random seed initializing the pseudo-random number generator used
        to generate the noise image.  ``seed`` can be an integer or an
        array (or other sequence) of integers of any length.  Separate
        function calls with the same noise parameters and ``seed`` will
        generate the identical noise image.  If ``seed`` is `None`, then
        a new random noise image will be generated each time.

    output_table : bool, optional
        Set to return a table of parameters for the randomly generated
        Gaussian sources.

    Returns
    -------
    image : `numpy.ndarray`
        Image containing 2D Gaussian sources and optional noise.

    table : `astropy.table.Table`
        A table of parameters for the randomly generated Gaussian
        sources.  Each row of the table corresponds to a Gaussian source
        whose parameters are defined by the column names, which must
        match the `astropy.modeling.functional_models.Gaussian2D`
        parameter names.  Returned *only* if ``output_table`` is `True`.

    Examples
    --------

    .. plot::
        :include-source:

        from photutils.datasets import make_random_gaussians
        shape = (300, 500)
        n_sources = 100
        amplitude_range = [50, 100]
        xstddev_range = [1, 5]
        ystddev_range = [1, 5]

        # make an image of random sources without noise, with Gaussian
        # noise, and with Poisson noise.  Note that "seed" is used here
        # to generate the same random sources across function calls.
        image1 = make_random_gaussians(shape, n_sources, amplitude_range,
                                       xstddev_range, ystddev_range,
                                       seed=12345)
        image2 = make_random_gaussians(shape, n_sources, amplitude_range,
                                       xstddev_range, ystddev_range,
                                       noise_stddev=5., seed=12345)
        image3 = make_random_gaussians(shape, n_sources, amplitude_range,
                                       xstddev_range, ystddev_range,
                                       noise_lambda=5., seed=12345)

        # plot the images
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        ax1.imshow(image1, origin='lower', interpolation='nearest')
        ax2.imshow(image2, origin='lower', interpolation='nearest')
        ax3.imshow(image3, origin='lower', interpolation='nearest')
    """

    if seed:
        prng = np.random.RandomState(seed)
    else:
        prng = np.random
    sources = Table()
    sources['amplitude'] = prng.uniform(amplitude_range[0],
                                        amplitude_range[1], n_sources)
    sources['x_mean'] = prng.uniform(0, image_shape[1], n_sources)
    sources['y_mean'] = prng.uniform(0, image_shape[0], n_sources)
    sources['x_stddev'] = prng.uniform(xstddev_range[0], xstddev_range[1],
                                       n_sources)
    sources['y_stddev'] = prng.uniform(ystddev_range[0], ystddev_range[1],
                                       n_sources)
    sources['theta'] = prng.uniform(0, 2.*np.pi, n_sources)
    image = make_gaussian_sources(image_shape, sources,
                                  noise_stddev=noise_stddev,
                                  noise_lambda=noise_lambda, seed=seed)
    if output_table:
        return image, sources
    else:
        return image
