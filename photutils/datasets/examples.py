# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making simulated example images for documentation examples and
tests.
"""

import pathlib

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from astropy.utils.data import get_pkg_data_path

from photutils.datasets import make_model_image

__all__ = ['make_4gaussians_image', 'make_100gaussians_image']

_DATASETS_DATA_DIR = pathlib.Path(get_pkg_data_path('datasets', 'data',
                                                    package='photutils'))


def _make_gaussians_image(filename, shape, noise, noise_stddev,
                          **kwargs):
    """
    Make an image containing 2D Gaussians from a parameter file.

    Parameters
    ----------
    filename : str
        The name of the ECSV file containing the Gaussian parameters.

    shape : 2-tuple of int
        The shape of the output image.

    noise : bool
        Whether to include noise in the output image.

    noise_stddev : float
        The standard deviation of the Gaussian noise.

    **kwargs
        Additional keyword arguments passed to `make_model_image`.

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing 2D Gaussian sources.
    """
    model = Gaussian2D()
    params = QTable.read(_DATASETS_DATA_DIR / filename,
                         format='ascii.ecsv')
    data = make_model_image(shape, model, params, x_name='x_mean',
                            y_name='y_mean', **kwargs)
    data += 5.0  # background

    if noise:
        rng = np.random.default_rng(seed=0)
        data += rng.normal(loc=0.0, scale=noise_stddev, size=shape)

    return data


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
    return _make_gaussians_image('4gaussians_params.ecsv', (100, 200),
                                 noise, noise_stddev=5.0)


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
    return _make_gaussians_image('100gaussians_params.ecsv', (300, 500),
                                 noise, noise_stddev=2.0, bbox_factor=6.0)
