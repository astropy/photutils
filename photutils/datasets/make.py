# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Make example datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from ..psf import GaussianPSF

__all__ = ['make_gaussian_image'
           ]


def make_gaussian_image(shape, table):
    """Make an image of Gaussian sources.

    The input table must contain parameters for one Gaussian
    source per row with column names matching the
    `~photutils.GaussianPSF` parameter names.

    Parameters
    ----------
    shape : tuple of int
        Output image shape
    table : `~astropy.table.Table`
        Gaussian source catalog

    Returns
    -------
    image : `numpy.array`
        Gaussian source model image

    Examples
    --------

    .. plot::
        :include-source:

        # Simulate a Gaussian source catalog
        from numpy.random import uniform
        from astropy.table import Table
        n_sources = 100
        shape = (100, 200) # axis order: (y, x)
        table = Table()
        table['sigma'] = uniform(1, 2, n_sources)
        table['amplitude'] = uniform(2, 3, n_sources)
        table['x_0'] = uniform(0, shape[1], n_sources)
        table['y_0'] = uniform(0, shape[0], n_sources)

        # Make an image of the sources
        from photutils.datasets import make_gaussian_image
        image = make_gaussian_image(shape, table)

        # Plot the image
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, origin='lower', interpolation='nearest')
        for source in table:
            circle = plt.Circle((source['x_0'], source['y_0']),
                                3 * source['sigma'],
                                color='white', fill=False)
            ax.add_patch(circle)
    """
    y, x = np.indices(shape)
    image = np.zeros(shape, dtype=np.float64)

    for source in table:
        model = GaussianPSF(amplitude=source['amplitude'],
                            x_0=source['x_0'],
                            y_0=source['y_0'],
                            sigma=source['sigma'],
                           )
        image += model(x, y)

    return image
