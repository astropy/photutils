# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for making tables of sources with random
model parameters.
"""

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from astropy.utils.decorators import deprecated

from photutils.utils._misc import _get_meta

__all__ = ['make_random_models_table', 'make_random_gaussians_table']


def make_random_models_table(n_sources, param_ranges, seed=None):
    """
    Make a `~astropy.table.QTable` containing randomly generated
    parameters for an Astropy model to simulate a set of sources.

    Each row of the table corresponds to a source whose parameters are
    defined by the column names. The parameters are drawn from a uniform
    distribution over the specified input ranges.

    The output table can be input into :func:`make_model_sources_image`
    to create an image containing the model sources.

    Parameters
    ----------
    n_sources : float
        The number of random model sources to generate.

    param_ranges : dict
        The lower and upper boundaries for each of the model parameters
        as a dictionary mapping the parameter name to its ``(lower,
        upper)`` bounds.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of parameters for the randomly generated sources. Each
        row of the table corresponds to a source whose model parameters
        are defined by the column names. The column names will be the
        keys of the dictionary ``param_ranges``.

    See Also
    --------
    make_random_gaussians_table

    Notes
    -----
    To generate identical parameter values from separate function
    calls, ``param_ranges`` must have the same parameter ranges and the
    ``seed`` must be the same.

    Examples
    --------
    >>> from photutils.datasets import make_random_models_table
    >>> n_sources = 5
    >>> param_ranges = {'amplitude': [500, 1000],
    ...                 'x_mean': [0, 500],
    ...                 'y_mean': [0, 300],
    ...                 'x_stddev': [1, 5],
    ...                 'y_stddev': [1, 5],
    ...                 'theta': [0, np.pi]}
    >>> sources = make_random_models_table(n_sources, param_ranges,
    ...                                    seed=0)
    >>> for col in sources.colnames:
    ...     sources[col].info.format = '%.8g'  # for consistent table output
    >>> print(sources)
    amplitude   x_mean    y_mean    x_stddev  y_stddev   theta
    --------- --------- ---------- --------- --------- ---------
    818.48084 456.37779  244.75607 1.7026225 1.1132787 1.2053586
    634.89336 303.31789 0.82155005 4.4527157 1.4971331 3.1328274
    520.48676 364.74828  257.22128 3.1658449 3.6824977 3.0813851
    508.26382  271.8125  10.075673 2.1988476  3.588758 2.1536937
    906.63512 467.53621  218.89663 2.6907489 3.4615404 2.0434781
    """
    rng = np.random.default_rng(seed)

    sources = QTable()
    sources.meta.update(_get_meta())  # keep sources.meta type
    for param_name, (lower, upper) in param_ranges.items():
        # Generate a column for every item in param_ranges, even if it
        # is not in the model (e.g., flux).
        sources[param_name] = rng.uniform(lower, upper, n_sources)

    return sources


@deprecated('1.13.0', alternative='make_random_models_table')
def make_random_gaussians_table(n_sources, param_ranges, seed=None):
    """
    Make a `~astropy.table.QTable` containing randomly generated
    parameters for 2D Gaussian sources.

    Each row of the table corresponds to a Gaussian source whose
    parameters are defined by the column names. The parameters are drawn
    from a uniform distribution over the specified input ranges.

    The output table will contain columns for both the Gaussian
    amplitude and flux.

    The output table can be input into
    :func:`make_gaussian_sources_image` to create an image containing
    the 2D Gaussian sources.

    Parameters
    ----------
    n_sources : float
        The number of random 2D Gaussian sources to generate.

    param_ranges : dict
        The lower and upper boundaries for each of the
        `~astropy.modeling.functional_models.Gaussian2D` parameters
        as a dictionary mapping the parameter name to its ``(lower,
        upper)`` bounds. The dictionary keys must be valid
        `~astropy.modeling.functional_models.Gaussian2D` parameter
        names or ``'flux'``. If ``'flux'`` is specified, but not
        ``'amplitude'`` then the 2D Gaussian amplitudes will be
        calculated and placed in the output table. If ``'amplitude'``
        is specified, then the 2D Gaussian fluxes will be calculated
        and placed in the output table. If both ``'flux'`` and
        ``'amplitude'`` are specified, then ``'flux'`` will be
        recalculated and overwritten. Model parameters not defined in
        ``param_ranges`` will be set to the default value.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of parameters for the randomly generated Gaussian
        sources. Each row of the table corresponds to a Gaussian source
        whose parameters are defined by the column names.

    See Also
    --------
    make_random_models_table

    Notes
    -----
    To generate identical parameter values from separate function
    calls, ``param_ranges`` must have the same parameter ranges and the
    ``seed`` must be the same.
    """
    sources = make_random_models_table(n_sources, param_ranges,
                                       seed=seed)
    model = Gaussian2D()

    # compute Gaussian2D amplitude to flux conversion factor
    if 'x_stddev' in sources.colnames:
        xstd = sources['x_stddev']
    else:
        xstd = model.x_stddev.value  # default
    if 'y_stddev' in sources.colnames:
        ystd = sources['y_stddev']
    else:
        ystd = model.y_stddev.value  # default
    gaussian_amplitude_to_flux = 2.0 * np.pi * xstd * ystd

    if 'amplitude' in param_ranges:
        sources['flux'] = sources['amplitude'] * gaussian_amplitude_to_flux

    if 'flux' in param_ranges and 'amplitude' not in param_ranges:
        sources['amplitude'] = sources['flux'] / gaussian_amplitude_to_flux

    return sources
