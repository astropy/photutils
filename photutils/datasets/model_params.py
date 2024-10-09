# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for making tables of model parameters or
making models from a table of model parameters.
"""

import numpy as np
from astropy.table import QTable

from photutils.utils._coords import make_random_xycoords
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import as_pair

__all__ = ['make_model_params', 'make_random_models_table',
           'params_table_to_models']


def make_model_params(shape, n_sources, *, x_name='x_0', y_name='y_0',
                      min_separation=1, border_size=(0, 0), seed=0, **kwargs):
    """
    Make a table of randomly generated model positions and additional
    parameters for simulated sources.

    By default, this function computes only a table of x_0 and y_0
    values. Additional parameters can be specified as keyword arguments
    with their lower and upper bounds as 2-tuples. The parameter values
    will be uniformly distributed between the lower and upper bounds,
    inclusively.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output image.

    n_sources : int
        The number of sources to generate. If ``min_separation`` is too
        large, the number of requested sources may not fit within the
        given ``shape`` and therefore the number of sources generated
        may be less than ``n_sources``.

    x_name : str, optional
        The name of the ``model`` parameter that corresponds to the x
        position of the sources. This will be the column name in the
        output table.

    y_name : str, optional
        The name of the ``model`` parameter that corresponds to the y
        position of the sources. This will be the column name in the
        output table.

    min_separation : float, optional
        The minimum separation between the centers of two sources. Note
        that if the minimum separation is too large, the number of
        sources generated may be less than ``n_sources``.

    border_size : tuple of 2 int or int, optional
        The (ny, nx) size of the border around the image where no
        sources will be generated (i.e., the source center will not be
        located within the border). If a single integer is provided, it
        will be used for both dimensions.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    **kwargs
        Keyword arguments are accepted for additional model parameters.
        The values should be 2-tuples of the lower and upper bounds for
        the parameter range. The parameter values will be uniformly
        distributed between the lower and upper bounds, inclusively.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table containing the model parameters of the generated
        sources. The table will also contain an ``'id'`` column with
        unique source IDs.

    Examples
    --------
    >>> from photutils.datasets import make_model_params
    >>> params = make_model_params((100, 100), 5, flux=(100, 500),
    ...                            min_separation=3, border_size=10, seed=0)
    >>> for col in params.colnames:
    ...     params[col].info.format = '%.8g'  # for consistent table output
    >>> print(params)
     id    x_0       y_0       flux
    --- --------- --------- ---------
      1 60.956935 72.967865 291.99517
      2 31.582937 29.149555 192.94917
      3 13.277882 80.118738 420.75223
      4 11.322211 14.685443 469.41206
      5 75.061619 36.889365 206.45211

    >>> params = make_model_params((100, 100), 5, flux=(100, 500),
    ...                            x_name='x_mean', y_name='y_mean',
    ...                            min_separation=3, border_size=10, seed=0)
    >>> for col in params.colnames:
    ...     params[col].info.format = '%.8g'  # for consistent table output
    >>> print(params)
     id   x_mean    y_mean     flux
    --- --------- --------- ---------
      1 60.956935 72.967865 291.99517
      2 31.582937 29.149555 192.94917
      3 13.277882 80.118738 420.75223
      4 11.322211 14.685443 469.41206
      5 75.061619 36.889365 206.45211

    >>> params = make_model_params((100, 100), 5, flux=(100, 500),
    ...                            sigma=(1, 2), alpha=(0, 1),
    ...                            min_separation=3, border_size=10, seed=0)
    >>> for col in params.colnames:
    ...     params[col].info.format = '%.5g'  # for consistent table output
    >>> print(params)
     id  x_0    y_0    flux  sigma   alpha
    --- ------ ------ ------ ------ --------
      1 60.957 72.968    292 1.5389  0.61437
      2 31.583  29.15 192.95 1.4428 0.028365
      3 13.278 80.119 420.75  1.931  0.71922
      4 11.322 14.685 469.41 1.0405 0.015992
      5 75.062 36.889 206.45  1.732  0.75795
    """
    shape = as_pair('shape', shape, lower_bound=(0, 1))
    border_size = as_pair('border_size', border_size, lower_bound=(0, 0))

    xrange = (border_size[1], shape[1] - border_size[1])
    yrange = (border_size[0], shape[0] - border_size[0])

    if xrange[0] >= xrange[1] or yrange[0] >= yrange[1]:
        raise ValueError('border_size is too large for the given shape')

    rng = np.random.default_rng(seed)
    xycoords = make_random_xycoords(n_sources, xrange, yrange,
                                    min_separation=min_separation,
                                    seed=rng)
    x, y = np.transpose(xycoords)

    model_params = QTable()
    model_params['id'] = np.arange(len(x)) + 1
    model_params[x_name] = x
    model_params[y_name] = y

    for param, prange in kwargs.items():
        if len(prange) != 2:
            raise ValueError(f'{param} must be a 2-tuple')
        vals = rng.uniform(*prange, len(model_params))
        model_params[param] = vals

    return model_params


def make_random_models_table(n_sources, param_ranges, seed=None):
    """
    Make a `~astropy.table.QTable` containing randomly generated
    parameters for an Astropy model to simulate a set of sources.

    Each row of the table corresponds to a source whose parameters are
    defined by the column names. The parameters are drawn from a uniform
    distribution over the specified input ranges, inclusively.

    The output table can be input into :func:`make_model_image` to
    create an image containing the model sources.

    Parameters
    ----------
    n_sources : float
        The number of random model sources to generate.

    param_ranges : dict
        The lower and upper boundaries for each of the model parameters
        as a dictionary mapping the parameter name to its ``(lower,
        upper)`` bounds. The parameter values will be uniformly
        distributed between these bounds, inclusively.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of parameters for the randomly generated sources. Each
        row of the table corresponds to a source whose model parameters
        are defined by the column names. The column names will be the
        keys of the dictionary ``param_ranges``. The table will also
        contain an ``'id'`` column with unique source IDs.

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
    >>> params = make_random_models_table(n_sources, param_ranges, seed=0)
    >>> for col in params.colnames:
    ...     params[col].info.format = '%.8g'  # for consistent table output
    >>> print(params)
     id amplitude   x_mean    y_mean    x_stddev  y_stddev   theta
    --- --------- --------- ---------- --------- --------- ---------
      1 818.48084 456.37779  244.75607 1.7026225 1.1132787 1.2053586
      2 634.89336 303.31789 0.82155005 4.4527157 1.4971331 3.1328274
      3 520.48676 364.74828  257.22128 3.1658449 3.6824977 3.0813851
      4 508.26382  271.8125  10.075673 2.1988476  3.588758 2.1536937
      5 906.63512 467.53621  218.89663 2.6907489 3.4615404 2.0434781
    """
    rng = np.random.default_rng(seed)

    sources = QTable()
    sources.meta.update(_get_meta())  # keep sources.meta type
    sources['id'] = np.arange(n_sources) + 1
    for param_name, (lower, upper) in param_ranges.items():
        # Generate a column for every item in param_ranges, even if it
        # is not in the model (e.g., flux).
        sources[param_name] = rng.uniform(lower, upper, n_sources)

    return sources


def params_table_to_models(params_table, model):
    """
    Create a list of models from a table of model parameters.

    Parameters
    ----------
    params_table : `~astropy.table.Table`
        A table containing the model parameters for each source.
        Each row of the table corresponds to a different model whose
        parameters are defined by the column names. Model parameters not
        defined in the table will be set to the ``model`` default value.
        To attach units to model parameters, ``params_table`` must be
        input as a `~astropy.table.QTable`. A column named 'name' can
        also be included in the table to assign a name to each model.

    model : `astropy.modeling.Model`
        The model whose parameters will be updated.

    Returns
    -------
    models : list of `astropy.modeling.Model`
        A list of models created from the input table of model
        parameters.

    Examples
    --------
    >>> from astropy.table import QTable
    >>> from photutils.datasets import params_table_to_models
    >>> from photutils.psf import CircularGaussianPSF
    >>> tbl = QTable()
    >>> tbl['x_0'] = [1, 2, 3]
    >>> tbl['y_0'] = [4, 5, 6]
    >>> tbl['flux'] = [100, 200, 300]
    >>> model = CircularGaussianPSF()
    >>> models = params_table_to_models(tbl, model)
    >>> models
    [<CircularGaussianPSF(flux=100., x_0=1., y_0=4., fwhm=1.)>,
     <CircularGaussianPSF(flux=200., x_0=2., y_0=5., fwhm=1.)>,
     <CircularGaussianPSF(flux=300., x_0=3., y_0=6., fwhm=1.)>]
    """
    param_names = set(model.param_names)
    colnames = set(params_table.colnames)
    if param_names.isdisjoint(colnames):
        raise ValueError('No matching model parameter names found in '
                         'params_table')

    param_names = [*list(param_names), 'name']
    models = []
    for row in params_table:
        new_model = model.copy()
        for param_name in param_names:
            if param_name not in colnames:
                continue
            setattr(new_model, param_name, row[param_name])
        models.append(new_model)

    return models
