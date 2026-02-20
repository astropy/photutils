# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for measuring non-parametric morphologies of sources.
"""

import numpy as np

__all__ = ['gini']


def gini(data, mask=None):
    r"""
    Calculate the `Gini coefficient
    <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of an array.

    The Gini coefficient is calculated using the prescription from `Lotz
    et al. 2004
    <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
    as:

    .. math::

        G = \frac{1}{\left | \bar{x} \right | n (n - 1)}
            \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

    where :math:`\bar{x}` is the mean over all pixel values :math:`x_i`.
    If the sum of all pixel values is zero, the Gini coefficient is
    zero.

    The Gini coefficient is a way of measuring the inequality in a given
    set of values. In the context of galaxy morphology, it measures how
    the light of a galaxy image is distributed among its pixels. A Gini
    coefficient value of 0 corresponds to a galaxy image with the light
    evenly distributed over all pixels while a Gini coefficient value of
    1 represents a galaxy image with all its light concentrated in just
    one pixel.

    Usually Gini's measurement needs some sort of preprocessing for
    defining the galaxy region in the image based on the quality of the
    input data. As there is not a general standard for doing this, this
    is left for the user.

    Invalid values (NaN and inf) in the input are automatically excluded
    from the calculation. Negative pixel values are used via their
    absolute value, consistent with the :math:`|x_i|` term in the Lotz
    et al. formula. If only a single finite pixel remains after
    filtering, the Gini coefficient is 0.0.

    Parameters
    ----------
    data : array_like
        The 1D or 2D data array or object that can be converted to an
        array.

    mask : array_like, optional
        A boolean mask with the same shape as ``data`` where `True`
        values indicate masked pixels. Masked pixels are excluded from
        the calculation.

    Returns
    -------
    result : float
        The Gini coefficient of the input array.

    Raises
    ------
    ValueError
        If ``mask`` is provided and does not have the same shape as
        ``data``.
    """
    data = np.asarray(data)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            msg = 'mask must have the same shape as data'
            raise ValueError(msg)
        values = np.ravel(data[~mask])
    else:
        values = np.ravel(data)

    # Exclude invalid values (NaN, inf)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan

    npix = values.size
    if npix == 1:
        return 0.0

    normalization = np.mean(np.abs(values)) * npix * (npix - 1)
    if normalization == 0.0:
        return 0.0

    kernel = ((2.0 * np.arange(1, npix + 1) - npix - 1)
              * np.sort(np.abs(values)))

    return np.sum(kernel) / normalization
