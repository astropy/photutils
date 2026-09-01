# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared helper functions for the photutils benchmark scripts.
"""

import os
import platform
import sys
import time

import numpy as np

from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                PolygonAperture, RectangularAnnulus,
                                RectangularAperture)
from photutils.datasets import make_wcs


def time_best(func, *, repeats=3):
    """
    Return the best wall-clock time of ``repeats`` calls to ``func``.

    Parameters
    ----------
    func : callable
        The zero-argument callable to time.

    repeats : int, optional
        The number of times to call ``func``.

    Returns
    -------
    result : float
        The best (minimum) wall-clock time in seconds.
    """
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


def make_image(shape, *, seed=0):
    """
    Return a Gaussian-noise image of the given shape.

    Parameters
    ----------
    shape : tuple of int
        The shape of the output image.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The noise image.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(100.0, 5.0, shape)


def print_environment():
    """
    Print information about the runtime environment.
    """
    import astropy
    import scipy

    import photutils
    from photutils.utils._optional_deps import HAS_BOTTLENECK

    gil = getattr(sys, '_is_gil_enabled', lambda: True)()
    print(f'python {platform.python_version()} (GIL enabled: {gil}), '
          f'{os.cpu_count()} CPUs')
    print(f'photutils {photutils.__version__}, numpy {np.__version__}, '
          f'scipy {scipy.__version__}, astropy {astropy.__version__}, '
          f'bottleneck: {HAS_BOTTLENECK}')
    print(f'photutils path: {os.path.dirname(photutils.__file__)}')


def parse_thread_counts(text):
    """
    Parse a comma-separated list of thread counts.

    Parameters
    ----------
    text : str
        The comma-separated thread counts (e.g., ``'1,2,4,8'``).

    Returns
    -------
    result : list of int
        The thread counts.
    """
    counts = [int(item) for item in text.split(',')]
    if any(count < 1 for count in counts):
        msg = 'thread counts must be positive integers'
        raise ValueError(msg)
    return counts


def format_sweep_cells(times):
    """
    Format thread-sweep times, with speedups relative to the first
    time.

    Parameters
    ----------
    times : list of float
        The wall-clock times in seconds, one per thread count.

    Returns
    -------
    result : list of str
        The formatted cells (e.g., ``['0.3270s', '0.1700s (1.9x)']``).
    """
    cells = [f'{times[0]:.4f}s']
    cells.extend(f'{t:.4f}s ({times[0] / t:.1f}x)' for t in times[1:])
    return cells


def make_apertures(positions):
    """
    Return all of the pixel-based aperture types at the given
    positions, with roughly comparable sizes.

    Parameters
    ----------
    positions : 2D `~numpy.ndarray`
        The (x, y) aperture positions.

    Returns
    -------
    result : list of (str, aperture) tuples
        The aperture name and instance pairs.
    """
    theta = 0.5
    return [
        ('CircularAperture', CircularAperture(positions, r=8.0)),
        ('CircularAnnulus',
         CircularAnnulus(positions, r_in=5.0, r_out=10.0)),
        ('EllipticalAperture',
         EllipticalAperture(positions, a=10.0, b=6.0, theta=theta)),
        ('EllipticalAnnulus',
         EllipticalAnnulus(positions, a_in=6.0, a_out=12.0, b_out=8.0,
                           theta=theta)),
        ('RectangularAperture',
         RectangularAperture(positions, w=14.0, h=10.0, theta=theta)),
        ('RectangularAnnulus',
         RectangularAnnulus(positions, w_in=8.0, w_out=16.0, h_out=12.0,
                            theta=theta)),
        ('PolygonAperture',
         PolygonAperture.from_regular_polygon(positions, n_vertices=6,
                                              radius=8.0)),
    ]


def make_aperture_inputs(size, n_sources, *, seed=0):
    """
    Return the data, error, wcs, and aperture positions used by the
    aperture benchmark scripts.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    n_sources : int
        The number of aperture positions.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data, error : 2D `~numpy.ndarray`
        The image and its error array.

    wcs : `~astropy.wcs.WCS`
        A WCS for the image.

    positions : 2D `~numpy.ndarray`
        The (x, y) aperture positions.
    """
    data = make_image((size, size), seed=seed)
    error = np.sqrt(np.abs(data))
    wcs = make_wcs(data.shape)
    rng = np.random.default_rng(seed)
    margin = 20.0
    positions = np.column_stack(
        [rng.uniform(margin, size - margin, n_sources),
         rng.uniform(margin, size - margin, n_sources)])
    return data, error, wcs, positions
