# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Result containers of the batch segmentation Cython drivers.
"""

from typing import NamedTuple

import numpy as np

__all__ = []


class BatchFluxRadiusArgs(NamedTuple):
    """
    The prepared per-source inputs of the flux-radius root-find, as
    returned by ``batch_flux_radius_prepare`` and consumed by
    ``batch_flux_radius_solve`` (see their docstrings).

    The cleaned cutouts of all sources are packed into one buffer.
    A source that cannot have a solution has a zero pixel count.
    """

    values: np.ndarray
    """The packed cleaned, background-subtracted cutout values of all
    sources, each in row-major order."""

    starts: np.ndarray
    """The start offset of each source in ``values``."""

    counts: np.ndarray
    """The number of cutout pixels of each source (zero for a source
    without a solution)."""

    nx: np.ndarray
    """The cutout width of each source."""

    ny: np.ndarray
    """The cutout height of each source."""

    grid_edges: np.ndarray
    """The ``(n_sources, 4)`` grid edges ``(xmin, xmax, ymin, ymax)``
    of each cutout, relative to the source centroid."""

    kronflux: np.ndarray
    """The Kron flux of each source."""

    max_radius: np.ndarray
    """The initial upper bracket radius of each source."""

    use_exact: int
    """Whether the root-find uses exact overlap fractions (1) or
    subpixel sampling (0)."""

    subpixels: int
    """The number of subpixels in each dimension when ``use_exact`` is
    0."""
