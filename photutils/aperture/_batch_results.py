# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Result containers of the batch aperture Cython drivers.
"""

from typing import NamedTuple

import numpy as np

__all__ = []


class BatchApertureSums(NamedTuple):
    """
    The per-source results of ``batch_aperture_sums`` (see its
    docstring for the full description of each field).

    The fields are the driver's return values in order, so the result
    can also be unpacked or indexed as a plain tuple.
    """

    sums: np.ndarray
    """The aperture sums (NaN where the bounding box does not overlap
    the data)."""

    sum_vars: np.ndarray
    """The aperture sum variances (NaN if ``error`` is `None`)."""

    areas: np.ndarray
    """The aperture areas within the data."""

    overlap: np.ndarray
    """Whether the aperture bounding box overlaps the data (bool)."""

    starts: np.ndarray
    """The start offset of each source in the packed per-pixel
    buffers."""

    sum_values: np.ndarray
    """The packed per-pixel background-subtracted values (empty unless
    ``emit_sum`` is nonzero)."""

    sum_fracs: np.ndarray
    """The packed per-pixel overlap fractions (empty unless
    ``emit_sum`` is nonzero)."""

    sum_errsq: np.ndarray
    """The packed per-pixel error variances (empty unless
    ``emit_sum`` is nonzero)."""

    sum_counts: np.ndarray
    """The number of packed per-pixel entries of each source (empty
    unless ``emit_sum`` is nonzero)."""

    flag_counts: np.ndarray
    """The ``(n_sources, N_FLAG_COLS)`` per-source flag counts (see the
    ``FLAG_COL_*`` constants)."""

    weights_out: np.ndarray
    """The per-source 0/1 indicator of nonzero aperture weights outside
    the data (uint8)."""
