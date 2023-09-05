# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the grouper module.
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from photutils.psf.groupers import SourceGrouper
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_empty():
    """
    Test case when there are no sources.
    """
    xx = np.array([])
    yy = np.array([])
    grouper = SourceGrouper(min_separation=10)
    match = 'x and y must not be empty'
    with pytest.raises(ValueError, match=match):
        grouper(xx, yy)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_one_source():
    """
    Test case when there is only one source.
    """
    xx = np.array([0])
    yy = np.array([0])
    gg = np.array([1])
    grouper = SourceGrouper(min_separation=10)
    groups = grouper(xx, yy)
    assert_equal(groups, gg)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_inputs():
    xx = np.array([1, 2, 3, 4])
    yy = np.array([1, 2])
    grouper = SourceGrouper(min_separation=10)
    match = 'x and y must have the same shape'
    with pytest.raises(ValueError, match=match):
        grouper(xx, yy)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_isolated_sources():
    """
    Test case when all sources are isolated.
    """
    xx = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                   -np.sqrt(2) / 4])
    yy = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                   -np.sqrt(2) / 4])
    gg = np.arange(len(xx), dtype=int) + 1
    grouper = SourceGrouper(min_separation=0.01)
    groups = grouper(xx, yy)
    assert_equal(groups, gg)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_one():
    """
        +---------+--------+---------+---------+--------+---------+
        |  *            *                        *             *  |
        |                                                         |
    0.2 +                                                         +
        |                                                         |
        |                                                         |
        |                                                         |
      0 +         *                                     *         +
        |                                                         |
        |                                                         |
        |                                                         |
   -0.2 +                                                         +
        |                                                         |
        |  *            *                        *             *  |
        +---------+--------+---------+---------+--------+---------+
                  0       0.5        1        1.5       2

    x and y axis are in pixel coordinates. Each asterisk represents
    the centroid of a star.
    """
    x1 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                   -np.sqrt(2) / 4])
    y1 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                   -np.sqrt(2) / 4])
    g1 = np.ones(len(x1), dtype=int)
    x2 = x1 + 2.0
    y2 = y1
    g2 = np.ones(len(x1), dtype=int) + 1

    xx = np.hstack([x1, x2])
    yy = np.hstack([y1, y2])
    gg = np.hstack([g1, g2])
    grouper = SourceGrouper(min_separation=0.6)
    groups = grouper(xx, yy)
    assert_equal(groups, gg)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_two():
    """
        +--------------+--------------+-------------+--------------+
      3 +                             *                            +
        |                             *                            |
    2.5 +                             *                            +
        |                             *                            |
      2 +                             *                            +
        |                                                          |
    1.5 +                                                          +
        |                                                          |
      1 +                             *                            +
        |                             *                            |
    0.5 +                             *                            +
        |                             *                            |
      0 +                             *                            +
        +--------------+--------------+-------------+--------------+
       -1            -0.5             0            0.5             1
    """
    x1 = np.zeros(5)
    y1 = np.linspace(0, 1, 5)
    g1 = np.ones(5, dtype=int)
    x2 = np.zeros(5)
    y2 = np.linspace(2, 3, 5)
    g2 = np.ones(5, dtype=int) + 1

    xx = np.hstack([x1, x2])
    yy = np.hstack([y1, y2])
    gg = np.hstack([g1, g2])
    grouper = SourceGrouper(min_separation=0.3)
    groups = grouper(xx, yy)
    assert_equal(groups, gg)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_three():
    """
      1 +--+-------+--------+--------+--------+-------+--------+--+
        |                                                         |
        |                                                         |
        |                                                         |
    0.5 +                                                         +
        |                                                         |
        |                                                         |
      0 +  *   *   *    *   *                 *   *   *    *   *  +
        |                                                         |
        |                                                         |
   -0.5 +                                                         +
        |                                                         |
        |                                                         |
        |                                                         |
     -1 +--+-------+--------+--------+--------+-------+--------+--+
           0      0.5       1       1.5       2      2.5       3
    """
    x1 = np.linspace(0, 1, 5)
    y1 = np.zeros(5)
    g1 = np.ones(5, dtype=int)
    x2 = np.linspace(2, 3, 5)
    y2 = np.zeros(5)
    g2 = np.ones(5, dtype=int) + 1

    xx = np.hstack([x1, x2])
    yy = np.hstack([y1, y2])
    gg = np.hstack([g1, g2])
    grouper = SourceGrouper(min_separation=0.3)
    groups = grouper(xx, yy)
    assert_equal(groups, gg)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_four():
    """
        +-+---------+---------+---------+---------+-+
      1 +                     *                     +
        |           *                   *           |
        |                                           |
        |                                           |
    0.5 +                                           +
        |                                           |
        |                                           |
        |                                           |
      0 + *                                       * +
        |                                           |
        |                                           |
   -0.5 +                                           +
        |                                           |
        |                                           |
        |           *                   *           |
     -1 +                     *                     +
        +-+---------+---------+---------+---------+-+
         -1       -0.5        0        0.5        1
    """
    x = np.linspace(-1.0, 1.0, 5)
    y = np.sqrt(1.0 - x**2)
    xx = np.hstack((x, x))
    yy = np.hstack((y, -y))
    gg = np.ones(len(xx), dtype=int)

    grouper = SourceGrouper(min_separation=2.5)
    groups = grouper(xx, yy)
    assert_equal(groups, gg)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_five():
    """
        +--+--------+--------+-------+--------+--------+--------+--+
      3 +                            *                             +
        |                            *                             |
    2.5 +                            *                             +
        |                            *                             |
      2 +                            *                             +
        |                                                          |
    1.5 +  *   *    *   *    *                *    *   *    *   *  +
        |                                                          |
      1 +                            *                             +
        |                            *                             |
    0.5 +                            *                             +
        |                            *                             |
      0 +                            *                             +
        +--+--------+--------+-------+--------+--------+--------+--+
            0       0.5       1      1.5       2       2.5       3
    """
    x1 = 1.5 * np.ones(5)
    y1 = np.linspace(0, 1, 5)
    g1 = np.ones(5, dtype=int)

    x2 = 1.5 * np.ones(5)
    y2 = np.linspace(2, 3, 5)
    g2 = np.ones(5, dtype=int) + 1

    x3 = np.linspace(0, 1, 5)
    y3 = 1.5 * np.ones(5)
    g3 = np.ones(5, dtype=int) + 2

    x4 = np.linspace(2, 3, 5)
    y4 = 1.5 * np.ones(5)
    g4 = np.ones(5, dtype=int) + 3

    xx = np.hstack([x1, x2, x3, x4])
    yy = np.hstack([y1, y2, y3, y4])
    gg = np.hstack([g1, g2, g3, g4])

    grouper = SourceGrouper(min_separation=0.3)
    groups = grouper(xx, yy)
    assert_equal(groups, gg)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_grouper_six():
    """
        +------+----------+----------+----------+----------+------+
        |  *       *             *       *             *       *  |
        |                                                         |
    0.2 +                                                         +
        |                                                         |
        |                                                         |
        |                                                         |
      0 +      *                     *                     *      +
        |                                                         |
        |                                                         |
        |                                                         |
   -0.2 +                                                         +
        |                                                         |
        |  *       *             *       *             *       *  |
        +------+----------+----------+----------+----------+------+
               0          1          2          3          4
    """
    x1 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                   -np.sqrt(2) / 4])
    y1 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                   -np.sqrt(2) / 4])
    g1 = np.ones(len(x1), dtype=int)

    x2 = x1 + 2.0
    y2 = y1
    g2 = np.ones(len(x1), dtype=int) + 1

    x3 = x1 + 4.0
    y3 = y1
    g3 = np.ones(len(x1), dtype=int) + 2

    xx = np.hstack([x1, x2, x3])
    yy = np.hstack([y1, y2, y3])
    gg = np.hstack([g1, g2, g3])
    grouper = SourceGrouper(min_separation=0.6)
    groups = grouper(xx, yy)
    assert_equal(groups, gg)
