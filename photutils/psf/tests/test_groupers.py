# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the groupers module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from photutils.psf import SourceGrouper, SourceGroups
from photutils.utils._optional_deps import HAS_MATPLOTLIB


class TestSourceGrouper:
    """
    Tests for the SourceGrouper class.
    """

    def test_initialization(self):
        """
        Test SourceGrouper initialization.
        """
        grouper = SourceGrouper(min_separation=10.0)
        assert grouper.min_separation == 10.0

    def test_repr(self):
        """
        Test string representation.
        """
        grouper = SourceGrouper(min_separation=5.5)
        repr_str = repr(grouper)
        assert 'SourceGrouper' in repr_str
        assert '5.5' in repr_str

    def test_empty_input(self):
        """
        Test that empty arrays raise ValueError.
        """
        xx = np.array([])
        yy = np.array([])
        grouper = SourceGrouper(min_separation=10)
        match = 'x and y must not be empty'
        with pytest.raises(ValueError, match=match):
            grouper(xx, yy)

    def test_single_source(self):
        """
        Test with a single source.
        """
        xx = np.array([0])
        yy = np.array([0])
        grouper = SourceGrouper(min_separation=10)
        result = grouper(xx, yy)

        # Default behavior returns array
        assert isinstance(result, np.ndarray)
        assert_equal(result, [1])

    def test_single_source_return_groups_object(self):
        """
        Test with a single source returning SourceGroups object.
        """
        xx = np.array([0])
        yy = np.array([0])
        grouper = SourceGrouper(min_separation=10)
        result = grouper(xx, yy, return_groups_object=True)

        assert isinstance(result, SourceGroups)
        assert_equal(result.groups, [1])
        assert result.n_sources == 1
        assert result.n_groups == 1

    def test_mismatched_shapes(self):
        """
        Test that mismatched x and y shapes raise ValueError.
        """
        xx = np.array([1, 2, 3, 4])
        yy = np.array([1, 2])
        grouper = SourceGrouper(min_separation=10)
        match = 'x and y must have the same shape'
        with pytest.raises(ValueError, match=match):
            grouper(xx, yy)

    def test_non_finite_x(self):
        """
        Test that non-finite x values raise ValueError.
        """
        xx = np.array([1, 2, np.nan])
        yy = np.array([1, 2, 3])
        grouper = SourceGrouper(min_separation=10)
        match = 'x coordinates must be finite'
        with pytest.raises(ValueError, match=match):
            grouper(xx, yy)

    def test_non_finite_x_inf(self):
        """
        Test that infinite x values raise ValueError.
        """
        xx = np.array([1, 2, np.inf])
        yy = np.array([1, 2, 3])
        grouper = SourceGrouper(min_separation=10)
        match = 'x coordinates must be finite'
        with pytest.raises(ValueError, match=match):
            grouper(xx, yy)

    def test_non_finite_y(self):
        """
        Test that non-finite y values raise ValueError.
        """
        xx = np.array([1, 2, 3])
        yy = np.array([1, np.nan, 3])
        grouper = SourceGrouper(min_separation=10)
        match = 'y coordinates must be finite'
        with pytest.raises(ValueError, match=match):
            grouper(xx, yy)

    def test_non_finite_y_inf(self):
        """
        Test that infinite y values raise ValueError.
        """
        xx = np.array([1, 2, 3])
        yy = np.array([1, np.inf, 3])
        grouper = SourceGrouper(min_separation=10)
        match = 'y coordinates must be finite'
        with pytest.raises(ValueError, match=match):
            grouper(xx, yy)

    def test_all_isolated_sources(self):
        """
        Test when all sources are isolated from each other.
        """
        xx = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                       -np.sqrt(2) / 4])
        yy = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                       -np.sqrt(2) / 4])
        gg = np.arange(len(xx), dtype=int) + 1
        grouper = SourceGrouper(min_separation=0.01)
        result = grouper(xx, yy)

        assert_equal(result, gg)
        assert len(np.unique(result)) == len(xx)

    def test_two_groups_horizontal(self):
        """
        Test sources forming two distinct horizontal groups.
        """
        #      +---------+--------+---------+---------+--------+---------+
        #      |  *            *                        *             *  |
        #      |                                                         |
        #  0.2 +                                                         +
        #      |                                                         |
        #      |                                                         |
        #      |                                                         |
        #    0 +         *                                     *         +
        #      |                                                         |
        #      |                                                         |
        #      |                                                         |
        # -0.2 +                                                         +
        #      |                                                         |
        #      |  *            *                        *             *  |
        #      +---------+--------+---------+---------+--------+---------+
        #                0       0.5        1        1.5       2

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
        result = grouper(xx, yy)

        assert_equal(result, gg)
        assert len(np.unique(result)) == 2

    def test_two_groups_vertical(self):
        """
        Test sources forming two distinct vertical groups.
        """
        #      +--------------+--------------+-------------+--------------+
        #    3 +                             *                            +
        #      |                             *                            |
        #  2.5 +                             *                            +
        #      |                             *                            |
        #    2 +                             *                            +
        #      |                                                          |
        #  1.5 +                                                          +
        #      |                                                          |
        #    1 +                             *                            +
        #      |                             *                            |
        #  0.5 +                             *                            +
        #      |                             *                            |
        #    0 +                             *                            +
        #      +--------------+--------------+-------------+--------------+
        #     -1            -0.5             0            0.5             1

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
        result = grouper(xx, yy)

        assert_equal(result, gg)
        assert len(np.unique(result)) == 2

    def test_two_groups_horizontal_separated(self):
        """
        Test two horizontally separated groups of sources.
        """
        #     1 +--+-------+--------+--------+--------+-------+--------+--+
        #       |                                                         |
        #       |                                                         |
        #       |                                                         |
        #   0.5 +                                                         +
        #       |                                                         |
        #       |                                                         |
        #     0 +  *   *   *    *   *                 *   *   *    *   *  +
        #       |                                                         |
        #       |                                                         |
        #  -0.5 +                                                         +
        #       |                                                         |
        #       |                                                         |
        #       |                                                         |
        #    -1 +--+-------+--------+--------+--------+-------+--------+--+
        #          0      0.5       1       1.5       2      2.5       3

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
        result = grouper(xx, yy)

        assert_equal(result, gg)
        assert len(np.unique(result)) == 2

    def test_circular_group(self):
        """
        Test sources arranged in a circle forming a single group.
        """
        #       +-+---------+---------+---------+---------+-+
        #     1 +                     *                     +
        #       |           *                   *           |
        #       |                                           |
        #       |                                           |
        #   0.5 +                                           +
        #       |                                           |
        #       |                                           |
        #       |                                           |
        #     0 + *                                       * +
        #       |                                           |
        #       |                                           |
        #  -0.5 +                                           +
        #       |                                           |
        #       |                                           |
        #       |           *                   *           |
        #    -1 +                     *                     +
        #       +-+---------+---------+---------+---------+-+
        #        -1       -0.5        0        0.5        1

        x = np.linspace(-1.0, 1.0, 5)
        y = np.sqrt(1.0 - x**2)
        xx = np.hstack((x, x))
        yy = np.hstack((y, -y))
        gg = np.ones(len(xx), dtype=int)

        grouper = SourceGrouper(min_separation=2.5)
        result = grouper(xx, yy)

        assert_equal(result, gg)
        assert len(np.unique(result)) == 1

    def test_four_groups_cross_pattern(self):
        """
        Test sources forming four groups in a cross pattern.
        """
        #      +--+--------+--------+-------+--------+--------+--------+--+
        #    3 +                            *                             +
        #      |                            *                             |
        #  2.5 +                            *                             +
        #      |                            *                             |
        #    2 +                            *                             +
        #      |                                                          |
        #  1.5 +  *   *    *   *    *                *    *   *    *   *  +
        #      |                                                          |
        #    1 +                            *                             +
        #      |                            *                             |
        #  0.5 +                            *                             +
        #      |                            *                             |
        #    0 +                            *                             +
        #      +--+--------+--------+-------+--------+--------+--------+--+
        #          0       0.5       1      1.5       2       2.5       3

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
        result = grouper(xx, yy)

        assert_equal(result, gg)
        assert len(np.unique(result)) == 4

    def test_three_groups_horizontal(self):
        """
        Test sources forming three horizontal groups.
        """
        #       +------+----------+----------+----------+----------+------+
        #       |  *       *             *       *             *       *  |
        #       |                                                         |
        #   0.2 +                                                         +
        #       |                                                         |
        #       |                                                         |
        #       |                                                         |
        #     0 +      *                     *                     *      +
        #       |                                                         |
        #       |                                                         |
        #       |                                                         |
        #  -0.2 +                                                         +
        #       |                                                         |
        #       |  *       *             *       *             *       *  |
        #       +------+----------+----------+----------+----------+------+
        #              0          1          2          3          4

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
        result = grouper(xx, yy)

        assert_equal(result, gg)
        assert len(np.unique(result)) == 3

    def test_list_input(self):
        """
        Test that list inputs are handled correctly.
        """
        xx = [0, 10, 20]
        yy = [0, 10, 20]
        grouper = SourceGrouper(min_separation=5)
        result = grouper(xx, yy)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        # Each source should be in its own group (separated by 10 units)
        assert len(np.unique(result)) == 3

    def test_return_groups_object_parameter(self):
        """
        Test the return_groups_object parameter.
        """
        xx = np.array([0, 10, 20])
        yy = np.array([0, 10, 20])
        grouper = SourceGrouper(min_separation=5)

        # Default behavior - returns array
        result_array = grouper(xx, yy)
        assert isinstance(result_array, np.ndarray)
        assert_equal(result_array, [1, 2, 3])

        # With return_groups_object=True - returns SourceGroups
        result_obj = grouper(xx, yy, return_groups_object=True)
        assert isinstance(result_obj, SourceGroups)
        assert result_obj.n_sources == 3
        assert result_obj.n_groups == 3
        assert_equal(result_obj.groups, [1, 2, 3])

    def test_return_groups_object_false_explicit(self):
        """
        Test that return_groups_object=False returns array.
        """
        xx = np.array([10, 15, 50])
        yy = np.array([20, 25, 60])
        grouper = SourceGrouper(min_separation=10)
        result = grouper(xx, yy, return_groups_object=False)

        assert isinstance(result, np.ndarray)
        assert_equal(result, [1, 1, 2])

    def test_different_min_separations(self):
        """
        Test that different min_separation values produce different
        groupings.
        """
        xx = np.array([0, 5, 10])
        yy = np.array([0, 0, 0])

        # Large separation: all in one group (threshold >= 5.0)
        grouper1 = SourceGrouper(min_separation=5)
        result1 = grouper1(xx, yy)
        assert len(np.unique(result1)) == 1

        # Small separation: all isolated (threshold < 5.0)
        grouper2 = SourceGrouper(min_separation=4)
        result2 = grouper2(xx, yy)
        assert len(np.unique(result2)) == 3

        # Very large separation: still one group
        grouper3 = SourceGrouper(min_separation=20)
        result3 = grouper3(xx, yy)
        assert len(np.unique(result3)) == 1

    def test_clustered_with_isolated(self):
        """
        Test mixture of clustered and isolated sources.
        """
        # Create a cluster of 5 sources close together
        cluster_x = np.array([0, 0.1, 0.2, 0.1, 0.2])
        cluster_y = np.array([0, 0.1, 0.0, 0.0, 0.1])

        # Add isolated sources far away
        isolated_x = np.array([10, 20, 30])
        isolated_y = np.array([10, 20, 30])

        xx = np.concatenate([cluster_x, isolated_x])
        yy = np.concatenate([cluster_y, isolated_y])

        grouper = SourceGrouper(min_separation=1.0)
        result = grouper(xx, yy)

        # Should have 4 groups: 1 cluster + 3 isolated
        assert len(np.unique(result)) == 4

        # First 5 sources should be in the same group
        assert len(np.unique(result[:5])) == 1

        # Last 3 sources should each be in different groups
        assert len(np.unique(result[5:])) == 3

    def test_very_small_separation(self):
        """
        Test with very small min_separation (almost touching sources).
        """
        xx = np.array([0, 0.001, 0.002])
        yy = np.array([0, 0.001, 0.002])

        grouper = SourceGrouper(min_separation=0.01)
        result = grouper(xx, yy)

        # All should be in one group
        assert len(np.unique(result)) == 1
        assert_equal(result, [1, 1, 1])

    def test_returns_array_by_default(self):
        """
        Test that __call__ returns array by default.
        """
        xx = np.array([0, 10])
        yy = np.array([0, 10])
        grouper = SourceGrouper(min_separation=5)
        result = grouper(xx, yy)

        assert isinstance(result, np.ndarray)
        assert_equal(result, [1, 2])

    def test_returns_sourcegroups_object_when_requested(self):
        """
        Test that __call__ returns SourceGroups when
        return_groups_object=True.
        """
        xx = np.array([0, 10])
        yy = np.array([0, 10])
        grouper = SourceGrouper(min_separation=5)
        result = grouper(xx, yy, return_groups_object=True)

        assert isinstance(result, SourceGroups)
        assert hasattr(result, 'groups')
        assert hasattr(result, 'x')
        assert hasattr(result, 'y')
        assert hasattr(result, 'n_sources')
        assert hasattr(result, 'n_groups')


@pytest.fixture
def sample_groups():
    """
    Fixture providing sample source grouping data for tests.

    Creates a simple dataset with 3 groups:
    - Group 1: 3 sources at (0, 0), (0.1, 0.1), (0.2, 0.0)
    - Group 2: 2 sources at (10, 10), (10.1, 10.1)
    - Group 3: 1 isolated source at (20, 20)

    Returns
    -------
    dict
        Dictionary containing:
        - 'x': x coordinates array
        - 'y': y coordinates array
        - 'groups_array': group IDs array
        - 'groups': SourceGroups object
    """
    x = np.array([0, 0.1, 0.2, 10, 10.1, 20])
    y = np.array([0, 0.1, 0.0, 10, 10.1, 20])
    groups_array = np.array([1, 1, 1, 2, 2, 3])

    grouper = SourceGrouper(min_separation=1.0)
    groups = grouper(x, y, return_groups_object=True)

    return {
        'x': x,
        'y': y,
        'groups_array': groups_array,
        'groups': groups,
    }


class TestSourceGroups:
    """
    Tests for the SourceGroups class.
    """

    def test_initialization(self, sample_groups):
        """
        Test SourceGroups initialization.
        """
        groups = SourceGroups(sample_groups['x'], sample_groups['y'],
                              sample_groups['groups_array'])

        assert_equal(groups.x, sample_groups['x'])
        assert_equal(groups.y, sample_groups['y'])
        assert_equal(groups.groups, sample_groups['groups_array'])
        assert groups.n_sources == 6
        assert groups.n_groups == 3

    def test_initialization_mismatched_shapes(self):
        """
        Test that mismatched array shapes raise ValueError.
        """
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        groups = np.array([1, 1, 1])

        match = 'x, y, and groups must have the same shape'
        with pytest.raises(ValueError, match=match):
            SourceGroups(x, y, groups)

    def test_initialization_groups_mismatch(self):
        """
        Test that groups array with wrong shape raises ValueError.
        """
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        groups = np.array([1, 1])

        match = 'x, y, and groups must have the same shape'
        with pytest.raises(ValueError, match=match):
            SourceGroups(x, y, groups)

    def test_repr(self, sample_groups):
        """
        Test string representation.
        """
        repr_str = repr(sample_groups['groups'])
        assert 'SourceGroups' in repr_str
        assert 'n_sources=6' in repr_str
        assert 'n_groups=3' in repr_str

    def test_len(self, sample_groups):
        """
        Test len() returns n_sources.
        """
        assert len(sample_groups['groups']) == 6

    def test_sizes_property(self, sample_groups):
        """
        Test sizes property.
        """
        sizes = sample_groups['groups'].sizes

        # Should return size for each source
        assert len(sizes) == 6
        assert_equal(sizes, [3, 3, 3, 2, 2, 1])

    def test_group_centers_property(self, sample_groups):
        """
        Test group_centers property.
        """
        centers = sample_groups['groups'].group_centers

        assert isinstance(centers, dict)
        assert len(centers) == 3

        # Check group 1 center (mean of first 3 sources)
        expected_x1 = np.mean([0, 0.1, 0.2])
        expected_y1 = np.mean([0, 0.1, 0.0])
        assert_allclose(centers[1], (expected_x1, expected_y1))

        # Check group 2 center
        expected_x2 = np.mean([10, 10.1])
        expected_y2 = np.mean([10, 10.1])
        assert_allclose(centers[2], (expected_x2, expected_y2))

        # Check group 3 center (single source)
        assert_allclose(centers[3], (20, 20))

    def test_get_group_sources(self, sample_groups):
        """
        Test get_group_sources method.
        """
        groups = sample_groups['groups']
        # Get sources from group 1
        x1, y1 = groups.get_group_sources(1)
        assert len(x1) == 3
        assert len(y1) == 3
        assert_equal(x1, [0, 0.1, 0.2])
        assert_equal(y1, [0, 0.1, 0.0])

        # Get sources from group 2
        x2, y2 = groups.get_group_sources(2)
        assert len(x2) == 2
        assert len(y2) == 2
        assert_equal(x2, [10, 10.1])
        assert_equal(y2, [10, 10.1])

        # Get sources from group 3 (isolated)
        x3, y3 = groups.get_group_sources(3)
        assert len(x3) == 1
        assert len(y3) == 1
        assert_equal(x3, [20])
        assert_equal(y3, [20])

    def test_get_group_sources_invalid_id(self, sample_groups):
        """
        Test that invalid group ID raises ValueError.
        """
        match = 'Group ID 99 not found in groups'
        with pytest.raises(ValueError, match=match):
            sample_groups['groups'].get_group_sources(99)

    def test_single_source(self):
        """
        Test SourceGroups with single source.
        """
        x = np.array([1.0])
        y = np.array([2.0])
        groups = np.array([1])

        result = SourceGroups(x, y, groups)

        assert result.n_sources == 1
        assert result.n_groups == 1
        assert len(result) == 1
        assert_equal(result.sizes, [1])

    def test_all_isolated_sources(self):
        """
        Test with all sources isolated (each in own group).
        """
        x = np.array([0, 10, 20, 30])
        y = np.array([0, 10, 20, 30])
        groups = np.array([1, 2, 3, 4])

        result = SourceGroups(x, y, groups)

        assert result.n_sources == 4
        assert result.n_groups == 4
        assert_equal(result.sizes, [1, 1, 1, 1])

    def test_all_in_one_group(self):
        """
        Test with all sources in a single group.
        """
        x = np.array([0, 0.1, 0.2, 0.3])
        y = np.array([0, 0.1, 0.2, 0.3])
        groups = np.array([1, 1, 1, 1])

        result = SourceGroups(x, y, groups)

        assert result.n_sources == 4
        assert result.n_groups == 1
        assert_equal(result.sizes, [4, 4, 4, 4])

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_basic(self, sample_groups):
        """
        Test basic plot functionality.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_ax = sample_groups['groups'].plot(radius=0.5, ax=ax)

        assert result_ax is ax
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_cmap(self, sample_groups):
        """
        Test cmap string input.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        groups = sample_groups['groups']
        result_ax = groups.plot(radius=0.5, ax=ax, cmap='Blues')

        assert result_ax is ax
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_with_label_offset(self, sample_groups):
        """
        Test plot with label offset.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_ax = sample_groups['groups'].plot(
            radius=0.5, ax=ax, label_groups=True, label_offset=(5, 0))

        assert result_ax is ax
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_with_custom_kwargs(self, sample_groups):
        """
        Test plot with custom styling kwargs.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        label_kwargs = {'fontsize': 12, 'fontweight': 'bold'}
        result_ax = sample_groups['groups'].plot(
            radius=0.5, ax=ax, label_groups=True,
            label_kwargs=label_kwargs, lw=3, alpha=0.5)

        assert result_ax is ax
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_no_axes(self, sample_groups):
        """
        Test plot creates axes if none provided.
        """
        import matplotlib.pyplot as plt

        result_ax = sample_groups['groups'].plot(radius=0.5)

        assert result_ax is not None
        plt.close('all')
