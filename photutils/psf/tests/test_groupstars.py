# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the groupstars module.
"""

import numpy as np
import pytest
from astropy.table import Table, vstack
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_almost_equal

from photutils.psf.groupstars import DAOGroup, DBSCANGroup
from photutils.utils._optional_deps import HAS_SKLEARN


def assert_table_almost_equal(table1, table2):
    assert table1.colnames == table2.colnames
    assert table1.meta == table2.meta
    for colname in table1.colnames:
        assert_almost_equal(table1[colname], table2[colname])


class TestDAOGROUP:
    def test_daogroup_one(self):
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
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            x_1 = x_0 + 2.0
            first_group = Table([x_0, y_0, np.arange(len(x_0)) + 1,
                                 np.ones(len(x_0), dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            second_group = Table([x_1, y_0, len(x_0) + np.arange(len(x_0)) + 1,
                                  2 * np.ones(len(x_0), dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            starlist = vstack([first_group, second_group])
            daogroup = DAOGroup(crit_separation=0.6)
            test_starlist = daogroup(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_daogroup_two(self):
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
        with pytest.warns(AstropyDeprecationWarning):
            first_group = Table([np.zeros(5), np.linspace(0, 1, 5),
                                 np.arange(5) + 1, np.ones(5, dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            second_group = Table([np.zeros(5), np.linspace(2, 3, 5),
                                  6 + np.arange(5), 2 * np.ones(5, dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            starlist = vstack([first_group, second_group])
            daogroup = DAOGroup(crit_separation=0.3)
            test_starlist = daogroup(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_daogroup_three(self):
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
        with pytest.warns(AstropyDeprecationWarning):
            first_group = Table([np.linspace(0, 1, 5), np.zeros(5),
                                 np.arange(5) + 1, np.ones(5, dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            second_group = Table([np.linspace(2, 3, 5), np.zeros(5),
                                  6 + np.arange(5), 2 * np.ones(5, dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            starlist = vstack([first_group, second_group])
            daogroup = DAOGroup(crit_separation=0.3)
            test_starlist = daogroup(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_daogroup_four(self):
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
        with pytest.warns(AstropyDeprecationWarning):
            x = np.linspace(-1.0, 1.0, 5)
            y = np.sqrt(1.0 - x**2)
            xx = np.hstack((x, x))
            yy = np.hstack((y, -y))
            starlist = Table([xx, yy, np.arange(10) + 1,
                              np.ones(10, dtype=int)],
                             names=('x_0', 'y_0', 'id', 'group_id'))
            daogroup = DAOGroup(crit_separation=2.5)
            test_starlist = daogroup(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_daogroup_five(self):
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
        with pytest.warns(AstropyDeprecationWarning):
            first_group = Table([1.5 * np.ones(5), np.linspace(0, 1, 5),
                                 np.arange(5) + 1, np.ones(5, dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            second_group = Table([1.5 * np.ones(5), np.linspace(2, 3, 5),
                                  6 + np.arange(5), 2 * np.ones(5, dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            third_group = Table([np.linspace(0, 1, 5), 1.5 * np.ones(5),
                                 11 + np.arange(5), 3 * np.ones(5, dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            fourth_group = Table([np.linspace(2, 3, 5), 1.5 * np.ones(5),
                                  16 + np.arange(5),
                                  4 * np.ones(5, dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            starlist = vstack([first_group, second_group, third_group,
                               fourth_group])
            daogroup = DAOGroup(crit_separation=0.3)
            test_starlist = daogroup(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_daogroup_six(self):
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
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            x_1 = x_0 + 2.0
            x_2 = x_0 + 4.0
            first_group = Table([x_0, y_0, np.arange(5) + 1,
                                 np.ones(5, dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            second_group = Table([x_1, y_0, 6 + np.arange(5),
                                  2 * np.ones(5, dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            third_group = Table([x_2, y_0, 11 + np.arange(5),
                                 3 * np.ones(5, dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            starlist = vstack([first_group, second_group, third_group])
            daogroup = DAOGroup(crit_separation=0.6)
            test_starlist = daogroup(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_isolated_sources(self):
        """
        Test case when all sources are isolated.
        """
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            starlist = Table([x_0, y_0, np.arange(len(x_0)) + 1,
                              np.arange(len(x_0)) + 1],
                             names=('x_0', 'y_0', 'id', 'group_id'))
            daogroup = DAOGroup(crit_separation=0.01)
            test_starlist = daogroup(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_id_column(self):
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            starlist = Table([x_0, y_0, np.arange(len(x_0)) + 1,
                              np.arange(len(x_0)) + 1],
                             names=('x_0', 'y_0', 'id', 'group_id'))
            daogroup = DAOGroup(crit_separation=0.01)
            test_starlist = daogroup(starlist['x_0', 'y_0'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_id_column_raise_error(self):
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            starlist = Table([x_0, y_0, np.arange(len(x_0)),
                              np.arange(len(x_0)) + 1],
                             names=('x_0', 'y_0', 'id', 'group_id'))
            daogroup = DAOGroup(crit_separation=0.01)
            with pytest.raises(ValueError):
                daogroup(starlist['x_0', 'y_0', 'id'])


@pytest.mark.skipif(not HAS_SKLEARN, reason='sklearn is required')
class TestDBSCANGroup:
    def test_group_stars_one(self):
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            x_1 = x_0 + 2.0
            first_group = Table([x_0, y_0, np.arange(len(x_0)) + 1,
                                np.ones(len(x_0), dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            second_group = Table([x_1, y_0, len(x_0) + np.arange(len(x_0)) + 1,
                                  2 * np.ones(len(x_0), dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            starlist = vstack([first_group, second_group])
            dbscan = DBSCANGroup(crit_separation=0.6)
            test_starlist = dbscan(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_group_stars_two(self):
        with pytest.warns(AstropyDeprecationWarning):
            first_group = Table([1.5 * np.ones(5), np.linspace(0, 1, 5),
                                 np.arange(5) + 1, np.ones(5, dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            second_group = Table([1.5 * np.ones(5), np.linspace(2, 3, 5),
                                  6 + np.arange(5), 2 * np.ones(5, dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            third_group = Table([np.linspace(0, 1, 5), 1.5 * np.ones(5),
                                 11 + np.arange(5), 3 * np.ones(5, dtype=int)],
                                names=('x_0', 'y_0', 'id', 'group_id'))
            fourth_group = Table([np.linspace(2, 3, 5), 1.5 * np.ones(5),
                                  16 + np.arange(5),
                                  4 * np.ones(5, dtype=int)],
                                 names=('x_0', 'y_0', 'id', 'group_id'))
            starlist = vstack([first_group, second_group, third_group,
                               fourth_group])
            dbscan = DBSCANGroup(crit_separation=0.3)
            test_starlist = dbscan(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_isolated_sources(self):
        """
        Test case when all sources are isolated.
        """
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            starlist = Table([x_0, y_0, np.arange(len(x_0)) + 1,
                              np.arange(len(x_0)) + 1],
                             names=('x_0', 'y_0', 'id', 'group_id'))
            dbscan = DBSCANGroup(crit_separation=0.01)
            test_starlist = dbscan(starlist['x_0', 'y_0', 'id'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_id_column(self):
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            starlist = Table([x_0, y_0, np.arange(len(x_0)) + 1,
                              np.arange(len(x_0)) + 1],
                             names=('x_0', 'y_0', 'id', 'group_id'))
            dbscan = DBSCANGroup(crit_separation=0.01)
            test_starlist = dbscan(starlist['x_0', 'y_0'])
            assert_table_almost_equal(starlist, test_starlist)

    def test_id_column_raise_error(self):
        with pytest.warns(AstropyDeprecationWarning):
            x_0 = np.array([0, np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            y_0 = np.array([0, np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4,
                            -np.sqrt(2) / 4])
            starlist = Table([x_0, y_0, np.arange(len(x_0)),
                              np.arange(len(x_0)) + 1],
                             names=('x_0', 'y_0', 'id', 'group_id'))
            dbscan = DBSCANGroup(crit_separation=0.01)
            with pytest.raises(ValueError):
                dbscan(starlist['x_0', 'y_0', 'id'])
