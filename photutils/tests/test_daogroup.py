# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
from numpy.testing import assert_array_equal
from astropy.table import Table, vstack
from daogroup import daogroup

class TestDAOGROUP(object):
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

        x_0 = np.array([0, np.sqrt(2)/4, np.sqrt(2)/4, -np.sqrt(2)/4,
                        -np.sqrt(2)/4])
        y_0 = np.array([0, np.sqrt(2)/4, -np.sqrt(2)/4, np.sqrt(2)/4,
                        -np.sqrt(2)/4])
        x_1 = x_0 + 2.0
        first_group = Table([x_0, y_0, np.arange(len(x_0))],
                            names=('x_0', 'y_0', 'id'))
        second_group = Table([x_1, y_0, len(x_0) + np.arange(len(x_0))],
                             names=('x_0', 'y_0', 'id'))
        starlist = vstack([first_group, second_group])
        true_groups = [first_group, second_group]
        daogroup_groups = daogroup(starlist, crit_separation=0.6)
        assert_array_equal(true_groups, daogroup_groups)


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

        first_group = Table([np.zeros(5), np.linspace(0,1,5), np.arange(5)],
                            names=('x_0', 'y_0', 'id'))
        second_group = Table([np.zeros(5), np.linspace(2,3,5),
                              5+np.arange(5)], names=('x_0', 'y_0', 'id'))
        starlist = vstack([first_group, second_group])
        true_groups = [first_group, second_group]
        daogroup_groups = daogroup(starlist, crit_separation=0.3)
        assert_array_equal(true_groups, daogroup_groups)


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
        
        first_group = Table([np.linspace(0,1,5), np.zeros(5), np.arange(5)],
                            names=('x_0', 'y_0', 'id'))
        second_group = Table([np.linspace(2,3,5), np.zeros(5),
                              5+np.arange(5)], names=('x_0', 'y_0', 'id'))
        starlist = vstack([first_group, second_group])
        true_groups = [first_group, second_group]
        daogroup_groups = daogroup(starlist, crit_separation=0.3)
        assert_array_equal(true_groups, daogroup_groups)

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

        x = np.linspace(-1.,1.,5)
        y = np.sqrt(1. - x**2)
        xx = np.hstack((x,x))
        yy = np.hstack((y,-y))
        starlist = Table([xx, yy, np.arange(10)], names=('x_0', 'y_0', 'id'))
        daogroup_group = daogroup(starlist, crit_separation=2.5)
        assert_array_equal([starlist], daogroup_group)

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

        first_group = Table([np.full((5,), 1.5), np.linspace(0,1,5),
                             np.arange(5)], names=('x_0', 'y_0', 'id'))
        second_group = Table([np.full((5,), 1.5), np.linspace(2,3,5),
                              5+np.arange(5)], names=('x_0', 'y_0', 'id'))
        third_group = Table([np.linspace(0,1,5), np.full((5,), 1.5),
                             10+np.arange(5)],
                            names=('x_0', 'y_0', 'id'))
        fourth_group = Table([np.linspace(2,3,5), np.full((5,), 1.5),
                              15+np.arange(5)], names=('x_0', 'y_0', 'id'))
        starlist = vstack([first_group, second_group, third_group,
                           fourth_group])
        true_groups = [first_group, second_group, third_group, fourth_group]
        daogroup_groups = daogroup(starlist, crit_separation=0.3)
        assert_array_equal(true_groups, daogroup_groups)

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

        x_0 = np.array([0, np.sqrt(2)/4, np.sqrt(2)/4, -np.sqrt(2)/4,
                        -np.sqrt(2)/4])
        y_0 = np.array([0, np.sqrt(2)/4, -np.sqrt(2)/4, np.sqrt(2)/4,
                        -np.sqrt(2)/4])
        x_1 = x_0 + 2.0
        x_2 = x_0 + 4.0
        first_group = Table([x_0, y_0, np.arange(5)],
                            names=('x_0', 'y_0', 'id'))
        second_group = Table([x_1, y_0, 5+np.arange(5)],
                             names=('x_0', 'y_0', 'id'))
        third_group = Table([x_2, y_0, 10+np.arange(5)],
                            names=('x_0', 'y_0', 'id'))
        starlist = vstack([first_group, second_group, third_group])
        true_groups = [first_group, second_group, third_group]
        daogroup_groups = daogroup(starlist, crit_separation=0.6)
        assert_array_equal(true_groups, daogroup_groups)
