# Licensed under a 3-clause BSD style license - see LICENSE.rst

# These tests test the wrappers that do not require Aperture objects but
# instead ensure that the apertures are created and broadcast correctly.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
import numpy as np

from ..aperture import aperture_circular, \
                       annulus_circular, \
                       aperture_elliptical, \
                       annulus_elliptical


class TestBroadcastingCircular(object):

    def setup_class(self):
        self.data = np.ones((10, 10), dtype=np.float)

    def test_single_object_single_same_aperture(self):
        # Input single object and single aperture
        flux = aperture_circular(self.data, 5., 5., 2.)
        assert np.isscalar(flux) == True

    def test_single_object_multiple_same_apertures(self):
        # One object, multiple apertures (same for each object)
        flux = aperture_circular(self.data, 2., 5., [[2.], [5.]])
        assert flux.shape == (2, 1)

    def test_multiple_objects_single_same_aperture(self):
        # Multiple objects, single aperture (same for each object)
        flux = aperture_circular(self.data, [2., 3., 4.], [5., 6., 7.], 5.)
        assert flux.shape == (3, )

    def test_multiple_objects_single_diff_apertures(self):
        # Multiple objects, single apertures (different for each object)
        flux = aperture_circular(self.data, [2., 3., 4.], [5., 6., 7.],
                                 [2., 3., 4.])
        assert flux.shape == (3, )

    def test_multiple_objects_multiple_aperture_per_object(self):
        # Multiple objects, multiple apertures per object (same for each object)
        flux = aperture_circular(self.data, [2., 3., 4.], [5., 6., 7.],
                                 [[2.], [5.]])
        assert flux.shape == (2, 3)

    def test_mismatch_object_apertures(self):
        # Mismatched number of objects and apertures
        with pytest.raises(ValueError) as exc:
            aperture_circular(self.data, [2., 3., 4.], [5., 6., 7.],
                              [2., 3.])
        assert exc.value.args[0] == "trailing dimension of 'apertures' must be 1 or match length of xc, yc"


class TestBroadcastingCircularAnnulus(object):

    def setup_class(self):
        self.data = np.ones((10, 10), dtype=np.float)

    def test_single_object_single_same_aperture(self):
        # Input single object and single aperture
        flux = annulus_circular(self.data, 5., 5., 2., 5.)
        assert np.isscalar(flux) == True

    def test_single_object_multiple_same_apertures(self):
        # One object, multiple apertures (same for each object)
        flux = annulus_circular(self.data, 2., 5.,
                                [[2.], [5.]], [[5.], [8.]])
        assert flux.shape == (2, 1)

    def test_multiple_objects_single_same_aperture(self):
        # Multiple objects, single aperture (same for each object)
        flux = annulus_circular(self.data, [2., 3., 4.], [5., 6., 7.],
                                2., 5.)
        assert flux.shape == (3, )

    def test_multiple_objects_single_diff_apertures(self):
        # Multiple objects, single apertures (different for each object)
        flux = annulus_circular(self.data, [2., 3., 4.], [5., 6., 7.],
                                [2., 3., 4.], [5., 6., 7.])
        assert flux.shape == (3, )

    def test_multiple_objects_multiple_aperture_per_object(self):
        # Multiple objects, multiple apertures per object (same for each object)
        flux = annulus_circular(self.data, [2., 3., 4.], [5., 6., 7.],
                                [[2.], [5.]], [[5.], [8.]])
        assert flux.shape == (2, 3)

    def test_mismatch_object_apertures(self):
        # Mismatched number of objects and apertures
        with pytest.raises(ValueError) as exc:
            annulus_circular(self.data, [2., 3., 4.], [5., 6., 7.],
                             [2., 3.], [5., 6.])
        assert exc.value.args[0] == "trailing dimension of 'apertures' must be 1 or match length of xc, yc"


class TestBroadcastingElliptical(object):

    def setup_class(self):
        self.data = np.ones((10, 10), dtype=np.float)

    def test_single_object_single_same_aperture(self):
        # Input single object and single aperture
        flux = aperture_elliptical(self.data, 5., 5., 2., 5., np.pi / 4.)
        assert np.isscalar(flux) == True

    def test_single_object_multiple_same_apertures(self):
        # One object, multiple apertures (same for each object)
        flux = aperture_elliptical(self.data, 2., 5.,
                                   [[2.], [5.]], [[5.], [8.]], np.pi / 2.)
        assert flux.shape == (2, 1)

    def test_multiple_objects_single_same_aperture(self):
        # Multiple objects, single aperture (same for each object)
        flux = aperture_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                                   2., 5., np.pi / 2.)
        assert flux.shape == (3, )

    def test_multiple_objects_single_diff_apertures(self):
        # Multiple objects, single apertures (different for each object)
        flux = aperture_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                                   [2., 3., 4.], [5., 6., 7.], np.pi / 2.)
        assert flux.shape == (3, )

    def test_multiple_objects_multiple_aperture_per_object(self):
        # Multiple objects, multiple apertures per object (same for each object)
        flux = aperture_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                                   [[2.], [5.]], [[5.], [8.]], np.pi / 2.)
        assert flux.shape == (2, 3)

    def test_mismatch_object_apertures(self):
        # Mismatched number of objects and apertures
        with pytest.raises(ValueError) as exc:
            aperture_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                               [2., 3.], [5., 6.], [5., 6.], np.pi / 2.)
        assert exc.value.args[0] == "trailing dimension of 'apertures' must be 1 or match length of xc, yc"


class TestBroadcastingEllipticalAnnulus(object):

    def setup_class(self):
        self.data = np.ones((10, 10), dtype=np.float)

    def test_single_object_single_same_aperture(self):
        # Input single object and single aperture
        flux = annulus_elliptical(self.data, 5., 5., 2., 5., 5., np.pi / 4.)
        assert np.isscalar(flux) == True

    def test_single_object_multiple_same_apertures(self):
        # One object, multiple apertures (same for each object)
        flux = annulus_elliptical(self.data, 2., 5.,
                           [[2.], [5.]], [[5.], [8.]], [[5.], [8.]], np.pi / 2.)
        assert flux.shape == (2, 1)

    def test_multiple_objects_single_same_aperture(self):
        # Multiple objects, single aperture (same for each object)
        flux = annulus_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                           2., 5., 5., np.pi / 2.)
        assert flux.shape == (3, )

    def test_multiple_objects_single_diff_apertures(self):
        # Multiple objects, single apertures (different for each object)
        flux = annulus_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                           [2., 3., 4.], [5., 6., 7.], [5., 6., 7.], np.pi / 2.)
        assert flux.shape == (3, )

    def test_multiple_objects_multiple_aperture_per_object(self):
        # Multiple objects, multiple apertures per object (same for each object)
        flux = annulus_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                           [[2.], [5.]], [[5.], [8.]], [[5.], [8.]], np.pi / 2.)
        assert flux.shape == (2, 3)

    def test_mismatch_object_apertures(self):
        # Mismatched number of objects and apertures
        with pytest.raises(ValueError) as exc:
            annulus_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                               [2., 3.], [5., 6.], [5., 6.], np.pi / 2.)
        assert exc.value.args[0] == "trailing dimension of 'apertures' must be 1 or match length of xc, yc"
