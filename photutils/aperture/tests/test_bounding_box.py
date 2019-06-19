# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the bounding_box module.
"""

from numpy.testing import assert_allclose
import pytest

from ..bounding_box import BoundingBox
from ..rectangle import RectangularAperture

try:
    import matplotlib  # noqa
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def test_bounding_box_init():
    bbox = BoundingBox(1, 10, 2, 20)
    assert bbox.ixmin == 1
    assert bbox.ixmax == 10
    assert bbox.iymin == 2
    assert bbox.iymax == 20


def test_bounding_box_init_minmax():
    with pytest.raises(ValueError):
        BoundingBox(100, 1, 1, 100)
    with pytest.raises(ValueError):
        BoundingBox(1, 100, 100, 1)


def test_bounding_box_inputs():
    with pytest.raises(TypeError):
        BoundingBox([1], [10], [2], [9])
    with pytest.raises(TypeError):
        BoundingBox([1, 2], 10, 2, 9)
    with pytest.raises(TypeError):
        BoundingBox(1.0, 10.0, 2.0, 9.0)
    with pytest.raises(TypeError):
        BoundingBox(1.3, 10, 2, 9)
    with pytest.raises(TypeError):
        BoundingBox(1, 10.3, 2, 9)
    with pytest.raises(TypeError):
        BoundingBox(1, 10, 2.3, 9)
    with pytest.raises(TypeError):
        BoundingBox(1, 10, 2, 9.3)


def test_bounding_box_from_float():
    # This is the example from the method docstring
    bbox = BoundingBox.from_float(xmin=1.0, xmax=10.0, ymin=2.0, ymax=20.0)
    assert bbox == BoundingBox(ixmin=1, ixmax=11, iymin=2, iymax=21)

    bbox = BoundingBox.from_float(xmin=1.4, xmax=10.4, ymin=1.6, ymax=10.6)
    assert bbox == BoundingBox(ixmin=1, ixmax=11, iymin=2, iymax=12)


def test_bounding_box_eq():
    bbox = BoundingBox(1, 10, 2, 20)
    assert bbox == BoundingBox(1, 10, 2, 20)
    assert bbox != BoundingBox(9, 10, 2, 20)
    assert bbox != BoundingBox(1, 99, 2, 20)
    assert bbox != BoundingBox(1, 10, 9, 20)
    assert bbox != BoundingBox(1, 10, 2, 99)

    with pytest.raises(TypeError):
        assert bbox == (1, 10, 2, 20)


def test_bounding_box_repr():
    bbox = BoundingBox(1, 10, 2, 20)
    assert repr(bbox) == 'BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)'


def test_bounding_box_shape():
    bbox = BoundingBox(1, 10, 2, 20)
    assert bbox.shape == (18, 9)


def test_bounding_box_slices():
    bbox = BoundingBox(1, 10, 2, 20)
    assert bbox.slices == (slice(2, 20), slice(1, 10))


def test_bounding_box_extent():
    bbox = BoundingBox(1, 10, 2, 20)
    assert_allclose(bbox.extent, (0.5, 9.5, 1.5, 19.5))


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_bounding_box_as_artist():
    bbox = BoundingBox(1, 10, 2, 20)
    patch = bbox.as_artist()

    assert_allclose(patch.get_xy(), (0.5, 1.5))
    assert_allclose(patch.get_width(), 9)
    assert_allclose(patch.get_height(), 18)


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_bounding_box_to_aperture():
    bbox = BoundingBox(1, 10, 2, 20)
    aper = RectangularAperture((5, 10.5), w=9., h=18., theta=0.)
    bbox_aper = bbox.to_aperture()
    assert_allclose(bbox_aper.positions, aper.positions)

    assert bbox_aper.w == aper.w
    assert bbox_aper.h == aper.h
    assert bbox_aper.theta == aper.theta


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_bounding_box_plot():
    # TODO: check the content of the plot
    bbox = BoundingBox(1, 10, 2, 20)
    bbox.plot()


def test_bounding_box_union():
    bbox1 = BoundingBox(1, 10, 2, 20)
    bbox2 = BoundingBox(5, 21, 7, 32)
    bbox_union_expected = BoundingBox(1, 21, 2, 32)
    bbox_union1 = bbox1 | bbox2
    bbox_union2 = bbox1.union(bbox2)

    assert bbox_union1 == bbox_union_expected
    assert bbox_union1 == bbox_union2

    with pytest.raises(TypeError):
        bbox1.union((5, 21, 7, 32))


def test_bounding_box_intersect():
    bbox1 = BoundingBox(1, 10, 2, 20)
    bbox2 = BoundingBox(5, 21, 7, 32)
    bbox_intersect_expected = BoundingBox(5, 10, 7, 20)
    bbox_intersect1 = bbox1 & bbox2
    bbox_intersect2 = bbox1.intersection(bbox2)

    assert bbox_intersect1 == bbox_intersect_expected
    assert bbox_intersect1 == bbox_intersect2

    with pytest.raises(TypeError):
        bbox1.intersection((5, 21, 7, 32))

    assert bbox1.intersection(BoundingBox(30, 40, 50, 60)) is None
