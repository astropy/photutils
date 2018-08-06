# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy.testing import assert_allclose
import pytest

from ..bounding_box import BoundingBox

try:
    import matplotlib    # noqa
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
    bbox = BoundingBox._from_float(xmin=1.0, xmax=10.0, ymin=2.0, ymax=20.0)
    assert bbox == BoundingBox(ixmin=1, ixmax=11, iymin=2, iymax=21)

    bbox = BoundingBox._from_float(xmin=1.4, xmax=10.4, ymin=1.6, ymax=10.6)
    assert bbox == BoundingBox(ixmin=1, ixmax=11, iymin=2, iymax=12)


def test_bounding_box_eq():
    bbox = BoundingBox(1, 10, 2, 20)
    assert bbox == bbox

    assert bbox != BoundingBox(9, 10, 2, 20)
    assert bbox != BoundingBox(1, 99, 2, 20)
    assert bbox != BoundingBox(1, 10, 9, 20)
    assert bbox != BoundingBox(1, 10, 2, 99)


def test_bounding_box_repr():
    bbox = BoundingBox(1, 10, 2, 20)

    assert repr(bbox) == 'BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)'
    assert eval(repr(bbox)) == bbox


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
def test_bounding_box_as_patch():
    bbox = BoundingBox(1, 10, 2, 20)

    patch = bbox.as_patch()
    assert_allclose(patch.get_xy(), (0.5, 1.5))
    assert_allclose(patch.get_width(), 9)
    assert_allclose(patch.get_height(), 18)
