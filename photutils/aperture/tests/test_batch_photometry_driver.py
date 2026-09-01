# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the low-level batch Cython aperture photometry driver.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from photutils.aperture._batch_photometry import (FLAG_COL_BBOX_CLIPPED,
                                                  FLAG_COL_N_PIXELS,
                                                  SHAPE_CIRCLE,
                                                  SHAPE_CIRCULAR_ANNULUS,
                                                  SHAPE_ELLIPSE,
                                                  SHAPE_ELLIPTICAL_ANNULUS,
                                                  SHAPE_RECTANGLE,
                                                  SHAPE_RECTANGULAR_ANNULUS,
                                                  batch_aperture_sums)

N_THREADS = 8
N_CALLS_PER_THREAD = 4


def _batch_inputs():
    """
    Build deterministic data, error, mask, and source positions for the
    batch-driver tests.
    """
    rng = np.random.default_rng(42)
    data = rng.random((80, 80))
    error = rng.random((80, 80)) + 0.1
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[::7, ::5] = 1
    positions = np.array([[20.0, 25.0], [40.0, 40.0], [55.0, 30.0],
                          [10.0, 60.0], [70.0, 70.0], [35.0, 15.0]])
    return data, error, mask, positions


# Aperture shape specs: (shape_code, params, ext_x, ext_y).
# The half-extents need only bound the aperture; they are identical
# across the baseline and concurrent calls, so the comparison is exact.
_BATCH_SPECS = [
    (SHAPE_CIRCLE, [8.0], 8.0, 8.0),
    (SHAPE_CIRCULAR_ANNULUS, [5.0, 8.0], 8.0, 8.0),
    (SHAPE_ELLIPSE, [8.0, 5.0, 0.7], 8.0, 8.0),
    (SHAPE_ELLIPTICAL_ANNULUS, [4.0, 2.0, 8.0, 5.0, 0.7], 8.0, 8.0),
    (SHAPE_RECTANGLE, [12.0, 7.0, 0.5], 7.0, 7.0),
    (SHAPE_RECTANGULAR_ANNULUS, [6.0, 4.0, 12.0, 7.0, 0.5], 7.0, 7.0),
]


class TestBatchApertureSums:
    """
    Tests for the batch_aperture_sums Cython driver.
    """

    @pytest.mark.parametrize('use_exact', [1, 0])
    def test_readonly_arrays(self, use_exact):
        """
        Test that the batch driver accepts read-only (non-writeable) data,
        error, positions, and params arrays and returns results identical to
        writeable arrays.

        The data, error, positions, and params arguments are declared as
        ``const`` typed memoryviews so that read-only arrays do not raise a
        ``ValueError``.
        """
        data, error, mask, positions = _batch_inputs()
        params = np.array([8.0], dtype=np.float64)

        expected = batch_aperture_sums(data, error, mask, positions,
                                       SHAPE_CIRCLE, params, 8.0, 8.0,
                                       0.0, 0.0, use_exact, 8)

        for arr in (data, error, positions, params):
            arr.setflags(write=False)
        result = batch_aperture_sums(data, error, mask, positions,
                                     SHAPE_CIRCLE, params, 8.0, 8.0,
                                     0.0, 0.0, use_exact, 8)

        for res_arr, exp_arr in zip(result, expected, strict=True):
            assert_array_equal(res_arr, exp_arr)

    @pytest.mark.parametrize(('shape_code', 'params', 'ext_x', 'ext_y'),
                             _BATCH_SPECS)
    @pytest.mark.parametrize('use_exact', [1, 0])
    def test_threadsafe(self, shape_code, params, ext_x, ext_y, use_exact):
        data, error, mask, positions = _batch_inputs()
        params = np.array(params, dtype=np.float64)

        def fn():
            return batch_aperture_sums(data, error, mask, positions,
                                       shape_code, params, ext_x, ext_y,
                                       0.0, 0.0, use_exact, 8)

        expected = fn()
        with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futures = [ex.submit(fn)
                       for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
            for future in futures:
                result = future.result()
                for res_arr, exp_arr in zip(result, expected, strict=True):
                    assert_array_equal(res_arr, exp_arr)

    def test_mixed_concurrent(self):
        """
        Run every aperture shape through the batch driver concurrently.

        Mixing shapes within the thread pool surfaces interference between
        calls if any shared mutable state (e.g., a module-level scratch
        buffer) were introduced into the ``nogil`` source loop.
        """
        data, error, mask, positions = _batch_inputs()

        def task(spec):
            shape_code, params, ext_x, ext_y = spec
            params = np.array(params, dtype=np.float64)
            return batch_aperture_sums(data, error, mask, positions,
                                       shape_code, params, ext_x, ext_y,
                                       0.0, 0.0, 1, 5)

        expected = {spec[0]: task(spec) for spec in _BATCH_SPECS}

        with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futures = {ex.submit(task, spec): spec[0]
                       for spec in _BATCH_SPECS for _ in range(N_THREADS)}
            for fut, shape_code in futures.items():
                result = fut.result()
                for res_arr, exp_arr in zip(result, expected[shape_code],
                                            strict=True):
                    assert_array_equal(res_arr, exp_arr)


def test_params_per_source_circle():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 40))
    positions = np.array([[10.0, 10.0], [25.0, 25.0], [30.0, 12.0]])
    radii = np.array([2.0, 3.5, 5.0])
    psrc = radii[:, None].copy()
    batch = batch_aperture_sums(
        data, None, None, positions, SHAPE_CIRCLE, None, 0.0, 0.0,
        0.0, 0.0, 1, 1, params_per_source=psrc)
    for i, r in enumerate(radii):
        single = batch_aperture_sums(
            data, None, None, positions[i:i + 1], SHAPE_CIRCLE,
            np.array([r]), r, r, 0.0, 0.0, 1, 1)
        assert_allclose(batch[0][i], single[0][0], rtol=1e-12)


def test_params_per_source_ellipse():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 40))
    positions = np.array([[10.0, 10.0], [25.0, 25.0], [30.0, 12.0]])
    psrc = np.array([[3.0, 2.0, 0.5],
                     [4.0, 1.0, -0.3],
                     [2.5, 2.5, 1.2]])
    batch = batch_aperture_sums(
        data, None, None, positions, SHAPE_ELLIPSE, None, 0.0, 0.0,
        0.0, 0.0, 1, 1, params_per_source=psrc)
    for i, (semi_a, semi_b, theta) in enumerate(psrc):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        ext_x = np.sqrt((semi_a * cos_theta)**2 + (semi_b * sin_theta)**2)
        ext_y = np.sqrt((semi_a * sin_theta)**2 + (semi_b * cos_theta)**2)
        single = batch_aperture_sums(
            data, None, None, positions[i:i + 1], SHAPE_ELLIPSE,
            np.array([semi_a, semi_b, theta]), ext_x, ext_y, 0.0, 0.0,
            1, 1)
        assert_allclose(batch[0][i], single[0][0], rtol=1e-12)


def test_params_per_source_invalid():
    data = np.ones((20, 20))
    positions = np.array([[10.0, 10.0], [12.0, 12.0]])
    psrc = np.array([[2.0], [3.0]])
    params = np.array([2.0])

    match = 'params must be given when params_per_source is None'
    with pytest.raises(ValueError, match=match):
        batch_aperture_sums(data, None, None, positions, SHAPE_CIRCLE,
                            None, 2.0, 2.0, 0.0, 0.0, 1, 1)

    match = 'give params or params_per_source, not both'
    with pytest.raises(ValueError, match=match):
        batch_aperture_sums(data, None, None, positions, SHAPE_CIRCLE,
                            params, 2.0, 2.0, 0.0, 0.0, 1, 1,
                            params_per_source=psrc)

    match = 'params_per_source supports only circle and ellipse shapes'
    with pytest.raises(ValueError, match=match):
        batch_aperture_sums(data, None, None, positions,
                            SHAPE_CIRCULAR_ANNULUS, None, 2.0, 2.0,
                            0.0, 0.0, 1, 1, params_per_source=psrc)

    match = 'params_per_source does not support emit_sum'
    with pytest.raises(ValueError, match=match):
        batch_aperture_sums(data, None, None, positions, SHAPE_CIRCLE,
                            None, 2.0, 2.0, 0.0, 0.0, 1, 1, None, None,
                            0, None, 1, params_per_source=psrc)

    match = 'params_per_source must have one row per position'
    with pytest.raises(ValueError, match=match):
        batch_aperture_sums(data, None, None, positions, SHAPE_CIRCLE,
                            None, 2.0, 2.0, 0.0, 0.0, 1, 1,
                            params_per_source=psrc[:1])

    match = 'params_per_source has the wrong column count'
    with pytest.raises(ValueError, match=match):
        batch_aperture_sums(data, None, None, positions, SHAPE_CIRCLE,
                            None, 2.0, 2.0, 0.0, 0.0, 1, 1,
                            params_per_source=np.zeros((2, 3)))
    with pytest.raises(ValueError, match=match):
        batch_aperture_sums(data, None, None, positions, SHAPE_ELLIPSE,
                            None, 2.0, 2.0, 0.0, 0.0, 1, 1,
                            params_per_source=np.zeros((2, 1)))


def test_weights_out():
    data = np.ones((20, 20))
    positions = np.array([[10.0, 10.0],   # interior
                          [1.0, 10.0],    # aperture off left edge
                          [19.4, 10.0]])  # aperture off right edge
    params = np.array([3.0])
    result = batch_aperture_sums(
        data, None, None, positions, SHAPE_CIRCLE, params, 3.0, 3.0,
        0.0, 0.0, 1, 1)
    weights_out = result.weights_out
    assert list(weights_out) == [0, 1, 1]


def test_weights_out_clipped_bbox_zero_weight():
    """
    Test that a bbox row can poke off-image while every off-image pixel
    has zero aperture fraction.

    ``weights_out`` must then be 0 even though ``FLAG_COL_BBOX_CLIPPED``
    is 1. With method='center' (``use_exact=0``, ``subpixels=1``), a
    circle of r=2.0 at y=1.4 has iymin = floor(1.4 - 2 + 0.5) = -1
    (clipped), but the off-image pixel centers at y=-1 lie 2.4 > r from
    the center, so their center-method fraction is zero.
    """
    data = np.ones((20, 20))
    positions = np.array([[10.0, 1.4]])
    params = np.array([2.0])
    result = batch_aperture_sums(
        data, None, None, positions, SHAPE_CIRCLE, params, 2.0, 2.0,
        0.0, 0.0, 0, 1)
    fcounts = result.flag_counts
    weights_out = result.weights_out
    assert fcounts[0, FLAG_COL_BBOX_CLIPPED] == 1
    assert weights_out[0] == 0

    # The same aperture with the exact method has positive area
    # off-image
    result = batch_aperture_sums(
        data, None, None, positions, SHAPE_CIRCLE, params, 2.0, 2.0,
        0.0, 0.0, 1, 1)
    assert result.weights_out[0] == 1


@pytest.mark.parametrize('radius', [2000.0, 8000.0, 20000.0])
def test_weights_out_large_aperture(radius):
    """
    Test an aperture whose bounding box is far larger than the data.

    Once a nonzero-fraction pixel outside the data has been found, the
    remaining outside pixels cannot change the outside-weight result,
    so the pixel loop narrows back to the part of the bounding box
    inside the data. Without that, the cost of these apertures would
    grow with the (enormous) bounding-box area rather than with the
    data area.
    """
    data = np.ones((200, 200))
    positions = np.array([[100.0, 100.0]])
    result = batch_aperture_sums(
        data, None, None, positions, SHAPE_CIRCLE, np.array([radius]),
        radius, radius, 0.0, 0.0, 1, 5)

    assert result.weights_out[0] == 1
    assert result.flag_counts[0, FLAG_COL_BBOX_CLIPPED] == 1
    # Every data pixel is well inside the aperture, so the sums are
    # unaffected by the outside-weight scan
    assert result.flag_counts[0, FLAG_COL_N_PIXELS] == data.size
    assert_allclose(result.sums[0], data.sum())
    assert_allclose(result.areas[0], data.size)
