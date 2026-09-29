# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch local background kernel.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from astropy.stats import SigmaClip
from numpy.testing import assert_allclose, assert_array_equal

from photutils.background import SExtractorBackground
from photutils.segmentation import (SegmentationImage, SourceCatalog,
                                    detect_sources)
from photutils.segmentation._batch_catalog import batch_local_background
from photutils.segmentation.tests._batch_scene import make_batch_scene


def _annulus_values(cat):
    """
    Gather the usable local background annulus values of each source
    with per-source aperture masks.

    This is a port of the pixel selection of the previous per-source
    loop that the batch kernel replaces, with the cutout data mask
    inlined and the aperture weights applied to the usable pixels only
    (the previous in-place multiplication of the whole cutout raised a
    warning for non-finite data within the aperture).
    """
    values = []
    for aperture in cat.local_background_aperture:
        aperture_mask = aperture.to_mask(method='center')
        slc_lg, slc_sm = aperture_mask.get_overlap_slices(cat._data.shape)

        data_cutout = cat._data[slc_lg].astype(float, copy=True)
        segm_mask_cutout = cat._segmentation_image.data[slc_lg].astype(bool)
        data_mask_cutout = ~np.isfinite(data_cutout)
        if cat._mask is not None:
            data_mask_cutout |= cat._mask[slc_lg]
        data_mask_cutout |= segm_mask_cutout

        aperweight_cutout = aperture_mask.data[slc_sm]
        good_mask = (aperweight_cutout > 0) & ~data_mask_cutout
        values.append(data_cutout[good_mask] * aperweight_cutout[good_mask])
    return values


def _reference_local_background(cat, *, sigma=3.0, maxiters=20):
    """
    Compute the local background of each source as the previous
    per-source loop did, with `SExtractorBackground` and its sigma
    clipping.

    The sigma clipping goes through astropy's fast C implementation,
    whose surviving values are the values within the final clipping
    bounds. It is the reference for the kernel.
    """
    sigma_clip = SigmaClip(sigma=sigma, cenfunc='median', maxiters=maxiters)
    bkg_func = SExtractorBackground(sigma_clip=sigma_clip)

    local_bkgs = []
    for data_values in _annulus_values(cat):
        if len(data_values) < 10:
            local_bkgs.append(0.0)
            continue
        local_bkgs.append(bkg_func(data_values))

    local_bkgs = np.array(local_bkgs)
    local_bkgs[cat._all_masked] = np.nan
    return local_bkgs


def _iterative_local_background(cat, *, sigma=3.0, maxiters=20):
    """
    Compute the local background of each source with iterative-removal
    sigma clipping, where a value dropped in an earlier iteration is
    never included again.

    This is the ``axis=None, masked=False`` code path of
    `~astropy.stats.SigmaClip`. It differs from the reference only when
    the clipping bounds widen between iterations.
    """
    sigma_clip = SigmaClip(sigma=sigma, cenfunc='median', maxiters=maxiters)

    local_bkgs = []
    for data_values in _annulus_values(cat):
        if len(data_values) < 10:
            local_bkgs.append(0.0)
            continue
        values = sigma_clip(data_values, axis=None, masked=False)
        median = np.median(values)
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            local_bkgs.append(mean)
        elif abs(mean - median) / std >= 0.3:
            local_bkgs.append(median)
        else:
            local_bkgs.append(2.5 * median - 1.5 * mean)

    local_bkgs = np.array(local_bkgs)
    local_bkgs[cat._all_masked] = np.nan
    return local_bkgs


def _kernel_local_background(cat, **kwargs):
    """
    Call the kernel directly with the catalog batch arrays.
    """
    arrays = cat._get_batch_arrays()
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    params = {'width': cat.local_bkg_width, 'scale': 1.5, 'sigma': 3.0,
              'maxiters': 20, 'min_pixels': 10}
    params.update(kwargs)
    return batch_local_background(
        arrays['data'], mask=arrays['mask'], segm=arrays['segm'],
        bbox_iymin=iymin, bbox_iymax=iymax, bbox_ixmin=ixmin,
        bbox_ixmax=ixmax, **params)


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('width', [1, 3, 8, 24])
@pytest.mark.parametrize('with_mask', [True, False])
def test_matches_reference(scene, width, with_mask):
    # The scene has sources touching every image edge, close pairs,
    # masked pixels, and non-finite data values
    cat = SourceCatalog(scene['data'], scene['segm'], error=scene['error'],
                        mask=scene['mask'] if with_mask else None,
                        local_bkg_width=width)
    expected = _reference_local_background(cat)
    assert np.all(np.isfinite(expected))
    assert np.any(expected != 0)
    # The reference sums the values in a different order, so the
    # results agree to rounding
    assert_allclose(cat.local_background, expected, rtol=1e-13, atol=0)
    assert_allclose(_kernel_local_background(cat), expected, rtol=1e-13,
                    atol=0)


def _make_noise_scene(seed):
    """
    Make sources of varied sizes on a sloped noisy background with
    outliers, so that the clipping iterates and the estimator branches
    are exercised.
    """
    rng = np.random.default_rng(seed)
    ny = nx = 121
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = rng.normal(0.0, 1.0, (ny, nx)) + 0.01 * xx + 0.02 * yy
    data[rng.random((ny, nx)) < 0.01] += 20.0
    for _ in range(12):
        xc, yc = rng.uniform(5, nx - 5, 2)
        sig = rng.uniform(1.0, 4.0)
        amp = rng.uniform(20.0, 200.0)
        data += amp * np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2)
                             / (2 * sig ** 2))
    segm = detect_sources(data, 5.0, n_pixels=5)
    mask = rng.random((ny, nx)) < 0.02
    return data, segm, mask


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_matches_reference_noise_scene(seed):
    data, segm, mask = _make_noise_scene(seed)
    for width in (2, 6):
        cat = SourceCatalog(data, segm, mask=mask, local_bkg_width=width)
        expected = _reference_local_background(cat)
        assert_allclose(cat.local_background, expected, rtol=1e-13,
                        atol=0)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_matches_reference_quantized_scene(seed):
    # Integer-valued data (e.g., raw counts) has many repeated values,
    # so the median jumps between iterations and the clipping bounds
    # can widen. The reference then includes values clipped in an
    # earlier iteration again, and the kernel must do the same.
    data, segm, mask = _make_noise_scene(seed)
    data = np.round(data * 4)
    for width in (2, 8, 24):
        cat = SourceCatalog(data, segm, mask=mask, local_bkg_width=width)
        expected = _reference_local_background(cat)
        assert_allclose(cat.local_background, expected, rtol=1e-13,
                        atol=0)


def test_final_bounds_readmit_clipped_value():
    # A small case where the clipping bounds widen. Iteration 3 drops
    # the value -1 at a lower bound of -0.84, then the median of the
    # remaining values falls from 1.5 to 1 while the standard deviation
    # shrinks only a little, so the final lower bound is -1.03 and the
    # -1 lies within the final bounds. The kernel must match the
    # reference, which includes the -1 again, rather than the
    # iterative-removal result.
    values = np.array([-1] + [0] * 8 + [1] * 2 + [2] * 2 + [3] * 7
                      + [4] * 2 + [8, 29], dtype=float)
    sigma = 1.5
    shape = (21, 21)
    segm_data = np.zeros(shape, dtype=int)
    segm_data[9:12, 9:12] = 1
    segm = SegmentationImage(segm_data)
    annulus = _annulus_pixels(9, 12, 9, 12, 3, shape)
    iy, ix = np.nonzero(annulus)
    assert iy.size > values.size
    rng = np.random.default_rng(0)
    order = rng.permutation(values.size)
    data = np.zeros(shape)
    data[iy[:values.size], ix[:values.size]] = values[order]
    mask = np.ones(shape, dtype=bool)
    mask[iy[:values.size], ix[:values.size]] = False
    mask[9:12, 9:12] = False  # the source segment
    cat = SourceCatalog(data, segm, mask=mask, local_bkg_width=3)

    sigma_clip = SigmaClip(sigma=sigma, cenfunc='median', maxiters=20)
    clipped = sigma_clip(values, axis=None, masked=False)
    assert clipped.min() == 0.0
    clipped = sigma_clip(values, axis=None, masked=True).compressed()
    assert clipped.min() == -1.0

    expected = SExtractorBackground(sigma_clip=sigma_clip)(values)
    iterative = _iterative_local_background(cat, sigma=sigma)[0]
    assert not np.isclose(expected, iterative)
    assert_allclose(_reference_local_background(cat, sigma=sigma)[0],
                    expected, rtol=1e-13, atol=0)
    assert_allclose(_kernel_local_background(cat, sigma=sigma)[0], expected,
                    rtol=1e-13, atol=0)


def test_maxiters_reached():
    # With a single iteration, the outliers of the noise scene are
    # still being clipped when maxiters is reached, so the statistics
    # of the values within the final bounds are computed anew
    data, segm, mask = _make_noise_scene(1)
    cat = SourceCatalog(data, segm, mask=mask, local_bkg_width=6)
    result = _kernel_local_background(cat, maxiters=1)
    expected = _reference_local_background(cat, maxiters=1)
    assert_allclose(result, expected, rtol=1e-13, atol=0)
    assert not np.allclose(result, _kernel_local_background(cat))


def test_all_values_dropped():
    # Ten annulus pixels of two values in equal number have a median
    # halfway between them and a standard deviation of half of their
    # difference, so a sigma below 1 drops every value in the first
    # iteration. Those bounds are final whether or not maxiters is
    # reached, and no value lies within them.
    shape = (11, 11)
    segm_data = np.zeros(shape, dtype=int)
    segm_data[4:7, 4:7] = 1
    segm = SegmentationImage(segm_data)
    annulus = _annulus_pixels(4, 7, 4, 7, 2, shape)
    iy, ix = np.nonzero(annulus)
    data = np.zeros(shape)
    data[iy[:10], ix[:10]] = np.tile([0.0, 1.0], 5)
    mask = np.ones(shape, dtype=bool)
    mask[iy[:10], ix[:10]] = False
    mask[4:7, 4:7] = False  # the source segment
    cat = SourceCatalog(data, segm, mask=mask, local_bkg_width=2)

    for maxiters in (1, 20):
        result = _kernel_local_background(cat, sigma=0.1, maxiters=maxiters)
        assert np.isnan(result[0])

    # A sigma of 1 keeps the two middle values, which lie exactly one
    # standard deviation from the median
    assert _kernel_local_background(cat, sigma=1.0)[0] == 0.5


def test_zero_width(scene):
    cat = SourceCatalog(scene['data'], scene['segm'], local_bkg_width=0)
    assert_array_equal(cat.local_background, np.zeros(cat.n_labels))


def test_all_masked_source(scene):
    segm = scene['segm']
    mask = scene['mask'].copy()
    slc = segm.slices[0]
    mask[slc] |= segm.data[slc] == segm.labels[0]
    cat = SourceCatalog(scene['data'], segm, mask=mask, local_bkg_width=5)
    result = cat.local_background
    assert cat._all_masked[0]
    assert np.isnan(result[0])
    assert np.all(np.isfinite(result[1:]))
    # The kernel itself still measures the annulus of the masked source
    assert np.isfinite(_kernel_local_background(cat)[0])


def test_few_pixels():
    # Fewer than min_pixels usable annulus pixels gives zero
    data = np.zeros((11, 11))
    data[4:7, 4:7] = 100.0
    segm_data = np.zeros((11, 11), dtype=int)
    segm_data[4:7, 4:7] = 1
    segm = SegmentationImage(segm_data)
    mask = np.ones((11, 11), dtype=bool)
    mask[3:8, 3:8] = False  # the inner rectangle only, no annulus pixel
    cat = SourceCatalog(data, segm, mask=mask, local_bkg_width=2)
    assert cat.local_background[0] == 0.0
    assert _reference_local_background(cat)[0] == 0.0
    assert _kernel_local_background(cat, min_pixels=1)[0] == 0.0
    # No usable pixel at all gives NaN when min_pixels allows it
    assert np.isnan(_kernel_local_background(cat, min_pixels=0)[0])

    # A single usable annulus pixel is measured only when min_pixels
    # allows it
    mask[2, 5] = False
    data[2, 5] = 2.0
    cat = SourceCatalog(data, segm, mask=mask, local_bkg_width=2)
    assert cat.local_background[0] == 0.0
    assert _reference_local_background(cat)[0] == 0.0
    assert _kernel_local_background(cat, min_pixels=2)[0] == 0.0
    assert _kernel_local_background(cat, min_pixels=1)[0] == 2.0


def _annulus_pixels(ixmin, ixmax, iymin, iymax, width, shape, *, scale=1.5):
    """
    Return the (y, x) indices of the pixel centers within the local
    background annulus of a segment bounding box, computed directly
    from the annulus definition.
    """
    xpos = 0.5 * (ixmin + ixmax - 1)
    ypos = 0.5 * (iymin + iymax - 1)
    half_w_in = 0.5 * (ixmax - ixmin) * scale
    half_h_in = 0.5 * (iymax - iymin) * scale
    half_w_out = half_w_in + width
    half_h_out = half_h_in + width
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dx = np.abs(xx - xpos)
    dy = np.abs(yy - ypos)
    outer = (dx < half_w_out) & (dy < half_h_out)
    inner = (dx < half_w_in) & (dy < half_h_in)
    return outer & ~inner


@pytest.mark.parametrize(('nx_src', 'ny_src', 'width'),
                         [(3, 3, 1), (2, 3, 1), (4, 2, 2), (5, 5, 3)])
def test_annulus_pixels_and_median(nx_src, ny_src, width):
    # Small annuli with even and odd pixel counts, with the pixel
    # membership checked against the annulus definition and the
    # estimator against the SExtractor mode of the pixel values
    shape = (21, 21)
    ixmin, iymin = 8, 7
    ixmax, iymax = ixmin + nx_src, iymin + ny_src
    rng = np.random.default_rng(nx_src * 10 + ny_src)
    data = rng.normal(10.0, 1.0, shape)
    segm_data = np.zeros(shape, dtype=int)
    segm_data[iymin:iymax, ixmin:ixmax] = 1
    segm = SegmentationImage(segm_data)
    cat = SourceCatalog(data, segm, local_bkg_width=width)

    annulus = _annulus_pixels(ixmin, ixmax, iymin, iymax, width, shape)
    values = data[annulus]
    assert values.size >= 10
    aperture_mask = cat.local_background_aperture[0].to_mask(
        method='center')
    assert_array_equal(aperture_mask.to_image(shape) > 0, annulus)

    sigma_clip = SigmaClip(sigma=3.0, cenfunc='median', maxiters=20)
    expected = SExtractorBackground(sigma_clip=sigma_clip)(values)
    assert_allclose(cat.local_background[0], expected, rtol=1e-13,
                    atol=0)


def test_estimator_branches():
    # A constant annulus (zero standard deviation) gives the mean, and
    # a strongly skewed annulus gives the median; both are exact
    shape = (31, 31)
    segm_data = np.zeros(shape, dtype=int)
    segm_data[13:18, 13:18] = 1
    segm = SegmentationImage(segm_data)

    data = np.full(shape, 3.0)
    cat = SourceCatalog(data, segm, local_bkg_width=4)
    assert cat.local_background[0] == 3.0

    data = np.zeros(shape)
    data[::2, ::2] = 1.5
    cat = SourceCatalog(data, segm, local_bkg_width=4)
    annulus = _annulus_pixels(13, 18, 13, 18, 4, shape)
    values = data[annulus]
    assert np.all(np.abs(values - np.median(values))
                  <= 3 * np.std(values))
    assert abs(np.mean(values) - np.median(values)) / np.std(values) >= 0.3
    assert cat.local_background[0] == np.median(values)


def test_clipping_removes_outliers():
    # Outliers in the annulus are clipped and do not bias the result
    shape = (41, 41)
    segm_data = np.zeros(shape, dtype=int)
    segm_data[18:23, 18:23] = 1
    segm = SegmentationImage(segm_data)
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 0.1, shape)
    data[10, 10] = 1000.0
    data[30, 30] = -1000.0
    cat = SourceCatalog(data, segm, local_bkg_width=8)
    assert_allclose(cat.local_background[0], 5.0, atol=0.05)
    expected = _reference_local_background(cat)
    assert_allclose(cat.local_background, expected, rtol=1e-13, atol=0)


def test_sliced_and_scalar_catalog(scene):
    cat = SourceCatalog(scene['data'], scene['segm'], mask=scene['mask'],
                        local_bkg_width=6)
    expected = cat.local_background
    assert_array_equal(cat[2:5].local_background, expected[2:5])
    assert_array_equal(cat[3].local_background, expected[3:4])


def test_input_dtypes(scene):
    # The kernel reads float32 data and int32 segmentation images
    # directly, with the same results as the same values input as
    # float64 and intp (to within rounding, because the compiler may
    # fuse multiply-adds differently in the two specializations)
    data32 = scene['data'].astype(np.float32)
    segm32 = SegmentationImage(scene['segm'].data.astype(np.int32))
    cat = SourceCatalog(data32, segm32, local_bkg_width=8)
    arrays = cat._get_batch_arrays()
    assert arrays['data'].dtype == np.float32
    assert arrays['segm'].dtype == np.int32

    segm_ref = SegmentationImage(scene['segm'].data.astype(np.intp))
    cat_ref = SourceCatalog(data32.astype(np.float64), segm_ref,
                            local_bkg_width=8)
    assert_allclose(cat.local_background, cat_ref.local_background,
                    rtol=1e-13, atol=0)
    assert_allclose(_kernel_local_background(cat),
                    _kernel_local_background(cat_ref), rtol=1e-13, atol=0)


def test_invalid_inputs(scene):
    cat = SourceCatalog(scene['data'], scene['segm'], local_bkg_width=3)
    arrays = cat._get_batch_arrays()
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    kwargs = {'width': 3, 'scale': 1.5, 'sigma': 3.0, 'maxiters': 20,
              'min_pixels': 10}

    match = 'bbox_ixmax must have the same length as bbox_iymin'
    with pytest.raises(ValueError, match=match):
        batch_local_background(arrays['data'], mask=arrays['mask'],
                               segm=arrays['segm'], bbox_iymin=iymin,
                               bbox_iymax=iymax, bbox_ixmin=ixmin,
                               bbox_ixmax=ixmax[:-1], **kwargs)

    match = 'mask must have the same shape as data'
    with pytest.raises(ValueError, match=match):
        batch_local_background(arrays['data'], mask=arrays['mask'][1:],
                               segm=arrays['segm'], bbox_iymin=iymin,
                               bbox_iymax=iymax, bbox_ixmin=ixmin,
                               bbox_ixmax=ixmax, **kwargs)

    match = 'maxiters must be at least 1'
    with pytest.raises(ValueError, match=match):
        _kernel_local_background(cat, maxiters=0)


def test_n_threads(scene):
    cat = SourceCatalog(scene['data'], scene['segm'], mask=scene['mask'],
                        local_bkg_width=6)
    cat_threaded = SourceCatalog(scene['data'], scene['segm'],
                                 mask=scene['mask'], local_bkg_width=6,
                                 n_threads=4)
    assert cat.n_labels >= 4
    assert_array_equal(cat_threaded.local_background,
                       cat.local_background)


def test_thread_safety(scene):
    cat = SourceCatalog(scene['data'], scene['segm'], mask=scene['mask'],
                        local_bkg_width=6)
    expected = _kernel_local_background(cat)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _kernel_local_background(cat),
                                range(8)))
    for result in results:
        assert_array_equal(result, expected)
