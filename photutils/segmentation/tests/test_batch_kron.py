# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch Kron photometry path.
"""

import math
import warnings

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture import BoundingBox, CircularAperture
from photutils.geometry import circular_overlap_grid, elliptical_overlap_grid
from photutils.segmentation import SourceCatalog
from photutils.segmentation.flags import SEGMENTATION_FLAGS
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)


def _reference_kron_photometry(cat, kron_aperture):
    """
    Compute the Kron flux, flux error, and flags for each source.

    This is a verbatim port of the per-source Python implementation
    that the batch Cython driver replaces. It is the numerical
    reference for the driver.

    Returns
    -------
    flux, flux_err, kron_flags : tuple of `~numpy.ndarray`
        The Kron flux, flux error, and bitwise quality flags.
    """
    labels = cat.labels

    _floor = math.floor
    max_size = max(cat._data.size, 1_000_000)
    ny_img, nx_img = cat._data.shape

    flux = []
    flux_err = []
    kron_flags = []
    for label, aperture, bkg in zip(labels, kron_aperture,
                                    cat._local_background, strict=True):
        if aperture is None:
            flux.append(np.nan)
            flux_err.append(np.nan)
            kron_flags.append(SEGMENTATION_FLAGS.KRON_UNDEFINED)
            continue

        xcen, ycen = aperture.positions

        # Compute the aperture mask directly, bypassing the
        # aperture's to_mask() method and ApertureMask/BoundingBox
        # property overhead.
        if isinstance(aperture, CircularAperture):
            r = aperture.r
            ixmin = _floor(xcen - r + 0.5)
            ixmax = _floor(xcen + r + 1.5)
            iymin = _floor(ycen - r + 0.5)
            iymax = _floor(ycen + r + 1.5)
            nx = ixmax - ixmin
            ny = iymax - iymin
            if nx * ny > max_size:
                flux.append(np.nan)
                flux_err.append(np.nan)
                kron_flags.append(SEGMENTATION_FLAGS.KRON_UNDEFINED)
                continue
            edges = (ixmin - 0.5 - xcen, ixmax - 0.5 - xcen,
                     iymin - 0.5 - ycen, iymax - 0.5 - ycen)
            mask_data = circular_overlap_grid(
                edges[0], edges[1], edges[2], edges[3],
                nx, ny, r, 1, 1)
        else:
            a = aperture.a
            b = aperture.b
            theta_val = aperture.theta
            theta_rad = (theta_val.to_value(u.radian)
                         if hasattr(theta_val, 'to')
                         else float(theta_val))
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            x_ext = math.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
            y_ext = math.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)
            ixmin = _floor(xcen - x_ext + 0.5)
            ixmax = _floor(xcen + x_ext + 1.5)
            iymin = _floor(ycen - y_ext + 0.5)
            iymax = _floor(ycen + y_ext + 1.5)
            nx = ixmax - ixmin
            ny = iymax - iymin
            if nx * ny > max_size:
                flux.append(np.nan)
                flux_err.append(np.nan)
                kron_flags.append(SEGMENTATION_FLAGS.KRON_UNDEFINED)
                continue
            edges = (ixmin - 0.5 - xcen, ixmax - 0.5 - xcen,
                     iymin - 0.5 - ycen, iymax - 0.5 - ycen)
            mask_data = elliptical_overlap_grid(
                edges[0], edges[1], edges[2], edges[3],
                nx, ny, a, b, theta_rad, 1, 1)

        bbox = BoundingBox(ixmin, ixmax, iymin, iymax)
        (data, error, mask, _, slc_sm,
         flag_masks) = cat._make_aperture_data(label, xcen, ycen, bbox,
                                               bkg)
        if data is None:
            flux.append(np.nan)
            flux_err.append(np.nan)
            kron_flags.append(SEGMENTATION_FLAGS.KRON_NO_OVERLAP)
            continue

        aperture_weights = mask_data[slc_sm]
        in_aperture = aperture_weights > 0
        pixel_mask = in_aperture & ~mask

        kron_flag = 0
        # The aperture bounding box extends beyond the data array.
        # The box corners can have zero aperture weight, so compare
        # the number of nonzero-weight pixels within the data to the
        # total number in the aperture. The overlap is partial only
        # if at least one nonzero-weight pixel falls both inside and
        # outside of the data.
        if (ixmin < 0 or iymin < 0 or ixmax > nx_img
                or iymax > ny_img):
            n_inside = np.count_nonzero(in_aperture)
            if n_inside == 0:
                kron_flag |= SEGMENTATION_FLAGS.KRON_NO_OVERLAP
            elif n_inside != np.count_nonzero(mask_data):
                kron_flag |= SEGMENTATION_FLAGS.KRON_PARTIAL_OVERLAP

        if np.any(flag_masks['data_mask'] & in_aperture):
            kron_flag |= SEGMENTATION_FLAGS.KRON_MASKED_PIXELS

        segm_mask = flag_masks['segm_mask']
        if segm_mask is not None and np.any(segm_mask & in_aperture):
            kron_flag |= SEGMENTATION_FLAGS.KRON_NEIGHBOR_PIXELS

        uncorrected_mask = flag_masks['uncorrected_mask']
        if (uncorrected_mask is not None
                and np.any(uncorrected_mask & in_aperture)):
            kron_flag |= SEGMENTATION_FLAGS.KRON_UNCORRECTED_PIXELS

        kron_flags.append(kron_flag)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = (aperture_weights * data)[pixel_mask]
            flux_ = np.nan if values.shape == (0,) else np.sum(values)
            flux.append(flux_)

            if error is None:
                flux_err_ = np.nan
            else:
                values = (aperture_weights**2 * error**2)[pixel_mask]
                if values.shape == (0,):
                    flux_err_ = np.nan
                else:
                    flux_err_ = np.sqrt(np.sum(values))
            flux_err.append(flux_err_)

    flux = np.array(flux)
    flux_err = np.array(flux_err)
    kron_flags = np.array(kron_flags, dtype=int)

    return flux, flux_err, kron_flags


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('method', ['correct', 'mask', 'none'])
@pytest.mark.parametrize('with_error', [True, False])
@pytest.mark.parametrize('with_mask', [True, False])
def test_matches_reference(scene, method, with_error, with_mask):
    cat = make_catalog(scene, aperture_mask_method=method,
                       with_error=with_error, with_mask=with_mask)
    kron_aperture = cat._array('kron_aperture')
    ref_flux, ref_err, ref_flags = _reference_kron_photometry(
        cat, kron_aperture)
    flux, flux_err, kron_flags = cat._calc_kron_photometry()
    assert_allclose(flux, ref_flux, rtol=1e-12, equal_nan=True)
    assert_allclose(flux_err, ref_err, rtol=1e-12, equal_nan=True)
    assert np.array_equal(kron_flags, ref_flags)


def test_custom_kron_params(scene):
    cat = make_catalog(scene)
    result = cat.kron_photometry((1.8, 1.0))
    apertures = cat._make_kron_apertures(
        cat._validate_kron_params((1.8, 1.0)))
    ref_flux, ref_err, _ = _reference_kron_photometry(cat, apertures)
    assert_allclose(result[0], ref_flux, rtol=1e-12, equal_nan=True)
    assert_allclose(result[1], ref_err, rtol=1e-12, equal_nan=True)


def test_mixed_aperture_types(scene):
    # A minimum circular radius makes the sources whose scaled Kron
    # ellipse is smaller than that radius circular, while the larger
    # sources stay elliptical; then both driver groups run in one
    # _calc_kron_photometry call
    cat = SourceCatalog(scene['data'], scene['segm'],
                        error=scene['error'], mask=scene['mask'],
                        aperture_mask_method='correct',
                        kron_params=(2.5, 1.4, 5.0))
    apertures = cat._array('kron_aperture')
    types = {type(ap).__name__ for ap in apertures if ap is not None}
    assert types == {'CircularAperture', 'EllipticalAperture'}
    ref = _reference_kron_photometry(cat, apertures)
    result = cat._calc_kron_photometry()
    for got, want in zip(result, ref, strict=True):
        assert_allclose(np.asarray(got, dtype=float),
                        np.asarray(want, dtype=float), rtol=1e-12,
                        equal_nan=True)
