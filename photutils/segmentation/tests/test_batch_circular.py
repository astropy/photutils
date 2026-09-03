# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch circular photometry path.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture import CircularAperture
from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog,
                                                       reference_aperture_data)


def _reference_aperture_to_mask(cat, aperture, **kwargs):
    # Verbatim port of the previous ``_aperture_to_mask`` method
    bbox = aperture.bbox
    max_size = max(cat._data.size, 1_000_000)
    if bbox.shape[0] * bbox.shape[1] > max_size:
        return None
    return aperture.to_mask(**kwargs)


def _reference_aperture_photometry(cat, apertures, **kwargs):
    """
    Compute the aperture flux and flux error for each source.

    This is a verbatim port of the per-source ``_aperture_photometry``
    method that the batch Cython driver replaces. It is the numerical
    reference for the driver.
    """
    flux = []
    flux_err = []
    for label, aperture, bkg in zip(cat.labels, apertures,
                                    cat._local_background, strict=True):
        if aperture is None:
            flux.append(np.nan)
            flux_err.append(np.nan)
            continue

        xcen, ycen = aperture.positions
        aperture_mask = _reference_aperture_to_mask(cat, aperture, **kwargs)
        if aperture_mask is None:
            flux.append(np.nan)
            flux_err.append(np.nan)
            continue

        data, error, mask, _, slc_sm, _ = reference_aperture_data(
            cat, label, xcen, ycen, aperture_mask.bbox, bkg)

        aperture_weights = aperture_mask.data[slc_sm]
        pixel_mask = (aperture_weights > 0) & ~mask
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

    return np.array(flux), np.array(flux_err)


def _reference_circular_photometry(cat, radius):
    apertures = cat._make_circular_apertures(radius)
    kwargs = cat._aperture_mask_kwargs['circ']
    return _reference_aperture_photometry(cat, apertures, **kwargs)


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('method', ['correct', 'mask', 'none'])
@pytest.mark.parametrize('with_error', [True, False])
@pytest.mark.parametrize('with_mask', [True, False])
@pytest.mark.parametrize('radius', [1.5, 4.0, 12.0])
def test_matches_reference(scene, method, with_error, with_mask, radius):
    cat = make_catalog(scene, aperture_mask_method=method,
                       with_error=with_error, with_mask=with_mask)
    ref_flux, ref_err = _reference_circular_photometry(cat, radius)
    flux, flux_err = cat.circular_photometry(radius)
    assert_allclose(flux, ref_flux, rtol=1e-12, equal_nan=True)
    assert_allclose(flux_err, ref_err, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize('kwargs', [{'method': 'center'},
                                    {'method': 'subpixel', 'subpixels': 5}])
def test_overlap_methods(scene, kwargs):
    cat = make_catalog(scene)
    cat._aperture_mask_kwargs['circ'] = kwargs
    ref_flux, ref_err = _reference_circular_photometry(cat, 4.0)
    flux, flux_err = cat.circular_photometry(4.0)
    assert_allclose(flux, ref_flux, rtol=1e-12, equal_nan=True)
    assert_allclose(flux_err, ref_err, rtol=1e-12, equal_nan=True)


def test_local_background(scene):
    cat = SourceCatalog(scene['data'], scene['segm'], error=scene['error'],
                        mask=scene['mask'], local_bkg_width=6)
    ref_flux, ref_err = _reference_circular_photometry(cat, 4.0)
    flux, flux_err = cat.circular_photometry(4.0)
    assert_allclose(flux, ref_flux, rtol=1e-12, equal_nan=True)
    assert_allclose(flux_err, ref_err, rtol=1e-12, equal_nan=True)


def test_all_masked_and_nonfinite_centroid(scene):
    # A fully masked source has no centroid and a NaN flux. A source
    # whose centroid is non-finite is also NaN
    data = scene['data'].copy()
    mask = scene['mask'].copy()
    segm = scene['segm']
    slc = segm.slices[0]
    mask[slc] |= segm.data[slc] == segm.labels[0]
    cat = SourceCatalog(data, segm, mask=mask, error=scene['error'])
    ref_flux, ref_err = _reference_circular_photometry(cat, 3.0)
    flux, flux_err = cat.circular_photometry(3.0)
    assert np.isnan(flux[0])
    assert np.isnan(flux_err[0])
    assert_allclose(flux, ref_flux, rtol=1e-12, equal_nan=True)
    assert_allclose(flux_err, ref_err, rtol=1e-12, equal_nan=True)


def test_oom_guard_and_off_image():
    # A circle whose bounding box exceeds the 1M-pixel guard is NaN,
    # and a centroid far outside the data gives a NaN flux
    data = np.zeros((21, 21))
    data[8:13, 8:13] = 10.0
    segm_data = np.zeros((21, 21), dtype=int)
    segm_data[8:13, 8:13] = 1
    segm = SegmentationImage(segm_data)
    cat = SourceCatalog(data, segm)
    assert_allclose(cat.circular_photometry(3.0)[0],
                    _reference_circular_photometry(cat, 3.0)[0])
    flux, flux_err = cat.circular_photometry(600.0)
    assert np.isnan(flux[0])
    assert np.isnan(flux_err[0])

    cat.__dict__['x_centroid'] = np.array([500.0])
    cat.__dict__['y_centroid'] = np.array([500.0])
    flux, flux_err = cat.circular_photometry(3.0)
    assert np.isnan(flux[0])
    assert np.isnan(flux_err[0])


def test_scalar_catalog(scene):
    cat = make_catalog(scene)[0]
    ref_flux, ref_err = _reference_circular_photometry(cat, 4.0)
    flux, flux_err = cat.circular_photometry(4.0)
    assert np.isscalar(flux)
    assert np.isscalar(flux_err)
    assert_allclose(flux, ref_flux[0], rtol=1e-12)
    assert_allclose(flux_err, ref_err[0], rtol=1e-12)


def test_aperture_objects_unchanged(scene):
    # The public make_circular_apertures helper still returns the
    # aperture objects (None where undefined)
    cat = make_catalog(scene)
    apertures = cat.make_circular_apertures(3.0)
    assert len(apertures) == cat.n_labels
    assert all(isinstance(aper, CircularAperture) for aper in apertures)
