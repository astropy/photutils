# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the AperturePhotometry class.
"""

from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData, StdDevUncertainty
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture.circle import (CircularAnnulus, CircularAperture,
                                       SkyCircularAperture)
from photutils.aperture.photometry import (AperturePhotometry,
                                           aperture_photometry)
from photutils.aperture.polygon import PolygonAperture
from photutils.aperture.stats import ApertureStats
from photutils.datasets import make_4gaussians_image, make_wcs
from photutils.segmentation import SegmentationImage
from photutils.utils._optional_deps import HAS_REGIONS


def make_scene():
    """
    Build a deterministic scene with a target source (label 1) and a
    bright neighbor source (label 2), on a nonzero background.
    """
    data = np.ones((50, 50))
    segm = np.zeros((50, 50), dtype=int)

    # Target source (label 1)
    data[18:25, 18:25] = 10.0
    segm[18:25, 18:25] = 1

    # Bright neighbor source (label 2)
    data[20:25, 26:32] = 100.0
    segm[20:25, 26:32] = 2

    return data, segm


class TestAperturePhotometryParity:
    """
    The class must return results identical to the legacy
    ``aperture_photometry`` function for all shared inputs.
    """

    def test_single_aperture(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        ref = aperture_photometry(data, aper)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert_allclose(phot.area.value, ref['area'].value)
        assert_equal(phot.flags, ref['flags'])

    def test_multiple_positions(self):
        data = make_4gaussians_image()
        aper = CircularAperture(((150, 25), (90, 60)), 10)
        phot = AperturePhotometry(data, aper)
        ref = aperture_photometry(data, aper)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert phot.flux.shape == (2,)

    def test_error_propagation(self):
        data = make_4gaussians_image()
        error = np.sqrt(np.abs(data))
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, error=error)
        ref = aperture_photometry(data, aper, error=error)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert_allclose(phot.flux_err, ref['aperture_sum_err'])

    def test_no_error_is_nan(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        assert np.all(np.isnan(phot.flux_err))

    def test_mask(self):
        data = make_4gaussians_image()
        mask = np.zeros(data.shape, dtype=bool)
        mask[25, 150] = True
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, mask=mask)
        ref = aperture_photometry(data, aper, mask=mask)
        assert_allclose(phot.flux, ref['aperture_sum'])

    def test_nomask(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, mask=np.ma.nomask)
        ref = aperture_photometry(data, aper)
        assert phot._mask is None
        assert_allclose(phot.flux, ref['aperture_sum'])

    @pytest.mark.parametrize(('method', 'subpixels'),
                             [('exact', 5), ('center', 5), ('subpixel', 7)])
    def test_method_variants(self, method, subpixels):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, method=method,
                                  subpixels=subpixels)
        ref = aperture_photometry(data, aper, method=method,
                                  subpixels=subpixels)
        assert_allclose(phot.flux, ref['aperture_sum'])

    def test_units(self):
        data = make_4gaussians_image() * u.Jy
        error = np.sqrt(np.abs(data.value)) * u.Jy
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, error=error)
        ref = aperture_photometry(data, aper, error=error)
        assert phot.flux.unit == u.Jy
        assert phot.flux_err.unit == u.Jy
        assert phot.area.unit == u.pix**2
        assert_allclose(phot.flux.value, ref['aperture_sum'].value)
        assert_allclose(phot.flux_err.value, ref['aperture_sum_err'].value)


class TestListOfApertures:
    def test_flux_shape(self):
        data = make_4gaussians_image()
        pos = ((150, 25), (90, 60))
        apers = [CircularAperture(pos, r) for r in (5, 8)]
        phot = AperturePhotometry(data, apers)
        assert phot.flux.shape == (2, 2)
        assert phot.area.shape == (2, 2)
        assert phot.flags.shape == (2, 2)

    def test_matches_per_aperture(self):
        data = make_4gaussians_image()
        pos = ((150, 25), (90, 60))
        apers = [CircularAperture(pos, r) for r in (5, 8)]
        phot = AperturePhotometry(data, apers)
        for i, aper in enumerate(apers):
            ref = aperture_photometry(data, aper)
            assert_allclose(phot.flux[:, i], ref['aperture_sum'])

    def test_identical_positions_required(self):
        data = make_4gaussians_image()
        apers = [CircularAperture((150, 25), 5),
                 CircularAperture((90, 60), 5)]
        match = 'Input apertures must all have identical positions'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, apers)


class TestSkyApertures:
    def test_sky_center_from_pixel_wcs(self):
        data = make_4gaussians_image()
        wcs = make_wcs(data.shape)
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, wcs=wcs)
        assert isinstance(phot.sky_center, SkyCoord)
        assert 'sky_center' in phot.to_table().colnames

    def test_no_wcs_sky_center_none(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        assert phot.sky_center is None
        assert 'sky_center' not in phot.to_table().colnames

    def test_sky_aperture(self):
        data = make_4gaussians_image()
        wcs = make_wcs(data.shape)
        skycoord = wcs.pixel_to_world(150, 25)
        sky_aper = SkyCircularAperture(skycoord, r=0.7 * u.arcsec)
        pix_aper = sky_aper.to_pixel(wcs)
        phot = AperturePhotometry(data, sky_aper, wcs=wcs)
        ref = aperture_photometry(data, pix_aper)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert isinstance(phot.sky_center, SkyCoord)
        assert phot.sky_center.isscalar is False

    def test_sky_aperture_multiple_positions(self):
        data = make_4gaussians_image()
        wcs = make_wcs(data.shape)
        skycoord = wcs.pixel_to_world([150, 90], [25, 60])
        sky_aper = SkyCircularAperture(skycoord, r=0.7 * u.arcsec)
        pix_aper = sky_aper.to_pixel(wcs)
        phot = AperturePhotometry(data, sky_aper, wcs=wcs)
        ref = aperture_photometry(data, pix_aper)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert phot.sky_center.isscalar is False
        assert len(phot.sky_center) == 2

    def test_sky_aperture_requires_wcs(self):
        data = np.ones((11, 11))
        wcs = make_wcs(data.shape)
        sky_aper = CircularAperture((5, 5), r=3).to_sky(wcs=wcs)
        match = 'A WCS transform must be defined'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, sky_aper)


class TestNDDataInput:
    def test_nddata(self):
        data = make_4gaussians_image()
        error = np.sqrt(np.abs(data))
        uncertainty = StdDevUncertainty(error)
        nddata = NDData(data, uncertainty=uncertainty)
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(nddata, aper)
        ref = aperture_photometry(data, aper, error=error)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert_allclose(phot.flux_err, ref['aperture_sum_err'])

    def test_nddata_ignored_keywords_warn(self):
        data = make_4gaussians_image()
        nddata = NDData(data)
        aper = CircularAperture((150, 25), 8)
        mask = np.zeros(data.shape, dtype=bool)
        match = 'is obtained from the input NDData object'
        with pytest.warns(AstropyUserWarning, match=match):
            AperturePhotometry(nddata, aper, mask=mask)

    def test_nddata_units(self):
        data = make_4gaussians_image()
        nddata = NDData(data * u.Jy)
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(nddata, aper)
        assert phot.flux.unit == u.Jy

    def test_nddata_uncertainty_with_unit(self):
        data = make_4gaussians_image()
        error = np.sqrt(np.abs(data))
        uncertainty = StdDevUncertainty(error, unit=u.Jy)
        nddata = NDData(data * u.Jy, uncertainty=uncertainty)
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(nddata, aper)
        ref = aperture_photometry(data * u.Jy, aper, error=error * u.Jy)
        assert phot.flux_err.unit == u.Jy
        assert_allclose(phot.flux_err.value, ref['aperture_sum_err'].value)


class TestSegmentationMasking:
    @pytest.mark.parametrize('use_segm_obj', [True, False])
    def test_mask_method_matches_manual(self, use_segm_obj):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        segm_in = SegmentationImage(segm) if use_segm_obj else segm
        phot = AperturePhotometry(data, aper, segmentation_image=segm_in,
                                  labels=[1], mask_method='mask')
        manual_mask = (segm > 0) & (segm != 1)
        ref = AperturePhotometry(data, aper, mask=manual_mask)
        assert_allclose(phot.flux, ref.flux)

    def test_source_only_matches_manual(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        phot = AperturePhotometry(data, aper, segmentation_image=segm,
                                  labels=[1], mask_method='source_only')
        manual_mask = segm != 1
        ref = AperturePhotometry(data, aper, mask=manual_mask)
        assert_allclose(phot.flux, ref.flux)

    def test_none_method_ignores_segmentation(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        phot = AperturePhotometry(data, aper, segmentation_image=segm,
                                  mask_method='none')
        ref = AperturePhotometry(data, aper)
        assert_allclose(phot.flux, ref.flux)

    def test_correct_matches_neighbor(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=10)
        none_phot = AperturePhotometry(data, aper, mask_method='none')
        corr_phot = AperturePhotometry(data, aper, segmentation_image=segm,
                                       labels=[1], mask_method='correct')
        assert corr_phot.flux[0] != none_phot.flux[0]

    def test_polygon_mask_path(self):
        data, segm = make_scene()
        offsets = np.array([[-7, -7], [9, -7], [9, 9], [-7, 9]])
        aper = PolygonAperture((21, 21), offsets)
        phot = AperturePhotometry(data, aper, segmentation_image=segm,
                                  labels=[1], mask_method='mask')
        manual_mask = (segm > 0) & (segm != 1)
        ref = AperturePhotometry(data, aper, mask=manual_mask)
        assert_allclose(phot.flux, ref.flux)

    def test_labels_required(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        match = 'labels must be input when segmentation_image is input'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper, segmentation_image=segm,
                               mask_method='mask')


class TestToTable:
    def test_default_columns(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper).to_table()
        assert tbl.colnames == ['id', 'x_center', 'y_center', 'flux',
                                'flux_err', 'area', 'flags']

    def test_default_columns_with_wcs(self):
        data = make_4gaussians_image()
        wcs = make_wcs(data.shape)
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper, wcs=wcs).to_table()
        assert tbl.colnames == ['id', 'x_center', 'y_center', 'sky_center',
                                'flux', 'flux_err', 'area', 'flags']

    def test_columns_subset(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper).to_table(columns=['id', 'flux'])
        assert tbl.colnames == ['id', 'flux']

    def test_columns_single_string(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper).to_table(columns='flux')
        assert tbl.colnames == ['flux']

    def test_multi_aperture_suffixes(self):
        data = make_4gaussians_image()
        pos = ((150, 25), (90, 60))
        apers = [CircularAperture(pos, r) for r in (5, 8)]
        tbl = AperturePhotometry(data, apers).to_table()
        assert tbl.colnames == ['id', 'x_center', 'y_center', 'flux_0',
                                'flux_1', 'flux_err_0', 'flux_err_1',
                                'area_0', 'area_1', 'flags_0', 'flags_1']

    def test_meta(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper).to_table()
        assert 'version' in tbl.meta
        assert 'aperture_photometry_args' in tbl.meta
        assert tbl.meta['aperture'] == 'CircularAperture'


class TestFlagsAndArea:
    def test_area_matches_area_overlap(self):
        data = np.ones((25, 25), dtype=float)
        aper = CircularAperture([(10, 10), (15, 15)], r=4.0)
        phot = AperturePhotometry(data, aper)
        assert phot.area.unit == u.pix**2
        assert_allclose(phot.area.value, aper.area_overlap(data))

    def test_partial_overlap_flag(self):
        data = np.ones((25, 25), dtype=float)
        aper = CircularAperture((0, 12), r=5.0)
        phot = AperturePhotometry(data, aper)
        assert phot.flags[0] != 0

    def test_decode_flags(self):
        data = np.ones((25, 25))
        mask = np.zeros(data.shape, dtype=bool)
        mask[12, 12] = True
        aper = CircularAperture([(12.0, 12.0), (0.0, 12.0)], r=3.0)
        phot = AperturePhotometry(data, aper, mask=mask)
        decoded = phot.decode_flags()
        assert decoded[0] == ['masked_pixels']
        assert decoded[1] == ['partial_overlap']

    def test_decode_flags_bit_values(self):
        data = np.ones((25, 25))
        mask = np.zeros(data.shape, dtype=bool)
        mask[12, 12] = True
        aper = CircularAperture([(12.0, 12.0)], r=3.0)
        phot = AperturePhotometry(data, aper, mask=mask)
        decoded = phot.decode_flags(return_bit_values=True)
        assert decoded[0] == [8]


class TestNonFiniteData:
    """
    Non-finite ``data`` values (NaN and inf) must be automatically
    masked (excluded from flux, flux_err, and area) and reported via the
    ``non_finite_data`` flag, mirroring ``ApertureStats``.
    """

    def test_batch_path_masks_nonfinite(self):
        data = np.ones((25, 25))
        data[12, 12] = np.nan
        data[11, 11] = np.inf
        aper = CircularAperture((12, 12), r=5)
        phot = AperturePhotometry(data, aper)
        # The two non-finite interior pixels (each of weight 1) are
        # excluded from the flux and area (all other pixels are 1.0).
        assert np.isfinite(phot.flux[0])
        assert_allclose(phot.flux[0], aper.area_overlap(data) - 2)
        assert_allclose(phot.area.value[0], phot.flux[0])
        assert phot.flags[0] == 32
        assert phot.decode_flags()[0] == ['non_finite_data']

    def test_mask_path_masks_nonfinite(self):
        # Disable the batch Cython driver to exercise the slower
        # mask-based code path (via a spec of None).
        aper = CircularAperture((12, 12), r=5)
        data = np.ones((25, 25))
        data[12, 12] = np.nan
        data[11, 11] = np.inf
        with patch.object(CircularAperture, '_batch_shape_params',
                          lambda _self: None):
            phot = AperturePhotometry(data, aper)
            flux = phot.flux[0]
            flags = phot.flags[0]
            decoded = phot.decode_flags()[0]
        assert np.isfinite(flux)
        assert_allclose(flux, aper.area_overlap(data) - 2)
        assert flags == 32
        assert decoded == ['non_finite_data']

    @pytest.mark.parametrize('aperture_type', ['circle', 'polygon'])
    def test_parity_with_aperture_stats(self, aperture_type):
        data = np.ones((25, 25))
        data[12, 12] = np.nan
        data[11, 11] = np.inf
        if aperture_type == 'circle':
            aper = CircularAperture((12, 12), r=5)
        else:
            offsets = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]])
            aper = PolygonAperture((12, 12), offsets)
        phot = AperturePhotometry(data, aper)
        stats = ApertureStats(data, aper)
        assert_allclose(phot.flux[0], stats.sum)
        assert_allclose(phot.area.value[0], stats.sum_aper_area.value)
        assert phot.flags[0] == stats.flags

    def test_nonfinite_outside_aperture_not_flagged(self):
        data = np.ones((25, 25))
        data[0, 0] = np.nan  # far outside the aperture
        aper = CircularAperture((12, 12), r=5)
        phot = AperturePhotometry(data, aper)
        assert phot.flags[0] == 0
        assert_allclose(phot.flux[0], aper.area_overlap(data))

    def test_combined_mask_and_nonfinite(self):
        data = np.ones((25, 25))
        data[12, 12] = np.nan
        mask = np.zeros(data.shape, dtype=bool)
        mask[11, 12] = True
        aper = CircularAperture((12, 12), r=5)
        phot = AperturePhotometry(data, aper, mask=mask)
        assert np.isfinite(phot.flux[0])
        # Both masked_pixels (8) and non_finite_data (32) are set.
        assert phot.flags[0] == 8 | 32
        assert set(phot.decode_flags()[0]) == {'masked_pixels',
                                               'non_finite_data'}

    def test_nonfinite_at_masked_pixel_counts_as_masked(self):
        # A non-finite pixel that is also input-masked is reported as
        # masked_pixels, not non_finite_data (matching ApertureStats).
        data = np.ones((25, 25))
        data[12, 12] = np.nan
        mask = np.zeros(data.shape, dtype=bool)
        mask[12, 12] = True
        aper = CircularAperture((12, 12), r=5)
        phot = AperturePhotometry(data, aper, mask=mask)
        assert phot.flags[0] == 8
        stats = ApertureStats(data, aper, mask=mask)
        assert phot.flags[0] == stats.flags

    def test_legacy_function_still_poisons_nonfinite(self):
        # The legacy aperture_photometry function keeps the 3.0.0
        # behavior: non-finite data poison the sum (not masked).
        data = np.ones((25, 25))
        data[12, 12] = np.nan
        aper = CircularAperture((12, 12), r=5)
        tbl = aperture_photometry(data, aper)
        assert np.isnan(tbl['aperture_sum'][0])

        # The mask-based path (PolygonAperture) also poisons.
        offsets = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]])
        poly = PolygonAperture((12, 12), offsets)
        tbl = aperture_photometry(data, poly)
        assert np.isnan(tbl['aperture_sum'][0])


class TestInputValidation:
    def test_data_not_2d(self):
        aper = CircularAperture((5, 5), r=3)
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(np.ones((5, 5, 5)), aper)

    def test_error_shape_mismatch(self):
        data = np.ones((11, 11))
        aper = CircularAperture((5, 5), r=3)
        match = 'data and error must have the same shape'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper, error=np.ones((5, 5)))

    def test_mask_shape_mismatch(self):
        data = np.ones((11, 11))
        aper = CircularAperture((5, 5), r=3)
        match = 'data and mask must have the same shape'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper, mask=np.zeros((5, 5), dtype=bool))

    def test_unit_mismatch(self):
        data = np.ones((11, 11)) * u.Jy
        aper = CircularAperture((5, 5), r=3)
        with pytest.raises(ValueError, match='must all have the same units'):
            AperturePhotometry(data, aper, error=np.ones((11, 11)))


class TestReprAndImmutability:
    def test_repr(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        assert 'AperturePhotometry' in repr(phot)
        assert "method='exact'" in repr(phot)

    def test_str(self):
        data = make_4gaussians_image()
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        assert 'AperturePhotometry' in str(phot)

    def test_no_new_attributes_after_init(self):
        """
        Only lazyproperty cache entries may appear after ``__init__``,
        which is required for the instance to be thread-safe.
        """
        data = make_4gaussians_image()
        aper = CircularAperture(((150, 25), (90, 60)), 8)
        phot = AperturePhotometry(data, aper, error=np.ones_like(data),
                                  wcs=make_wcs(data.shape))
        init_keys = set(vars(phot).keys())

        # Touch every public and private lazy property/attribute
        for name in ('id', 'x_center', 'y_center', 'sky_center', 'flux',
                     'flux_err', 'area', 'flags', 'n_positions'):
            getattr(phot, name)
        phot.to_table()
        phot.decode_flags()

        new_keys = set(vars(phot).keys()) - init_keys
        lazy_names = {'_photometry_results', '_positions', 'n_positions',
                      'id', 'x_center', 'y_center', 'sky_center', 'flux',
                      'flux_err', 'area', 'flags'}
        assert new_keys.issubset(lazy_names)

    def test_concurrent_access(self):
        from concurrent.futures import ThreadPoolExecutor

        data = make_4gaussians_image()
        aper = CircularAperture(((150, 25), (90, 60)), 8)
        phot = AperturePhotometry(data, aper)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: phot.flux, range(8)))
        for result in results:
            assert_allclose(result, results[0])


@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
class TestRegionInput:
    def test_region_matches_aperture(self):
        from regions import CirclePixelRegion, PixCoord

        data = np.ones((40, 40), dtype=float)
        error = np.ones(data.shape, dtype=float)
        position = (20.0, 20.0)
        r = 10.0
        region = CirclePixelRegion(PixCoord(*position), r)
        aper = CircularAperture(position, r)
        phot = AperturePhotometry(data, region, error=error)
        ref = AperturePhotometry(data, aper, error=error)
        assert_allclose(phot.flux, ref.flux)
        assert_allclose(phot.flux_err, ref.flux_err)

    def test_annulus_region(self):
        from regions import CircleAnnulusPixelRegion, PixCoord

        data = np.ones((40, 40), dtype=float)
        position = (20.0, 20.0)
        region = CircleAnnulusPixelRegion(PixCoord(*position), 8.0, 10.0)
        aper = CircularAnnulus(position, 8.0, 10.0)
        phot = AperturePhotometry(data, region)
        ref = AperturePhotometry(data, aper)
        assert_allclose(phot.flux, ref.flux)
