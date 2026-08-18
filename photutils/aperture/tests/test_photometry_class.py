# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the AperturePhotometry class.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData, StdDevUncertainty
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture.circle import (CircularAnnulus, CircularAperture,
                                       SkyCircularAperture)
from photutils.aperture.flags import APERTURE_FLAGS
from photutils.aperture.photometry import (AperturePhotometry,
                                           aperture_photometry)
from photutils.aperture.polygon import PolygonAperture
from photutils.aperture.stats import ApertureStats
from photutils.aperture.tests.conftest import (NoBatchCircularAperture,
                                               make_scene)
from photutils.datasets import make_wcs
from photutils.segmentation import SegmentationImage
from photutils.utils._optional_deps import HAS_REGIONS


class TestAperturePhotometryParity:
    """
    The class must return results identical to the legacy
    ``aperture_photometry`` function for all shared inputs.
    """

    def test_single_aperture(self, data):
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        ref = aperture_photometry(data, aper)
        assert_allclose(phot.flux, ref['aperture_sum'])

    def test_multiple_positions(self, data):
        aper = CircularAperture(((150, 25), (90, 60)), 10)
        phot = AperturePhotometry(data, aper)
        ref = aperture_photometry(data, aper)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert phot.flux.shape == (2,)

    def test_error_propagation(self, data):
        error = np.sqrt(np.abs(data))
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, error=error)
        ref = aperture_photometry(data, aper, error=error)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert_allclose(phot.flux_err, ref['aperture_sum_err'])

    def test_no_error_is_nan(self, data):
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        assert np.all(np.isnan(phot.flux_err))

    def test_mask(self, data):
        mask = np.zeros(data.shape, dtype=bool)
        mask[25, 150] = True
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, mask=mask)
        ref = aperture_photometry(data, aper, mask=mask)
        assert_allclose(phot.flux, ref['aperture_sum'])

    def test_nomask(self, data):
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, mask=np.ma.nomask)
        ref = aperture_photometry(data, aper)
        assert phot._mask is None
        assert_allclose(phot.flux, ref['aperture_sum'])

    @pytest.mark.parametrize(('method', 'subpixels'),
                             [('exact', 5), ('center', 5), ('subpixel', 7)])
    def test_method_variants(self, method, subpixels, data):
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, method=method,
                                  subpixels=subpixels)
        ref = aperture_photometry(data, aper, method=method,
                                  subpixels=subpixels)
        assert_allclose(phot.flux, ref['aperture_sum'])

    def test_units(self, data):
        data = data * u.Jy
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
    def test_flux_shape(self, data):
        pos = ((150, 25), (90, 60))
        apers = [CircularAperture(pos, r) for r in (5, 8)]
        phot = AperturePhotometry(data, apers)
        assert phot.flux.shape == (2, 2)
        assert phot.area.shape == (2, 2)
        assert phot.flags.shape == (2, 2)

    def test_matches_per_aperture(self, data):
        pos = ((150, 25), (90, 60))
        apers = [CircularAperture(pos, r) for r in (5, 8)]
        phot = AperturePhotometry(data, apers)
        for i, aper in enumerate(apers):
            ref = aperture_photometry(data, aper)
            assert_allclose(phot.flux[:, i], ref['aperture_sum'])

    def test_identical_positions_required(self, data):
        apers = [CircularAperture((150, 25), 5),
                 CircularAperture((90, 60), 5)]
        match = 'Input apertures must all have identical positions'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, apers)


class TestSkyApertures:
    def test_sky_center_from_pixel_wcs(self, data):
        wcs = make_wcs(data.shape)
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper, wcs=wcs)
        assert isinstance(phot.sky_center, SkyCoord)
        assert 'sky_center' in phot.to_table().colnames

    def test_no_wcs_sky_center_none(self, data):
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        assert phot.sky_center is None
        assert 'sky_center' not in phot.to_table().colnames

    def test_sky_aperture(self, data):
        wcs = make_wcs(data.shape)
        skycoord = wcs.pixel_to_world(150, 25)
        sky_aper = SkyCircularAperture(skycoord, r=0.7 * u.arcsec)
        pix_aper = sky_aper.to_pixel(wcs)
        phot = AperturePhotometry(data, sky_aper, wcs=wcs)
        ref = aperture_photometry(data, pix_aper)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert isinstance(phot.sky_center, SkyCoord)
        assert phot.sky_center.isscalar is True

    def test_sky_aperture_multiple_positions(self, data):
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
    def test_nddata(self, data):
        error = np.sqrt(np.abs(data))
        uncertainty = StdDevUncertainty(error)
        nddata = NDData(data, uncertainty=uncertainty)
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(nddata, aper)
        ref = aperture_photometry(data, aper, error=error)
        assert_allclose(phot.flux, ref['aperture_sum'])
        assert_allclose(phot.flux_err, ref['aperture_sum_err'])

    def test_nddata_ignored_keywords_warn(self, data):
        nddata = NDData(data)
        aper = CircularAperture((150, 25), 8)
        mask = np.zeros(data.shape, dtype=bool)
        match = 'is obtained from the input NDData object'
        with pytest.warns(AstropyUserWarning, match=match):
            AperturePhotometry(nddata, aper, mask=mask)

    def test_nddata_error_keyword_is_ignored(self, data):
        """
        Test that the ``error`` keyword is ignored, as warned, when the
        NDData object has no StdDevUncertainty, matching the handling of
        the ``mask`` and ``wcs`` keywords.
        """
        nddata = NDData(data)
        aper = CircularAperture((150, 25), 8)
        error = np.sqrt(np.abs(data))
        match = 'is obtained from the input NDData object'
        with pytest.warns(AstropyUserWarning, match=match):
            phot = AperturePhotometry(nddata, aper, error=error)
        assert np.isnan(phot.flux_err)

        with pytest.warns(AstropyUserWarning, match=match):
            stats = ApertureStats(nddata, aper, error=error)
        assert np.isnan(stats.sum_err)

    def test_nddata_units(self, data):
        nddata = NDData(data * u.Jy)
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(nddata, aper)
        assert phot.flux.unit == u.Jy

    def test_nddata_uncertainty_with_unit(self, data):
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

    def test_correct_matches_manual_mirror(self):
        """
        Test that mask_method='correct' reproduces plain photometry on
        a manually mirror-corrected image.

        The scene is constructed so that the mirror of every neighbor
        pixel is an unmasked non-neighbor pixel, so every neighbor pixel
        is corrected (none are excluded).
        """
        data, segm = make_scene()
        xycen = (21, 21)
        aper = CircularAperture([xycen], r=10)
        corr_phot = AperturePhotometry(data, aper, segmentation_image=segm,
                                       labels=[1], mask_method='correct')

        # Replace every neighbor pixel with its value mirrored across
        # the aperture center. Pixels outside the aperture have zero
        # weight, so correcting them globally does not change the flux.
        corrected = data.copy()
        yidx, xidx = np.nonzero((segm > 0) & (segm != 1))
        corrected[yidx, xidx] = data[2 * xycen[1] - yidx,
                                     2 * xycen[0] - xidx]
        ref_phot = AperturePhotometry(corrected, aper)
        assert_allclose(corr_phot.flux, ref_phot.flux, rtol=1e-12)

        # The correction changes the flux relative to no masking
        none_phot = AperturePhotometry(data, aper, mask_method='none')
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
    def test_default_columns(self, data):
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper).to_table()
        assert tbl.colnames == ['id', 'x_center', 'y_center', 'flux',
                                'flux_err', 'area', 'flags']

    def test_default_columns_with_wcs(self, data):
        wcs = make_wcs(data.shape)
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper, wcs=wcs).to_table()
        assert tbl.colnames == ['id', 'x_center', 'y_center', 'sky_center',
                                'flux', 'flux_err', 'area', 'flags']

    def test_columns_subset(self, data):
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper).to_table(columns=['id', 'flux'])
        assert tbl.colnames == ['id', 'flux']

    def test_columns_single_string(self, data):
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper).to_table(columns='flux')
        assert tbl.colnames == ['flux']

    def test_invalid_column(self, data):
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        match = 'Invalid column name'
        with pytest.raises(ValueError, match=match):
            phot.to_table(columns='invalid')
        with pytest.raises(ValueError, match=match):
            phot.to_table(columns=['id', 'subpixels'])

    def test_multi_aperture_suffixes(self, data):
        pos = ((150, 25), (90, 60))
        apers = [CircularAperture(pos, r) for r in (5, 8)]
        tbl = AperturePhotometry(data, apers).to_table()
        assert tbl.colnames == ['id', 'x_center', 'y_center', 'flux_0',
                                'flux_1', 'flux_err_0', 'flux_err_1',
                                'area_0', 'area_1', 'flags_0', 'flags_1']

    def test_meta(self, data):
        aper = CircularAperture((150, 25), 8)
        tbl = AperturePhotometry(data, aper).to_table()
        assert 'version' in tbl.meta
        assert 'aperture_photometry_args' in tbl.meta
        assert tbl.meta['aperture'] == 'CircularAperture'

    def test_sky_center_requires_wcs(self, data):
        """
        Test that requesting the sky_center column without a WCS raises
        a clear error.
        """
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        match = "the 'sky_center' column requires a WCS"
        with pytest.raises(ValueError, match=match):
            phot.to_table(columns=['id', 'sky_center'])


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
        assert phot.flags == APERTURE_FLAGS.PARTIAL_OVERLAP

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


class TestScalarBehavior:
    """
    A single scalar aperture position yields scalar output attributes,
    while array positions yield array outputs.
    """

    def test_single_scalar_position(self, data):
        error = 0.1 * np.ones_like(data)
        aper = CircularAperture((150, 25), r=8)
        phot = AperturePhotometry(data, aper, error=error)
        assert phot.isscalar is True
        assert phot.n_positions == 1
        for attr in ('id', 'x_center', 'y_center', 'flux', 'flux_err'):
            assert np.ndim(getattr(phot, attr)) == 0
        assert phot.area.isscalar
        assert np.ndim(phot.flags) == 0

    def test_array_position_single_element(self, data):
        # A length-1 list of positions is not scalar.
        aper = CircularAperture([(150, 25)], r=8)
        phot = AperturePhotometry(data, aper)
        assert phot.isscalar is False
        assert phot.flux.shape == (1,)
        assert phot.flux_err.shape == (1,)
        assert phot.area.shape == (1,)
        assert phot.flags.shape == (1,)

    def test_multiple_positions_not_scalar(self, data):
        aper = CircularAperture(((150, 25), (90, 60)), r=8)
        phot = AperturePhotometry(data, aper)
        assert phot.isscalar is False
        assert phot.flux.shape == (2,)

    def test_scalar_matches_array(self, data):
        """
        Test that the scalar output from a single position matches
        the first element of the array output from a length-1 list of
        positions.
        """
        scalar = AperturePhotometry(data, CircularAperture((150, 25), r=8))
        array = AperturePhotometry(data, CircularAperture([(150, 25)], r=8))
        assert_allclose(scalar.flux, array.flux[0])
        assert scalar.flags == array.flags[0]
        assert_allclose(scalar.x_center, array.x_center[0])

    def test_list_of_apertures_scalar_position(self, data):
        apers = [CircularAperture((150, 25), r=r) for r in (5, 8)]
        phot = AperturePhotometry(data, apers)
        assert phot.isscalar is True

        # id/x_center/y_center collapse to scalars
        assert np.ndim(phot.id) == 0
        assert np.ndim(phot.x_center) == 0
        assert np.ndim(phot.y_center) == 0

        # The per-aperture attributes keep only the trailing aperture axis
        assert phot.flux.shape == (2,)
        assert phot.flux_err.shape == (2,)
        assert phot.area.shape == (2,)
        assert phot.flags.shape == (2,)

    def test_list_of_apertures_array_position(self, data):
        pos = ((150, 25), (90, 60))
        apers = [CircularAperture(pos, r=r) for r in (5, 8)]
        phot = AperturePhotometry(data, apers)
        assert phot.isscalar is False
        assert phot.flux.shape == (2, 2)

    def test_sky_center_scalar(self, data):
        wcs = make_wcs(data.shape)
        phot = AperturePhotometry(data, CircularAperture((150, 25), r=8),
                                  wcs=wcs)
        assert isinstance(phot.sky_center, SkyCoord)
        assert phot.sky_center.isscalar is True

    def test_sky_center_array(self, data):
        wcs = make_wcs(data.shape)
        phot = AperturePhotometry(data, CircularAperture([(150, 25)], r=8),
                                  wcs=wcs)
        assert phot.sky_center.isscalar is False
        assert len(phot.sky_center) == 1

    def test_to_table_scalar_single_aperture(self, data):
        aper = CircularAperture((150, 25), r=8)
        tbl = AperturePhotometry(data, aper).to_table()
        assert len(tbl) == 1
        assert tbl.colnames == ['id', 'x_center', 'y_center', 'flux',
                                'flux_err', 'area', 'flags']

    def test_to_table_scalar_list_of_apertures(self, data):
        apers = [CircularAperture((150, 25), r=r) for r in (5, 8)]
        tbl = AperturePhotometry(data, apers).to_table()
        assert len(tbl) == 1
        assert tbl.colnames == ['id', 'x_center', 'y_center', 'flux_0',
                                'flux_1', 'flux_err_0', 'flux_err_1',
                                'area_0', 'area_1', 'flags_0', 'flags_1']

    def test_decode_flags_scalar(self):
        data = np.ones((25, 25))
        mask = np.zeros(data.shape, dtype=bool)
        mask[12, 12] = True
        aper = CircularAperture((12, 12), r=3)
        phot = AperturePhotometry(data, aper, mask=mask)
        assert phot.decode_flags() == [['masked_pixels']]

    @pytest.mark.parametrize('use_segm_obj', [True, False])
    def test_scalar_input_attributes_not_collapsed(self, use_segm_obj):
        """
        Test that the ``segmentation_image`` and ``labels`` inputs are
        echoed back unchanged for a scalar instance, i.e., that they are
        not treated as per-position output arrays.
        """
        data, segm = make_scene()
        segm_in = SegmentationImage(segm) if use_segm_obj else segm
        aper = CircularAperture((21, 21), r=8)
        phot = AperturePhotometry(data, aper, segmentation_image=segm_in,
                                  labels=np.array([1]), mask_method='mask')
        assert phot.isscalar is True
        assert phot.segmentation_image is segm_in
        assert_equal(phot.labels, np.array([1]))

        # The photometry itself is unaffected
        ref = AperturePhotometry(data, aper,
                                 mask=(segm > 0) & (segm != 1))
        assert_allclose(phot.flux, ref.flux)

    def test_isscalar_matches_aperture_stats(self, data):
        scalar_aper = CircularAperture((150, 25), r=8)
        array_aper = CircularAperture([(150, 25)], r=8)
        assert (AperturePhotometry(data, scalar_aper).isscalar
                == ApertureStats(data, scalar_aper).isscalar)
        assert (AperturePhotometry(data, array_aper).isscalar
                == ApertureStats(data, array_aper).isscalar)


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
        assert np.isfinite(phot.flux)
        assert_allclose(phot.flux, aper.area_overlap(data) - 2)
        assert_allclose(phot.area.value, phot.flux)
        assert phot.flags == 32
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
            flux = phot.flux
            flags = phot.flags
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
        assert_allclose(phot.flux, stats.sum)
        assert_allclose(phot.area.value, stats.sum_aper_area.value)
        assert phot.flags == stats.flags

    def test_nonfinite_outside_aperture_not_flagged(self):
        data = np.ones((25, 25))
        data[0, 0] = np.nan  # far outside the aperture
        aper = CircularAperture((12, 12), r=5)
        phot = AperturePhotometry(data, aper)
        assert phot.flags == 0
        assert_allclose(phot.flux, aper.area_overlap(data))

    def test_combined_mask_and_nonfinite(self):
        data = np.ones((25, 25))
        data[12, 12] = np.nan
        mask = np.zeros(data.shape, dtype=bool)
        mask[11, 12] = True
        aper = CircularAperture((12, 12), r=5)
        phot = AperturePhotometry(data, aper, mask=mask)
        assert np.isfinite(phot.flux)
        # Both masked_pixels (8) and non_finite_data (32) are set.
        assert phot.flags == 8 | 32
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
        assert phot.flags == 8
        stats = ApertureStats(data, aper, mask=mask)
        assert phot.flags == stats.flags

    def test_legacy_function_still_corrupts_nonfinite(self):
        """
        Test that the legacy aperture_photometry function still returns
        NaN for the aperture sum when there are non-finite values in the
        aperture, even though the new AperturePhotometry class masks
        them and returns a finite sum.

        This is to ensure backward compatibility with existing code that
        relies on the old behavior.
        """
        data = np.ones((25, 25))
        data[12, 12] = np.nan

        aper = CircularAperture((12, 12), r=5)
        tbl = aperture_photometry(data, aper)
        assert np.isnan(tbl['aperture_sum'][0])

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
        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper, error=np.ones((11, 11)))

        # The converse: unitless data with a unit error
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(np.ones((11, 11)), aper,
                               error=np.ones((11, 11)) * u.Jy)

        # Same-dimension but different units are also rejected
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper,
                               error=np.ones((11, 11)) * u.mJy)

    @pytest.mark.parametrize('method', ['exact ', 'Exact', 'invalid'])
    def test_invalid_method_at_init(self, method):
        """
        Test that an invalid method is reported at construction rather
        than at the first access of a measured attribute.
        """
        data = np.ones((11, 11))
        aper = CircularAperture((5, 5), r=3)
        with pytest.raises(ValueError, match=f'Invalid method: {method!r}'):
            AperturePhotometry(data, aper, method=method)
        with pytest.raises(ValueError,
                           match=f'Invalid sum_method: {method!r}'):
            ApertureStats(data, aper, sum_method=method)

    @pytest.mark.parametrize('subpixels', [0, -1, 2.5, True])
    def test_invalid_subpixels_at_init(self, subpixels):
        """
        Test that an invalid subpixels value is reported at
        construction.
        """
        data = np.ones((11, 11))
        aper = CircularAperture((5, 5), r=3)
        match = 'subpixels must be a strictly positive integer'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper, method='subpixel',
                               subpixels=subpixels)
        with pytest.raises(ValueError, match=match):
            ApertureStats(data, aper, sum_method='subpixel',
                          subpixels=subpixels)

    def test_invalid_mask_method_at_init(self):
        data = np.ones((11, 11))
        aper = CircularAperture((5, 5), r=3)
        match = 'mask_method must be one of'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper, mask_method='invalid')
        with pytest.raises(ValueError, match=match):
            ApertureStats(data, aper, mask_method='invalid')

    @pytest.mark.parametrize('apertures', [[], (), np.array([])])
    def test_empty_aperture_list(self, apertures):
        """
        Test that an empty aperture list is reported at construction.
        """
        data = np.ones((11, 11))
        match = 'apertures must not be empty'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, apertures)

    def test_labels_not_1d(self):
        """
        Test that a 2D labels array is reported at construction rather
        than failing later in the batch driver.
        """
        data = np.ones((11, 11))
        segm = np.zeros(data.shape, dtype=int)
        segm[4:7, 4:7] = 1
        aper = CircularAperture((5, 5), r=3)
        match = 'labels must be a 1D array'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper, segmentation_image=segm,
                               labels=np.array([[1, 2]]),
                               mask_method='mask')
        with pytest.raises(ValueError, match=match):
            ApertureStats(data, aper, segmentation_image=segm,
                          labels=np.array([[1, 2]]), mask_method='mask')


class TestReadOnlyInputs:
    """
    End-to-end tests that both classes accept read-only (non-writeable)
    input arrays on both the batch and mask-based code paths, and that
    the inputs are never modified.
    """

    @staticmethod
    def _make_readonly_inputs():
        """
        Build a full set of read-only input arrays, including a masked
        pixel and a non-finite data value.
        """
        data, segm = make_scene()
        rng = np.random.default_rng(7)
        data = data + rng.normal(0.0, 0.1, data.shape)
        data[30, 5] = np.nan
        error = np.full(data.shape, 0.1)
        mask = np.zeros(data.shape, dtype=bool)
        mask[20, 20] = True
        arrays = {'data': data, 'error': error, 'mask': mask,
                  'segmentation_image': segm.astype(np.intp),
                  'labels': np.array([1]),
                  'local_bkg': np.array([1.0]),
                  'positions': np.array([(21.0, 21.0)])}
        for arr in arrays.values():
            arr.setflags(write=False)
        return arrays

    @pytest.mark.parametrize('aper_cls', [CircularAperture,
                                          NoBatchCircularAperture])
    @pytest.mark.parametrize('mask_method', ['none', 'mask', 'source_only',
                                             'correct'])
    def test_aperture_photometry(self, aper_cls, mask_method):
        arrays = self._make_readonly_inputs()
        originals = {key: arr.copy() for key, arr in arrays.items()}

        kwargs = {}
        if mask_method != 'none':
            kwargs = {'segmentation_image': arrays['segmentation_image'],
                      'labels': arrays['labels'],
                      'mask_method': mask_method}
        aper = aper_cls(arrays['positions'], r=10)
        phot = AperturePhotometry(arrays['data'], aper,
                                  error=arrays['error'],
                                  mask=arrays['mask'], **kwargs)
        assert np.isfinite(phot.flux[0])
        for attr in ('flux_err', 'area', 'flags'):
            _ = getattr(phot, attr)
        _ = phot.to_table()

        for key, arr in arrays.items():
            assert_equal(arr, originals[key])

    @pytest.mark.parametrize('aper_cls', [CircularAperture,
                                          NoBatchCircularAperture])
    @pytest.mark.parametrize('with_sigma_clip', [False, True])
    def test_aperture_stats(self, aper_cls, with_sigma_clip):
        arrays = self._make_readonly_inputs()
        originals = {key: arr.copy() for key, arr in arrays.items()}

        sigma_clip = (SigmaClip(sigma=3.0, maxiters=5) if with_sigma_clip
                      else None)
        aper = aper_cls(arrays['positions'], r=10)
        stats = ApertureStats(arrays['data'], aper, error=arrays['error'],
                              mask=arrays['mask'], sigma_clip=sigma_clip,
                              local_bkg=arrays['local_bkg'],
                              segmentation_image=(
                                  arrays['segmentation_image']),
                              labels=arrays['labels'],
                              mask_method='correct')
        for attr in ('sum', 'sum_err', 'mean', 'median', 'std', 'mad_std',
                     'biweight_location', 'gini', 'centroid',
                     'semimajor_axis', 'fwhm', 'flags'):
            _ = getattr(stats, attr)
        assert np.isfinite(stats.mean)
        _ = stats.to_table()

        for key, arr in arrays.items():
            assert_equal(arr, originals[key])


class TestEmptyPositions:
    """
    Regression tests for apertures with zero positions, which must
    flow through both classes (including the batch Cython drivers) and
    the legacy function, returning empty outputs.
    """

    def test_aperture_photometry_class(self):
        data = np.ones((11, 11))
        aper = CircularAperture(np.empty((0, 2)), r=3)
        phot = AperturePhotometry(data, aper, error=np.ones_like(data))
        assert phot.n_positions == 0
        assert phot.flux.shape == (0,)
        assert phot.flux_err.shape == (0,)
        assert phot.area.shape == (0,)
        assert phot.flags.shape == (0,)
        assert len(phot.to_table()) == 0

    def test_aperture_stats(self):
        data = np.ones((11, 11))
        aper = CircularAperture(np.empty((0, 2)), r=3)
        stats = ApertureStats(data, aper)
        assert stats.n_positions == 0
        assert stats.mean.shape == (0,)
        assert stats.median.shape == (0,)
        assert stats.sum.shape == (0,)
        assert stats.flags.shape == (0,)
        assert len(stats.to_table()) == 0

    def test_legacy_function(self):
        data = np.ones((11, 11))
        aper = CircularAperture(np.empty((0, 2)), r=3)
        tbl = aperture_photometry(data, aper)
        assert len(tbl) == 0


class TestSegmentationAttributes:
    """
    Both classes echo the segmentation-masking inputs back as public
    attributes.
    """

    @pytest.mark.parametrize('cls', [AperturePhotometry, ApertureStats])
    def test_defaults(self, cls):
        data = np.ones((11, 11))
        aper = CircularAperture((5, 5), r=3)
        obj = cls(data, aper)
        assert obj.segmentation_image is None
        assert obj.labels is None
        assert obj.mask_method == 'none'

    @pytest.mark.parametrize('cls', [AperturePhotometry, ApertureStats])
    def test_inputs_echoed(self, cls):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        obj = cls(data, aper, segmentation_image=segm, labels=[1],
                  mask_method='mask')
        assert obj.segmentation_image is segm
        assert_equal(obj.labels, [1])
        assert obj.mask_method == 'mask'

    def test_stats_slicing_slices_labels(self):
        """
        Test that slicing an ApertureStats also slices the per-aperture
        ``labels``, so the sliced object reports the labels of the
        apertures it contains.
        """
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (21, 21)], r=8)
        stats = ApertureStats(data, aper, segmentation_image=segm,
                              labels=[1, 2], mask_method='mask')
        assert_equal(stats.labels, [1, 2])
        assert_equal(stats[1:].labels, [2])
        assert stats[0].labels == 1
        assert stats[0].mask_method == 'mask'
        assert stats[0].segmentation_image is segm

    def test_photometry_meta_records_mask_method(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        phot = AperturePhotometry(data, aper, segmentation_image=segm,
                                  labels=[1], mask_method='mask')
        args = phot.to_table().meta['aperture_photometry_args']
        assert "method='exact'" in args
        assert 'subpixels=5' in args
        assert "mask_method='mask'" in args


class TestReprAndImmutability:
    def test_repr(self, data):
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        assert 'AperturePhotometry' in repr(phot)
        assert "method='exact'" in repr(phot)

    def test_str(self, data):
        aper = CircularAperture((150, 25), 8)
        phot = AperturePhotometry(data, aper)
        assert 'AperturePhotometry' in str(phot)

    def test_no_new_attributes_after_init(self, data):
        """
        Only cached-property cache entries may appear after ``__init__``,
        which is required for the instance to be thread-safe.
        """
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
                      'flux_err', 'area', 'flags', 'isscalar'}
        assert new_keys.issubset(lazy_names)

    def test_concurrent_access(self, data):
        """
        Test that a single shared AperturePhotometry instance can be
        read concurrently.

        The cached-property caches fill under contention and every
        thread sees identical values.
        """
        aper = CircularAperture(((150, 25), (90, 60)), 8)
        phot = AperturePhotometry(data, aper, error=np.ones_like(data))

        def read(_):
            return {attr: np.asarray(getattr(phot, attr))
                    for attr in ('flux', 'flux_err', 'flags', 'id')}

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(read, range(16)))
        for result in results:
            for attr, values in result.items():
                assert_allclose(values, results[0][attr])


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


class TestNThreads:
    """
    Tests for the n_threads keyword.
    """

    @staticmethod
    def make_inputs():
        """
        Build a deterministic image (with non-finite values), error,
        mask, and positions (including off-edge positions).
        """
        rng = np.random.default_rng(0)
        data = rng.normal(100.0, 5.0, (120, 130))
        data[3, 3] = np.nan
        data[50, 50] = np.inf
        error = np.abs(rng.normal(5.0, 0.5, data.shape)) + 0.1
        mask = np.zeros(data.shape, dtype=bool)
        mask[60:65, 85:90] = True
        positions = np.column_stack(
            [rng.uniform(-5, data.shape[1] + 5, 57),
             rng.uniform(-5, data.shape[0] + 5, 57)])
        return data, error, mask, positions

    @pytest.mark.parametrize('n_threads', [2, 8])
    def test_identical_results(self, n_threads):
        """
        Test that multithreaded photometry gives results identical to
        the single-threaded computation, including for off-edge
        positions, masked pixels, and non-finite data values.
        """
        data, error, mask, positions = self.make_inputs()
        aper = CircularAperture(positions, r=7.0)
        phot1 = AperturePhotometry(data, aper, error=error, mask=mask)
        phot2 = AperturePhotometry(data, aper, error=error, mask=mask,
                                   n_threads=n_threads)
        assert phot2.n_threads == n_threads
        assert_equal(phot1.flux, phot2.flux)
        assert_equal(phot1.flux_err, phot2.flux_err)
        assert_equal(phot1.area, phot2.area)
        assert_equal(phot1.flags, phot2.flags)

    def test_more_threads_than_positions(self):
        """
        Test that n_threads larger than the number of positions gives
        identical results.
        """
        data = np.ones((40, 40))
        aper = CircularAperture([(20, 20), (10, 10), (30, 25)], r=5.0)
        phot1 = AperturePhotometry(data, aper)
        phot2 = AperturePhotometry(data, aper, n_threads=8)
        assert_equal(phot1.flux, phot2.flux)
        assert_equal(phot1.flags, phot2.flags)

    def test_scalar_position(self):
        """
        Test that a scalar aperture position with n_threads > 1 falls
        back to a single-chunk (serial) computation.
        """
        data = np.ones((40, 40))
        aper = CircularAperture((20, 20), r=5.0)
        phot1 = AperturePhotometry(data, aper)
        phot2 = AperturePhotometry(data, aper, n_threads=4)
        assert_equal(phot1.flux, phot2.flux)

    def test_aperture_list(self):
        """
        Test multithreading with a list of input apertures.
        """
        data, error, mask, positions = self.make_inputs()
        apers = [CircularAperture(positions, r=r) for r in (3.0, 7.0)]
        phot1 = AperturePhotometry(data, apers, error=error, mask=mask)
        phot2 = AperturePhotometry(data, apers, error=error, mask=mask,
                                   n_threads=4)
        assert_equal(phot1.flux, phot2.flux)
        assert_equal(phot1.flux_err, phot2.flux_err)
        assert_equal(phot1.area, phot2.area)
        assert_equal(phot1.flags, phot2.flags)

    def test_segmentation_masking(self):
        """
        Test that per-source segmentation labels are chunked together
        with the positions.
        """
        data, segm = make_scene()
        positions = [(21.0, 21.0), (28.0, 22.0), (21.0, 21.5),
                     (28.5, 22.0), (20.5, 21.0)]
        labels = [1, 2, 1, 2, 1]
        aper = CircularAperture(positions, r=8.0)
        phot1 = AperturePhotometry(data, aper, segmentation_image=segm,
                                   labels=labels, mask_method='mask')
        phot2 = AperturePhotometry(data, aper, segmentation_image=segm,
                                   labels=labels, mask_method='mask',
                                   n_threads=3)
        assert_equal(phot1.flux, phot2.flux)
        assert_equal(phot1.area, phot2.area)
        assert_equal(phot1.flags, phot2.flags)

    def test_mask_based_fallback(self):
        """
        Test that apertures that do not support the batch code path
        give correct results with n_threads > 1 (the mask-based code
        path stays serial).
        """
        data, _, _, positions = self.make_inputs()
        aper = NoBatchCircularAperture(positions, r=7.0)
        ref = AperturePhotometry(data, CircularAperture(positions, r=7.0))
        phot = AperturePhotometry(data, aper, n_threads=4)
        assert_allclose(phot.flux, ref.flux, equal_nan=True)

    def test_empty_positions(self):
        """
        Test that an aperture with zero positions works with n_threads >
        1 (zero chunks must fall back to the serial path).
        """
        data = np.ones((11, 11))
        aper = CircularAperture(np.empty((0, 2)), r=3.0)
        phot = AperturePhotometry(data, aper, n_threads=4)
        assert phot.n_positions == 0
        assert phot.flux.shape == (0,)

    def test_invalid_n_threads(self):
        """
        Test that an error is raised if n_threads is not a positive
        integer.
        """
        data = np.ones((40, 40))
        aper = CircularAperture((20, 20), r=5.0)
        match = 'n_threads must be a positive integer'
        for n_threads in (0, -1, 2.5):
            with pytest.raises(ValueError, match=match):
                AperturePhotometry(data, aper, n_threads=n_threads)

    def test_repr_and_meta(self):
        """
        Test that n_threads appears in the repr and in the table
        metadata calling arguments.
        """
        data = np.ones((40, 40))
        aper = CircularAperture((20, 20), r=5.0)
        phot = AperturePhotometry(data, aper, n_threads=4)
        assert 'n_threads=4' in repr(phot)
        assert 'n_threads=4' in phot.to_table().meta[
            'aperture_photometry_args']
