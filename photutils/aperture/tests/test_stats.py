# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the stats module.
"""

import sys
from functools import cached_property
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import NDData, StdDevUncertainty
from astropy.stats import SigmaClip, biweight_location, biweight_scale, mad_std
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture.bounding_box import BoundingBox
from photutils.aperture.circle import CircularAnnulus, CircularAperture
from photutils.aperture.core import _enable_batch_photometry
from photutils.aperture.ellipse import (EllipticalAnnulus, EllipticalAperture,
                                        SkyEllipticalAnnulus)
from photutils.aperture.flags import APERTURE_FLAGS
from photutils.aperture.photometry import AperturePhotometry
from photutils.aperture.rectangle import (RectangularAnnulus,
                                          RectangularAperture)
from photutils.aperture.stats import _MAD_STD_SCALE, ApertureStats
from photutils.aperture.tests.conftest import NoBatchCircularAperture
from photutils.datasets import make_100gaussians_image, make_wcs
from photutils.utils._optional_deps import HAS_REGIONS


@_enable_batch_photometry
class _NoneSpecCircular(CircularAperture):
    """
    A `CircularAperture` subclass that opts in to the fast batch driver
    but whose ``_batch_shape_params`` hook returns `None`, so the batch
    driver is not used and the mask-based code path is exercised.
    """

    def _batch_shape_params(self):
        return None


@pytest.fixture(scope='class')
def stats_data(request):
    """
    Build the shared data and ApertureStats objects on the test class.
    """
    cls = request.cls
    cls.data = make_100gaussians_image()
    cls.error = np.sqrt(np.abs(cls.data))
    cls.wcs = make_wcs(cls.data.shape)
    cls.positions = ((145.1, 168.3), (84.7, 224.1), (48.3, 200.3))
    cls.aperture = CircularAperture(cls.positions, r=5)

    cls.sigclip = SigmaClip(sigma=3.0, maxiters=10)
    cls.apstats1 = ApertureStats(cls.data, cls.aperture,
                                 error=cls.error, wcs=cls.wcs,
                                 sigma_clip=None)
    cls.apstats2 = ApertureStats(cls.data, cls.aperture,
                                 error=cls.error, wcs=cls.wcs,
                                 sigma_clip=cls.sigclip)

    cls.unit = u.Jy
    cls.apstats1_units = ApertureStats(cls.data * cls.unit,
                                       cls.aperture,
                                       error=cls.error * cls.unit,
                                       wcs=cls.wcs, sigma_clip=None)
    cls.apstats2_units = ApertureStats(cls.data * cls.unit,
                                       cls.aperture,
                                       error=cls.error * cls.unit,
                                       wcs=cls.wcs,
                                       sigma_clip=cls.sigclip)


class _ExpandedBboxCircular(CircularAperture):
    """
    A `CircularAperture` subclass whose ``bbox`` is expanded by one
    pixel on each side, overriding the default implementation.

    Used to test that the ``ApertureStats`` bounding-box bounds honor a
    ``bbox`` override instead of using the vectorized fast path.
    """

    @cached_property
    def bbox(self):
        boxes = [BoundingBox(ixmin=box.ixmin - 1, ixmax=box.ixmax + 1,
                             iymin=box.iymin - 1, iymax=box.iymax + 1)
                 for box in self._bbox]
        if self.isscalar:
            return boxes[0]
        return boxes


@pytest.mark.usefixtures('stats_data')
class BaseApertureStatsData:
    """
    Shared data, aperture, and ApertureStats objects used by the
    ApertureStats test classes below.

    The ``stats_data`` fixture builds the objects once per test class
    rather than at import time, so collection stays cheap and each test
    class gets its own instances (no cached-property state is shared
    across classes).
    """


class TestProperties(BaseApertureStatsData):
    """
    Tests for the computed source properties.
    """

    @pytest.mark.parametrize('with_units', [True, False])
    @pytest.mark.parametrize('with_sigmaclip', [True, False])
    def test_properties(self, with_units, with_sigmaclip):
        apstats = [self.apstats1.copy(), self.apstats2.copy(),
                   self.apstats1_units.copy(), self.apstats2_units.copy()]
        index = [1, 3] if with_sigmaclip else [0, 2]
        index = index[1] if with_units else index[0]
        apstats1 = apstats[index]
        apstats2 = apstats1.copy()

        idx = 1

        scalar_props = ('isscalar', 'n_positions')

        # Evaluate (cache) properties before slice
        for prop in apstats1.properties:
            _ = getattr(apstats1, prop)
        apstats3 = apstats1[idx]
        for prop in apstats1.properties:
            if prop in scalar_props:
                continue
            assert_equal(getattr(apstats1, prop)[idx], getattr(apstats3, prop))

        # Slice catalog before evaluating catalog properties
        apstats4 = apstats2[idx]
        for prop in apstats1.properties:
            if prop in scalar_props:
                continue
            assert_equal(getattr(apstats4, prop), getattr(apstats1, prop)[idx])

    def test_skyaperture(self):
        pix_apstats = ApertureStats(self.data, self.aperture, wcs=self.wcs)
        skyaper = self.aperture.to_sky(self.wcs)
        sky_apstats = ApertureStats(self.data, skyaper, wcs=self.wcs)

        exclude_props = ('bbox', 'error_sum_cutout', 'sum_err',
                         'sky_centroid', 'sky_centroid_icrs')
        for prop in pix_apstats.properties:
            if prop in exclude_props:
                continue
            assert_allclose(getattr(pix_apstats, prop),
                            getattr(sky_apstats, prop), atol=1e-7)

        match = 'A wcs is required when using a SkyAperture'
        with pytest.raises(ValueError, match=match):
            _ = ApertureStats(self.data, skyaper)

    def test_minimal_inputs(self):
        apstats = ApertureStats(self.data, self.aperture)
        props = ('sky_centroid', 'sky_centroid_icrs', 'error_sum_cutout')
        for prop in props:
            assert set(getattr(apstats, prop)) == {None}
        assert np.all(np.isnan(apstats.sum_err))
        assert set(apstats._variance_cutout) == {None}

        apstats = ApertureStats(self.data, self.aperture, sum_method='center')
        assert set(apstats._variance_cutout_center) == {None}

    def test_tiny_source(self):
        data = np.zeros((21, 21))
        data[5, 5] = 1.0
        aperture = CircularAperture(((5, 5), (15, 15)), r=1)
        apstats = ApertureStats(data, aperture)
        assert_allclose(apstats.sum, (1.0, 0.0))
        assert_allclose(apstats[0].covariance,
                        [(1 / 12, 0), (0, 1 / 12)] * u.pix**2)
        assert_allclose(apstats[1].covariance,
                        [(np.nan, np.nan), (np.nan, np.nan)] * u.pix**2)
        assert_allclose(apstats.fwhm, [0.67977799, np.nan] * u.pix)

    def test_data_dtype(self):
        """
        Regression test that input ``data`` with int dtype does not
        raise UFuncTypeError due to subtraction of float array from int
        array.
        """
        data = np.ones((25, 25), dtype=np.uint16)
        aper = CircularAperture((12, 12), 5)
        apstats = ApertureStats(data, aper)
        assert apstats.min == 1.0
        assert apstats.max == 1.0
        assert apstats.mean == 1.0
        assert apstats.x_centroid == 12.0
        assert apstats.y_centroid == 12.0

    def test_cached_properties_class_cache(self):
        """
        Test that _cached_properties is cached on the class and shared
        across instances.
        """
        apstats2 = ApertureStats(self.data, self.aperture)
        result1 = self.apstats1._cached_properties
        result2 = apstats2._cached_properties
        assert result1 is result2

    def test_repr_str(self):
        assert repr(self.apstats1) == str(self.apstats1)
        assert 'Length: 3' in repr(self.apstats1)


class TestSumMethod(BaseApertureStatsData):
    """
    Tests for the sum_method aperture-mask footprint.
    """

    @pytest.mark.parametrize('sum_method', ['exact', 'subpixel'])
    def test_sum_method(self, sum_method):
        apstats1 = ApertureStats(self.data, self.aperture, error=self.error,
                                 sum_method='center')
        apstats2 = ApertureStats(self.data, self.aperture, error=self.error,
                                 sum_method=sum_method, subpixels=4)

        scalar_props = ('isscalar', 'n_positions')

        # Evaluate (cache) properties before slice
        for prop in apstats1.properties:
            if prop in scalar_props:
                continue
            if 'sum' in prop:
                # The sum-footprint properties must differ between the
                # two sum methods
                equal = True
                try:
                    assert_equal(getattr(apstats1, prop),
                                 getattr(apstats2, prop))
                except AssertionError:
                    equal = False
                assert not equal, f'{prop} is unexpectedly equal'
            else:
                assert_equal(getattr(apstats1, prop), getattr(apstats2, prop))

    def test_sum_method_photometry(self):
        for method in ('center', 'exact', 'subpixel'):
            subpixels = 4
            apstats = ApertureStats(self.data, self.aperture,
                                    error=self.error, sum_method=method,
                                    subpixels=subpixels)
            phot = AperturePhotometry(self.data, self.aperture,
                                      error=self.error, method=method,
                                      subpixels=subpixels)
            assert_allclose(apstats.sum, phot.flux)
            assert_allclose(apstats.sum_err, phot.flux_err)

    def test_sum_aper_area_center_masked(self):
        """
        Regression test that ``sum_aper_area`` is consistent with
        ``sum`` (and identical between the fast and mask-based code
        paths) when all center-method pixels are masked but the
        ``sum_method`` aperture still contains unmasked fractional
        boundary pixels.

        Previously, the mask-based path returned NaN for the area (via
        the center-method all-masked flag) while returning a finite
        ``sum`` from the same surviving fractional pixels.
        """
        data = np.ones((11, 11))
        aperture = CircularAperture((5.0, 5.0), r=1.4)
        # Mask exactly the center-method pixels. The fractional
        # (exact-method) boundary pixels remain unmasked.
        mask = aperture.to_mask(method='center').to_image(data.shape) > 0

        fast = ApertureStats(data, aperture, mask=mask)
        with patch.object(ApertureStats, '_batch_inputs',
                          property(lambda _: None)):
            slow = ApertureStats(data, aperture, mask=mask)
            slow_sum = slow.sum
            slow_area = slow.sum_aper_area

        assert fast._fast_sum is not None
        assert fast.sum > 0
        # data is all ones, so the sum equals the surviving area
        assert_allclose(fast.sum_aper_area.value, fast.sum)
        assert_allclose(slow_sum, fast.sum)
        assert_allclose(slow_area, fast.sum_aper_area)
        # The center-method statistics remain all-masked (NaN)
        assert np.isnan(fast.center_aper_area.value)
        assert np.isnan(fast.mean)


class TestMasking(BaseApertureStatsData):
    """
    Tests for the mask keyword, local background, and apertures
    that do not overlap the data.
    """

    def test_mask(self):
        mask = np.zeros(self.data.shape, dtype=bool)
        mask[225:240, 80:90] = True  # partially mask id=2
        mask[190:210, 40:60] = True  # completely mask id=3

        apstats = ApertureStats(self.data, self.aperture, mask=mask,
                                error=self.error)

        # id=2 is partially masked
        assert apstats[1].sum < self.apstats1[1].sum
        assert apstats[1].sum_err < self.apstats1[1].sum_err

        exclude = ('isscalar', 'n_positions', 'sky_centroid',
                   'sky_centroid_icrs', 'flags')
        apstats1 = apstats[2]
        for prop in apstats1.properties:
            if (prop in exclude or 'bbox' in prop or 'cutout' in prop
                    or 'moments' in prop):
                continue
            assert np.all(np.isnan(getattr(apstats1, prop)))

        # id=2 has masked pixels; id=3 is completely masked
        assert apstats[1].flags == APERTURE_FLAGS.MASKED_PIXELS
        assert apstats[2].flags == (APERTURE_FLAGS.MASKED_PIXELS
                                    | APERTURE_FLAGS.ALL_MASKED)

        # Test that mask=None is the same as mask=np.ma.nomask
        apstats1 = ApertureStats(self.data, self.aperture, mask=None)
        apstats2 = ApertureStats(self.data, self.aperture, mask=np.ma.nomask)
        assert_equal(apstats1.centroid, apstats2.centroid)

    def test_local_bkg(self):
        data = np.ones(self.data.shape) * 100.0
        local_bkg = (10, 20, 30)
        apstats = ApertureStats(data, self.aperture, local_bkg=local_bkg)

        for i, locbkg in enumerate(local_bkg):
            apstats0 = ApertureStats(data - locbkg, self.aperture[i],
                                     local_bkg=None)
            for prop in apstats.properties:
                assert_equal(getattr(apstats[i], prop),
                             getattr(apstats0, prop))

        # Test broadcasting
        local_bkg = (12, 12, 12)
        apstats1 = ApertureStats(data, self.aperture, local_bkg=local_bkg)
        apstats2 = ApertureStats(data, self.aperture, local_bkg=local_bkg[0])
        assert_equal(apstats1.sum, apstats2.sum)

        match = 'local_bkg must be scalar or have the same length as the'
        with pytest.raises(ValueError, match=match):
            _ = ApertureStats(data, self.aperture, local_bkg=(10, 20))

        match = 'local_bkg must not contain any non-finite'
        with pytest.raises(ValueError, match=match):
            _ = ApertureStats(data, self.aperture[0:2], local_bkg=(10, np.nan))
        with pytest.raises(ValueError, match=match):
            _ = ApertureStats(data, self.aperture[0:2],
                              local_bkg=(-np.inf, 10))

        match = 'local_bkg must be a 1D array'
        with pytest.raises(ValueError, match=match):
            _ = ApertureStats(data, self.aperture[0:2],
                              local_bkg=np.ones((3, 3)))

    def test_local_bkg_units(self):
        """
        Test that local_bkg must carry the same units as a Quantity
        ``data`` input and that the results match the unitless case.
        """
        unit = u.Jy
        data = np.ones(self.data.shape) * 100.0
        local_bkg = (10, 20, 30)
        apstats1 = ApertureStats(data << unit, self.aperture,
                                 local_bkg=local_bkg * unit)
        apstats2 = ApertureStats(data, self.aperture, local_bkg=local_bkg)
        assert apstats1.sum.unit == unit
        assert_allclose(apstats1.sum.value, apstats2.sum)

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            ApertureStats(data << unit, self.aperture, local_bkg=local_bkg)
        with pytest.raises(ValueError, match=match):
            ApertureStats(data << unit, self.aperture,
                          local_bkg=local_bkg * u.mJy)
        with pytest.raises(ValueError, match=match):
            ApertureStats(data, self.aperture, local_bkg=local_bkg * unit)

    def test_no_aperture_overlap(self):
        aperture = CircularAperture(((0, 0), (100, 100), (-100, -100)), r=5)
        apstats = ApertureStats(self.data, aperture)
        assert_equal(apstats._overlap, [True, True, False])

        exclude = ('isscalar', 'n_positions', 'sky_centroid',
                   'sky_centroid_icrs', 'flags')
        apstats1 = apstats[2]
        for prop in apstats1.properties:
            if (prop in exclude or 'bbox' in prop or 'cutout' in prop
                    or 'moments' in prop):
                continue
            assert np.all(np.isnan(getattr(apstats1, prop)))

        assert apstats[0].flags == APERTURE_FLAGS.PARTIAL_OVERLAP
        assert apstats[2].flags == (APERTURE_FLAGS.NO_OVERLAP
                                    | APERTURE_FLAGS.NO_PIXELS)


class TestTable(BaseApertureStatsData):
    """
    Tests for the to_table output and the properties list.
    """

    def test_to_table(self):
        tbl = self.apstats1.to_table()
        assert tbl.colnames == self.apstats1.default_columns
        assert len(tbl) == len(self.apstats1) == 3

        columns = ['id', 'min', 'max', 'mean', 'median', 'std', 'sum']
        tbl = self.apstats1.to_table(columns=columns)
        assert tbl.colnames == columns
        assert len(tbl) == len(self.apstats1) == 3

        tbl = self.apstats1.to_table(columns='sum')
        assert tbl.colnames == ['sum']
        assert len(tbl) == len(self.apstats1) == 3

    @pytest.mark.parametrize('columns', ['invalid',
                                         'sum_method',
                                         'isscalar',
                                         'n_positions',
                                         ['id', 'subpixels']])
    def test_invalid_column(self, columns):
        match = 'Invalid column name'
        with pytest.raises(ValueError, match=match):
            self.apstats1.to_table(columns=columns)

    def test_properties_excludes_scalar_attrs(self):
        """
        Regression test to ensure that ``isscalar`` and ``n_positions``
        (scalar values for the whole object, not per-source values) are
        not included in ``properties`` and so are not valid ``to_table``
        columns.
        """
        assert 'isscalar' not in self.apstats1.properties
        assert 'n_positions' not in self.apstats1.properties

    def test_deprecated_column(self):
        """
        Regression test to ensure that deprecated column names are still
        accepted by ``to_table``, not rejected by the new column-name
        validation.
        """
        match = "'xcentroid' attribute was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            tbl = self.apstats1.to_table(columns='xcentroid')
        assert tbl.colnames == ['x_centroid']


class TestIndexing(BaseApertureStatsData):
    """
    Tests for indexing, slicing, and scalar ApertureStats objects.
    """

    def test_slicing(self):
        apstats = self.apstats1
        _ = apstats.to_table()
        apstat0 = apstats[1]
        assert apstat0.n_positions == 1
        assert apstat0.id == np.array([2])
        apstat1 = apstats.select_id(2)
        assert apstat1.n_positions == 1
        assert apstat0.sum_aper_area == apstat1.sum_aper_area

        apstat0 = apstats[0:1]
        assert len(apstat0) == 1

        apstat0 = apstats[0:2]
        assert len(apstat0) == 2

        apstat0 = apstats[0:3]
        assert len(apstat0) == 3

        apstat0 = apstats[1:]
        apstat1 = apstats.select_ids([1, 2])
        assert len(apstat0) == len(apstat1) == 2

        apstat0 = apstats[1:]
        apstat1 = apstats.select_ids([1, 2])
        assert len(apstat0) == len(apstat1) == 2

        apstat0 = apstats[[2, 1, 0]]
        apstat1 = apstats.select_ids([3, 2, 1])
        assert len(apstat0) == len(apstat1) == 3
        assert_equal(apstat0.id, [3, 2, 1])
        assert_equal(apstat1.id, [3, 2, 1])

        # Test select_ids when ids are not sorted
        apstat0 = apstats[[2, 1, 0]]
        apstat1 = apstat0.select_ids(2)
        assert apstat1.id == 2

        mask = apstats.id >= 2
        apstat0 = apstats[mask]
        assert len(apstat0) == 2
        assert_equal(apstat0.id, [2, 3])

        # Test iter
        for (i, apstat) in enumerate(apstats):
            assert apstat.isscalar
            assert apstat.id == (i + 1)

        match = "Scalar 'ApertureStats' object has no len"
        with pytest.raises(TypeError, match=match):
            _ = len(apstats[0])

        apstat0 = apstats[0]
        match = "A scalar 'ApertureStats' object cannot be indexed"
        with pytest.raises(TypeError, match=match):
            apstat1 = apstat0[0]
        apstat0 = apstats[0]
        with pytest.raises(TypeError, match=match):
            apstat1 = apstat0[0]  # can't slice scalar object

        match = '-1 is not a valid source ID number'
        with pytest.raises(ValueError, match=match):
            apstat0 = apstats.select_ids([-1, 0])

    def test_fancy_slicing_list_cache(self):
        """
        Test fancy-index slicing of an evaluated list-valued cached
        property (e.g., the ``bbox`` BoundingBox list), which cannot
        be indexed directly with an array index.
        """
        apstats = self.apstats1.copy()
        bbox = apstats.bbox  # cache the per-source BoundingBox list
        sliced = apstats[[2, 1, 0]]
        assert sliced.bbox == [bbox[2], bbox[1], bbox[0]]

    def test_scalar_aperture_stats(self):
        apstats = self.apstats1[0]
        assert apstats.n_positions == 1
        assert apstats.id == np.array([1])
        tbl = apstats.to_table()
        assert len(tbl) == 1

    def test_select_ids_scalar(self):
        """
        Test that select_id(s) on a scalar object raises the same clear
        TypeError as indexing it.
        """
        apstats = self.apstats1[0]
        match = "A scalar 'ApertureStats' object cannot be indexed"
        with pytest.raises(TypeError, match=match):
            apstats.select_ids(1)
        with pytest.raises(TypeError, match=match):
            apstats.select_id(1)

    @pytest.mark.parametrize('cached', ['moments', 'moments_central',
                                        'cutout_centroid', 'centroid',
                                        'covariance_eigvals', 'data_cutout',
                                        'bbox', 'x_centroid'])
    def test_scalar_slice_after_cache(self, cached):
        """
        Test that per-source properties are computed correctly for a
        scalar instance obtained by slicing a multi-source instance when
        only *some* properties were cached on the parent before slicing.

        The ``_array`` accessor is used internally by several properties
        to index along the source axis. A per-source array can be stored
        either with its leading length-1 axis intact (freshly computed
        for a scalar instance) or with that axis removed (a multi-source
        value cached on the parent and then sliced). If ``_array`` does
        not normalize both cases, moment-derived properties such as
        ``cutout_centroid`` raise an ``IndexError`` (or return wrong
        values) when a dependency was cached on the parent but the
        dependent property was not.

        This parametrizes over the dependencies that feed other
        properties through ``_array``, caches exactly one of them on
        the parent, slices to a scalar, and checks that every property
        matches the same scalar computed from scratch.
        """
        scalar_props = ('isscalar', 'n_positions')

        # Reference scalar instance that recomputes every property from
        # scratch (nothing cached on the parent before slicing).
        ref = ApertureStats(self.data, self.aperture, error=self.error,
                            wcs=self.wcs, sigma_clip=None)[0]

        # Cache exactly one dependency on the parent, then slice.
        parent = ApertureStats(self.data, self.aperture, error=self.error,
                               wcs=self.wcs, sigma_clip=None)
        _ = getattr(parent, cached)
        scalar = parent[0]

        for prop in ref.properties:
            if prop in scalar_props:
                continue
            assert_equal(getattr(scalar, prop), getattr(ref, prop))


class TestDeprecations(BaseApertureStatsData):
    """
    Tests for the deprecated attributes and columns.
    """

    def test_deprecated_attributes(self):
        """
        Test that deprecated attributes are still available and
        give the same value as the new attributes, but raise an
        AstropyDeprecationWarning.
        """
        apstats = ApertureStats(self.data, self.aperture, error=self.error)
        match = 'attribute was deprecated'
        deprecated_map = {
            'covar_sigx2': 'covariance_xx',
            'covar_sigxy': 'covariance_xy',
            'covar_sigy2': 'covariance_yy',
            'cxx': 'ellipse_cxx',
            'cxy': 'ellipse_cxy',
            'cyy': 'ellipse_cyy',
            'data_sumcutout': 'data_sum_cutout',
            'error_sumcutout': 'error_sum_cutout',
            'get_id': 'select_id',
            'get_ids': 'select_ids',
            'semimajor_sigma': 'semimajor_axis',
            'semiminor_sigma': 'semiminor_axis',
            'xcentroid': 'x_centroid',
            'ycentroid': 'y_centroid',
        }
        for old_name, new_name in deprecated_map.items():
            with pytest.warns(AstropyDeprecationWarning, match=match):
                old_val = getattr(apstats, old_name)
            new_val = getattr(apstats, new_name)
            assert_equal(old_val, new_val)

    def test_deprecated_ids(self):
        """
        Test that the ``ids`` attribute is deprecated and returns the
        same value as the ``id`` attribute.
        """
        apstats = ApertureStats(self.data, self.aperture)
        match = 'deprecated in version 3.1'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            old_val = apstats.ids
        assert_equal(old_val, apstats.id)

        # A scalar instance returns the scalar id
        scalar = apstats[0]
        with pytest.warns(AstropyDeprecationWarning, match=match):
            assert scalar.ids == scalar.id

    def test_deprecated_n_apertures(self):
        """
        Test that the ``n_apertures`` attribute is deprecated and
        returns the same value as the ``n_positions`` attribute.
        """
        apstats = ApertureStats(self.data, self.aperture)
        match = 'deprecated in version 3.1'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            old_val = apstats.n_apertures
        assert old_val == apstats.n_positions


class TestInputValidation(BaseApertureStatsData):
    """
    Tests for the input validation and NDData input.
    """

    def test_invalid_inputs(self):
        match = 'aperture must be an Aperture or Region object'
        with pytest.raises(TypeError, match=match):
            ApertureStats(self.data, 10.0)

        with (patch.dict(sys.modules, {'regions': None}),
              pytest.raises(TypeError, match=match)):
            ApertureStats(self.data, 10.0)

        match = 'sigma_clip must be a SigmaClip instance'
        with pytest.raises(TypeError, match=match):
            ApertureStats(self.data, self.aperture, sigma_clip=10)

        match = 'error must be a 2D array'
        with pytest.raises(ValueError, match=match):
            ApertureStats(self.data, self.aperture, error=10.0)

        match = 'error must be a 2D array'
        with pytest.raises(ValueError, match=match):
            ApertureStats(self.data, self.aperture, error=np.ones(3))

        match = 'data and error must have the same shape'
        with pytest.raises(ValueError, match=match):
            ApertureStats(self.data, self.aperture, error=np.ones((3, 3)))

    @pytest.mark.parametrize('with_units', [True, False])
    def test_nddata_input(self, with_units):
        mask = np.zeros(self.data.shape, dtype=bool)
        mask[225:240, 80:90] = True  # partially mask id=2

        data = self.data
        error = self.error
        if with_units:
            unit = u.Jy
            data <<= unit
            error <<= unit
        else:
            unit = None

        apstats1 = ApertureStats(data, self.aperture, error=error,
                                 mask=mask, wcs=self.wcs, sigma_clip=None)

        uncertainty = StdDevUncertainty(self.error)
        nddata = NDData(self.data, uncertainty=uncertainty, mask=mask,
                        wcs=self.wcs, unit=unit)
        apstats2 = ApertureStats(nddata, self.aperture, sigma_clip=None)

        assert_allclose(apstats1.x_centroid, apstats2.x_centroid)
        assert_allclose(apstats1.y_centroid, apstats2.y_centroid)
        assert_allclose(apstats1.sum, apstats2.sum)

        if with_units:
            assert apstats1.sum.unit == unit

        match = 'keyword is ignored. Its value is obtained from the input'
        nddata = NDData(self.data, uncertainty=uncertainty, mask=mask,
                        wcs=self.wcs, unit=unit)
        with pytest.warns(AstropyUserWarning, match=match):
            ApertureStats(nddata, self.aperture, mask=mask)


class TestMaskPathParity(BaseApertureStatsData):
    """
    The mask-based code path must give the same results as the
    fast batch code path.
    """

    @pytest.mark.parametrize('aperture', [
        CircularAperture(((145.1, 168.3), (84.7, 224.1), (48.3, 200.3)), r=6),
        CircularAnnulus(((145.1, 168.3), (84.7, 224.1), (48.3, 200.3)),
                        r_in=4, r_out=8),
        EllipticalAperture(((145.1, 168.3), (84.7, 224.1), (48.3, 200.3)),
                           8, 4, theta=0.5),
        EllipticalAnnulus(((145.1, 168.3), (84.7, 224.1), (48.3, 200.3)),
                          4, 8, 5, theta=0.5),
        RectangularAperture(((145.1, 168.3), (84.7, 224.1), (48.3, 200.3)),
                            10, 6, theta=0.3),
        RectangularAnnulus(((145.1, 168.3), (84.7, 224.1), (48.3, 200.3)),
                           8, 12, 9, theta=0.3),
    ])
    @pytest.mark.parametrize('sigma_clip', [
        None,
        SigmaClip(sigma=3.0, maxiters=5),
        SigmaClip(sigma=2.5, cenfunc='mean', stdfunc='mad_std',
                  maxiters=None),
    ])
    def test_fast_matches_mask(self, aperture, sigma_clip):
        """
        Regression test that the fast Cython statistics path produces the
        same results as the mask-based path for every property, including
        when sigma-clipping is applied.
        """
        numeric_props = (
            'sum', 'sum_err', 'sum_aper_area', 'center_aper_area',
            'min', 'max', 'mean', 'median', 'mode', 'std', 'var',
            'mad_std', 'biweight_location', 'biweight_midvariance', 'gini',
            'x_centroid', 'y_centroid', 'fwhm', 'semimajor_axis',
            'semiminor_axis', 'orientation', 'eccentricity', 'elongation',
            'ellipticity', 'covariance_xx', 'covariance_yy',
            'covariance_xy', 'moments', 'moments_central',
            'mean_err', 'median_err')

        fast = ApertureStats(self.data, aperture, error=self.error,
                             sigma_clip=sigma_clip)

        # Force the mask-based path by disabling the batch-driver
        # inputs, which disables both the fast center-value gather
        # and the fast sum path. Patching only ``_fast_gather`` is
        # insufficient because ``_fast_sum`` would still handle the
        # ``sum``, ``sum_err``, and ``sum_aper_area`` comparisons.
        with patch.object(ApertureStats, '_batch_inputs',
                          property(lambda _: None)):
            slow = ApertureStats(self.data, aperture, error=self.error,
                                 sigma_clip=sigma_clip)
            assert slow._fast_gather is None
            assert slow._fast_sum is None
            slow_values = {prop: getattr(slow, prop)
                           for prop in numeric_props}

        for prop in numeric_props:
            assert_allclose(getattr(fast, prop), slow_values[prop],
                            rtol=1e-7, atol=1e-9, equal_nan=True,
                            err_msg=f'mismatch in {prop}')

    @pytest.mark.parametrize('sigma_clip', [
        SigmaClip(sigma=3.0, cenfunc=np.nanmedian),
        SigmaClip(sigma=3.0, stdfunc=np.nanstd),
        SigmaClip(sigma=3.0, grow=1.5),
    ])
    def test_sigma_clip_fallback(self, sigma_clip):
        """
        Test that a SigmaClip not supported by the fast clipping kernel
        (callable cenfunc/stdfunc or spatial growing) falls back to the
        mask-based code path.
        """
        apstats = ApertureStats(self.data, self.aperture, error=self.error,
                                sigma_clip=sigma_clip)
        assert apstats._fast_gather is None
        assert np.all(np.isfinite(apstats.mean))
        assert np.all(np.isfinite(apstats.mean_err))

    def test_sigma_clip_callable(self):
        """
        Test that a SigmaClip with callable cenfunc/stdfunc (mask-based
        path) gives the same results as the equivalent string
        cenfunc/stdfunc (fast path).
        """
        fast = ApertureStats(self.data, self.aperture, error=self.error,
                             sigma_clip=SigmaClip(sigma=3.0))
        slow = ApertureStats(self.data, self.aperture, error=self.error,
                             sigma_clip=SigmaClip(sigma=3.0,
                                                  cenfunc=np.nanmedian,
                                                  stdfunc=np.nanstd))
        assert fast._fast_gather is not None
        assert slow._fast_gather is None
        for prop in ('mean', 'median', 'std', 'mean_err', 'median_err'):
            assert_allclose(getattr(fast, prop), getattr(slow, prop),
                            rtol=1e-7, equal_nan=True, err_msg=prop)

    def test_batch_hook_fallback(self):
        """
        Test that apertures that do not opt into the batch driver (no
        own-class hook, or a hook returning None) use the mask-based
        path and give the same results as the fast path.
        """
        nohook = ApertureStats(self.data,
                               NoBatchCircularAperture(self.positions, r=5))
        nospec = ApertureStats(self.data,
                               _NoneSpecCircular(self.positions, r=5))
        assert nohook._fast_gather is None
        assert nospec._fast_gather is None
        ref = ApertureStats(self.data, self.aperture)
        assert_allclose(nohook.mean, ref.mean, equal_nan=True)
        assert_allclose(nospec.mean, ref.mean, equal_nan=True)

    def test_data_mask_fallback(self):
        """
        Test that data/mask inputs not supported by the fast batch
        driver fall back to the mask-based code path.
        """
        mdata = np.ma.MaskedArray(
            self.data, mask=np.zeros(self.data.shape, dtype=bool))
        assert ApertureStats(mdata, self.aperture)._fast_gather is None
        mask = np.zeros(self.data.shape, dtype=int)
        assert ApertureStats(self.data, self.aperture,
                             mask=mask)._fast_gather is None

    @pytest.mark.parametrize('sum_method', ['exact', 'center'])
    @pytest.mark.parametrize('with_units', [False, True])
    def test_slow_path_extended(self, sum_method, with_units):
        """
        Test that the mask-based code path matches the fast path for data
        units, ``sum_method='center'``, and a non-overlapping source,
        exercising the unit-application and no-overlap branches.
        """
        positions = [*self.positions, (-50.0, -50.0)]  # last has no overlap
        data = self.data
        error = self.error
        if with_units:
            data = data * self.unit
            error = error * self.unit
        fast = ApertureStats(data, CircularAperture(positions, r=5),
                             error=error, sum_method=sum_method)
        slow = ApertureStats(data, NoBatchCircularAperture(positions, r=5),
                             error=error, sum_method=sum_method)
        assert fast._fast_gather is not None
        assert slow._fast_gather is None
        props = ('sum', 'sum_err', 'center_aper_area', 'sum_aper_area',
                 'min', 'max', 'mean', 'median', 'mode', 'std', 'var',
                 'mad_std', 'mean_err', 'median_err', 'moments',
                 'moments_central')
        for prop in props:
            assert_allclose(np.asarray(getattr(fast, prop)),
                            np.asarray(getattr(slow, prop)),
                            rtol=1e-7, atol=1e-9, equal_nan=True,
                            err_msg=prop)

    def test_slow_path_scalar(self):
        """
        Test the mask-based code path for a scalar (single-aperture)
        catalog, exercising the scalar branches of the mask-based
        helpers.
        """
        slow = ApertureStats(self.data,
                             NoBatchCircularAperture(self.positions[0], r=5),
                             error=self.error)
        fast = ApertureStats(self.data,
                             CircularAperture(self.positions[0], r=5),
                             error=self.error)
        assert slow.isscalar
        assert slow._fast_gather is None
        for prop in ('mean', 'median', 'std', 'mean_err', 'median_err',
                     'moments', 'sum'):
            assert_allclose(np.asarray(getattr(fast, prop)),
                            np.asarray(getattr(slow, prop)),
                            rtol=1e-7, equal_nan=True, err_msg=prop)

    def test_slow_path_segmentation(self):
        """
        Test that the mask-based code path applies segmentation-based
        masking identically to the fast path.
        """
        data = np.ones((50, 50))
        segm = np.zeros((50, 50), dtype=int)
        data[18:25, 18:25] = 10.0
        segm[18:25, 18:25] = 1
        data[20:25, 26:32] = 100.0
        segm[20:25, 26:32] = 2
        positions = [(21, 21), (28, 22)]
        kwargs = {'segmentation_image': segm, 'labels': [1, 2],
                  'mask_method': 'mask'}
        fast = ApertureStats(data, CircularAperture(positions, r=6),
                             **kwargs)
        slow = ApertureStats(data, NoBatchCircularAperture(positions, r=6),
                             **kwargs)
        assert slow._fast_gather is None
        for prop in ('sum', 'mean', 'median', 'std', 'mean_err',
                     'median_err'):
            assert_allclose(getattr(fast, prop), getattr(slow, prop),
                            rtol=1e-7, equal_nan=True, err_msg=prop)

    def test_slow_path_mask(self):
        """
        Test that the mask-based code path applies the input ``mask``
        identically to the fast path.
        """
        rng = np.random.default_rng(123)
        mask = rng.random(self.data.shape) > 0.7
        fast = ApertureStats(self.data, CircularAperture(self.positions, r=5),
                             error=self.error, mask=mask)
        slow_aper = NoBatchCircularAperture(self.positions, r=5)
        slow = ApertureStats(self.data, slow_aper,
                             error=self.error, mask=mask)
        assert slow._fast_gather is None
        for prop in ('sum', 'mean', 'median', 'std', 'mean_err',
                     'median_err'):
            assert_allclose(getattr(fast, prop), getattr(slow, prop),
                            rtol=1e-7, equal_nan=True, err_msg=prop)

    def test_mask_nonfinite_data(self):
        """
        Test that the fast batch driver correctly combines a
        user-supplied ``mask`` with the mask derived from non-finite
        ``data`` values, giving the same result as the mask-based code
        path.
        """
        data = self.data.copy()
        data[168, 145] = np.nan
        rng = np.random.default_rng(123)
        mask = rng.random(data.shape) > 0.7
        fast = ApertureStats(data, CircularAperture(self.positions, r=5),
                             error=self.error, mask=mask)
        assert fast._batch_inputs is not None
        slow_aper = NoBatchCircularAperture(self.positions, r=5)
        slow = ApertureStats(data, slow_aper,
                             error=self.error, mask=mask)
        assert slow._fast_gather is None
        for prop in ('sum', 'mean', 'median', 'std', 'mean_err',
                     'median_err'):
            assert_allclose(getattr(fast, prop), getattr(slow, prop),
                            rtol=1e-7, equal_nan=True, err_msg=prop)


class TestStandardErrors(BaseApertureStatsData):
    """
    Tests for the mean_err and median_err standard errors.
    """

    def test_mean_err_median_err(self):
        """
        Test the standard-error properties against their definitions.
        """
        apstats = ApertureStats(self.data, self.aperture, error=self.error)
        n_pixels = apstats.center_aper_area.value
        sample_std = apstats.std * np.sqrt(n_pixels / (n_pixels - 1.0))
        expected_mean_err = sample_std / np.sqrt(n_pixels)
        assert_allclose(apstats.mean_err, expected_mean_err)
        assert_allclose(apstats.median_err,
                        np.sqrt(np.pi / 2.0) * expected_mean_err)

    def test_mean_err_single_pixel(self):
        """
        Test that the standard error is NaN for apertures with fewer than
        two unmasked pixels.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 1.0
        apstats = ApertureStats(data, CircularAperture((5, 5), r=0.5))
        assert apstats.center_aper_area.value < 2
        assert np.isnan(apstats.mean_err)
        assert np.isnan(apstats.median_err)

    def test_mean_err_units(self):
        """
        Test that the standard-error properties carry the data unit.
        """
        apstats = ApertureStats(self.data * self.unit, self.aperture,
                                error=self.error * self.unit)
        assert apstats.mean_err.unit == self.unit
        assert apstats.median_err.unit == self.unit


class TestDdof(BaseApertureStatsData):
    """
    Tests for the ddof keyword used by the var and std properties.
    """

    @pytest.mark.parametrize('with_units', [False, True])
    def test_ddof(self, with_units):
        """
        Test that the ``ddof`` keyword rescales ``var`` and ``std`` by
        ``N / (N - ddof)`` and that the standard errors are unaffected.
        """
        data = self.data
        error = self.error
        if with_units:
            data = data * self.unit
            error = error * self.unit
        apstats0 = ApertureStats(data, self.aperture, error=error)
        apstats1 = ApertureStats(data, self.aperture, error=error, ddof=1)
        n_pixels = apstats0.center_aper_area.value
        var0 = apstats0.var
        std0 = apstats0.std
        var1 = apstats1.var
        std1 = apstats1.std
        if with_units:
            assert var1.unit == self.unit**2
            assert std1.unit == self.unit
            var0, std0 = var0.value, std0.value
            var1, std1 = var1.value, std1.value
        assert_allclose(var1, var0 * n_pixels / (n_pixels - 1.0))
        assert_allclose(std1, std0 * np.sqrt(n_pixels / (n_pixels - 1.0)))
        # The standard errors are defined with the sample std and are
        # independent of the ``ddof`` keyword
        assert_allclose(apstats0.mean_err, apstats1.mean_err, equal_nan=True)
        assert_allclose(apstats0.median_err, apstats1.median_err,
                        equal_nan=True)

    def test_ddof_slow_path(self):
        """
        Test the ``ddof`` rescaling on the mask-based (slow) code path.
        """
        fast = ApertureStats(self.data, CircularAperture(self.positions, r=5),
                             ddof=1)
        slow_aper = NoBatchCircularAperture(self.positions, r=5)
        slow = ApertureStats(self.data, slow_aper, ddof=1)
        assert slow._fast_gather is None
        assert_allclose(fast.var, slow.var, equal_nan=True)
        assert_allclose(fast.std, slow.std, equal_nan=True)

    def test_ddof_undefined(self):
        """
        Test that ``var`` and ``std`` are NaN when ``N <= ddof``.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 1.0
        aper = CircularAperture((5, 5), r=0.5)
        apstats = ApertureStats(data, aper, ddof=1)
        assert apstats.center_aper_area.value < 2
        assert np.isnan(apstats.var)
        assert np.isnan(apstats.std)
        # The default (ddof=0) gives a finite (zero) variance
        assert ApertureStats(data, aper).var == 0.0

    def test_ddof_invalid(self):
        """
        Test that an invalid ``ddof`` raises a ``ValueError``.
        """
        match = 'ddof must be a non-negative integer'
        for bad in (-1, 1.5, True, False):
            with pytest.raises(ValueError, match=match):
                ApertureStats(self.data, self.aperture, ddof=bad)


class TestMadStdScale:
    """
    The mad_std scale factor must match astropy.stats.mad_std.
    """

    def test_scale(self):
        """
        Test that the hard-coded MAD-to-sigma scale factor matches
        `astropy.stats.mad_std`.

        The factor is duplicated in ``photutils.aperture.stats`` and
        ``photutils.aperture._batch_stats`` (a Cython constant cannot be
        shared directly); the fast-vs-mask regression tests cover the
        Cython copy.
        """
        rng = np.random.default_rng(0)
        values = rng.normal(size=101)
        mad = np.median(np.abs(values - np.median(values)))
        assert_allclose(_MAD_STD_SCALE * mad, mad_std(values), rtol=1e-14)


class TestRegionInput:
    """
    A regions.Region input must give the same results as the
    equivalent aperture.
    """

    @pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
    def test_region_matches_aperture(self):
        from regions import CirclePixelRegion, PixCoord
        region = CirclePixelRegion(center=PixCoord(5, 5), radius=3)
        aperture = CircularAperture((5, 5), r=3)
        data = np.ones((10, 10))
        apstats1 = ApertureStats(data, region)
        apstats2 = ApertureStats(data, aperture)
        tbl = apstats1.to_table()
        for colname in tbl.colnames:
            val1 = getattr(apstats1, colname)
            if val1 is not None:
                assert_allclose(val1, getattr(apstats2, colname))


class TestApertureMetadata:
    """
    Tests for the aperture metadata stored in the output table.
    """

    def test_metadata(self):
        x = [10, 20, 3]
        y = [3, 5, 10]
        xypos = list(zip(x, y, strict=False))
        a1 = CircularAperture(xypos, r=3)
        a2 = CircularAperture(xypos, r=4)
        a3 = CircularAnnulus(xypos, 5, 10)
        a4 = EllipticalAperture(xypos, 10, 5, theta=10 * u.deg)
        a5 = EllipticalAnnulus(xypos, a_in=5, a_out=10, b_in=3, b_out=5,
                               theta=20 * u.deg)
        a6 = RectangularAperture(xypos, 10, 5, theta=30 * u.deg)
        a7 = RectangularAnnulus(xypos, w_in=5, w_out=10, h_in=3, h_out=5,
                                theta=40 * u.deg)
        apers = (a1, a2, a3, a4, a5, a6, a7)
        data = np.ones((50, 50))

        for aper in apers:
            apstats = ApertureStats(data, aper)
            tbl = apstats.to_table()
            assert tbl.meta['aperture'] == aper.__class__.__name__
            params = aper._params
            for param in params:
                if param != 'positions':
                    assert (tbl.meta[f'aperture_{param}']
                            == getattr(aper, param))

        wcs = make_wcs(data.shape)
        skycoord = wcs.pixel_to_world(10, 10)
        unit = u.arcsec
        saper = SkyEllipticalAnnulus(skycoord, a_in=0.1 * unit,
                                     a_out=0.2 * unit,
                                     b_in=0.05 * unit, b_out=0.1 * unit,
                                     theta=10 * u.deg)
        apstats = ApertureStats(data, saper, wcs=wcs)
        tbl = apstats.to_table()
        assert tbl.meta['aperture'] == saper.__class__.__name__
        assert tbl.meta['aperture_a_in'] == saper.a_in
        assert tbl.meta['aperture_a_out'] == saper.a_out
        assert tbl.meta['aperture_b_out'] == saper.b_out
        assert tbl.meta['aperture_theta'] == saper.theta


class TestSigmaClipSharedInstance:
    """
    Tests for calling a shared SigmaClip instance from the astropy
    fallback clipping path.

    Astropy SigmaClip stores internal state on the instance during
    calls, and the fallback path is reachable from two different cached
    properties (the center- and sum-footprint cutouts) that do not share
    a lock. ApertureStats must therefore never call the user's SigmaClip
    instance directly.
    """

    def test_user_sigma_clip_instance_not_called(self, monkeypatch):
        """
        Test that the user's SigmaClip instance is never called directly
        (an internal copy must be used).
        """
        data = np.ones((51, 51))
        data[20:30, 20:30] = 100.0
        aper = CircularAperture([(25, 25), (10, 10)], r=8)
        # A callable cenfunc is unsupported by the batch clipping
        # kernels and forces the astropy SigmaClip fallback path.
        sigclip = SigmaClip(sigma=3.0, maxiters=5, cenfunc=np.nanmedian)

        called_ids = []
        orig_call = SigmaClip.__call__

        def spy(self, *args, **kwargs):
            called_ids.append(id(self))
            return orig_call(self, *args, **kwargs)

        monkeypatch.setattr(SigmaClip, '__call__', spy)
        apstats = ApertureStats(data, aper, sigma_clip=sigclip,
                                sum_method='exact')
        _ = apstats.mean
        _ = apstats.sum

        assert len(called_ids) > 0  # the fallback path was exercised
        assert id(sigclip) not in called_ids

    def test_fallback_thread_safety(self):
        """
        Test that computing a center-footprint and a sum-footprint
        property concurrently on the same instance gives the same
        results as the serial computation.
        """
        from concurrent.futures import ThreadPoolExecutor

        rng = np.random.default_rng(0)
        n_src = 200
        ny = 51
        yc = ny // 2
        xs = np.arange(25, 25 + 50 * n_src, 50)
        data = rng.normal(0.0, 1.0, (ny, 50 * n_src + 50))
        for i, x in enumerate(xs):
            # Strong per-source scale differences make any clipping
            # bound contamination between sources visible
            scale = 10.0 ** (i % 6)
            data[:, x - 20:x + 20] = rng.normal(scale, scale / 10.0,
                                                (ny, 40))
            data[yc - 2:yc + 2, x - 2:x + 2] = scale * 50.0  # outliers

        positions = np.column_stack([xs, np.full(n_src, yc)])
        aper = CircularAperture(positions, r=15)
        sigclip = SigmaClip(sigma=2.0, maxiters=10, cenfunc=np.nanmedian)

        ref = ApertureStats(data, aper, sigma_clip=sigclip,
                            sum_method='exact')
        ref_mean = ref.mean
        ref_sum = ref.sum

        for _ in range(3):
            apstats = ApertureStats(data, aper, sigma_clip=sigclip,
                                    sum_method='exact')
            with ThreadPoolExecutor(max_workers=2) as executor:
                mean_future = executor.submit(getattr, apstats, 'mean')
                sum_future = executor.submit(getattr, apstats, 'sum')
                mean = mean_future.result()
                total = sum_future.result()
            assert_allclose(mean, ref_mean)
            assert_allclose(total, ref_sum)


class TestSigmaClipBiweightStrings:
    """
    Tests for the SigmaClip 'biweight' cenfunc/stdfunc string options in
    the fast clipping kernels.
    """

    @staticmethod
    def _make_sigma_clip_biweight(**kwargs):
        """
        Create a SigmaClip instance using the 'biweight' cenfunc and
        stdfunc string options, emulating them on astropy versions that
        do not support these options.
        """

        def nanbiloc(data, axis=None):
            return biweight_location(data, axis=axis, ignore_nan=True)

        def nanbiscale(data, axis=None):
            return biweight_scale(data, axis=axis, ignore_nan=True)

        try:
            return SigmaClip(cenfunc='biweight', stdfunc='biweight',
                             **kwargs)
        except ValueError:
            sigma_clip = SigmaClip(**kwargs)
            sigma_clip.cenfunc = 'biweight'
            sigma_clip.stdfunc = 'biweight'
            sigma_clip._cenfunc_parsed = nanbiloc
            sigma_clip._stdfunc_parsed = nanbiscale
            return sigma_clip

    def test_biweight_matches_fallback(self, monkeypatch):
        """
        Test that the 'biweight' string options are accepted by the fast
        clipping kernels and match the mask-based fallback path, which
        clips using the astropy SigmaClip instance.
        """
        rng = np.random.default_rng(0)
        data = rng.normal(10.0, 2.0, (101, 101))
        data[::7, ::7] = 200.0  # outliers
        positions = [(20.0, 20.0), (50.0, 50.0), (80.5, 80.5)]
        aper = CircularAperture(positions, r=12)
        sigclip = self._make_sigma_clip_biweight(sigma=2.0, maxiters=5)

        apstats1 = ApertureStats(data, aper, sigma_clip=sigclip,
                                 sum_method='exact')
        assert apstats1._fast_clip_spec() is not None
        mean1 = apstats1.mean
        std1 = apstats1.std
        sum1 = apstats1.sum

        # Force the mask-based fallback path
        monkeypatch.setattr(ApertureStats, '_fast_clip_spec',
                            lambda _self: None)
        apstats2 = ApertureStats(data, aper, sigma_clip=sigclip,
                                 sum_method='exact')
        assert apstats2._fast_clip_spec() is None

        assert_allclose(mean1, apstats2.mean, rtol=1e-10)
        assert_allclose(std1, apstats2.std, rtol=1e-10)
        assert_allclose(sum1, apstats2.sum, rtol=1e-10)


def test_sigma_clip_sorted_values_reuse(monkeypatch):
    """
    Test that the order statistics computed from the sigma-clip kernel's
    sorted output (reused by _sorted_values instead of re-sorting the
    clipped values) match the mask-based fallback path.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 2.0, (101, 101))
    data[::5, ::5] = 200.0  # outliers
    positions = [(20.0, 20.0), (50.0, 50.0), (80.5, 80.5), (3.0, 3.0)]
    aper = CircularAperture(positions, r=12)
    sigclip = SigmaClip(sigma=2.0, maxiters=10)

    apstats1 = ApertureStats(data, aper, sigma_clip=sigclip)
    props = ('min', 'max', 'median', 'mad_std', 'biweight_location',
             'std', 'mean')
    results1 = [getattr(apstats1, prop) for prop in props]

    # Force the mask-based fallback path
    monkeypatch.setattr(ApertureStats, '_fast_clip_spec',
                        lambda _self: None)
    apstats2 = ApertureStats(data, aper, sigma_clip=sigclip)
    for prop, result1 in zip(props, results1, strict=True):
        assert_allclose(result1, getattr(apstats2, prop), rtol=1e-10)


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

    @staticmethod
    def assert_stats_equal(stats1, stats2):
        """
        Assert that all (non-cutout) properties of two ApertureStats
        instances are exactly equal.
        """
        for prop in stats1.properties:
            value1 = getattr(stats1, prop)
            value2 = getattr(stats2, prop)
            if prop.endswith('cutout'):
                for arr1, arr2 in zip(value1, value2, strict=True):
                    arr1 = np.ma.filled(arr1, np.nan)
                    arr2 = np.ma.filled(arr2, np.nan)
                    assert_equal(arr1, arr2)
            elif prop.startswith('sky_'):
                # SkyCoord without a wcs is None or an array of None
                if hasattr(value1, 'ra'):
                    assert_equal(value1.ra.deg, value2.ra.deg)
                    assert_equal(value1.dec.deg, value2.dec.deg)
                else:
                    assert_equal(value1, value2)
            else:
                assert_equal(value1, value2)

    @pytest.mark.parametrize('n_threads', [2, 8])
    @pytest.mark.parametrize('use_sigma_clip', [False, True])
    def test_identical_results(self, n_threads, use_sigma_clip):
        """
        Test that multithreaded statistics are identical to the
        single-threaded computation for every property, including for
        off-edge positions, masked pixels, and non-finite data values.
        """
        data, error, mask, positions = self.make_inputs()
        wcs = make_wcs(data.shape)
        sigma_clip = (SigmaClip(sigma=3.0, maxiters=10) if use_sigma_clip
                      else None)
        aper = CircularAperture(positions, r=7.0)
        stats1 = ApertureStats(data, aper, error=error, mask=mask,
                               wcs=wcs, sigma_clip=sigma_clip)
        stats2 = ApertureStats(data, aper, error=error, mask=mask,
                               wcs=wcs, sigma_clip=sigma_clip,
                               n_threads=n_threads)
        assert stats2.n_threads == n_threads
        self.assert_stats_equal(stats1, stats2)

    def test_local_bkg_ddof_sum_method(self):
        """
        Test that the per-source local background values are chunked
        together with the positions.
        """
        data, error, _, positions = self.make_inputs()
        local_bkg = np.linspace(-1.0, 1.0, positions.shape[0])
        aper = CircularAperture(positions, r=6.0)
        kwargs = {'error': error, 'local_bkg': local_bkg, 'ddof': 1,
                  'sum_method': 'center'}
        stats1 = ApertureStats(data, aper, **kwargs)
        stats2 = ApertureStats(data, aper, n_threads=4, **kwargs)
        for prop in ('sum', 'sum_err', 'std', 'median', 'mean'):
            assert_equal(getattr(stats1, prop), getattr(stats2, prop))

    def test_segmentation_masking(self):
        """
        Test that the per-source segmentation labels are chunked
        together with the positions.
        """
        data = np.ones((50, 50))
        segm = np.zeros((50, 50), dtype=int)
        data[18:25, 18:25] = 10.0
        segm[18:25, 18:25] = 1
        data[20:25, 26:32] = 100.0
        segm[20:25, 26:32] = 2
        positions = [(21.0, 21.0), (28.0, 22.0), (21.0, 21.5),
                     (28.5, 22.0), (20.5, 21.0)]
        labels = [1, 2, 1, 2, 1]
        aper = CircularAperture(positions, r=8.0)
        kwargs = {'segmentation_image': segm, 'labels': labels,
                  'mask_method': 'mask'}
        stats1 = ApertureStats(data, aper, **kwargs)
        stats2 = ApertureStats(data, aper, n_threads=3, **kwargs)
        for prop in ('sum', 'mean', 'median', 'flags'):
            assert_equal(getattr(stats1, prop), getattr(stats2, prop))

    def test_more_threads_than_positions(self):
        """
        Test that n_threads larger than the number of positions gives
        identical results.
        """
        data = np.ones((40, 40))
        aper = CircularAperture([(20, 20), (10, 10), (30, 25)], r=5.0)
        stats1 = ApertureStats(data, aper)
        stats2 = ApertureStats(data, aper, n_threads=8)
        self.assert_stats_equal(stats1, stats2)

    def test_scalar_aperture(self):
        """
        Test that a scalar aperture position with n_threads > 1 falls
        back to a single-chunk (serial) computation.
        """
        data = np.ones((40, 40))
        aper = CircularAperture((20, 20), r=5.0)
        stats1 = ApertureStats(data, aper)
        stats2 = ApertureStats(data, aper, n_threads=4)
        assert_equal(stats1.sum, stats2.sum)
        assert_equal(stats1.median, stats2.median)

    def test_mask_based_fallback(self):
        """
        Test that apertures that do not support the batch code path
        give correct results with n_threads > 1 (the mask-based code
        path stays serial).
        """
        data, error, _, positions = self.make_inputs()
        aper = NoBatchCircularAperture(positions, r=7.0)
        stats = ApertureStats(data, aper, error=error, n_threads=4)
        assert stats._batch_inputs is None
        ref = ApertureStats(data, CircularAperture(positions, r=7.0),
                            error=error)
        assert_allclose(stats.sum, ref.sum, equal_nan=True)
        assert_allclose(stats.median, ref.median, equal_nan=True)

    def test_empty_positions(self):
        """
        Test that an aperture with zero positions works with
        n_threads > 1 (regression test for the chunking helper, which
        must fall back to the serial path for zero chunks).
        """
        data = np.ones((11, 11))
        aper = CircularAperture(np.empty((0, 2)), r=3.0)
        stats = ApertureStats(data, aper, n_threads=4)
        assert stats.n_positions == 0
        assert stats.sum.shape == (0,)
        assert stats.median.shape == (0,)

    def test_indexing_preserves_n_threads(self):
        """
        Test that slicing an ApertureStats instance preserves the
        n_threads value.
        """
        data = np.ones((40, 40))
        aper = CircularAperture([(20, 20), (10, 10), (30, 25)], r=5.0)
        stats = ApertureStats(data, aper, n_threads=4)
        assert stats[1:].n_threads == 4
        assert stats[0].n_threads == 4

    def test_invalid_n_threads(self):
        """
        Test that an error is raised if n_threads is not a positive
        integer.
        """
        data = np.ones((40, 40))
        aper = CircularAperture((20, 20), r=5.0)
        match = 'n_threads must be a positive integer'
        for n_threads in (0, -1, 2.5, True):
            with pytest.raises(ValueError, match=match):
                ApertureStats(data, aper, n_threads=n_threads)


def test_overridden_bbox_fallback():
    """
    Test that the bounding-box bounds honor an aperture subclass that
    overrides ``bbox``.

    The bounds are normally computed from the aperture's vectorized
    integer bounds without creating per-position BoundingBox objects;
    an overridden ``bbox`` must fall back to reading the objects.
    """
    data = np.ones((60, 60))
    positions = [(20.0, 20.0), (35.5, 24.2), (10.1, 40.7)]
    stats1 = ApertureStats(data, CircularAperture(positions, r=5.0))
    stats2 = ApertureStats(data, _ExpandedBboxCircular(positions, r=5.0))
    assert_equal(stats2.bbox_xmin, stats1.bbox_xmin - 1)
    assert_equal(stats2.bbox_xmax, stats1.bbox_xmax + 1)
    assert_equal(stats2.bbox_ymin, stats1.bbox_ymin - 1)
    assert_equal(stats2.bbox_ymax, stats1.bbox_ymax + 1)


class TestCenterCutoutParity:
    """
    The center-method cutouts reconstructed from the packed gather
    buffers must match the mask-based cutouts exactly, including the
    masked-pixel data values.
    """

    @staticmethod
    def make_inputs():
        """
        Build a deterministic image (with non-finite data and error
        values), error, mask, and positions (including partial-overlap
        and no-overlap positions).
        """
        rng = np.random.default_rng(0)
        data = rng.normal(100.0, 5.0, (80, 90))
        data[10, 10] = np.nan
        data[40, 41] = np.inf
        error = np.abs(rng.normal(5.0, 0.5, data.shape)) + 0.1
        error[12, 12] = np.inf
        mask = np.zeros(data.shape, dtype=bool)
        mask[30:35, 30:35] = True
        positions = [(5.0, 5.0), (-20.0, -20.0), (33.0, 32.0),
                     (45.5, 41.2), (88.0, 3.0), (12.3, 11.1)]
        return data, error, mask, positions

    @staticmethod
    def assert_cutout_lists_equal(fast_list, slow_list):
        """
        Assert two per-source cutout lists are exactly equal,
        including the data values under the mask.
        """
        for fast, slow in zip(fast_list, slow_list, strict=True):
            assert_equal(np.ma.getdata(fast), np.ma.getdata(slow))
            assert_equal(np.ma.getmaskarray(fast),
                         np.ma.getmaskarray(slow))

    @staticmethod
    def get_cutouts(stats):
        """
        Return the center-method cutout lists of an instance.
        """
        return {'data': stats.data_cutout,
                'variance': stats._variance_cutout_center,
                'mask': stats._mask_cutout_center,
                'weight': stats._weight_cutout_center}

    @pytest.mark.parametrize('use_sigma_clip', [False, True])
    @pytest.mark.parametrize('use_error', [False, True])
    def test_matches_mask_path(self, use_sigma_clip, use_error):
        """
        Test that the buffer-reconstructed cutouts equal the
        mask-based cutouts, including non-finite pixels, partial
        overlaps, and the no-overlap sentinel arrays.
        """
        data, error, mask, positions = self.make_inputs()
        sigma_clip = (SigmaClip(sigma=2.0, maxiters=10) if use_sigma_clip
                      else None)
        kwargs = {'error': error if use_error else None, 'mask': mask,
                  'sigma_clip': sigma_clip,
                  'local_bkg': np.linspace(-1.0, 1.0, len(positions))}
        aperture = CircularAperture(positions, r=6.0)

        fast = ApertureStats(data, aperture, **kwargs)
        assert fast._fast_cutouts_center is not None
        fast_cutouts = self.get_cutouts(fast)

        with patch.object(ApertureStats, '_batch_inputs',
                          property(lambda _: None)):
            slow = ApertureStats(data, aperture, **kwargs)
            assert slow._fast_cutouts_center is None
            # The mask-based path multiplies non-finite data values by
            # the boolean masks (NaN * 0), which emits a numpy
            # RuntimeWarning
            with np.errstate(invalid='ignore'):
                slow_cutouts = self.get_cutouts(slow)

        for key in fast_cutouts:
            if not use_error and key == 'variance':
                assert all(value is None for value in fast_cutouts[key])
                assert all(value is None for value in slow_cutouts[key])
                continue
            self.assert_cutout_lists_equal(fast_cutouts[key],
                                           slow_cutouts[key])

    @pytest.mark.parametrize('mask_method', ['mask', 'source_only'])
    def test_segmentation_masking(self, mask_method):
        """
        Test that segmentation-excluded pixels land in the cutout mask
        identically to the mask-based path.
        """
        data = np.ones((50, 50))
        segm = np.zeros((50, 50), dtype=int)
        data[18:25, 18:25] = 10.0
        segm[18:25, 18:25] = 1
        data[20:25, 26:32] = 100.0
        segm[20:25, 26:32] = 2
        aperture = CircularAperture([(21.0, 21.0), (28.0, 22.0)], r=8.0)
        kwargs = {'segmentation_image': segm, 'labels': [1, 2],
                  'mask_method': mask_method}

        fast = ApertureStats(data, aperture, **kwargs)
        assert fast._fast_cutouts_center is not None
        fast_data = fast.data_cutout

        with patch.object(ApertureStats, '_batch_inputs',
                          property(lambda _: None)):
            slow = ApertureStats(data, aperture, **kwargs)
            slow_data = slow.data_cutout

        self.assert_cutout_lists_equal(fast_data, slow_data)

    def test_correct_mode_fallback(self):
        """
        Test that mask_method='correct' uses the mask-based cutout
        path (the mirrored error values are not recoverable from the
        packed buffers) and still produces the corrected cutouts.
        """
        data = np.ones((50, 50))
        segm = np.zeros((50, 50), dtype=int)
        data[18:25, 18:25] = 10.0
        segm[18:25, 18:25] = 1
        data[20:25, 26:32] = 100.0
        segm[20:25, 26:32] = 2
        aperture = CircularAperture([(21.0, 21.0)], r=8.0)
        stats = ApertureStats(data, aperture, segmentation_image=segm,
                              labels=[1], mask_method='correct')
        assert stats._fast_gather is not None
        assert stats._fast_cutouts_center is None
        # The bright neighbor pixel values must not leak into the
        # cutout (they are replaced by their mirrored values)
        cutout = stats.data_cutout[0]
        assert np.ma.getdata(cutout).max() <= 10.0


class TestSumErrPartialPixelWeights:
    """
    Tests that ``sum_err`` weights the pixel variances by the squared
    overlap fractions.

    The reported sum is sum(w * data) over the aperture pixels, where
    ``w`` is the pixel overlap fraction. With independent pixel errors,
    its variance is sum(w**2 * sigma**2), so boundary pixels must
    contribute their variance weighted by the squared fraction.
    """

    position = (12.2, 11.7)
    radius = 3.3

    @staticmethod
    def _expected_error(aperture, error):
        """
        Compute sqrt(sum(w**2 * sigma**2)) from the exact aperture mask.
        """
        apermask = aperture.to_mask(method='exact')
        slc_large, slc_small = apermask.get_overlap_slices(error.shape)
        weights = apermask.data[slc_small]
        return np.sqrt(np.sum(weights**2 * error[slc_large] ** 2))

    @pytest.fixture(name='scene')
    def fixture_scene(self):
        """
        A deterministic uniform image and non-uniform error array.

        The data are uniform so that sigma clipping never rejects a
        pixel, keeping the expected error identical for the clipped and
        unclipped code paths.
        """
        rng = np.random.default_rng(seed=456)
        data = np.ones((25, 25))
        error = rng.uniform(0.5, 1.5, (25, 25))
        return data, error

    def test_batch_path(self, scene):
        data, error = scene
        aperture = CircularAperture(self.position, self.radius)
        apstats = ApertureStats(data, aperture, error=error)
        expected = self._expected_error(aperture, error)
        assert_allclose(apstats.sum_err, expected)

    def test_mask_path(self, scene):
        data, error = scene
        aperture = NoBatchCircularAperture(self.position, self.radius)
        apstats = ApertureStats(data, aperture, error=error)
        expected = self._expected_error(aperture, error)
        assert_allclose(apstats.sum_err, expected)

    def test_sigma_clip_batch_path(self, scene):
        data, error = scene
        aperture = CircularAperture(self.position, self.radius)
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        apstats = ApertureStats(data, aperture, error=error,
                                sigma_clip=sigma_clip)
        expected = self._expected_error(aperture, error)
        assert_allclose(apstats.sum_err, expected)

    def test_error_sum_cutout(self, scene):
        """
        The ``error_sum_cutout`` values are the pixel errors weighted by
        the overlap fractions, so that the quadrature sum of the cutout
        equals ``sum_err``.
        """
        data, error = scene
        aperture = CircularAperture(self.position, self.radius)
        apstats = ApertureStats(data, aperture, error=error)

        apermask = aperture.to_mask(method='exact')
        slc_large, slc_small = apermask.get_overlap_slices(error.shape)
        expected = apermask.data[slc_small] * error[slc_large]

        cutout = apstats.error_sum_cutout
        assert_allclose(cutout.filled(0.0), expected)
        assert_allclose(np.sqrt(np.sum(cutout.filled(0.0) ** 2)),
                        apstats.sum_err)
