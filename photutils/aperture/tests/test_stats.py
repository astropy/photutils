# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the stats module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import NDData, StdDevUncertainty
from astropy.stats import SigmaClip
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture.circle import CircularAperture
from photutils.aperture.stats import ApertureStats
from photutils.datasets import make_100gaussians_image, make_wcs


class TestApertureStats:
    data = make_100gaussians_image()
    error = np.sqrt(np.abs(data))
    wcs = make_wcs(data.shape)
    positions = [(145.1, 168.3), (84.7, 224.1), (48.3, 200.3)]
    aperture = CircularAperture(positions, r=5)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    apstats1 = ApertureStats(data, aperture, error=error, wcs=wcs,
                             sigma_clip=None)
    apstats2 = ApertureStats(data, aperture, error=error, wcs=wcs,
                             sigma_clip=sigclip)

    unit = u.Jy
    apstats1_units = ApertureStats(data * u.Jy, aperture,
                                   error=error * u.Jy, wcs=wcs,
                                   sigma_clip=None)
    apstats2_units = ApertureStats(data * u.Jy, aperture,
                                   error=error * u.Jy, wcs=wcs,
                                   sigma_clip=sigclip)

    @pytest.mark.parametrize('with_units', (True, False))
    @pytest.mark.parametrize('with_sigmaclip', (True, False))
    def test_properties(self, with_units, with_sigmaclip):
        apstats = [self.apstats1.copy(), self.apstats2.copy(),
                   self.apstats1_units.copy(), self.apstats2_units.copy()]
        if with_sigmaclip:
            index = [1, 3]
        else:
            index = [0, 2]
        if with_units:
            index = index[1]
        else:
            index = index[0]
        apstats1 = apstats[index]
        apstats2 = apstats1.copy()

        idx = 1

        scalar_props = ('isscalar', 'n_apertures')

        # evaluate (cache) properties before slice
        for prop in apstats1.properties:
            _ = getattr(apstats1, prop)
        apstats3 = apstats1[idx]
        for prop in apstats1.properties:
            if prop in scalar_props:
                continue
            assert_equal(getattr(apstats1, prop)[idx], getattr(apstats3, prop))

        # slice catalog before evaluating catalog properties
        apstats4 = apstats2[idx]
        for prop in apstats1.properties:
            if prop in scalar_props:
                continue
            assert_equal(getattr(apstats4, prop), getattr(apstats1, prop)[idx])

    def test_skyaperture(self):
        pix_apstats = ApertureStats(self.data, self.aperture, wcs=self.wcs)
        skyaper = self.aperture.to_sky(self.wcs)
        sky_apstats = ApertureStats(self.data, skyaper, wcs=self.wcs)

        exclude_props = ('bbox', 'error_sumcutout', 'sum_error',
                         'sky_centroid', 'sky_centroid_icrs')
        for prop in pix_apstats.properties:
            if prop in exclude_props:
                continue
            assert_allclose(getattr(pix_apstats, prop),
                            getattr(sky_apstats, prop), atol=1e-7)

        with pytest.raises(ValueError):
            _ = ApertureStats(self.data, skyaper)

    def test_minimal_inputs(self):
        apstats = ApertureStats(self.data, self.aperture)
        props = ('sky_centroid', 'sky_centroid_icrs', 'error_sumcutout')
        for prop in props:
            assert set(getattr(apstats, prop)) == {None}
        assert np.all(np.isnan(apstats.sum_err))
        assert set(apstats._variance_cutout) == {None}

        apstats = ApertureStats(self.data, self.aperture, sum_method='center')
        assert set(apstats._variance_cutout_center) == {None}

    @pytest.mark.parametrize('sum_method', ('exact', 'subpixel'))
    def test_sum_method(self, sum_method):
        apstats1 = ApertureStats(self.data, self.aperture, error=self.error,
                                 sum_method='center')
        apstats2 = ApertureStats(self.data, self.aperture, error=self.error,
                                 sum_method=sum_method, subpixels=4)

        scalar_props = ('isscalar', 'n_apertures')

        # evaluate (cache) properties before slice
        for prop in apstats1.properties:
            if prop in scalar_props:
                continue
            if 'sum' in prop:
                # test that these properties are not equal
                with pytest.raises(AssertionError):
                    assert_equal(getattr(apstats1, prop),
                                 getattr(apstats2, prop))
            else:
                assert_equal(getattr(apstats1, prop), getattr(apstats2, prop))

    def test_sum_method_photometry(self):
        for method in ('center', 'exact', 'subpixel'):
            subpixels = 4
            apstats = ApertureStats(self.data, self.aperture,
                                    error=self.error, sum_method=method,
                                    subpixels=subpixels)
            apsum, apsum_err = self.aperture.do_photometry(self.data,
                                                           self.error,
                                                           method=method,
                                                           subpixels=subpixels)
            assert_allclose(apstats.sum, apsum)
            assert_allclose(apstats.sum_err, apsum_err)

    def test_mask(self):
        mask = np.zeros(self.data.shape, dtype=bool)
        mask[225:240, 80:90] = True  # partially mask id=2
        mask[190:210, 40:60] = True  # completely mask id=3

        apstats = ApertureStats(self.data, self.aperture, mask=mask,
                                error=self.error)

        # id=2 is partially masked
        assert apstats[1].sum < self.apstats1[1].sum
        assert apstats[1].sum_err < self.apstats1[1].sum_err

        exclude = ('isscalar', 'n_apertures', 'sky_centroid',
                   'sky_centroid_icrs')
        apstats1 = apstats[2]
        for prop in apstats1.properties:
            if (prop in exclude or 'bbox' in prop or 'cutout' in prop
                    or 'moments' in prop):
                continue
            assert np.all(np.isnan(getattr(apstats1, prop)))

        # test that mask=None is the same as mask=np.ma.nomask
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

        # test broadcasting
        local_bkg = (12, 12, 12)
        apstats1 = ApertureStats(data, self.aperture, local_bkg=local_bkg)
        apstats2 = ApertureStats(data, self.aperture, local_bkg=local_bkg[0])
        assert_equal(apstats1.sum, apstats2.sum)

        with pytest.raises(ValueError):
            _ = ApertureStats(data, self.aperture, local_bkg=(10, 20))
        with pytest.raises(ValueError):
            _ = ApertureStats(data, self.aperture[0:2], local_bkg=(10, np.nan))
        with pytest.raises(ValueError):
            _ = ApertureStats(data, self.aperture[0:2],
                              local_bkg=(-np.inf, 10))
        with pytest.raises(ValueError):
            _ = ApertureStats(data, self.aperture[0:2],
                              local_bkg=np.ones((3, 3)))

    def test_no_aperture_overlap(self):
        aperture = CircularAperture(((0, 0), (100, 100), (-100, -100)), r=5)
        apstats = ApertureStats(self.data, aperture)
        assert_equal(apstats._overlap, [True, True, False])

        exclude = ('isscalar', 'n_apertures', 'sky_centroid',
                   'sky_centroid_icrs')
        apstats1 = apstats[2]
        for prop in apstats1.properties:
            if (prop in exclude or 'bbox' in prop or 'cutout' in prop
                    or 'moments' in prop):
                continue
            assert np.all(np.isnan(getattr(apstats1, prop)))

    def test_to_table(self):
        tbl = self.apstats1.to_table()
        assert tbl.colnames == self.apstats1.default_columns
        assert len(tbl) == len(self.apstats1) == 3

        columns = ['id', 'min', 'max', 'mean', 'median', 'std', 'sum']
        tbl = self.apstats1.to_table(columns=columns)
        assert tbl.colnames == columns
        assert len(tbl) == len(self.apstats1) == 3

    def test_slicing(self):
        apstats = self.apstats1
        _ = apstats.to_table()
        apstat0 = apstats[1]
        assert apstat0.n_apertures == 1
        assert apstat0.ids == np.array([2])
        apstat1 = apstats.get_id(2)
        assert apstat1.n_apertures == 1
        assert apstat0.sum_aper_area == apstat1.sum_aper_area

        apstat0 = apstats[0:1]
        assert len(apstat0) == 1

        apstat0 = apstats[0:2]
        assert len(apstat0) == 2

        apstat0 = apstats[0:3]
        assert len(apstat0) == 3

        apstat0 = apstats[1:]
        apstat1 = apstats.get_ids([1, 2])
        assert len(apstat0) == len(apstat1) == 2

        apstat0 = apstats[1:]
        apstat1 = apstats.get_ids([1, 2])
        assert len(apstat0) == len(apstat1) == 2

        apstat0 = apstats[[2, 1, 0]]
        apstat1 = apstats.get_ids([3, 2, 1])
        assert len(apstat0) == len(apstat1) == 3
        assert_equal(apstat0.ids, [3, 2, 1])
        assert_equal(apstat1.ids, [3, 2, 1])

        # test get_ids when ids are not sorted
        apstat0 = apstats[[2, 1, 0]]
        apstat1 = apstat0.get_ids(2)
        assert apstat1.ids == 2

        mask = apstats.id >= 2
        apstat0 = apstats[mask]
        assert len(apstat0) == 2
        assert_equal(apstat0.ids, [2, 3])

        # test iter
        for (i, apstat) in enumerate(apstats):
            assert apstat.isscalar
            assert apstat.id == (i + 1)

        with pytest.raises(TypeError):
            _ = len(apstats[0])

        with pytest.raises(TypeError):
            apstat0 = apstats[0]
            apstat1 = apstat0[0]

        with pytest.raises(TypeError):
            apstat0 = apstats[0]
            apstat1 = apstat0[0]  # can't slice scalar object

        with pytest.raises(ValueError):
            apstat0 = apstats.get_ids([-1, 0])

    def test_scalar_aperture_stats(self):
        apstats = self.apstats1[0]
        assert apstats.n_apertures == 1
        assert apstats.ids == np.array([1])
        tbl = apstats.to_table()
        assert len(tbl) == 1

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            ApertureStats(self.data, 10.0)
        with pytest.raises(TypeError):
            ApertureStats(self.data, self.aperture, sigma_clip=10)
        with pytest.raises(ValueError):
            ApertureStats(self.data, self.aperture, error=10.0)
        with pytest.raises(ValueError):
            ApertureStats(self.data, self.aperture, error=np.ones(3))
        with pytest.raises(ValueError):
            ApertureStats(self.data, self.aperture, error=np.ones((3, 3)))

    def test_repr_str(self):
        assert repr(self.apstats1) == str(self.apstats1)
        assert 'Length: 3' in repr(self.apstats1)

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
        assert apstats.xcentroid == 12.0
        assert apstats.ycentroid == 12.0

    @pytest.mark.parametrize('with_units', (True, False))
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

        assert_allclose(apstats1.xcentroid, apstats2.xcentroid)
        assert_allclose(apstats1.ycentroid, apstats2.ycentroid)
        assert_allclose(apstats1.sum, apstats2.sum)

        if with_units:
            assert apstats1.sum.unit == unit
