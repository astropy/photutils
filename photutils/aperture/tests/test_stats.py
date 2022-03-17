# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the stats module.
"""

from astropy.stats import SigmaClip
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pytest

from ...datasets import make_100gaussians_image, make_wcs
from ..bounding_box import BoundingBox
from ..circle import CircularAperture, CircularAnnulus
from ..mask import ApertureMask
from ..stats import ApertureStats


class TestApertureStats:
    data = make_100gaussians_image()
    error = np.sqrt(data)
    wcs = make_wcs(data.shape)
    positions = [(145.1, 168.3), (84.5, 224.1), (48.3, 200.3)]
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
    def test_properties(self, with_units):
        if with_units:
            apstats1 = self.apstats1_units.copy()
            apstats2 = self.apstats1_units.copy()
        else:
            apstats1 = self.apstats1.copy()
            apstats2 = self.apstats1.copy()

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

    def test_minimal_inputs(self):
        apstats = ApertureStats(self.data, self.aperture)
        props = ('sky_centroid', 'sky_centroid_icrs', 'error_sumcutout')
        for prop in props:
            assert set(getattr(apstats, prop)) == set([None])
        assert np.all(np.isnan(apstats.sum_err))

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

    def test_table(self):
        tbl = self.apstats1.to_table()
        assert tbl.colnames == self.apstats1.default_columns
        assert len(tbl) == len(self.apstats1) == 3

    def test_slicing(self):
        apstats = self.apstats1
        _ = apstats.to_table()
        apstat0 = apstats[1]
        assert apstat0.n_apertures == 1
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

        apstat0 = apstats[[2, 1, 0]]
        apstat1 = apstats.get_ids([3, 2, 1])
        assert len(apstat0) == len(apstat1) == 3
        assert_equal(apstat0.ids, [3, 2, 1])
        assert_equal(apstat1.ids, [3, 2, 1])

        mask = apstats.id >= 2
        apstat0 = apstats[mask]
        assert len(apstat0) == 2
        assert_equal(apstat0.ids, [2, 3])

        with pytest.raises(TypeError):
            apstat0 = apstats[0]
            apstat1 = apstat0[0]

        with pytest.raises(TypeError):
            apstat0 = apstats[0]
            apstat1 = apstat0[0]  # can't slice scalar object

        with pytest.raises(ValueError):
            apstat0 = apstats.get_ids([-1, 0])
