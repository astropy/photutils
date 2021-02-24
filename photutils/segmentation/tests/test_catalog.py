# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the catalog module.
"""

from copy import deepcopy

import astropy.units as u
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from numpy.testing import assert_allclose, assert_equal
import numpy as np
import pytest

from ..catalog import SourceCatalog
from ..core import SegmentationImage
from ..detect import detect_sources
from ...aperture import CircularAperture, EllipticalAperture
from ...datasets import make_gwcs, make_wcs, make_noise_image

try:
    import scipy  # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import gwcs  # noqa
    HAS_GWCS = True
except ImportError:
    HAS_GWCS = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestSourceCatalog:
    def setup_class(self):
        xcen = 51.
        ycen = 52.7
        major_sigma = 8.
        minor_sigma = 3.
        theta = np.pi / 6.
        g1 = Gaussian2D(111., xcen, ycen, major_sigma, minor_sigma,
                        theta=theta)
        g2 = Gaussian2D(50, 20, 80, 5.1, 4.5)
        g3 = Gaussian2D(70, 75, 18, 9.2, 4.5)
        g4 = Gaussian2D(111., 11.1, 12.2, major_sigma, minor_sigma,
                        theta=theta)
        g5 = Gaussian2D(81., 61, 42.7, major_sigma, minor_sigma, theta=theta)
        g6 = Gaussian2D(107., 75, 61, major_sigma, minor_sigma, theta=-theta)
        g7 = Gaussian2D(107., 90, 90, 4, 2, theta=-theta)

        yy, xx = np.mgrid[0:101, 0:101]
        self.data = (g1(xx, yy) + g2(xx, yy) + g3(xx, yy) + g4(xx, yy)
                     + g5(xx, yy) + g6(xx, yy) + g7(xx, yy))
        threshold = 27.
        self.segm = detect_sources(self.data, threshold, npixels=5)
        self.error = make_noise_image(self.data.shape, mean=0, stddev=2.,
                                      seed=123)
        self.background = np.ones(self.data.shape) * 5.1
        self.mask = np.zeros(self.data.shape, dtype=bool)
        self.mask[0:30, 0:30] = True

        self.wcs = make_wcs(self.data.shape)
        self.cat = SourceCatalog(self.data, self.segm, error=self.error,
                                 background=self.background, mask=self.mask,
                                 wcs=self.wcs, localbkg_width=24)
        unit = u.nJy
        self.cat_units = SourceCatalog(self.data << unit, self.segm,
                                       error=self.error << unit,
                                       background=self.background << unit,
                                       mask=self.mask, wcs=self.wcs,
                                       localbkg_width=24)

    @pytest.mark.parametrize('with_units', (True, False))
    def test_catalog(self, with_units):
        props1 = ('background_centroid', 'background_mean', 'background_sum',
                  'bbox', 'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'cxx',
                  'cxy', 'cyy', 'ellipticity', 'elongation',
                  'equivalent_radius', 'gini', 'kron_radius', 'maxval_xindex',
                  'maxval_yindex', 'minval_xindex', 'minval_yindex',
                  'perimeter', 'sky_bbox_ll', 'sky_bbox_lr', 'sky_bbox_ul',
                  'sky_bbox_ur', 'sky_centroid_icrs')

        props2 = ('centroid', 'covariance', 'covariance_eigvals',
                  'cutout_centroid', 'cutout_maxval_index',
                  'cutout_minval_index', 'inertia_tensor', 'maxval_index',
                  'minval_index', 'moments', 'moments_central', 'background',
                  'background_ma', 'convdata', 'convdata_ma', 'data',
                  'data_ma', 'error', 'error_ma', 'segment', 'segment_ma')

        props = tuple(self.cat.default_columns) + props1 + props2

        if with_units:
            cat1 = deepcopy(self.cat_units)
            cat2 = deepcopy(self.cat_units)
        else:
            cat1 = deepcopy(self.cat)
            cat2 = deepcopy(self.cat)

        idx = 1

        # evaluate (cache) catalog properties before slice
        obj = cat1[idx]
        for prop in props:
            assert_equal(getattr(cat1, prop)[idx], getattr(obj, prop))

        # slice catalog before evaluating catalog properties
        obj = cat2[idx]
        for prop in props:
            assert_equal(getattr(obj, prop), getattr(cat2, prop)[idx])

    def test_minimal_catalog(self):
        cat = SourceCatalog(self.data, self.segm)
        obj = cat[4]
        props = ('background', 'background_ma', 'error', 'error_ma')
        for prop in props:
            assert getattr(obj, prop) is None

        props = ('background_mean', 'background_sum', 'background_centroid',
                 'segment_fluxerr', 'kron_fluxerr')
        for prop in props:
            assert np.isnan(getattr(obj, prop))

        assert obj.local_background_aperture is None
        assert obj.local_background == 0.
