# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for deprecation warnings when optional arguments are passed
positionally.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.datasets.images import _model_shape_from_bbox
from photutils.datasets.load import load_irac_psf
from photutils.datasets.model_params import make_random_models_table
from photutils.datasets.noise import apply_poisson_noise, make_noise_image
from photutils.datasets.wcs import make_gwcs, make_wcs
from photutils.utils._optional_deps import HAS_GWCS


class TestLoadIracPsfPositionalKwargs:
    """
    Test that load_irac_psf warns for positional optional args.
    """

    @pytest.mark.remote_data
    def test_positional_warns(self):
        match = 'load_irac_psf'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            load_irac_psf(1, False)  # noqa: FBT003

    @pytest.mark.remote_data
    def test_keyword_no_warning(self):
        load_irac_psf(1, show_progress=False)


class TestMakeRandomModelsTablePositionalKwargs:
    """
    Test that make_random_models_table warns for positional optional
    args.
    """

    def test_positional_warns(self):
        match = 'make_random_models_table'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            make_random_models_table(5, {'x_mean': [0, 100]}, 0)

    def test_keyword_no_warning(self):
        make_random_models_table(5, {'x_mean': [0, 100]}, seed=0)


class TestApplyPoissonNoisePositionalKwargs:
    """
    Test that apply_poisson_noise warns for positional optional args.
    """

    def test_positional_warns(self):
        data = np.ones((10, 10))
        match = 'apply_poisson_noise'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            apply_poisson_noise(data, 0)

    def test_keyword_no_warning(self):
        data = np.ones((10, 10))
        apply_poisson_noise(data, seed=0)


class TestMakeNoiseImagePositionalKwargs:
    """
    Test that make_noise_image warns for positional optional args.
    """

    def test_positional_warns(self):
        match = 'make_noise_image'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            make_noise_image((10, 10), 'gaussian', mean=0.0, stddev=2.0)

    def test_keyword_no_warning(self):
        make_noise_image((10, 10), distribution='gaussian', mean=0.0,
                         stddev=2.0)


class TestMakeWcsPositionalKwargs:
    """
    Test that make_wcs warns for positional optional args.
    """

    def test_positional_warns(self):
        match = 'make_wcs'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            make_wcs((100, 100), False)  # noqa: FBT003

    def test_keyword_no_warning(self):
        make_wcs((100, 100), galactic=False)


class TestMakeGwcsPositionalKwargs:
    """
    Test that make_gwcs warns for positional optional args.
    """

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_positional_warns(self):
        match = 'make_gwcs'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            make_gwcs((100, 100), False)  # noqa: FBT003

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_keyword_no_warning(self):
        make_gwcs((100, 100), galactic=False)


class TestModelShapeFromBboxPositionalKwargs:
    """
    Test that _model_shape_from_bbox warns for positional optional args.
    """

    def test_positional_warns(self):
        model = Gaussian2D()
        match = '_model_shape_from_bbox'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            _model_shape_from_bbox(model, 5.0)

    def test_keyword_no_warning(self):
        model = Gaussian2D()
        _model_shape_from_bbox(model, bbox_factor=5.0)
