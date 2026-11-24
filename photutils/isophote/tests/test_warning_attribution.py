# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the deprecation warnings of the decorated public callables
are attributed to the line of the caller.
"""

import warnings

import astropy.units as u
import numpy as np
import pytest

from photutils.isophote import (Ellipse, EllipseFitter, EllipseGeometry,
                                EllipseSample, build_ellipse_model)
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning


@pytest.fixture(name='data', scope='module')
def fixture_data():
    return make_test_image(nx=128, ny=128, sma=10.0, seed=0)


@pytest.fixture(name='isolist', scope='module')
def fixture_isolist(data):
    ellipse = Ellipse(data)
    return ellipse.fit_image(sma_start=10.0, max_sma=15.0)


def _make_geometry():
    return EllipseGeometry(x0=64.0, y0=64.0, sma=10.0, ellipticity=0.2,
                           orientation=(np.pi / 2) * u.rad)


def _record(func):
    """
    Call ``func`` and return the recorded deprecation warnings.
    """
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        func()
    return [record for record in records
            if issubclass(record.category, PhotutilsDeprecationWarning)]


def _assert_attributed(func, *, n_warnings):
    """
    Assert that ``func`` emits ``n_warnings`` deprecation warnings and
    that each one is attributed to this file.
    """
    records = _record(func)
    assert len(records) == n_warnings
    for record in records:
        assert record.filename == __file__, str(record.message)


# The renamed keywords of each callable, as (old name, value) pairs. The
# values are those of the new names in the base keywords below.
GEOMETRY_KWARGS = {'x0': 64.0, 'y0': 64.0, 'sma': 10.0,
                   'ellipticity': 0.2, 'orientation': 0.5 * u.rad,
                   'step': 0.1,
                   'fix_orientation': False, 'fix_ellipticity': False}
GEOMETRY_RENAMES = {'eps': 'ellipticity', 'pa': 'orientation',
                    'astep': 'step', 'fix_pa': 'fix_orientation',
                    'fix_eps': 'fix_ellipticity'}

SAMPLE_KWARGS = {'x0': 64.0, 'y0': 64.0, 'step': 0.1, 'ellipticity': 0.2,
                 'orientation': 0.5 * u.rad,
                 'integration_mode': 'bilinear'}
SAMPLE_RENAMES = {'astep': 'step',
                  'eps': 'ellipticity', 'position_angle': 'orientation',
                  'integrmode': 'integration_mode'}

# The deprecated sigma-clipping keywords, which are replaced by the
# sigma_clip keyword
LEGACY_CLIP_KEYWORDS = ['sclip', 'n_clip', 'nclip']

FIT_IMAGE_KWARGS = {'sma_start': 10.0, 'min_sma': 8.0, 'max_sma': 12.0,
                    'convergence': 0.05, 'miniters': 10, 'maxiters': 50,
                    'min_valid_fraction': 0.7, 'max_gradient_rel_err': 0.5,
                    'integration_mode': 'bilinear', 'linear_growth': False,
                    'max_iterative_sma': 100.0, 'fix_orientation': False,
                    'fix_ellipticity': False}
FIT_IMAGE_RENAMES = {'sma0': 'sma_start', 'minsma': 'min_sma',
                     'maxsma': 'max_sma', 'conver': 'convergence',
                     'minit': 'miniters', 'maxit': 'maxiters',
                     'fflag': 'min_valid_fraction',
                     'maxgerr': 'max_gradient_rel_err',
                     'integrmode': 'integration_mode',
                     'linear': 'linear_growth',
                     'maxrit': 'max_iterative_sma',
                     'fix_pa': 'fix_orientation',
                     'fix_eps': 'fix_ellipticity'}

FIT_ISOPHOTE_KWARGS = {'convergence': 0.05, 'miniters': 10,
                       'maxiters': 50, 'min_valid_fraction': 0.7,
                       'integration_mode': 'bilinear',
                       'linear_growth': False, 'max_iterative_sma': 100.0}
FIT_ISOPHOTE_RENAMES = {'conver': 'convergence', 'minit': 'miniters',
                        'maxit': 'maxiters', 'fflag': 'min_valid_fraction',
                        'integrmode': 'integration_mode',
                        'linear': 'linear_growth',
                        'maxrit': 'max_iterative_sma'}

FITTER_KWARGS = {'convergence': 0.05, 'miniters': 10, 'maxiters': 50,
                 'min_valid_fraction': 0.7}
FITTER_RENAMES = {'conver': 'convergence', 'minit': 'miniters',
                  'maxit': 'maxiters', 'fflag': 'min_valid_fraction'}


def _renamed(kwargs, old, new):
    """
    Return ``kwargs`` with ``new`` replaced by ``old``.
    """
    kwargs = dict(kwargs)
    kwargs[old] = kwargs.pop(new)
    return kwargs


class TestEllipseGeometry:
    @pytest.mark.parametrize(('old', 'new'), GEOMETRY_RENAMES.items())
    def test_old_keyword(self, old, new):
        kwargs = _renamed(GEOMETRY_KWARGS, old, new)
        _assert_attributed(lambda: EllipseGeometry(**kwargs), n_warnings=1)

    def test_positional(self):
        _assert_attributed(
            lambda: EllipseGeometry(64.0, 64.0, 10.0, 0.2, 0.5 * u.rad,
                                    0.1),
            n_warnings=1)

    def test_positional_and_old_keyword(self):
        _assert_attributed(
            lambda: EllipseGeometry(64.0, 64.0, 10.0, 0.2, 0.5 * u.rad,
                                    0.1, fix_pa=False),
            n_warnings=2)

    def test_orientation_number(self):
        _assert_attributed(
            lambda: EllipseGeometry(64.0, 64.0, 10.0, 0.2, 0.5),
            n_warnings=1)

    def test_orientation_number_and_old_keyword(self):
        _assert_attributed(
            lambda: EllipseGeometry(64.0, 64.0, 10.0, 0.2, 0.5,
                                    fix_pa=False),
            n_warnings=2)

    def test_orientation_number_and_positional(self):
        _assert_attributed(
            lambda: EllipseGeometry(64.0, 64.0, 10.0, 0.2, 0.5, 0.1),
            n_warnings=2)

    def test_orientation_number_positional_and_old_keyword(self):
        _assert_attributed(
            lambda: EllipseGeometry(64.0, 64.0, 10.0, 0.2, 0.5, 0.1,
                                    fix_pa=False),
            n_warnings=3)

    def test_find_center_positional(self, data):
        geometry = _make_geometry()
        _assert_attributed(lambda: geometry.find_center(data, 0.1),
                           n_warnings=1)


class TestEllipseSample:
    @pytest.mark.parametrize(('old', 'new'), SAMPLE_RENAMES.items())
    def test_old_keyword(self, data, old, new):
        kwargs = _renamed(SAMPLE_KWARGS, old, new)
        _assert_attributed(lambda: EllipseSample(data, 10.0, **kwargs),
                           n_warnings=1)

    @pytest.mark.parametrize('old', LEGACY_CLIP_KEYWORDS)
    def test_removed_keyword(self, data, old):
        _assert_attributed(lambda: EllipseSample(data, 10.0, **{old: 1}),
                           n_warnings=1)

    def test_positional(self, data):
        _assert_attributed(lambda: EllipseSample(data, 10.0, 64.0, 64.0),
                           n_warnings=1)

    def test_positional_and_old_keyword(self, data):
        _assert_attributed(
            lambda: EllipseSample(data, 10.0, 64.0, 64.0, eps=0.2),
            n_warnings=2)

    def test_update_old_keyword(self, data):
        sample = EllipseSample(data, 10.0)
        fixed = np.zeros(4, dtype=bool)
        _assert_attributed(lambda: sample.update(fixed_parameters=fixed),
                           n_warnings=1)

    def test_update_positional(self, data):
        sample = EllipseSample(data, 10.0)
        fixed = np.zeros(4, dtype=bool)
        _assert_attributed(lambda: sample.update(fixed), n_warnings=1)


class TestEllipse:
    def test_init_old_keyword(self, data):
        _assert_attributed(lambda: Ellipse(data, threshold=0.1),
                           n_warnings=1)

    def test_init_positional(self, data):
        geometry = _make_geometry()
        _assert_attributed(lambda: Ellipse(data, geometry), n_warnings=1)

    def test_init_positional_and_old_keyword(self, data):
        geometry = _make_geometry()
        _assert_attributed(lambda: Ellipse(data, geometry, threshold=0.1),
                           n_warnings=2)

    @pytest.mark.parametrize(('old', 'new'), FIT_IMAGE_RENAMES.items())
    def test_fit_image_old_keyword(self, data, old, new):
        ellipse = Ellipse(data)
        kwargs = _renamed(FIT_IMAGE_KWARGS, old, new)
        _assert_attributed(lambda: ellipse.fit_image(**kwargs),
                           n_warnings=1)

    @pytest.mark.parametrize('old', LEGACY_CLIP_KEYWORDS)
    def test_fit_image_removed_keyword(self, data, old):
        ellipse = Ellipse(data)
        _assert_attributed(
            lambda: ellipse.fit_image(max_sma=12.0, **{old: 1}),
            n_warnings=1)

    def test_fit_image_positional(self, data):
        ellipse = Ellipse(data)
        _assert_attributed(lambda: ellipse.fit_image(10.0, 8.0, 12.0),
                           n_warnings=1)

    def test_fit_image_positional_and_old_keyword(self, data):
        ellipse = Ellipse(data)
        _assert_attributed(lambda: ellipse.fit_image(10.0, 8.0, maxsma=12.0),
                           n_warnings=2)

    @pytest.mark.parametrize(('old', 'new'), FIT_ISOPHOTE_RENAMES.items())
    def test_fit_isophote_old_keyword(self, data, old, new):
        ellipse = Ellipse(data)
        kwargs = _renamed(FIT_ISOPHOTE_KWARGS, old, new)
        _assert_attributed(lambda: ellipse.fit_isophote(10.0, **kwargs),
                           n_warnings=1)

    @pytest.mark.parametrize('old', ['maxgerr', 'going_inwards',
                                     *LEGACY_CLIP_KEYWORDS])
    def test_fit_isophote_removed_keyword(self, data, old):
        ellipse = Ellipse(data)
        _assert_attributed(lambda: ellipse.fit_isophote(10.0, **{old: 1}),
                           n_warnings=1)

    def test_fit_isophote_positional(self, data):
        ellipse = Ellipse(data)
        _assert_attributed(lambda: ellipse.fit_isophote(10.0, 0.1),
                           n_warnings=1)

    def test_fit_isophote_positional_and_old_keyword(self, data):
        ellipse = Ellipse(data)
        _assert_attributed(lambda: ellipse.fit_isophote(10.0, 0.1, nclip=0),
                           n_warnings=2)


class TestEllipseFitter:
    @pytest.mark.parametrize(('old', 'new'), FITTER_RENAMES.items())
    def test_old_keyword(self, data, old, new):
        kwargs = _renamed(FITTER_KWARGS, old, new)
        sample = EllipseSample(data, 10.0)
        sample.update()
        _assert_attributed(lambda: EllipseFitter(sample).fit(**kwargs),
                           n_warnings=2)

    @pytest.mark.parametrize('old', ['maxgerr', 'going_inwards'])
    def test_removed_keyword(self, data, old):
        sample = EllipseSample(data, 10.0)
        sample.update()
        _assert_attributed(lambda: EllipseFitter(sample).fit(**{old: 1}),
                           n_warnings=2)


class TestBuildEllipseModel:
    def test_positional(self, isolist):
        _assert_attributed(
            lambda: build_ellipse_model((128, 128), isolist, 0.0),
            n_warnings=2)


class TestIsophoteList:
    def test_to_table_positional(self, isolist):
        _assert_attributed(lambda: isolist.to_table(['sma']),
                           n_warnings=1)
