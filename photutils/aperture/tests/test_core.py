# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_allclose

from photutils.aperture import (Aperture, CircularAperture, EllipticalAperture,
                                SkyCircularAperture)
from photutils.aperture.core import (_aperture_metadata,
                                     _update_method_subpixels_docstring)

POSITIONS = [(5, 5), (10, 10), (15, 15)]
SCALAR_POS = (5, 5)


class MinimalAperture(Aperture):
    """
    Minimal concrete Aperture subclass that is neither PixelAperture nor
    SkyAperture.

    Used to exercise bare-Aperture code paths.
    """

    _params = ('positions',)

    @property
    def positions(self):
        """
        Return a fixed single position.
        """
        return np.array([[5.0, 5.0]])


class RaisesOnCompare:
    """
    Helper object that raises TypeError on any != comparison, used to
    exercise the except-TypeError branch in Aperture.__eq__.
    """

    def __ne__(self, other):
        """
        Raise TypeError unconditionally.
        """
        msg = 'incompatible types'
        raise TypeError(msg)


class TestAperture:
    """
    Tests for branches of the Aperture base class not covered elsewhere.
    """

    def test_positions_str_raises_for_unknown_type(self):
        """
        Test that _positions_str raises TypeError when the aperture is
        not a PixelAperture or SkyAperture subclass.
        """
        aper = MinimalAperture()
        match = 'Aperture must be a subclass of PixelAperture or SkyAperture'
        with pytest.raises(TypeError, match=match):
            aper._positions_str()

    def test_eq_different_class(self):
        """
        Test that __eq__ returns False when compared to a different
        class (exercises the isinstance early-return branch).
        """
        aper1 = CircularAperture(SCALAR_POS, r=3)
        aper2 = EllipticalAperture(SCALAR_POS, a=3, b=2, theta=0)
        assert aper1 != aper2

    def test_eq_different_params(self):
        """
        Test that __eq__ returns False when the two instances have
        different _params tuples (exercises the params-mismatch branch).
        """
        aper1 = CircularAperture(SCALAR_POS, r=3)
        aper2 = CircularAperture(SCALAR_POS, r=3)
        # Inject an extended _params tuple at the instance level so that
        # isinstance passes (both CircularAperture instances, same type so
        # Python does not invoke subclass reflection) while the params
        # check diverges, hitting the mismatch return at line 104.
        aper2.__dict__['_params'] = (*CircularAperture._params, 'extra')
        assert aper1 != aper2

    def test_eq_comparison_type_error(self):
        """
        Test that __eq__ returns False (rather than propagating the
        exception) when the position comparison raises TypeError.
        """
        aper1 = CircularAperture(SCALAR_POS, r=3)
        aper2 = CircularAperture(SCALAR_POS, r=3)
        # Bypass the descriptor and inject a position object that raises
        # TypeError on != comparison, mimicking incompatible SkyCoords.
        aper1.__dict__['positions'] = RaisesOnCompare()
        aper2.__dict__['positions'] = RaisesOnCompare()
        assert aper1 != aper2


class TestPixelAperture:
    """
    Tests for branches of the PixelAperture class not covered elsewhere.
    """

    def test_to_mask_invalid_method(self):
        """
        Test that to_mask raises ValueError for an unrecognised
        method string (exercises the invalid-method branch in
        _translate_mask_method).
        """
        aper = CircularAperture(SCALAR_POS, r=3)
        match = 'Invalid mask method'
        with pytest.raises(ValueError, match=match):
            aper.to_mask(method='invalid')

    def test_bbox_multi_position(self):
        """
        Test that the bbox property returns a list for a multi-position
        aperture (exercises the non-scalar branch).
        """
        aper = CircularAperture(POSITIONS, r=3)
        bbox = aper.bbox
        assert isinstance(bbox, list)
        assert len(bbox) == len(POSITIONS)


class TestPixelAperturePhotometry:
    """
    Tests for error-handling branches of PixelAperture.photometry.
    """

    def setup_method(self):
        """
        Set up a simple scalar aperture and matching data array.
        """
        self.aper = CircularAperture(SCALAR_POS, r=3)
        self.data = np.ones((20, 20))

    def test_photometry_1d_data_error(self):
        """
        Test that photometry raises ValueError when data is not a
        2D array.
        """
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            self.aper.photometry(np.ones(20))

    def test_photometry_error_shape_mismatch(self):
        """
        Test that photometry raises ValueError when the error array
        does not match the data shape.
        """
        match = 'error and data must have the same shape'
        with pytest.raises(ValueError, match=match):
            self.aper.photometry(self.data, error=np.ones((5, 5)))

    def test_photometry_unit_mismatch_error(self):
        """
        Test that photometry raises ValueError when data and error
        have different units.
        """
        import astropy.units as u

        match = 'they both must have the same units'
        with pytest.raises(ValueError, match=match):
            self.aper.photometry(
                self.data * u.Jy,
                error=self.data * u.ct,
            )

    def test_photometry_basic(self):
        """
        Test that photometry returns the expected aperture sum for a
        uniform data array with no error input, and that the returned
        error array has the same length as the flux array, filled with
        NaN.
        """
        sums, errs = self.aper.photometry(self.data)
        assert_allclose(sums[0], np.pi * 9, rtol=1e-3)
        assert len(errs) == len(sums)
        assert_allclose(errs, np.nan)

    def test_photometry_result_backward_compatible(self):
        """
        Test that the photometry result unpacks as a 2-tuple and
        supports len() and integer indexing like the legacy tuple.
        """
        result = self.aper.photometry(self.data)
        assert len(result) == 2

        sums, errs = result
        assert_allclose(sums, result.aperture_sum)
        assert_allclose(errs, result.aperture_sum_err, equal_nan=True)
        assert_allclose(result[0], result.aperture_sum)
        assert_allclose(result[1], result.aperture_sum_err, equal_nan=True)

    def test_photometry_area_matches_area_overlap(self):
        """
        Test that the result area matches area_overlap for a simple
        non-masked, fully-overlapping aperture.
        """
        result = self.aper.photometry(self.data)
        expected = self.aper.area_overlap(self.data)
        assert result.area.unit == u.pix**2
        assert_allclose(result.area.value, expected)

    def test_photometry_area_no_overlap_nan(self):
        """
        Test that the result area is NaN for an aperture with no overlap
        with the data.
        """
        aper = CircularAperture((-50, -50), r=3)
        result = aper.photometry(self.data)
        assert np.isnan(result.area.value).all()

    def test_photometry_area_units(self):
        """
        Test that the result area always has units of pix**2 regardless
        of whether the data/error are Quantity or plain arrays.
        """
        result = self.aper.photometry(self.data)
        assert result.area.unit == u.pix**2

        result_q = self.aper.photometry(self.data * u.Jy,
                                        error=self.data * u.Jy)
        assert result_q.area.unit == u.pix**2
        assert_allclose(result.area.value, result_q.area.value,
                        equal_nan=True)

    def test_photometry_area_batch_vs_mask_path(self):
        """
        Test that the batch and mask code paths produce identical area
        values.
        """
        aper = CircularAperture(POSITIONS, r=5.5)
        mask = np.zeros(self.data.shape, dtype=bool)
        mask[5, 5] = True

        batch = aper._do_batch_photometry(self.data, error=None, mask=mask,
                                          method='exact', subpixels=5)
        assert batch is not None
        legacy = aper._do_mask_photometry(self.data, error=None, mask=mask,
                                          method='exact', subpixels=5)
        # areas are the third element of each result
        assert_allclose(batch[2], legacy[2], rtol=1e-12, equal_nan=True)

    def test_do_photometry_deprecated(self):
        """
        Test that the deprecated do_photometry method emits a
        deprecation warning and returns the same results as photometry.
        """
        error = np.full(self.data.shape, 0.1)
        expected = self.aper.photometry(self.data, error=error)
        match = r'Use photometry instead'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            result = self.aper.do_photometry(self.data, error=error)
        assert_allclose(result.aperture_sum, expected.aperture_sum)
        assert_allclose(result.aperture_sum_err, expected.aperture_sum_err)
        assert_allclose(result.area.value, expected.area.value)


class TestApertureReprStr:
    """
    Tests for __repr__ and __str__ of various aperture types.
    """

    def test_repr_scalar(self):
        """
        Test __repr__ for a scalar CircularAperture.
        """
        aper = CircularAperture(SCALAR_POS, r=3)
        result = repr(aper)
        assert 'CircularAperture' in result
        assert 'r=3.0' in result

    def test_repr_multi(self):
        """
        Test __repr__ for a multi-position CircularAperture.
        """
        aper = CircularAperture(POSITIONS, r=3)
        result = repr(aper)
        assert 'CircularAperture' in result

    def test_str_scalar(self):
        """
        Test __str__ for a scalar CircularAperture.
        """
        aper = CircularAperture(SCALAR_POS, r=3)
        result = str(aper)
        assert 'Aperture: CircularAperture' in result
        assert 'r: 3.0' in result

    def test_str_multi(self):
        """
        Test __str__ for a multi-position CircularAperture.
        """
        aper = CircularAperture(POSITIONS, r=3)
        result = str(aper)
        assert 'Aperture: CircularAperture' in result

    def test_repr_sky(self):
        """
        Test __repr__ for a SkyCircularAperture.
        """
        pos = SkyCoord(ra=10, dec=20, unit='deg')
        aper = SkyCircularAperture(pos, r=1.0 * u.arcsec)
        result = repr(aper)
        assert 'SkyCircularAperture' in result


class TestApertureIteration:
    """
    Tests for __iter__ and __len__ of Aperture objects.
    """

    def test_iter(self):
        """
        Test that iterating over a multi-position aperture yields
        scalar apertures.
        """
        aper = CircularAperture(POSITIONS, r=3)
        items = list(aper)
        assert len(items) == len(POSITIONS)
        for item in items:
            assert item.isscalar

    def test_copy(self):
        """
        Test that copy creates a deep copy with independent data.
        """
        aper = CircularAperture(POSITIONS, r=3)
        aper_copy = aper.copy()
        assert aper == aper_copy
        aper_copy.r = 5.0
        assert aper != aper_copy
        assert aper.r == 3.0

    def test_copy_sky(self):
        """
        Test that copy works for SkyAperture objects.
        """
        pos = SkyCoord(ra=[10, 20], dec=[30, 40], unit='deg')
        aper = SkyCircularAperture(pos, r=1.0 * u.arcsec)
        aper_copy = aper.copy()
        assert aper == aper_copy


class TestApertureMetadata:
    """
    Tests for the _aperture_metadata helper function.
    """

    def test_metadata_keys(self):
        """
        Test that _aperture_metadata returns the expected keys.
        """
        aper = CircularAperture(SCALAR_POS, r=3)
        meta = _aperture_metadata(aper)
        assert 'aperture' in meta
        assert meta['aperture'] == 'CircularAperture'
        assert 'aperture_r' in meta
        assert meta['aperture_r'] == 3.0

    def test_metadata_with_index(self):
        """
        Test that _aperture_metadata uses the index in keys.
        """
        aper = CircularAperture(SCALAR_POS, r=3)
        meta = _aperture_metadata(aper, index='_0')
        assert 'aperture_0' in meta
        assert meta['aperture_0'] == 'CircularAperture'
        assert 'aperture_0_r' in meta

    def test_metadata_class_name_not_repeated(self):
        """
        Test that aperture class name key is set exactly once.
        """
        aper = EllipticalAperture(SCALAR_POS, a=5, b=3, theta=0)
        meta = _aperture_metadata(aper)
        assert meta['aperture'] == 'EllipticalAperture'
        assert 'aperture_a' in meta
        assert 'aperture_b' in meta
        assert 'aperture_theta' in meta


class TestMethodSubpixelsDocstring:
    """
    Tests for the _update_method_subpixels_docstring decorator.
    """

    def test_insert_descriptions(self):
        """
        Test that the placeholder is replaced by the method and
        subpixels descriptions.
        """
        @_update_method_subpixels_docstring
        def func():
            """
            Summary.

            Parameters
            ----------
            <method_subpixels_descriptions>
            """

        doc = func.__doc__
        assert '<method_subpixels_descriptions>' not in doc
        assert "method : {'exact', 'center', 'subpixel'}, optional" in doc
        assert 'subpixels : int, optional' in doc

    def test_indentation_preserved(self):
        """
        Test that the inserted text is indented to match the
        placeholder.

        The docstring is assigned manually to avoid the compile-time
        docstring dedenting introduced in Python 3.13.
        """
        def func():
            pass

        func.__doc__ = ('Summary.\n\n'
                        '        Parameters\n'
                        '        ----------\n'
                        '        <method_subpixels_descriptions>\n')
        _update_method_subpixels_docstring(func)

        lines = func.__doc__.splitlines()
        method_line = next(line for line in lines
                           if line.lstrip().startswith('method :'))
        assert method_line.startswith('        method :')
        subpixels_line = next(line for line in lines
                              if line.lstrip().startswith('subpixels :'))
        assert subpixels_line.startswith('        subpixels :')

    def test_no_placeholder_noop(self):
        """
        Test that the decorator is a no-op if no placeholder is present.
        """
        @_update_method_subpixels_docstring
        def func():
            """Summary with no placeholder."""

        assert func.__doc__ == 'Summary with no placeholder.'

    def test_none_docstring_noop(self):
        """
        Test that the decorator is a no-op if no docstring exists.
        """
        @_update_method_subpixels_docstring
        def func():
            pass

        assert func.__doc__ is None

    def test_class_docstring(self):
        """
        Test that the decorator also updates a class docstring.
        """
        @_update_method_subpixels_docstring
        class Example:
            """
            Summary.

            Parameters
            ----------
            <method_subpixels_descriptions>
            """

        assert '<method_subpixels_descriptions>' not in Example.__doc__
        assert 'subpixels : int, optional' in Example.__doc__

    def test_method_bullets_placeholder(self):
        """
        Test that the method-bullets placeholder is replaced by only the
        method bullet list, without the method header or the subpixels
        description.
        """
        @_update_method_subpixels_docstring
        def func():
            """
            Summary.

            Parameters
            ----------
            <method_bullets>
            """

        doc = func.__doc__
        assert '<method_bullets>' not in doc
        assert "* ``'exact'`` (default):" in doc
        assert "* ``'center'``:" in doc
        assert "* ``'subpixel'``:" in doc
        assert "method : {'exact', 'center', 'subpixel'}, optional" not in doc
        assert 'subpixels : int, optional' not in doc

    def test_subpixels_description_placeholder(self):
        """
        Test that the subpixels-description placeholder is replaced by
        only the subpixels parameter description.
        """
        @_update_method_subpixels_docstring
        def func():
            """
            Summary.

            Parameters
            ----------
            <subpixels_description>
            """

        doc = func.__doc__
        assert '<subpixels_description>' not in doc
        assert 'subpixels : int, optional' in doc
        assert "method : {'exact', 'center', 'subpixel'}, optional" not in doc
        assert "* ``'exact'`` (default):" not in doc
