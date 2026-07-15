# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the polygon aperture module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from photutils.aperture.polygon import (PolygonAperture, SkyPolygonAperture,
                                        _polygon_is_simple,
                                        _segments_intersect,
                                        _signed_polygon_area,
                                        _validate_simple_polygon,
                                        _vertices_centroid)
from photutils.aperture.rectangle import RectangularAperture
from photutils.aperture.tests.test_aperture_common import BaseTestAperture
from photutils.utils._optional_deps import HAS_MATPLOTLIB

POSITIONS = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (70.0, 80.0)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec
SQUARE_OFFSETS = np.array([[-1.0, -1.0], [1.0, -1.0],
                           [1.0, 1.0], [-1.0, 1.0]])
TRIANGLE_OFFSETS = np.array([[0.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def _make_wcs():
    """
    Tangent-plane WCS at (180, 0).
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.5, 50.5]
    wcs.wcs.cdelt = [-0.01, 0.01]
    wcs.wcs.crval = [180.0, 0.0]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    return wcs


def test_signed_polygon_area_ccw_positive():
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert_allclose(_signed_polygon_area(verts), 1.0)


def test_signed_polygon_area_cw_negative():
    verts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    assert_allclose(_signed_polygon_area(verts), -1.0)


def test_vertices_centroid_square():
    verts = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    assert_allclose(_vertices_centroid(verts), [2.0, 2.0])


def test_validate_simple_polygon_reorders_clockwise():
    cw = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    out = _validate_simple_polygon(cw)
    # Expect CCW order
    assert _signed_polygon_area(out) > 0


def test_validate_simple_polygon_rejects_wrong_shape():
    with pytest.raises(ValueError, match='must have shape'):
        _validate_simple_polygon(np.array([1.0, 2.0, 3.0]))


def test_validate_simple_polygon_rejects_too_few():
    with pytest.raises(ValueError, match='at least 3 vertices'):
        _validate_simple_polygon(np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_validate_simple_polygon_rejects_nonfinite():
    bad = np.array([[0.0, 0.0], [np.nan, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match='non-finite'):
        _validate_simple_polygon(bad)


def test_validate_simple_polygon_rejects_collinear():
    bad = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match='degenerate'):
        _validate_simple_polygon(bad)


def test_validate_simple_polygon_accepts_nonconvex():
    """
    Test that a non-convex (L-shaped) polygon is accepted.
    """
    arrow = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0],
                      [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]])
    out = _validate_simple_polygon(arrow)
    assert out.shape == arrow.shape


def _pentagram():
    """
    A five-pointed (self-intersecting) star drawn by connecting every
    other vertex of a regular pentagon.

    Its edges self-intersect, but its signed area is non-zero so it is
    not caught by the degeneracy check.
    """
    ang = np.pi / 2 + np.arange(5) * 2 * np.pi * 2 / 5
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _concave_star():
    """
    A ten-pointed star outline (alternating inner/outer radii). It is
    concave but simple (non-self-intersecting).
    """
    ang = np.pi / 2 + np.arange(10) * np.pi / 5
    radius = np.where(np.arange(10) % 2 == 0, 5.0, 2.0)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def test_polygon_is_simple_true_for_square():
    assert _polygon_is_simple(SQUARE_OFFSETS)


def test_polygon_is_simple_true_for_concave_star():
    assert _polygon_is_simple(_concave_star())


def test_polygon_is_simple_false_for_pentagram():
    star = _pentagram()
    # Sanity check: the pentagram has non-zero area, so it is not caught
    # by the degeneracy check and must be rejected by the simplicity
    # check.
    assert _signed_polygon_area(star) != 0.0
    assert not _polygon_is_simple(star)


def test_validate_simple_polygon_rejects_self_intersecting():
    with pytest.raises(ValueError, match='self-intersect'):
        _validate_simple_polygon(_pentagram())


def test_pixel_polygon_rejects_self_intersecting():
    with pytest.raises(ValueError, match='self-intersect'):
        PolygonAperture((0.0, 0.0), _pentagram())


def test_sky_polygon_rejects_self_intersecting():
    pos = SkyCoord(ra=180.0, dec=0.0, unit='deg')
    with pytest.raises(ValueError, match='self-intersect'):
        SkyPolygonAperture(pos, _pentagram() * u.arcsec)


def test_pixel_polygon_accepts_concave_star():
    aper = PolygonAperture((0.0, 0.0), _concave_star())
    assert aper.vertex_offsets.shape == (10, 2)


@pytest.mark.parametrize('n_spikes', [5, 6, 7, 8])
def test_pixel_polygon_from_star(n_spikes):
    """
    Test that ``_from_star`` constructs a star-shaped polygon with the
    correct area and that it raises for invalid parameters.
    """
    xypos = (5, 5)

    match = 'must be at least 2'
    with pytest.raises(ValueError, match=match):
        PolygonAperture._from_star(xypos, 1, 5.0, 2.0)

    match = 'must be a finite positive number'
    with pytest.raises(ValueError, match=match):
        PolygonAperture._from_star(xypos, 5, -2.0, 2.0)
    with pytest.raises(ValueError, match=match):
        PolygonAperture._from_star(xypos, 5, 5.0, -2.0)

    match = 'inner_radius must be less than outer_radius'
    with pytest.raises(ValueError, match=match):
        PolygonAperture._from_star(xypos, 5, 5.0, 6.0)

    match = 'optimal_shape and collinear_edges cannot both be True'
    with pytest.raises(ValueError, match=match):
        PolygonAperture._from_star(xypos, 5, 5.0, 2.0, optimal_shape=True,
                                   collinear_edges=True)

    match = 'collinear_edges requires n_spikes'
    with pytest.raises(ValueError, match=match):
        PolygonAperture._from_star(xypos, 4, 3.0, 2.0, collinear_edges=True)

    match = 'inner_radius must be provided'
    with pytest.raises(ValueError, match=match):
        PolygonAperture._from_star(xypos, 5, 3.0)

    data = np.ones((11, 11))
    outer_radius = 5.0
    inner_radius = 2.0
    aper = PolygonAperture._from_star(xypos, n_spikes, outer_radius,
                                      inner_radius)
    flux, _ = aper.photometry(data, method='exact')
    assert_allclose(flux, aper.area)

    aper1 = PolygonAperture._from_star(xypos, n_spikes, outer_radius,
                                       inner_radius, theta=10 * u.deg)
    flux, _ = aper1.photometry(data, method='exact')
    assert_allclose(flux, aper1.area)

    aper2 = PolygonAperture._from_star(xypos, n_spikes, outer_radius,
                                       optimal_shape=True)
    flux, _ = aper2.photometry(data, method='exact')
    assert_allclose(flux, aper2.area)

    aper3 = PolygonAperture._from_star(xypos, n_spikes, outer_radius,
                                       collinear_edges=True)
    flux, _ = aper3.photometry(data, method='exact')
    assert_allclose(flux, aper3.area)


@pytest.mark.parametrize(
    ('p1', 'p2', 'p3', 'p4', 'expected'),
    [
        # Proper crossing (segments straddle each other).
        ((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0), True),
        # Disjoint, non-collinear segments.
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), False),
        # p1 lies on segment p3-p4 (d1 == 0).
        ((1.0, 0.0), (1.0, 1.0), (0.0, 0.0), (2.0, 0.0), True),
        # p2 lies on segment p3-p4 (d2 == 0).
        ((1.0, 1.0), (1.0, 0.0), (0.0, 0.0), (2.0, 0.0), True),
        # p3 lies on segment p1-p2 (d3 == 0).
        ((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (1.0, 1.0), True),
        # p4 lies on segment p1-p2 (d4 == 0).
        ((0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, 0.0), True),
        # Collinear but non-overlapping segments.
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), False),
    ])
def test_segments_intersect(p1, p2, p3, p4, expected):
    assert _segments_intersect(p1, p2, p3, p4) is expected


class TestPolygonAperture(BaseTestAperture):
    aperture = PolygonAperture(POSITIONS, SQUARE_OFFSETS)

    def test_index(self):
        aper = self.aperture[2]
        assert isinstance(aper, PolygonAperture)
        assert aper.isscalar
        assert_allclose(aper.positions, self.aperture.positions[2])
        assert_allclose(aper.vertex_offsets, self.aperture.vertex_offsets)

    def test_slice(self):
        aper = self.aperture[0:2]
        assert isinstance(aper, PolygonAperture)
        assert len(aper) == 2
        assert_allclose(aper.positions, self.aperture.positions[0:2])
        assert_allclose(aper.vertex_offsets, self.aperture.vertex_offsets)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.vertex_offsets = SQUARE_OFFSETS * 2
        assert aper != self.aperture

    def test_repr(self):
        assert 'PolygonAperture' in repr(self.aperture)
        assert 'PolygonAperture' in str(self.aperture)

    def test_perimeter(self):
        # Unit square (side 2) has perimeter 8.
        assert_allclose(self.aperture.perimeter, 8.0)

    def test_perimeter_triangle(self):
        aper = PolygonAperture((0.0, 0.0), TRIANGLE_OFFSETS)
        expected = 2.0 * np.sqrt(5.0) + 2.0
        assert_allclose(aper.perimeter, expected, rtol=1e-12)

    def test_perimeter_regular_polygon(self):
        # Regular hexagon: perimeter = n * 2 * r * sin(pi / n).
        aper = PolygonAperture.from_regular_polygon((0.0, 0.0), 6, 5.0)
        expected = 6.0 * 2.0 * 5.0 * np.sin(np.pi / 6)
        assert_allclose(aper.perimeter, expected, rtol=1e-12)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self.aperture.plot(ax=ax)
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_returns_patches(self):
        from matplotlib.patches import Patch

        patches = self.aperture._to_patch()
        assert isinstance(patches, list)
        for patch in patches:
            assert isinstance(patch, Patch)


def test_construct_scalar():
    aper = PolygonAperture((10.0, 20.0), SQUARE_OFFSETS)
    assert aper.isscalar
    assert aper.shape == ()
    assert aper.vertex_offsets.shape == (4, 2)
    assert_allclose(aper.area, 4.0)
    assert aper._xy_extents == (1.0, 1.0)


def test_construct_array_positions():
    aper = PolygonAperture(POSITIONS, SQUARE_OFFSETS)
    assert not aper.isscalar
    assert aper.shape == (4,)
    assert len(aper) == 4
    assert aper.vertices.shape == (4, 4, 2)
    expected_first = np.asarray(POSITIONS[0]) + SQUARE_OFFSETS
    assert_allclose(aper.vertices[0], expected_first)


def test_vertices_scalar():
    aper = PolygonAperture((10.0, 20.0), SQUARE_OFFSETS)
    assert aper.vertices.shape == (4, 2)
    assert_allclose(aper.vertices, np.array([10.0, 20.0]) + SQUARE_OFFSETS)


def test_vertex_offsets_normalized_to_ccw():
    cw = SQUARE_OFFSETS[::-1].copy()
    aper = PolygonAperture((0, 0), cw)
    assert _signed_polygon_area(aper.vertex_offsets) > 0


def test_invalid_vertex_offsets_quantity_raises():
    match = 'must not be a Quantity'
    with pytest.raises(TypeError, match=match):
        PolygonAperture((0, 0), SQUARE_OFFSETS * u.pix)


def test_invalid_vertex_offsets_nonfinite_raises():
    bad = np.array([[0.0, 0.0], [np.nan, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match='non-finite'):
        PolygonAperture((0, 0), bad)


def test_polygon_aperture_accepts_nonconvex():
    """
    The aperture supports non-convex (L-shaped) polygons.
    """
    l_shape = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0],
                        [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]])
    aper = PolygonAperture((0.0, 0.0), l_shape)
    assert aper.vertex_offsets.shape == (6, 2)
    assert_allclose(aper.area, 12.0)
    mask = aper.to_mask(method='exact')
    assert_allclose(mask.data.sum(), 12.0, atol=1e-12)


def test_invalid_vertex_offsets_too_few_raises():
    match = 'at least 3 vertices'
    with pytest.raises(ValueError, match=match):
        PolygonAperture((0, 0), [[0.0, 0.0], [1.0, 1.0]])


def test_invalid_vertex_offsets_unparseable():
    match = 'must be convertible to a numeric'
    with pytest.raises(TypeError, match=match):
        PolygonAperture((0, 0), [['a', 'b'], ['c', 'd'], ['e', 'f']])


def test_from_vertices_round_trip():
    verts = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]])
    aper = PolygonAperture.from_vertices(verts)
    assert_allclose(aper.vertices, verts)


def test_from_vertices_invalid_shape():
    match = 'vertices must have shape'
    with pytest.raises(ValueError, match=match):
        PolygonAperture.from_vertices(np.array([1.0, 2.0]))


def test_from_vertices_rejects_degenerate():
    """
    Test that ``from_vertices`` rejects collinear vertices.

    The ``from_vertices`` method must raise a clean ValueError
    rather than a ZeroDivisionError from the area-weighted centroid
    computation.
    """
    match = 'degenerate'
    with pytest.raises(ValueError, match=match):
        PolygonAperture.from_vertices([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])


def test_to_mask_exact_matches_rectangle():
    """
    Test that exact polygon mask for a square equals the rectangular
    result.
    """
    poly = PolygonAperture((5.0, 5.0), SQUARE_OFFSETS)
    rect = RectangularAperture((5.0, 5.0), w=2.0, h=2.0)
    assert_allclose(poly.to_mask(method='exact').data,
                    rect.to_mask(method='exact').data)


def test_to_mask_subpixel_approximates_exact():
    poly = PolygonAperture((5.5, 5.5), TRIANGLE_OFFSETS)
    exact = poly.to_mask(method='exact').data
    sub = poly.to_mask(method='subpixel', subpixels=32).data
    assert np.abs(exact - sub).max() < 0.05


def test_to_mask_center_method():
    poly = PolygonAperture((5.0, 5.0), SQUARE_OFFSETS)
    mask = poly.to_mask(method='center').data
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_to_mask_center_matches_rectangle():
    """
    Test that the ``center`` mask for a rectangle matches the equivalent
    polygon.

    A `RectangularAperture` and a `PolygonAperture` built from the same
    four corners must produce identical ``center`` masks. This includes
    pixel centers that lie exactly on the aperture boundary, which both
    apertures must exclude.
    """
    shape = (12, 12)

    # Axis-aligned rectangle whose edges fall exactly on pixel centers
    # (x in [1, 5], y in [2, 4]), so many pixel centers lie on the
    # boundary and must be excluded by both apertures.
    rect = RectangularAperture((3.0, 3.0), w=4.0, h=2.0)
    corners = [(1.0, 2.0), (5.0, 2.0), (5.0, 4.0), (1.0, 4.0)]
    poly = PolygonAperture.from_vertices(corners)
    rect_mask = rect.to_mask(method='center').to_image(shape)
    poly_mask = poly.to_mask(method='center').to_image(shape)
    assert_allclose(poly_mask, rect_mask)

    # Boundary centers are excluded: only the strictly interior centers
    # (x in {2, 3, 4}, y = 3) are counted.
    assert rect_mask.sum() == 3

    # Rotated rectangle, compared against the equivalent polygon.
    theta = 0.6
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    half_w, half_h = 2.5, 1.5
    local = [(-half_w, -half_h), (half_w, -half_h),
             (half_w, half_h), (-half_w, half_h)]
    abs_corners = [(6.0 + x * cos_t - y * sin_t,
                    6.0 + x * sin_t + y * cos_t) for x, y in local]
    rect2 = RectangularAperture((6.0, 6.0), w=5.0, h=3.0, theta=theta)
    poly2 = PolygonAperture.from_vertices(abs_corners)
    rect2_mask = rect2.to_mask(method='center').to_image(shape)
    poly2_mask = poly2.to_mask(method='center').to_image(shape)
    assert_allclose(poly2_mask, rect2_mask)


def test_to_mask_array_returns_list():
    aper = PolygonAperture(POSITIONS, SQUARE_OFFSETS)
    masks = aper.to_mask(method='exact')
    assert isinstance(masks, list)
    assert len(masks) == len(POSITIONS)


def test_indexing_and_iteration():
    aper = PolygonAperture(POSITIONS, SQUARE_OFFSETS)
    one = aper[1]
    assert one.isscalar
    assert_allclose(one.positions, POSITIONS[1])
    assert_allclose(one.vertex_offsets, aper.vertex_offsets)
    assert list(aper)[0].isscalar
    sliced = aper[0:2]
    assert len(sliced) == 2


def test_indexing_scalar_raises():
    aper = PolygonAperture((0.0, 0.0), SQUARE_OFFSETS)
    match = 'cannot be indexed'
    with pytest.raises(TypeError, match=match):
        aper[0]
    match = 'no len'
    with pytest.raises(TypeError, match=match):
        len(aper)


def test_photometry():
    data = np.ones((30, 30))
    aper = PolygonAperture((10.0, 10.0), SQUARE_OFFSETS)
    flux, _ = aper.photometry(data, method='exact')
    assert_allclose(flux, [4.0])


def test_photometry_subpixel_on_horizontal_boundary():
    """
    Test that ``photometry`` excludes subpixel centers landing
    exactly on a horizontal polygon edge.

    This is the ``photometry`` counterpart of
    ``test_polygon_overlap_subpixel_on_horizontal_boundary`` in
    ``photutils/geometry/tests/test_polygon_overlap_grid.py``,
    using the same wide rectangle (half-widths 2 in x, 1 in y) and
    ``subpixels=3``.

    At an integer position the subpixel centers fall on all multiples
    of 1/3, so the horizontal edges at y = center +/- 1 coincide with
    a row of subpixel centers and are excluded. The strictly interior
    subpixel centers are the 11 x-positions and 5 y-positions inside the
    open rectangle, giving 55 of every 9 subpixels, i.e., 55/9. Were the
    on-boundary centers counted instead, the result would be 91/9.
    """
    data = np.ones((31, 31))
    offsets = np.array([[-2.0, -1.0], [2.0, -1.0],
                        [2.0, 1.0], [-2.0, 1.0]])
    aper = PolygonAperture((15.0, 15.0), offsets)
    flux, _ = aper.photometry(data, method='subpixel', subpixels=3)
    assert_allclose(flux, [55.0 / 9.0])


def test_photometry_subpixel_on_vertical_boundary():
    """
    Test that ``photometry`` excludes subpixel centers landing
    exactly on a vertical polygon edge.

    This is the ``photometry`` counterpart of
    ``test_polygon_overlap_subpixel_on_vertical_boundary``, using
    the same tall rectangle (half-widths 1 in x, 2 in y) and
    ``subpixels=3``. The vertical edges at x = center +/- 1 coincide
    with a column of subpixel centers and are excluded, giving 55/9 (the
    boundary-included value would be 91/9).
    """
    data = np.ones((31, 31))
    offsets = np.array([[-1.0, -2.0], [1.0, -2.0],
                        [1.0, 2.0], [-1.0, 2.0]])
    aper = PolygonAperture((15.0, 15.0), offsets)
    flux, _ = aper.photometry(data, method='subpixel', subpixels=3)
    assert_allclose(flux, [55.0 / 9.0])


def test_photometry_subpixel_on_all_boundaries():
    """
    Test that ``photometry`` excludes subpixel centers on horizontal
    edges, vertical edges, and at corners.

    This is the ``photometry`` counterpart of
    ``test_polygon_overlap_subpixel_on_all_boundaries``, using the same
    unit square (half-widths 1 in both axes) and ``subpixels=3``. At
    an integer position the four edges and four corners coincide with
    subpixel centers and are all excluded, leaving the 5 x-positions
    and 5 y-positions strictly inside the open square, i.e., 25/9 (the
    boundary-included value would be 49/9).
    """
    data = np.ones((31, 31))
    aper = PolygonAperture((15.0, 15.0), SQUARE_OFFSETS)
    flux, _ = aper.photometry(data, method='subpixel', subpixels=3)
    assert_allclose(flux, [25.0 / 9.0])


@pytest.mark.parametrize(('method', 'kwargs'),
                         [('exact', {}),
                          ('subpixel', {'subpixels': 7}),
                          ('center', {})])
def test_array_photometry_matches_scalar(method, kwargs):
    """
    Test that multi-position polygon photometry (batch Cython driver)
    matches the per-position mask-based (grid) result for the exact,
    subpixel, and center methods, including for a non-convex polygon.

    This exercises the batch driver's per-position buffer reuse and
    confirms it is numerically identical to the ``polygon_overlap_grid``
    code path used by ``to_mask``.
    """
    rng = np.random.default_rng(0)
    data = rng.random((60, 60))
    positions = [(15.0, 18.0), (30.0, 32.0), (45.0, 22.0)]
    offsets = _concave_star()

    aper = PolygonAperture(positions, offsets)
    multi, _ = aper.photometry(data, method=method, **kwargs)

    ref = [PolygonAperture(pos, offsets)
           .to_mask(method=method, **kwargs).multiply(data).sum()
           for pos in positions]
    assert_allclose(multi, ref, rtol=1e-12)


def test_invalid_mask_method():
    aper = PolygonAperture((0.0, 0.0), SQUARE_OFFSETS)
    match = 'Invalid mask method'
    with pytest.raises(ValueError, match=match):
        aper.to_mask(method='nonsense')


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_to_patch_scalar():
    import matplotlib.patches as mpatches
    aper = PolygonAperture((0.0, 0.0), SQUARE_OFFSETS)
    patch = aper._to_patch()
    assert isinstance(patch, mpatches.Polygon)


class TestSkyPolygonAperture(BaseTestAperture):
    aperture = SkyPolygonAperture(SKYCOORD, SQUARE_OFFSETS * UNIT)

    def test_index(self):
        aper = self.aperture[2]
        assert isinstance(aper, SkyPolygonAperture)
        assert aper.isscalar
        assert_quantity_allclose(aper.positions.ra,
                                 self.aperture.positions[2].ra)
        assert_quantity_allclose(aper.vertex_offsets,
                                 self.aperture.vertex_offsets)

    def test_slice(self):
        aper = self.aperture[0:2]
        assert isinstance(aper, SkyPolygonAperture)
        assert len(aper) == 2
        assert_quantity_allclose(aper.positions.ra,
                                 self.aperture.positions[0:2].ra)
        assert_quantity_allclose(aper.vertex_offsets,
                                 self.aperture.vertex_offsets)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.vertex_offsets = SQUARE_OFFSETS * 2 * UNIT
        assert aper != self.aperture

    def test_perimeter(self):
        # Unit square (side 2 arcsec): perimeter = 8 arcsec.
        assert_quantity_allclose(self.aperture.perimeter, 8.0 * u.arcsec)

    def test_perimeter_triangle(self):
        aper = SkyPolygonAperture(SKYCOORD[0], TRIANGLE_OFFSETS * u.arcsec)
        expected = (2.0 * np.sqrt(5.0) + 2.0) * u.arcsec
        assert_quantity_allclose(aper.perimeter, expected, rtol=1e-12)

    def test_perimeter_unit(self):
        assert self.aperture.perimeter.unit == u.arcsec

    def test_perimeter_regular_polygon(self):
        # Regular hexagon: perimeter = n * 2 * r * sin(pi / n).
        aper = SkyPolygonAperture.from_regular_polygon(
            SKYCOORD[0], 6, 5.0 * u.arcsec)
        expected = 6.0 * 2.0 * 5.0 * np.sin(np.pi / 6) * u.arcsec
        assert_quantity_allclose(aper.perimeter, expected, rtol=1e-12)


def test_sky_construct():
    pos = SkyCoord(ra=180.0, dec=0.0, unit='deg')
    offsets = TRIANGLE_OFFSETS * u.arcsec
    aper = SkyPolygonAperture(pos, offsets)
    assert aper.isscalar
    assert aper.vertex_offsets.unit.is_equivalent(u.arcsec)


def test_sky_invalid_offsets_not_quantity():
    pos = SkyCoord(ra=180.0, dec=0.0, unit='deg')
    match = 'angular Quantity'
    with pytest.raises(TypeError, match=match):
        SkyPolygonAperture(pos, TRIANGLE_OFFSETS)


def test_sky_invalid_offsets_wrong_unit():
    pos = SkyCoord(ra=180.0, dec=0.0, unit='deg')
    match = 'angular units'
    with pytest.raises(u.UnitsError, match=match):
        SkyPolygonAperture(pos, TRIANGLE_OFFSETS * u.m)


def test_sky_invalid_offsets_wrong_shape():
    pos = SkyCoord(ra=180.0, dec=0.0, unit='deg')
    match = 'shape'
    with pytest.raises(ValueError, match=match):
        SkyPolygonAperture(pos, TRIANGLE_OFFSETS.flatten() * u.arcsec)


def test_sky_invalid_positions():
    with pytest.raises(TypeError, match='SkyCoord'):
        SkyPolygonAperture((1.0, 2.0), TRIANGLE_OFFSETS * u.arcsec)


def test_sky_vertices_scalar():
    pos = SkyCoord(ra=180.0, dec=0.0, unit='deg')
    offsets = SQUARE_OFFSETS * u.arcsec
    aper = SkyPolygonAperture(pos, offsets)
    assert aper.vertices.shape == (4,)


def test_sky_vertices_array():
    pos = SkyCoord(ra=[180.0, 181.0], dec=[0.0, 1.0], unit='deg')
    offsets = SQUARE_OFFSETS * u.arcsec
    aper = SkyPolygonAperture(pos, offsets)
    assert aper.vertices.shape == (2, 4)


def test_sky_from_vertices_round_trip():
    verts = SkyCoord(ra=[180.0, 180.001, 180.0],
                     dec=[0.0, 0.0, 0.001], unit='deg')
    aper = SkyPolygonAperture.from_vertices(verts)
    assert_quantity_allclose(aper.vertices.ra, verts.ra, atol=1e-9 * u.deg)
    assert_quantity_allclose(aper.vertices.dec, verts.dec, atol=1e-9 * u.deg)


def test_sky_from_vertices_requires_skycoord():
    with pytest.raises(TypeError, match='SkyCoord'):
        SkyPolygonAperture.from_vertices(np.array([[0.0, 0.0], [1.0, 0.0],
                                                   [1.0, 1.0]]))


def test_pixel_to_sky_to_pixel_round_trip():
    """
    Test that round-trip through to_sky / to_pixel reproduces original.
    """
    wcs = _make_wcs()
    aper = PolygonAperture([(50.0, 50.0), (60.0, 60.0)], SQUARE_OFFSETS)
    sky = aper.to_sky(wcs)
    back = sky.to_pixel(wcs)
    assert_allclose(back.positions, aper.positions, atol=1e-9)
    assert_allclose(back.vertex_offsets, aper.vertex_offsets, atol=1e-9)


def test_pixel_to_sky_scalar():
    wcs = _make_wcs()
    aper = PolygonAperture((50.0, 50.0), SQUARE_OFFSETS)
    sky = aper.to_sky(wcs)
    assert sky.isscalar
    back = sky.to_pixel(wcs)
    assert_allclose(back.positions, aper.positions, atol=1e-9)


def test_to_sky_vertices_match_pixel_to_world():
    """
    For a scalar polygon, the absolute sky vertices from ``to_sky``
    match feeding the absolute pixel vertices directly to
    ``wcs.pixel_to_world``.

    The comparison is order-independent. The ``_make_wcs`` WCS flips
    parity, so the counter-clockwise normalization applied when setting
    ``vertex_offsets`` may reverse the vertex order relative to the
    input. The two sets of vertices must still describe the same
    polygon.
    """
    wcs = _make_wcs()
    aper = PolygonAperture((50.0, 50.0), SQUARE_OFFSETS)

    abs_pix = aper.vertices
    direct = wcs.pixel_to_world(abs_pix[:, 0], abs_pix[:, 1])
    got = aper.to_sky(wcs).vertices

    # Each direct vertex must coincide with one of the to_sky vertices
    nearest = [np.min(coord.separation(got).to_value(u.arcsec))
               for coord in direct]
    assert_allclose(nearest, 0.0, atol=1e-9)


def test_sky_to_pixel_to_sky_round_trip():
    """
    Test that round-trip through to_pixel / to_sky reproduces the
    original sky aperture.
    """
    wcs = _make_wcs()
    positions = SkyCoord(ra=[180.0, 180.01], dec=[0.0, 0.01], unit='deg')
    aper = SkyPolygonAperture(positions, SQUARE_OFFSETS * u.arcsec)
    pixel = aper.to_pixel(wcs)
    back = pixel.to_sky(wcs)
    assert_quantity_allclose(back.positions.ra, aper.positions.ra,
                             atol=1e-9 * u.deg)
    assert_quantity_allclose(back.positions.dec, aper.positions.dec,
                             atol=1e-9 * u.deg)
    assert_quantity_allclose(back.vertex_offsets, aper.vertex_offsets,
                             atol=1e-6 * u.arcsec)


def test_polygon_matches_rotated_rectangle():
    """
    Test that the exact polygon mask for a rotated rectangle matches the
    exact rectangular mask.

    This is a non-trivial test of the polygon masking machinery, since
    the rectangle vertices are not axis-aligned and thus require the
    full generality of the polygon code.
    """
    rect = RectangularAperture((5.5, 5.5), w=4.0, h=2.0, theta=np.pi / 6)
    # Build the equivalent polygon from the rotated rectangle vertices.
    cos_t = np.cos(rect.theta.to_value(u.radian))
    sin_t = np.sin(rect.theta.to_value(u.radian))
    hw = rect.w / 2
    hh = rect.h / 2
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rot = np.column_stack([local[:, 0] * cos_t - local[:, 1] * sin_t,
                           local[:, 0] * sin_t + local[:, 1] * cos_t])
    poly = PolygonAperture((5.5, 5.5), rot)
    assert_allclose(poly.to_mask(method='exact').data,
                    rect.to_mask(method='exact').data)


def test_sky_indexing():
    pos = SkyCoord(ra=[180.0, 181.0], dec=[0.0, 1.0], unit='deg')
    offsets = SQUARE_OFFSETS * u.arcsec
    aper = SkyPolygonAperture(pos, offsets)
    one = aper[0]
    assert one.isscalar
    assert_quantity_allclose(one.positions.ra, 180.0 * u.deg)


def test_sky_vertex_offsets_reset_lazyproperties():
    """
    Test that changing vertex_offsets on a SkyPolygonAperture clears the
    cached vertices, which are stored as a SkyCoord.
    """
    pos = SkyCoord(ra=180.0, dec=0.0, unit='deg')
    aper = SkyPolygonAperture(pos, SQUARE_OFFSETS * u.arcsec)
    _ = aper.vertices  # cache it
    aper.vertex_offsets = TRIANGLE_OFFSETS * u.arcsec
    assert aper.vertices.shape == (3,)


def test_pixel_vertex_offsets_reset_lazyproperties():
    aper = PolygonAperture((0.0, 0.0), SQUARE_OFFSETS)
    _ = aper.vertices
    aper.vertex_offsets = TRIANGLE_OFFSETS
    assert aper.vertices.shape == (3, 2)


@pytest.mark.parametrize('n_vertices', [3, 4, 5, 6, 8, 12])
def test_pixel_from_regular_polygon_basic(n_vertices):
    aper = PolygonAperture.from_regular_polygon((10.0, 20.0), n_vertices, 5.0)
    assert aper.vertex_offsets.shape == (n_vertices, 2)
    assert aper.n_vertices == n_vertices
    assert aper.is_regular
    assert_allclose(aper.outer_radius, 5.0)
    expected_inner_radius = 5.0 * np.cos(np.pi / n_vertices)
    assert_allclose(aper.inner_radius, expected_inner_radius)
    expected_side = 2.0 * 5.0 * np.sin(np.pi / n_vertices)
    assert_allclose(aper.side_length, expected_side)
    assert_quantity_allclose(aper.interior_angle,
                             ((n_vertices - 2) / n_vertices) * 180.0 * u.deg)
    assert_quantity_allclose(aper.exterior_angle,
                             (360.0 / n_vertices) * u.deg)
    assert_allclose(aper.vertices[0], (10.0, 25.0), atol=1e-12)


def test_pixel_from_regular_polygon_with_angle_quantity():
    aper = PolygonAperture.from_regular_polygon((0.0, 0.0), 4, 1.0,
                                                theta=45.0 * u.deg)
    assert aper.is_regular
    expected = np.array([[-np.sqrt(2) / 2, np.sqrt(2) / 2],
                         [-np.sqrt(2) / 2, -np.sqrt(2) / 2],
                         [np.sqrt(2) / 2, -np.sqrt(2) / 2],
                         [np.sqrt(2) / 2, np.sqrt(2) / 2]])
    assert_allclose(aper.vertex_offsets, expected, atol=1e-12)


def test_pixel_from_regular_polygon_with_angle_float():
    aper = PolygonAperture.from_regular_polygon((0.0, 0.0), 4, 1.0,
                                                theta=np.pi / 4)
    assert_allclose(aper.vertex_offsets[0],
                    (-np.sqrt(2) / 2, np.sqrt(2) / 2), atol=1e-12)


def test_pixel_from_regular_polygon_invalid_n_vertices():
    with pytest.raises(ValueError, match='at least 3'):
        PolygonAperture.from_regular_polygon((0.0, 0.0), 2, 1.0)


def test_pixel_from_regular_polygon_invalid_radius():
    with pytest.raises(ValueError, match='positive'):
        PolygonAperture.from_regular_polygon((0.0, 0.0), 4, 0.0)
    with pytest.raises(ValueError, match='positive'):
        PolygonAperture.from_regular_polygon((0.0, 0.0), 4, -1.0)
    with pytest.raises(ValueError, match='positive'):
        PolygonAperture.from_regular_polygon((0.0, 0.0), 4, np.nan)


def test_pixel_is_regular_false_cases():
    aper = PolygonAperture.from_vertices(
        np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]]))
    assert not aper.is_regular

    aper2 = PolygonAperture((0.0, 0.0),
                            np.array([[1.0, 2.0], [-1.0, 2.0],
                                      [-1.0, -2.0], [1.0, -2.0]]))
    assert not aper2.is_regular

    aper3 = PolygonAperture((0.0, 0.0),
                            np.array([[0.0, 0.0], [4.0, 0.0],
                                      [4.0, 2.0], [2.0, 2.0],
                                      [2.0, 4.0], [0.0, 4.0]]))
    assert not aper3.is_regular


def test_pixel_regular_attrs_raise_when_not_regular():
    aper = PolygonAperture.from_vertices(
        np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]]))
    for attr in ('outer_radius', 'inner_radius', 'side_length',
                 'interior_angle', 'exterior_angle'):
        with pytest.raises(ValueError, match='regular polygon'):
            getattr(aper, attr)


def test_pixel_n_vertices_works_for_any_polygon():
    aper = PolygonAperture.from_vertices(
        np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]]))
    assert aper.n_vertices == 3


@pytest.mark.parametrize('n_vertices', [3, 4, 5, 6])
def test_sky_from_regular_polygon_basic(n_vertices):
    pos = SkyCoord(ra=10.0 * u.deg, dec=30.0 * u.deg)
    aper = SkyPolygonAperture.from_regular_polygon(pos, n_vertices,
                                                   2.0 * u.arcsec)
    assert aper.vertex_offsets.shape == (n_vertices, 2)
    assert aper.n_vertices == n_vertices
    assert aper.is_regular
    assert_quantity_allclose(aper.outer_radius, 2.0 * u.arcsec)
    assert_quantity_allclose(aper.inner_radius,
                             2.0 * np.cos(np.pi / n_vertices) * u.arcsec)
    assert_quantity_allclose(aper.side_length,
                             2.0 * 2.0 * np.sin(np.pi / n_vertices)
                             * u.arcsec)
    assert_quantity_allclose(aper.interior_angle,
                             ((n_vertices - 2) / n_vertices) * 180.0 * u.deg)
    assert_quantity_allclose(aper.exterior_angle,
                             (360.0 / n_vertices) * u.deg)


def test_sky_from_regular_polygon_with_angle():
    pos = SkyCoord(ra=10.0 * u.deg, dec=30.0 * u.deg)
    aper = SkyPolygonAperture.from_regular_polygon(pos, 4,
                                                   1.0 * u.arcmin,
                                                   theta=45.0 * u.deg)
    assert aper.is_regular
    assert aper.vertex_offsets.unit == u.arcmin


def test_sky_from_regular_polygon_invalid_n_vertices():
    pos = SkyCoord(ra=10.0 * u.deg, dec=30.0 * u.deg)
    with pytest.raises(ValueError, match='at least 3'):
        SkyPolygonAperture.from_regular_polygon(pos, 2, 1.0 * u.arcsec)


def test_sky_from_regular_polygon_invalid_radius():
    pos = SkyCoord(ra=10.0 * u.deg, dec=30.0 * u.deg)
    with pytest.raises(TypeError, match='angular Quantity'):
        SkyPolygonAperture.from_regular_polygon(pos, 4, 1.0)
    with pytest.raises(TypeError, match='angular Quantity'):
        SkyPolygonAperture.from_regular_polygon(pos, 4, 1.0 * u.kg)
    with pytest.raises(ValueError, match='positive'):
        SkyPolygonAperture.from_regular_polygon(pos, 4, 0.0 * u.arcsec)
    with pytest.raises(ValueError, match='positive'):
        SkyPolygonAperture.from_regular_polygon(pos, 4, -1.0 * u.arcsec)


def test_sky_from_regular_polygon_invalid_angle():
    pos = SkyCoord(ra=10.0 * u.deg, dec=30.0 * u.deg)
    with pytest.raises(TypeError, match='angular Quantity'):
        SkyPolygonAperture.from_regular_polygon(pos, 4,
                                                1.0 * u.arcsec,
                                                theta=0.5)
    with pytest.raises(TypeError, match='angular Quantity'):
        SkyPolygonAperture.from_regular_polygon(pos, 4,
                                                1.0 * u.arcsec,
                                                theta=1.0 * u.kg)


def test_sky_is_regular_false_and_attrs_raise():
    pos = SkyCoord(ra=10.0 * u.deg, dec=30.0 * u.deg)
    aper = SkyPolygonAperture(pos,
                              np.array([[0.0, 0.0], [4.0, 0.0],
                                        [4.0, 3.0]]) * u.arcsec)
    assert not aper.is_regular
    for attr in ('outer_radius', 'inner_radius', 'side_length',
                 'interior_angle', 'exterior_angle'):
        with pytest.raises(ValueError, match='regular polygon'):
            getattr(aper, attr)


def test_sky_n_vertices_works_for_any_polygon():
    pos = SkyCoord(ra=10.0 * u.deg, dec=30.0 * u.deg)
    aper = SkyPolygonAperture(pos,
                              np.array([[0.0, 0.0], [4.0, 0.0],
                                        [4.0, 3.0]]) * u.arcsec)
    assert aper.n_vertices == 3


def _matches_up_to_cyclic_reversal(a, b, *, atol=1e-10):
    n = len(a)
    if len(b) != n:
        return False
    for direction in (b, b[::-1]):
        for k in range(n):
            shifted = np.roll(direction, -k, axis=0)
            if np.allclose(a, shifted, atol=atol):
                return True
    return False


@pytest.mark.skipif(not pytest.importorskip('regions', minversion='0.11'),
                    reason='requires regions')
def test_pixel_from_regular_polygon_matches_regions():
    from regions import PixCoord, RegularPolygonPixelRegion

    for n in (3, 5, 6, 8):
        for angle in (0.0, 30.0, 90.0):
            aper = PolygonAperture.from_regular_polygon(
                (10.0, 20.0), n, 5.0, theta=angle * u.deg)
            reg = RegularPolygonPixelRegion(
                PixCoord(10.0, 20.0), n, 5.0, angle=angle * u.deg)
            reg_xy = np.column_stack([reg.vertices.x, reg.vertices.y])
            ours = aper.vertices
            assert _matches_up_to_cyclic_reversal(ours, reg_xy)


class TestPolygonApertureTheta:
    """
    Tests for the ``theta`` attribute of `PolygonAperture`.
    """

    def test_theta_zero(self):
        aper = PolygonAperture.from_regular_polygon((10, 20), 6, 5.0,
                                                    theta=0.0)
        assert isinstance(aper.theta, u.Quantity)
        assert_quantity_allclose(aper.theta, 0.0 * u.deg, atol=1e-10 * u.deg)

    def test_theta_unit(self):
        aper = PolygonAperture.from_regular_polygon((10, 20), 5, 3.0,
                                                    theta=np.pi / 3)
        assert aper.theta.unit == u.deg

    @pytest.mark.parametrize('theta_rad', [
        0.0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 2,
        np.pi,
        3 * np.pi / 2,
    ])
    def test_theta_roundtrip_radians(self, theta_rad):
        aper = PolygonAperture.from_regular_polygon((10, 20), 6, 5.0,
                                                    theta=theta_rad)
        expected = np.degrees(theta_rad) % 360.0
        assert_quantity_allclose(aper.theta, expected * u.deg,
                                 atol=1e-10 * u.deg)

    @pytest.mark.parametrize('theta_deg', [0.0, 45.0, 90.0, 135.0, 180.0,
                                           270.0, 359.0])
    def test_theta_roundtrip_quantity(self, theta_deg):
        aper = PolygonAperture.from_regular_polygon((10, 20), 6, 5.0,
                                                    theta=theta_deg * u.deg)
        assert_quantity_allclose(aper.theta, theta_deg * u.deg,
                                 atol=1e-10 * u.deg)

    def test_theta_negative_normalized(self):
        # Negative theta is normalized to [0, 360) deg.
        aper = PolygonAperture.from_regular_polygon((10, 20), 6, 5.0,
                                                    theta=-np.pi / 4)
        assert_quantity_allclose(aper.theta, 315.0 * u.deg,
                                 atol=1e-10 * u.deg)

    @pytest.mark.parametrize('n_vertices', [3, 4, 5, 6, 8, 10])
    def test_theta_different_n_vertices(self, n_vertices):
        aper = PolygonAperture.from_regular_polygon((0, 0), n_vertices, 1.0,
                                                    theta=0.0)
        assert_quantity_allclose(aper.theta, 0.0 * u.deg,
                                 atol=1e-10 * u.deg)

    def test_theta_nonregular_raises(self):
        # Non-regular polygon raises ValueError.
        offsets = [(1.0, 0.0), (2.0, 1.0), (0.0, 2.0)]
        aper = PolygonAperture((10, 20), offsets)
        match = "'theta' is only defined for a regular polygon aperture"
        with pytest.raises(ValueError, match=match):
            _ = aper.theta


class TestSkyPolygonApertureTheta:
    """
    Tests for the ``theta`` attribute of `SkyPolygonAperture`.
    """

    def test_theta_zero(self):
        aper = SkyPolygonAperture.from_regular_polygon(
            SKYCOORD[0], 6, 5.0 * u.arcsec, theta=0.0 * u.deg)
        assert isinstance(aper.theta, u.Quantity)
        assert_quantity_allclose(aper.theta, 0.0 * u.deg, atol=1e-10 * u.deg)

    def test_theta_unit(self):
        aper = SkyPolygonAperture.from_regular_polygon(
            SKYCOORD[0], 5, 3.0 * u.arcsec, theta=60.0 * u.deg)
        assert aper.theta.unit == u.deg

    @pytest.mark.parametrize('theta_deg', [
        0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 270.0,
    ])
    def test_theta_roundtrip(self, theta_deg):
        aper = SkyPolygonAperture.from_regular_polygon(
            SKYCOORD[0], 6, 5.0 * u.arcsec, theta=theta_deg * u.deg)
        assert_quantity_allclose(aper.theta, theta_deg * u.deg,
                                 atol=1e-10 * u.deg)

    def test_theta_negative_normalized(self):
        aper = SkyPolygonAperture.from_regular_polygon(
            SKYCOORD[0], 6, 5.0 * u.arcsec, theta=-45.0 * u.deg)
        assert_quantity_allclose(aper.theta, 315.0 * u.deg,
                                 atol=1e-10 * u.deg)

    def test_theta_nonregular_raises(self):
        offsets = np.array([(0.5, 0.0), (1.0, 0.5), (0.0, 1.0)]) * u.arcsec
        aper = SkyPolygonAperture(SKYCOORD[0], offsets)
        match = "'theta' is only defined for a regular polygon aperture"
        with pytest.raises(ValueError, match=match):
            _ = aper.theta

    @pytest.mark.parametrize('n_vertices', [3, 4, 5, 6, 8])
    def test_theta_different_n_vertices(self, n_vertices):
        aper = SkyPolygonAperture.from_regular_polygon(
            SKYCOORD[0], n_vertices, 1.0 * u.arcsec, theta=0.0 * u.deg)
        assert_quantity_allclose(aper.theta, 0.0 * u.deg,
                                 atol=1e-10 * u.deg)
