# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Polygon apertures in both pixel and sky coordinates.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils import lazyproperty

from photutils.aperture._batch_photometry import SHAPE_POLYGON
from photutils.aperture.attributes import (ApertureAttribute, PixelPositions,
                                           SkyCoordPositions)
from photutils.aperture.core import PixelAperture, SkyAperture
from photutils.geometry._polygon_overlap import polygon_overlap_grid

__all__ = [
    'PolygonAperture',
    'SkyPolygonAperture',
]


def _signed_polygon_area(verts):
    """
    Calculate the signed area of a polygon (positive for
    counter-clockwise vertices) via the shoelace formula.

    Parameters
    ----------
    verts : `~numpy.ndarray`
        ``(n_vertices, 2)`` array of polygon vertices.

    Returns
    -------
    area : float
        The signed area of the polygon.
    """
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orientation(a, b, c):
    """
    Return twice the signed area of the triangle ``a``, ``b``, ``c``.

    The result is positive if ``a -> b -> c`` traverses
    counter-clockwise, negative if clockwise, and zero if the three
    points are collinear.
    """
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a, b, c):
    """
    Return `True` if point ``c``, assumed collinear with ``a`` and
    ``b``, lies within the closed bounding box of the segment ``a b``.
    """
    return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))


def _segments_intersect(p1, p2, p3, p4):
    """
    Return `True` if the closed line segments ``p1 p2`` and ``p3 p4``
    intersect, including the cases where they merely touch at a point or
    overlap collinearly.
    """
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)

    # The segments straddle each other (proper crossing).
    if (((d1 > 0) and (d2 < 0)) or ((d1 < 0) and (d2 > 0))) and \
       (((d3 > 0) and (d4 < 0)) or ((d3 < 0) and (d4 > 0))):
        return True

    # Collinear / touching endpoints.
    if d1 == 0.0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0.0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0.0 and _on_segment(p1, p2, p3):
        return True
    return bool(d4 == 0.0 and _on_segment(p1, p2, p4))


def _polygon_is_simple(verts):
    """
    Return `True` if the closed polygon defined by ``verts`` is simple
    (no two non-adjacent edges intersect).

    Every pair of non-adjacent polygon edges is tested for intersection
    using orientation predicates, including the case where an endpoint
    of one edge lies on another edge. Adjacent edges (which share a
    vertex by construction) are not compared with each other, so a
    polygon that folds straight back onto an adjacent edge is not
    detected here; such degenerate cases are instead excluded by the
    zero-area check in `_validate_simple_polygon`.

    The cost is ``O(n^2)`` in the number of vertices ``n``. This is
    negligible for the polygon sizes used in aperture photometry, and
    the check is performed only once, when the attribute is set.

    Parameters
    ----------
    verts : `~numpy.ndarray`
        ``(n_vertices, 2)`` array of polygon vertices.

    Returns
    -------
    is_simple : bool
        `True` if the polygon is simple, `False` otherwise.
    """
    n = verts.shape[0]
    for i in range(n):
        a1 = verts[i]
        a2 = verts[(i + 1) % n]
        # Compare edge i only with later, non-adjacent edges. Edges i
        # and i+1 share a vertex, and (for i == 0) edges 0 and n-1 share
        # a vertex, so those pairs are skipped.
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b1 = verts[j]
            b2 = verts[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return False
    return True


def _validate_simple_polygon(verts, *, name='vertex_offsets',
                             check_simple=True):
    """
    Verify that ``verts`` defines a simple, non-degenerate polygon.

    The polygon is required to be simple (i.e., non-self-intersecting),
    but it is not required to be convex. Self-intersection is checked
    explicitly with an ``O(n^2)`` edge-pair test (see
    `_polygon_is_simple`); this is inexpensive for the polygon sizes
    used in aperture photometry and runs only once, when the attribute
    is set.

    Parameters
    ----------
    verts : `~numpy.ndarray`
        ``(n_vertices, 2)`` array of polygon vertices.

    name : str
        Name to use in error messages.

    check_simple : bool, optional
        Whether to run the ``O(n^2)`` self-intersection test. Set to
        `False` only for offsets that are already known to define
        a simple polygon (e.g., a circle, ellipse, or rectangle
        discretized into vertices); the cheap shape, vertex-count,
        finiteness, and zero-area checks are still performed.

    Returns
    -------
    verts_ccw : `~numpy.ndarray`
        The vertices reordered to be counter-clockwise.
    """
    if verts.ndim != 2 or verts.shape[1] != 2:
        msg = (f'{name!r} must have shape (n_vertices, 2), '
               f'got {verts.shape}')
        raise ValueError(msg)

    n = verts.shape[0]
    if n < 3:
        msg = (f'{name!r} must define a polygon with at least 3 '
               f'vertices, got {n}')
        raise ValueError(msg)

    if not np.all(np.isfinite(verts)):
        msg = f'{name!r} must not contain any non-finite values'
        raise ValueError(msg)

    signed_area = _signed_polygon_area(verts)
    if signed_area == 0.0:
        msg = (f'{name!r} defines a degenerate polygon with zero '
               'area (vertices are collinear or coincident)')
        raise ValueError(msg)

    if check_simple and not _polygon_is_simple(verts):
        msg = (f'{name!r} must define a simple polygon, but its edges '
               'self-intersect')
        raise ValueError(msg)

    if signed_area < 0:
        verts = verts[::-1].copy()

    return verts


class PixelVertexOffsets(ApertureAttribute):
    """
    Validate and set ``vertex_offsets`` for `PolygonAperture`.

    The value is converted to an ``(n_vertices, 2)`` `~numpy.ndarray` of
    pixel offsets and reordered into counter-clockwise vertex order.
    """

    def __set__(self, instance, value):
        value = self._validate(value)
        if self.name in instance.__dict__:
            self._reset_lazyproperties(instance)
        instance.__dict__[self.name] = value

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            msg = f'{self.name!r} must not be a Quantity'
            raise TypeError(msg)

        try:
            value = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            msg = (f'{self.name!r} must be convertible to a numeric '
                   f'(n_vertices, 2) array')
            raise TypeError(msg) from exc

        return _validate_simple_polygon(value, name=self.name)


class SkyVertexOffsets(ApertureAttribute):
    """
    Validate and set ``vertex_offsets`` for `SkyPolygonAperture`.

    The value must be an angular `~astropy.units.Quantity` of shape
    ``(n_vertices, 2)``. The two columns are the ``(d_lon, d_lat)``
    offsets relative to the corresponding ``positions`` coordinate.
    """

    def __set__(self, instance, value):
        value = self._validate(value)
        if self.name in instance.__dict__:
            self._reset_lazyproperties(instance)
        instance.__dict__[self.name] = value

    def _validate(self, value):
        if not isinstance(value, u.Quantity):
            msg = (f'{self.name!r} must be an angular Quantity with '
                   'shape (n_vertices, 2)')
            raise TypeError(msg)

        if not value.unit.is_equivalent(u.deg):
            msg = f'{self.name!r} must have angular units'
            raise u.UnitsError(msg)

        if value.ndim != 2 or value.shape[1] != 2:
            msg = (f'{self.name!r} must have shape (n_vertices, 2), '
                   f'got {value.shape}')
            raise ValueError(msg)

        # Validate using arcsec values; angular geometry on the local
        # tangent plane is preserved.
        verts = value.to_value(u.arcsec)
        verts = _validate_simple_polygon(verts, name=self.name)

        return (verts * u.arcsec).to(value.unit)


def _vertices_centroid(verts):
    """
    Return the area-weighted geometric center of a polygon.

    For a polygon with vertices ``v_i`` and area ``A``, the centroid is:

        C_x = (1/(6A)) sum_i (x_i + x_{i+1}) (x_i y_{i+1} - x_{i+1} y_i)
        C_y = (1/(6A)) sum_i (y_i + y_{i+1}) (x_i y_{i+1} - x_{i+1} y_i)
    """
    x = verts[:, 0]
    y = verts[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * float(np.sum(cross))
    cx = float(np.sum((x + x1) * cross)) / (6.0 * area)
    cy = float(np.sum((y + y1) * cross)) / (6.0 * area)

    return np.array([cx, cy])


def _regular_polygon_offsets(n_vertices, radius, angle_rad):
    """
    Return the ``(n_vertices, 2)`` array of vertex offsets for a regular
    polygon with the given circumradius and rotation angle.

    The first vertex sits at angle ``pi/2 + angle_rad`` (i.e., directly
    above the center for ``angle_rad == 0``), and subsequent vertices
    are placed counter-clockwise at uniform angular spacing.
    """
    base = np.pi / 2.0 + float(angle_rad)
    step = 2.0 * np.pi / n_vertices
    thetas = base + np.arange(n_vertices) * step
    return np.column_stack([radius * np.cos(thetas),
                            radius * np.sin(thetas)])


def _is_regular_polygon(offsets, *, rtol=1.0e-7, atol=1.0e-10):
    """
    Return `True` if ``offsets`` defines a regular polygon centered at
    the origin (equal vertex distances and uniform angular spacing).
    """
    n = offsets.shape[0]
    radii = np.hypot(offsets[:, 0], offsets[:, 1])

    if not np.allclose(radii, radii[0], rtol=rtol, atol=atol):
        return False

    # Sort vertex angles in [0, 2*pi) and check uniform spacing.
    angles = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * np.pi)
    angles_sorted = np.sort(angles)
    diffs = np.diff(np.concatenate([angles_sorted,
                                    angles_sorted[:1] + 2.0 * np.pi]))
    expected = 2.0 * np.pi / n

    return np.allclose(diffs, expected, rtol=rtol, atol=atol)


class _RegularPolygonGeometry:
    """
    Compute the geometric properties of a regular polygon from its
    vertex offsets.

    This small helper centralizes the regular-polygon math shared by
    `PolygonAperture` and `SkyPolygonAperture`, avoiding duplicated
    trigonometric formulas in the two classes.

    Parameters
    ----------
    offsets : `~numpy.ndarray`
        The ``(n_vertices, 2)`` array of vertex offsets relative to
        the polygon center, as plain numbers in the aperture's native
        length unit (pixels for `PolygonAperture`, arcseconds for
        `SkyPolygonAperture`).

    Notes
    -----
    The length properties (`outer_radius`, `inner_radius`,
    `side_length`) are returned as plain floats in the same length
    unit as ``offsets``. The angular properties (`interior_angle`,
    `exterior_angle`, `theta`) are returned as `~astropy.units.Quantity`
    in degrees and are independent of the length unit.
    """

    def __init__(self, offsets):
        self.offsets = offsets
        self.n_vertices = offsets.shape[0]

    @property
    def is_regular(self):
        return _is_regular_polygon(self.offsets)

    @property
    def side_length(self):
        return (2.0 * self.outer_radius
                * float(np.sin(np.pi / self.n_vertices)))

    @property
    def outer_radius(self):
        return float(np.hypot(self.offsets[0, 0], self.offsets[0, 1]))

    @property
    def inner_radius(self):
        return self.outer_radius * float(np.cos(np.pi / self.n_vertices))

    @property
    def interior_angle(self):
        return ((self.n_vertices - 2) / self.n_vertices) * 180.0 * u.deg

    @property
    def exterior_angle(self):
        return (360.0 / self.n_vertices) * u.deg

    @property
    def theta(self):
        dx, dy = self.offsets[0]
        theta_rad = np.arctan2(dy, dx) - np.pi / 2
        return (np.degrees(theta_rad) % 360.0) * u.deg


class PolygonAperture(PixelAperture):
    """
    A polygon aperture defined in pixel coordinates.

    The aperture has a single fixed shape, defined by ``vertex_offsets``
    relative to each ``positions`` entry, but it can have multiple
    positions.

    Parameters
    ----------
    positions : array_like
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs

    vertex_offsets : array_like
        Shape ``(n_vertices, 2)``. The pixel-space offsets ``(dx, dy)``
        of each polygon vertex relative to ``positions``. The polygon
        must be simple (non-self-intersecting) with at least 3 vertices.
        The polygon may be convex or non-convex. Either clockwise or
        counter-clockwise vertex orderings are accepted; the input is
        normalized to counter-clockwise internally.

    Raises
    ------
    ValueError : `ValueError`
        If ``vertex_offsets`` is not a valid simple polygon with at
        least 3 vertices.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.aperture import PolygonAperture
    >>> # A regular hexagon centered on (10, 20) with circumradius 5.
    >>> theta = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
    >>> offsets = np.column_stack([5.0 * np.cos(theta),
    ...                            5.0 * np.sin(theta)])
    >>> aper = PolygonAperture((10.0, 20.0), offsets)

    Construct a single polygon directly from absolute vertex
    coordinates::

    >>> verts = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]
    >>> aper = PolygonAperture.from_vertices(verts)
    """

    _params = ('positions', 'vertex_offsets')
    positions = PixelPositions('The center pixel position(s).')
    vertex_offsets = PixelVertexOffsets(
        'The (n_vertices, 2) array of pixel offsets of the polygon '
        'vertices, measured relative to ``positions``.')

    def __init__(self, positions, vertex_offsets):
        self.positions = positions
        self.vertex_offsets = vertex_offsets

    @classmethod
    def _from_convex_offsets(cls, positions, offsets):
        """
        Construct a `PolygonAperture` from vertex offsets that are
        known to define a simple polygon, skipping the ``O(n^2)``
        self-intersection check.

        This is a fast-path internal constructor used by the
        ``to_polygon`` methods of the built-in apertures, where the
        generated polygon (a discretized circle, ellipse, or rectangle)
        is guaranteed to be simple by construction. The cheap shape,
        vertex-count, finiteness, zero-area, and counter-clockwise
        normalization checks are still applied.
        """
        offsets = np.asarray(offsets, dtype=float)
        offsets = _validate_simple_polygon(offsets, name='vertex_offsets',
                                           check_simple=False)
        obj = cls.__new__(cls)
        obj.positions = positions
        obj.__dict__['vertex_offsets'] = offsets
        return obj

    @classmethod
    def from_vertices(cls, vertices):
        """
        Construct a `PolygonAperture` from a single set of absolute
        pixel vertices.

        The aperture's ``positions`` is set to the area-weighted
        centroid of the input vertices, and ``vertex_offsets`` is set to
        the vertices minus the centroid.

        Parameters
        ----------
        vertices : array_like
            Shape ``(n_vertices, 2)`` array of absolute pixel ``(x, y)``
            vertex coordinates.

        Returns
        -------
        aperture : `PolygonAperture`
            A scalar polygon aperture whose ``vertices`` property
            reproduces the input.
        """
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2:
            msg = ('vertices must have shape (n_vertices, 2), '
                   f'got {verts.shape}')
            raise ValueError(msg)

        # Validate before computing the centroid: the area-weighted
        # centroid formula divides by the polygon area, which is zero
        # for a degenerate (collinear or coincident) polygon.
        verts = _validate_simple_polygon(verts, name='vertices')
        center = _vertices_centroid(verts)
        offsets = verts - center

        return cls(tuple(center), offsets)

    @classmethod
    def from_regular_polygon(cls, positions, n_vertices, radius, *, theta=0.0):
        """
        Construct a regular `PolygonAperture` from a center, vertex
        count, circumradius, and rotation angle.

        The first vertex is placed directly above the center (in
        the ``+y`` direction) for ``theta == 0`` and is rotated
        counter-clockwise by ``theta``. This convention matches
        `regions.RegularPolygonPixelRegion`.

        Parameters
        ----------
        positions : array_like
            The pixel coordinates of the aperture center(s) (see
            `PolygonAperture` for the accepted formats).

        n_vertices : int
            The number of vertices (must be at least 3).

        radius : float
            The circumradius (outer radius) of the polygon in pixels
            (must be positive).

        theta : float or `~astropy.units.Quantity`, optional
            The rotation angle of the polygon, measured counter-
            clockwise from the ``+y`` axis. A scalar float is
            interpreted in radians.

        Returns
        -------
        aperture : `PolygonAperture`
            The regular polygon aperture.
        """
        n_vertices = int(n_vertices)
        if n_vertices < 3:
            msg = f'n_vertices must be at least 3, got {n_vertices}'
            raise ValueError(msg)

        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0.0:
            msg = f'radius must be a finite positive number, got {radius}'
            raise ValueError(msg)

        if isinstance(theta, u.Quantity):
            theta_rad = float(theta.to_value(u.rad))
        else:
            theta_rad = float(theta)

        offsets = _regular_polygon_offsets(n_vertices, radius, theta_rad)

        return cls(positions, offsets)

    @classmethod
    def _from_star(cls, positions, n_spikes, outer_radius, inner_radius=None,
                   *, theta=0.0, optimal_shape=False, collinear_edges=False):
        """
        Construct a star-shaped `PolygonAperture`.

        The star has ``n_spikes`` spikes, with vertices alternating
        between ``outer_radius`` and ``inner_radius``. The first (outer)
        vertex is placed directly above the center (in the ``+y``
        direction) for ``theta == 0`` and is rotated counter-clockwise
        by ``theta``.

        Parameters
        ----------
        positions : array_like
            The pixel coordinates of the aperture center(s) (see
            `PolygonAperture` for the accepted formats).

        n_spikes : int
            The number of spikes (must be at least 2).

        outer_radius : float
            The distance from the center to the outer (spike) vertices
            in pixels (must be positive).

        inner_radius : float, optional
            The distance from the center to the inner vertices in pixels
            (must be positive and less than ``outer_radius``). If
            either ``optimal_shape`` or ``collinear_edges`` is `True`,
            this parameter is ignored and the inner radius is computed
            automatically.

        theta : float or `~astropy.units.Quantity`, optional
            The rotation angle of the star, measured counter-clockwise
            from the ``+y`` axis to the first (outer) vertex. A scalar
            float is interpreted in radians.

        optimal_shape : bool, optional
            If `True`, compute ``inner_radius`` automatically to
            produce a star with naturally-proportioned geometry. The
            spike angle (the angular width at each point) is set to
            180°/``n_spikes``, which is the exterior angle of a
            regular ``n_spikes``-gon. This creates a balanced, visually
            appealing star where the indentations between spikes are
            symmetric and proportional to the spikes themselves.

            The formula is:

                inner_radius = outer_radius / (1 + 2·cos(π/``n_spikes``))

            For a 5-pointed star, this gives a radius ratio of φ + 1
            (where φ is the golden ratio ≈ 1.618), resulting in the
            inner_radius ≈ 3.82 for outer_radius = 10. This ratio is
            aesthetically pleasing and appears naturally in classical
            star designs. Default is `False`.

        collinear_edges : bool, optional
            If `True`, compute ``inner_radius`` automatically to make
            the two polygon edges adjacent to each spike lie on a single
            straight line. At ``theta=0``, these edges are horizontal
            at the sides of the top spike, creating a symmetric _/\\_
            profile. When rotated (``theta > 0``), the edges remain
            collinear but at a different angle.

            This proportionality constraint makes the indentation
            between spikes flat-bottomed, resulting in a sharp,
            classical star appearance. The formula is:

                inner_radius = (outer_radius · cos(2π/``n_spikes``)
                                / cos(π/``n_spikes``))

            This keyword requires ``n_spikes >= 5``. Default is `False`.

        Returns
        -------
        aperture : `PolygonAperture`
            The star-shaped polygon aperture.
        """
        n_spikes = int(n_spikes)
        if n_spikes < 2:
            msg = f'n_spikes must be at least 2, got {n_spikes}'
            raise ValueError(msg)

        outer_radius = float(outer_radius)
        if not np.isfinite(outer_radius) or outer_radius <= 0.0:
            msg = ('outer_radius must be a finite positive number, '
                   f'got {outer_radius}')
            raise ValueError(msg)

        # Determine which automatic mode to use, if any
        num_auto_modes = sum([optimal_shape, collinear_edges])
        if num_auto_modes > 1:
            msg = 'optimal_shape and collinear_edges cannot both be True'
            raise ValueError(msg)

        if optimal_shape:
            # Compute inner radius to make spike angles = 180°/n_spikes
            inner_radius = (outer_radius
                            / (1.0 + 2.0 * np.cos(np.pi / n_spikes)))
        elif collinear_edges:
            # Compute inner radius to make the polygon edges adjacent
            # to each spike collinear (lie on the same straight line).
            # This creates a flat-bottomed indentation between spikes.
            # At theta=0, these edges are horizontal; when rotated, they
            # remain collinear but at a different angle. This requires
            # n_spikes >= 5 because for smaller values the formula
            # produces a negative or zero inner radius.
            if n_spikes < 5:
                msg = ('collinear_edges requires n_spikes >= 5, '
                       f'got {n_spikes}')
                raise ValueError(msg)
            inner_radius = (outer_radius
                            * np.cos(2 * np.pi / n_spikes)
                            / np.cos(np.pi / n_spikes))
        else:
            if inner_radius is None:
                msg = ('inner_radius must be provided if both optimal_shape '
                       'and horizontal_edges are False')
                raise ValueError(msg)

            inner_radius = float(inner_radius)
            if not np.isfinite(inner_radius) or inner_radius <= 0.0:
                msg = ('inner_radius must be a finite positive number, '
                       f'got {inner_radius}')
                raise ValueError(msg)

            if inner_radius >= outer_radius:
                msg = ('inner_radius must be less than outer_radius '
                       f'({inner_radius} >= {outer_radius})')
                raise ValueError(msg)

        if isinstance(theta, u.Quantity):
            theta_rad = float(theta.to_value(u.rad))
        else:
            theta_rad = float(theta)

        # Create star vertices alternating between outer and inner radii.
        # The star has 2*n_spikes vertices total.
        # For theta=0, the first (outer) vertex points straight up (+y),
        # at angle pi/2 from the x-axis.
        n_vertices = 2 * n_spikes
        step = 2.0 * np.pi / n_vertices
        base = np.pi / 2.0 + theta_rad

        # Create alternating radii: outer, inner, outer, inner, ...
        radii = np.zeros(n_vertices)
        radii[::2] = outer_radius  # even indices: outer vertices
        radii[1::2] = inner_radius  # odd indices: inner vertices

        # Place vertices at the appropriate angles
        thetas = base + np.arange(n_vertices) * step
        offsets = np.column_stack([radii * np.cos(thetas),
                                   radii * np.sin(thetas)])

        return cls(positions, offsets)

    @lazyproperty
    def vertices(self):
        """
        The absolute pixel ``(x, y)`` vertices of the polygon at each
        aperture position.

        For a scalar aperture, the returned array has shape
        ``(n_vertices, 2)``. For an aperture with ``n_positions``
        positions, the returned array has shape ``(n_positions,
        n_vertices, 2)``.
        """
        # vertex_offsets shape: (n_v, 2); positions shape: (..., 2).
        # Broadcast position to vertices.
        offsets = self.vertex_offsets
        if self.isscalar:
            pos = np.asarray(self.positions, dtype=float)
            return pos + offsets

        # positions shape: (n_positions, 2); result: (n_positions, n_v, 2)
        return self.positions[:, np.newaxis, :] + offsets[np.newaxis, :, :]

    @lazyproperty
    def _xy_extents(self):
        """
        Half the extents of the polygon's axis-aligned bounding box.
        """
        x_extent = float(np.max(np.abs(self.vertex_offsets[:, 0])))
        y_extent = float(np.max(np.abs(self.vertex_offsets[:, 1])))
        return x_extent, y_extent

    def _batch_shape_params(self):
        # The vertex offsets are counter-clockwise (normalized when the
        # attribute is set) and flattened as (x0, y0, x1, y1, ...) for
        # the batch Cython photometry driver.
        return SHAPE_POLYGON, tuple(self.vertex_offsets.ravel())

    @lazyproperty
    def area(self):
        """
        The exact geometric area of the aperture shape.
        """
        return abs(_signed_polygon_area(self.vertex_offsets))

    @lazyproperty
    def perimeter(self):
        """
        The perimeter of the polygon in pixels.

        The perimeter is computed as the sum of the Euclidean distances
        between consecutive vertices.
        """
        offsets = self.vertex_offsets
        diff = np.roll(offsets, -1, axis=0) - offsets
        return float(np.sum(np.hypot(diff[:, 0], diff[:, 1])))

    @lazyproperty
    def n_vertices(self):
        """
        The number of vertices of the polygon.
        """
        return int(self.vertex_offsets.shape[0])

    @lazyproperty
    def _regular_geometry(self):
        """
        Helper that computes the regular-polygon geometric properties
        from the pixel vertex offsets.
        """
        return _RegularPolygonGeometry(self.vertex_offsets)

    @lazyproperty
    def is_regular(self):
        """
        `True` if the polygon is regular (equal-length sides and equal
        interior angles, with all vertices at the same distance from
        ``positions``).
        """
        return self._regular_geometry.is_regular

    def _check_regular(self, name):
        if not self.is_regular:
            msg = (f'{name!r} is only defined for a regular polygon '
                   'aperture')
            raise ValueError(msg)

    @lazyproperty
    def outer_radius(self):
        """
        The outer (circumscribed-circle) radius of the regular polygon
        in pixels, i.e., the distance from ``positions`` to each vertex.

        This is also called the `circumradius
        <https://mathworld.wolfram.com/Circumradius.html>`_.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('outer_radius')
        return self._regular_geometry.outer_radius

    @lazyproperty
    def inner_radius(self):
        """
        The inner (inscribed-circle) radius of the regular polygon in
        pixels, i.e., the distance from ``positions`` to the midpoint of
        each side.

        This is also called the `inradius
        <https://mathworld.wolfram.com/Inradius.html>`_

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('inner_radius')
        return self._regular_geometry.inner_radius

    @lazyproperty
    def side_length(self):
        """
        The side length of the regular polygon, in pixels.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('side_length')
        return self._regular_geometry.side_length

    @lazyproperty
    def interior_angle(self):
        """
        The interior angle of the regular polygon as an angular
        `~astropy.units.Quantity` in degrees.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('interior_angle')
        return self._regular_geometry.interior_angle

    @lazyproperty
    def exterior_angle(self):
        """
        The exterior angle of the regular polygon as an angular
        `~astropy.units.Quantity` in degrees.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('exterior_angle')
        return self._regular_geometry.exterior_angle

    @lazyproperty
    def theta(self):
        """
        The rotation angle of the regular polygon as an angular
        `~astropy.units.Quantity` in degrees.

        The angle is measured counter-clockwise from the ``+y``
        axis to the first vertex, in the range ``[0, 360) deg``.
        This convention matches the ``theta`` parameter of
        `~PolygonAperture.from_regular_polygon`.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('theta')
        return self._regular_geometry.theta

    def _compute_overlap(self, edges, nx, ny, use_exact, subpixels):
        """
        Compute the overlap of the aperture on the pixel grid.

        Parameters
        ----------
        edges : list of 4 1D `~numpy.ndarray`
            The edges of the pixel grid in the form of
            ``[x_edges, y_edges, x_centers, y_centers]``.

        nx, ny : int
            The number of pixels in the x and y directions.

        use_exact : bool
            Whether to use the exact method for calculating the overlap.

        subpixels : int
            The number of subpixels to use in each dimension for the
            subpixel method.

        Returns
        -------
        overlap : 2D `~numpy.ndarray`
            The overlap of the aperture on the pixel grid. The values
            will be between 0 and 1, where 0 means no overlap and 1
            means full overlap.
        """
        verts_x = np.ascontiguousarray(self.vertex_offsets[:, 0])
        verts_y = np.ascontiguousarray(self.vertex_offsets[:, 1])
        return polygon_overlap_grid(edges[0], edges[1], edges[2], edges[3],
                                    nx, ny, verts_x, verts_y,
                                    use_exact, subpixels)

    def _to_patch(self, *, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.Polygon` patch (or list of them)
        for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.Patch` or list of \
                `~matplotlib.patches.Patch`
            A patch for the aperture. If the aperture is scalar then a
            single `~matplotlib.patches.Patch` is returned, otherwise a
            list of `~matplotlib.patches.Patch` is returned.
        """
        import matplotlib.patches as mpatches

        xy_positions, patch_kwargs = self._define_patch_params(origin=origin,
                                                               **kwargs)
        offsets = self.vertex_offsets
        patches = [mpatches.Polygon(pos + offsets, closed=True,
                                    **patch_kwargs)
                   for pos in xy_positions]

        if self.isscalar:
            return patches[0]

        return patches

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyPolygonAperture` object defined in
        celestial coordinates.

        The conversion uses ``wcs.pixel_to_world`` directly on the
        absolute pixel vertex coordinates, evaluated at the first aperture
        position.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `SkyPolygonAperture`
            The equivalent sky-coordinate polygon aperture.
        """
        xpos, ypos = np.transpose(self.positions)
        positions = wcs.pixel_to_world(xpos, ypos)

        first_pos = np.atleast_2d(self.positions)[0]
        first_skypos = positions if self.isscalar else positions[0]
        abs_x = first_pos[0] + self.vertex_offsets[:, 0]
        abs_y = first_pos[1] + self.vertex_offsets[:, 1]
        sky_vertices = wcs.pixel_to_world(abs_x, abs_y)

        d_lon, d_lat = first_skypos.spherical_offsets_to(sky_vertices)
        offsets = u.Quantity([d_lon, d_lat]).T

        return SkyPolygonAperture(positions=positions,
                                  vertex_offsets=offsets)


class SkyPolygonAperture(SkyAperture):
    """
    A polygon aperture defined in sky coordinates.

    The aperture has a single fixed shape, defined by angular
    ``vertex_offsets`` relative to each ``positions`` entry, but it can
    have multiple positions. The angular offsets are interpreted on the
    local tangent plane at each position.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    vertex_offsets : `~astropy.units.Quantity`
        Shape ``(n_vertices, 2)`` angular Quantity of polygon vertex
        offsets ``(d_lon, d_lat)`` relative to ``positions``, applied as
        spherical offsets on the local tangent plane. The polygon must
        be simple (non-self-intersecting) with at least 3 vertices. The
        polygon may be convex or non-convex.

    Examples
    --------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> from astropy.coordinates import SkyCoord
    >>> from photutils.aperture import SkyPolygonAperture
    >>> positions = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    >>> theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    >>> r = 1.0
    >>> offsets = np.column_stack([r * np.cos(theta),
    ...                            r * np.sin(theta)]) * u.arcsec
    >>> aper = SkyPolygonAperture(positions, offsets)
    """

    _params = ('positions', 'vertex_offsets')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    vertex_offsets = SkyVertexOffsets(
        'The (n_vertices, 2) angular Quantity of polygon vertex offsets '
        '(d_lon, d_lat) relative to ``positions``.')

    def __init__(self, positions, vertex_offsets):
        self.positions = positions
        self.vertex_offsets = vertex_offsets

    @classmethod
    def _from_convex_offsets(cls, positions, offsets):
        """
        Construct a `SkyPolygonAperture` from angular vertex offsets
        that are known to define a simple polygon, skipping the
        ``O(n^2)`` self-intersection check.

        This is a fast-path internal constructor used by the
        ``to_polygon`` methods of the built-in sky apertures, where the
        generated polygon (a discretized circle, ellipse, or rectangle)
        is guaranteed to be simple by construction. The cheap shape,
        vertex-count, finiteness, zero-area, and counter-clockwise
        normalization checks are still applied.

        Parameters
        ----------
        positions : `~astropy.coordinates.SkyCoord`
            The center position(s) in sky coordinates.

        offsets : `~astropy.units.Quantity`
            The ``(n_vertices, 2)`` angular Quantity of vertex offsets
            ``(d_lon, d_lat)`` relative to ``positions``.

        Returns
        -------
        aperture : `SkyPolygonAperture`
            The sky polygon aperture.
        """
        verts = offsets.to_value(u.arcsec)
        verts = _validate_simple_polygon(verts, name='vertex_offsets',
                                         check_simple=False)
        offsets = (verts * u.arcsec).to(offsets.unit)
        obj = cls.__new__(cls)
        obj.positions = positions
        obj.__dict__['vertex_offsets'] = offsets
        return obj

    @classmethod
    def from_vertices(cls, vertices):
        """
        Construct a `SkyPolygonAperture` from a single set of absolute
        sky vertices.

        The aperture's ``positions`` is set to an approximate centroid
        of the input vertices, computed as a single tangent-plane
        mean of the spherical offsets from the first vertex (i.e.,
        the vertices' spherical offsets from this centroid average
        to approximately zero), and ``vertex_offsets`` is set to the
        spherical offsets of the vertices from this centroid.

        Parameters
        ----------
        vertices : `~astropy.coordinates.SkyCoord`
            A SkyCoord with shape ``(n_vertices,)`` of absolute polygon
            vertex coordinates.

        Returns
        -------
        aperture : `SkyPolygonAperture`
            A scalar sky polygon aperture whose ``vertices`` property
            reproduces the input.
        """
        if not isinstance(vertices, SkyCoord) or vertices.ndim != 1:
            msg = ('vertices must be a 1-D SkyCoord with at least 3 '
                   'vertices')
            raise TypeError(msg)

        # Provisional centroid: spherical mean approximated via mean of
        # offsets from the first vertex.
        ref = vertices[0]
        d_lon, d_lat = ref.spherical_offsets_to(vertices)
        center_d_lon = d_lon.mean()
        center_d_lat = d_lat.mean()
        center = ref.spherical_offsets_by(center_d_lon, center_d_lat)
        d_lon, d_lat = center.spherical_offsets_to(vertices)
        offsets = u.Quantity([d_lon, d_lat]).T
        return cls(positions=center, vertex_offsets=offsets)

    @classmethod
    def from_regular_polygon(cls, positions, n_vertices, radius, *,
                             theta=0.0 * u.deg):
        """
        Construct a regular `SkyPolygonAperture` from a center, vertex
        count, angular circumradius, and rotation angle.

        The first vertex is placed at the ``+lat`` edge of the local
        tangent plane (typically northward for standard right-handed
        celestial coordinates) for ``theta == 0`` and is rotated
        counter-clockwise by ``theta``.

        Parameters
        ----------
        positions : `~astropy.coordinates.SkyCoord`
            The celestial coordinates of the aperture center(s).

        n_vertices : int
            The number of vertices (must be at least 3).

        radius : `~astropy.units.Quantity`
            The angular circumradius (outer radius) of the polygon (must
            be a positive angular Quantity).

        theta : `~astropy.units.Quantity`, optional
            The rotation angle of the polygon, measured counter-
            clockwise from the ``+lat`` axis on the local tangent
            plane. This assumes a right-handed coordinate system (e.g.,
            standard celestial coordinates).

        Returns
        -------
        aperture : `SkyPolygonAperture`
            The regular sky polygon aperture.
        """
        n_vertices = int(n_vertices)
        if n_vertices < 3:
            msg = f'n_vertices must be at least 3, got {n_vertices}'
            raise ValueError(msg)

        if (not isinstance(radius, u.Quantity)
                or not radius.unit.is_equivalent(u.deg)):
            msg = 'radius must be an angular Quantity'
            raise TypeError(msg)

        radius_arcsec = float(radius.to_value(u.arcsec))
        if not np.isfinite(radius_arcsec) or radius_arcsec <= 0.0:
            msg = f'radius must be a finite positive Quantity, got {radius}'
            raise ValueError(msg)

        if (not isinstance(theta, u.Quantity)
                or not theta.unit.is_equivalent(u.deg)):
            msg = 'theta must be an angular Quantity'
            raise TypeError(msg)

        theta_rad = float(theta.to_value(u.radian))
        offsets_arcsec = _regular_polygon_offsets(n_vertices, radius_arcsec,
                                                  theta_rad)
        offsets = (offsets_arcsec * u.arcsec).to(radius.unit)

        return cls(positions=positions, vertex_offsets=offsets)

    @lazyproperty
    def n_vertices(self):
        """
        The number of vertices of the polygon.
        """
        return int(self.vertex_offsets.shape[0])

    @lazyproperty
    def perimeter(self):
        """
        The angular perimeter of the polygon as a
        `~astropy.units.Quantity`.

        The perimeter is computed as the sum of the angular distances
        between consecutive vertices on the local tangent plane.
        """
        offsets = self.vertex_offsets.to_value(u.arcsec)
        diff = np.roll(offsets, -1, axis=0) - offsets
        total = float(np.sum(np.hypot(diff[:, 0], diff[:, 1])))
        return total * u.arcsec

    @lazyproperty
    def _regular_geometry(self):
        """
        Helper that computes the regular-polygon geometric properties
        from the angular vertex offsets (in arcseconds).
        """
        return _RegularPolygonGeometry(
            self.vertex_offsets.to_value(u.arcsec))

    @lazyproperty
    def is_regular(self):
        """
        `True` if the polygon is regular (equal-length sides and equal
        interior angles, with all vertices at the same angular distance
        from ``positions``).
        """
        return self._regular_geometry.is_regular

    def _check_regular(self, name):
        if not self.is_regular:
            msg = (f'{name!r} is only defined for a regular polygon '
                   'aperture')
            raise ValueError(msg)

    @lazyproperty
    def outer_radius(self):
        """
        The angular outer (circumscribed-circle) radius of the regular
        polygon as a `~astropy.units.Quantity`, i.e., the distance from
        ``positions`` to each vertex.

        This is also called the `circumradius
        <https://mathworld.wolfram.com/Circumradius.html>`_.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('outer_radius')
        return self._regular_geometry.outer_radius * u.arcsec

    @lazyproperty
    def inner_radius(self):
        """
        The angular inner (inscribed-circle) radius of the regular
        polygon as a `~astropy.units.Quantity`, i.e., the distance from
        ``positions`` to the midpoint of each side.

        This is also called the `inradius
        <https://mathworld.wolfram.com/Inradius.html>`_

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('inner_radius')
        return self._regular_geometry.inner_radius * u.arcsec

    @lazyproperty
    def side_length(self):
        """
        The angular side length of the regular polygon as a
        `~astropy.units.Quantity`.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('side_length')
        return self._regular_geometry.side_length * u.arcsec

    @lazyproperty
    def interior_angle(self):
        """
        The interior angle of the regular polygon as an angular
        `~astropy.units.Quantity` in degrees.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('interior_angle')
        return self._regular_geometry.interior_angle

    @lazyproperty
    def exterior_angle(self):
        """
        The exterior angle of the regular polygon as an angular
        `~astropy.units.Quantity` in degrees.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('exterior_angle')
        return self._regular_geometry.exterior_angle

    @lazyproperty
    def theta(self):
        """
        The rotation angle of the regular polygon as an angular
        `~astropy.units.Quantity` in degrees.

        The angle is measured counter-clockwise from the ``+lat``
        axis on the local tangent plane to the first vertex, in
        the range ``[0, 360) deg``. This assumes a right-handed
        coordinate system (e.g., standard celestial coordinates).
        This convention matches the ``theta`` parameter of
        `~SkyPolygonAperture.from_regular_polygon`.

        This attribute is only defined for regular polygons. Accessing
        it for a non-regular polygon raises `ValueError`.
        """
        self._check_regular('theta')
        return self._regular_geometry.theta

    @lazyproperty
    def vertices(self):
        """
        The absolute sky vertices of the polygon at each aperture
        position, as a `~astropy.coordinates.SkyCoord`.

        For a scalar aperture, the returned SkyCoord has shape
        ``(n_vertices,)``. For an aperture with ``n_positions``
        positions, the returned SkyCoord has shape ``(n_positions,
        n_vertices)``.
        """
        d_lon = self.vertex_offsets[:, 0]
        d_lat = self.vertex_offsets[:, 1]

        if self.isscalar:
            return self.positions.spherical_offsets_by(d_lon, d_lat)

        # Broadcast to shape (n_positions, n_vertices).
        positions = self.positions[:, np.newaxis]
        d_lon_b = d_lon[np.newaxis, :]
        d_lat_b = d_lat[np.newaxis, :]

        return positions.spherical_offsets_by(d_lon_b, d_lat_b)

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `PolygonAperture` object defined in
        pixel coordinates.

        The conversion uses ``wcs.world_to_pixel`` directly on the
        absolute sky vertex coordinate, evaluated at the first aperture
        position.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `PolygonAperture`
            The equivalent pixel-coordinate polygon aperture.
        """
        xpos, ypos = wcs.world_to_pixel(self.positions)
        positions = np.transpose((xpos, ypos))

        first_skypos = self.positions if self.isscalar else self.positions[0]
        first_x = float(np.atleast_1d(xpos)[0])
        first_y = float(np.atleast_1d(ypos)[0])
        d_lon = self.vertex_offsets[:, 0]
        d_lat = self.vertex_offsets[:, 1]
        abs_sky_vertices = first_skypos.spherical_offsets_by(d_lon, d_lat)
        vx, vy = wcs.world_to_pixel(abs_sky_vertices)
        offsets = np.column_stack([vx - first_x, vy - first_y])
        return PolygonAperture(positions=positions,
                               vertex_offsets=offsets)
