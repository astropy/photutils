# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Cython kernels that evaluate the bivariate B-splines of the PSF image
models at scattered points.

The splines are the ones built by
`scipy.interpolate.RectBivariateSpline`, passed in as their FITPACK
knot vectors and coefficient arrays. The PSF image models build bicubic
interpolating splines (``kx=ky=3``, ``s=0``), and the kernels evaluate
only bicubic splines.

The evaluation follows the FITPACK ``fpbisp`` and ``fpbspl`` routines
that ``RectBivariateSpline.__call__`` uses. The coordinates are clamped
to the knot range (constant extrapolation), the knot interval is
located, and the nonzero B-spline basis functions are computed with
the Cox-de Boor recursion. The partial derivatives are computed from
the derivative of the basis functions in the same pass, which is
mathematically identical to evaluating the derivative splines that
``RectBivariateSpline.partial_derivative`` builds. The one difference
is beyond the knot range, where the clamped spline is constant along
the clamped axis. The partial derivative along that axis is zero there,
while the derivative splines give the derivative at the edge.

The kernels combine the splines of the (up to four) bounding grid PSFs
of a gridded PSF model with their bilinear weights in a single call,
instead of one spline call per grid PSF and per derivative. A single
image PSF is the case of one spline with unit weight. For the small
fit regions used in PSF fitting the spline call overhead dominates the
evaluation, so this reduces the model evaluation cost by several times.

The kernels run without the GIL and use no global mutable state, so this
module is safe to use from multiple threads, including on free-threaded
Python builds.
"""

__all__ = ['bispline_sum', 'bispline_sum_deriv']

cdef enum:
    # The degree of the splines along both axes
    DEGREE = 3


cdef inline Py_ssize_t _find_interval(const double[::1] t,
                                      double x) noexcept nogil:
    """
    Return the index ``l`` of the knot interval ``t[l] <= x < t[l + 1]``
    for a coordinate already clamped to the knot range.

    The result is limited to the valid intervals ``k <= l <= n - k - 2``
    for the spline degree ``k``, which matches the FITPACK search.
    """
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t lo = DEGREE
    cdef Py_ssize_t hi = n - DEGREE - 1  # exclusive upper bound of l + 1
    cdef Py_ssize_t mid

    # Binary search for the last knot <= x within t[k:n-k-1]. The
    # upper bound keeps the result at or below n - k - 2.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


cdef inline void _raise_degree(const double[::1] t, Py_ssize_t l, double x,
                               int j, double *h) noexcept nogil:
    """
    Replace the ``j`` nonzero B-spline basis functions of degree
    ``j - 1`` on the knot interval ``l`` at ``x``, in ``h``, with the
    ``j + 1`` basis functions of degree ``j``.

    This is one step of the FITPACK ``fpbspl`` recursion.
    """
    cdef double hh[DEGREE]
    cdef double f, denom
    cdef int i
    cdef Py_ssize_t li, lj

    for i in range(j):
        hh[i] = h[i]
    h[0] = 0.0
    for i in range(j):
        li = l + i + 1
        lj = li - j
        denom = t[li] - t[lj]
        if denom == 0.0:
            h[i + 1] = 0.0
        else:
            f = hh[i] / denom
            h[i] += f * (t[li] - x)
            h[i + 1] = f * (x - t[lj])


cdef inline void _basis(const double[::1] t, Py_ssize_t l, double x,
                        double *h) noexcept nogil:
    """
    Compute the four nonzero cubic B-spline basis functions on the
    knot interval ``l`` at ``x``.

    ``h[i]`` is the basis function ``B[l - 3 + i, 3]``.
    """
    cdef int j

    h[0] = 1.0
    for j in range(1, DEGREE + 1):
        _raise_degree(t, l, x, j, h)


cdef inline void _basis_deriv(const double[::1] t, Py_ssize_t l, double x,
                              double *h, double *dh) noexcept nogil:
    """
    Compute the four nonzero cubic B-spline basis functions on the
    knot interval ``l`` at ``x`` (``h``) and their first derivatives
    (``dh``).
    """
    cdef double f, denom
    cdef int i, j
    cdef Py_ssize_t li, lj

    h[0] = 1.0
    for j in range(1, DEGREE):
        _raise_degree(t, l, x, j, h)

    # The quadratic basis functions give the derivatives of the cubic
    # basis functions
    for i in range(DEGREE + 1):
        dh[i] = 0.0
    for i in range(DEGREE):
        li = l + i + 1
        lj = li - DEGREE
        denom = t[li] - t[lj]
        if denom != 0.0:
            f = DEGREE * h[i] / denom
            dh[i] -= f
            dh[i + 1] += f

    _raise_degree(t, l, x, DEGREE, h)


cdef inline double _clamp(const double[::1] t, double x) noexcept nogil:
    cdef Py_ssize_t n = t.shape[0]
    if x < t[DEGREE]:
        return t[DEGREE]
    if x > t[n - DEGREE - 1]:
        return t[n - DEGREE - 1]
    return x


def bispline_sum(const double[::1] tx, const double[::1] ty,
                 const double[:, ::1] coeffs, const Py_ssize_t[::1] grid_idx,
                 const double[::1] weights, const double[::1] x,
                 const double[::1] y, double[::1] out):
    """
    Evaluate a weighted sum of bicubic B-splines at scattered points.

    Parameters
    ----------
    tx, ty : 1D ndarray of float64 (C-contiguous)
        The knot vectors along the first (x) and second (y) variables
        of bicubic splines (``kx=ky=3``), in the FITPACK format of
        ``RectBivariateSpline.get_knots()``. The knots
        must be in non-decreasing order, which is not checked. The knot
        interval is located with a binary search, which gives wrong
        values for unordered knots. A scipy smoothing fit (``s > 0``)
        can return such knots.

    coeffs : 2D ndarray of float64 (C-contiguous)
        The coefficient arrays, one row per spline, each in the FITPACK
        layout (x-major) of ``RectBivariateSpline.get_coeffs()``.

    grid_idx : 1D ndarray of intp (C-contiguous)
        The rows of ``coeffs`` to combine.

    weights : 1D ndarray of float64 (C-contiguous)
        The weight of each combined spline. Splines with a zero weight
        are skipped.

    x, y : 1D ndarray of float64 (C-contiguous)
        The coordinates of the evaluation points.

    out : 1D ndarray of float64 (C-contiguous)
        Output. Filled with ``sum_g weights[g] * S_g(x, y)``, where
        ``S_g`` is the spline of row ``grid_idx[g]``.
    """
    cdef Py_ssize_t n_points = x.shape[0]
    cdef Py_ssize_t n_splines = grid_idx.shape[0]
    cdef Py_ssize_t nky1 = ty.shape[0] - DEGREE - 1
    cdef Py_ssize_t p, g, lx, ly, row, base
    cdef int i, j
    cdef double xp, yp, val, s, sy, w
    cdef double hx[DEGREE + 1]
    cdef double hy[DEGREE + 1]

    # The kernels index the arrays without bounds checks, so the checks
    # below must cover every index that the loops compute. The grid_idx
    # check comes first because placing it after the other checks makes
    # the compiler generate a 60% slower evaluation loop.
    for g in range(n_splines):
        if grid_idx[g] < 0 or grid_idx[g] >= coeffs.shape[0]:
            msg = 'grid_idx has an index that is not a row of coeffs'
            raise ValueError(msg)
    if (y.shape[0] != n_points or out.shape[0] != n_points
            or weights.shape[0] != n_splines):
        msg = 'x, y, out, grid_idx, and weights have inconsistent lengths'
        raise ValueError(msg)
    if tx.shape[0] < 2 * (DEGREE + 1) or ty.shape[0] < 2 * (DEGREE + 1):
        msg = 'the knot vectors are too short for bicubic splines'
        raise ValueError(msg)
    if coeffs.shape[1] != (tx.shape[0] - DEGREE - 1) * nky1:
        msg = 'coeffs does not match the knot vectors'
        raise ValueError(msg)

    with nogil:
        for p in range(n_points):
            xp = _clamp(tx, x[p])
            yp = _clamp(ty, y[p])
            lx = _find_interval(tx, xp)
            ly = _find_interval(ty, yp)
            _basis(tx, lx, xp, hx)
            _basis(ty, ly, yp, hy)
            val = 0.0
            for g in range(n_splines):
                w = weights[g]
                if w == 0.0:
                    continue
                row = grid_idx[g]
                s = 0.0
                for i in range(DEGREE + 1):
                    base = (lx - DEGREE + i) * nky1 + ly - DEGREE
                    sy = 0.0
                    for j in range(DEGREE + 1):
                        sy += coeffs[row, base + j] * hy[j]
                    s += hx[i] * sy
                val += w * s
            out[p] = val


def bispline_sum_deriv(const double[::1] tx, const double[::1] ty,
                       const double[:, ::1] coeffs,
                       const Py_ssize_t[::1] grid_idx,
                       const double[::1] weights, const double[::1] dw_dx,
                       const double[::1] dw_dy, double scale_x,
                       double scale_y, const double[::1] x,
                       const double[::1] y, double[::1] out,
                       double[::1] out_dx, double[::1] out_dy):
    """
    Evaluate a weighted sum of bicubic B-splines and its derivatives
    with respect to a shift of the spline coordinates and to the
    weights, at scattered points.

    This is the `fit_deriv` kernel of the gridded PSF models. With
    ``S_g`` the spline of row ``grid_idx[g]``, the outputs are

    * ``out = sum_g weights[g] * S_g``
    * ``out_dx = sum_g dw_dx[g] * S_g - scale_x * weights[g] * dS_g/dx``
    * ``out_dy = sum_g dw_dy[g] * S_g - scale_y * weights[g] * dS_g/dy``

    where ``scale_x`` and ``scale_y`` are the factors that convert a
    model position shift to a shift of the spline coordinates (the
    oversampling), with the sign of the shift already included.

    The coordinates are clamped to the knot range, so ``dS_g/dx`` is
    zero for a point beyond the knot range along x, and likewise for
    ``dS_g/dy`` along y.

    Parameters
    ----------
    tx, ty, coeffs, grid_idx, weights, x, y
        See `bispline_sum`.

    dw_dx, dw_dy : 1D ndarray of float64 (C-contiguous)
        The derivatives of the weights with respect to the model x and
        y positions. A spline is skipped only if its weight and both
        weight derivatives are zero.

    scale_x, scale_y : float
        The spline coordinate shift per unit model position shift.

    out, out_dx, out_dy : 1D ndarray of float64 (C-contiguous)
        Output. The weighted sum and its two derivatives.
    """
    cdef Py_ssize_t n_points = x.shape[0]
    cdef Py_ssize_t n_splines = grid_idx.shape[0]
    cdef Py_ssize_t nky1 = ty.shape[0] - DEGREE - 1
    cdef Py_ssize_t p, g, lx, ly, row, base
    cdef int i, j
    cdef double xp, yp, w, wx, wy, c
    cdef double s, s_dx, s_dy, sy, sy_dy, val, val_dx, val_dy
    cdef double hx[DEGREE + 1]
    cdef double hy[DEGREE + 1]
    cdef double dhx[DEGREE + 1]
    cdef double dhy[DEGREE + 1]

    # The kernels index the arrays without bounds checks, so the checks
    # below must cover every index that the loops compute. The grid_idx
    # check comes first because placing it after the other checks makes
    # the compiler generate a 60% slower evaluation loop.
    for g in range(n_splines):
        if grid_idx[g] < 0 or grid_idx[g] >= coeffs.shape[0]:
            msg = 'grid_idx has an index that is not a row of coeffs'
            raise ValueError(msg)
    if (y.shape[0] != n_points or out.shape[0] != n_points
            or out_dx.shape[0] != n_points or out_dy.shape[0] != n_points
            or weights.shape[0] != n_splines or dw_dx.shape[0] != n_splines
            or dw_dy.shape[0] != n_splines):
        msg = ('x, y, the outputs, grid_idx, weights, dw_dx, and dw_dy '
               'have inconsistent lengths')
        raise ValueError(msg)
    if tx.shape[0] < 2 * (DEGREE + 1) or ty.shape[0] < 2 * (DEGREE + 1):
        msg = 'the knot vectors are too short for bicubic splines'
        raise ValueError(msg)
    if coeffs.shape[1] != (tx.shape[0] - DEGREE - 1) * nky1:
        msg = 'coeffs does not match the knot vectors'
        raise ValueError(msg)

    with nogil:
        for p in range(n_points):
            xp = _clamp(tx, x[p])
            yp = _clamp(ty, y[p])
            lx = _find_interval(tx, xp)
            ly = _find_interval(ty, yp)
            _basis_deriv(tx, lx, xp, hx, dhx)
            _basis_deriv(ty, ly, yp, hy, dhy)
            # A clamped coordinate does not change with the input
            # coordinate, so the spline is constant along that axis
            if x[p] < xp or x[p] > xp:
                for i in range(DEGREE + 1):
                    dhx[i] = 0.0
            if y[p] < yp or y[p] > yp:
                for j in range(DEGREE + 1):
                    dhy[j] = 0.0
            val = 0.0
            val_dx = 0.0
            val_dy = 0.0
            for g in range(n_splines):
                w = weights[g]
                wx = dw_dx[g]
                wy = dw_dy[g]
                if w == 0.0 and wx == 0.0 and wy == 0.0:
                    continue
                row = grid_idx[g]
                s = 0.0
                s_dx = 0.0
                s_dy = 0.0
                for i in range(DEGREE + 1):
                    base = (lx - DEGREE + i) * nky1 + ly - DEGREE
                    sy = 0.0
                    sy_dy = 0.0
                    for j in range(DEGREE + 1):
                        c = coeffs[row, base + j]
                        sy += c * hy[j]
                        sy_dy += c * dhy[j]
                    s += hx[i] * sy
                    s_dx += dhx[i] * sy
                    s_dy += hx[i] * sy_dy
                val += w * s
                val_dx += wx * s - scale_x * w * s_dx
                val_dy += wy * s - scale_y * w * s_dy
            out[p] = val
            out_dx[p] = val_dx
            out_dy[p] = val_dy
