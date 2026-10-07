# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Gridded PSF models.
"""

import bisect
import copy
import itertools
import threading
from functools import cached_property

import numpy as np
from astropy.io import registry
from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import NDData
from scipy.interpolate import RectBivariateSpline

from photutils.psf._bispline import bispline_sum, bispline_sum_deriv
from photutils.psf._bispline_inputs import ONE_PLANE, UNIT_WEIGHT, ZERO_WEIGHT
from photutils.psf.model_io import (GriddedPSFModelRead, _get_metadata,
                                    _read_stdpsf, is_stdpsf, is_webbpsf,
                                    stdpsf_reader, webbpsf_reader)
from photutils.psf.model_plotting import (_ModelGridPlotter,
                                          _plot_grid_docstring)
from photutils.psf.utils import _copy_model_sharing_data, _out_of_grid_mask
from photutils.utils._parameters import as_pair

__all__ = ['GriddedPSFModel', 'STDPSFGrid']
__doctest_skip__ = ['STDPSFGrid']

# The instance attributes that hold the spline cache of a GriddedPSFModel
_SPLINE_CACHE_NAMES = ('_spline_knots', '_spline_coeffs', '_spline_filled',
                       '_spline_lock')


class GriddedPSFModel(Fittable2DModel):
    """
    A model representing a grid of 2D ePSF models.

    The ePSF models are defined at fiducial detector locations specified
    by their ``(x, y)`` detector coordinates. The fiducial locations
    must form a rectangular grid.

    This model has three parameters: an image intensity scaling factor
    (``flux``), which scales the input image, and two positional
    parameters (``x_0`` and ``y_0``), which specify the location of the
    feature in the coordinate grid where the model is evaluated.

    When evaluating this model, the input ``x`` and ``y`` arrays must
    have no more than two dimensions.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object containing the reference ePSF
        grid. Its ``data`` attribute must be a 3D `~numpy.ndarray` with
        shape ``(N_psf, ePSF_ny, ePSF_nx)``, where each plane is a 2D
        ePSF image. The x and y dimensions of each ePSF image must both
        be at least 4 pixels. ``N_psf`` must not be 2 or 3. All values
        in ``data`` must be finite. The ePSF peak is assumed to be
        centered in each input image. See the Notes section for details
        on the required normalization of the input ePSF images.

        If ``N_psf`` is 1, the single ePSF image is used at all detector
        positions. This is equivalent to using `~photutils.psf.ImagePSF`
        with the same ePSF image.

        The ``meta`` attribute must be a dictionary containing:

        * ``'grid_xypos'``: A sequence of the fiducial ``(x, y)``
          detector coordinates for each reference ePSF. The order must
          match the first axis of ``data``. That is, ``grid_xypos[i]``
          gives the detector coordinates of ``nddata.data[i]``. The
          coordinates must form a rectangular grid.

        * ``'oversampling'``: The integer oversampling factor(s) of the
          input ePSF images. If a scalar is provided, it is applied to
          both axes. If two values are provided, they must be in ``(y,
          x)`` order.

        The ``meta`` dictionary may also contain additional metadata,
        such as the telescope, instrument, detector, or filter.

    flux : float, optional
        The flux scaling factor. This corresponds to the total
        source flux, assuming the input ePSF images are properly
        normalized.

    x_0, y_0 : float, optional
        The ``(x, y)`` coordinates of the ePSF peak in the output
        coordinate grid where the model is evaluated.

    fill_value : float or `None`, optional
        The value used for points outside the input pixel grid. The
        default is 0.0. If `None`, a point outside the input pixel grid
        takes the value of the spline at the nearest point on the edge
        of the grid.

    Methods
    -------
    read(*args, **kwargs)
        Class method to create a `GriddedPSFModel`
        instance from a STDPSF FITS file. This method uses
        :func:`~photutils.psf.stdpsf_reader` with the provided
        parameters.

    See Also
    --------
    ImagePSF : A model for a single ePSF image.
    make_epsf_from_psf : Make an ePSF image from a sampled PSF.

    Notes
    -----
    The input images must be effective PSFs (ePSFs). Each value of an
    ePSF is the fraction of the source flux that falls in a whole
    detector pixel centered at that position relative to the source,
    even when the image is oversampled. The model interpolates the
    input images and does not integrate them over the detector pixels.
    Evaluating the model at a position ``(x, y)`` therefore gives the
    flux in a detector pixel centered at ``(x, y)``, which can be any
    fractional pixel position.

    Because each value is the flux in a whole detector pixel, the model
    values sum to ``flux`` only when the model is evaluated on a grid
    with a spacing of one detector pixel. That holds for any values of
    ``x_0`` and ``y_0``. On a finer grid the pixels overlap, and the
    sum is larger than ``flux`` by the ratio of the pixel area to the
    area of a grid cell.

    Oversampled images that are not ePSFs, such as PSFs sampled at the
    points of a fine grid or binned into subpixels, are not converted
    to ePSFs by this model. The model is then sharper than a source
    in the data. For an undersampled PSF, the sum of the model values
    over the detector pixels can also change with the subpixel position
    of the source. Such images should first be integrated over the
    area of a detector pixel centered at each of their grid points,
    which is what `make_epsf_from_psf` does.

    The fitted ``flux`` parameter represents the total source flux,
    provided the input ePSF images are properly normalized. The fitted
    flux is a multiplicative scale factor applied to the input ePSF
    after accounting for any oversampling.

    For a fully sampled ePSF (i.e., no oversampling), the sum of
    the ePSF values over an infinite grid is 1.0. Because ePSFs are
    represented by finite images in practice, the sum of the array
    values may be less than 1.0.

    For oversampled ePSFs, the normalization should instead be such
    that the sum of the array values over an infinite grid equals the
    product of the oversampling factors (e.g., ``oversampling**2`` when
    the oversampling is the same along both axes). Again, a finite image
    will generally have a smaller sum because it does not contain the
    full PSF wings.

    If the input ePSF image covers only a finite region of the PSF,
    correction factors based on the encircled or ensquared energy
    can be used to estimate the missing flux and obtain the proper
    normalization.

    Internally, the ePSF grid is reordered so that the reference ePSFs
    are sorted first by their y detector coordinate and then by their x
    detector coordinate.

    Each grid plane is interpolated with a bicubic spline
    (`~scipy.interpolate.RectBivariateSpline` with ``kx=ky=3`` and
    ``s=0``). The spline coefficients of each evaluated grid plane are
    cached on the model, in a float64 array of the same shape as the
    grid data, and shared across copies made with the `copy` method.
    When every grid plane has been evaluated, the cache roughly doubles
    the model's memory footprint for float64 grid data and triples it
    for float32 grid data. The cache is not included when the model is
    pickled, and a deep copy gets its own copy of it. The splines of
    the four bounding grid planes are evaluated together by a compiled
    kernel, which also computes the analytic partial derivatives used by
    `fit_deriv`.
    """

    flux = Parameter(description='Intensity scaling factor for the ePSF '
                     'model.', default=1.0)
    x_0 = Parameter(description='x position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)
    y_0 = Parameter(description='y position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)

    read = registry.UnifiedReadWriteMethod(GriddedPSFModelRead)

    def __init__(self, nddata, *, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, fill_value=0.0):

        self._data, self._grid_xypos = self._define_grid(nddata)
        self._meta = nddata.meta.copy()  # _meta to avoid the meta descriptor
        # Drop a stale user-supplied key. The grid shape is derived
        # from grid_xypos
        self._meta.pop('grid_shape', None)
        # The setter also stores the normalized (y, x) pair in meta
        self.oversampling = nddata.meta['oversampling']
        self.fill_value = fill_value

        self._xgrid = np.unique(self.grid_xypos[:, 0])  # sorted
        self._ygrid = np.unique(self.grid_xypos[:, 1])  # sorted
        self._grid_shape = (len(self._ygrid), len(self._xgrid))
        # Store the sorted grid positions so that meta always matches
        # the grid_xypos attribute, regardless of the input form
        self.meta['grid_xypos'] = self.grid_xypos

        self._init_spline_cache()

        super().__init__(flux, x_0, y_0)

    def _init_spline_cache(self):
        """
        Create the empty cache of the grid-plane splines.

        The cache holds the knot vectors, which are the same for every
        grid plane, and the spline coefficients of each grid plane.
        It is filled on first use by `_fill_spline_coefficients`. The
        containers are created here rather than lazily so that the model
        copies made with `copy` share them.

        The knots and the filled flags are kept in lists, not arrays. A
        list item is replaced atomically, also on free-threaded builds,
        so a thread that reads a filled flag or the knots without the
        lock sees values that are completely written.
        """
        n_grid, ny, nx = self._data.shape
        # A one-item holder for the (tx, ty) tuple of knot vectors
        self._spline_knots = [None]
        # A bicubic interpolating spline has as many coefficients as
        # data points
        self._spline_coeffs = np.empty((n_grid, ny * nx), dtype=float)
        self._spline_filled = [False] * n_grid
        self._spline_lock = threading.Lock()

    def __getstate__(self):
        """
        Return the model state for pickling, without the spline cache.

        The cache is as large as the grid data and can be rebuilt from
        it, so it is left out of pickles.
        """
        state = self.__dict__.copy()
        for name in _SPLINE_CACHE_NAMES:
            del state[name]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Discard the spline objects that are in the pickles of models
        # made by earlier versions, which cached them on the model.
        for name in ('_interpolator', '_deriv_interpolators'):
            self.__dict__.pop(name, None)
        self._init_spline_cache()

    def __copy__(self):
        """
        Return a shallow copy that shares the spline cache.

        A shallow copy is made from the pickle state by default, which
        leaves the spline cache out, so it is added to the copy.
        """
        new_model = object.__new__(self.__class__)
        new_model.__setstate__(self.__getstate__())
        for name in _SPLINE_CACHE_NAMES:
            new_model.__dict__[name] = self.__dict__[name]
        return new_model

    def __deepcopy__(self, memo):
        """
        Return a deep copy with its own copy of the spline cache.

        The coefficients of the planes that are already cached are
        copied, so the deep copy does not rebuild their splines. A
        compound model that contains this model is deep copied for every
        source in PSF photometry. The caches of the two models are
        independent afterward.

        The copy is made from the pickle state, so that the
        ``__getstate__`` and ``__setstate__`` methods of a subclass
        apply to deep copies as well.
        """
        new_model = object.__new__(self.__class__)
        memo[id(self)] = new_model
        new_model.__setstate__(copy.deepcopy(self.__getstate__(), memo))

        # A plane that is flagged as filled is completely written and
        # is never written again, so it can be copied while other
        # threads fill other planes.
        filled = list(self._spline_filled)
        for gidx, is_filled in enumerate(filled):
            if is_filled:
                new_model._spline_coeffs[gidx] = self._spline_coeffs[gidx]
        new_model._spline_filled[:] = filled
        # The knot vectors are never modified, so they are shared
        new_model._spline_knots[0] = self._spline_knots[0]
        return new_model

    @staticmethod
    def _validate_data(data):
        """
        Validate the input ePSF data.

        Parameters
        ----------
        data : `~astropy.nddata.NDData`
            The input NDData object containing the ePSF data.

        Raises
        ------
        TypeError
            If the input data is not an NDData instance.
        ValueError
            If the input data is not a 3D numpy ndarray or if the input
            data contains NaNs or infs.
        """
        if not isinstance(data, NDData):
            msg = 'data must be an NDData instance'
            raise TypeError(msg)

        if data.data.ndim != 3:
            msg = 'The NDData data attribute must be a 3D numpy ndarray'
            raise ValueError(msg)

        if not np.all(np.isfinite(data.data)):
            msg = 'All elements of input data must be finite'
            raise ValueError(msg)

        if data.data.shape[0] in (2, 3):
            msg = 'The number of ePSFs must not be 2 or 3'
            raise ValueError(msg)

        # The minimum number of data points required is 4 along each
        # axis. This is because RectBivariateSpline requires at least 4
        # points along each axis for cubic spline interpolation (kx=3,
        # ky=3).
        if np.any(np.array(data.data.shape[1:]) < 4):
            msg = ('The length of the PSF x and y axes must both be at '
                   'least 4')
            raise ValueError(msg)

        if 'oversampling' not in data.meta:
            msg = "'oversampling' must be in the nddata meta dictionary"
            raise ValueError(msg)

    @staticmethod
    def _is_rectangular_grid(grid_xypos):
        """
        Determine if the given (x, y) pixel positions form an axis-
        aligned rectangular grid.

        Spacing does not need to be uniform along x or y, but there must
        be at least two unique x and y values, and all combinations of x
        and y must be present.

        Parameters
        ----------
        grid_xypos : (N, 2) array of (x, y) pairs
            The fiducial (x, y) positions of the ePSFs.

        Returns
        -------
        result : bool
            Returns `True` if the input ``grid_xypos`` forms a
            rectangular grid.
        """
        # Fewer than 4 positions cannot form a 2D rectangular grid
        if len(grid_xypos) < 4:
            return False

        x_vals = np.unique(grid_xypos[:, 0])  # sorted
        y_vals = np.unique(grid_xypos[:, 1])  # sorted
        # Must have at least 2 unique x and y values to form a 2D grid
        if len(x_vals) < 2 or len(y_vals) < 2:
            return False

        expected_points = {(x, y) for x in x_vals for y in y_vals}
        return set(map(tuple, grid_xypos)) == expected_points

    def _validate_grid(self, data):
        """
        Validate the input ePSF grid.

        Parameters
        ----------
        data : `~astropy.nddata.NDData`
            The input NDData object containing the ePSF data.

        Raises
        ------
        ValueError
            If the input grid_xypos does not form a rectangular grid.
        """
        try:
            grid_xypos = np.asarray(data.meta['grid_xypos'])
        except KeyError as exc:
            msg = "'grid_xypos' must be in the nddata meta dictionary"
            raise ValueError(msg) from exc

        if len(grid_xypos) != data.data.shape[0]:
            msg = ('The length of grid_xypos must match the number of '
                   'input ePSFs')
            raise ValueError(msg)

        if len({tuple(pos) for pos in grid_xypos}) != len(grid_xypos):
            msg = 'grid_xypos must not contain duplicate positions'
            raise ValueError(msg)

        if len(grid_xypos) != 1 and not self._is_rectangular_grid(grid_xypos):
            msg = ('grid_xypos must form a rectangular grid, i.e., there '
                   'must be at least two unique x and y positions and '
                   'every combination of them must be present')
            raise ValueError(msg)

    def _define_grid(self, nddata):
        """
        Sort the input ePSF data into a rectangular grid where the ePSFs
        are sorted first by y and then by x.

        Parameters
        ----------
        nddata : `~astropy.nddata.NDData`
            The input NDData object containing the ePSF data.

        Returns
        -------
        data : 3D `~numpy.ndarray`
            The 3D array of ePSFs.
        grid_xypos : array of (x, y) pairs
            The (x, y) positions of the ePSFs, sorted first by y and
            then by x.
        """
        self._validate_data(nddata)
        self._validate_grid(nddata)

        grid_xypos = np.asarray(nddata.meta['grid_xypos'])
        # Sort by y and then by x (last key is primary)
        idx = np.lexsort((grid_xypos[:, 0], grid_xypos[:, 1]))
        return nddata.data[idx], grid_xypos[idx]

    def __str__(self):
        keywords = []
        oversampling = tuple(int(value) for value in self.oversampling)

        keys = ('STDPSF', 'instrument', 'detector', 'filter')
        for key in keys:
            if key in self.meta:
                name = key.capitalize() if key != 'STDPSF' else key
                keywords.append((name, self.meta[key]))

        keywords.extend([('Number of PSFs', len(self.grid_xypos)),
                         ('Grid shape', self.grid_shape),
                         ('Grid positions', self.grid_xypos),
                         ('PSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', oversampling),
                         ('Fill Value', self.fill_value)])

        with np.printoptions(threshold=25, edgeitems=5):
            return self._format_str(keywords=keywords)

    def __repr__(self):
        kwargs = {'oversampling': self.oversampling.tolist(),
                  'fill_value': self.fill_value}
        return self._format_repr(args=[], kwargs=kwargs)

    @property
    def data(self):
        """
        The 3D array of ePSFs.

        The shape is ``(N_psf, ePSF_ny, ePSF_nx)``.
        """
        return self._data

    @property
    def grid_xypos(self):
        """
        The (x, y) positions of the ePSFs.

        The order of positions should match the first axis of the 3D
        `~numpy.ndarray` of ePSFs. In other words, ``grid_xypos[i]``
        should be the (x, y) position of the reference ePSF defined in
        ``nddata.data[i]``. The grid positions must form a rectangular
        grid.
        """
        return self._grid_xypos

    @property
    def grid_shape(self):
        """
        The ``(ny, nx)`` shape of the ePSF grid.
        """
        return self._grid_shape

    def copy(self):
        """
        Return a copy of this model where only the model parameters are
        copied.

        All other copied model attributes are references to the
        original model. This prevents copying the ePSF grid data, which
        may contain a large array.

        This method is useful if one is interested in only changing
        the model parameters in a model copy. It is used in the PSF
        photometry classes during model fitting.

        Use the `deepcopy` method if you want to copy all the model
        attributes, including the ePSF grid data.

        Returns
        -------
        result : `GriddedPSFModel`
            A copy of this model with only the model parameters copied.
        """
        new_model = _copy_model_sharing_data(self)

        # Give the copy its own meta dictionary so that setting the
        # oversampling on the copy does not mutate the original meta
        new_model._meta = dict(self._meta)

        return new_model

    def deepcopy(self):
        """
        Return a deep copy of this model.

        The deep copy gets its own copy of the cached spline
        coefficients, so it does not rebuild the splines that this model
        already has.

        Returns
        -------
        result : `GriddedPSFModel`
            A deep copy of this model.
        """
        return copy.deepcopy(self)

    @property
    def oversampling(self):
        """
        The integer oversampling factor(s) of the input ePSF images.

        If ``oversampling`` is a scalar then it will be used for both
        axes. If ``oversampling`` has two elements, they must be in
        ``(y, x)`` order.
        """
        return self._oversampling

    @oversampling.setter
    def oversampling(self, value):
        """
        Set the oversampling factor(s) of the input ePSF images.

        Parameters
        ----------
        value : int or tuple of int
            The integer oversampling factor(s) of the input ePSF images.
            If ``oversampling`` is a scalar then it will be used for both
            axes. If ``oversampling`` has two elements, they must be in
            ``(y, x)`` order.
        """
        self._oversampling = as_pair('oversampling', value, lower_bound=(0, 0))
        # Keep meta in sync with the normalized (y, x) pair
        self.meta['oversampling'] = tuple(int(val)
                                          for val in self._oversampling)

    def _calc_bounding_box(self):
        """
        Return a bounding box defining the limits of the model.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        dy, dx = np.array(self.data.shape[1:]) / 2 / self.oversampling
        return ((self.y_0 - dy, self.y_0 + dy), (self.x_0 - dx, self.x_0 + dx))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from itertools import product
        >>> import numpy as np
        >>> from astropy.nddata import NDData
        >>> from photutils.psf import GaussianPSF, GriddedPSFModel
        >>> psfs = []
        >>> yy, xx = np.mgrid[0:101, 0:101]
        >>> for i in range(16):
        ...     theta = np.deg2rad(i * 10.0)
        ...     gmodel = GaussianPSF(flux=1, x_0=50, y_0=50, x_fwhm=10,
        ...                          y_fwhm=5, theta=theta)
        ...     psfs.append(gmodel(xx, yy))
        >>> xgrid = [0, 40, 160, 200]
        >>> ygrid = [0, 60, 140, 200]
        >>> meta = {}
        >>> meta['grid_xypos'] = list(product(xgrid, ygrid))
        >>> meta['oversampling'] = 4
        >>> nddata = NDData(psfs, meta=meta)
        >>> model = GriddedPSFModel(nddata, flux=1, x_0=0, y_0=0)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-12.625, upper=12.625)
                y: Interval(lower=-12.625, upper=12.625)
            }
            model=GriddedPSFModel(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box()

    @cached_property
    def origin(self):
        """
        The (x, y) pixel coordinates, as a 1D `~numpy.ndarray`, of the
        origin of the coordinate system within the model image.
        """
        # data.shape is (N_psf, ePSF_ny, ePSF_nx). The leading axis is
        # excluded so that the result is the (x, y) image center
        xyorigin = (np.array(self.data.shape[1:]) - 1) / 2
        return xyorigin[::-1]

    @cached_property
    def _interp_xyidx(self):
        """
        The x and y indices for the interpolator.
        """
        xidx = np.arange(self.data.shape[2])
        yidx = np.arange(self.data.shape[1])
        return xidx, yidx

    def _calc_interpolator(self, grid_idx):
        """
        Calculate the `~scipy.interpolate.RectBivariateSpline`
        interpolator for an input ePSF image at the given reference (x,
        y) position.

        Parameters
        ----------
        grid_idx : int
            The index of the ePSF image in the reference grid.

        Returns
        -------
        interp : `~scipy.interpolate.RectBivariateSpline`
            The interpolator for the input ePSF image.
        """
        # RectBivariateSpline expects the data to be in (x, y) axis order
        data = self.data[grid_idx]
        return RectBivariateSpline(*self._interp_xyidx, data.T, kx=3, ky=3,
                                   s=0)

    def _fill_spline_coefficients(self, grid_idx, *weights):
        """
        Ensure that the spline knots and the spline coefficients of the
        given grid planes are in the spline cache.

        Only the knots and coefficients are kept. The spline object that
        computes them holds its own copy of the coefficients and is
        discarded.

        The spline of a plane is not built if all of its weights are
        zero, because the kernels skip such a plane.

        Several threads may build the spline of the same plane at the
        same time, but only the first one stores it. A plane that is
        flagged as filled is never written again, so the threads that
        evaluate it need no lock.

        Parameters
        ----------
        grid_idx : `~numpy.ndarray`
            The indices of the grid planes.

        *weights : `~numpy.ndarray`
            The weights of the planes that the kernel is called with
            (the bilinear weights and, for the derivative kernel, their
            derivatives).
        """
        for i, gidx in enumerate(grid_idx):
            gidx = int(gidx)
            if self._spline_filled[gidx]:
                continue
            if not any(weight[i] != 0.0 for weight in weights):
                continue
            interp = self._calc_interpolator(gidx)
            tx, ty = interp.get_knots()
            coeffs = interp.get_coeffs()
            with self._spline_lock:
                if self._spline_filled[gidx]:
                    continue
                if self._spline_knots[0] is None:
                    # Every grid plane has the same shape, so the knots
                    # are the same for all of them
                    self._spline_knots[0] = (np.ascontiguousarray(tx),
                                             np.ascontiguousarray(ty))
                self._spline_coeffs[gidx] = coeffs
                self._spline_filled[gidx] = True

    def _bounding_weights(self, x_0, y_0, *, derivs=False):
        """
        Return the grid indices, bilinear weights, and weight
        derivatives of the grid planes that bound a model position.

        For a single-plane grid, the one plane has unit weight and zero
        weight derivatives.

        Parameters
        ----------
        x_0, y_0 : float
            The (x, y) position of the model.

        derivs : bool, optional
            Whether to calculate the weight derivatives.

        Returns
        -------
        grid_idx, weights, dw_dx, dw_dy : `~numpy.ndarray` or `None`
            The plane indices and their weights and weight derivatives
            with respect to the model x and y positions. The weight
            derivatives are `None` if ``derivs`` is `False`.
        """
        if self.data.shape[0] == 1:
            dw_dx = ZERO_WEIGHT if derivs else None
            return ONE_PLANE, UNIT_WEIGHT, dw_dx, dw_dx
        grid_idx, grid_xy = self._find_bounding_points(x_0, y_0)
        weights = self._calc_bilinear_weights(x_0, y_0, grid_xy)
        if not derivs:
            return grid_idx, weights, None, None
        dw_dx, dw_dy = self._calc_bilinear_weight_derivs(x_0, y_0, grid_xy)
        return grid_idx, weights, dw_dx, dw_dy

    @cached_property
    def _xgrid_list(self):
        """
        A plain Python list of the sorted unique x grid positions.

        This is used for fast scalar lookups with `bisect.bisect_right`,
        which is faster than `numpy.searchsorted` for scalar inputs.
        """
        return [float(v) for v in self._xgrid]

    @cached_property
    def _ygrid_list(self):
        """
        A plain Python list of the sorted unique y grid positions.

        This is used for fast scalar lookups with `bisect.bisect_right`,
        which is faster than `numpy.searchsorted` for scalar inputs.
        """
        return [float(v) for v in self._ygrid]

    @cached_property
    def _bounding_lookup(self):
        """
        A precomputed lookup table mapping grid-cell indices to the
        source indices of the four bounding ePSF models.

        The array has shape ``(nx - 1, ny - 1, 4)`` and dtype intp,
        where ``nx`` and ``ny`` are the number of unique x and y grid
        positions, respectively. For a grid cell ``(xidx, yidx)``, the
        last axis contains the source indices of the four bounding ePSFs
        in the order (lower-left, lower-right, upper-left, upper-right).

        Precomputing this table avoids the repeated `numpy.where`
        searches over ``grid_xypos`` that would otherwise be performed
        for every model evaluation.
        """
        nx = len(self._xgrid)
        ny = len(self._ygrid)

        # Map each reference (x, y) grid position to its source index.
        # The grid is rectangular, so every corner is present.
        pos_to_idx = {(float(x), float(y)): idx
                      for idx, (x, y) in enumerate(self.grid_xypos)}

        # The spline kernels take the indices as intp
        out = np.empty((nx - 1, ny - 1, 4), dtype=np.intp)
        for ix in range(nx - 1):
            x0 = float(self._xgrid[ix])
            x1 = float(self._xgrid[ix + 1])
            for iy in range(ny - 1):
                y0 = float(self._ygrid[iy])
                y1 = float(self._ygrid[iy + 1])
                # Corners in (lower-left, lower-right, upper-left,
                # upper-right) order
                out[ix, iy] = (pos_to_idx[(x0, y0)], pos_to_idx[(x1, y0)],
                               pos_to_idx[(x0, y1)], pos_to_idx[(x1, y1)])
        return out

    def _find_bounding_points(self, x, y):
        """
        Find the grid indices and reference (x, y) points of the four
        bounding grid points for a given (x, y) coordinate.

        If the point is outside the grid, the nearest grid cell is
        selected.

        This method is a scalar-only fast path. ``x`` and ``y`` must be
        scalar values (the model ``x_0`` and ``y_0`` positions), which
        is always the case when called from ``_calc_model_values``.

        Parameters
        ----------
        x, y : float
            The scalar (x_0, y_0) position of the model.

        Returns
        -------
        grid_idx : `~numpy.ndarray`
            The indices of the four bounding points in the sorted
            grid. The order is lower-left, lower-right, upper-left,
            upper-right.

        grid_xy : tuple of 4 float
            The x and y coordinates of the four bounding points. The
            order is left, right, bottom, top.
        """
        # Scalar fast path: use bisect on the precomputed float list and
        # the precomputed corner lookup table for efficient grid-cell
        # indexing. This avoids the numpy call overhead (searchsorted,
        # clip, and where) that dominates for scalar inputs. The indices
        # are clamped so that out-of-grid inputs select the nearest grid
        # cell.
        nx = len(self._xgrid)
        ny = len(self._ygrid)
        xidx = bisect.bisect_right(self._xgrid_list, float(x)) - 1
        yidx = bisect.bisect_right(self._ygrid_list, float(y)) - 1
        if xidx < 0:
            xidx = 0
        elif xidx > nx - 2:
            xidx = nx - 2
        if yidx < 0:
            yidx = 0
        elif yidx > ny - 2:
            yidx = ny - 2

        # The coordinates are taken from the float lists so that the
        # bilinear weights computed from them are float64 for grid
        # positions of any dtype, which the spline kernels require. They
        # are returned as Python floats because the arithmetic of the
        # weights is faster with them than with NumPy scalars.
        x0 = self._xgrid_list[xidx]
        x1 = self._xgrid_list[xidx + 1]
        y0 = self._ygrid_list[yidx]
        y1 = self._ygrid_list[yidx + 1]
        grid_idx = self._bounding_lookup[xidx, yidx]
        return grid_idx, (x0, x1, y0, y1)

    def _calc_bilinear_weights(self, xi, yi, grid_xy):
        """
        Calculate the bilinear interpolation weights for a given (xi,
        yi) coordinate and the four bounding grid points.

        This method is a scalar-only fast path. ``xi`` and ``yi`` must
        be scalar values (the model ``x_0`` and ``y_0`` positions).

        Parameters
        ----------
        xi, yi : float
            The scalar (x_0, y_0) position of the model.

        grid_xy : sequence of 4 float
            The x and y coordinates of the four bounding points. The
            order is left, right, bottom, top.

        Returns
        -------
        weights : `~numpy.ndarray`
            The bilinear interpolation weights for the four bounding
            points. The order is lower-left, lower-right, upper-left,
            upper-right.
        """
        x0, x1, y0, y1 = grid_xy

        # Scalar fast path: clamp the coordinates to the grid bounds
        # using scalar conditionals instead of numpy.clip, which avoids
        # the numpy call overhead for scalar inputs.
        xi = float(xi)
        yi = float(yi)
        if xi < x0:
            xi = x0
        elif xi > x1:
            xi = x1
        if yi < y0:
            yi = y0
        elif yi > y1:
            yi = y1

        # Weights of the four bounding points for bilinear
        # interpolation. The order is lower-left, lower-right,
        # upper-left, upper-right.
        inv_norm = 1.0 / ((x1 - x0) * (y1 - y0))
        ll = (x1 - xi) * (y1 - yi) * inv_norm
        lr = (xi - x0) * (y1 - yi) * inv_norm
        ul = (x1 - xi) * (yi - y0) * inv_norm
        ur = (xi - x0) * (yi - y0) * inv_norm
        return np.array((ll, lr, ul, ur))

    def _calc_bilinear_weight_derivs(self, xi, yi, grid_xy):
        """
        Calculate the partial derivatives of the bilinear interpolation
        weights with respect to the (xi, yi) coordinate.

        This method is a scalar-only fast path. ``xi`` and ``yi`` must
        be scalar values (the model ``x_0`` and ``y_0`` positions).

        `_calc_bilinear_weights` clamps the coordinates to the grid
        cell, so the weights are constant outside the cell and the
        corresponding derivatives are zero there.

        Parameters
        ----------
        xi, yi : float
            The scalar (x_0, y_0) position of the model.

        grid_xy : sequence of 4 float
            The x and y coordinates of the four bounding points. The
            order is left, right, bottom, top.

        Returns
        -------
        dw_dx, dw_dy : `~numpy.ndarray`
            The partial derivatives of the bilinear weights for the four
            bounding points with respect to ``xi`` and ``yi``. The order
            is lower-left, lower-right, upper-left, upper-right.
        """
        x0, x1, y0, y1 = grid_xy
        xi = float(xi)
        yi = float(yi)
        inv_norm = 1.0 / ((x1 - x0) * (y1 - y0))

        # Clamp the coordinates as in _calc_bilinear_weights. The
        # clamped values are used in the derivatives along the other
        # axis.
        x_inside = x0 <= xi <= x1
        y_inside = y0 <= yi <= y1
        xc = min(max(xi, x0), x1)
        yc = min(max(yi, y0), y1)

        if x_inside:
            dw_dx = np.array((-(y1 - yc), (y1 - yc),
                              -(yc - y0), (yc - y0))) * inv_norm
        else:
            dw_dx = np.zeros(4)
        if y_inside:
            dw_dy = np.array((-(x1 - xc), -(xc - x0),
                              (x1 - xc), (xc - x0))) * inv_norm
        else:
            dw_dy = np.zeros(4)
        return dw_dx, dw_dy

    def _calc_model_values(self, x_0, y_0, xi, yi):
        """
        Calculate the ePSF model at a given (x_0, y_0) model coordinate
        and the input (xi, yi) coordinate.

        Parameters
        ----------
        x_0, y_0 : float
            The (x, y) position of the model.

        xi, yi : `~numpy.ndarray`
            The input (x, y) coordinates at which the model is
            evaluated. The two arrays must have the same shape.

        Returns
        -------
        result : `~numpy.ndarray`
            The interpolated ePSF model at the input (x_0, y_0)
            coordinate.
        """
        grid_idx, weights, _, _ = self._bounding_weights(x_0, y_0)
        self._fill_spline_coefficients(grid_idx, weights)
        tx, ty = self._spline_knots[0]
        xi = np.ascontiguousarray(xi, dtype=float)
        yi = np.ascontiguousarray(yi, dtype=float)
        result = np.empty(xi.shape, dtype=float)
        bispline_sum(tx, ty, self._spline_coeffs, grid_idx, weights,
                     xi.ravel(), yi.ravel(), result.ravel())
        return result

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Calculate the ePSF model at the input coordinates for the given
        model parameters.

        Parameters
        ----------
        x, y : float or `~numpy.ndarray`
            The x and y positions at which to evaluate the model.

        flux : float
            The flux scaling factor for the model.

        x_0, y_0 : float
            The (x, y) position of the model.

        Returns
        -------
        evaluated_model : `~numpy.ndarray`
            The evaluated model.
        """
        # Promote scalar inputs to 1D arrays so that the interpolator
        # returns an array that supports masked assignment below,
        # regardless of the scipy version
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.ndim > 2:
            msg = 'x and y must be 1D or 2D'
            raise ValueError(msg)

        # The base Model.__call__() method converts scalar inputs to
        # size-1 arrays before calling evaluate(), but we need scalar
        # values for the interpolator.
        if not np.isscalar(x_0):
            x_0 = x_0[0]
        if not np.isscalar(y_0):
            y_0 = y_0[0]

        # Now evaluate the ePSF at the (x_0, y_0) subpixel position on
        # the input (x, y) values.
        xi = self.oversampling[1] * (np.asarray(x, dtype=float) - x_0)
        yi = self.oversampling[0] * (np.asarray(y, dtype=float) - y_0)
        xi += self.origin[0]
        yi += self.origin[1]
        if xi.shape != yi.shape:
            xi, yi = np.broadcast_arrays(xi, yi)

        evaluated_model = flux * self._calc_model_values(x_0, y_0, xi, yi)

        if self.fill_value is not None:
            # Set pixels that are outside the input pixel grid to the
            # fill_value to avoid extrapolation
            invalid = _out_of_grid_mask(xi, yi, self.data.shape[1:])
            evaluated_model[invalid] = self.fill_value

        return evaluated_model

    def fit_deriv(self, x, y, flux, x_0, y_0):
        """
        Calculate the partial derivatives of the ePSF model with respect
        to the model parameters.

        Providing this analytic Jacobian allows the fitter to avoid the
        finite-difference approximation, which requires additional model
        evaluations.

        The position derivatives include both the shift of the ePSF with
        the model position and the change of the bilinearly interpolated
        ePSF itself with the model position across the reference grid.

        Parameters
        ----------
        x, y : float or `~numpy.ndarray`
            The x and y positions at which to evaluate the model.

        flux : float
            The flux scaling factor for the model.

        x_0, y_0 : float
            The (x, y) position of the model.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to the
            ``flux``, ``x_0``, and ``y_0`` parameters.
        """
        # Promote scalar inputs to 1D arrays so that the interpolator
        # returns an array that supports masked assignment below,
        # regardless of the scipy version
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.ndim > 2:
            msg = 'x and y must be 1D or 2D'
            raise ValueError(msg)

        # The fitting machinery may pass the parameters as size-1
        # arrays, but we need scalar values for the interpolator.
        if not np.isscalar(x_0):
            x_0 = x_0[0]
        if not np.isscalar(y_0):
            y_0 = y_0[0]

        xi = self.oversampling[1] * (np.asarray(x, dtype=float) - x_0)
        yi = self.oversampling[0] * (np.asarray(y, dtype=float) - y_0)
        xi += self.origin[0]
        yi += self.origin[1]
        if xi.shape != yi.shape:
            xi, yi = np.broadcast_arrays(xi, yi)

        # The ePSF value contributes to the flux derivative through its
        # bilinear weight and to the position derivatives through the
        # weight derivatives. The chain rule adds the shift terms from
        # the spline partial derivatives (dxi/dx_0 = -oversampling[1],
        # dyi/dy_0 = -oversampling[0]). The kernel computes all of these
        # in one pass over the bounding planes.
        grid_idx, weights, dw_dx, dw_dy = self._bounding_weights(
            x_0, y_0, derivs=True)
        self._fill_spline_coefficients(grid_idx, weights, dw_dx, dw_dy)
        tx, ty = self._spline_knots[0]
        xi = np.ascontiguousarray(xi, dtype=float)
        yi = np.ascontiguousarray(yi, dtype=float)
        d_flux = np.empty(xi.shape, dtype=float)
        deriv_x = np.empty(xi.shape, dtype=float)
        deriv_y = np.empty(xi.shape, dtype=float)
        bispline_sum_deriv(tx, ty, self._spline_coeffs, grid_idx, weights,
                           dw_dx, dw_dy, float(self.oversampling[1]),
                           float(self.oversampling[0]), xi.ravel(),
                           yi.ravel(), d_flux.ravel(), deriv_x.ravel(),
                           deriv_y.ravel())

        d_x_0 = flux * deriv_x
        d_y_0 = flux * deriv_y

        if self.fill_value is not None:
            # Outside the input pixel grid the model is constant
            # (fill_value), so all derivatives are zero there
            invalid = _out_of_grid_mask(xi, yi, self.data.shape[1:])
            d_flux[invalid] = 0.0
            d_x_0[invalid] = 0.0
            d_y_0[invalid] = 0.0

        return [d_flux, d_x_0, d_y_0]

    @_plot_grid_docstring
    def plot_grid(self, *, ax=None, vmax_scale=None, peak_norm=False,
                  deltas=False, cmap='viridis', dividers=True,
                  divider_color='darkgray', divider_ls='-', figsize=None):
        plotter = _ModelGridPlotter(self)
        return plotter.plot_grid(ax=ax, vmax_scale=vmax_scale,
                                 peak_norm=peak_norm, deltas=deltas,
                                 cmap=cmap, dividers=dividers,
                                 divider_color=divider_color,
                                 divider_ls=divider_ls, figsize=figsize)


class STDPSFGrid:
    """
    Class to read and plot ePSF model grids stored in the STDPSF format.

    STDPSF files are FITS files containing a 3D array of ePSF models.
    The FITS header specifies the fiducial detector coordinates
    associated with each ePSF in the grid.

    For STDPSF files, the oversampling factor is assumed to be 4 along
    both axes.

    Parameters
    ----------
    filename : str or path-like
        The name or URL of a STDPSF FITS file.

    Examples
    --------
    >>> from photutils.psf import STDPSFGrid
    >>> psfgrid = STDPSFGrid('STDPSF_ACSWFC_F814W.fits')
    >>> fig = psfgrid.plot_grid()
    """

    # STDPSF files are assumed to have an oversampling factor of 4 along
    # both axes
    _default_oversampling = (4, 4)

    def __init__(self, filename):
        grid_data = _read_stdpsf(filename)
        xgrid = grid_data['xgrid']
        ygrid = grid_data['ygrid']

        # itertools.product iterates over the last input first
        grid_xypos = np.array([yx[::-1]
                               for yx in itertools.product(ygrid, xgrid)])

        # Try to get additional metadata from the filename because this
        # information is not currently available in the FITS headers.
        meta = _get_metadata(filename, None) or {}

        self._init_grid(grid_data['data'], grid_xypos,
                        (len(ygrid), len(xgrid)),
                        self._default_oversampling, meta)

    @classmethod
    def _from_asdf(cls, data, meta):
        """
        Create a `STDPSFGrid` from the contents of an ASDF file.

        Parameters
        ----------
        data : `~numpy.ndarray`
            A 3D array containing the ePSF grid.

        meta : dict
            A metadata dictionary. It must contain a ``'grid_xypos'``
            key holding an ``(N, 2)`` array of the fiducial ``(x, y)``
            detector coordinates and a ``'grid_shape'`` key holding the
            ``(ny, nx)`` shape of the grid. It may also contain an
            ``'oversampling'`` key. If absent, the default is ``(4,
            4)``. These three keys are consumed here and are not
            retained in the ``meta`` attribute.

        Returns
        -------
        result : `STDPSFGrid`
            The ePSF grid.
        """
        meta = dict(meta)
        for key in ('grid_xypos', 'grid_shape'):
            if key not in meta:
                msg = f'{key!r} must be in the meta dictionary'
                raise ValueError(msg)

        grid_xypos = np.asarray(meta.pop('grid_xypos'))
        grid_shape = meta.pop('grid_shape')
        oversampling = meta.pop('oversampling', cls._default_oversampling)

        obj = cls.__new__(cls)
        obj._init_grid(data, grid_xypos, grid_shape, oversampling, meta)
        return obj

    def _init_grid(self, data, grid_xypos, grid_shape, oversampling, meta):
        """
        Set the ePSF grid attributes.

        Parameters
        ----------
        data : `~numpy.ndarray`
            A 3D array containing the ePSF grid.

        grid_xypos : `~numpy.ndarray`
            An ``(N, 2)`` array of the fiducial ``(x, y)`` detector
            coordinates of each ePSF, ordered along y and then x.

        grid_shape : tuple of int
            The ``(ny, nx)`` shape of the ePSF grid.

        oversampling : int or array_like of int
            The integer oversampling factor(s) of the ePSF images.

        meta : dict
            The metadata dictionary.
        """
        self._data = data
        self._grid_xypos = grid_xypos
        self.meta = meta
        self._grid_shape = tuple(int(value) for value in grid_shape)

        # The grid axes are extracted from the first row and column
        # rather than with np.unique because a coordinate can be
        # repeated where two detectors abut (e.g., ACS/WFC).
        xypos = grid_xypos.reshape(*self._grid_shape, 2)
        self._xgrid = xypos[0, :, 0]
        self._ygrid = xypos[:, 0, 1]

        self._oversampling = as_pair('oversampling', oversampling,
                                     lower_bound=(0, 0))

    @property
    def data(self):
        """
        The 3D array of ePSFs.

        The shape is ``(N_psf, ePSF_ny, ePSF_nx)``.
        """
        return self._data

    @property
    def grid_xypos(self):
        """
        The (x, y) positions of the ePSFs.

        The order of positions matches the first axis of the 3D
        `~numpy.ndarray` of ePSFs. In other words, ``grid_xypos[i]``
        is the (x, y) position of the reference ePSF defined in
        ``data[i]``.
        """
        return self._grid_xypos

    @property
    def grid_shape(self):
        """
        The ``(ny, nx)`` shape of the ePSF grid.
        """
        return self._grid_shape

    @property
    def oversampling(self):
        """
        The integer oversampling factor(s) of the input ePSF images.

        Returns
        -------
        oversampling : `~numpy.ndarray`
            The oversampling factors in ``(y, x)`` order.
        """
        return self._oversampling

    @_plot_grid_docstring
    def plot_grid(self, *, ax=None, vmax_scale=None, peak_norm=False,
                  deltas=False, cmap='viridis', dividers=True,
                  divider_color='darkgray', divider_ls='-', figsize=None):
        plotter = _ModelGridPlotter(self)
        return plotter.plot_grid(ax=ax, vmax_scale=vmax_scale,
                                 peak_norm=peak_norm, deltas=deltas,
                                 cmap=cmap, dividers=dividers,
                                 divider_color=divider_color,
                                 divider_ls=divider_ls, figsize=figsize)

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        cls_info = []
        # Use int to avoid printing numpy int64 values in the string
        # representation
        oversampling = tuple(int(value) for value in self.oversampling)

        keys = ('STDPSF', 'instrument', 'detector', 'filter')
        for key in keys:
            if key in self.meta:
                name = key.capitalize() if key != 'STDPSF' else key
                cls_info.append((name, self.meta[key]))

        cls_info.extend([('Number of PSFs', len(self.grid_xypos)),
                         ('Grid shape', self.grid_shape),
                         ('PSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', oversampling)])

        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'{key}: {val}' for key, val in cls_info]

        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()


with registry.delay_doc_updates(GriddedPSFModel):
    registry.register_reader('stdpsf', GriddedPSFModel, stdpsf_reader)
    registry.register_identifier('stdpsf', GriddedPSFModel, is_stdpsf)
    registry.register_reader('webbpsf', GriddedPSFModel, webbpsf_reader)
    registry.register_identifier('webbpsf', GriddedPSFModel, is_webbpsf)
