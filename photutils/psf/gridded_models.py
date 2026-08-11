# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Gridded PSF models.
"""

import bisect
import copy
import itertools

import numpy as np
from astropy.io import registry
from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import NDData
from astropy.utils.decorators import lazyproperty
from scipy.interpolate import RectBivariateSpline

from photutils.psf.model_io import (GriddedPSFModelRead, _get_metadata,
                                    _read_stdpsf, is_stdpsf, is_webbpsf,
                                    stdpsf_reader, webbpsf_reader)
from photutils.psf.model_plotting import (_ModelGridPlotter,
                                          _plot_grid_docstring)
from photutils.utils._parameters import as_pair

__all__ = ['GriddedPSFModel', 'STDPSFGrid']
__doctest_skip__ = ['STDPSFGrid']


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
          match the first axis of ``data``; that is, ``grid_xypos[i]``
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

    fill_value : float, optional
        The value used for points outside the input pixel grid. The
        default is 0.0.

    Methods
    -------
    read(*args, **kwargs)
        Class method to create a `GriddedPSFModel`
        instance from a STDPSF FITS file. This method uses
        :func:`~photutils.psf.stdpsf_reader` with the provided
        parameters.

    Notes
    -----
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
        # The setter also stores the normalized (y, x) pair in meta
        self.oversampling = nddata.meta['oversampling']
        self.fill_value = fill_value

        self._xgrid = np.unique(self.grid_xypos[:, 0])  # sorted
        self._ygrid = np.unique(self.grid_xypos[:, 1])  # sorted
        self.meta['grid_shape'] = (len(self._ygrid), len(self._xgrid))
        # Store the sorted grid positions so that meta always matches
        # the grid_xypos attribute, regardless of the input form
        self.meta['grid_xypos'] = self.grid_xypos

        self._interpolator = {}

        super().__init__(flux, x_0, y_0)

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
                         ('Grid shape', self.meta['grid_shape']),
                         ('Grid positions', self.grid_xypos.tolist()),
                         ('PSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', oversampling),
                         ('Fill Value', self.fill_value)])

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
        newcls = object.__new__(self.__class__)

        for key, val in self.__dict__.items():
            if key in self.param_names:  # copy only the parameter values
                newcls.__dict__[key] = copy.copy(val)
            else:
                newcls.__dict__[key] = val

        return newcls

    def deepcopy(self):
        """
        Return a deep copy of this model.

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
        Set a bounding box defining the limits of the model.

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

    @lazyproperty
    def origin(self):
        """
        A 1D `~numpy.ndarray` (x, y) pixel coordinates within the
        model's 2D image of the origin of the coordinate system.
        """
        # data.shape is (N_psf, ePSF_ny, ePSF_nx); the leading axis is
        # excluded so that the result is the (x, y) image center
        xyorigin = (np.array(self.data.shape[1:]) - 1) / 2
        return xyorigin[::-1]

    @lazyproperty
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

        The resulting interpolator is cached in the `_interpolator`
        dictionary for reuse.

        Parameters
        ----------
        grid_idx : int
            The index of the ePSF image in the reference grid.

        Returns
        -------
        interp : `~scipy.interpolate.RectBivariateSpline`
            The interpolator for the input ePSF image.
        """
        # Check if the interpolator is already cached
        if grid_idx in self._interpolator:
            return self._interpolator[grid_idx]

        # RectBivariateSpline expects the data to be in (x, y) axis order
        data = self.data[grid_idx]
        interp = RectBivariateSpline(*self._interp_xyidx, data.T, kx=3, ky=3,
                                     s=0)

        # Cache the interpolator for reuse
        self._interpolator[grid_idx] = interp

        return interp

    @lazyproperty
    def _xgrid_list(self):
        """
        A plain Python list of the sorted unique x grid positions.

        This is used for fast scalar lookups with `bisect.bisect_right`,
        which is faster than `numpy.searchsorted` for scalar inputs.
        """
        return [float(v) for v in self._xgrid]

    @lazyproperty
    def _ygrid_list(self):
        """
        A plain Python list of the sorted unique y grid positions.

        This is used for fast scalar lookups with `bisect.bisect_right`,
        which is faster than `numpy.searchsorted` for scalar inputs.
        """
        return [float(v) for v in self._ygrid]

    @lazyproperty
    def _bounding_lookup(self):
        """
        A precomputed lookup table mapping grid-cell indices to the
        source indices of the four bounding ePSF models.

        The array has shape ``(nx - 1, ny - 1, 4)`` and dtype int64,
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

        out = np.empty((nx - 1, ny - 1, 4), dtype=np.int64)
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

        This method is a scalar-only fast path: ``x`` and ``y`` must be
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

        grid_xy : `~numpy.ndarray`
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
        x0 = self._xgrid[xidx]
        x1 = self._xgrid[xidx + 1]
        y0 = self._ygrid[yidx]
        y1 = self._ygrid[yidx + 1]
        grid_idx = self._bounding_lookup[xidx, yidx]
        grid_xy = np.array((x0, x1, y0, y1))
        return grid_idx, grid_xy

    def _calc_bilinear_weights(self, xi, yi, grid_xy):
        """
        Calculate the bilinear interpolation weights for a given (xi,
        yi) coordinate and the four bounding grid points.

        This method is a scalar-only fast path: ``xi`` and ``yi`` must
        be scalar values (the model ``x_0`` and ``y_0`` positions).

        Parameters
        ----------
        xi, yi : float
            The scalar (x_0, y_0) position of the model.

        grid_xy : `~numpy.ndarray`
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

    def _calc_model_values(self, x_0, y_0, xi, yi):
        """
        Calculate the ePSF model at a given (x_0, y_0) model coordinate
        and the input (xi, yi) coordinate.

        Parameters
        ----------
        x_0, y_0 : float
            The (x, y) position of the model.

        xi, yi : float
            The input (x, y) coordinates at which the model is
            evaluated.

        Returns
        -------
        result : float or `~numpy.ndarray`
            The interpolated ePSF model at the input (x_0, y_0)
            coordinate.
        """
        grid_idx, grid_xy = self._find_bounding_points(x_0, y_0)
        weights = self._calc_bilinear_weights(x_0, y_0, grid_xy)

        # Accumulate the weighted interpolator evaluations using a plain
        # Python loop. This avoids the overhead of building intermediate
        # arrays and fancy-indexing (np.array, np.where), which is
        # called once per model evaluation during fitting.
        result = 0.0
        for gidx, weight in zip(grid_idx, weights, strict=True):
            if weight == 0.0:
                continue
            interp = self._calc_interpolator(int(gidx))
            result += interp(xi, yi, grid=False) * weight

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

        if self.data.shape[0] == 1:
            # If there is only one ePSF, we do not need to perform
            # the bilinear interpolation.
            evaluated_model = flux * self._calc_interpolator(0)(xi, yi,
                                                                grid=False)
        else:
            evaluated_model = flux * self._calc_model_values(x_0, y_0, xi, yi)

        if self.fill_value is not None:
            # Set pixels that are outside the input pixel grid to the
            # fill_value to avoid extrapolation; these bounds match the
            # RegularGridInterpolator bounds.
            ny, nx = self.data.shape[1:]
            invalid = (xi < 0) | (xi > nx - 1) | (yi < 0) | (yi > ny - 1)
            evaluated_model[invalid] = self.fill_value

        return evaluated_model

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
            ``'oversampling'`` key; if absent, the default is ``(4,
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
