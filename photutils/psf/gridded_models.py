# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define gridded PSF models.
"""

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
from photutils.psf.model_plotting import ModelGridPlotMixin
from photutils.utils._parameters import as_pair

__all__ = ['GriddedPSFModel', 'STDPSFGrid']
__doctest_skip__ = ['STDPSFGrid']


class GriddedPSFModel(ModelGridPlotMixin, Fittable2DModel):
    """
    A model for a grid of 2D ePSF models.

    The ePSF models are defined at fiducial detector locations and are
    bilinearly interpolated to calculate an ePSF model at an arbitrary
    (x, y) detector position. The fiducial detector locations are must
    form a rectangular grid.

    The model has three model parameters: an image intensity scaling
    factor (``flux``) which is applied to the input image, and two
    positional parameters (``x_0`` and ``y_0``) indicating the location
    of a feature in the coordinate grid on which the model is evaluated.

    When evaluating this model, it cannot be called with x and y arrays
    that have greater than 2 dimensions.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object containing the grid of
        reference ePSF arrays. The data attribute must contain a 3D
        `~numpy.ndarray` containing a stack of the 2D ePSFs with a shape
        of ``(N_psf, ePSF_ny, ePSF_nx)``. The length of the x and y
        axes must both be at least 4. ``N_psf`` must not be 2 or 3. All
        elements of the input image data must be finite. The PSF peak is
        assumed to be located at the center of the input image. Please
        see the Notes section below for details on the normalization of
        the input image data.

        If ``N_psf`` is 1, the model will be evaluated using the single
        ePSF image at every (x, y) position. This is equivalent to using
        the `~photutils.psf.ImagePSF` model with the single ePSF.

        The meta attribute must be dictionary containing the following:

        * ``'grid_xypos'``: A list of the (x, y) grid positions of
          each reference ePSF. The order of positions should match the
          first axis of the 3D `~numpy.ndarray` of ePSFs. In other words,
          ``grid_xypos[i]`` should be the (x, y) position of the reference
          ePSF defined in ``nddata.data[i]``. The grid positions must form
          a rectangular grid.

        * ``'oversampling'``: The integer oversampling factor(s) of the
          input ePSF images. If ``oversampling`` is a scalar then it will
          be used for both axes. If ``oversampling`` has two elements,
          they must be in ``(y, x)`` order.

        The meta attribute may contain other properties such as the
        telescope, instrument, detector, and filter of the ePSF.

    flux : float, optional
        The flux scaling factor for the model. This is the total flux
        of the source, assuming the input ePSF images are properly
        normalized.

    x_0, y_0 : float, optional
        The (x, y) position of the PSF peak in the image in the output
        coordinate grid on which the model is evaluated.

    fill_value : float, optional
        The value to use for points outside of the input pixel grid.
        The default is 0.0.

    Methods
    -------
    read(*args, **kwargs)
        Class method to create a `GriddedPSFModel`
        instance from a STDPSF FITS file. This method uses
        :func:`~photutils.psf.stdpsf_reader` with the provided
        parameters.

    Notes
    -----
    The fitted PSF model flux represents the total flux of the source,
    assuming the input image was properly normalized. This flux is
    determined as a multiplicative scale factor applied to the input
    image PSF, after accounting for any oversampling. Theoretically,
    the sum of all values in the PSF image over an infinite grid should
    equal 1.0 (assuming no oversampling). However, when the PSF is
    represented over a finite region, the sum of the values may be less
    than 1.0. For oversampled PSF images, the normalization should be
    adjusted so that the sum of the array values equals the product
    of the oversampling factors (e.g., oversampling squared if the
    oversampling is the same along both axes). If the input image only
    covers a finite region of the PSF, the sum may again be less than
    the product of the oversampling factors. Correction factors based on
    the encircled or ensquared energy of the PSF can be used to estimate
    the proper scaling for the finite region of the input PSF image and
    ensure correct flux normalization.

    Internally, the grid of ePSFs will be arranged and stored such that
    it is sorted first by the y reference pixel coordinate and then by
    the x reference pixel coordinate.
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
        self._oversampling = as_pair('oversampling',
                                     nddata.meta['oversampling'],
                                     lower_bound=(0, 1))
        self.fill_value = fill_value

        self._xgrid = np.unique(self.grid_xypos[:, 0])  # sorted
        self._ygrid = np.unique(self.grid_xypos[:, 1])  # sorted
        self.meta['grid_shape'] = (len(self._ygrid), len(self._xgrid))

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

        # this is required by RectBivariateSpline for kx=3, ky=3
        if np.any(np.array(data.data.shape[1:]) < 4):
            msg = ('The length of the PSF x and y axes must both be at '
                   'least 4')
            raise ValueError(msg)

        if 'oversampling' not in data.meta:
            msg = '"oversampling" must be in the nddata meta dictionary'
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
        if len(grid_xypos) < 4:  # pragma: no cover
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
            grid_xypos = np.array(data.meta['grid_xypos'])
        except KeyError as exc:
            msg = '"grid_xypos" must be in the nddata meta dictionary'
            raise ValueError(msg) from exc

        if len(grid_xypos) != data.data.shape[0]:
            msg = ('The length of grid_xypos must match the number of '
                   'input ePSFs')
            raise ValueError(msg)

        if len(grid_xypos) != 1 and not self._is_rectangular_grid(grid_xypos):
            msg = 'grid_xypos must form a rectangular grid'
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

        grid_xypos = np.array(nddata.meta['grid_xypos'])
        # sort by y and then by x (last key is primary)
        idx = np.lexsort((grid_xypos[:, 0], grid_xypos[:, 1]))
        return nddata.data[idx], grid_xypos[idx]

    def _cls_info(self):
        cls_info = []

        keys = ('STDPSF', 'instrument', 'detector', 'filter', 'grid_shape')
        for key in keys:
            if key in self.meta:
                name = key.capitalize() if key != 'STDPSF' else key
                cls_info.append((name, self.meta[key]))

        cls_info.extend([('Number of PSFs', len(self.grid_xypos)),
                         ('PSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', tuple(self.oversampling))])
        return cls_info

    def __str__(self):
        return self._format_str(keywords=self._cls_info())

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

        Use the `deepcopy` method if you want to copy all of the model
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
        self._oversampling = as_pair('oversampling', value, lower_bound=(0, 1))

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
        >>> model.bounding_box  # doctest: +FLOAT_CMP
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
        xyorigin = (np.array(self.data.shape) - 1) / 2
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
        # check if the interpolator is already cached
        if grid_idx in self._interpolator:
            return self._interpolator[grid_idx]

        # RectBivariateSpline expects the data to be in (x, y) axis order
        data = self.data[grid_idx]
        interp = RectBivariateSpline(*self._interp_xyidx, data.T, kx=3, ky=3,
                                     s=0)

        # cache the interpolator for reuse
        self._interpolator[grid_idx] = interp

        return interp

    def _find_bounding_points(self, x, y):
        """
        Find the grid indices and reference (x, y) points of the four
        bounding grid points for a given (x, y) coordinate.

        If the point is outside the grid, the nearest grid points are
        selected. The input grid points do not need to be sorted.

        Parameters
        ----------
        x, y : float
            The (x_0, y_0) position of the model.

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
        # Find the insertion indices for x and y in the sorted grids
        xidx = np.searchsorted(self._xgrid, x) - 1
        yidx = np.searchsorted(self._ygrid, y) - 1

        # Clip the indices to valid ranges
        xidx = np.clip(xidx, 0, len(self._xgrid) - 2)
        yidx = np.clip(yidx, 0, len(self._ygrid) - 2)

        # Find the four bounding points in the sorted grid
        # (x0, y0) is the lower-left corner of the grid
        # (x1, y1) is the upper-right corner of the grid
        x0, x1 = self._xgrid[xidx], self._xgrid[xidx + 1]
        y0, y1 = self._ygrid[yidx], self._ygrid[yidx + 1]

        # Find the indices of these points in grid_xypos
        xcoords, ycoords = self.grid_xypos.T
        lower_left = np.where((xcoords == x0) & (ycoords == y0))[0][0]
        lower_right = np.where((xcoords == x1) & (ycoords == y0))[0][0]
        upper_left = np.where((xcoords == x0) & (ycoords == y1))[0][0]
        upper_right = np.where((xcoords == x1) & (ycoords == y1))[0][0]

        grid_idx = np.array((lower_left, lower_right, upper_left, upper_right))
        grid_xy = np.array((x0, x1, y0, y1))

        return grid_idx, grid_xy

    def _calc_bilinear_weights(self, xi, yi, grid_xy):
        """
        Calculate the bilinear interpolation weights for a given (xi,
        yi) coordinate and the four bounding grid points.

        Parameters
        ----------
        xi, yi : float
            The (x_0, y_0) position of the model.

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

        xi = np.clip(xi, x0, x1)
        yi = np.clip(yi, y0, y1)

        norm = (x1 - x0) * (y1 - y0)
        # lower-left, lower-right, upper-left, upper-right
        return np.array([(x1 - xi) * (y1 - yi), (xi - x0) * (y1 - yi),
                         (x1 - xi) * (yi - y0), (xi - x0) * (yi - y0)]) / norm

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
        interpolators = np.array([self._calc_interpolator(gidx)
                                  for gidx in grid_idx])
        weights = self._calc_bilinear_weights(x_0, y_0, grid_xy)

        idx = np.where(weights != 0)
        interpolators = interpolators[idx]
        weights = weights[idx]

        result = 0
        for interp, weight in zip(interpolators, weights, strict=True):
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

        # the base Model.__call__() method converts scalar inputs to
        # size-1 arrays before calling evaluate(), but we need scalar
        # values for the interpolator
        if not np.isscalar(x_0):
            x_0 = x_0[0]
        if not np.isscalar(y_0):
            y_0 = y_0[0]

        # now evaluate the ePSF at the (x_0, y_0) subpixel position on
        # the input (x, y) values
        xi = self.oversampling[1] * (np.asarray(x, dtype=float) - x_0)
        yi = self.oversampling[0] * (np.asarray(y, dtype=float) - y_0)
        xi += self.origin[0]
        yi += self.origin[1]

        if self.data.shape[0] == 1:
            # if there is only one ePSF, we do not need to perform
            # the bilinear interpolation
            evaluated_model = flux * self._calc_interpolator(0)(xi, yi,
                                                                grid=False)
        else:
            evaluated_model = flux * self._calc_model_values(x_0, y_0, xi, yi)

        if self.fill_value is not None:
            # set pixels that are outside the input pixel grid to the
            # fill_value to avoid extrapolation; these bounds match the
            # RegularGridInterpolator bounds
            ny, nx = self.data.shape[1:]
            invalid = (xi < 0) | (xi > nx - 1) | (yi < 0) | (yi > ny - 1)
            evaluated_model[invalid] = self.fill_value

        return evaluated_model


class STDPSFGrid(ModelGridPlotMixin):
    """
    Class to read and plot "STDPSF" format ePSF model grids.

    STDPSF files are FITS files that contain a 3D array of ePSFs with
    the header detailing where the fiducial ePSFs are located in the
    detector coordinate frame.

    The oversampling factor for STDPSF FITS files is assumed to be 4.

    Parameters
    ----------
    filename : str
        The name of the STDPDF FITS file. A URL can also be used.

    Examples
    --------
    >>> from photutils.psf import STDPSFGrid
    >>> psfgrid = STDPSFGrid('STDPSF_ACSWFC_F814W.fits')
    >>> fig = psfgrid.plot_grid()
    >>> fig.show()
    """

    def __init__(self, filename):
        grid_data = _read_stdpsf(filename)
        self.data = grid_data['data']
        self._xgrid = grid_data['xgrid']
        self._ygrid = grid_data['ygrid']
        xy_grid = [yx[::-1] for yx in itertools.product(self._ygrid,
                                                        self._xgrid)]
        oversampling = 4  # assumption for STDPSF files
        self.grid_xypos = xy_grid
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        meta = {'grid_shape': (len(self._ygrid), len(self._xgrid)),
                'grid_xypos': xy_grid,
                'oversampling': oversampling}

        # try to get additional metadata from the filename because this
        # information is not currently available in the FITS headers
        file_meta = _get_metadata(filename, None)
        if file_meta is not None:
            meta.update(file_meta)

        self.meta = meta

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        cls_info = []

        keys = ('STDPSF', 'detector', 'filter', 'grid_shape')
        for key in keys:
            if key in self.meta:
                name = key.capitalize() if key != 'STDPSF' else key
                cls_info.append((name, self.meta[key]))

        cls_info.extend([('Number of PSFs', len(self.grid_xypos)),
                         ('PSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', self.oversampling)])

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
