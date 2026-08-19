# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image-based PSF models.
"""

import copy
from functools import cached_property

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from scipy.interpolate import RectBivariateSpline

from photutils.psf.utils import _out_of_grid_mask
from photutils.utils._parameters import as_pair

__all__ = ['ImagePSF']


class ImagePSF(Fittable2DModel):
    """
    A model representing a 2D image PSF.

    This class evaluates a 2D image PSF at arbitrary positions,
    including fractional pixel coordinates, using spline interpolation
    provided by `~scipy.interpolate.RectBivariateSpline`.

    This model has three parameters: an image intensity scaling factor
    (``flux``), which scales the input image, and two positional
    parameters (``x_0`` and ``y_0``), which specify the location of the
    feature in the coordinate grid where the model is evaluated.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        A 2D array containing the PSF image. The x and y dimensions
        must both be at least 4 pixels. All values must be finite. By
        default, the PSF peak is assumed to be centered in the input
        image (see ``origin``). See the Notes section for details on the
        required normalization of the input image.

    flux : float, optional
        The flux scaling factor. This corresponds to the total source flux,
        assuming the input PSF image is properly normalized.

    x_0, y_0 : float, optional
        The x and y positions of a feature in the image in the output
        coordinate grid on which the model is evaluated. Typically, this
        refers to the position of the PSF peak, which is assumed to be
        located at the center of the input image (see the ``origin``
        keyword).

    origin : tuple of 2 float or None, optional
        The ``(x, y)`` coordinate in the input image corresponding to the
        reference pixel.

        The reference pixel is placed at the model ``x_0`` and ``y_0``
        coordinates in the output coordinate grid.

        In most cases, the PSF should be centered in the input image, so
        ``origin`` should be set to the central pixel of ``data``.

        If `None`, ``origin`` is set to the center of the input image,
        ``((n_x - 1) / 2, (n_y - 1) / 2)``.

    oversampling : int or array_like (int), optional
        The integer oversampling factor(s) of the input PSF image. If a
        scalar is provided, it is applied to both axes. If two values are
        provided, they must be in ``(y, x)`` order.

    fill_value : float or `None`, optional
        The value used for points outside the input pixel grid. The default
        is 0.0. If `None`, values outside the input pixel grid are
        extrapolated from the spline fit.

    **kwargs : dict, optional
        Additional keyword arguments passed to the
        `~astropy.modeling.Model` base class.

    See Also
    --------
    GriddedPSFModel : A model for a grid of ePSF models.

    Notes
    -----
    The fitted ``flux`` parameter represents the total source flux,
    provided the input PSF image is properly normalized. The fitted flux
    is a multiplicative scale factor applied to the input PSF after
    accounting for any oversampling.

    For a fully sampled ePSF (i.e., no oversampling), the sum of
    the ePSF values over an infinite grid is 1.0. Because ePSFs are
    represented by finite images in practice, the sum of the array
    values may be less than 1.0.

    For oversampled ePSF images, the normalization should instead be
    such that the sum of the array values over an infinite grid equals
    the product of the oversampling factors (e.g., ``oversampling**2``
    when the oversampling is the same along both axes). Again, a finite
    image will generally have a smaller sum because it does not contain
    the full PSF wings.

    If the input PSF image covers only a finite region of the PSF,
    correction factors based on the encircled or ensquared energy
    can be used to estimate the missing flux and obtain the proper
    normalization.

    Examples
    --------
    In this simple example, we create a PSF image model from a Circular
    Gaussian PSF. In this case, one should use the `CircularGaussianPSF`
    model directly as a PSF model. However, this example demonstrates
    how to create an image PSF model from an input image.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import CircularGaussianPSF, ImagePSF

        gaussian_psf = CircularGaussianPSF(x_0=12, y_0=12, fwhm=3.2)
        yy, xx = np.mgrid[:25, :25]
        psf_data = gaussian_psf(xx, yy)
        psf_model = ImagePSF(psf_data, x_0=12, y_0=12, flux=10)
        data = psf_model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """

    flux = Parameter(default=1,
                     description='Intensity scaling factor of the image.')
    x_0 = Parameter(default=0,
                    description=('Position of a feature in the image along '
                                 'the x axis'))
    y_0 = Parameter(default=0,
                    description=('Position of a feature in the image along '
                                 'the y axis'))

    def __init__(self, data, *, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, origin=None, oversampling=1,
                 fill_value=0.0, **kwargs):

        self.data = data
        self.origin = origin
        self.oversampling = oversampling
        self.fill_value = fill_value

        super().__init__(flux, x_0, y_0, **kwargs)

    @staticmethod
    def _validate_data(data):
        if not isinstance(data, np.ndarray):
            msg = 'Input data must be a 2D numpy array'
            raise TypeError(msg)

        if data.ndim != 2:
            msg = 'Input data must be a 2D numpy array'
            raise ValueError(msg)

        if not np.all(np.isfinite(data)):
            msg = 'All elements of input data must be finite'
            raise ValueError(msg)

        # The minimum number of data points required is 4 along each
        # axis. This is because RectBivariateSpline requires at least 4
        # points along each axis for cubic spline interpolation (kx=3,
        # ky=3).
        if np.any(np.array(data.shape) < 4):
            msg = 'The length of the x and y axes must both be at least 4'
            raise ValueError(msg)

    def __str__(self):
        keywords = [('PSF shape (oversampled pixels)', self.data.shape),
                    ('Origin', self.origin.tolist()),
                    ('Oversampling', tuple(self.oversampling.tolist())),
                    ('Fill Value', self.fill_value),
                    ]
        return self._format_str(keywords=keywords)

    def __repr__(self):
        kwargs = {'origin': self.origin.tolist(),
                  'oversampling': self.oversampling.tolist(),
                  'fill_value': self.fill_value}
        return self._format_repr(kwargs=kwargs)

    def copy(self):
        """
        Return a copy of this model where only the model parameters are
        copied.

        All other copied model attributes are references to the original
        model. This prevents copying the image data, which may be a
        large array.

        This method is useful if one is interested in only changing
        the model parameters in a model copy. It is used in the PSF
        photometry classes during model fitting.

        Use the `deepcopy` method if you want to copy all the model
        attributes, including the PSF image data.

        Returns
        -------
        result : `ImagePSF`
            A copy of this model with only the model parameters copied.
        """
        newcls = object.__new__(self.__class__)

        # Snapshot so concurrent cached_property fills cannot resize
        # the dict during iteration
        for key, val in dict(self.__dict__).items():
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
        result : `ImagePSF`
            A deep copy of this model.
        """
        return copy.deepcopy(self)

    @property
    def data(self):
        """
        The 2D image of the PSF.

        Setting this attribute revalidates the input array and discards
        the cached `interpolator`. The `origin` is not updated, so it
        should be reset explicitly if the new image has a different
        shape or a different reference pixel.
        """
        return self._data

    @data.setter
    def data(self, value):
        """
        Set the 2D image of the PSF.

        Parameters
        ----------
        value : 2D `~numpy.ndarray`
            The 2D image of the PSF.
        """
        self._validate_data(value)
        self._data = value
        # Discard the cached interpolators, which are tied to the old
        # data
        self.__dict__.pop('interpolator', None)
        self.__dict__.pop('_deriv_interpolators', None)

    @property
    def shape(self):
        """
        The shape of the (oversampled) PSF data array.

        Returns
        -------
        shape : tuple
            The shape of the (oversampled) PSF data array.
        """
        return self.data.shape

    @property
    def oversampling(self):
        """
        The integer oversampling factor(s) of the input PSF image.

        If ``oversampling`` is a scalar then it will be used for both
        axes. If ``oversampling`` has two elements, they must be in
        ``(y, x)`` order.
        """
        return self._oversampling

    @oversampling.setter
    def oversampling(self, value):
        """
        Set the oversampling factor(s) of the input PSF image.

        Parameters
        ----------
        value : int or tuple of int
            The integer oversampling factor(s) of the input PSF image.
            If ``oversampling`` is a scalar then it will be used for
            both axes. If ``oversampling`` has two elements, they must
            be in ``(y, x)`` order.
        """
        self._oversampling = as_pair('oversampling', value,
                                     lower_bound=(0, 0))

    @property
    def origin(self):
        """
        The (x, y) pixel coordinates, as a 1D `~numpy.ndarray`, of the
        origin of the coordinate system within the model image.

        The reference ``origin`` pixel will be placed at the model
        ``x_0`` and ``y_0`` coordinates in the output coordinate system
        on which the model is evaluated.

        Most typically, the input PSF should be centered in the input
        image, and thus the origin should be set to the central pixel of
        the ``data`` array.

        If the origin is set to `None`, then the origin will be set to
        the center of the ``data`` array (``(n_pixels - 1) / 2.0``).
        """
        return self._origin

    @origin.setter
    def origin(self, origin):
        if origin is None:
            origin = (np.array(self.data.shape) - 1.0) / 2.0
            origin = origin[::-1]  # flip to (x, y) order
        else:
            origin = np.asarray(origin)
            if origin.ndim != 1 or len(origin) != 2:
                msg = 'origin must be 1D and have 2-elements'
                raise ValueError(msg)
            if not np.all(np.isfinite(origin)):
                msg = 'All elements of origin must be finite'
                raise ValueError(msg)
        self._origin = origin

    @cached_property
    def interpolator(self):
        """
        The interpolating spline function.

        The interpolator is computed with a 3rd-degree
        `~scipy.interpolate.RectBivariateSpline` (kx=3, ky=3, s=0) using
        the input image data. The interpolator is used to evaluate
        the model at arbitrary locations, including fractional pixel
        positions.

        Notes
        -----
        This property can be overridden in a subclass to define
        custom interpolators. A custom interpolator must provide
        a `~scipy.interpolate.RectBivariateSpline`-compatible
        ``partial_derivative`` method to support `fit_deriv`. Otherwise,
        the subclass should also set ``fit_deriv = None`` to fall back
        to the fitter's finite-difference Jacobian.
        """
        x = np.arange(self.data.shape[1])
        y = np.arange(self.data.shape[0])
        # RectBivariateSpline expects the data to be in (x, y) axis order
        return RectBivariateSpline(x, y, self.data.T, kx=3, ky=3, s=0)

    @cached_property
    def _deriv_interpolators(self):
        """
        The spline partial-derivative interpolators.

        The interpolators evaluate the partial derivatives of
        `interpolator` with respect to its first (x) and second (y)
        variables. They are precomputed here because evaluating them
        is faster than passing ``dx=1`` or ``dy=1`` to `interpolator`,
        which computes the derivative on the fly. They are used by
        `fit_deriv`.
        """
        return (self.interpolator.partial_derivative(1, 0),
                self.interpolator.partial_derivative(0, 1))

    def _calc_bounding_box(self):
        """
        Return a bounding box defining the limits of the model.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        dy, dx = np.array(self.data.shape) / 2 / self.oversampling

        # Apply the origin shift. If origin is None, the origin is set
        # to the center of the image and the shift is 0.
        xshift = (self.data.shape[1] - 1) / 2 - self.origin[0]
        yshift = (self.data.shape[0] - 1) / 2 - self.origin[1]
        xshift /= self.oversampling[1]
        yshift /= self.oversampling[0]

        return ((self.y_0 - dy + yshift, self.y_0 + dy + yshift),
                (self.x_0 - dx + xshift, self.x_0 + dx + xshift))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import ImagePSF
        >>> psf_data = np.arange(30, dtype=float).reshape(5, 6)
        >>> psf_data /= np.sum(psf_data)
        >>> model = ImagePSF(psf_data, flux=1, x_0=0, y_0=0)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-3.0, upper=3.0)
                y: Interval(lower=-2.5, upper=2.5)
            }
            model=ImagePSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box()

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Calculate the value of the image model at the input coordinates
        for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            The total flux of the source, assuming the input image
            was properly normalized.

        x_0, y_0 : float
            The x and y positions of the feature in the image in the
            output coordinate grid on which the model is evaluated.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        # Promote scalar inputs to 1D arrays so that the interpolator
        # returns an array that supports masked assignment below,
        # regardless of the scipy version
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        xi = self.oversampling[1] * (x - x_0)
        yi = self.oversampling[0] * (y - y_0)
        xi += self._origin[0]
        yi += self._origin[1]

        evaluated_model = flux * self.interpolator(xi, yi, grid=False)

        if self.fill_value is not None:
            # Set pixels that are outside the input pixel grid to the
            # fill_value to avoid extrapolation
            invalid = _out_of_grid_mask(xi, yi, self.data.shape)
            evaluated_model[invalid] = self.fill_value

        return evaluated_model

    def fit_deriv(self, x, y, flux, x_0, y_0):
        """
        Calculate the partial derivatives of the image model with
        respect to the model parameters.

        Providing this analytic Jacobian allows the fitter to avoid the
        finite-difference approximation, which requires additional model
        evaluations.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            The total flux of the source, assuming the input image
            was properly normalized.

        x_0, y_0 : float
            The x and y positions of the feature in the image in the
            output coordinate grid on which the model is evaluated.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to the
            ``flux``, ``x_0``, and ``y_0`` parameters.
        """
        # Promote scalar inputs to 1D arrays so that the interpolator
        # returns an array that supports masked assignment below,
        # regardless of the scipy version
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        xi = self.oversampling[1] * (x - x_0)
        yi = self.oversampling[0] * (y - y_0)
        xi += self._origin[0]
        yi += self._origin[1]

        # The spline interpolation is linear in flux, and the chain rule
        # gives the x_0 and y_0 derivatives from the spline partial
        # derivatives (dxi/dx_0 = -oversampling[1], dyi/dy_0 =
        # -oversampling[0])
        dx_interp, dy_interp = self._deriv_interpolators
        d_flux = self.interpolator(xi, yi, grid=False)
        d_x_0 = (-flux * self.oversampling[1]
                 * dx_interp(xi, yi, grid=False))
        d_y_0 = (-flux * self.oversampling[0]
                 * dy_interp(xi, yi, grid=False))

        if self.fill_value is not None:
            # Outside the input pixel grid the model is constant
            # (fill_value), so all derivatives are zero there
            invalid = _out_of_grid_mask(xi, yi, self.data.shape)
            d_flux[invalid] = 0.0
            d_x_0[invalid] = 0.0
            d_y_0[invalid] = 0.0

        return [d_flux, d_x_0, d_y_0]
