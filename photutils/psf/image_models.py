# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image-based PSF models.
"""

import copy
import warnings
from functools import cached_property

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.utils.exceptions import AstropyDeprecationWarning
from scipy.interpolate import RectBivariateSpline

from photutils.psf._bispline import bispline_sum, bispline_sum_deriv
from photutils.psf._bispline_inputs import ONE_PLANE, UNIT_WEIGHT, ZERO_WEIGHT
from photutils.psf.utils import _copy_model_sharing_data, _out_of_grid_mask
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
        A 2D array containing the ePSF image. The x and y dimensions
        must both be at least 4 pixels. All values must be finite. By
        default, the ePSF peak is assumed to be centered in the input
        image (see ``origin``). See the Notes section for the definition
        of an ePSF and for details on the required normalization of the
        input image.

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
        The value used for points outside the input pixel grid. The
        default is 0.0. If `None`, a point outside the input pixel grid
        takes the value of the spline at the nearest point on the edge
        of the grid.

    **kwargs : dict, optional
        Additional keyword arguments passed to the
        `~astropy.modeling.Model` base class.

    See Also
    --------
    GriddedPSFModel : A model for a grid of ePSF models.
    make_epsf_from_psf : Make an ePSF image from a sampled PSF.

    Notes
    -----
    The input image must be an effective PSF (ePSF). Each value of an
    ePSF is the fraction of the source flux that falls in a whole
    detector pixel centered at that position relative to the source,
    even when the image is oversampled. The model interpolates the
    input image and does not integrate it over the detector pixels.
    Evaluating the model at a position ``(x, y)`` therefore gives the
    flux in a detector pixel centered at ``(x, y)``, which can be any
    fractional pixel position.

    Because each value is the flux in a whole detector pixel, the model
    values sum to ``flux`` only when the model is evaluated on a grid
    with a spacing of one detector pixel. That holds for any values of
    ``x_0`` and ``y_0``. On a finer grid the pixels overlap, and the
    sum is larger than ``flux`` by the ratio of the pixel area to the
    area of a grid cell.

    An oversampled image that is not an ePSF, such as a PSF sampled at
    the points of a fine grid or binned into subpixels, is not converted
    to an ePSF by this model. The model is then sharper than a source
    in the data. For an undersampled PSF, the sum of the model values
    over the detector pixels can also change with the subpixel position
    of the source. Such an image should first be integrated over the
    area of a detector pixel centered at each of its grid points, which
    is what `make_epsf_from_psf` does.

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
    In this simple example, we create a PSF image model from a circular
    Gaussian that is integrated over the pixels, which is an ePSF with
    no oversampling. In this case, one should use the
    `CircularGaussianPRF` model directly as a PSF model. However, this
    example demonstrates how to create an image PSF model from an input
    image.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import CircularGaussianPRF, ImagePSF

        gaussian_prf = CircularGaussianPRF(x_0=12, y_0=12, fwhm=3.2)
        yy, xx = np.mgrid[:25, :25]
        psf_data = gaussian_prf(xx, yy)
        psf_model = ImagePSF(psf_data, x_0=12, y_0=12, flux=10)
        data = psf_model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')

    An oversampled PSF whose values are samples of the PSF, such as
    the output of an optical model, must be converted to an ePSF before
    it is used as the input image. Here, a narrow Gaussian PSF is
    sampled on a grid that is oversampled by a factor of 4 and then
    integrated over the detector pixels with `make_epsf_from_psf`:

    >>> import numpy as np
    >>> from photutils.psf import ImagePSF, make_epsf_from_psf
    >>> oversampling = 4
    >>> yy, xx = np.mgrid[-30:31, -30:31] / oversampling
    >>> sigma = 0.5  # detector pixels
    >>> psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    >>> psf *= oversampling**2 / psf.sum()
    >>> epsf = make_epsf_from_psf(psf, oversampling=oversampling)
    >>> model = ImagePSF(epsf, oversampling=oversampling, x_0=0.3,
    ...                  y_0=-0.4)

    The model values on a grid of detector pixels sum to the model
    flux for any subpixel position of the source:

    >>> yy, xx = np.mgrid[-7:8, -7:8]
    >>> print(f'{model(xx, yy).sum():.3f}')
    1.000
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

        if type(self).interpolator is not ImagePSF.interpolator:
            msg = ('Overriding the ImagePSF.interpolator attribute in a '
                   'subclass is deprecated since version 3.1 and will be '
                   'removed in version 4.0.')
            warnings.warn(msg, AstropyDeprecationWarning, stacklevel=2)

        super().__init__(flux, x_0, y_0, **kwargs)

    def __setattr__(self, name, value):
        if name == 'interpolator':
            msg = ('Assigning a custom interpolator to the '
                   'ImagePSF.interpolator attribute is deprecated since '
                   'version 3.1 and will be removed in version 4.0.')
            warnings.warn(msg, AstropyDeprecationWarning, stacklevel=2)
            # The model calls an assigned interpolator instead of
            # evaluating the spline of the image data with the kernel.
            self.__dict__['_interpolator_assigned'] = True
        super().__setattr__(name, value)

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

        The cached spline is one of the shared attributes. It is built
        on this model first, if it is not already cached, so that every
        copy shares it instead of building its own.

        Use the `deepcopy` method if you want to copy all the model
        attributes, including the PSF image data.

        Returns
        -------
        result : `ImagePSF`
            A copy of this model with only the model parameters copied.
        """
        if not self._has_custom_interpolator:
            _ = self._spline
        return _copy_model_sharing_data(self)

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

        The cached `interpolator` is not updated if the array is
        modified in place (e.g., ``model.data[:] = values``) after the
        model has been evaluated or copied with `copy`. Assign a new
        array to this attribute instead.
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
        # data. That includes an interpolator assigned to the model.
        self.__dict__.pop('interpolator', None)
        self.__dict__.pop('_interpolator_assigned', None)
        self.__dict__.pop('_deriv_interpolators', None)
        self.__dict__.pop('_spline', None)

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
        The model evaluates the knots and coefficients of this spline
        with a compiled kernel, which also computes the partial
        derivatives for `fit_deriv`. The spline object itself is not
        called.

        .. deprecated:: 3.1
            Defining a custom interpolator, either by overriding this
            property in a subclass or by assigning an interpolator
            to this attribute of a model, is deprecated and will
            be removed in version 4.0. A model with a custom
            interpolator calls it instead of using the compiled
            kernel. An assigned interpolator is discarded when
            `data` is set. The custom interpolator must provide a
            `~scipy.interpolate.RectBivariateSpline`-compatible
            ``partial_derivative`` method to support `fit_deriv`.
            Otherwise, ``fit_deriv`` should also be set to `None` to
            fall back to the fitter's finite-difference Jacobian.
        """
        x = np.arange(self.data.shape[1])
        y = np.arange(self.data.shape[0])
        # RectBivariateSpline expects the data to be in (x, y) axis order
        return RectBivariateSpline(x, y, self.data.T, kx=3, ky=3, s=0)

    @property
    def _has_custom_interpolator(self):
        """
        Whether this model has a custom interpolator.

        A custom interpolator is one defined by a subclass that
        overrides `interpolator`, or one that was assigned to the
        ``interpolator`` attribute of this model. Such a model calls its
        interpolator. Otherwise, the spline built from the image data is
        evaluated by the compiled kernel.
        """
        return (type(self).interpolator is not ImagePSF.interpolator
                or self.__dict__.get('_interpolator_assigned', False))

    @cached_property
    def _spline(self):
        """
        The knots and coefficients of the `interpolator` spline, as
        the ``(tx, ty, coeffs)`` tuple that the compiled kernel takes.

        The arrays are the ones that the spline object holds, not
        copies of them.
        """
        interp = self.interpolator
        tx, ty = interp.get_knots()
        return (np.ascontiguousarray(tx, dtype=float),
                np.ascontiguousarray(ty, dtype=float),
                np.ascontiguousarray(interp.get_coeffs(), dtype=float))

    @cached_property
    def _deriv_interpolators(self):
        """
        The spline partial-derivative interpolators of a custom
        interpolator.

        The interpolators evaluate the partial derivatives of
        `interpolator` with respect to its first (x) and second (y)
        variables. They are precomputed here because evaluating them
        is faster than passing ``dx=1`` or ``dy=1`` to `interpolator`,
        which computes the derivative on the fly. They are used by
        `fit_deriv`.
        """
        return (self.interpolator.partial_derivative(1, 0),
                self.interpolator.partial_derivative(0, 1))

    def _precompute_interpolators(self):
        """
        Compute and cache the interpolators of a custom interpolator.

        The cached interpolators are shared by the model copies made
        with `copy` (e.g., by the fitters), so calling this method
        before fitting the model to many sources builds the splines
        once instead of once per copy. The derivative interpolators are
        computed only when `fit_deriv` is enabled.

        A model without a custom interpolator needs no such step,
        because `copy` builds its spline.
        """
        if not self._has_custom_interpolator:
            return
        _ = self.interpolator
        if self.fit_deriv is not None:
            _ = self._deriv_interpolators

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
        # regardless of the scipy version.
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        xi = self.oversampling[1] * (x - x_0)
        yi = self.oversampling[0] * (y - y_0)
        xi += self._origin[0]
        yi += self._origin[1]
        if xi.shape != yi.shape:
            xi, yi = np.broadcast_arrays(xi, yi)

        if self._has_custom_interpolator:
            evaluated_model = flux * self.interpolator(xi, yi, grid=False)
        else:
            tx, ty, coeffs = self._spline
            values = np.empty(xi.shape, dtype=float)
            bispline_sum(tx, ty, coeffs[np.newaxis], ONE_PLANE,
                         UNIT_WEIGHT, xi.ravel(), yi.ravel(),
                         values.ravel())
            # The flux may have units or be an array, so the product
            # cannot be stored in the plain array of the kernel output
            evaluated_model = flux * values

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
        if xi.shape != yi.shape:
            xi, yi = np.broadcast_arrays(xi, yi)

        # The spline interpolation is linear in flux, and the chain rule
        # gives the x_0 and y_0 derivatives from the spline partial
        # derivatives (dxi/dx_0 = -oversampling[1], dyi/dy_0 =
        # -oversampling[0])
        if self._has_custom_interpolator:
            dx_interp, dy_interp = self._deriv_interpolators
            d_flux = self.interpolator(xi, yi, grid=False)
            d_x_0 = (-flux * self.oversampling[1]
                     * dx_interp(xi, yi, grid=False))
            d_y_0 = (-flux * self.oversampling[0]
                     * dy_interp(xi, yi, grid=False))
        else:
            tx, ty, coeffs = self._spline
            d_flux = np.empty(xi.shape, dtype=float)
            deriv_x = np.empty(xi.shape, dtype=float)
            deriv_y = np.empty(xi.shape, dtype=float)
            bispline_sum_deriv(tx, ty, coeffs[np.newaxis], ONE_PLANE,
                               UNIT_WEIGHT, ZERO_WEIGHT, ZERO_WEIGHT,
                               float(self.oversampling[1]),
                               float(self.oversampling[0]),
                               xi.ravel(), yi.ravel(), d_flux.ravel(),
                               deriv_x.ravel(), deriv_y.ravel())
            # The flux may have units or be an array, so the products
            # cannot be stored in the plain arrays of the kernel output
            d_x_0 = flux * deriv_x
            d_y_0 = flux * deriv_y

        if self.fill_value is not None:
            # Outside the input pixel grid the model is constant
            # (fill_value), so all derivatives are zero there
            invalid = _out_of_grid_mask(xi, yi, self.data.shape)
            d_flux[invalid] = 0.0
            d_x_0[invalid] = 0.0
            d_y_0[invalid] = 0.0

        return [d_flux, d_x_0, d_y_0]
