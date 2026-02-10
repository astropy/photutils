# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image-based PSF models.
"""

import copy

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.utils.decorators import lazyproperty
from scipy.interpolate import RectBivariateSpline

from photutils.utils._parameters import as_pair

__all__ = ['ImagePSF']


class ImagePSF(Fittable2DModel):
    """
    A model for a 2D image PSF.

    This class takes 2D image data and computes the values of the model
    at arbitrary locations, including fractional pixel positions, within
    the image using spline interpolation provided by
    :py:class:`~scipy.interpolate.RectBivariateSpline`.

    The model has three model parameters: an image intensity scaling
    factor (``flux``) which is applied to the input image, and two
    positional parameters (``x_0`` and ``y_0``) indicating the location
    of a feature in the coordinate grid on which the model is evaluated.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        Array containing the 2D image. The length of the x and y axes
        must both be at least 4. All elements of the input image data
        must be finite. By default, the PSF peak is assumed to be
        located at the center of the input image (see the ``origin``
        keyword). Please see the Notes section below for details on the
        normalization of the input image data.

    flux : float, optional
        The total flux of the source, assuming the input image
        was properly normalized.

    x_0, y_0 : float
        The x and y positions of a feature in the image in the output
        coordinate grid on which the model is evaluated. Typically, this
        refers to the position of the PSF peak, which is assumed to be
        located at the center of the input image (see the ``origin``
        keyword).

    origin : tuple of 2 float or None, optional
        The ``(x, y)`` coordinate with respect to the input image data
        array that represents the reference pixel of the input data.

        The reference ``origin`` pixel will be placed at the model
        ``x_0`` and ``y_0`` coordinates in the output coordinate system
        on which the model is evaluated.

        Most typically, the input PSF should be centered in the input
        image, and thus the origin should be set to the central pixel of
        the ``data`` array.

        If the origin is set to `None`, then the origin will be set to
        the center of the ``data`` array (``(npix - 1) / 2.0``).

    oversampling : int or array_like (int), optional
        The integer oversampling factor(s) of the input PSF image. If
        ``oversampling`` is a scalar then it will be used for both axes.
        If ``oversampling`` has two elements, they must be in ``(y, x)``
        order.

    fill_value : float, optional
        The value to use for points outside the input pixel grid. The
        default is 0.0.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    GriddedPSFModel : A model for a grid of ePSF models.

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
        plt.imshow(data, origin='lower')
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

        self._validate_data(data)
        self.data = data
        self.origin = origin
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
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

        # this is required by RectBivariateSpline for kx=3, ky=3
        if np.any(np.array(data.shape) < 4):
            msg = 'The length of the x and y axes must both be at least 4'
            raise ValueError(msg)

    def __str__(self):
        keywords = [('PSF shape (oversampled pixels)', self.data.shape),
                    ('Origin', self.origin.tolist()),
                    ('Oversampling', self.oversampling.tolist()),
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
        result : `ImagePSF`
            A deep copy of this model.
        """
        return copy.deepcopy(self)

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
    def origin(self):
        """
        A 1D `~numpy.ndarray` (x, y) pixel coordinates within the
        model's 2D image of the origin of the coordinate system.

        The reference ``origin`` pixel will be placed at the model
        ``x_0`` and ``y_0`` coordinates in the output coordinate system
        on which the model is evaluated.

        Most typically, the input PSF should be centered in the input
        image, and thus the origin should be set to the central pixel of
        the ``data`` array.

        If the origin is set to `None`, then the origin will be set to
        the center of the ``data`` array (``(npix - 1) / 2.0``).
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

    @lazyproperty
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
        This property can be overridden in a subclass to define custom
        interpolators.
        """
        x = np.arange(self.data.shape[1])
        y = np.arange(self.data.shape[0])
        # RectBivariateSpline expects the data to be in (x, y) axis order
        return RectBivariateSpline(x, y, self.data.T, kx=3, ky=3, s=0)

    def _calc_bounding_box(self):
        """
        Set a bounding box defining the limits of the model.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        dy, dx = np.array(self.data.shape) / 2 / self.oversampling

        # apply the origin shift
        # if origin is None, the origin is set to the center of the
        # image and the shift is 0
        xshift = np.array(self.data.shape[1] - 1) / 2 - self.origin[0]
        yshift = np.array(self.data.shape[0] - 1) / 2 - self.origin[1]
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
        >>> model.bounding_box  # doctest: +FLOAT_CMP
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
        xi = self.oversampling[1] * (np.asarray(x, dtype=float) - x_0)
        yi = self.oversampling[0] * (np.asarray(y, dtype=float) - y_0)
        xi += self._origin[0]
        yi += self._origin[1]

        evaluated_model = flux * self.interpolator(xi, yi, grid=False)

        if self.fill_value is not None:
            # set pixels that are outside the input pixel grid to the
            # fill_value to avoid extrapolation; these bounds match the
            # RegularGridInterpolator bounds
            ny, nx = self.data.shape
            invalid = (xi < 0) | (xi > nx - 1) | (yi < 0) | (yi > ny - 1)
            evaluated_model[invalid] = self.fill_value

        return evaluated_model
