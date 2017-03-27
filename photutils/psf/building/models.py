"""
This module provides fittable models based on 2D images.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
import logging
import copy

import numpy as np
from astropy.modeling import Fittable2DModel
from astropy.modeling.parameters import Parameter

__all__ = ['FittableImageModel2D', 'NonNormalizable']

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler(level=logging.INFO))


class NonNormalizable(Warning):
    """
    Used to undicate that a :py:class:`FittableImageModel2D` model is
    non-normalizable.

    """
    pass


class FittableImageModel2D(Fittable2DModel):
    """
    A fittable 2D model of an image allowing for image intensity scaling
    and image translations.

    This class takes 2D image data and computes the
    values of the model at arbitrary locations (including at intra-pixel,
    fractional positions) within this image using spline interpolation
    provided by :py:class:`~scipy.interpolate.RectBivariateSpline`.

    The fittable model provided by this class has three model parameters:
    an image intensity scaling factor (`flux`) which is applied to
    (normalized) image, and two positional parameters (`x_0` and `y_0`)
    indicating the location of a feature in the coordinate grid on which
    the model is to be evaluated.

    If this class is initialized with `flux` (intensity scaling factor)
    set to `None`, then `flux` is be estimated as ``|sum(data)|``.

    Parameters
    ----------
    data : numpy.ndarray
        Array containing 2D image.

    origin : tuple, None, optional
        A reference point in the input image ``data`` array. When origin is
        `None`, origin will be set at the middle of the image array.

        If `origin` represents the location of a feature (e.g., the position
        of an intensity peak) in the input ``data``, then model parameters
        `x_0` and `y_0` show the location of this peak in an another target
        image to which this model was fitted. Fundamentally, it is the
        coordinate in the model's image data that should map to
        coordinate (`x_0`, `y_0`) of the output coordinate system on which the
        model is evaluated.

        Alternatively, when `origin` is set to ``(0,0)``, then model parameters
        `x_0` and `y_0` are shifts by which model's image should be translated
        in order to match a target image.

    normalize : bool, optional
        Indicates whether or not the model should be build on normalized
        input image data. If true, then the normalization constant (*N*) is
        computed so that

        .. math::
            N \cdot C \cdot |\Sigma_{i,j}D_{i,j}| = 1,

        where *N* is the normalization constant, *C* is correction factor
        given by the parameter ``correction_factor``, and :math:`D_{i,j}` are
        the elements of the input image ``data`` array.

    correction_factor : float, optional
        A strictly positive number that represents correction that needs to
        be applied to model's `flux`. This parameter affects the value of
        the normalization factor (see ``normalize`` for more details).

        A possible application for this parameter is to account for aperture
        correction. Assuming model's data represent a PSF to be fitted to
        some target star, we set ``correction_factor`` to the aperture
        correction that needs to be applied to the model.
        Then, best fitted value of the `flux` model
        parameter will represent an aperture-corrected flux of the target star.

    fill_value : float, optional
        The value to be returned by the `evaluate` or
        ``astropy.modeling.Model.__call__`` methods
        when evaluation is performed outside the definition domain of the
        model.

    ikwargs : dict, optional

        Additional optional keyword arguments to be passed directly to the
        `compute_interpolator` method. See `compute_interpolator` for more
        details.

    """
    flux = Parameter(description='Intensity scaling factor for image data.',
                     default=None)
    x_0 = Parameter(description='X-position of a feature in the image in '
                    'the output coordinate grid on which the model is '
                    'evaluated.', default=0.0)
    y_0 = Parameter(description='Y-position of a feature in the image in '
                    'the output coordinate grid on which the model is '
                    'evaluated.', default=0.0)

    def __init__(self, data, flux=flux.default,
                 x_0=x_0.default, y_0=y_0.default,
                 normalize=False, correction_factor=1.0,
                 origin=None, fill_value=0.0, ikwargs={}):
        self._fill_value = fill_value
        self._img_norm = None
        self._normalization_status = 0 if normalize else 2
        self._store_interpolator_kwargs(ikwargs)

        if correction_factor <= 0:
            raise ValueError("'correction_factor' must be strictly positive.")
        self._correction_factor = correction_factor

        self._data = np.array(data, copy=True, dtype=np.float64)

        if not np.all(np.isfinite(self._data)):
            raise ValueError("All elements of input 'data' must be finite.")

        # set input image related parameters:
        self._ny, self._nx = self._data.shape
        self._shape = self._data.shape
        if self._data.size < 1:
            raise ValueError("Image data array cannot be zero-sized.")

        # set the origin of the coordinate system in image's pixel grid:
        self.origin = origin

        if flux is None:
            if self._img_norm is None:
                self._img_norm = self._compute_raw_image_norm(self._data)
            flux = self._img_norm

        self._compute_normalization(normalize)

        super(FittableImageModel2D, self).__init__(flux, x_0, y_0)

        # initialize interpolator:
        self.compute_interpolator(ikwargs)

    def _compute_raw_image_norm(self, data):
        """
        Helper function that computes the uncorrected inverse normalization
        factor of input image data. This quantity is computed as the
        *absolute value* of the *sum of all pixel values*.

        .. note::
            This function is intended to be overriden in a subclass if one
            desires to change the way the normalization factor is computed.

        """
        return np.abs(np.sum(self._data, dtype=np.float64))

    def _compute_normalization(self, normalize):
        """
        Helper function that computes the inverse normalization factor of the
        original image data. This quantity is computed as the *absolute value*
        of the the sum of pixel values. Computation is performed only if this
        sum has not been previously computed. Otherwise, the existing value is
        not modified as :py:class:`FittableImageModel2D` does not allow image
        data to be modified after the object is created.

        .. note::
            Normally, this function should not be called by the end-user. It
            is intended to be overriden in a subclass if one desires to change
            the way the normalization factor is computed.

        """
        self._normalization_constant = 1.0 / self._correction_factor

        if normalize:
            # compute normalization constant so that
            # N*C*sum(data) = 1:
            if self._img_norm is None:
                self._img_norm = self._compute_raw_image_norm(self._data)

            if self._img_norm != 0.0 and np.isfinite(self._img_norm):
                self._normalization_constant /= self._img_norm
                self._normalization_status = 0

            else:
                self._normalization_constant = 1.0
                self._normalization_status = 1
                warnings.warn("Overflow encountered while computing "
                              "normalization constant. Normalization "
                              "constant will be set to 1.", NonNormalizable)

        else:
            self._normalization_status = 2

    @property
    def data(self):
        """ Get original image data. """
        return self._data

    @property
    def normalized_data(self):
        """ Get normalized and/or intensity-corrected image data. """
        return (self._normalization_constant * self._data)

    @property
    def normalization_constant(self):
        """ Get normalization constant. """
        return self._normalization_constant

    @property
    def normalization_status(self):
        """
        Get normalization status. Possible status values are:

        - 0: **Performed**. Model has been successfuly normalized at
          user's request.
        - 1: **Failed**. Attempt to normalize has failed.
        - 2: **NotRequested**. User did not request model to be normalized.

        """
        return self._normalization_status

    @property
    def correction_factor(self):
        """
        Set/Get flux correction factor.

        .. note::
            When setting correction factor, model's flux will be adjusted
            accordingly such that if this model was a good fit to some target
            image before, then it will remain a good fit after correction
            factor change.

        """
        return self._correction_factor

    @correction_factor.setter
    def correction_factor(self, correction_factor):
        old_cf = self._correction_factor
        self._correction_factor = correction_factor
        self._compute_normalization(normalize=self._normalization_status != 2)

        # adjust model's flux so that if this model was a good fit to some
        # target image, then it will remain a good fit after correction factor
        # change:
        self.flux *= correction_factor / old_cf

    @property
    def shape(self):
        """A tuple of dimensions of the data array in numpy style (ny, nx)."""
        return self._shape

    @property
    def nx(self):
        """Number of columns in the data array."""
        return self._nx

    @property
    def ny(self):
        """Number of rows in the data array."""
        return self._ny

    @property
    def origin(self):
        """
        A tuple of ``x`` and ``y`` coordinates of the origin of the coordinate
        system in terms of pixels of model's image.

        When setting the coordinate system origin, a tuple of two `int` or
        `float` may be used. If origin is set to `None`, the origin of the
        coordinate system will be set to the middle of the data array
        (``(npix-1)/2.0``).

        .. warning::
            Modifying `origin` will not adjust (modify) model's parameters
            `x_0` and `y_0`.
        """
        return (self._x_origin, self._y_origin)

    @origin.setter
    def origin(self, origin):
        if origin is None:
            self._x_origin = (self._nx - 1) / 2.0
            self._y_origin = (self._ny - 1) / 2.0
        elif hasattr(origin, '__iter__') and len(origin) == 2:
            self._x_origin, self._y_origin = origin
        else:
            raise TypeError("Parameter 'origin' must be either None or an "
                            "iterable with two elements.")

    @property
    def x_origin(self):
        """X-coordinate of the origin of the coordinate system."""
        return self._x_origin

    @property
    def y_origin(self):
        """Y-coordinate of the origin of the coordinate system."""
        return self._y_origin

    @property
    def fill_value(self):
        """Fill value to be returned for coordinates outside of the domain of
        definition of the interpolator. If ``fill_value`` is `None`, then
        values outside of the domain of definition are the ones returned
        by the interpolator.

        """
        return self._fill_value

    @fill_value.setter
    def fill_value(self, fill_value):
        self._fill_value = fill_value

    def _store_interpolator_kwargs(self, ikwargs):
        """
        This function should be called in a subclass whenever model's
        interpolator is (re-)computed.
        """
        self._interpolator_kwargs = copy.deepcopy(ikwargs)

    @property
    def interpolator_kwargs(self):
        """
        Get current interpolator's arguments used when interpolator was
        created.
        """
        return self._interpolator_kwargs

    def compute_interpolator(self, ikwargs={}):
        """
        Compute/define the interpolating spline. This function can be overriden
        in a subclass to define custom interpolators.

        Parameters
        ----------
        ikwargs : dict, optional

            Additional optional keyword arguments. Possible values are:

            - **degree** : int, tuple, optional
                Degree of the interpolating spline. A tuple can be used to
                provide different degrees for the X- and Y-axes.
                Default value is degree=3.

            - **s** : float, optional
                Non-negative smoothing factor. Default value s=0 corresponds to
                interpolation.
                See :py:class:`~scipy.interpolate.RectBivariateSpline` for more
                details.

        Notes
        -----
            * When subclassing :py:class:`FittableImageModel2D` for the
              purpose of overriding :py:func:`compute_interpolator`,
              the :py:func:`evaluate` may need to overriden as well depending
              on the behavior of the new interpolator. In addition, for
              improved future compatibility, make sure
              that the overriding method stores keyword arguments ``ikwargs``
              by calling ``_store_interpolator_kwargs`` method.

            * Use caution when modifying interpolator's degree or smoothness in
              a computationally intensive part of the code as it may decrease
              code performance due to the need to recompute interpolator.

        """
        from scipy.interpolate import RectBivariateSpline

        if 'degree' in ikwargs:
            degree = ikwargs['degree']
            if hasattr(degree, '__iter__') and len(degree) == 2:
                degx = int(degree[0])
                degy = int(degree[1])
            else:
                degx = int(degree)
                degy = int(degree)
            if degx < 0 or degy < 0:
                raise ValueError("Interpolator degree must be a non-negative "
                                 "integer")
        else:
            degx = 3
            degy = 3

        if 's' in ikwargs:
            smoothness = ikwargs['s']
        else:
            smoothness = 0

        x = np.arange(self._nx, dtype=np.float)
        y = np.arange(self._ny, dtype=np.float)
        self.interpolator = RectBivariateSpline(
            x, y, self._data.T, kx=degx, ky=degx, s=smoothness
        )

        self._store_interpolator_kwargs(ikwargs)

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Evaluate the model on some input variables and provided model
        parameters.

        """
        xi = np.asarray(x, dtype=np.float) + (self._x_origin - x_0)
        yi = np.asarray(y, dtype=np.float) + (self._y_origin - y_0)

        f = flux * self._normalization_constant

        evaluated_model = f * self.interpolator.ev(xi, yi)

        if self._fill_value is not None:
            # find indices of pixels that are outside the input pixel grid and
            # set these pixels to the 'fill_value':
            invalid = (((xi < 0) | (xi > self._nx - 1)) |
                       ((yi < 0) | (yi > self._ny - 1)))
            evaluated_model[invalid] = self._fill_value

        return evaluated_model
