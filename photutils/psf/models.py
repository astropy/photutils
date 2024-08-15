# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides models for doing PSF/PRF-fitting photometry.
"""

import copy
import warnings

import astropy.units as u
import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.utils import ellipse_extent
from astropy.units import UnitsError
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.utils._parameters import as_pair

__all__ = ['GaussianPSF', 'CircularGaussianPSF', 'GaussianPRF',
           'CircularGaussianPRF', 'FittableImageModel', 'EPSFModel',
           'IntegratedGaussianPRF', 'PRFAdapter']


GAUSSIAN_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


class GaussianPSF(Fittable2DModel):
    r"""
    A 2D Gaussian PSF model.

    This model is evaluated by sampling the 2D Gaussian at the input
    coordinates. The Gaussian is normalized such that the analytical
    integral over the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x axis.

    y_0 : float, optional
        Position of the peak along the y axis.

    x_fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian along the
        x axis.

    y_fwhm : float, optional
        FWHM of the Gaussian along the y axis.

    theta : float, optional
        The counterclockwise rotation angle either as a float (in
        degrees) or a `~astropy.units.Quantity` angle (optional).

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    CircularGaussianPSF, GaussianPRF, CircularGaussianPRF

    Notes
    -----
    The Gaussian function is defined as:

    .. math::

        f(x, y) = \frac{F}{2 \pi \sigma_{x} \sigma_{y}}
                  \exp \left( -a\left(x - x_{0}\right)^{2}
                  - b \left(x - x_{0}\right) \left(y - y_{0}\right)
                  - c \left(y - y_{0}\right)^{2} \right)

    where :math:`F` is the total integrated flux, :math:`(x_{0},
    y_{0})` is the position of the peak, and :math:`\sigma_{x}` and
    :math:`\sigma_{y}` are the standard deviations along the x and y
    axes, respectively.

    .. math::

        a = \frac{\cos^{2}{\theta}}{2 \sigma_{x}^{2}}
            + \frac{\sin^{2}{\theta}}{2 \sigma_{y}^{2}}

        b = \frac{\sin{2 \theta}}{2 \sigma_{x}^{2}}
            - \frac{\sin{2 \theta}}{2 \sigma_{y}^{2}}

        c = \frac{\sin^{2}{\theta}} {2 \sigma_{x}^{2}}
            + \frac{\cos^{2}{\theta}}{2 \sigma_{y}^{2}}

    where :math:`\theta` is the rotation angle of the Gaussian.

    The FWHMs of the Gaussian along the x and y axes are given by:

    .. math::

        \rm{FWHM}_{x} = 2 \sigma_{x} \sqrt{2 \ln{2}}

        \rm{FWHM}_{y} = 2 \sigma_{y} \sqrt{2 \ln{2}}

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) dx dy = F

    The ``x_fwhm``, ``y_fwhm``, and ``theta`` parameters are fixed by
    default. If you wish to fit these parameters, set the ``fixed``
    attribute to `False`, e.g.,::

        >>> from photutils.psf import GaussianPSF
        >>> model = GaussianPSF()
        >>> model.x_fwhm.fixed = False
        >>> model.y_fwhm.fixed = False
        >>> model.theta.fixed = False

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import GaussianPSF
        model = GaussianPSF(flux=71.4, x_0=24.3, y_0=25.2, x_fwhm=10.1,
                            y_fwhm=5.82, theta=21.7)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        plt.imshow(data, origin='lower', interpolation='nearest')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    x_fwhm = Parameter(
        default=1, description='FWHM of the Gaussian along the x axis',
        fixed=True)
    y_fwhm = Parameter(
        default=1, description='FWHM of the Gaussian along the y axis',
        fixed=True)
    theta = Parameter(
        default=0.0, description=('CCW rotation angle either as a float (in '
                                  'degrees) or a Quantity angle (optional)'),
        fixed=True)

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 x_fwhm=x_fwhm.default, y_fwhm=y_fwhm.default,
                 theta=theta.default, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, x_fwhm=x_fwhm,
                         y_fwhm=y_fwhm, theta=theta, **kwargs)

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return self.flux / (2 * np.pi * self.x_sigma * self.y_sigma)

    @property
    def x_sigma(self):
        """
        Gaussian sigma (standard deviation) along the x axis.
        """
        return self.x_fwhm * GAUSSIAN_FWHM_TO_SIGMA

    @property
    def y_sigma(self):
        """
        Gaussian sigma (standard deviation) along the y axis.
        """
        return self.y_fwhm * GAUSSIAN_FWHM_TO_SIGMA

    def bounding_box(self, factor=5.5):
        """
        Return a bounding box defining the limits of the model.

        The default offset from the mean is 5.5-sigma, corresponding to
        a relative error < 1e-7. The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y standard deviations used to
            define the limits. The default is 5.5.

        Returns
        -------
        bounding_box : tuple
            A bounding box defining the limits of the model in each
            dimension as ``((y_low, y_high), (x_low, x_high))``.

        Examples
        --------
        >>> from photutils.psf import GaussianPSF
        >>> model = GaussianPSF(x_0=0, y_0=0, x_fwhm=1, y_fwhm=2)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-2.33563, upper=2.33563)
                y: Interval(lower=-4.67127, upper=4.67127)
            }
            model=GaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        a = factor * self.x_sigma
        b = factor * self.y_sigma
        dx, dy = ellipse_extent(a, b, self.theta)

        return ((self.y_0 - dy, self.y_0 + dy),
                (self.x_0 - dx, self.x_0 + dx))

    def evaluate(self, x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)
        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        sin2t = np.sin(2.0 * theta)
        xstd = x_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        ystd = y_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        xstd2 = xstd ** 2
        ystd2 = ystd ** 2
        xdiff = x - x_0
        ydiff = y - y_0
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
        amplitude = flux / (2 * np.pi * xstd * ystd)
        return amplitude * np.exp(
            -(a * xdiff**2) - (b * xdiff * ydiff) - (c * ydiff**2))

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the partial derivatives of the 2D Gaussian function
        with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)

        cost = np.cos(theta)
        sint = np.sin(theta)
        cost2 = cost ** 2
        sint2 = sint ** 2
        cos2t = np.cos(2.0 * theta)
        sin2t = np.sin(2.0 * theta)
        xstd = x_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        ystd = y_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        xstd2 = xstd ** 2
        ystd2 = ystd ** 2
        xstd3 = xstd ** 3
        ystd3 = ystd ** 3
        xdiff = x - x_0
        ydiff = y - y_0
        xdiff2 = xdiff ** 2
        ydiff2 = ydiff ** 2
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
        amplitude = flux / (2 * np.pi * xstd * ystd)
        exp = np.exp(-(a * xdiff2) - (b * xdiff * ydiff) - (c * ydiff2))
        g = amplitude * exp

        da_dtheta = sint * cost * ((1.0 / ystd2) - (1.0 / xstd2))
        db_dtheta = (cos2t / xstd2) - (cos2t / ystd2)
        dc_dtheta = -da_dtheta

        da_dxstd = -cost2 / xstd3
        db_dxstd = -sin2t / xstd3
        dc_dxstd = -sint2 / xstd3

        da_dystd = -sint2 / ystd3
        db_dystd = sin2t / ystd3
        dc_dystd = -cost2 / ystd3

        dg_dflux = g / flux
        dg_dx_0 = g * ((2.0 * a * xdiff) + (b * ydiff))
        dg_dy_0 = g * ((b * xdiff) + (2.0 * c * ydiff))

        damp_dxstd = -amplitude / xstd
        damp_dystd = -amplitude / ystd
        dexp_dxstd = -exp * (da_dxstd * xdiff2
                             + db_dxstd * xdiff * ydiff
                             + dc_dxstd * ydiff2)
        dexp_dystd = -exp * (da_dystd * xdiff2
                             + db_dystd * xdiff * ydiff
                             + dc_dystd * ydiff2)
        dg_dxstd = damp_dxstd * exp + amplitude * dexp_dxstd
        dg_dystd = damp_dystd * exp + amplitude * dexp_dystd

        # chain rule for change of variables from sigma to fwhm
        # std => fwhm * GAUSSIAN_FWHM_TO_SIGMA
        # dstd/dfwhm => GAUSSIAN_FWHM_TO_SIGMA
        dg_dxfwhm = dg_dxstd * GAUSSIAN_FWHM_TO_SIGMA
        dg_dyfwhm = dg_dystd * GAUSSIAN_FWHM_TO_SIGMA

        dg_dtheta = g * (-(da_dtheta * xdiff2 + db_dtheta * xdiff * ydiff
                           + dc_dtheta * ydiff2))
        # chain rule for unit change;
        # theta[rad] => theta[deg] * pi / 180; drad/dtheta = pi / 180
        dg_dtheta *= np.pi / 180.0

        return [dg_dflux, dg_dx_0, dg_dy_0, dg_dxfwhm, dg_dyfwhm, dg_dtheta]

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            raise UnitsError("Units of 'x' and 'y' inputs should match")

        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'x_fwhm': inputs_unit[self.inputs[0]],
                'y_fwhm': inputs_unit[self.inputs[0]],
                'theta': u.deg,
                'flux': outputs_unit[self.outputs[0]]}


class CircularGaussianPSF(Fittable2DModel):
    r"""
    A circular 2D Gaussian PSF model.

    This model is evaluated by sampling the 2D Gaussian at the input
    coordinates. The Gaussian is normalized such that the analytical
    integral over the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x axis.

    y_0 : float, optional
        Position of the peak along the y axis.

    fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    GaussianPSF, GaussianPRF, CircularGaussianPRF

    Notes
    -----
    The circular Gaussian function is defined as:

    .. math::

        f(x, y) = \frac{F}{2 \pi \sigma^{2}}
                  \exp \left( {\frac{-(x - x_{0})^{2} - (y - y_{0})^{2}}
                             {2 \sigma^{2}}} \right)

    where :math:`F` is the total integrated flux, :math:`(x_{0}, y_{0})`
    is the position of the peak, and :math:`\sigma` is the standard
    deviation, respectively.

    The FWHM of the Gaussian is given by:

    .. math::

        \rm{FWHM} = 2 \sigma \sqrt{2 \ln{2}}

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) dx dy = F

    The ``fwhm`` parameter is fixed by default. If you wish to fit this
    parameter, set the ``fixed`` attribute to `False`, e.g.,::

        >>> from photutils.psf import CircularGaussianPSF
        >>> model = CircularGaussianPSF()
        >>> model.fwhm.fixed = False

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import CircularGaussianPSF
        model = CircularGaussianPSF(flux=71.4, x_0=24.3, y_0=25.2, fwhm=10.1)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        plt.imshow(data, origin='lower', interpolation='nearest')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    fwhm = Parameter(
        default=1, description='FWHM of the Gaussian', fixed=True)

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 fwhm=fwhm.default, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, fwhm=fwhm, **kwargs)

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return self.flux / (2 * np.pi * self.x_sigma * self.y_sigma)

    @property
    def sigma(self):
        """
        Gaussian sigma (standard deviation).
        """
        return self.fwhm * GAUSSIAN_FWHM_TO_SIGMA

    def bounding_box(self, factor=5.5):
        """
        Return a bounding box defining the limits of the model.

        The default offset from the mean is 5.5-sigma, corresponding to
        a relative error < 1e-7.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviation used to define the
            limits. The default is 5.5.

        Returns
        -------
        bounding_box : tuple
            A bounding box defining the limits of the model in each
            dimension as ``((y_low, y_high), (x_low, x_high))``.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPSF
        >>> model = CircularGaussianPSF(x_0=0, y_0=0, fwhm=2)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.67127, upper=4.67127)
                y: Interval(lower=-4.67127, upper=4.67127)
            }
            model=CircularGaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        dx = factor * self.sigma
        return ((self.y_0 - dx, self.y_0 + dx),
                (self.x_0 - dx, self.x_0 + dx))

    def evaluate(self, x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        return GaussianPSF().evaluate(x, y, flux, x_0, y_0, fwhm, fwhm, 0.0)

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the partial derivatives of the 2D Gaussian function
        with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        return GaussianPSF().fit_deriv(x, y, flux, x_0, y_0, fwhm, fwhm,
                                       0.0)[:-2]

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'fwhm': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}


class GaussianPRF(Fittable2DModel):
    r"""
    A 2D Gaussian PSF model integrated over pixels.

    This model is evaluated by integrating the 2D Gaussian over the
    input coordinate pixels, and is equivalent to assuming the PSF is
    2D Gaussian at a *sub-pixel* level. Because it is integrated over
    pixels, this model is considered a PRF instead of a PSF.

    The Gaussian is normalized such that the analytical integral over
    the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x axis.

    y_0 : float, optional
        Position of the peak along the y axis.

    x_fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian along the
        x axis.

    y_fwhm : float, optional
        FWHM of the Gaussian along the y axis.

    theta : float, optional
        The counterclockwise rotation angle either as a float (in
        degrees) or a `~astropy.units.Quantity` angle (optional).

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    GaussianPSF, CircularGaussianPSF, CircularGaussianPRF

    Notes
    -----
    The Gaussian function is defined as:

    .. math::

        f(x, y) =
            \frac{F}{4}
            \left[
                {\rm erf} \left(
                    \frac{x^\prime + 0.5}{\sqrt{2} \sigma_{x}} \right) -
                {\rm erf} \left(
                    \frac{x^\prime - 0.5}{\sqrt{2} \sigma_{x}} \right)
            \right]
            \left[
                {\rm erf} \left(
                    \frac{y^\prime + 0.5}{\sqrt{2} \sigma_{y}} \right) -
                {\rm erf} \left(
                    \frac{y^\prime - 0.5}{\sqrt{2} \sigma_{y}} \right)
            \right]

    where :math:`F` is the total integrated flux, :math:`\sigma_{x}`
    and :math:`\sigma_{y}` are the standard deviations along the x
    and y axes, respectively, and :math:`{\rm erf}` denotes the error
    function.

    .. math::

        x^\prime = (x - x_0) \cos(\theta) + (y - y_0) \sin(\theta)

        y^\prime = -(x - x_0) \sin(\theta) + (y - y_0) \cos(\theta)

    where :math:`(x_{0}, y_{0})` is the position of the peak and
    :math:`\theta` is the rotation angle of the Gaussian.

    The FWHMs of the Gaussian along the x and y axes are given by:

    .. math::

        \rm{FWHM}_{x} = 2 \sigma_{x} \sqrt{2 \ln{2}}

        \rm{FWHM}_{y} = 2 \sigma_{y} \sqrt{2 \ln{2}}

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) dx dy = F

    The ``x_fwhm``, ``y_fwhm``, and ``theta`` parameters are fixed by
    default. If you wish to fit these parameters, set the ``fixed``
    attribute to `False`, e.g.,::

        >>> from photutils.psf import GaussianPRF
        >>> model = GaussianPRF()
        >>> model.x_fwhm.fixed = False
        >>> model.y_fwhm.fixed = False
        >>> model.theta.fixed = False

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import GaussianPRF
        model = GaussianPRF(flux=71.4, x_0=24.3, y_0=25.2, x_fwhm=10.1,
                            y_fwhm=5.82, theta=21.7)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        plt.imshow(data, origin='lower', interpolation='nearest')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    x_fwhm = Parameter(
        default=1, description='FWHM of the Gaussian along the x axis',
        fixed=True)
    y_fwhm = Parameter(
        default=1, description='FWHM of the Gaussian along the y axis',
        fixed=True)
    theta = Parameter(
        default=0.0, description=('CCW rotation angle either as a float (in '
                                  'degrees) or a Quantity angle (optional)'),
        fixed=True)

    _erf = None

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 x_fwhm=x_fwhm.default, y_fwhm=y_fwhm.default,
                 theta=theta.default, **kwargs):

        if self._erf is None:
            from scipy.special import erf
            self.__class__._erf = erf

        super().__init__(flux=flux, x_0=x_0, y_0=y_0, x_fwhm=x_fwhm,
                         y_fwhm=y_fwhm, theta=theta, **kwargs)

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return self.flux / (2 * np.pi * self.x_sigma * self.y_sigma)

    @property
    def x_sigma(self):
        """
        Gaussian sigma (standard deviation) along the x axis.
        """
        return self.x_fwhm * GAUSSIAN_FWHM_TO_SIGMA

    @property
    def y_sigma(self):
        """
        Gaussian sigma (standard deviation) along the y axis.
        """
        return self.y_fwhm * GAUSSIAN_FWHM_TO_SIGMA

    def bounding_box(self, factor=5.5):
        """
        Return a bounding box defining the limits of the model.

        The default offset from the mean is 5.5-sigma, corresponding to
        a relative error < 1e-7. The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y standard deviations used to
            define the limits. The default is 5.5.

        Returns
        -------
        bounding_box : tuple
            A bounding box defining the limits of the model in each
            dimension as ``((y_low, y_high), (x_low, x_high))``.

        Examples
        --------
        >>> from photutils.psf import GaussianPRF
        >>> model = GaussianPRF(x_0=0, y_0=0, x_fwhm=1, y_fwhm=2)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-2.33563, upper=2.33563)
                y: Interval(lower=-4.67127, upper=4.67127)
            }
            model=GaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        a = factor * self.x_sigma
        b = factor * self.y_sigma
        dx, dy = ellipse_extent(a, b, self.theta)

        return ((self.y_0 - dy, self.y_0 + dy),
                (self.x_0 - dx, self.x_0 + dx))

    def evaluate(self, x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)

        x_sigma = x_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        y_sigma = y_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        dx = x - x_0
        dy = y - y_0
        cost = np.cos(theta)
        sint = np.sin(theta)
        x0 = dx * cost + dy * sint
        y0 = -dx * sint + dy * cost

        return (flux / 4.0
                * ((self._erf((x0 + 0.5) / (np.sqrt(2) * x_sigma))
                    - self._erf((x0 - 0.5) / (np.sqrt(2) * x_sigma)))
                   * (self._erf((y0 + 0.5) / (np.sqrt(2) * y_sigma))
                      - self._erf((y0 - 0.5) / (np.sqrt(2) * y_sigma)))))

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            raise UnitsError("Units of 'x' and 'y' inputs should match")

        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'x_fwhm': inputs_unit[self.inputs[0]],
                'y_fwhm': inputs_unit[self.inputs[0]],
                'theta': u.deg,
                'flux': outputs_unit[self.outputs[0]]}


class CircularGaussianPRF(Fittable2DModel):
    r"""
    A circular 2D Gaussian PSF model integrated over pixels.

    This model is evaluated by integrating the 2D Gaussian over the
    input coordinate pixels, and is equivalent to assuming the PSF is
    2D Gaussian at a *sub-pixel* level. Because it is integrated over
    pixels, this model is considered a PRF instead of a PSF.

    The Gaussian is normalized such that the analytical integral over
    the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x axis.

    y_0 : float, optional
        Position of the peak along the y axis.

    fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    GaussianPRF, GaussianPSF, CircularGaussianPSF

    Notes
    -----
    The Gaussian function is defined as:

    .. math::

        f(x, y) =
            \frac{F}{4}
            \left[
                {\rm erf} \left(
                    \frac{x - x_0 + 0.5}{\sqrt{2} \sigma} \right) -
                {\rm erf} \left(
                    \frac{x - x_0 - 0.5}{\sqrt{2} \sigma} \right)
            \right]
            \left[
                {\rm erf} \left(
                    \frac{y - y_0 + 0.5}{\sqrt{2} \sigma} \right) -
                {\rm erf} \left(
                    \frac{y - y_0 - 0.5}{\sqrt{2} \sigma} \right)
            \right]

    where :math:`F` is the total integrated flux, :math:`(x_{0},
    y_{0})` is the position of the peak, :math:`\sigma` is the standard
    deviation of the Gaussian, and :math:`{\rm erf}` denotes the error
    function.

    The FWHMs of the Gaussian is given by:

    .. math::

        \rm{FWHM} = 2 \sigma \sqrt{2 \ln{2}}

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) dx dy = F

    The ``fwhm`` parameter is fixed by default. If you wish to fit this
    parameter, set the ``fixed`` attribute to `False`, e.g.,::

        >>> from photutils.psf import CircularGaussianPRF
        >>> model = CircularGaussianPRF()
        >>> model.fwhm.fixed = False

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import CircularGaussianPRF
        model = CircularGaussianPRF(flux=71.4, x_0=24.3, y_0=25.2, fwhm=10.1)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        plt.imshow(data, origin='lower', interpolation='nearest')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    fwhm = Parameter(
        default=1, description='FWHM of the Gaussian', fixed=True)

    _erf = None

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 fwhm=fwhm.default, **kwargs):

        if self._erf is None:
            from scipy.special import erf
            self.__class__._erf = erf

        super().__init__(flux=flux, x_0=x_0, y_0=y_0, fwhm=fwhm, **kwargs)

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return self.flux / (2 * np.pi * self.sigma**2)

    @property
    def sigma(self):
        """
        Gaussian sigma (standard deviation).
        """
        return self.fwhm * GAUSSIAN_FWHM_TO_SIGMA

    def bounding_box(self, factor=5.5):
        """
        Return a bounding box defining the limits of the model.

        The default offset from the mean is 5.5-sigma, corresponding to
        a relative error < 1e-7. The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y standard deviations used to
            define the limits. The default is 5.5.

        Returns
        -------
        bounding_box : tuple
            A bounding box defining the limits of the model in each
            dimension as ``((y_low, y_high), (x_low, x_high))``.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPRF
        >>> model = CircularGaussianPRF(x_0=0, y_0=0, fwhm=1)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-2.33563, upper=2.33563)
                y: Interval(lower=-2.33563, upper=2.33563)
            }
            model=CircularGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        dx = factor * self.sigma
        return ((self.y_0 - dx, self.y_0 + dx),
                (self.x_0 - dx, self.x_0 + dx))

    def evaluate(self, x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        return GaussianPRF().evaluate(x, y, flux, x_0, y_0, fwhm, fwhm, 0.0)

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'fwhm': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}


class IntegratedGaussianPRF(Fittable2DModel):
    r"""
    A circular 2D Gaussian PSF model integrated over pixels.

    This model is evaluated by integrating the 2D Gaussian over the
    input coordinate pixels, and is equivalent to assuming the PSF is
    2D Gaussian at a *sub-pixel* level. Because it is integrated over
    pixels, this model is considered a PRF instead of a PSF.

    The Gaussian is normalized such that the analytical integral over
    the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak in x direction.

    y_0 : float, optional
        Position of the peak in y direction.

    sigma : float, optional
        Width of the Gaussian PSF.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` parent class.

    See Also
    --------
    GaussianPSF, GaussianPRF, CircularGaussianPSF, CircularGaussianPRF

    Notes
    -----
    This model is evaluated according to the following formula:

    .. math::

        f(x, y) =
            \frac{F}{4}
            \left[
                {\rm erf} \left(\frac{x - x_0 + 0.5}
                                     {\sqrt{2} \sigma} \right)  -
                {\rm erf} \left(\frac{x - x_0 - 0.5}
                                     {\sqrt{2} \sigma} \right)
            \right]
            \left[
                {\rm erf} \left(\frac{y - y_0 + 0.5}
                                     {\sqrt{2} \sigma} \right) -
                {\rm erf} \left(\frac{y - y_0 - 0.5}
                                     {\sqrt{2} \sigma} \right)
            \right]

    where :math:`F` is the total integrated flux, :math:`(x_{0},
    y_{0})` is the position of the peak, :math:`\sigma` is the standard
    deviation of the Gaussian, and :math:`{\rm erf}` denotes the error
    function.

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) dx dy = F

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import IntegratedGaussianPRF
        model = IntegratedGaussianPRF(flux=71.4, x_0=24.3, y_0=25.2,
                                      sigma=5.1)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        plt.imshow(data, origin='lower', interpolation='nearest')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    sigma = Parameter(
        default=1, description='Sigma (standard deviation) of the Gaussian',
        fixed=True)

    _erf = None

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 sigma=sigma.default, **kwargs):

        if self._erf is None:
            from scipy.special import erf
            self.__class__._erf = erf

        super().__init__(sigma=sigma, x_0=x_0, y_0=y_0, flux=flux, **kwargs)

    def bounding_box(self, factor=5.5):
        """
        Return a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float
            The multiple of `sigma` used to define the limits. The
            default is 5.5, corresponding to a relative flux error less
            than 5e-9.

        Returns
        -------
        bounding_box : `astropy.modeling.bounding_box.ModelBoundingBox`
            A bounding box defining the limits of the model.

        Examples
        --------
        >>> from photutils.psf import IntegratedGaussianPRF
        >>> model = IntegratedGaussianPRF(x_0=0, y_0=0, sigma=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-11.0, upper=11.0)
                y: Interval(lower=-11.0, upper=11.0)
            }
            model=IntegratedGaussianPRF(inputs=('x', 'y'))
            order='C'
        )

        This range can be set directly (see: `Model.bounding_box
        <astropy.modeling.Model.bounding_box>`) or by using a different
        factor, like:

        >>> model.bounding_box = model.bounding_box(factor=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.0, upper=4.0)
                y: Interval(lower=-4.0, upper=4.0)
            }
            model=IntegratedGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        delta = factor * self.sigma
        return (
            (self.y_0 - delta, self.y_0 + delta),
            (self.x_0 - delta, self.x_0 + delta),
        )

    def evaluate(self, x, y, flux, x_0, y_0, sigma):
        """
        Model function Gaussian PSF model.

        Parameters
        ----------
        x, y : float or array_like
            The coordinates at which to evaluate the model.

        flux : float
            The total flux of the star.

        x_0, y_0 : float
            The position of the star.

        sigma : float
            The width of the Gaussian PRF.

        Returns
        -------
        evaluated_model : `~numpy.ndarray`
            The evaluated model.
        """
        return (flux / 4
                * ((self._erf((x - x_0 + 0.5) / (np.sqrt(2) * sigma))
                    - self._erf((x - x_0 - 0.5) / (np.sqrt(2) * sigma)))
                   * (self._erf((y - y_0 + 0.5) / (np.sqrt(2) * sigma))
                      - self._erf((y - y_0 - 0.5) / (np.sqrt(2) * sigma)))))


class FittableImageModel(Fittable2DModel):
    r"""
    A fittable image model allowing for intensity scaling and
    translations.

    This class takes 2D image data and computes the values of
    the model at arbitrary locations, including fractional pixel
    positions, within the image using spline interpolation provided by
    :py:class:`~scipy.interpolate.RectBivariateSpline`.

    The fittable model provided by this class has three model
    parameters: an image intensity scaling factor (``flux``) which
    is applied to (normalized) image, and two positional parameters
    (``x_0`` and ``y_0``) indicating the location of a feature in the
    coordinate grid on which the model is to be evaluated.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        Array containing the 2D image.

    flux : float, optional
        Intensity scaling factor for image data. If ``flux`` is `None`,
        then the normalization constant will be computed so that the
        total flux of the model's image data is 1.0.

    x_0, y_0 : float, optional
        Position of a feature in the image in the output coordinate grid
        on which the model is evaluated.

    normalize : bool, optional
        Indicates whether or not the model should be build on normalized
        input image data. If true, then the normalization constant (*N*)
        is computed so that

        .. math::
            N \cdot C \cdot \sum\limits_{i,j} D_{i,j} = 1,

        where *N* is the normalization constant, *C* is correction
        factor given by the parameter ``normalization_correction``, and
        :math:`D_{i,j}` are the elements of the input image ``data``
        array.

    normalization_correction : float, optional
        A strictly positive number that represents correction that needs
        to be applied to model's data normalization (see *C* in the
        equation in the comments to ``normalize`` for more details).
        A possible application for this parameter is to account for
        aperture correction. Assuming model's data represent a PSF to be
        fitted to some target star, we set ``normalization_correction``
        to the aperture correction that needs to be applied to the
        model. That is, ``normalization_correction`` in this case should
        be set to the ratio between the total flux of the PSF (including
        flux outside model's data) to the flux of model's data. Then,
        best fitted value of the ``flux`` model parameter will represent
        an aperture-corrected flux of the target star. In the case of
        aperture correction, ``normalization_correction`` should be a
        value larger than one, as the total flux, including regions
        outside of the aperture, should be larger than the flux inside
        the aperture, and thus the correction is applied as an inversely
        multiplied factor.

    origin : tuple, None, optional
        A reference point in the input image ``data`` array. When origin
        is `None`, origin will be set at the middle of the image array.

        If ``origin`` represents the location of a feature (e.g., the
        position of an intensity peak) in the input ``data``, then
        model parameters ``x_0`` and ``y_0`` show the location of this
        peak in an another target image to which this model was fitted.
        Fundamentally, it is the coordinate in the model's image data
        that should map to coordinate (``x_0``, ``y_0``) of the output
        coordinate system on which the model is evaluated.

        Alternatively, when ``origin`` is set to ``(0, 0)``, then model
        parameters ``x_0`` and ``y_0`` are shifts by which model's image
        should be translated in order to match a target image.

    oversampling : int or array_like (int)
        The integer oversampling factor(s) of the ePSF relative to the
        input ``stars`` along each axis. If ``oversampling`` is a scalar
        then it will be used for both axes. If ``oversampling`` has two
        elements, they must be in ``(y, x)`` order.

    fill_value : float, optional
        The value to be returned by the `evaluate` or
        ``astropy.modeling.Model.__call__`` methods when evaluation is
        performed outside the definition domain of the model.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed directly to
        the `compute_interpolator` method. See `compute_interpolator`
        for more details.
    """

    flux = Parameter(description='Intensity scaling factor for image data.',
                     default=1.0)
    x_0 = Parameter(description='X-position of a feature in the image in '
                    'the output coordinate grid on which the model is '
                    'evaluated.', default=0.0)
    y_0 = Parameter(description='Y-position of a feature in the image in '
                    'the output coordinate grid on which the model is '
                    'evaluated.', default=0.0)

    def __init__(self, data, *, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, normalize=False,
                 normalization_correction=1.0, origin=None, oversampling=1,
                 fill_value=0.0, **kwargs):

        self._fill_value = fill_value
        self._img_norm = None
        self._normalization_status = 0 if normalize else 2
        self._store_interpolator_kwargs(**kwargs)
        self._oversampling = as_pair('oversampling', oversampling,
                                     lower_bound=(0, 1))

        if normalization_correction <= 0:
            raise ValueError("'normalization_correction' must be strictly "
                             'positive.')
        self._normalization_correction = normalization_correction
        self._normalization_constant = 1.0 / self._normalization_correction

        self._data = np.array(data, copy=True, dtype=float)

        if not np.all(np.isfinite(self._data)):
            raise ValueError("All elements of input 'data' must be finite.")

        # set input image related parameters:
        self._ny, self._nx = self._data.shape
        self._shape = self._data.shape
        if self._data.size < 1:
            raise ValueError('Image data array cannot be zero-sized.')

        # set the origin of the coordinate system in image's pixel grid:
        self.origin = origin

        flux = self._initial_norm(flux, normalize)

        super().__init__(flux, x_0, y_0)

        # initialize interpolator:
        self.compute_interpolator(**kwargs)

    def _initial_norm(self, flux, normalize):

        if flux is None:
            if self._img_norm is None:
                self._img_norm = self._compute_raw_image_norm()
            flux = self._img_norm

        self._compute_normalization(normalize)

        return flux

    def _compute_raw_image_norm(self):
        """
        Helper function that computes the uncorrected inverse
        normalization factor of input image data. This quantity is
        computed as the *sum of all pixel values*.

        .. note::
            This function is intended to be overridden in a subclass if
            one desires to change the way the normalization factor is
            computed.
        """
        return np.sum(self._data, dtype=float)

    def _compute_normalization(self, normalize=True):
        r"""
        Helper function that computes (corrected) normalization factor
        of the original image data.

        This quantity is computed as the inverse "raw image norm"
        (or total "flux" of model's image) corrected by the
        ``normalization_correction``:

        .. math::
            N = 1/(\Phi * C),

        where :math:`\Phi` is the "total flux" of model's image as
        computed by `_compute_raw_image_norm` and *C* is the
        normalization correction factor. :math:`\Phi` is computed only
        once if it has not been previously computed. Otherwise, the
        existing (stored) value of :math:`\Phi` is not modified as
        :py:class:`FittableImageModel` does not allow image data to be
        modified after the object is created.

        .. note::
            Normally, this function should not be called by the
            end-user. It is intended to be overridden in a subclass if
            one desires to change the way the normalization factor is
            computed.
        """
        self._normalization_constant = 1.0 / self._normalization_correction

        if normalize:
            # compute normalization constant so that
            # N*C*sum(data) = 1:
            if self._img_norm is None:
                self._img_norm = self._compute_raw_image_norm()

            if self._img_norm != 0.0 and np.isfinite(self._img_norm):
                self._normalization_constant /= self._img_norm
                self._normalization_status = 0

            else:
                self._normalization_constant = 1.0
                self._normalization_status = 1
                warnings.warn('Overflow encountered while computing '
                              'normalization constant. Normalization '
                              'constant will be set to 1.', AstropyUserWarning)

        else:
            self._normalization_status = 2

    @property
    def oversampling(self):
        """
        The factor by which the stored image is oversampled.

        An input to this model is multiplied by this factor to yield the
        index into the stored image.
        """
        return self._oversampling

    @property
    def data(self):
        """
        Get original image data.
        """
        return self._data

    @property
    def normalized_data(self):
        """
        Get normalized and/or intensity-corrected image data.
        """
        return self._normalization_constant * self._data

    @property
    def normalization_constant(self):
        """
        Get normalization constant.
        """
        return self._normalization_constant

    @property
    def normalization_status(self):
        """
        Get normalization status.

        Possible status values are:

        - 0: **Performed**. Model has been successfully normalized at
             user's request.
        - 1: **Failed**. Attempt to normalize has failed.
        - 2: **NotRequested**. User did not request model to be normalized.
        """
        return self._normalization_status

    @property
    def normalization_correction(self):
        """
        Set/Get flux correction factor.

        .. note::
            When setting correction factor, model's flux will be
            adjusted accordingly such that if this model was a good fit
            to some target image before, then it will remain a good fit
            after correction factor change.
        """
        return self._normalization_correction

    @normalization_correction.setter
    def normalization_correction(self, normalization_correction):
        old_cf = self._normalization_correction
        self._normalization_correction = normalization_correction
        self._compute_normalization(normalize=self._normalization_status != 2)

        # adjust model's flux so that if this model was a good fit to
        # some target image, then it will remain a good fit after
        # correction factor change:
        self.flux *= normalization_correction / old_cf

    @property
    def shape(self):
        """
        A tuple of dimensions of the data array in numpy style (ny, nx).
        """
        return self._shape

    @property
    def nx(self):
        """
        Number of columns in the data array.
        """
        return self._nx

    @property
    def ny(self):
        """
        Number of rows in the data array.
        """
        return self._ny

    @property
    def origin(self):
        """
        A tuple of ``x`` and ``y`` coordinates of the origin of the
        coordinate system in terms of pixels of model's image.

        When setting the coordinate system origin, a tuple of two
        integers or floats may be used. If origin is set to `None`, the
        origin of the coordinate system will be set to the middle of the
        data array (``(npix-1)/2.0``).

        .. warning::
            Modifying ``origin`` will not adjust (modify) model's
            parameters ``x_0`` and ``y_0``.
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
            raise TypeError('Parameter "origin" must be either None or an '
                            'iterable with two elements.')

    @property
    def x_origin(self):
        """
        X-coordinate of the origin of the coordinate system.
        """
        return self._x_origin

    @property
    def y_origin(self):
        """
        Y-coordinate of the origin of the coordinate system.
        """
        return self._y_origin

    @property
    def fill_value(self):
        """
        Fill value to be returned for coordinates outside of the domain
        of definition of the interpolator.

        If ``fill_value`` is `None`, then values outside of the domain
        of definition are the ones returned by the interpolator.
        """
        return self._fill_value

    @fill_value.setter
    def fill_value(self, fill_value):
        self._fill_value = fill_value

    def _store_interpolator_kwargs(self, **kwargs):
        """
        Store interpolator keyword arguments.

        This function should be called in a subclass whenever model's
        interpolator is (re-)computed.
        """
        self._interpolator_kwargs = copy.deepcopy(kwargs)

    @property
    def interpolator_kwargs(self):
        """
        Get current interpolator's arguments used when interpolator was
        created.
        """
        return self._interpolator_kwargs

    def compute_interpolator(self, **kwargs):
        """
        Compute/define the interpolating spline.

        This function can be overridden in a subclass to define custom
        interpolators.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional optional keyword arguments:

            - **degree** : int, tuple, optional
                Degree of the interpolating spline. A tuple can be used
                to provide different degrees for the X- and Y-axes.
                Default value is degree=3.

            - **s** : float, optional
                Non-negative smoothing factor. Default
                value s=0 corresponds to interpolation. See
                :py:class:`~scipy.interpolate.RectBivariateSpline` for
                more details.

        Notes
        -----
        * When subclassing :py:class:`FittableImageModel` for the
          purpose of overriding :py:func:`compute_interpolator`, the
          :py:func:`evaluate` may need to overridden as well depending
          on the behavior of the new interpolator. In addition, for
          improved future compatibility, make sure that the overriding
          method stores keyword arguments ``kwargs`` by calling
          ``_store_interpolator_kwargs`` method.

        * Use caution when modifying interpolator's degree or smoothness
          in a computationally intensive part of the code as it may
          decrease code performance due to the need to recompute
          interpolator.
        """
        from scipy.interpolate import RectBivariateSpline

        if 'degree' in kwargs:
            degree = kwargs['degree']
            if hasattr(degree, '__iter__') and len(degree) == 2:
                degx = int(degree[0])
                degy = int(degree[1])
            else:
                degx = int(degree)
                degy = int(degree)
            if degx < 0 or degy < 0:
                raise ValueError('Interpolator degree must be a non-negative '
                                 'integer')
        else:
            degx = 3
            degy = 3

        smoothness = kwargs.get('s', 0)

        x = np.arange(self._nx, dtype=float)
        y = np.arange(self._ny, dtype=float)
        self.interpolator = RectBivariateSpline(
            x, y, self._data.T, kx=degx, ky=degy, s=smoothness
        )

        self._store_interpolator_kwargs(**kwargs)

    def evaluate(self, x, y, flux, x_0, y_0, *, use_oversampling=True):
        """
        Evaluate the model on some input variables and provided model
        parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            The total flux of the source.

        x_0, y_0 : float
            The x and y positions of the feature in the image in the
            output coordinate grid on which the model is evaluated.

        use_oversampling : bool, optional
            Whether to use the oversampling factor to calculate the
            model pixel indices. The default is `True`, which means the
            input indices will be multiplied by this factor.

        Returns
        -------
        evaluated_model : `~numpy.ndarray`
            The evaluated model.
        """
        if use_oversampling:
            xi = self._oversampling[1] * (np.asarray(x) - x_0)
            yi = self._oversampling[0] * (np.asarray(y) - y_0)
        else:
            xi = np.asarray(x) - x_0
            yi = np.asarray(y) - y_0

        xi = xi.astype(float)
        yi = yi.astype(float)
        xi += self._x_origin
        yi += self._y_origin

        f = flux * self._normalization_constant
        evaluated_model = f * self.interpolator.ev(xi, yi)

        if self._fill_value is not None:
            # find indices of pixels that are outside the input pixel grid and
            # set these pixels to the 'fill_value':
            invalid = (((xi < 0) | (xi > self._nx - 1))
                       | ((yi < 0) | (yi > self._ny - 1)))
            evaluated_model[invalid] = self._fill_value

        return evaluated_model


class EPSFModel(FittableImageModel):
    """
    A class that models an effective PSF (ePSF).

    The EPSFModel is normalized such that the sum of the PSF over the
    (undersampled) pixels within the input ``norm_radius`` is 1.0.
    This means that when the EPSF is fit to stars, the resulting flux
    corresponds to aperture photometry within a circular aperture of
    radius ``norm_radius``.

    While this class is a subclass of `FittableImageModel`, it is very
    similar. The primary differences/motivation are a few additional
    parameters necessary specifically for ePSFs.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        Array containing the 2D image.

    flux : float, optional
        Intensity scaling factor for image data.

    x_0, y_0 : float, optional
        Position of a feature in the image in the output coordinate grid
        on which the model is evaluated.

    normalize : bool, optional
        Indicates whether or not the model should be build on normalized
        input image data.

    normalization_correction : float, optional
        A strictly positive number that represents correction that needs
        to be applied to model's data normalization.

    origin : tuple, None, optional
        A reference point in the input image ``data`` array. When origin
        is `None`, origin will be set at the middle of the image array.

    oversampling : int or array_like (int)
        The integer oversampling factor(s) of the ePSF relative to the
        input ``stars`` along each axis. If ``oversampling`` is a scalar
        then it will be used for both axes. If ``oversampling`` has two
        elements, they must be in ``(y, x)`` order.

    fill_value : float, optional
        The value to be returned when evaluation is performed outside
        the domain of the model.

    norm_radius : float, optional
        The radius inside which the ePSF is normalized by the sum over
        undersampled integer pixel values inside a circular aperture.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed directly to
        the `compute_interpolator` method. See `compute_interpolator`
        for more details.
    """

    def __init__(self, data, *, flux=1.0, x_0=0.0, y_0=0.0, normalize=True,
                 normalization_correction=1.0, origin=None, oversampling=1,
                 fill_value=0.0, norm_radius=5.5, **kwargs):

        self._norm_radius = norm_radius

        super().__init__(data=data, flux=flux, x_0=x_0, y_0=y_0,
                         normalize=normalize,
                         normalization_correction=normalization_correction,
                         origin=origin, oversampling=oversampling,
                         fill_value=fill_value, **kwargs)

    def _initial_norm(self, flux, normalize):
        if flux is None:
            if self._img_norm is None:
                self._img_norm = self._compute_raw_image_norm()
            flux = self._img_norm

        if normalize:
            self._compute_normalization()
        else:
            self._img_norm = self._compute_raw_image_norm()

        return flux

    def _compute_raw_image_norm(self):
        """
        Compute the normalization of input image data as the flux within
        a given radius.
        """
        xypos = (self._nx / 2.0, self._ny / 2.0)
        # TODO: generalize "radius" (ellipse?) is oversampling is
        # different along x/y axes
        radius = self._norm_radius * self.oversampling[0]
        aper = CircularAperture(xypos, r=radius)
        flux, _ = aper.do_photometry(self._data, method='exact')
        return flux[0] / np.prod(self.oversampling)

    def _compute_normalization(self, normalize=True):
        """
        Helper function that computes (corrected) normalization factor
        of the original image data.

        For the ePSF this is defined as the sum over the inner N
        (default=5.5) pixels of the non-oversampled image. Will re-
        normalize the data to the value calculated.
        """
        if normalize:
            if self._img_norm is None:
                if np.sum(self._data) == 0:
                    self._img_norm = 1
                else:
                    self._img_norm = self._compute_raw_image_norm()

            if self._img_norm != 0.0 and np.isfinite(self._img_norm):
                self._data /= (self._img_norm * self._normalization_correction)
                self._normalization_status = 0
            else:
                self._normalization_status = 1
                self._img_norm = 1
                warnings.warn('Overflow encountered while computing '
                              'normalization constant. Normalization '
                              'constant will be set to 1.', AstropyUserWarning)
        else:
            self._normalization_status = 2

    @property
    def normalized_data(self):
        """
        Overloaded dummy function that also returns self._data, as the
        normalization occurs within _compute_normalization in EPSFModel,
        and as such self._data will sum, accounting for
        under/oversampled pixels, to 1/self._normalization_correction.
        """
        return self._data

    @FittableImageModel.origin.setter
    def origin(self, origin):
        if origin is None:
            self._x_origin = (self._nx - 1) / 2.0 / self.oversampling[1]
            self._y_origin = (self._ny - 1) / 2.0 / self.oversampling[0]
        elif (hasattr(origin, '__iter__') and len(origin) == 2):
            self._x_origin, self._y_origin = origin
        else:
            raise TypeError('Parameter "origin" must be either None or an '
                            'iterable with two elements.')

    def compute_interpolator(self, **kwargs):
        """
        Compute/define the interpolating spline.

        This function can be overridden in a subclass to define custom
        interpolators.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional optional keyword arguments:

            - **degree** : int, tuple, optional
                Degree of the interpolating spline. A tuple can be used
                to provide different degrees for the X- and Y-axes.
                Default value is degree=3.

            - **s** : float, optional
                Non-negative smoothing factor. Default
                value s=0 corresponds to interpolation. See
                :py:class:`~scipy.interpolate.RectBivariateSpline` for
                more details.

        Notes
        -----
        * When subclassing :py:class:`FittableImageModel` for the
          purpose of overriding :py:func:`compute_interpolator`, the
          :py:func:`evaluate` may need to overridden as well depending
          on the behavior of the new interpolator. In addition, for
          improved future compatibility, make sure that the overriding
          method stores keyword arguments ``kwargs`` by calling
          ``_store_interpolator_kwargs`` method.

        * Use caution when modifying interpolator's degree or smoothness
          in a computationally intensive part of the code as it may
          decrease code performance due to the need to recompute
          interpolator.
        """
        from scipy.interpolate import RectBivariateSpline

        if 'degree' in kwargs:
            degree = kwargs['degree']
            if hasattr(degree, '__iter__') and len(degree) == 2:
                degx = int(degree[0])
                degy = int(degree[1])
            else:
                degx = int(degree)
                degy = int(degree)
            if degx < 0 or degy < 0:
                raise ValueError('Interpolator degree must be a non-negative '
                                 'integer')
        else:
            degx = 3
            degy = 3

        smoothness = kwargs.get('s', 0)

        # Interpolator must be set to interpolate on the undersampled
        # pixel grid, going from 0 to len(undersampled_grid)
        x = np.arange(self._nx, dtype=float) / self.oversampling[1]
        y = np.arange(self._ny, dtype=float) / self.oversampling[0]
        self.interpolator = RectBivariateSpline(
            x, y, self._data.T, kx=degx, ky=degy, s=smoothness)

        self._store_interpolator_kwargs(**kwargs)

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Evaluate the model on some input variables and provided model
        parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            The total flux of the source.

        x_0, y_0 : float
            The x and y positions of the feature in the image in the
            output coordinate grid on which the model is evaluated.

        Returns
        -------
        evaluated_model : `~numpy.ndarray`
            The evaluated model.
        """
        xi = np.asarray(x) - x_0 + self._x_origin
        yi = np.asarray(y) - y_0 + self._y_origin

        evaluated_model = flux * self.interpolator.ev(xi, yi)

        if self._fill_value is not None:
            # find indices of pixels that are outside the input pixel
            # grid and set these pixels to the 'fill_value':
            invalid = (((xi < 0) | (xi > (self._nx - 1)
                                    / self.oversampling[1]))
                       | ((yi < 0) | (yi > (self._ny - 1)
                                      / self.oversampling[0])))
            evaluated_model[invalid] = self._fill_value

        return evaluated_model


class PRFAdapter(Fittable2DModel):
    """
    A model that adapts a supplied PSF model to act as a PRF.

    It integrates the PSF model over pixel "boxes". A critical built-in
    assumption is that the PSF model scale and location parameters are
    in *pixel* units.

    Parameters
    ----------
    psfmodel : a 2D model
        The model to assume as representative of the PSF.

    renormalize_psf : bool, optional
        If True, the model will be integrated from -inf to inf and
        re-scaled so that the total integrates to 1. Note that this
        renormalization only occurs *once*, so if the total flux of
        ``psfmodel`` depends on position, this will *not* be correct.

    flux : float, optional
        The total flux of the star.

    x_0 : float, optional
        The x position of the star.

    y_0 : float, optional
        The y position of the star.

    xname : str or None, optional
        The name of the ``psfmodel`` parameter that corresponds to the
        x-axis center of the PSF. If None, the model will be assumed to
        be centered at x=0.

    yname : str or None, optional
        The name of the ``psfmodel`` parameter that corresponds to the
        y-axis center of the PSF. If None, the model will be assumed to
        be centered at y=0.

    fluxname : str or None, optional
        The name of the ``psfmodel`` parameter that corresponds to
        the total flux of the star. If None, a scaling factor will
        be applied by the ``PRFAdapter`` instead of modifying the
        ``psfmodel``.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` parent class.

    Notes
    -----
    This current implementation of this class (using numerical
    integration for each pixel) is extremely slow, and only suited for
    experimentation over relatively few small regions.
    """

    flux = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)

    def __init__(self, psfmodel, *, renormalize_psf=True, flux=flux.default,
                 x_0=x_0.default, y_0=y_0.default, xname=None, yname=None,
                 fluxname=None, **kwargs):

        self.psfmodel = psfmodel.copy()

        if renormalize_psf:
            from scipy.integrate import dblquad
            self._psf_scale_factor = 1.0 / dblquad(self.psfmodel,
                                                   -np.inf, np.inf,
                                                   lambda x: -np.inf,
                                                   lambda x: np.inf)[0]
        else:
            self._psf_scale_factor = 1

        self.xname = xname
        self.yname = yname
        self.fluxname = fluxname

        # these can be used to adjust the integration behavior. Might be
        # used in the future to expose how the integration happens
        self._dblquadkwargs = {}

        super().__init__(n_models=1, x_0=x_0, y_0=y_0, flux=flux, **kwargs)

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        The evaluation function for PRFAdapter.

        Parameters
        ----------
        x, y : float or array_like
            The coordinates at which to evaluate the model.

        flux : float
            The total flux of the star.

        x_0, y_0 : float
            The position of the star.

        Returns
        -------
        evaluated_model : `~numpy.ndarray`
            The evaluated model.
        """
        if not np.isscalar(flux):
            flux = flux[0]
        if not np.isscalar(x_0):
            x_0 = x_0[0]
        if not np.isscalar(y_0):
            y_0 = y_0[0]

        if self.xname is None:
            dx = x - x_0
        else:
            dx = x
            setattr(self.psfmodel, self.xname, x_0)

        if self.xname is None:
            dy = y - y_0
        else:
            dy = y
            setattr(self.psfmodel, self.yname, y_0)

        if self.fluxname is None:
            return (flux * self._psf_scale_factor
                    * self._integrated_psfmodel(dx, dy))

        setattr(self.psfmodel, self.yname, flux * self._psf_scale_factor)
        return self._integrated_psfmodel(dx, dy)

    def _integrated_psfmodel(self, dx, dy):
        from scipy.integrate import dblquad

        # infer type/shape from the PSF model. Seems wasteful, but the
        # integration step is a *lot* more expensive so its just peanuts
        out = np.empty_like(self.psfmodel(dx, dy))
        outravel = out.ravel()
        for i, (xi, yi) in enumerate(zip(dx.ravel(), dy.ravel(), strict=True)):
            outravel[i] = dblquad(self.psfmodel,
                                  xi - 0.5, xi + 0.5,
                                  lambda x: yi - 0.5, lambda x: yi + 0.5,
                                  **self._dblquadkwargs)[0]
        return out
