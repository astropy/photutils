# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides mathematical functional PSF models.
"""

import astropy.units as u
import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.utils import ellipse_extent
from astropy.units import UnitsError

__all__ = ['GaussianPSF', 'CircularGaussianPSF', 'GaussianPRF',
           'CircularGaussianPRF', 'IntegratedGaussianPRF', 'MoffatPSF']

__doctest_requires__ = {'*': ['scipy']}

GAUSSIAN_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _gaussian_amplitude(flux, xsigma, ysigma):
    # output units should match the input flux units
    if isinstance(xsigma, u.Quantity):
        xsigma = xsigma.value
        ysigma = ysigma.value

    return flux / (2.0 * np.pi * xsigma * ysigma)


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
    CircularGaussianPSF, GaussianPRF, CircularGaussianPRF, MoffatPSF

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
        return _gaussian_amplitude(self.flux, self.x_sigma, self.y_sigma)

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

        The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y standard deviations (sigma) used
            to define the limits.

        Returns
        -------
        bounding_box : `astropy.modeling.bounding_box.ModelBoundingBox`
            A bounding box defining the limits of the model.

        Examples
        --------
        >>> from photutils.psf import GaussianPSF
        >>> model = GaussianPSF(x_0=0, y_0=0, x_fwhm=2, y_fwhm=3)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.6712699, upper=4.6712699)
                y: Interval(lower=-7.0069049, upper=7.0069048)
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

        # output units should match the input flux units
        if isinstance(xstd, u.Quantity):
            xstd = xstd.value
            ystd = ystd.value

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
    GaussianPSF, GaussianPRF, CircularGaussianPRF, MoffatPSF

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
        return _gaussian_amplitude(self.flux, self.sigma, self.sigma)

    @property
    def sigma(self):
        """
        Gaussian sigma (standard deviation).
        """
        return self.fwhm * GAUSSIAN_FWHM_TO_SIGMA

    def bounding_box(self, factor=5.5):
        """
        Return a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

        Returns
        -------
        bounding_box : `astropy.modeling.bounding_box.ModelBoundingBox`
            A bounding box defining the limits of the model.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPSF
        >>> model = CircularGaussianPSF(x_0=0, y_0=0, fwhm=2)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.6712699, upper=4.6712699)
                y: Interval(lower=-4.6712699, upper=4.6712699)
            }
            model=CircularGaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        delta = factor * self.sigma
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

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
    GaussianPSF, CircularGaussianPSF, CircularGaussianPRF, MoffatPSF

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
        return _gaussian_amplitude(self.flux, self.x_sigma, self.y_sigma)

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

        The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y standard deviations (sigma) used
            to define the limits.

        Returns
        -------
        bounding_box : `astropy.modeling.bounding_box.ModelBoundingBox`
            A bounding box defining the limits of the model.

        Examples
        --------
        >>> from photutils.psf import GaussianPRF
        >>> model = GaussianPRF(x_0=0, y_0=0, x_fwhm=2, y_fwhm=3)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.6712699, upper=4.6712699)
                y: Interval(lower=-7.0069049, upper=7.0069048)
            }
            model=GaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        a = factor * self.x_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        b = factor * self.y_fwhm * GAUSSIAN_FWHM_TO_SIGMA
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

        dpix = 0.5
        if isinstance(x_0, u.Quantity):
            dpix *= x_0.unit

        return (flux / 4.0
                * ((self._erf((x0 + dpix) / (np.sqrt(2) * x_sigma))
                    - self._erf((x0 - dpix) / (np.sqrt(2) * x_sigma)))
                   * (self._erf((y0 + dpix) / (np.sqrt(2) * y_sigma))
                      - self._erf((y0 - dpix) / (np.sqrt(2) * y_sigma)))))

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
    GaussianPRF, GaussianPSF, CircularGaussianPSF, MoffatPSF

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
        return _gaussian_amplitude(self.flux, self.sigma, self.sigma)

    @property
    def sigma(self):
        """
        Gaussian sigma (standard deviation).
        """
        return self.fwhm * GAUSSIAN_FWHM_TO_SIGMA

    def bounding_box(self, factor=5.5):
        """
        Return a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

        Returns
        -------
        bounding_box : `astropy.modeling.bounding_box.ModelBoundingBox`
            A bounding box defining the limits of the model.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPRF
        >>> model = CircularGaussianPRF(x_0=0, y_0=0, fwhm=2)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.6712699, upper=4.6712699)
                y: Interval(lower=-4.6712699, upper=4.6712699)
            }
            model=CircularGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        delta = factor * self.fwhm * GAUSSIAN_FWHM_TO_SIGMA
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

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

    This model is equivalent to `CircularGaussianPRF`, but it is
    parameterized in terms of the standard deviation (sigma) instead of
    the full width at half maximum (FWHM).

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

    The ``sigma`` parameter is fixed by default. If you wish to fit this
    parameter, set the ``fixed`` attribute to `False`, e.g.,::

        >>> from photutils.psf import IntegratedGaussianPRF
        >>> model = IntegratedGaussianPRF()
        >>> model.sigma.fixed = False

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

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return _gaussian_amplitude(self.flux, self.sigma, self.sigma)

    @property
    def fwhm(self):
        """
        Gaussian FWHM.
        """
        return self.sigma / GAUSSIAN_FWHM_TO_SIGMA

    def bounding_box(self, factor=5.5):
        """
        Return a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

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
        delta = factor * self.fwhm * GAUSSIAN_FWHM_TO_SIGMA
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

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
        dpix = 0.5
        if isinstance(x_0, u.Quantity):
            dpix *= x_0.unit

        return (flux / 4
                * ((self._erf((x - x_0 + dpix) / (np.sqrt(2) * sigma))
                    - self._erf((x - x_0 - dpix) / (np.sqrt(2) * sigma)))
                   * (self._erf((y - y_0 + dpix) / (np.sqrt(2) * sigma))
                      - self._erf((y - y_0 - dpix) / (np.sqrt(2) * sigma)))))

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
                'sigma': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}


class MoffatPSF(Fittable2DModel):
    r"""
    A 2D Moffat PSF model.

    This model is evaluated by sampling the 2D Moffat function at the
    input coordinates. The Moffat profile is normalized such that the
    analytical integral over the entire 2D plane is equal to the total
    flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x axis.

    y_0 : float, optional
        Position of the peak along the y axis.

    alpha : float, optional
        The characteristic radius of the Moffat profile.

    beta : float, optional
        The asymptotic power-law slope of the Moffat profile wings at
        large radial distances. Larger values provide less flux in the
        profile wings. For large ``beta``, this profile approaches a
        Gaussian profile. ``beta`` must be greater than 1. If ``beta``
        is set to 1, then the Moffat profile is a Lorentz function,
        whose integral is infinite. For this normalized model, if
        ``beta`` is set to 1, then the profile will be zero everywhere.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    GaussianPSF, CircularGaussianPSF, GaussianPRF, CircularGaussianPRF

    Notes
    -----
    The Moffat profile is defined as:

    .. math::

       f(x, y) = F \frac{\beta - 1}{\pi \alpha^2}
           \left(1 + \frac{\left(x - x_{0}\right)^{2}
               + \left(y - y_{0}\right)^{2}}{\alpha^{2}}\right)^{-\beta}

    where :math:`F` is the total integrated flux and :math:`(x_{0},
    y_{0})` is the position of the peak. Note that :math:`\beta` must be
    greater than 1.

    The FWHM of the Moffat profile is given by:

    .. math::

        \rm{FWHM} = 2 \alpha \sqrt{2^{1 / \beta} - 1}

    The model is normalized such that, for :math:`\beta > 1`:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) dx dy = F

    The ``alpha`` and ``beta`` parameters are fixed by default. If
    you wish to fit these parameters, set the ``fixed`` attribute to
    `False`, e.g.,::

        >>> from photutils.psf import MoffatPSF
        >>> model = MoffatPSF()
        >>> model.alpha.fixed = False
        >>> model.beta.fixed = False

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Moffat_distribution

    .. [2] https://ui.adsabs.harvard.edu/abs/1969A%26A.....3..455M/abstract

    .. [3] https://ned.ipac.caltech.edu/level5/Stetson/Stetson2_2_1.html

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import MoffatPSF
        model = MoffatPSF(flux=71.4, x_0=24.3, y_0=25.2, alpha=5.1, beta=3.2)
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
    alpha = Parameter(
        default=1, description='Characteristic radius of the Moffat profile',
        fixed=True)
    beta = Parameter(
        default=1, description='Power-law index of the Moffat profile',
        fixed=True)

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 alpha=alpha.default, beta=beta.default, **kwargs):

        super().__init__(flux=flux, x_0=x_0, y_0=y_0, alpha=alpha, beta=beta,
                         **kwargs)

    @property
    def fwhm(self):
        """
        The FWHM of the Moffat profile.
        """
        return 2.0 * self.alpha * np.sqrt(2 ** (1.0 / self.beta) - 1)

    def bounding_box(self, factor=3.0):
        """
        Return a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the FWHM used to define the limits.

        Returns
        -------
        bounding_box : `astropy.modeling.bounding_box.ModelBoundingBox`
            A bounding box defining the limits of the model.

        Examples
        --------
        >>> from photutils.psf import MoffatPSF
        >>> model = MoffatPSF(x_0=0, y_0=0, alpha=2, beta=3)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-11.21614, upper=11.21614)
                y: Interval(lower=-11.21614, upper=11.21614)
            }
            model=GaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        delta = factor * self.fwhm
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    def evaluate(self, x, y, flux, x_0, y_0, alpha, beta):
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

        alpha : float, optional
            The characteristic radius of the Moffat profile.

        beta : float, optional
            The asymptotic power-law slope of the Moffat profile wings
            at large radial distances. Larger values provide less flux
            in the profile wings. For large ``beta``, this profile
            approaches a Gaussian profile. ``beta`` must be greater
            than 1. If ``beta`` is set to 1, then the Moffat profile is
            a Lorentz function, whose integral is infinite. For this
            normalized model, if ``beta`` is set to 1, then the profile
            will be zero everywhere.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        # output units should match the input flux units
        alpha2 = alpha.copy()
        if isinstance(alpha, u.Quantity):
            alpha2 = alpha.value

        amp = flux * (beta - 1) / (np.pi * alpha2 ** 2)
        r2 = (x - x_0) ** 2 + (y - y_0) ** 2
        return amp * (1 + (r2 / alpha**2)) ** (-beta)

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
                'alpha': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}