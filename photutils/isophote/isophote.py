# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for storing the results of isophote fits.
"""

import warnings

import astropy.units as u
import numpy as np
from astropy.table import Table

from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.harmonics import (_first_and_second_harmonic_function,
                                          _fit_first_and_second_harmonics,
                                          _fit_upper_harmonic)
from photutils.isophote.model import _make_model_image
from photutils.utils._deprecation import (create_empty_deprecated_qtable,
                                          deprecated, deprecated_attribute,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          warn_deprecated_attribute)
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import validate_table_columns
from photutils.utils.exceptions import PhotutilsDeprecationWarning

__all__ = ['Isophote', 'IsophoteList']


# Remove in 4.0
_DEPRECATED_ATTRIBUTES = {
    'grad_error': 'gradient_err',
    'grad_r_error': 'gradient_rel_err',
    'niter': 'n_iter',
    'ndata': 'n_data',
    'nflag': 'n_flagged',
    'intens': 'intensity',
    'int_err': 'intensity_err',
    'eps': 'ellipticity',
    'ellip_err': 'ellipticity_err',
    'grad': 'gradient',
    'pix_stddev': 'pixel_stddev',
    'sarea': 'sector_area',
    'n_flag': 'n_flagged',
    'tflux_e': 'total_flux_ellipse',
    'tflux_c': 'total_flux_circle',
    'npix_e': 'n_pixels_ellipse',
    'npix_c': 'n_pixels_circle',
    'pa': 'orientation',
    'pa_err': 'orientation_err',
}

# The version in which each attribute name was deprecated
_DEPRECATED_SINCE = {
    **dict.fromkeys(('grad_error', 'grad_r_error', 'niter', 'ndata',
                     'nflag'), '3.0'),
    **dict.fromkeys(('tflux_e', 'tflux_c', 'npix_e', 'npix_c', 'intens',
                     'int_err', 'eps', 'ellip_err', 'grad', 'pix_stddev',
                     'sarea', 'n_flag', 'pa', 'pa_err'), '3.2'),
}

# Remove in 4.0
# The deprecated table column names and the version in which each was
# deprecated. The 'grad_rerror' and 'intens_err' names were table
# column names, not attributes.
_DEPRECATED_COLUMNS = {**_DEPRECATED_ATTRIBUTES,
                       'grad_rerror': 'gradient_rel_err',
                       'intens_err': 'intensity_err'}
_DEPRECATED_COLUMNS_SINCE = {**_DEPRECATED_SINCE, 'grad_rerror': '3.0',
                             'intens_err': '3.2'}

# Remove in 4.0
# The deprecated angle names are not aliases of the new names. They
# return floats in radians, from these private attributes, instead of
# a Quantity in degrees.
_DEPRECATED_RADIANS = {'pa': '_orientation_rad',
                       'pa_err': '_orientation_err_rad'}


def _nan_if_none(value):
    """
    Return NaN in place of `None`, otherwise return the value.
    """
    return np.nan if value is None else value


def _to_degrees(angle):
    """
    Convert an angle (float or array) in radians to a
    `~astropy.units.Quantity` in degrees.
    """
    return np.rad2deg(angle) << u.deg


class Isophote:
    """
    Container class to store the results of a single isophote fit.

    The extracted data sample at the given isophote (sampled intensities
    along the elliptical path on the image) is also kept as an attribute
    of this class. The container concept helps in segregating
    information directly related to the sample from information that
    more closely relates to the fitting process, such as status codes,
    errors for isophote parameters, and the like.

    Parameters
    ----------
    sample : `~photutils.isophote.EllipseSample` instance
        The sample information.
    n_iter : int
        The number of iterations used to fit the isophote.
    valid : bool
        The status of the fitting operation.
    stop_code : int
        The fitting stop code:

        * ``0``: Normal.

        * ``1``: Fewer than the prespecified fraction of the extracted
          data points are valid. When
          :meth:`~photutils.isophote.Ellipse.fit_image` grows outward,
          such an isophote is extracted again with the geometry of the
          previous isophote. If it is the first isophote, an empty list
          is returned instead.

        * ``2``: Exceeded maximum number of iterations.

        * ``3``: The harmonic fit failed (e.g., a singular matrix) or
          there are too few valid data points to fit. Such an isophote
          is always invalid (``valid=False``) and is not included in
          the results of :meth:`~photutils.isophote.Ellipse.fit_image`.
          It is also returned by
          :meth:`~photutils.isophote.Ellipse.measure_isophotes` for an
          ellipse with no valid data points.

        * ``4``: The isophote was extracted with a fixed geometry,
          without fitting. This happens when the semimajor axis length
          is larger than ``max_iterative_sma``, with ``noniterate=True``
          in :meth:`~photutils.isophote.Ellipse.fit_isophote`,
          when :meth:`~photutils.isophote.Ellipse.fit_image`
          switches to noniterative mode, when all the ellipse
          parameters are fixed, and for isophotes measured with
          :meth:`~photutils.isophote.Ellipse.measure_isophotes`.

        * ``5``: The fit stopped with a code of -1 (see below) inside
          :meth:`~photutils.isophote.Ellipse.fit_image`. The isophote is
          extracted with the geometry of a previously fitted isophote.
          After two consecutive isophotes with this code, the outward
          growth stops or continues in noniterative mode.

        * ``-1``: No meaningful gradient could be measured, or the
          corrected ellipse diverged (its ellipticity exceeded 0.95
          or its center moved outside of the image). The isophote
          has the best geometry found before that. This code is
          returned by :meth:`~photutils.isophote.Ellipse.fit_isophote`.
          :meth:`~photutils.isophote.Ellipse.fit_image` replaces such an
          isophote with one that has the geometry of a previously fitted
          isophote and a stop code of 5. If it is the first isophote, an
          empty list is returned instead.

    Attributes
    ----------
    geometry : `~photutils.isophote.EllipseGeometry`
        The geometry of the isophote. It is the same object as the
        geometry of the ``sample``. The central pixel has ``sma``,
        ``ellipticity``, and ``orientation`` of zero.
    intensity : float
        The mean intensity along the elliptical path.
    intens : float
        .. deprecated:: 3.2
            Use ``intensity`` instead.
    rms : float
        The root-mean-square scatter (standard deviation) of the
        intensity values about their mean along the elliptical path.
    intensity_err : float
        The error of the mean (rms / sqrt(# data points)).
    int_err : float
        .. deprecated:: 3.2
            Use ``intensity_err`` instead.
    ellipticity : float
        The ellipticity of the ellipse.
    eps : float
        .. deprecated:: 3.2
            Use ``ellipticity`` instead.
    ellipticity_err : float
        The ellipticity error.
    ellip_err : float
        .. deprecated:: 3.2
            Use ``ellipticity_err`` instead.
    orientation : `~astropy.units.Quantity`
        The orientation of the semimajor axis in degrees, measured
        counterclockwise from the positive x axis.
    pa : float
        .. deprecated:: 3.2
            Use ``orientation`` instead. This attribute is the
            orientation in radians.
    orientation_err : `~astropy.units.Quantity`
        The orientation error in degrees.
    pa_err : float
        .. deprecated:: 3.2
            Use ``orientation_err`` instead. This attribute is the
            orientation error in radians.
    x0_err : float
        The error associated with the center x coordinate.
    y0_err : float
        The error associated with the center y coordinate.
    pixel_stddev : float
        The estimate of pixel standard deviation (rms * sqrt(average
        sector integration area)).
    pix_stddev : float
        .. deprecated:: 3.2
            Use ``pixel_stddev`` instead.
    gradient : float
        The local radial intensity gradient.
    grad : float
        .. deprecated:: 3.2
            Use ``gradient`` instead.
    grad_error : float
        .. deprecated:: 3.0
            Use ``gradient_err`` instead.
    gradient_err : float
        The measurement error of the local radial intensity gradient.
        It is NaN if it could not be computed.
    grad_r_error : float
        .. deprecated:: 3.0
            Use ``gradient_rel_err`` instead.
    gradient_rel_err : float
        The relative error of the local radial intensity gradient. It
        is NaN if it could not be computed.
    tflux_e : float
        .. deprecated:: 3.2
            Use ``total_flux_ellipse`` instead.
    total_flux_ellipse : float
        The sum of the valid (unmasked and finite) pixel values inside
        the ellipse.
    npix_e : int
        .. deprecated:: 3.2
            Use ``n_pixels_ellipse`` instead.
    n_pixels_ellipse : int or float
        The total number of valid pixels inside the ellipse. It is NaN
        for the central pixel.
    tflux_c : float
        .. deprecated:: 3.2
            Use ``total_flux_circle`` instead.
    total_flux_circle : float
        The sum of the valid (unmasked and finite) pixel values inside
        a circle with the same ``sma`` as the ellipse.
    npix_c : int
        .. deprecated:: 3.2
            Use ``n_pixels_circle`` instead.
    n_pixels_circle : int or float
        The total number of valid pixels inside a circle with the same
        ``sma`` as the ellipse. It is NaN for the central pixel.
    sector_area : float
        The average sector area on the isophote (pixel**2).
    sarea : float
        .. deprecated:: 3.2
            Use ``sector_area`` instead.
    niter : int
        .. deprecated:: 3.0
            Use ``n_iter`` instead.
    ndata : int
        .. deprecated:: 3.0
            Use ``n_data`` instead.
    n_data : int
        The number of extracted data points.
    nflag : int
        .. deprecated:: 3.0
            Use ``n_flagged`` instead.
    n_flag : int
        .. deprecated:: 3.2
            Use ``n_flagged`` instead.
    n_flagged : int
        The number of discarded data points. Data points can be
        discarded either because they are physically outside the image
        frame boundaries, because they were rejected by sigma clipping,
        or because they are masked or nonfinite.
    a3, b3, a4, b4 : float
        The higher-order harmonics that measure the deviations from a
        perfect ellipse. These values are actually the raw harmonic
        amplitudes divided by the absolute value of the local radial
        gradient and the semimajor axis length, so they can directly be
        compared with each other. The ``b4`` parameter is positive for
        galaxies with disky (kite-like) isophotes and negative for
        galaxies with boxy isophotes. They are NaN if they could not
        be computed.
    a3_err, b3_err, a4_err, b4_err : float
        The errors associated with the ``a3``, ``b3``, ``a4``, and
        ``b4`` attributes. They are NaN if they could not be
        computed.
    """

    def __init__(self, sample, n_iter, valid, stop_code):
        self.sample = sample
        self.n_iter = n_iter
        self.valid = valid
        self.stop_code = stop_code
        self.geometry = sample.geometry

        # An isophote with a negative or NaN semimajor axis length has
        # no measured values.
        sma = sample.geometry.sma
        if sma == 0:
            self._set_central_values(sample)
        elif sma > 0 and sample.n_valid == 0:
            self._set_empty_values(sample)
        elif sma > 0:
            self._set_measured_values(sample)

    def _set_measured_values(self, sample):
        self.intensity = sample.intensity
        self.rms = np.std(sample.intensities)
        self.intensity_err = self.rms / np.sqrt(sample.n_valid)
        self.pixel_stddev = self.rms * np.sqrt(sample.sector_area)
        self.gradient = sample.gradient
        self.gradient_err = _nan_if_none(sample.gradient_err)

        self.gradient_rel_err = _nan_if_none(sample.gradient_rel_err)
        self.sector_area = sample.sector_area
        self.n_data = sample.n_valid
        self.n_flagged = sample.n_total - sample.n_valid

        # Flux contained inside ellipse and circle
        (self.total_flux_ellipse, self.total_flux_circle,
         self.n_pixels_ellipse,
         self.n_pixels_circle) = self._compute_fluxes()

        self._compute_errors()

        # Deviations from a perfect ellipse
        (self.a3, self.b3, self.a3_err,
         self.b3_err) = self._compute_deviations(sample, 3)
        (self.a4, self.b4, self.a4_err,
         self.b4_err) = self._compute_deviations(sample, 4)

    def _set_empty_values(self, sample):
        # No valid sampled points (e.g., a trial geometry that fell
        # entirely outside the image). Operating on the empty sample
        # arrays would emit numpy warnings (e.g., "Mean of empty
        # slice"), so set nonmeaningful values directly and skip the
        # harmonic computations.
        self.intensity = sample.intensity
        self.rms = np.nan
        self.intensity_err = np.nan
        self.pixel_stddev = np.nan
        self.gradient = sample.gradient
        self.gradient_err = _nan_if_none(sample.gradient_err)
        self.gradient_rel_err = _nan_if_none(sample.gradient_rel_err)
        self.sector_area = sample.sector_area
        self.n_data = sample.n_valid
        self.n_flagged = sample.n_total - sample.n_valid
        self.total_flux_ellipse = self.total_flux_circle = 0.0
        self.n_pixels_ellipse = self.n_pixels_circle = 0
        self.x0_err = self.y0_err = self._orientation_err_rad = 0.0
        self.ellipticity_err = 0.0
        self.a3 = self.b3 = self.a3_err = self.b3_err = np.nan
        self.a4 = self.b4 = self.a4_err = self.b4_err = np.nan

    @classmethod
    def _from_values(cls, geometry, values):
        """
        Create an isophote without a sample from its result values.

        Parameters
        ----------
        geometry : `~photutils.isophote.EllipseGeometry`
            The geometry of the isophote. It supplies ``sma``,
            ``x0``, ``y0``, ``ellipticity``, and ``orientation``.

        values : dict
            The scalar values of the other isophote attributes, keyed
            by attribute name. The orientation error is stored as a
            float in radians under the ``'_orientation_err_rad'`` key.

        Returns
        -------
        result : `~photutils.isophote.Isophote`
            The isophote, whose ``sample`` is `None`.
        """
        # __init__ computes the values from a sample, so it is skipped.
        isophote = cls.__new__(cls)
        isophote.sample = None
        isophote.geometry = geometry
        for name, value in values.items():
            setattr(isophote, name, value)
        return isophote

    def _set_central_values(self, sample):
        # The central pixel holds a single intensity value. Some values
        # are set to zero to ease certain tasks such as model building
        # and plotting magnitude errors.
        self.intensity = sample.intensity
        self.rms = np.nan
        self.intensity_err = 0.0
        self.pixel_stddev = np.nan
        self.gradient = 0.0
        self.gradient_err = np.nan
        self.gradient_rel_err = np.nan
        self.sector_area = np.nan
        self.n_data = sample.n_valid
        self.n_flagged = sample.n_total - sample.n_valid

        self.total_flux_ellipse = self.total_flux_circle = np.nan
        self.n_pixels_ellipse = self.n_pixels_circle = np.nan

        self.a3 = self.b3 = 0.0
        self.a4 = self.b4 = 0.0
        self.a3_err = self.b3_err = 0.0
        self.a4_err = self.b4_err = 0.0

        self.ellipticity_err = 0.0
        self._orientation_err_rad = 0.0
        self.x0_err = 0.0
        self.y0_err = 0.0

    # Remove in 4.0
    @property
    def eps(self):
        """
        The ellipticity of the ellipse.

        .. deprecated:: 3.2
            Use ``ellipticity`` instead.
        """
        warn_deprecated_attribute('eps', 'ellipticity', since='3.2',
                                  until='4.0')
        return self.ellipticity

    # Remove in 4.0
    @property
    def pa(self):
        """
        The orientation of the semimajor axis as a float in radians.

        .. deprecated:: 3.2
            Use ``orientation`` instead.
        """
        warn_deprecated_attribute('pa', 'orientation', since='3.2',
                                  until='4.0')
        return self._orientation_rad

    # Remove in 4.0
    @property
    def pa_err(self):
        """
        The orientation error as a float in radians.

        .. deprecated:: 3.2
            Use ``orientation_err`` instead.
        """
        warn_deprecated_attribute('pa_err', 'orientation_err', since='3.2',
                                  until='4.0')
        return self._orientation_err_rad

    @pa_err.setter
    def pa_err(self, value):  # numpydoc ignore=GL08
        warn_deprecated_attribute('pa_err', 'orientation_err', since='3.2',
                                  until='4.0')
        self._orientation_err_rad = value

    # The comparison methods are deprecated. Compare the ``sma`` values
    # instead. The isophotes are compared by their semimajor axis
    # length. Objects without a semimajor axis length are not
    # comparable.
    @staticmethod
    def _warn_comparison():
        msg = ('Comparing isophotes was deprecated in version 3.2 and '
               'will be removed in version 4.0. Compare the isophote.sma '
               'values instead.')
        warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=3)

    def __lt__(self, other):
        self._warn_comparison()
        try:
            return self.sma < other.sma
        except AttributeError:
            return NotImplemented

    def __gt__(self, other):
        self._warn_comparison()
        try:
            return self.sma > other.sma
        except AttributeError:
            return NotImplemented

    def __le__(self, other):
        self._warn_comparison()
        try:
            return self.sma <= other.sma
        except AttributeError:
            return NotImplemented

    def __ge__(self, other):
        self._warn_comparison()
        try:
            return self.sma >= other.sma
        except AttributeError:
            return NotImplemented

    def __eq__(self, other):
        # Equality with an object that has no semimajor axis length is
        # always False, so it is not deprecated.
        if hasattr(other, 'sma'):
            self._warn_comparison()
        try:
            return self.sma == other.sma
        except AttributeError:
            return NotImplemented

    def __str__(self):
        return str(self.to_table())

    @property
    def sma(self):
        """
        The semimajor axis length (pixels).
        """
        return self.geometry.sma

    @property
    def ellipticity(self):
        """
        The ellipticity of the ellipse.
        """
        return self.geometry.ellipticity

    @property
    def orientation(self):
        """
        The orientation of the semimajor axis as an angular
        `~astropy.units.Quantity` in degrees, measured counterclockwise
        from the positive x axis.
        """
        return _to_degrees(self.geometry._orientation_rad)

    @property
    def _orientation_rad(self):
        return self.geometry._orientation_rad

    @property
    def orientation_err(self):
        """
        The orientation error as an angular `~astropy.units.Quantity`
        in degrees.
        """
        return _to_degrees(self._orientation_err_rad)

    @property
    def x0(self):
        """
        The center x coordinate (pixel).
        """
        return self.geometry.x0

    @property
    def y0(self):
        """
        The center y coordinate (pixel).
        """
        return self.geometry.y0

    def _compute_fluxes(self):
        """
        Compute integrated flux inside ellipse, as well as inside a
        circle defined with the same semimajor axis.

        Pixels in a square section enclosing the circle are scanned. The
        distance of each pixel to the isophote center is compared both
        with the semimajor axis length and with the length of the
        ellipse radius vector, and integrals are updated if the pixel
        distance is not larger. Masked and nonfinite pixels are
        excluded.
        """
        # Compute limits of square array that encloses circle.
        sma = self.sample.geometry.sma
        x0 = self.sample.geometry.x0
        y0 = self.sample.geometry.y0
        xsize = self.sample.image.shape[1]
        ysize = self.sample.image.shape[0]

        imin = max(0, int(x0 - sma - 0.5) - 1)
        jmin = max(0, int(y0 - sma - 0.5) - 1)
        imax = min(xsize, int(x0 + sma + 0.5) + 1)
        jmax = min(ysize, int(y0 + sma + 0.5) + 1)

        # Integrate.
        if (jmax - jmin > 1) and (imax - imin) > 1:
            y, x = np.mgrid[jmin:jmax, imin:imax]
            radius, angle = self.sample.geometry.to_polar(x, y)
            radius_e = self.sample.geometry.radius(angle)

            midx = (radius <= sma)
            # Exclude masked and nonfinite pixels.
            values = np.ma.masked_invalid(self.sample.image[y[midx], x[midx]])
            tflux_c = np.ma.sum(values)
            npix_c = np.ma.count(values)

            midx2 = (radius <= radius_e)
            values = np.ma.masked_invalid(
                self.sample.image[y[midx2], x[midx2]])
            tflux_e = np.ma.sum(values)
            npix_e = np.ma.count(values)
        else:
            tflux_e = 0.0
            tflux_c = 0.0
            npix_e = 0
            npix_c = 0

        return tflux_e, tflux_c, npix_e, npix_c

    def _compute_deviations(self, sample, n):
        """
        Compute deviations from a perfect ellipse, based on the
        amplitudes and errors for harmonic "n".

        The upper harmonic is fit to the sampled intensities together
        with a constant term.
        """
        # The residual variance needs more data points than fitted
        # coefficients.
        if sample.n_valid <= 3:
            return np.nan, np.nan, np.nan, np.nan

        try:
            # Upper (third and fourth) harmonics
            up_coeffs, up_inv_hessian = _fit_upper_harmonic(sample.angles,
                                                            sample.intensities,
                                                            n)

            scale = self.sma * abs(sample.gradient)
            a = up_coeffs[1] / scale
            b = up_coeffs[2] / scale

            phi = sample.angles
            up_model = (up_coeffs[0] + up_coeffs[1] * np.sin(n * phi)
                        + up_coeffs[2] * np.cos(n * phi))
            up_var_residual = np.std(up_model - sample.intensities,
                                     ddof=len(up_coeffs))**2
            up_covariance = up_inv_hessian * up_var_residual

            ce = np.sqrt(np.diag(up_covariance))

            # This comes from the old code. Likely it was based on
            # empirical experience with the STSDAS task, so we leave
            # it here without too much thought.
            # The sample value is tested because a gradient error that
            # could not be measured is `None` there, while a measured
            # relative error can also be NaN. Both are NaN on the
            # isophote.
            gre = (sample.gradient_rel_err
                   if sample.gradient_rel_err is not None else 0.8)

            # The relative amplitude error is combined in quadrature with
            # the relative gradient error. The errors are written in a
            # form that does not divide by the amplitudes, which can be
            # zero.
            a_err = np.sqrt((ce[1] / scale)**2 + (a * gre)**2)
            b_err = np.sqrt((ce[2] / scale)**2 + (b * gre)**2)

        except Exception:  # we want to catch everything
            a = b = a_err = b_err = np.nan

        return a, b, a_err, b_err

    def _compute_errors(self):
        """
        Compute parameter errors based on the diagonal of the covariance
        matrix of the four harmonic coefficients for harmonics n=1 and
        n=2.
        """
        # The residual variance needs more data points than fitted
        # coefficients.
        if self.sample.n_valid <= 5:
            self.x0_err = self.y0_err = self._orientation_err_rad = 0.0
            self.ellipticity_err = 0.0
            return

        try:
            coeffs, covariance = _fit_first_and_second_harmonics(
                self.sample.angles, self.sample.intensities)
            model = _first_and_second_harmonic_function(self.sample.angles,
                                                        coeffs)
            var_residual = np.std(self.sample.intensities - model,
                                  ddof=len(coeffs)) ** 2
            errors = np.sqrt(np.diagonal(covariance * var_residual))

            eps = self.sample.geometry.ellipticity
            orientation_rad = self.sample.geometry._orientation_rad

            # Parameter errors result from direct projection of
            # coefficient errors. These showed to be the error estimators
            # that best convey the errors measured in Monte Carlo
            # experiments (see Busko 1996, ASPC 101, 139).
            ea = abs(errors[2] / self.gradient)
            eb = abs(errors[1] * (1.0 - eps) / self.gradient)
            self.x0_err = np.sqrt((ea * np.cos(orientation_rad))**2
                                  + (eb * np.sin(orientation_rad))**2)
            self.y0_err = np.sqrt((ea * np.sin(orientation_rad))**2
                                  + (eb * np.cos(orientation_rad))**2)
            self.ellipticity_err = (abs(2.0 * errors[4] * (1.0 - eps)
                                        / self.sma / self.gradient))
            if abs(eps) > np.finfo(float).resolution:
                self._orientation_err_rad = (
                    abs(2.0 * errors[3] * (1.0 - eps) / self.sma
                        / self.gradient / (1.0 - (1.0 - eps)**2)))
            else:
                self._orientation_err_rad = 0.0
        except Exception:  # we want to catch everything
            self.x0_err = self.y0_err = self._orientation_err_rad = 0.0
            self.ellipticity_err = 0.0

    def _fix_geometry(self, isophote):
        """
        Fix the geometry of this isophote to be identical to the input
        isophote. See the deprecated ``fix_geometry`` method.
        """
        self.geometry.ellipticity = isophote.geometry.ellipticity
        self.geometry._orientation_rad = isophote.geometry._orientation_rad
        self.geometry.x0 = isophote.geometry.x0
        self.geometry.y0 = isophote.geometry.y0

        # The initial polar radius and sector width depend on the
        # ellipticity, so they must be recomputed for the new geometry.
        self.geometry._reset_sma_dependent_attributes()

    @deprecated('3.2', until='4.0')
    def fix_geometry(self, isophote):
        """
        Fix the geometry of a problematic isophote to be identical to
        the input isophote.

        This method should be called when the fitting goes berserk and
        delivers an isophote with bad geometry, such as ellipticity > 1
        or another meaningless situation. This is not a problem in
        itself when fitting any given isophote, but will create an error
        when the affected isophote is used as starting guess for the
        next fit.

        The semimajor axis length is not changed, and the sampled data
        and derived attributes (e.g., ``intensity``) are not recomputed.

        Parameters
        ----------
        isophote : `~photutils.isophote.Isophote` instance
            The isophote from which to take the geometry information.
        """
        self._fix_geometry(isophote)

    def sampled_coordinates(self):
        """
        Return the (x, y) coordinates where the image was sampled in
        order to get the intensities associated with this isophote.

        Returns
        -------
        x, y : 1D `~numpy.ndarray`
            The x and y coordinates as 1D arrays.

        Raises
        ------
        ValueError
            If the isophote has no sample, which is the case for
            isophotes created by
            `~photutils.isophote.IsophoteList.from_table`.
        """
        if self.sample is None:
            msg = ('The isophote has no sample. Use '
                   'Ellipse.measure_isophotes to sample an image along '
                   'it.')
            raise ValueError(msg)
        return self.sample.coordinates()

    def to_table(self):
        """
        Return the default isophote parameters as an astropy
        `~astropy.table.QTable`.

        Returns
        -------
        result : `~astropy.table.QTable`
            A one-row table with the default isophote parameters (see
            ``IsophoteList.default_columns``). The
            ``orientation`` and ``orientation_err`` columns are
            `~astropy.units.Quantity` columns in degrees.
        """
        return _isophote_list_to_table([self])


# Remove in 4.0
# The other deprecated names were plain instance attributes, so they are
# properties that read and write the new attributes.
for _name, _new_name in _DEPRECATED_ATTRIBUTES.items():
    if _name not in ('eps', 'pa', 'pa_err'):
        setattr(Isophote, _name,
                deprecated_attribute(_name, _new_name,
                                     _DEPRECATED_SINCE[_name], until='4.0'))

del _name, _new_name


class IsophoteList:
    """
    Container class that provides the same attributes as the
    `~photutils.isophote.Isophote` class, but for a list of isophotes.

    The attributes of this class are arrays representing the values of
    the attributes for the entire list of `~photutils.isophote.Isophote`
    instances. See the `~photutils.isophote.Isophote` class for a
    description of the attributes.

    The class provides basic list behavior, such as indexing, slicing,
    appending, and the ``+`` and ``+=`` operators.

    Parameters
    ----------
    iso_list : list of `~photutils.isophote.Isophote`
        A list of `~photutils.isophote.Isophote` instances.

    Attributes
    ----------
    default_columns : list of str
        The names of the properties included in the output of
        `to_table` when ``columns`` is `None`. The list can be changed.
        It is copied to the lists created by slicing or adding.
    """

    def __init__(self, iso_list):
        self._list = iso_list
        self.default_columns = list(_DEFAULT_COLUMNS)

    # Remove in 4.0
    def __getattr__(self, name):
        if name in _DEPRECATED_RADIANS:
            warn_deprecated_attribute(name, _DEPRECATED_ATTRIBUTES[name],
                                      since=_DEPRECATED_SINCE[name],
                                      until='4.0')
            return self._collect_as_array(_DEPRECATED_RADIANS[name])
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since=_DEPRECATED_SINCE, until='4.0')

    def __len__(self):
        return len(self._list)

    def __delitem__(self, index):
        self._list.__delitem__(index)

    def __setitem__(self, index, value):
        self._list.__setitem__(index, value)

    def __getitem__(self, index):
        if isinstance(index, slice):
            result = IsophoteList(self._list[index])
            result.default_columns = _as_column_list(self.default_columns)
            return result
        return self._list.__getitem__(index)

    def __iter__(self):
        return self._list.__iter__()

    def sort(self):
        """
        Sort the list of isophotes by semimajor axis length.
        """
        self._list.sort(key=lambda iso: iso.sma)

    def insert(self, index, value):
        """
        Insert an isophote at a given index.

        Parameters
        ----------
        index : int
            The index where to insert the isophote.
        value : `~photutils.isophote.Isophote`
            The isophote to be inserted.
        """
        self._list.insert(index, value)

    def append(self, value):
        """
        Append an isophote to the list.

        Parameters
        ----------
        value : `~photutils.isophote.Isophote`
            The isophote to be appended.
        """
        self._list.append(value)

    def extend(self, value):
        """
        Extend the list with the isophotes from another
        `~photutils.isophote.IsophoteList` instance.

        Parameters
        ----------
        value : `~photutils.isophote.IsophoteList`
            The isophotes to be appended.
        """
        self._list.extend(value._list)

    def __iadd__(self, value):
        self.extend(value)
        return self

    def __add__(self, value):
        temp = self._list[:]  # shallow copy
        temp.extend(value._list)
        result = IsophoteList(temp)
        result.default_columns = _as_column_list(self.default_columns)
        return result

    def get_closest(self, sma):
        """
        Return the `~photutils.isophote.Isophote` instance that has the
        closest semimajor axis length to the input semimajor axis.

        Parameters
        ----------
        sma : float
            The semimajor axis length.

        Returns
        -------
        isophote : `~photutils.isophote.Isophote` instance
            The isophote with the closest semimajor axis value.
        """
        index = (np.abs(self.sma - sma)).argmin()
        return self._list[index]

    def _collect_as_array(self, attr_name, *, dtype=float):
        return np.array(self._collect_as_list(attr_name), dtype=dtype)

    def _collect_as_list(self, attr_name):
        return [getattr(iso, attr_name) for iso in self._list]

    @property
    def sample(self):
        """
        The isophote `~photutils.isophote.EllipseSample` information.
        """
        return self._collect_as_list('sample')

    @property
    def properties(self):
        """
        A list of the names of the properties of an
        `~photutils.isophote.IsophoteList` instance.

        The ``sample`` property is excluded.
        """
        return list(_PROPERTY_NAMES)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def to_table(self, columns=None):
        """
        Convert an `~photutils.isophote.IsophoteList` instance to a
        `~astropy.table.QTable`.

        Parameters
        ----------
        columns : list of str, `None`, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`. The allowed column names are the
            names in the ``properties`` attribute. If ``columns`` is
            `None`, then a default list of properties (as defined by
            the ``default_columns`` attribute) will be used. A
            deprecated name gives a column with its new name and emits
            a deprecation warning. The deprecated ``'main'`` and
            ``'all'`` values are equivalent to `None` and to the
            `properties` attribute, respectively.

        Returns
        -------
        result : `~astropy.table.QTable`
            A table with one row per isophote. The ``orientation`` and
            ``orientation_err`` columns are `~astropy.units.Quantity`
            columns in degrees.

        Raises
        ------
        ValueError
            If any name in ``columns`` is not a valid column name, or if
            ``'main'`` or ``'all'`` is input together with other names.
        """
        return _isophote_list_to_table(self, columns=columns)

    @classmethod
    def from_table(cls, table, *, step=0.1, linear_growth=False):
        """
        Create an `~photutils.isophote.IsophoteList` from a table.

        This is the inverse of `to_table`. The table has one row per
        isophote and the isophotes keep the order of the rows.

        The isophotes have no sample, so ``sample`` is `None` and
        ``sampled_coordinates`` raises an error. Use
        `Ellipse.measure_isophotes
        <photutils.isophote.Ellipse.measure_isophotes>` to sample an
        image along them.

        Parameters
        ----------
        table : `~astropy.table.Table` or `~astropy.table.QTable`
            The table. The ``sma``, ``x0``, ``y0``, ``ellipticity``,
            and ``orientation`` columns are required. The other
            columns are the names in the ``properties`` attribute. A
            missing float column is NaN, a missing ``n_data``,
            ``n_flagged``, ``n_iter``, or ``stop_code`` column is 0,
            and a missing ``valid`` column is `True`. Columns that are
            not isophote properties are ignored. Deprecated column
            names are accepted and emit a deprecation warning.

            The ``orientation`` and ``orientation_err`` columns must
            have an angular unit. Plain text formats such as CSV do
            not keep units, so set the unit after reading the table
            (e.g., ``table['orientation'].unit = 'deg'``). Any other
            column that has a unit, either a `~astropy.units.Quantity`
            column or a column with its ``unit`` attribute set, must
            be dimensionless, or in pixels for the ``sma``, ``x0``,
            ``x0_err``, ``y0``, and ``y0_err`` columns.

            The ``sma`` values must be finite and nonnegative, the
            ``x0``, ``y0``, and ``orientation`` values must be finite,
            and the ``ellipticity`` values must be at least 0 and less
            than 1. The ``orientation_err`` values may be NaN but not
            infinite.

        step : float, optional
            The positive step value of the geometry of each isophote,
            which sets the sampling annulus used by
            `Ellipse.measure_isophotes
            <photutils.isophote.Ellipse.measure_isophotes>`. See
            `~photutils.isophote.EllipseGeometry`.

        linear_growth : bool, optional
            Whether ``step`` is a linear increment of the semimajor
            axis length instead of a relative one. See
            `~photutils.isophote.EllipseGeometry`.

        Returns
        -------
        result : `~photutils.isophote.IsophoteList`
            The isophotes.

        Raises
        ------
        TypeError
            If ``table`` is not an astropy `~astropy.table.Table` or
            `~astropy.table.QTable`.

        ValueError
            If a required column is missing, if a column has a unit
            that is not allowed, if a geometry value is out of range
            (the error names the column and the first invalid row), if
            ``step`` is not a positive finite number, or if a value
            cannot be represented with the type of its property (e.g.,
            NaN in the ``n_data`` column).
        """
        return _isophote_list_from_table(cls, table, step=step,
                                         linear_growth=linear_growth)

    def make_model_image(self, shape, *, fill_value=0.0,
                         high_harmonics=False, sma_interval=0.1):
        """
        Make a model elliptical galaxy image from the isophotes.

        The isophote list must contain at least 4 isophotes, sorted
        by strictly increasing semimajor axis length. Isophotes with a
        nonfinite intensity (e.g., the central pixel if it is masked)
        are ignored.

        The intensity, ellipticity, orientation, and center of the
        isophotes are interpolated with least-squares cubic splines
        onto a grid of semimajor axis lengths spaced by
        ``sma_interval``. Each interpolated ellipse is traced on the
        output image and its intensity is distributed over the
        overlapping pixels with bilinear weights. The pixel values are
        then normalized by the accumulated weights.

        Parameters
        ----------
        shape : 2-tuple
            The (ny, nx) shape of the output model image, usually the
            shape of the array used to generate the isophotes.

        fill_value : float, optional
            The constant value to fill empty pixels. If an output pixel
            has no contribution from any isophote, it will be assigned
            this value. The default is 0.

        high_harmonics : bool, optional
            Whether to add the higher-order harmonics (i.e., ``a3``,
            ``b3``, ``a4``, and ``b4``) to the result. See
            `~photutils.isophote.Isophote` for details. The harmonic
            amplitudes are scaled by the ``gradient`` of the isophotes,
            which must be finite. Harmonics that could not be computed
            (NaN) are treated as zero. The default is `False`.

        sma_interval : float, optional
            The step in semimajor axis length of the grid on which the
            isophote parameters are interpolated and the model ellipses
            are drawn. It must be positive. Larger values are faster but
            can leave gaps (filled with ``fill_value``) in the model.
            The default is 0.1.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The image with the model galaxy.

        Raises
        ------
        ValueError
            If the list contains fewer than 4 isophotes with a finite
            intensity, if its isophotes are not sorted by strictly
            increasing semimajor axis length, or if ``high_harmonics``
            is `True` and the ``gradient`` of an isophote with a finite
            intensity is not finite.
        """
        return _make_model_image(self, shape, fill_value=fill_value,
                                 high_harmonics=high_harmonics,
                                 sma_interval=sma_interval)

    @deprecated('3.2', alternative='the ``properties`` attribute',
                until='4.0')
    def get_names(self):
        """
        Get the names of the properties of an
        `~photutils.isophote.IsophoteList` instance.

        The ``sample`` property is excluded.

        Returns
        -------
        list_names : list
            A list of the names of the properties.
        """
        return self.properties


# The properties of IsophoteList, as (name, dtype, unit, description)
# entries in the order of IsophoteList.properties.
_ISOPHOTE_PROPERTIES = (
    ('sma', float, None,
     'The semimajor axis length (pixels).'),
    ('intensity', float, None,
     'The mean intensity value along the elliptical path.'),
    ('intensity_err', float, None,
     ('The error of the mean intensity (rms / sqrt(# data '
      'points)).')),
    ('ellipticity', float, None,
     'The ellipticity of the ellipse.'),
    ('ellipticity_err', float, None,
     'The ellipticity error.'),
    ('orientation', float, u.deg,
     ('The orientation (degrees) of the semimajor axis, measured '
      'counterclockwise from the positive x axis.')),
    ('orientation_err', float, u.deg,
     'The orientation error (degrees).'),
    ('x0', float, None,
     'The center x coordinate (pixel).'),
    ('x0_err', float, None,
     'The error associated with the center x coordinate.'),
    ('y0', float, None,
     'The center y coordinate (pixel).'),
    ('y0_err', float, None,
     'The error associated with the center y coordinate.'),
    ('rms', float, None,
     ('The root-mean-square scatter (standard deviation) of the '
      'intensity values about their mean along the elliptical '
      'path.')),
    ('pixel_stddev', float, None,
     ('The estimate of pixel standard deviation (rms * '
      'sqrt(average sector integration area)).')),
    ('gradient', float, None,
     'The local radial intensity gradient.'),
    ('gradient_err', float, None,
     ('The measurement error of the local radial intensity '
      'gradient.')),
    ('gradient_rel_err', float, None,
     'The relative error of the local radial intensity gradient.'),
    ('sector_area', float, None,
     'The average sector area on the isophote (pixel**2).'),
    ('n_data', int, None,
     'The number of extracted data points.'),
    ('n_flagged', int, None,
     ('The number of discarded data points.\n\n'
      'Data points can be discarded either because they are '
      'physically outside the image frame boundaries, because '
      'they were rejected by sigma clipping, or because they are '
      'masked or nonfinite.')),
    ('n_iter', int, None,
     'The number of iterations used to fit the isophote.'),
    ('valid', bool, None,
     'The status of the fitting operation.'),
    ('stop_code', int, None,
     'The fitting stop code.'),
    ('total_flux_ellipse', float, None,
     ('The sum of the valid (unmasked and finite) pixel values '
      'inside the ellipse.')),
    ('total_flux_circle', float, None,
     ('The sum of the valid (unmasked and finite) pixel values '
      'inside a circle with the same ``sma`` as the ellipse.')),
    ('n_pixels_ellipse', float, None,
     'The total number of valid pixels inside the ellipse.'),
    ('n_pixels_circle', float, None,
     ('The total number of valid pixels inside a circle with the '
      'same ``sma`` as the ellipse.')),
    ('a3', float, None,
     ('The third-order harmonic amplitude of the sine term, '
      'divided by the semimajor axis length and the absolute '
      'value of the local radial intensity gradient.')),
    ('b3', float, None,
     ('The third-order harmonic amplitude of the cosine term, '
      'divided by the semimajor axis length and the absolute '
      'value of the local radial intensity gradient.')),
    ('a4', float, None,
     ('The fourth-order harmonic amplitude of the sine term, '
      'divided by the semimajor axis length and the absolute '
      'value of the local radial intensity gradient.')),
    ('b4', float, None,
     ('The fourth-order harmonic amplitude of the cosine term, '
      'divided by the semimajor axis length and the absolute '
      'value of the local radial intensity gradient.\n\n'
      'It is positive for disky isophotes and negative for boxy '
      'isophotes.')),
    ('a3_err', float, None,
     ('The error associated with '
      '`~photutils.isophote.IsophoteList.a3`.')),
    ('b3_err', float, None,
     ('The error associated with '
      '`~photutils.isophote.IsophoteList.b3`.')),
    ('a4_err', float, None,
     ('The error associated with '
      '`~photutils.isophote.IsophoteList.a4`.')),
    ('b4_err', float, None,
     ('The error associated with '
      '`~photutils.isophote.IsophoteList.b4`.')),
)


def _make_property(name, dtype, unit, description):
    if unit is None:
        def getter(self):
            return self._collect_as_array(name, dtype=dtype)
    elif unit == u.deg:
        # The properties in degrees are the angles. The isophotes store
        # them as floats in radians in private ``_<name>_rad``
        # attributes, and the unit is attached once to the whole array.
        def getter(self):
            return _to_degrees(self._collect_as_array(f'_{name}_rad',
                                                      dtype=dtype))
    else:
        msg = (f'The {name!r} isophote property has the unit {unit}. '
               'Only angle properties in degrees, stored in radians in '
               f"a private '_{name}_rad' attribute, can have a unit.")
        raise ValueError(msg)

    getter.__name__ = name
    getter.__qualname__ = f'IsophoteList.{name}'
    getter.__doc__ = description
    return property(getter)


for _name, _dtype, _unit, _description in _ISOPHOTE_PROPERTIES:
    setattr(IsophoteList, _name,
            _make_property(_name, _dtype, _unit, _description))

del _name, _dtype, _unit, _description

# The names of the generated IsophoteList properties.
_PROPERTY_NAMES = tuple(entry[0] for entry in _ISOPHOTE_PROPERTIES)


# The isophote_list properties included in to_table() by default.
_DEFAULT_COLUMNS = ('sma', 'intensity', 'intensity_err', 'ellipticity',
                    'ellipticity_err', 'orientation', 'orientation_err',
                    'gradient', 'gradient_err', 'gradient_rel_err', 'x0',
                    'x0_err', 'y0', 'y0_err', 'n_data', 'n_flagged',
                    'n_iter', 'stop_code')


def _as_column_list(columns):
    """
    Return a new list of column names from a name or an iterable of
    names.
    """
    return [columns] if isinstance(columns, str) else list(columns)


def _warn_deprecated_shortcut(name, replacement):
    # Remove in 4.0
    # The stacklevel points at the caller of IsophoteList.to_table,
    # through the deprecated_positional_kwargs wrapper.
    msg = (f'The columns={name!r} value was deprecated in version 3.2. '
           f'Use {replacement} instead. It will be removed in version '
           '4.0.')
    warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=5)


def _isophote_list_to_table(isophote_list, *, columns=None):
    """
    Convert an `~photutils.isophote.IsophoteList` instance to a
    `~astropy.table.QTable`.

    Parameters
    ----------
    isophote_list : list of `~photutils.isophote.Isophote` or \
            `~photutils.isophote.IsophoteList` instance
        A list of isophotes.

    columns : list of str, `None`, optional
        The names of the isophote properties to include. If `None`,
        the ``default_columns`` of ``isophote_list`` are used. A
        deprecated name gives a column with its new name and emits a
        deprecation warning.

    Returns
    -------
    result : `~astropy.table.QTable`
        A table with one row per isophote. The ``orientation`` and
        ``orientation_err`` columns are `~astropy.units.Quantity`
        columns in degrees.

    Raises
    ------
    ValueError
        If any name in ``columns`` is not a valid (or deprecated)
        column name, or if ``'main'`` or ``'all'`` is input together
        with other names.
    """
    # The IsophoteList properties are arrays with a numerical dtype,
    # where values that are not defined are NaN.
    if not isinstance(isophote_list, IsophoteList):
        isophote_list = IsophoteList(isophote_list)

    all_names = _PROPERTY_NAMES

    # The deprecated 'main' and 'all' shortcuts are allowed only in
    # the columns argument, not in default_columns.
    allowed_columns = {*all_names, 'main', 'all'}
    if columns is None:
        columns = isophote_list.default_columns
        allowed_columns = set(all_names)

    table_columns = validate_table_columns(
        columns, allowed_columns, deprecated_names=_DEPRECATED_COLUMNS)
    if len(table_columns) > 1 and {'main', 'all'} & set(table_columns):
        msg = ("'main' and 'all' must be input by themselves, not in a "
               'list with other column names.')
        raise ValueError(msg)

    # Replace with QTable in 4.0
    since = _DEPRECATED_COLUMNS_SINCE
    until = '4.0'
    isotable = create_empty_deprecated_qtable(
        _DEPRECATED_COLUMNS, since=since, until=until)

    isotable.meta.update(_get_meta())  # keep isotable.meta type

    # The names of the properties that are the table columns.
    if table_columns == ['all']:
        _warn_deprecated_shortcut('all', "the 'properties' attribute")
        names = list(all_names)
    elif table_columns == ['main']:
        _warn_deprecated_shortcut('main', 'columns=None')
        names = _as_column_list(isophote_list.default_columns)
    else:
        # A deprecated name uses the new name, with a single deprecation
        # warning instead of one per isophote.
        names = []
        for name in table_columns:
            if name in _DEPRECATED_COLUMNS:
                new_name = _DEPRECATED_COLUMNS[name]
                # The names that are not attributes were only columns.
                noun = ('attribute' if name in _DEPRECATED_ATTRIBUTES
                        else 'column')
                msg = (f'The {name!r} {noun} was deprecated in version '
                       f'{since[name]}. Use {new_name!r} instead. It will be '
                       f'removed in version {until}.')
                warnings.warn(msg, PhotutilsDeprecationWarning,
                              stacklevel=4)
                names.append(new_name)
            else:
                names.append(name)

    for name in names:
        isotable[name] = getattr(isophote_list, name)

    return isotable


_REQUIRED_COLUMNS = ('sma', 'x0', 'y0', 'ellipticity', 'orientation')

# The columns that may be a Quantity in pixels. The other columns of
# properties without a unit may be a dimensionless Quantity.
_PIXEL_COLUMNS = ('sma', 'x0', 'x0_err', 'y0', 'y0_err')

# The checks of the column values, as (test, requirement) pairs. The
# test returns a boolean array that is True for the valid values. The
# angles are tested in radians.
_VALUE_CHECKS = {
    'sma': (lambda value: np.isfinite(value) & (value >= 0),
            'finite, nonnegative values'),
    'x0': (np.isfinite, 'finite values'),
    'y0': (np.isfinite, 'finite values'),
    'ellipticity': (lambda value: (np.isfinite(value) & (value >= 0)
                                   & (value < 1)),
                    'finite values that are at least 0 and less than 1'),
    'orientation': (np.isfinite, 'finite values'),
    'orientation_err': (lambda value: ~np.isinf(value),
                        'finite or NaN values'),
}

# The values of the properties that are not in a table.
_MISSING_VALUES = {name: {float: np.nan, int: 0, bool: True}[dtype]
                   for name, dtype, _, _ in _ISOPHOTE_PROPERTIES}


def _unitless_column_value(data, unit, name, column_name):
    """
    Return the values of a column that has a unit for a property that
    has no unit.

    The column must be dimensionless, or in pixels for the
    ``_PIXEL_COLUMNS``.
    """
    allowed_units = [u.dimensionless_unscaled]
    requirement = 'dimensionless'
    if name in _PIXEL_COLUMNS:
        allowed_units.append(u.pix)
        requirement = 'dimensionless or in pixels'
    for allowed_unit in allowed_units:
        if unit == allowed_unit:
            return data
        if unit.is_equivalent(allowed_unit):
            return u.Quantity(data, unit).to_value(allowed_unit)
    msg = f'The {column_name!r} column must be {requirement}, not {unit}.'
    raise ValueError(msg)


def _isophote_list_from_table(cls, table, *, step, linear_growth):
    """
    Create an `~photutils.isophote.IsophoteList` from a table. See
    `~photutils.isophote.IsophoteList.from_table`.
    """
    if not isinstance(table, Table):
        msg = ('An astropy Table or QTable is required, not '
               f'{type(table).__name__}.')
        raise TypeError(msg)

    msg = f'step must be a positive finite number, got {step!r}.'
    try:
        step = float(step)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not (np.isfinite(step) and step > 0):
        raise ValueError(msg)

    # Remove in 4.0
    # Only the names in table.colnames are used to read the table,
    # because a table made by to_table warns on an old column name.
    matches = {}
    for name in table.colnames:
        new_name = (name if name in _PROPERTY_NAMES
                    else _DEPRECATED_COLUMNS.get(name))
        if new_name is not None:
            matches.setdefault(new_name, []).append(name)

    columns = {}
    for new_name, names in matches.items():
        if len(names) > 1:
            msg = (f'The table columns {names[0]!r} and {names[1]!r} both '
                   f'give {new_name!r}. Remove one of them.')
            raise ValueError(msg)
        name = names[0]
        if name != new_name:
            msg = (f'The {name!r} column name was deprecated in version '
                   f'{_DEPRECATED_COLUMNS_SINCE[name]}. Use {new_name!r} '
                   'instead. It will be removed in version 4.0.')
            warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=3)
        columns[new_name] = name

    for name in _REQUIRED_COLUMNS:
        if name not in columns:
            msg = f'The table must have a {name!r} column.'
            raise ValueError(msg)

    # Each value is a list of Python scalars, with angles in radians.
    dtypes = {name: dtype for name, dtype, _, _ in _ISOPHOTE_PROPERTIES}
    values = {}
    for name, column_name in columns.items():
        column = table[column_name]
        if hasattr(column, 'unmasked'):  # astropy Masked Quantity
            data = np.asarray(column.unmasked)
        else:
            data = np.ma.getdata(column)
        mask = getattr(column, 'mask', None)
        if mask is None or mask is np.ma.nomask:
            mask = np.zeros(len(data), dtype=bool)
        mask = np.asarray(mask, dtype=bool)

        # A float column may hold numeric text, e.g., from a text file.
        if dtypes[name] is float:
            try:
                data = np.asarray(data, dtype=float)
            except (TypeError, ValueError):
                msg = f'The {column_name!r} column must contain numbers.'
                raise ValueError(msg) from None

        if name in ('orientation', 'orientation_err'):
            unit = getattr(column, 'unit', None)
            if unit is None:
                msg = (f'The {column_name!r} column must have an angular '
                       'unit. Set its unit, e.g., '
                       f"table[{column_name!r}].unit = 'deg'.")
                raise ValueError(msg)
            try:
                data = u.Quantity(data, unit).to_value(u.rad)
            except u.UnitConversionError:
                unit_name = unit.to_string() or 'a dimensionless unit'
                msg = (f'The {column_name!r} column must have an angular '
                       f'unit, not {unit_name}.')
                raise ValueError(msg) from None
        elif (unit := getattr(column, 'unit', None)) is not None:
            data = _unitless_column_value(data, unit, name, column_name)

        if mask.any():
            if dtypes[name] is not float or name in _REQUIRED_COLUMNS:
                msg = (f'The {column_name!r} column has masked values, '
                       'which are not allowed in this column.')
                raise ValueError(msg)
            data = np.where(mask, np.nan, data.astype(float))

        if name == 'valid' and data.dtype.kind in 'USO':
            mapping = {'true': True, 'false': False}
            try:
                data = [mapping[str(item).strip().lower()]
                        for item in data]
            except KeyError:
                msg = ("The 'valid' column must contain True or False "
                       'values.')
                raise ValueError(msg) from None
        elif name == 'valid' and (data.dtype.kind not in 'biuf'
                                  or not np.all((data == 0) | (data == 1))):
            msg = "The 'valid' column must contain True or False values."
            raise ValueError(msg)

        data = np.asarray(data)
        if name in _VALUE_CHECKS:
            test, requirement = _VALUE_CHECKS[name]
            if not np.all(is_valid := test(data)):
                index = np.flatnonzero(~is_valid)[0]
                msg = (f'The {column_name!r} column must contain '
                       f'{requirement}. The value in row {index} is '
                       f'{data[index]}.')
                raise ValueError(msg)
        if dtypes[name] is int:
            integral = data.dtype.kind in 'biu' or (
                data.dtype.kind == 'f'
                and np.all(np.isfinite(data) & (data == np.round(data))))
            if not integral:
                msg = (f'The {column_name!r} column must contain integer '
                       'values.')
                raise ValueError(msg)
        values[name] = data.astype(dtypes[name]).tolist()

    n_rows = len(table)
    isophotes = []
    for i in range(n_rows):
        geometry = EllipseGeometry._from_radians(
            values['x0'][i], values['y0'][i], values['sma'][i],
            values['ellipticity'][i], values['orientation'][i],
            step=step, linear_growth=linear_growth)
        row = {}
        for name, missing in _MISSING_VALUES.items():
            if name in _REQUIRED_COLUMNS:
                continue
            attr = name
            if name == 'orientation_err':
                attr = '_orientation_err_rad'
            row[attr] = values[name][i] if name in values else missing
        isophotes.append(Isophote._from_values(geometry, row))

    return cls(isophotes)
