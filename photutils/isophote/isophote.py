# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections import OrderedDict

import numpy as np
from astropy.table import QTable
import astropy.units as u

from .harmonics import (fit_first_and_second_harmonics,
                        first_and_second_harmonic_function,
                        fit_upper_harmonic)


__all__ = ['Isophote', 'IsophoteList']


class Isophote:
    """
    Container class to store the results of single isophote fit.

    The extracted data sample at the given isophote (sampled intensities
    along the elliptical path on the image) is also kept as an attribute
    of this class. The container concept helps in segregating
    information directly related to the sample, from information that
    more closely relates to the fitting process, such as status codes,
    errors for isophote parameters, and the like.

    Parameters
    ----------
    sample : `~photutils.isophote.EllipseSample` instance
        The sample information.
    niter : int
        The number of iterations used to fit the isophote.
    valid : bool
        The status of the fitting operation.
    stop_code : int
        The fitting stop code:

           *  0: Normal.
           *  1: Fewer than the pre-specified fraction of the extracted
              data points are valid.
           *  2: Exceeded maximum number of iterations.
           *  3: Singular matrix in harmonic fit, results may not be
              valid.  This also signals an insufficient number of
              data points to fit.
           *  4: Small or wrong gradient, or ellipse diverged. Subsequent
              ellipses at larger or smaller semimajor axis may have
              the same constant geometric parameters. It's also used
              when the user turns off the fitting algorithm via the
              ``maxrit`` fitting parameter (see the
              `~photutils.isophote.Ellipse` class).
           *  5: Ellipse diverged; not even the minimum number of
              iterations could be executed. Subsequent ellipses at
              larger or smaller semimajor axis may have the same
              constant geometric parameters.
           * -1: Internal use.

    Attributes
    ----------
    rms : float
        The root-mean-square of intensity values along the elliptical
        path.
    int_err : float
        The error of the mean (rms / sqrt(# data points)).
    ellip_err : float
        The ellipticity error.
    pa_err : float
        The position angle error (radians).
    x0_err : float
        The error associated with the center x coordinate.
    y0_err : float
        The error associated with the center y coordinate.
    pix_stddev : float
        The estimate of pixel standard deviation (rms * sqrt(average
        sector integration area)).
    grad : float
        The local radial intensity gradient.
    grad_error : float
        The measurement error of the local radial intensity gradient.
    grad_r_error : float
        The relative error of local radial intensity gradient.
    tflux_e : float
        The sum of all pixels inside the ellipse.
    npix_e : int
        The total number of valid pixels inside the ellipse.
    tflux_c : float
        The sum of all pixels inside a circle with the same ``sma`` as
        the ellipse.
    npix_c : int
        The total number of valid pixels inside a circle with the same
        ``sma`` as the ellipse.
    sarea : float
        The average sector area on the isophote (pixel**2).
    ndata : int
        The number of extracted data points.
    nflag : int
        The number of discarded data points. Data points can be
        discarded either because they are physically outside the image
        frame boundaries, because they were rejected by sigma-clipping,
        or they are masked.
    a3, b3, a4, b4 : float
        The higher order harmonics that measure the deviations from a
        perfect ellipse.  These values are actually the raw harmonic
        amplitudes divided by the local radial gradient and the
        semimajor axis length, so they can directly be compared with
        each other.
    a3_err, b3_err, a4_err, b4_err : float
        The errors associated with the ``a3``, ``b3``, ``a4``, and
        ``b4`` attributes.
    """

    def __init__(self, sample, niter, valid, stop_code):
        self.sample = sample
        self.niter = niter
        self.valid = valid
        self.stop_code = stop_code

        self.intens = sample.mean
        self.rms = np.std(sample.values[2])
        self.int_err = self.rms / np.sqrt(sample.actual_points)
        self.pix_stddev = self.rms * np.sqrt(sample.sector_area)
        self.grad = sample.gradient
        self.grad_error = sample.gradient_error

        self.grad_r_error = sample.gradient_relative_error
        self.sarea = sample.sector_area
        self.ndata = sample.actual_points
        self.nflag = sample.total_points - sample.actual_points

        # flux contained inside ellipse and circle
        (self.tflux_e, self.tflux_c, self.npix_e,
         self.npix_c) = self._compute_fluxes()

        self._compute_errors()

        # deviations from a perfect ellipse
        (self.a3, self.b3, self.a3_err,
         self.b3_err) = self._compute_deviations(sample, 3)
        (self.a4, self.b4, self.a4_err,
         self.b4_err) = self._compute_deviations(sample, 4)

    # This method is useful for sorting lists of instances. Note
    # that __lt__ is the python3 way of supporting sorting.
    def __lt__(self, other):
        if hasattr(other, 'sma'):
            return self.sma < other.sma

    def __str__(self):
        return str(self.to_table())

    @property
    def sma(self):
        """The semimajor axis length (pixels)."""
        return self.sample.geometry.sma

    @property
    def eps(self):
        """The ellipticity of the ellipse."""
        return self.sample.geometry.eps

    @property
    def pa(self):
        """The position angle (radians) of the ellipse."""
        return self.sample.geometry.pa

    @property
    def x0(self):
        """The center x coordinate (pixel)."""
        return self.sample.geometry.x0

    @property
    def y0(self):
        """The center y coordinate (pixel)."""
        return self.sample.geometry.y0

    def _compute_fluxes(self):
        """
        Compute integrated flux inside ellipse, as well as inside a
        circle defined with the same semimajor axis.

        Pixels in a square section enclosing circle are scanned; the
        distance of each pixel to the isophote center is compared both
        with the semimajor axis length and with the length of the
        ellipse radius vector, and integrals are updated if the pixel
        distance is smaller.
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

        # Integrate
        if (jmax-jmin > 1) and (imax-imin) > 1:
            y, x = np.mgrid[jmin:jmax, imin:imax]
            radius, angle = self.sample.geometry.to_polar(x, y)
            radius_e = self.sample.geometry.radius(angle)

            midx = (radius <= sma)
            values = self.sample.image[y[midx], x[midx]]
            tflux_c = np.ma.sum(values)
            npix_c = np.ma.count(values)

            midx2 = (radius <= radius_e)
            values = self.sample.image[y[midx2], x[midx2]]
            tflux_e = np.ma.sum(values)
            npix_e = np.ma.count(values)
        else:
            tflux_e = 0.
            tflux_c = 0.
            npix_e = 0
            npix_c = 0

        return tflux_e, tflux_c, npix_e, npix_c

    def _compute_deviations(self, sample, n):
        """
        Compute deviations from a perfect ellipse, based on the
        amplitudes and errors for harmonic "n". Note that we first
        subtract the first and second harmonics from the raw data.
        """

        try:
            coeffs = fit_first_and_second_harmonics(self.sample.values[0],
                                                    self.sample.values[2])
            coeffs = coeffs[0]
            model = first_and_second_harmonic_function(self.sample.values[0],
                                                       coeffs)
            residual = self.sample.values[2] - model

            c = fit_upper_harmonic(residual, sample.values[2], n)
            covariance = c[1]
            ce = np.diagonal(covariance)
            c = c[0]

            a = c[1] / self.sma / sample.gradient
            b = c[2] / self.sma / sample.gradient

            # this comes from the old code. Likely it was based on
            # empirical experience with the STSDAS task, so we leave
            # it here without too much thought.
            gre = self.grad_r_error if self.grad_r_error is not None else 0.64

            a_err = abs(a) * np.sqrt((ce[1] / c[1])**2 + gre**2)
            b_err = abs(b) * np.sqrt((ce[2] / c[2])**2 + gre**2)

        except Exception as e:    # we want to catch everything
            a = b = a_err = b_err = None

        return a, b, a_err, b_err

    def _compute_errors(self):
        """
        Compute parameter errors based on the diagonal of the covariance
        matrix of the four harmonic coefficients for harmonics n=1 and
        n=2.
        """

        try:
            coeffs = fit_first_and_second_harmonics(self.sample.values[0],
                                                    self.sample.values[2])
            covariance = coeffs[1]
            coeffs = coeffs[0]
            model = first_and_second_harmonic_function(self.sample.values[0],
                                                       coeffs)
            residual_rms = np.std(self.sample.values[2] - model)
            errors = np.diagonal(covariance) * residual_rms

            eps = self.sample.geometry.eps
            pa = self.sample.geometry.pa

            # parameter errors result from direct projection of
            # coefficient errors.  These showed to be the error estimators
            # that best convey the errors measured in Monte Carlo
            # experiments (see Busko 1996; ASPC 101, 139).
            ea = abs(errors[2] / self.grad)
            eb = abs(errors[1] * (1. - eps) / self.grad)
            self.x0_err = np.sqrt((ea * np.cos(pa))**2 + (eb * np.sin(pa))**2)
            self.y0_err = np.sqrt((ea * np.sin(pa))**2 + (eb * np.cos(pa))**2)
            self.ellip_err = (abs(2. * errors[4] * (1. - eps) / self.sma /
                                  self.grad))
            if (abs(eps) > np.finfo(float).resolution):
                self.pa_err = (abs(2. * errors[3] * (1. - eps) / self.sma /
                                   self.grad / (1. - (1. - eps)**2)))
            else:
                self.pa_err = 0.
        except Exception as e:    # we want to catch everything
            self.x0_err = self.y0_err = self.pa_err = self.ellip_err = 0.

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

        Parameters
        ----------
        isophote : `~photutils.isophote.Isophote` instance
            The isophote from which to take the geometry information.
        """

        self.sample.geometry.eps = isophote.sample.geometry.eps
        self.sample.geometry.pa = isophote.sample.geometry.pa
        self.sample.geometry.x0 = isophote.sample.geometry.x0
        self.sample.geometry.y0 = isophote.sample.geometry.y0

    def sampled_coordinates(self):
        """
        Return the (x, y) coordinates where the image was sampled in
        order to get the intensities associated with this isophote.

        Returns
        -------
        x, y : 1D `~numpy.ndarray`
            The x and y coordinates as 1D arrays.
        """

        return self.sample.coordinates()

    def to_table(self):
        """
        Return the main isophote parameters as an astropy
        `~astropy.table.QTable`.

        Returns
        -------
        result : `~astropy.table.QTable`
            An astropy `~astropy.table.QTable` containing the main
            isophote paramters.
        """

        return _isophote_list_to_table([self])


class CentralPixel(Isophote):
    """
    Specialized Isophote class for the galaxy central pixel.

    This class holds only a single intensity value at the central
    position.  Thus, most of its attributes are hardcoded to `None` or a
    default value when appropriate.

    Parameters
    ----------
    sample : `~photutils.utils.EllipseSample` instance
        The sample information.
    """

    def __init__(self, sample):
        self.sample = sample
        self.niter = 0
        self.valid = True
        self.stop_code = 0

        self.intens = sample.mean

        # some values are set to zero to ease certain tasks
        # such as model building and plotting magnitude errors
        self.rms = None
        self.int_err = 0.0
        self.pix_stddev = None
        self.grad = 0.0
        self.grad_error = None
        self.grad_r_error = None
        self.sarea = None
        self.ndata = sample.actual_points
        self.nflag = sample.total_points - sample.actual_points

        self.tflux_e = self.tflux_c = self.npix_e = self.npix_c = None

        self.a3 = self.b3 = 0.0
        self.a4 = self.b4 = 0.0
        self.a3_err = self.b3_err = 0.0
        self.a4_err = self.b4_err = 0.0

        self.ellip_err = 0.
        self.pa_err = 0.
        self.x0_err = 0.
        self.y0_err = 0.

    @property
    def eps(self):
        return 0.

    @property
    def pa(self):
        return 0.

    @property
    def x0(self):
        return self.sample.geometry.x0


class IsophoteList(Isophote, list):
    """
    Container class that provides the same attributes as the
    `~photutils.isophote.Isophote` class, but for a list of isophotes.

    The attributes of this class are arrays representing the values of
    the attributes for the entire list of `~photutils.isophote.Isophote`
    instances.  See the `~photutils.isophote.Isophote` class for a
    description of the attributes.

    The class extends the `list` functionality, thus provides basic list
    behavior such as slicing, appending, and support for '+' and '+='
    operators.

    Parameters
    ----------
    iso_list : list of `~photutils.isophote.Isophote`
        A list of `~photutils.isophote.Isophote` instances.
    """

    def __init__(self, iso_list):
        self._list = iso_list

    def __len__(self):
        return len(self._list)

    def __delitem__(self, index):
        self._list.__delitem__(index)

    def __setitem__(self, index, value):
        self._list.__setitem__(index, value)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IsophoteList(self._list[index])
        return self._list.__getitem__(index)

    def __iter__(self):
        return self._list.__iter__()

    def sort(self):
        self._list.sort()

    def insert(self, index, value):
        self._list.insert(index, value)

    def append(self, value):
        self.insert(len(self) + 1, value)

    def extend(self, value):
        self._list.extend(value._list)

    def __iadd__(self, value):
        self.extend(value)
        return self

    def __add__(self, value):
        temp = self._list[:]    # shallow copy
        temp.extend(value._list)
        return IsophoteList(temp)

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

    def _collect_as_array(self, attr_name):
        return np.array(self._collect_as_list(attr_name), dtype=np.float64)

    def _collect_as_list(self, attr_name):
        return [getattr(iso, attr_name) for iso in self._list]

    @property
    def sample(self):
        """
        The isophote `~photutils.isophote.EllipseSample` information.
        """
        return self._collect_as_list('sample')

    @property
    def sma(self):
        """The semimajor axis length (pixels)."""
        return self._collect_as_array('sma')

    @property
    def intens(self):
        """The mean intensity value along the elliptical path."""
        return self._collect_as_array('intens')

    @property
    def int_err(self):
        """The error of the mean intensity (rms / sqrt(# data points))."""
        return self._collect_as_array('int_err')

    @property
    def eps(self):
        """The ellipticity of the ellipse."""
        return self._collect_as_array('eps')

    @property
    def ellip_err(self):
        """The ellipticity error."""
        return self._collect_as_array('ellip_err')

    @property
    def pa(self):
        """The position angle (radians) of the ellipse."""
        return self._collect_as_array('pa')

    @property
    def pa_err(self):
        """The position angle error (radians)."""
        return self._collect_as_array('pa_err')

    @property
    def x0(self):
        """The center x coordinate (pixel)."""
        return self._collect_as_array('x0')

    @property
    def x0_err(self):
        """The error associated with the center x coordinate."""
        return self._collect_as_array('x0_err')

    @property
    def y0(self):
        """The center y coordinate (pixel)."""
        return self._collect_as_array('y0')

    @property
    def y0_err(self):
        """The error associated with the center y coordinate."""
        return self._collect_as_array('y0_err')

    @property
    def rms(self):
        """
        The root-mean-square of intensity values along the elliptical
        path.
        """
        return self._collect_as_array('rms')

    @property
    def pix_stddev(self):
        """
        The estimate of pixel standard deviation (rms * sqrt(average
        sector integration area)).
        """
        return self._collect_as_array('pix_stddev')

    @property
    def grad(self):
        """The local radial intensity gradient."""
        return self._collect_as_array('grad')

    @property
    def grad_error(self):
        """
        The measurement error of the local radial intensity
        gradient.
        """
        return self._collect_as_array('grad_error')

    @property
    def grad_r_error(self):
        """
        The relative error of local radial intensity gradient.
        """
        return self._collect_as_array('grad_r_error')

    @property
    def sarea(self):
        """The average sector area on the isophote (pixel**2)."""
        return self._collect_as_array('sarea')

    @property
    def ndata(self):
        """The number of extracted data points."""
        return self._collect_as_array('ndata')

    @property
    def nflag(self):
        """
        The number of discarded data points. Data points can be
        discarded either because they are physically outside the image
        frame boundaries, because they were rejected by sigma-clipping,
        or they are masked.
        """
        return self._collect_as_array('nflag')

    @property
    def niter(self):
        """The number of iterations used to fit the isophote."""

        return self._collect_as_array('niter')

    @property
    def valid(self):
        """The status of the fitting operation."""
        return self._collect_as_array('valid')

    @property
    def stop_code(self):
        """The fitting stop code."""
        return self._collect_as_array('stop_code')

    @property
    def tflux_e(self):
        """The sum of all pixels inside the ellipse."""
        return self._collect_as_array('tflux_e')

    @property
    def tflux_c(self):
        """
        The sum of all pixels inside a circle with the same ``sma`` as
        the ellipse.
        """
        return self._collect_as_array('tflux_c')

    @property
    def npix_e(self):
        """The total number of valid pixels inside the ellipse."""
        return self._collect_as_array('npix_e')

    @property
    def npix_c(self):
        """
        The total number of valid pixels inside a circle with the same
        ``sma`` as the ellipse.
        """
        return self._collect_as_array('npix_c')

    @property
    def a3(self):
        """
        A third-order harmonic coefficent.  See the
        :func:`~photutils.isophote.fit_upper_harmonic` function for
        details.
        """
        return self._collect_as_array('a3')

    @property
    def b3(self):
        """
        A third-order harmonic coefficent.  See the
        :func:`~photutils.isophote.fit_upper_harmonic` function for
        details.
        """
        return self._collect_as_array('b3')

    @property
    def a4(self):
        """
        A fourth-order harmonic coefficent.  See the
        :func:`~photutils.isophote.fit_upper_harmonic` function for
        details.
        """
        return self._collect_as_array('a4')

    @property
    def b4(self):
        """
        A fourth-order harmonic coefficent.  See the
        :func:`~photutils.isophote.fit_upper_harmonic` function for
        details.
        """
        return self._collect_as_array('b4')

    @property
    def a3_err(self):
        """
        The error associated with `~photutils.isophote.IsophoteList.a3`.
        """
        return self._collect_as_array('a3_err')

    @property
    def b3_err(self):
        """
        The error associated with `~photutils.isophote.IsophoteList.b3`.
        """
        return self._collect_as_array('b3_err')

    @property
    def a4_err(self):
        """
        The error associated with `~photutils.isophote.IsophoteList.a4`.
        """
        return self._collect_as_array('a4_err')

    @property
    def b4_err(self):
        """
        The error associated with `~photutils.isophote.IsophoteList.b3`.
        """
        return self._collect_as_array('b4_err')

    def to_table(self):
        """
        Convert an `~photutils.isophote.IsophoteList` instance to a
        `~astropy.table.QTable` with the main isophote parameters.

        Returns
        -------
        result : `~astropy.table.QTable`
            An astropy QTable with the main isophote parameters.
        """
        return _isophote_list_to_table(self)


def _isophote_list_to_table(isophote_list):
    """
    Convert an `~photutils.isophote.IsophoteList` instance to
    a `~astropy.table.QTable`.

    Parameters
    ----------
    isophote_list : list of `~photutils.isophote.Isophote` or a `~photutils.isophote.IsophoteList` instance
        A list of isophotes.

    Returns
    -------
    result : `~astropy.table.QTable`
        An astropy QTable with the main isophote parameters.
    """

    properties = OrderedDict()
    properties['sma'] = 'sma'
    properties['intens'] = 'intens'
    properties['int_err'] = 'intens_err'
    properties['eps'] = 'ellipticity'
    properties['ellip_err'] = 'ellipticity_err'
    properties['pa'] = 'pa'
    properties['pa_err'] = 'pa_err'
    properties['grad_r_error'] = 'grad_rerr'
    properties['ndata'] = 'ndata'
    properties['nflag'] = 'flag'
    properties['niter'] = 'niter'
    properties['stop_code'] = 'stop_code'

    isotable = QTable()

    for k, v in properties.items():
        isotable[v] = np.array([getattr(iso, k) for iso in isophote_list])

        if k in ('pa', 'pa_err'):
            isotable[v] = isotable[v] * 180. / np.pi * u.deg

    return isotable
