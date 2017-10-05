# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .harmonics import (fit_1st_and_2nd_harmonics,
                        first_and_2nd_harmonic_function, fit_upper_harmonic)


__all__ = ['Isophote', 'IsophoteList']


def print_header():
    """
    Helper function that prints at stdout a table header to tell the
    user what are the columns printed by the iterating algorithm.
    """

    print('#')
    print('# Semi-      Isophote         Ellipticity    Position     Grad.'
          '   Data  Flag Iter. Stop')
    print('# major        mean                           Angle        rel.'
          '                    code')
    print('# axis       intensity                                    error')
    print('#(pixel)                                     (degree)')
    print('#')


class Isophote(object):
    """
    This class is basically a container that holds the results of a
    single isophote fit.  The actual extracted sample at the given
    isophote (sampled intensities along the elliptical path on the
    image) is kept as an attribute of this class. The container concept
    helps in segregating information directly related to the sample,
    from information that more closely relates to the fitting process,
    such as status codes, errors for isophote parameters (as defined by
    the old STSDAS code), and the like.

    Parameters
    ----------
    sample : instance of Sample
        the sample information
    niter : int
        number of iterations used to fit the isophote
    valid : boolean
        status of the fitting operation
    stop_code : int
        stop code
            0. normal.
            1. less than pre-specified fraction of the extracted data
               points are valid.
            2. exceeded maximum number of iterations.
            3. singular matrix in harmonic fit, results may not be valid.
               Also signals insufficient number of data points to fit.
            4. small or wrong gradient, or ellipse diverged. Subsequent
               ellipses at larger or smaller semi-major axis may have the
               same constant geometric parameters. It's also used when the
               user turns off the fitting algorithm via the `maxrit`
               fitting parameter (see Ellipse class).
            5. ellipse diverged; not even the minimum number of iterations
               could be executed. Subsequent ellipses at larger or
               smaller semi-major axis may have the same constant
               geometric parameters.
           -1. internal use.

    Attributes
    ----------
    sma : float
        semi-major axis length (pixels)
    intens : float
        mean intensity value along the elliptical path
    eps : float
        ellipticity
    pa : float
        position angle (rad)
    x0 : float
        center coordinate (pixel)
    y0 : float
        center coordinate (pixel)
    rms : float
        root-mean-sq of intensity values along the elliptical path
    int_err : float
        error of the mean (rms / sqrt(# data points)))
    ellip_err : float
        ellipticity error
    pa_err : float
        position angle error (rad)
    x0_err : float
        error of center coordinate X
    y0_err : float
        error of center coordinate Y
    pix_stddev : float
        estimate of pixel standard deviation (rms * sqrt(average sector
        integration area))
    grad : float
        local radial intensity gradient
    grad_error : float
        measurement error of local radial intensity gradient
    grad_r_error : float
        relative error of local radial intensity gradient
    tflux_e : float
        sum of all pixels inside ellipse
    npix_e : int
        total number of valid pixels inside ellipse
    tflux_c : float
        sum of all pixels inside circle with same `sma` as ellipse
    npix_c : int
        total number of valid pixels inside circle
    sarea : float
        average sector area on isophote (pixel)
    ndata : int
        number of actual (extracted) data points
    nflag : int
        number of discarded data points. Data points can be discarded
        either because they are physically outside the image frame
        boundaries, because they were rejected by sigma-clipping, or
        they are masked.
    a3, b3, a4, b4 : float higher order harmonics that measure the
        deviations from a perfect ellipse.  These values ar actually the
        raw harmonic amplitude divided by the local radial gradient and
        the semi-major axis length, so they can directly be compared
        with each other.
    a3_err, b3_err, a4_err, b4_err : float
        errors of the a3, b3, a4, b4 attributes
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

    @property
    def eps(self):
        return self.sample.geometry.eps

    @property
    def pa(self):
        return self.sample.geometry.pa

    @property
    def x0(self):
        return self.sample.geometry.x0

    @property
    def y0(self):
        return self.sample.geometry.y0

    def _compute_fluxes(self):
        # Compute integrated flux inside ellipse, as well as inside
        # circle defined by semi-major axis. Pixels in a square section
        # enclosing circle are scanned; the distance of each pixel to
        # the isophote center is compared both with the semi-major axis
        # length and with the length of the ellipse radius vector, and
        # integrals are updated if the pixel distance is smaller.

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

        return tflux_e, tflux_c, npix_e, npix_c

    def _compute_deviations(self, sample, n):
        # compute deviations from a perfect ellipse, based on the
        # amplitudes and errors for harmonic `n`. Note that we first
        # subtract the 1st and 2nd harmonics from the raw data.
        try:
            coeffs = fit_1st_and_2nd_harmonics(self.sample.values[0],
                                               self.sample.values[2])
            coeffs = coeffs[0]
            model = first_and_2nd_harmonic_function(self.sample.values[0],
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
        # compute parameter errors based on the diagonal of the
        # covariance matrix of the four harmonic coefficients for
        # harmonics n=1 and n=2.
        try:
            coeffs = fit_1st_and_2nd_harmonics(self.sample.values[0],
                                               self.sample.values[2])
            covariance = coeffs[1]
            coeffs = coeffs[0]
            model = first_and_2nd_harmonic_function(self.sample.values[0],
                                                    coeffs)
            residual_rms = np.std(self.sample.values[2] - model)
            errors = np.diagonal(covariance) * residual_rms

            eps = self.sample.geometry.eps
            pa = self.sample.geometry.pa

            # parameter errors result from direct projection of
            # coefficient errors.  These showed to be the error estimators
            # that best convey the errors measured in Monte Carlo
            # experiments (see reference in ellipse help page).
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

    def __repr__(self):
        return "sma=%7.2f" % (self.sma)

    def __str__(self):
        if self.grad_r_error:
            s = ('{0:7.2f}   {1:9.2f} ({2:5.2f}) {3:5.3f} ({4:5.3f}) '
                 '{5:6.2f} ({6:4.1f})  {7:5.3f}  {8:4d}  {9:4d} {10:4d}'
                 '  {11:4d}').format(self.sample.geometry.sma, self.intens,
                                     self.int_err, self.sample.geometry.eps,
                                     self.ellip_err,
                                     self.sample.geometry.pa / np.pi * 180.,
                                     self.pa_err / np.pi * 180.,
                                     self.grad_r_error, self.ndata,
                                     self.nflag, self.niter, self.stop_code)
        else:
            s = ('{0:7.2f}   {1:9.2f} ({2:5.2f}) {3:5.3f} ({4:5.3f}) '
                 '{5:6.2f} ({6:4.1f})  None   {7:4d}  {8:4d} {9:4d}  {10:4d}'
                 .format(self.sample.geometry.sma, self.intens, self.int_err,
                         self.sample.geometry.eps, self.ellip_err,
                         self.sample.geometry.pa / np.pi * 180.,
                         self.pa_err / np.pi * 180., self.ndata, self.nflag,
                         self.niter, self.stop_code))
        return s

    def fix_geometry(self, isophote):
        """
        This method should be called when the fitting goes berserk and
        delivers an isophote with bad geometry, such as with eps>1 or
        another meaningless situation. This is not a problem in itself
        when fitting any given isophote, but will create an error when
        the affected isophote is used as starting guess for the next
        fit.

        The method will set the geometry of the affected isophote to be
        identical with the geometry of the isophote used as input value.

        Parameters
        ----------
        isophote : instance of Isophote
            the isophote where to take geometry information from
        """

        self.sample.geometry.eps = isophote.sample.geometry.eps
        self.sample.geometry.pa = isophote.sample.geometry.pa
        self.sample.geometry.x0 = isophote.sample.geometry.x0
        self.sample.geometry.y0 = isophote.sample.geometry.y0

    def sampled_coordinates(self):
        """
        Returns the X-Y coordinates where the image was sampled in
        order to get the intensities associated with this isophote.

        Returns
        -------
        1-D numpy arrays
            two arrays with the X and Y coordinates, respectively
        """

        return self.sample.coordinates()

    # These two methods are useful for sorting lists of instances. Note
    # that __lt__ is the python3 way of supporting sorting. This might
    # not work under python2.
    @property
    def sma(self):
        return self.sample.geometry.sma

    def __lt__(self, other):
        if hasattr(other, 'sma'):
            return self.sma < other.sma


class CentralPixel(Isophote):
    """
    Container for the central pixel in the galaxy image.

    For convenience, the CentralPixel class inherits from
    the Isophote class, although it's not really a true
    isophote but just a single intensity value at the central
    position. Thus, most of its attributes are hardcoded to
    None, or other default value when appropriate.

    Parameters
    ----------
    sample : instance of Sample
        the sample information
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

    def __str__(self):
        return '   0.00   {0:9.2f}'.format(self.intens)


class IsophoteList(Isophote, list):
    """
    This class is a convenience container that provides the same
    attributes that the Isophote class offers, except that scalar
    attributes are replaced by numpy array attributes. These arrays
    reflect the values of the given attribute across the entire list of
    Isophote instances provided at constructor time.

    The class extends the 'list' built-in, thus providing basic list
    behavior such as slicing, appending, and support for '+' and '+='
    operators.

    Parameters
    ----------
    iso_list : list
        a python list with Isophote instances
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
        temp = self._list.copy()    # shallow copy!
        temp.extend(value._list)
        return IsophoteList(temp)

    def get_closest(self, sma):
        """
        Returns the Isophote instance that has the closest semi-major
        axis length to the passed parameter

        Parameters
        ----------
        sma : float
            a value for the semi-major axis length

        Returns
        -------
        Isophote instance
            the instance with the closest sma value
        """

        index = (np.abs(self.sma - sma)).argmin()
        return self._list[index]

    def _collect_as_array(self, attr_name):
        return np.array(self._collect_as_list(attr_name), dtype=np.float64)

    def _collect_as_list(self, attr_name):
        result = []
        for k in range(len(self._list)):
            result.append(getattr(self._list[k], attr_name))
        return result

    @property
    def sample(self):
        return self._collect_as_list('sample')

    @property
    def sma(self):
        return self._collect_as_array('sma')

    @property
    def intens(self):
        return self._collect_as_array('intens')

    @property
    def eps(self):
        return self._collect_as_array('eps')

    @property
    def pa(self):
        return self._collect_as_array('pa')

    @property
    def x0(self):
        return self._collect_as_array('x0')

    @property
    def y0(self):
        return self._collect_as_array('y0')

    @property
    def rms(self):
        return self._collect_as_array('rms')

    @property
    def int_err(self):
        return self._collect_as_array('int_err')

    @property
    def pix_stddev(self):
        return self._collect_as_array('pix_stddev')

    @property
    def ellip_err(self):
        return self._collect_as_array('ellip_err')

    @property
    def pa_err(self):
        return self._collect_as_array('pa_err')

    @property
    def x0_err(self):
        return self._collect_as_array('x0_err')

    @property
    def y0_err(self):
        return self._collect_as_array('y0_err')

    @property
    def grad(self):
        return self._collect_as_array('grad')

    @property
    def grad_error(self):
        return self._collect_as_array('grad_error')

    @property
    def grad_r_error(self):
        return self._collect_as_array('grad_r_error')

    @property
    def sarea(self):
        return self._collect_as_array('sarea')

    @property
    def ndata(self):
        return self._collect_as_array('ndata')

    @property
    def nflag(self):
        return self._collect_as_array('nflag')

    @property
    def niter(self):
        return self._collect_as_array('niter')

    @property
    def valid(self):
        return self._collect_as_array('valid')

    @property
    def stop_code(self):
        return self._collect_as_array('stop_code')

    @property
    def tflux_e(self):
        return self._collect_as_array('tflux_e')

    @property
    def tflux_c(self):
        return self._collect_as_array('tflux_c')

    @property
    def npix_e(self):
        return self._collect_as_array('npix_e')

    @property
    def npix_c(self):
        return self._collect_as_array('npix_c')

    @property
    def a3(self):
        return self._collect_as_array('a3')

    @property
    def b3(self):
        return self._collect_as_array('b3')

    @property
    def a4(self):
        return self._collect_as_array('a4')

    @property
    def b4(self):
        return self._collect_as_array('b4')

    @property
    def a3_err(self):
        return self._collect_as_array('a3_err')

    @property
    def b3_err(self):
        return self._collect_as_array('b3_err')

    @property
    def a4_err(self):
        return self._collect_as_array('a4_err')

    @property
    def b4_err(self):
        return self._collect_as_array('b4_err')
