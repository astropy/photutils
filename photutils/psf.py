# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for performing PSF fitting photometry on 2D arrays."""

from __future__ import division
import warnings
import numpy as np
from astropy.modeling.parameters import Parameter
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Fittable2DModel
from astropy.nddata.utils import extract_array, add_array, subpixel_indices
from .utils import mask_to_mirrored_num


__all__ = ['DiscretePRF', 'create_prf', 'psf_photometry',
           'GaussianPSF', 'subtract_psf']


class DiscretePRF(Fittable2DModel):
    """
    A discrete PRF model.

    The discrete PRF model stores images of the PRF at different subpixel
    positions or offsets as a lookup table. The resolution is given by the
    subsampling parameter, which states in how many subpixels a pixel is
    divided.

    The discrete PRF model class in initialized with a 4 dimensional
    array, that contains the PRF images at different subpixel positions.
    The definition of the axes is as following:

        1. Axis: y subpixel position
        2. Axis: x subpixel position
        3. Axis: y direction of the PRF image
        4. Axis: x direction of the PRF image

    The total array therefore has the following shape
    (subsampling, subsampling, prf_size, prf_size)

    Parameters
    ----------
    prf_array : ndarray
        Array containing PRF images.
    normalize : bool
        Normalize PRF images to unity.
    subsampling : int, optional
        Factor of subsampling. Default = 1.
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')
    linear = True

    def __init__(self, prf_array, normalize=True, subsampling=1):

        # Array shape and dimension check
        if subsampling == 1:
            if prf_array.ndim == 2:
                prf_array = np.array([[prf_array]])
        if prf_array.ndim != 4:
            raise TypeError('Array must have 4 dimensions.')
        if prf_array.shape[:2] != (subsampling, subsampling):
            raise TypeError('Incompatible subsampling and array size')
        if np.isnan(prf_array).any():
            raise Exception("Array contains NaN values. Can't create PRF.")

        # Normalize if requested
        if normalize:
            for i in range(prf_array.shape[0]):
                for j in range(prf_array.shape[1]):
                    prf_array[i, j] /= prf_array[i, j].sum()

        # Set PRF asttributes
        self._prf_array = prf_array
        self.subsampling = subsampling

        constraints = {'fixed': {'x_0': True, 'y_0': True}}
        x_0 = 0
        y_0 = 0
        amplitude = 1
        super(DiscretePRF, self).__init__(n_models=1, x_0=x_0, y_0=y_0,
                                          amplitude=amplitude, **constraints)
        self.fitter = LevMarLSQFitter()

        # Fix position per default
        self.x_0.fixed = True
        self.y_0.fixed = True

    @property
    def shape(self):
        """
        Shape of the PRF image.
        """
        return self._prf_array.shape[-2:]

    def evaluate(self, x, y, amplitude, x_0, y_0):
        """
        Discrete PRF model evaluation.

        Given a certain position and amplitude the corresponding image of
        the PSF is chosen and scaled to the amplitude. If x and y are
        outside the boundaries of the image, zero will be returned.

        Parameters
        ----------
        x : float
            x coordinate array in pixel coordinates.
        y : float
            y coordinate array in pixel coordinates.
        amplitude : float
            Model amplitude.
        x_0 : float
            x position of the center of the PRF.
        y_0 : float
            y position of the center of the PRF.
        """
        # Convert x and y to index arrays
        x = (x - x_0 + 0.5 + self.shape[1] // 2).astype('int')
        y = (y - y_0 + 0.5 + self.shape[0] // 2).astype('int')

        # Get subpixel indices
        y_sub, x_sub = subpixel_indices((y_0, x_0), self.subsampling)

        # Out of boundary masks
        x_bound = np.logical_or(x < 0, x >= self.shape[1])
        y_bound = np.logical_or(y < 0, y >= self.shape[0])
        out_of_bounds = np.logical_or(x_bound, y_bound)

        # Set out of boundary indices to zero
        x[x_bound] = 0
        y[y_bound] = 0
        result = amplitude * self._prf_array[int(y_sub), int(x_sub)][y, x]

        # Set out of boundary values to zero
        result[out_of_bounds] = 0
        return result

    def fit(self, data, indices):
        """
        Fit PSF/PRF to data.

        Fits the PSF/PRF to the data and returns the best fitting flux.
        If the data contains NaN values or if the source is not completely
        contained in the image data the fitting is omitted and a flux of 0
        is returned.

        For reasons of performance, indices for the data have to be created
        outside and passed to the function.

        The fit is performed on a slice of the data with the same size as
        the PRF.

        Parameters
        ----------
        data : ndarray
            Array containig image data.
        indices : ndarray
            Array with indices of the data. As
            returned by np.indices(data.shape)
        """
        # Extract sub array of the data of the size of the PRF grid
        sub_array_data = extract_array(data, self.shape,
                                       (self.y_0.value, self.x_0.value))

        # Fit only if PSF is completely contained in the image and no NaN
        # values are present
        if (sub_array_data.shape == self.shape and
                not np.isnan(sub_array_data).any()):
            y = extract_array(indices[0], self.shape,
                              (self.y_0.value, self.x_0.value))
            x = extract_array(indices[1], self.shape,
                              (self.y_0.value, self.x_0.value))
            # TODO: It should be discussed whether this is the right
            # place to fix the warning.  Maybe it should be handled better
            # in astropy.modeling.fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyUserWarning)
                m = self.fitter(self, x, y, sub_array_data)
            return m.amplitude.value
        else:
            return 0


class GaussianPSF(Fittable2DModel):
    """
    Symmetrical Gaussian PSF model.

    The PSF is evaluated by using the `scipy.special.erf` function
    on a fixed grid of the size of 1 pixel to assure flux conservation
    on subpixel scale.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian PSF.
    amplitude : float (default 1)
        The peak amplitude of the PSF.
    x_0 : float (default 0)
        Position of the peak in x direction.
    y_0 : float (default 0)
        Position of the peak in y direction.

    Notes
    -----
    The PSF model is evaluated according to the following formula:

        .. math::

            f(x, y) =
                \\frac{A}{0.02538010595464}
                \\left[
                \\textnormal{erf} \\left(\\frac{x - x_0 + 0.5}
                {\\sqrt{2} \\sigma} \\right) -
                \\textnormal{erf} \\left(\\frac{x - x_0 - 0.5}
                {\\sqrt{2} \\sigma} \\right)
                \\right]
                \\left[
                \\textnormal{erf} \\left(\\frac{y - y_0 + 0.5}
                {\\sqrt{2} \\sigma} \\right) -
                \\textnormal{erf} \\left(\\frac{y - y_0 - 0.5}
                {\\sqrt{2} \\sigma} \\right)
                \\right]

    Where ``erf`` denotes the error function and ``A`` is the amplitude.
    """

    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')
    sigma = Parameter('sigma')

    _erf = None

    def __init__(self, sigma, amplitude=1, x_0=0, y_0=0):
        if self._erf is None:
            from scipy.special import erf
            self.__class__._erf = erf

        constraints = {'fixed': {'x_0': True, 'y_0': True, 'sigma': True}}
        super(GaussianPSF, self).__init__(n_models=1, sigma=sigma,
                                          x_0=x_0, y_0=y_0,
                                          amplitude=amplitude, **constraints)

        # Default size is 8 * sigma
        self.shape = (int(8 * sigma) + 1, int(8 * sigma) + 1)
        self.fitter = LevMarLSQFitter()

        # Fix position per default
        self.x_0.fixed = True
        self.y_0.fixed = True

    def evaluate(self, x, y, amplitude, x_0, y_0, sigma):
        """
        Model function Gaussian PSF model.
        """
        psf = (1.0 *
               ((self._erf((x - x_0 + 0.5) / (np.sqrt(2) * sigma)) -
                 self._erf((x - x_0 - 0.5) / (np.sqrt(2) * sigma))) *
                (self._erf((y - y_0 + 0.5) / (np.sqrt(2) * sigma)) -
                 self._erf((y - y_0 - 0.5) / (np.sqrt(2) * sigma)))))
        return amplitude * psf / psf.max()

    def fit(self, data, indices):
        """
        Fit PSF/PRF to data.

        Fits the PSF/PRF to the data and returns the best fitting flux.
        If the data contains NaN values or if the source is not completely
        contained in the image data the fitting is omitted and a flux of 0
        is returned.

        For reasons of performance, indices for the data have to be created
        outside and passed to the function.

        The fit is performed on a slice of the data with the same size as
        the PRF.

        Parameters
        ----------
        data : ndarray
            Array containig image data.
        indices : ndarray
            Array with indices of the data. As
            returned by np.indices(data.shape)

        Returns
        -------
        flux : float
            Best fit flux value. Returns flux = 0 if PSF is not completely
            contained in the image or if NaN values are present.
        """
        # Set position
        position = (self.y_0.value, self.x_0.value)

        # Extract sub array with data of interest
        sub_array_data = extract_array(data, self.shape, position)

        # Fit only if PSF is completely contained in the image and no NaN
        # values are present
        if (sub_array_data.shape == self.shape and
                not np.isnan(sub_array_data).any()):
            y = extract_array(indices[0], self.shape, position)
            x = extract_array(indices[1], self.shape, position)
            m = self.fitter(self, x, y, sub_array_data)
            return m.amplitude.value
        else:
            return 0


def psf_photometry(data, positions, psf, mask=None, mode='sequential',
                   tune_coordinates=False):
    """
    Perform PSF/PRF photometry on the data.

    Given a PSF or PRF model, the model is fitted simultaneously or
    sequentially to the given positions to obtain an estimate of the
    flux. If required, coordinates are also tuned to match best the data.

    If the data contains NaN values or the PSF/PRF is not completely
    contained in the image, a flux of zero is returned.

    Parameters
    ----------
    data : ndarray
        Image data array
    positions : List or array
        List of positions in pixel coordinates
        where to fit the PSF/PRF.
    psf : `photutils.psf.DiscretePRF` or `photutils.psf.GaussianPSF`
        PSF/PRF model to fit to the data.
    mask : ndarray, optional
        Mask to be applied to the data.
    mode : {'sequential', 'simultaneous'}
        One of the following modes to do PSF/PRF photometry:
            * 'simultaneous'
                Fit PSF/PRF simultaneous to all given positions.
            * 'sequential' (default)
                Fit PSF/PRF one after another to the given positions.
    tune_coordinates : boolean
        If ``True`` the peak position of the PSF will be fit, if ``False``,
        it is frozen to the input value.

    Examples
    --------
    See `Spitzer PSF Photometry <http://nbviewer.ipython.org/gist/adonath/
    6550989/PSFPhotometrySpitzer.ipynb>`_ for a short tutorial.
    """
    # Check input array type and dimension.
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))

    # Fit coordinates if requested
    if tune_coordinates:
        psf.fixed['x_0'] = False
        psf.fixed['y_0'] = False
    else:
        psf.fixed['x_0'] = True
        psf.fixed['y_0'] = True

    # Actual photometry
    result = np.array([])
    indices = np.indices(data.shape)

    if mode == 'simultaneous':
        raise NotImplementedError('Simultaneous mode not implemented')
    elif mode == 'sequential':
        for position in positions:
                psf.x_0, psf.y_0 = position
                flux = psf.fit(data, indices)
                result = np.append(result, flux)
    else:
        raise Exception('Invalid photometry mode.')
    return result


def create_prf(data, positions, size, fluxes=None, mask=None, mode='mean',
               subsampling=1, fix_nan=False):
    """
    Estimate point response function (PRF) from image data.

    Given a list of positions and size this function estimates an image of
    the PRF by extracting and combining the individual PRFs from the given
    positions. Different modes of combining are available.

    NaN values are either ignored by passing a mask or can be replaced by
    the mirrored value with respect to the center of the PRF.

    Furthermore it is possible to specify fluxes to have a correct
    normalization of the individual PRFs. Otherwise the flux is estimated from
    a quadratic aperture of the same size as the PRF image.

    Parameters
    ----------
    data : array
        Data array
    positions : List or array
        List of pixel coordinate source positions to use in creating the PRF.
    size : odd int
        Size of the quadratic PRF image in pixels.
    mask : bool array, optional
        Boolean array to mask out bad values.
    fluxes : array, optional
        Object fluxes to normalize extracted PRFs.
    mode : {'mean', 'median'}
        One of the following modes to combine the extracted PRFs:
            * 'mean'
                Take the pixelwise mean of the extracted PRFs.
            * 'median'
                Take the pixelwise median of the extracted PRFs.
    subsampling : int
        Factor of subsampling of the PRF (default = 1).
    fix_nan : bool
        Fix NaN values in the data by replacing it with the
        mirrored value. Assuming that the PRF is symmetrical.

    Returns
    -------
    prf : `photutils.psf.DiscretePRF`
        Discrete PRF model estimated from data.

    Notes
    -----
    In Astronomy different definitions of Point Spread Function (PSF) and
    Point Response Function (PRF) are used. Here we assume that the PRF is
    an image of a point source after discretization e.g. with a CCD. This
    definition is equivalent to the `Spitzer definiton of the PRF
    <http://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/mopex/mopexusersguide/89/>`_.

    References
    ----------
    `Spitzer PSF vs. PRF
    <http://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/PRF_vs_PSF.pdf>`_

    `Kepler PSF calibration
    <http://keplerscience.arc.nasa.gov/CalibrationPSF.shtml>`_

    `The Kepler Pixel Response Function
    <http://adsabs.harvard.edu/abs/2010ApJ...713L..97B>`_
    """

    # Check input array type and dimension.
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))
    if size % 2 == 0:
        raise TypeError("Size must be odd.")

    if fluxes is not None and len(fluxes) != len(positions):
        raise TypeError("Position and flux arrays must be of equal length.")

    if mask is None:
        mask = np.isnan(data)

    if isinstance(positions, (list, tuple)):
        positions = np.array(positions)

    if isinstance(fluxes, (list, tuple)):
        fluxes = np.array(fluxes)

    if mode == 'mean':
        combine = np.ma.mean
    elif mode == 'median':
        combine = np.ma.median
    else:
        raise Exception('Invalid mode to combine prfs.')

    data_internal = np.ma.array(data=data, mask=mask)
    prf_model = np.ndarray(shape=(subsampling, subsampling, size, size))
    positions_subpixel_indices = np.array([subpixel_indices(_, subsampling)
                                           for _ in positions], dtype=np.int)

    for i in range(subsampling):
        for j in range(subsampling):
            extracted_sub_prfs = []
            sub_prf_indices = np.all(positions_subpixel_indices == [j, i],
                                     axis=1)
            positions_sub_prfs = positions[sub_prf_indices]
            for k, position in enumerate(positions_sub_prfs):
                x, y = position
                extracted_prf = extract_array(data_internal, (size, size),
                                              (y, x))
                # Check shape to exclude incomplete PRFs at the boundaries
                # of the image
                if (extracted_prf.shape == (size, size) and
                        np.ma.sum(extracted_prf) != 0):
                    # Replace NaN values by mirrored value, with respect
                    # to the prf's center
                    if fix_nan:
                        prf_nan = extracted_prf.mask
                        if prf_nan.any():
                            if (prf_nan.sum() > 3 or
                                    prf_nan[size // 2, size // 2]):
                                continue
                            else:
                                extracted_prf = mask_to_mirrored_num(
                                    extracted_prf, prf_nan,
                                    (size // 2, size // 2))
                    # Normalize and add extracted PRF to data cube
                    if fluxes is None:
                        extracted_prf_norm = (np.ma.copy(extracted_prf) /
                                              np.ma.sum(extracted_prf))
                    else:
                        fluxes_sub_prfs = fluxes[sub_prf_indices]
                        extracted_prf_norm = (np.ma.copy(extracted_prf) /
                                              fluxes_sub_prfs[k])
                    extracted_sub_prfs.append(extracted_prf_norm)
                else:
                    continue
            prf_model[i, j] = np.ma.getdata(
                combine(np.ma.dstack(extracted_sub_prfs), axis=2))
    return DiscretePRF(prf_model, subsampling=subsampling)


def subtract_psf(data, psf, positions, fluxes, mask=None):
    """
    Removes PSF/PRF at the given positions.

    To calculate residual images the PSF/PRF model is subtracted from the data
    at the given positions.

    Parameters
    ----------
    data : ndarray
        Image data.
    psf : `photutils.psf.DiscretePRF` or `photutils.psf.GaussianPSF`
        PSF/PRF model to be substracted from the data.
    positions : ndarray
        List of center positions where PSF/PRF is removed.
    fluxes : ndarray
        List of fluxes of the sources, for correct
        normalization.
    """
    # Set up indices
    indices = np.indices(data.shape)
    data_ = data.copy()
    # Loop over position
    for i, position in enumerate(positions):
        x_0, y_0 = position
        y = extract_array(indices[0], psf.shape, (y_0, x_0))
        x = extract_array(indices[1], psf.shape, (y_0, x_0))
        psf.amplitude.value = fluxes[i]
        psf.x_0.value, psf.y_0.value = x_0, y_0
        psf_image = psf(x, y)
        data_ = add_array(data_, -psf_image, (y_0, x_0))
    return data_
