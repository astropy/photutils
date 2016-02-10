# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for performing PSF fitting photometry on 2D arrays."""

from __future__ import division
import warnings

import numpy as np

from .utils import mask_to_mirrored_num

from astropy.table import Table, Column
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Parameter, Fittable2DModel
from astropy.utils.exceptions import AstropyUserWarning
from astropy.nddata.utils import extract_array, add_array, subpixel_indices


__all__ = ['DiscretePRF', 'create_prf', 'psf_photometry',
           'IntegratedGaussianPSF', 'subtract_psf']


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
        raise NotImplementedError("DiscretePRF is not yet compatible with psf module changes." )

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


class IntegratedGaussianPSF(Fittable2DModel):
    r"""
    Circular Gaussian model integrated over pixels.

    This model is a Gaussian *integrated* over an area of ``1`` (in units
    of the model input coordinates).  This is in contrast to the apparently
    similar `astropy.modeling.functional_models.Gaussian2D`, which is the value
    of a 2D Gaussian *at* the input coordinates, with no integration.  So this
    model is equivalent to assuming the sub-pixel PSF is Gaussian.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian PSF.
    flux : float (default 1)
        Total integrated flux over the entire PSF
    x_0 : float (default 0)
        Position of the peak in x direction.
    y_0 : float (default 0)
        Position of the peak in y direction.

    Notes
    -----
    The PSF model is evaluated according to the following formula:

        .. math::

            f(x, y) =
                \frac{F}{4}
                \left[
                {\rm erf} \left(\frac{x - x_0 + 0.5}
                {\sqrt{2} \sigma} \right) -
                {\rm erf} \left(\frac{x - x_0 - 0.5}
                {\sqrt{2} \sigma} \right)
                \right]
                \left[
                {\rm erf} \left(\frac{y - y_0 + 0.5}
                {\sqrt{2} \sigma} \right) -
                {\rm erf} \left(\frac{y - y_0 - 0.5}
                {\sqrt{2} \sigma} \right)
                \right]

    where ``erf`` denotes the error function and ``F`` the total
    integrated flux.
    """

    flux = Parameter(default=1)
    x_0 = Parameter(default=0, fixed=True)
    y_0 = Parameter(default=0, fixed=True)
    sigma = Parameter(default=1, fixed=True)

    _erf = None
    fit_deriv = None

    @property
    def bounding_box(self):
        halfwidth = 4 * self.sigma
        return ((int(self.y_0 - halfwidth), int(self.y_0 + halfwidth)),
                (int(self.x_0 - halfwidth), int(self.x_0 + halfwidth)))

    def __init__(self, sigma=sigma.default,
                 x_0=x_0.default, y_0=y_0.default, flux=flux.default,
                 **kwargs):
        if self._erf is None:
            from scipy.special import erf
            self.__class__._erf = erf

        super(IntegratedGaussianPSF, self).__init__(n_models=1, sigma=sigma,
                                                    x_0=x_0, y_0=y_0,
                                                    flux=flux, **kwargs)

    def evaluate(self, x, y, flux, x_0, y_0, sigma):
        """
        Model function Gaussian PSF model.
        """
        return (flux / 4 *
                ((self._erf((x - x_0 + 0.5) / (np.sqrt(2) * sigma)) -
                  self._erf((x - x_0 - 0.5) / (np.sqrt(2) * sigma))) *
                 (self._erf((y - y_0 + 0.5) / (np.sqrt(2) * sigma)) -
                  self._erf((y - y_0 - 0.5) / (np.sqrt(2) * sigma)))))


def _extract_psf_fitting_names(psf):
    """
    Determine the names of the x coordinate, y coordinate, and flux from a
    model.  Returns (xname, yname, fluxname)
    """
    if hasattr(psf, 'psf_xname'):
        xname = psf.psf_xname
    elif 'x_0' in psf.param_names:
        xname = 'x_0'
    else:
        raise ValueError('Could not determine x coordinate name for psf_photometry.')

    if hasattr(psf, 'psf_yname'):
        yname = psf.psf_yname
    elif 'y_0' in psf.param_names:
        yname = 'y_0'
    else:
        raise ValueError('Could not determine y coordinate name for psf_photometry.')

    if hasattr(psf, 'psf_fluxname'):
        fluxname = psf.psf_fluxname
    elif 'flux' in psf.param_names:
        fluxname = 'flux'
    else:
        raise ValueError('Could not determine flux name for psf_photometry.')

    return xname, yname, fluxname


def psf_photometry(data, positions, psf, fitter=LevMarLSQFitter(),
                   mask=None, mode='sequential'):
    """
    Perform PSF/PRF photometry on the data.

    Given a PSF or PRF model, the model is fitted simultaneously or
    sequentially to the given positions to obtain an estimate of the
    flux. If required, coordinates are also tuned to match best the data.

    If the data contains NaN values or the PSF/PRF is not completely
    contained in the image, a flux of zero is returned.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or array (must be 2D)
        Image data.
    positions : Array-like of shape (2, N) or `~astropy.table.Table`
        Positions at which to *start* the fit, in pixel coordinates.  If a
        table, the columns 'x_0' and 'y_0' must be present.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit to the data. Examples for such models are
        `photutils.psf.DiscretePRF` or `photutils.psf.GaussianPSF`.
    fitter : an astropy fitter
        This could be a `astropy.modeling.fitting.Fitter` instance.
    mask : ndarray, optional
        Mask to be applied to the data.
    mode : {'sequential'}
        One of the following modes to do PSF/PRF photometry:
            * 'sequential' (default)
                Fit PSF/PRF separately for the given positions.
            * (No other modes are yet implemented)

    Returns
    -------
    result_tab : `~astropy.table.Table`
        The results of the fitting procedure.  The fitted flux is in the column
        'flux_fit', and the centroids are in 'x_fit' and 'y_fit'. If `positions`
        was a table, any columns in that table will be carried over to this
        table.

    Examples
    --------
    See `Spitzer PSF Photometry <http://nbviewer.ipython.org/gist/adonath/
    6550989/PSFPhotometrySpitzer.ipynb>`_ for a short tutorial.
    """
    # accept nddata.  The isinstance is necessary because arrays have a `data`
    # attribute but it's not an array like in an NDData
    # we also can't check its type directly because on py2 it's a buffer but
    # in py3 it's a memoryview
    if hasattr(data, 'data') and hasattr(data.data, 'dtype'):
        fluxunit = data.unit
        data = data.data
    else:
        fluxunit = getattr(data, 'unit', None)

    # Check input array type and dimension.
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. Only 2-d arrays can be '
                         'passed to psf_photometry.'.format(data.ndim))

    # determine the names of the model's relevant attributes
    xname, yname, fluxname = _extract_psf_fitting_names(psf)

    # Prep the index arrays and the table for output
    indices = np.indices(data.shape)
    if hasattr(positions, 'colnames'):
        # quacks like a table, so assume it's a table
        if 'x_0' not in positions.colnames:
            raise ValueError('Input table does not have x0 column')
        if 'y_0' not in positions.colnames:
            raise ValueError('Input table does not have y0 column')
        result_tab = positions.copy()
    else:
        positions = np.array(positions, copy=False)
        result_tab = Table()
        result_tab['x_0'] = positions[0]
        result_tab['y_0'] = positions[1]

    result_tab['x_fit'] = result_tab['x_0']
    result_tab['y_fit'] = result_tab['y_0']
    result_tab.add_column(Column(name='flux_fit', unit=fluxunit,
                                 data=np.empty(len(result_tab), dtype=data.dtype)))

    # prep for fitting
    psf = psf.copy()  # don't want to muck up whatever PSF the user gives us
    setflux = 'flux_0' in result_tab.colnames

    if mode == 'sequential':
        for row in result_tab:
            setattr(psf, xname, row['x_0'])
            setattr(psf, yname, row['y_0'])
            if setflux:
                setattr(psf, fluxname, row['flux_0'])

            fitted = fitter(psf, indices[0], indices[1], data)
            row['x_fit'] = getattr(fitted, xname).value
            row['y_fit'] = getattr(fitted, yname).value
            row['flux_fit'] = getattr(fitted, fluxname).value

        # Set position
        #position = (self.y_0.value, self.x_0.value)

        # Extract sub array with data of interest
        #sub_array_data = extract_array(data, self.shape, position)

        # Fit only if PSF is completely contained in the image and no NaN
        # values are present
        #if (sub_array_data.shape == self.shape and
        #        not np.isnan(sub_array_data).any()):
        #    y = extract_array(indices[0], self.shape, position)
        #    x = extract_array(indices[1], self.shape, position)
        #    m = self.fitter(self, x, y, sub_array_data)
        #    return m.amplitude.value
        #else:
        #    return 0


    else:
        raise ValueError('Invalid photometry mode.')

    return result_tab, fitted


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


def subtract_psf(data, psf, posflux, subshape=None):
    """
    Subtracts PSF/PRFs from an image.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or array (must be 2D)
        Image data.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF/PRF model to be substracted from the data.
    posflux : Array-like of shape (3, N) or `~astropy.table.Table`
        Positions and fluxes for the objects to subtract.  If an array, it is
        interpreted as ``(x, y, flux)``  If a table, the columns 'x_fit',
        'y_fit', and 'flux_fit' must be present.
    subshape : 2-tuple or None
        The shape of the region around the center of the location to subtract
        the PSF from.  If None, subtract from the whole image.

    Returns
    -------
    subdata : same as `data`
        The image with the PSF subtracted
    """
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. Only 2-d arrays can be '
                         'passed to subtract_psf.'.format(data.ndim))

    #  translate array input into table
    if hasattr(posflux, 'colnames'):
        if 'x_fit' not in posflux.colnames:
            raise ValueError('Input table does not have x_fit')
        if 'y_fit' not in posflux.colnames:
            raise ValueError('Input table does not have y_fit')
        if 'flux_fit' not in posflux.colnames:
            raise ValueError('Input table does not have flux_fit')
    else:
        posflux = Table(names=['x_fit', 'y_fit', 'flux_fit'], data=posflux)

    # Set up contstants across the loop
    psf = psf.copy()
    xname, yname, fluxname = _extract_psf_fitting_names(psf)
    indices = np.indices(data.shape)
    subbeddata = data.copy()

    if subshape is None:
        indicies_reversed = indices[::-1]

        for row in posflux:
            getattr(psf, xname).value = row['x_fit']
            getattr(psf, yname).value = row['y_fit']
            getattr(psf, fluxname).value = row['flux_fit']

            subbeddata -= psf(*indicies_reversed)
    else:
        for row in posflux:
            x_0, y_0 = row['x_fit'], row['y_fit']

            y = extract_array(indices[0], subshape, (y_0, x_0))
            x = extract_array(indices[1], subshape, (y_0, x_0))

            getattr(psf, xname).value = x_0
            getattr(psf, yname).value = y_0
            getattr(psf, fluxname).value = row['flux_fit']

            subbeddata = add_array(subbeddata, -psf(x, y), (y_0, x_0))

    return subbeddata
