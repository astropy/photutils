# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for performing PSF fitting photometry on 2-D arrays."""

from __future__ import division

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.core import Parametric2DModel
from astropy.modeling.parameters import Parameter
from .arrayutils import (extract_array_2D, subpixel_indices, 
                        add_array_2D, fix_prf_nan)


__all__ = ['DiscretePRF', 'create_prf', 'psf_photometry', 'GaussianPSF', 'remove_prf']


class DiscretePRF(Parametric2DModel):
    """
    A discrete PRF model.
    
    The discrete PRF model stores images of the PRF at different subpixel
    positions or offsets as a lookup table. The resolution is given by the 
    subsampling parameter, which states in how many subpixels a pixel is divided.
    
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
        super(DiscretePRF, self).__init__(param_dim=1, x_0=x_0, y_0=y_0, 
                                    amplitude=amplitude, **constraints)
        self.linear = True
        self.fitter = fitting.NonLinearLSQFitter()
        
        # Fix position per default
        self.x_0.fixed = True
        self.y_0.fixed = True
    
    @property
    def shape(self):
        """
        Shape of the PRF image.
        """
        return self._prf_array.shape[-2:]
        
    def eval(self, x, y, amplitude, x_0, y_0):
        """
        Discrete PRF model evaluation.
        
        Given a certain position and amplitude the corresponding image of the PSF 
        is chosen and scaled to the amplitude. If x and y are outside the 
        boundaries of the image, zero will be returned.
        
        Parameters
        ----------
        x : float
            x in pixel coordinates.
        y : float 
            y in pixel coordinates.
        amplitude : float
            Model amplitude. 
        x_0 : float
            x position of the center.
        y_0 : float
            y position of the center.
        """
        # Convert x and y to index arrays
        x = (x - int(x_0)).astype('int') + self.shape[1] // 2
        y = (y - int(y_0)).astype('int') + self.shape[0] // 2
        
        # Get subpixel indices
        y_sub, x_sub = subpixel_indices((x_0, y_0), self.subsampling)
        
        # Out of boundary masks
        x_bound = np.logical_or(x < 0, x >= self.shape[1])
        y_bound = np.logical_or(y < 0, y >= self.shape[0])
        out_of_bounds = np.logical_or(x_bound, y_bound)
        
        # Set out of boundary indices to zero
        x[x_bound] = 0
        y[y_bound] = 0
        result = amplitude * self._prf_array[y_sub, x_sub][y, x]
        
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
        
        The fit is performed on a slice of the data with the same size as the PRF.
         
        Parameters
        ----------
        data : ndarray
            Array containig image data.
        indices : ndarray
            Array with indices of the data. As
            returned by np.indices(data.shape)
        """
        # Extract sub array of the data of the size of the PRF grid
        sub_array_data = extract_array_2D(data, self.shape, 
                                          (self.x_0.value, self.y_0.value))
        
        # Fit only if PSF is completely contained in the image and no NaN values
        # are present
        if sub_array_data.shape == self.shape and not np.isnan(sub_array_data).any():
            y = extract_array_2D(indices[0], self.shape, 
                                 (self.x_0.value, self.y_0.value))
            x = extract_array_2D(indices[1], self.shape, 
                                 (self.x_0.value, self.y_0.value))
            m = self.fitter(self, x, y, sub_array_data)
            return m.amplitude.value
        else:
            return 0
    

class GaussianPSF(Parametric2DModel):
    """
    Symmetrical Gaussian PSF model.
    
    The PSF is evaluated by using the 'scipy.special.erf` function 
    on a fixed grid of the size of 1 pixel to assure flux conservation
    on subpixel scale.
    
    Parameters
    ----------
    amplitude : float (default 1)
        Amplitude at the peak value.
    x_0 : float (default 0)
        Position of the peak in x direction.
    y_0 : float (default 0)
        Position of the peak in y direction.
    sigma : float
        Width of the Gaussian PSF.
        
    Notes
    -----
    The PSF model is evaluated according to the following formula:
    
        .. math:: 
            
            f(x, y) =
                \\frac{A}{4} 
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
    
    Where `erf` denotes the error function.  
    
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')
    sigma = Parameter('sigma')

    _erf = None

    def __init__(self, sigma):
        if self._erf is None:
            try:
                from scipy.special import erf
                self.__class__._erf = erf
            except (ValueError, ImportError):
                raise ImportError("Gaussian PSF model requires scipy.")
        x_0 = 0
        y_0 = 0
        amplitude = 1
        constraints = {'fixed': {'x_0': True, 'y_0': True, 'sigma': True}}
        super(GaussianPSF, self).__init__(param_dim=1, sigma=sigma, x_0=x_0, y_0=y_0, 
                                          amplitude=amplitude, **constraints)
        
        # Default size is 8 * sigma
        self.shape = (int(8 * sigma) + 1, int(8 * sigma) + 1)
        self.fitter = fitting.NonLinearLSQFitter()
        
        # Fix position per default
        self.x_0.fixed = True
        self.y_0.fixed = True
                
    def eval(self, x, y, amplitude, x_0, y_0, sigma):
        """
        Model function Gaussian PSF model.
        """
        return amplitude / 4 * ((self._erf((x - x_0 + 0.5) / (np.sqrt(2) * sigma)) 
                            - self._erf((x - x_0 - 0.5) / (np.sqrt(2) * sigma))) 
                            * (self._erf((y - y_0 + 0.5) / (np.sqrt(2) * sigma)) 
                            - self._erf((y - y_0 - 0.5) / (np.sqrt(2) * sigma))))
        
    def fit(self, data, indices):
        """
        Fit PSF/PRF to data.
        
        Fits the PSF/PRF to the data and returns the best fitting flux. 
        If the data contains NaN values or if the source is not completely 
        contained in the image data the fitting is omitted and a flux of 0 
        is returned.  
        
        For reasons of performance, indices for the data have to be created 
        outside and passed to the function. 
        
        The fit is performed on a slice of the data with the same size as the PRF.
         
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
        position = (self.x_0.value, self.y_0.value)
        
        # Extract sub array with data of interest
        sub_array_data = extract_array_2D(data, self.shape, position)
        
        # Fit only if PSF is completely contained in the image and no NaN values
        # are present
        if sub_array_data.shape == self.shape and not np.isnan(sub_array_data).any():
            y = extract_array_2D(indices[0], self.shape, position)
            x = extract_array_2D(indices[1], self.shape, position)
            m = self.fitter(self, x, y, sub_array_data)
            return m.amplitude.value
        else:
            return 0
    

def psf_photometry(data, positions, prf, mask=None, mode='sequential', 
                   tune_coordinates=False):
    """
    Perform PSF photometry on the data.
    
    Given a PSF or PRF model, the model is fitted simultaneously or sequentially
    to the given positions to obtain an estimate of the flux. If required, coordinates 
    are also tuned to match best the data. 
    
    If the data contains NaN values or the PSF/PRF is not completely contained in
    the image, a flux of zero is returned.
    
    
    Parameters
    ----------
    data : ndarray
        Image data array
    positions : List or array
        List of positions in pixel coordinates
        where to fit the PSF.
    prf : `photutils.psf.DiscretePRF` or `photutils.psf.GaussianPSF` 
        PSF model to fit to the data.
    mask : ndarray, optional
        Mask to be applied to the data.
    mode : {'sequential', 'simultaneous'}
         One of the following modes to do PSF photometry:
            * 'simultaneous' 
                Fit PSF simultaneous to all given positions.
            * 'sequential' (default)
                Fit PSF one after another to the given positions .
    
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
        prf.fixed['x_0'] = False
        prf.fixed['y_0'] = False
    
    # Actual photometry
    result = np.array([])
    indices = np.indices(data.shape)
    
    if mode == 'simultaneous':
        raise NotImplementedError('Simultaneous mode not implemented')
    elif mode == 'sequential':
        for i, position in enumerate(positions):
                prf.x_0, prf.y_0 = position
                flux = prf.fit(data, indices)
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
        List of spurce positions in pixel coordinates
        where to create the PRFs from.
    size : odd int
        Size of the quadratic PRF image in pixels.
    mask : bool array, optional 
        Boolean array to mask out bad values.
    fluxes : array, optional
        Object fluxes to normalize extracted prfs. 
    mode : string
        One of the following modes to combine
        the extracted PRFs:
            * 'mean' (default)
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
    In contrast to the Point Spread Function (PSF) the Point Response Function (PRF)
    is a map or an image of a point source after discretization e.g. with a CCD.
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
    
    # Setup data structure for extracted PRFs
    extracted_prfs = np.ndarray(shape=(len(positions), 
                                       subsampling, subsampling, size, size))
    
    # Setup data structure for extracted masks
    extracted_masks = np.ndarray(shape=(len(positions), 
                                        subsampling, subsampling, size, size))
    extracted_masks.fill(False)
        
    
    if mask is None:
        # Extract PRFs at given pixel positions
        for i, position in enumerate(positions):
            extracted_prf = extract_array_2D(data, (size, size), position)
            
            # Check shape to exclude incomplete PRFs at the boundaries of the image
            if extracted_prf.shape == (size, size) and extracted_prf.sum() != 0:
                # Replace NaN values by mirrored value, with respect 
                # to the prf's center
                if fix_nan:
                    prf_nan = np.isnan(extracted_prf)
                    if prf_nan.any():
                        if prf_nan.sum() > 3 or prf_nan[size / 2, size / 2]:
                            continue
                        else:
                            extracted_prf = fix_prf_nan(extracted_prf, prf_nan)
                # Normalize and add extracted PRF to data cube
                if fluxes is None:
                    extracted_prf_copy = extracted_prf.copy() / extracted_prf.sum()
                else:
                    extracted_prf_copy = extracted_prf.copy() / fluxes[i]
                    
                y_sub, x_sub = subpixel_indices(position, subsampling)
                extracted_prfs[i, y_sub, x_sub] = extracted_prf_copy
                
                # Mark values as set otherwise the later calculation of the mean
                # will go wrong
                extracted_masks[i, y_sub, x_sub]= True
            else:
                continue
        
    else:
        for i, position in enumerate(positions):
            extracted_prf = extract_array_2D(data, (size, size), position)
            extracted_mask = extract_array_2D(mask, (size, size), position)
            # Check shape to exclude incomplete PRFs at the boundaries of the image
            if extracted_prf.shape == (size, size) and extracted_prf.sum() != 0:
                # Normalize and add extracted PRF to data cube
                if fluxes is None:
                    extracted_prf_copy = extracted_prf.copy() / extracted_prf.sum()
                else:
                    extracted_prf_copy = extracted_prf.copy() / fluxes[i]
                y_sub, x_sub = subpixel_indices(position, subsampling)
                extracted_prfs[i, y_sub, x_sub] = extracted_prf_copy
                extracted_masks[i, y_sub, x_sub]= extracted_mask
            else:
                continue
            
    # Choose combination mode
    if mode == 'mean':
        prf = np.nansum(extracted_prfs, axis=0) / np.sum(extracted_masks, axis=0)
    elif mode == 'median':
        prf = np.median(extracted_prfs, axis=0)
    else:
        raise Exception('Invalid mode to combine prfs.') 
    return DiscretePRF(prf, subsampling=subsampling)


def remove_prf(data, prf, positions, fluxes, mask=None):
    """
    Removes PSF/PRF at the given positions.
    
    To calculate residual images the PSF/PRF model is subtracted from the data
    at the given positions. 
    
    Parameters
    ----------
    data : ndarray
        Image data.
    prf : `photutils.psf.DiscretePRF` or `photutils.psf.GaussianPSF`
        PSF/PRF model to be substracted from the data.
    positions : ndarray
        List of center positions where PRF is removed.
    fluxes : ndarray
        List of fluxes of the sources, for correct 
        normalization.
    
    """
    # Set up indices
    indices = np.indices(data.shape)
    
    # Loop over position
    for i, position in enumerate(positions):
        y = extract_array_2D(indices[0], prf.shape, position)
        x = extract_array_2D(indices[1], prf.shape, position)
        prf_image = prf.eval(x, y, fluxes[i], position[0], position[1])
        data = add_array_2D(data, -prf_image, position)
    return data
