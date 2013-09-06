# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for performing PSF fitting photometry on 2-D arrays."""

from __future__ import division

import numpy as np

from astropy.nddata.convolution.core import Kernel2D
from astropy.modeling import fitting
from astropy.modeling.core import Parametric2DModel
from .photometryutils import extract_array_2D
from IPython import embed


__all__ = ['DiscretePRF', 'create_prf', 'psf_photometry', 'GaussianPSF']



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
    subsampling : int
        Factor of subsampling.
    """
    param_names = ['amplitude', 'x_0', 'y_0']
    
    def __init__(self, prf_array, normalize=True, subsampling=5):

        # Array shape and dimension check
        if subsampling == 1:
            if prf_array.ndim == 2:
                prf_array = np.array([[prf_array]])
        if prf_array.ndim != 4:
            raise TypeError('Array must have 4 dimensions.')
        if prf_array.shape[:2] != (subsampling, subsampling):
            raise TypeError('Incompatible subsampling and array size')
       
        # Set PRF asttributes
        self._prf_array = prf_array
        self.subsampling = subsampling
        
        Parametric2DModel.__init__(self, {'amplitude': 1, 'x_0': 0, 'y_0': 0,
                                   'fixed': {'x_0': True, 'y_0': True}})
        self.linear = True
        self.fitter = fitting.NonLinearLSQFitter(self)
    
    @property
    def shape(self):
        """
        """
        return self._prf_array.shape[-2:]
        
    def eval(self, x, y, amplitude, x_0, y_0):
        """
        Discrete PRF model evaluation.
        
        Parameters
        ----------
        x : float
            x in pixel coordinates.
        y : float 
            y in pixel coordinates.
        amplitude : float
            Model amplitude. 
        """
        x = x.astype('int')
        y = y.astype('int')
        y_sub, x_sub = subpixel_indices(x_0, y_0, self.subsampling)
        return amplitude * self._prf_array[y_sub, x_sub][y, x]
        
    def fit(self, data, indices):
        """
        Fit PRF to data.
        
        Parameters
        ----------
        data : ndarray
            Data subarray with the same size as PSF image.
        """
        sub_array_data = extract_array_2D(data, self.shape, (self.x_0.value, self.y_0.value))
        if sub_array_data.shape == self.shape and not np.isnan(sub_array_data).any():
            y, x = np.indices(self.shape)
            self.fitter(x, y, sub_array_data)
            return self.amplitude.value
        else:
            return 0
        
def subpixel_indices(x, y, subsampling):
    """
    """
    x_frac, _ = np.modf(x)
    y_frac, _ = np.modf(y)
    x_sub = np.int(x_frac * subsampling)
    y_sub = np.int(y_frac * subsampling)
    return y_sub, x_sub
    

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
    
    Model Formula:
    
        .. math:: 
            
            f(x, y) =
                \\frac{A}{4} 
                \\left[
                \\textnormal{erf} \\left(\\frac{x - x_0 + 0.5}{\\sqrt{2} \\sigma} \\right) -
                \\textnormal{erf} \\left(\\frac{x - x_0 - 0.5}{\\sqrt{2} \\sigma} \\right)
                \\right]
                \\left[
                \\textnormal{erf} \\left(\\frac{y - y_0 + 0.5}{\\sqrt{2} \\sigma} \\right) -
                \\textnormal{erf} \\left(\\frac{y - y_0 - 0.5}{\\sqrt{2} \\sigma} \\right)
                \\right]    
    
    Where `erf` denotes the error function.  
    
    """
    param_names = ['amplitude', 'x_0', 'y_0', 'sigma']

    def __init__(self, sigma):
        try:
            from scipy.special import erf
            self.erf = erf
        except ImportError:
            raise Exception('Gaussian PSF model requires scipy.')
        Parametric2DModel.__init__(self, {'sigma': sigma, 'x_0': 0, 'y_0': 0,
                                          'amplitude': 1})
        self.shape = (int(8 * sigma) + 1, int(8 * sigma) + 1)
        self.fitter = fitting.NonLinearLSQFitter(self)
        
    def eval(self, x, y, amplitude, x_0, y_0, sigma):
        """
        Model function Gaussian PSF model.
        """
        return amplitude / 4 * ((self.erf((x - x_0 + 0.5) / (np.sqrt(2) * sigma)) 
                            - self.erf((x - x_0 - 0.5) / (np.sqrt(2) * sigma))) 
                            * (self.erf((y - y_0 + 0.5) / (np.sqrt(2) * sigma)) 
                            - self.erf((y - y_0 - 0.5) / (np.sqrt(2) * sigma))))
        
    def fit(self, data, indices):
        """
        Fit PSF to data.
        
        Tries to fit the PSF to the data and returns the best fitting flux. 
        If the data contains NaN values or if the source is not completely 
        contained in the image data the fitting is omitted and a flux of 0 
        is returned.  
        
        Indices have to be passed. 
         
        Parameters
        ----------
        data : ndarray
            Array containig image data.
        indices : ndarray
            Array with indices of the data.
        """
        sub_array_data = extract_array_2D(data, self.shape, (self.x_0.value, self.y_0.value))
        if sub_array_data.shape == self.shape and not np.isnan(sub_array_data).any():
            y = extract_array_2D(indices[0], self.shape, (self.x_0.value, self.y_0.value))
            x = extract_array_2D(indices[1], self.shape, (self.x_0.value, self.y_0.value))
            self.fitter(x, y, sub_array_data)
            return self.amplitude.value
        else:
            return 0
    
    
class LorentzianPSF():
    """
    Lorentzian PSF model.
    """
    def __init__(self):
        raise NotImplementedError


class MoffatPSF():
    """
    Moffat PSF model.
    """
    def __init__(self):
        raise NotImplementedError


class SpitzerPSF():
    """
    Spitzer PSF model.
    """
    def __init__(self):
        raise NotImplementedError


class FermiPSF():
    """
    Fermi PSF model.
    """
    def __init__(self):
        raise NotImplementedError


def psf_photometry(data, positions, prf, mask=None, mode='sequential', tune_coordinates=False):
    """
    Perform PSF photometry on the data.
    
    Parameters
    ----------
    data : array
        Data array
    positions : List or array
        List of positions in pixel coordinates
        where to fit the PSF.
    psf : photutils.psf instance
        PSF model to fit to the data.
    mode : string
         One of the following modes to do PSF photometry:
            * 'simultaneous' (default)
                Fit PSF simultaneous to all given positions.
            * 'sequential'
                Fit PSF one after another to the given positions .
    
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
    
    result = np.array([])
    # Actual photometry
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


def create_prf(data, positions, size, mask=None, fluxes=None, mode='mean', subsampling=1, fix_nan=False):
    """
    Estimate point response function (PRF) from image data.
    
    Given a list of positions and size this function estimates an image of
    the PRF by extracting and combining the individual PRFs from the given 
    positions. Different modes of combining are available.    
     
    NaN values are either ignored by passing a mask or can be replaced by 
    the mirrored value     with respect to the center of the PRF. 
    
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
    prf : DiscretePRF
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
        raise TypeError("Positions and fluxes must be of equal length.")
    
    # Setup data structure for extracted PRFs
    extracted_prfs = np.ndarray(shape=(len(positions), subsampling, subsampling, size, size))
    
    # Setup data structure for extracted masks
    extracted_masks = np.ndarray(shape=(len(positions), subsampling, subsampling, size, size))
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
                        # Allow at most 3 NaN values to prevent the unlikely case, 
                        # that the mirrored values are also NaN. 
                        if prf_nan.sum() > 3 or prf_nan[size / 2, size / 2]:
                            continue
                        else:
                            y_nan_coords, x_nan_coords = np.where(prf_nan==True)
                            for y_nan, x_nan in zip(y_nan_coords, x_nan_coords):
                                if not np.isnan(extracted_prf[-y_nan - 1, -x_nan - 1]):
                                    extracted_prf[y_nan, x_nan] = \
                                     extracted_prf[-y_nan - 1, -x_nan - 1]
                                elif not np.isnan(extracted_prf[y_nan, -x_nan]):
                                    extracted_prf[y_nan, x_nan] = \
                                    extracted_prf[y_nan, -x_nan - 1]
                                else:
                                    extracted_prf[y_nan, x_nan] = \
                                    extracted_prf[-y_nan - 1, x_nan]                    
                # Normalize and add extracted PRF to data cube
                if fluxes is None:
                    extracted_prf_copy = extracted_prf.copy() / extracted_prf.sum()
                else:
                    extracted_prf_copy = extracted_prf.copy() / fluxes[i]
                    
                y_sub, x_sub = subpixel_indices(position[0], position[1], subsampling)
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
                y_sub, x_sub = subpixel_indices(position[0], position[1], subsampling)
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
        
    