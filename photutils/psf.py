# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing PSF fitting photometry on 2-D arrays."""

from __future__ import division
import abc

import numpy as np

from astropy.nddata.convolution.core import Kernel2D
from astropy.modeling import fitting
from astropy.modeling.core import Parametric2DModel
from .photometryutils import extract_array_2D
from IPython import embed



class AnalyticalPSF(Kernel2D, Parametric2DModel):
    """
    An abstract base class for an analytical PSF.
    """
    __metaclass__ = abc.ABCMeta

    param_names = ['amplitude']
    def __init__(self, psf_model):
        Kernel2D.__init__(self, array=array)
        Parametric2DModel.__init__(self, {'amplitude': 1})
        self.normalize()


    def eval(self, x, y, amplitude):
        """
        Analytic PSF model evaluation.
        
        Parameters
        ----------
        x : float
            x in pixel coordinates.
        y : float 
            y in pixel coordinates.
        amplitude : float
            Model amplitude. 
        """
        return amplitude * self._array[y, x]
        

class DiscretePRF(Parametric2DModel):
    """
    Base class for discrete PSF models.
    
    Parameters
    ----------
    array : ndarray
        Array containing PSF image.
    """
    param_names = ['amplitude', 'x_0', 'y_0']
    
    def __init__(self, array, normalize=True, subsampling=5):
        self._prf_array = prf_array
        self.subsampling = subsampling
        Parametric2DModel.__init__(self, {'amplitude': 1, 'x_0': 0, 'y_0': 0})
        self.linear = True
        self.fitter = fitting.NonLinearLSQFitter(self)
    
    @property
    def shape(self):
        """
        """
        return self._prf_array.shape[2:5]
        
    def normalize(self):
        """
        """
        pass
        
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
        x_frac, x_int = np.modf(x - x_0 + self.shape[1] // 2)
        y_frac, y_int = np.modf(y - y_0 + self.shape[0] // 2)
        x_sub = (x_frac * self.subsampling).astype('int')
        y_sub = (y_frac * self.subsampling).astype('int')
        return amplitude * self._prf_array
  
    def fit(self, data):
        """
        Fit PSF to data.
        
        Parameters
        ----------
        data : ndarray
            Data subarray with the same size as PSF image.
        """
        sub_array_data = extract_array_2D(data, self.shape, (self.x_0, self.y_0))
        if sub_array_data.shape == self.shape and not np.isnan(sub_array_data).any():
            y, x = np.indices(self.shape)
            self.fitter(x, y, sub_array_data)
            return self.amplitude.value
        else:
            return 0
        
def subpixel_indices(position, subsampling):
    """
    """
    x_frac, _  = np.modf(position[0])
    y_frac, _ = np.modf(position[1])
    x_sub = (x_frac * subsampling).astype('int')
    y_sub = (y_frac * subsampling).astype('int')
    return y_sub, x_sub
    
    

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


class GaussianPSF():
    """
    Gaussian PSF model.
    """
    def __init__(self):
        raise NotImplementedError
        
    
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


def psf_photometry(data, positions, prf, mask=None, mode='sequential'):
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
    result = np.array([])
    # Actual photometry
    if mode == 'simultaneous':
        raise NotImplementedError('Simultaneous mode not implemented')
    elif mode == 'sequential':
        for i, position in enumerate(positions):
                prf.x_0, prf.y_0 = position
                flux = prf.fit(data)
                result = np.append(result, flux)
    else:
        raise Exception('Invalid photometry mode.')
    return result


def create_prf(data, positions, size, mask=None, fluxes=None, mode='mean', subsampling=1):
    """
    Estimate point spread function (PSF) from image data.
    
    Given a list of positions and grid size this function estimates an image of
    the PSF by extracting and combining the individual PSFs from the given 
    positions. Different combining modes are available.    
     
    NaN values are replaced by the mirrored value, with respect to the center 
    of the PSF. Furthermore it is possible to specify fluxes to have a correct
    normalization of the individual PSFs. Otherwise the flux is estimated from
    a quadratic aperture of the specified size. 
        
    Parameters
    ----------
    data : array
        Data array
    positions : List or array
        List of positions in pixel coordinates
        where to create the PRF from.
    size : odd int
        Size of the quadratic PRF grid in pixels.
    fluxes : array (optional)
        Object fluxes to normalize extracted prfs. 
    mode : string
        One of the following modes to combine
        the extracted PRFs:
            * 'mean' (default)
                Take the pixelwise mean of the extracted PRFs.
            * 'median'
                Take the pixelwise median of the extracted PRFs.
    
    Returns
    -------
    prf : DiscretePRF
        Discrete PRF model estimated from data.
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
    
    if mask is None:
        # Extract PRFs at given pixel positions
        for i, position in enumerate(positions):
            extracted_prf = extract_array_2D(data, (size, size), position)
            
            # Check shape to exclude incomplete PRFs at the boundaries of the image
            if extracted_prf.shape == (size, size) and extracted_prf.sum() != 0:
                # Normalize and add extracted PRF to data cube
                if fluxes is None:
                    extracted_prf_copy = np.copy(extracted_prf, extracted_prf.sum())
                else:
                    extracted_prf_copy = np.copy(extracted_prf, fluxes[i])
                y_sub, x_sub = subpixel_indices(position, subsampling)
                extracted_prfs[i, y_sub, x_sub] = extracted_prf_copy
            else:
                continue
        
    else:
        extracted_masks = np.ndarray(shape=(len(positions), subsampling, subsampling, size, size))
        for i, position in enumerate(positions):
            position[1] -= 1./subsampling
            position[0] -= 1.5/subsampling
            extracted_prf = extract_array_2D(data, (size, size), position)
            extracted_mask = extract_array_2D(mask, (size, size), position)
            
            # Check shape to exclude incomplete PRFs at the boundaries of the image
            if extracted_prf.shape == (size, size) and extracted_prf.sum() != 0:
                # Normalize and add extracted PRF to data cube
                if fluxes is None:
                    extracted_prf_copy = np.copy(extracted_prf, extracted_prf.sum())
                else:
                    extracted_prf_copy = np.copy(extracted_prf, fluxes[i])
                y_sub, x_sub = subpixel_indices(position, subsampling)
                extracted_prfs[i, y_sub, x_sub] = extracted_prf_copy
                extracted_masks[i, y_sub, x_sub]= extracted_mask
            else:
                continue
    # Choose combination mode
    if mode == 'mean':
        if mask is None:
            prf = np.mean(extracted_prfs, axis=0)
        else:
            prf = np.nansum(extracted_prfs, axis=0) / np.sum(extracted_masks, axis=0)
    elif mode == 'median':
        prf = np.median(extracted_prfs, axis=0)
    else:
        raise Exception('Invalid mode to combine prfs.') 
    return prf
        
        
        
class PSFFitter(fitting.Fitter):
    """
    PSF fitter.
    """
    def __init__(self):
        """
        """
        pass
    
    def errorfunc(self, fps, *args):
        self.fitpars = fps
        meas = args[-1]
        if self.weights is None:
            return np.ravel(self.model(*args[: -1]) - meas)
        else:
            return np.ravel(self.weights * (self.model(*args[: -1]) - meas))
    
    
    def errorfunc(self, fps, *args):
        """
        Error function.
        """
        self.fitpars = fps
        meas = args[-1]
        return np.ravel(self.model(*args[: -1]) - meas)
    
    def __call__(self):
        """
        """
        pass    
    
#            # Replace NaN values by mirrored value, with respect 
#            #to the psf's center
#            psf_nan = np.isnan(extracted_psf)
#            if psf_nan.any():
#                
#                # Allow at most 3 NaN values to prevent the unlikely case, 
#                # that the mirrored values are also NaN. 
#                if psf_nan.sum() > 3 or psf_nan[size / 2, size / 2]:
#                    continue
#                else:
#                    y_nan_coords, x_nan_coords = np.where(psf_nan==True)
#                    for y_nan, x_nan in zip(y_nan_coords, x_nan_coords):
#                        if not np.isnan(extracted_psf[-y_nan - 1, -x_nan - 1]):
#                            extracted_psf[y_nan, x_nan] = \
#                             extracted_psf[-y_nan - 1, -x_nan - 1]
#                        elif not np.isnan(extracted_psf[y_nan, -x_nan]):
#                            extracted_psf[y_nan, x_nan] = \
#                            extracted_psf[y_nan, -x_nan - 1]
#                        else:
#                            extracted_psf[y_nan, x_nan] = \
#                            extracted_psf[-y_nan - 1, x_nan]    
    
    
    
    
        
        
    