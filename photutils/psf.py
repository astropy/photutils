# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing PSF fitting photometry on 2-D arrays."""

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
        

class DiscretePSF(Kernel2D, Parametric2DModel):
    """
    Base class for discrete PSF models.
    
    Parameters
    ----------
    array : ndarray
        Array containing PSF image.
    """
    param_names = ['amplitude']
    
    def __init__(self, array, normalize=True):
        Kernel2D.__init__(self, array=array)
        Parametric2DModel.__init__(self, {'amplitude': 1})
        if normalize:
            self.normalize()
        self.linear = True
        self.fitter = fitting.NonLinearLSQFitter(self)
        
    def eval(self, x, y, amplitude):
        """
        Discrete PSF model evaluation.
        
        Parameters
        ----------
        x : float
            x in pixel coordinates.
        y : float 
            y in pixel coordinates.
        amplitude : float
            Model amplitude. 
        """
        x_int = np.rint(x).astype('int16')
        y_int = np.rint(y).astype('int16')
        return amplitude * self._array[y_int, x_int]
  
    def fit(self, data):
        """
        Fit PSF to data.
        
        Parameters
        ----------
        data : ndarray
            Data subarray with the same size as PSF image.
        """
        y, x = np.indices(self.shape)
        self.fitter(x, y, data)
        return self.amplitude


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


def psf_photometry(data, positions, psf, mode='sequential'):
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
    data = np.asarray(data)
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
            sub_array_data = extract_array_2D(data, psf.shape, position)
            if sub_array_data.shape == psf.shape:
                flux = psf.fit(sub_array_data)
                result = np.append(result, flux.value)
            else:
                continue
                 
    else:
        raise Exception('Invalid photometry mode.')
    return result


def create_psf(data, positions, size, fluxes=None, mode='mean'):
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
        where to create the PSF from.
    size : odd int
        Size of the quadratic PSF grid in pixels.
    fluxes : array (optional)
        Object fluxes to normalize extracted psfs. 
    mode : string
        One of the following modes to combine
        the extracted PSFs:
            * 'mean' (default)
                Take the pixelwise mean of the extracted PSFs.
            * 'median'
                Take the pixelwise median of the extracted PSFs.
    
    Returns
    -------
    psf : DiscretePSF
        Discrete PSF model estimated from data.
    """
    # Check input array type and dimension.
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))
    if size % 2 == 0:
        raise TypeError("Size must be odd.")
        
    if fluxes is not None and len(fluxes) != len(positions):
        raise TypeError("Positions and fluxes must be of equal length.")
    
    # Setup data cube for extracted PSFs
    extracted_psfs = np.ndarray(shape=(0, size, size))
    
    # Extract PSFs at given pixel positions
    for i, position in enumerate(positions):
        extracted_psf = extract_array_2D(data, (size, size), position)
        
        # Check shape to exclude incomplete psfs at the boundaries of the image
        if extracted_psf.shape == (size, size) and extracted_psf.sum() != 0:
            
            # Replace NaN values by mirrored value, with respect 
            #to the psf's center
            psf_nan = np.isnan(extracted_psf)
            if psf_nan.any():
                
                # Allow at most 3 NaN values to prevent the unlikely case, 
                # that the mirrored values are also NaN. 
                if psf_nan.sum() > 3 or psf_nan[size / 2, size / 2]:
                    continue
                else:
                    y_nan_coords, x_nan_coords = np.where(psf_nan==True)
                    for y_nan, x_nan in zip(y_nan_coords, x_nan_coords):
                        if not np.isnan(extracted_psf[-y_nan - 1, -x_nan - 1]):
                            extracted_psf[y_nan, x_nan] = \
                             extracted_psf[-y_nan - 1, -x_nan - 1]
                        elif not np.isnan(extracted_psf[y_nan, -x_nan]):
                            extracted_psf[y_nan, x_nan] = \
                            extracted_psf[y_nan, -x_nan - 1]
                        else:
                            extracted_psf[y_nan, x_nan] = \
                            extracted_psf[-y_nan - 1, x_nan]
            
            # Normalize and add extracted psf to data cube
            if fluxes is None:
                extracted_psf_copy = np.copy(extracted_psf, extracted_psf.sum())
            else:
                extracted_psf_copy = np.copy(extracted_psf, fluxes[i])
            extracted_psf_copy.shape = (1, size, size)
            extracted_psfs = np.append(extracted_psfs, extracted_psf_copy , axis=0)
        else:
            continue
    
    # Choose combination mode
    if mode == 'mean':
        psf = np.mean(extracted_psfs, axis=0)
    elif mode == 'median':
        psf = np.median(extracted_psfs, axis=0)
    else:
        raise Exception('Invalid mode to combine PSFs.') 
    return DiscretePSF(psf)
        
        
        
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
    
    
    
    
    
    
        
        
    