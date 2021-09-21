# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module stores work related to photutils.psf that is not quite ready
for prime-time (i.e., is not considered a stable public API), but is
included either for experimentation or as legacy code.
"""

from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import extract_array, subpixel_indices
from astropy.table import Table
import numpy as np

from ..segmentation._utils import mask_to_mirrored_value

__all__ = ['DiscretePRF', 'Reproject']

__doctest_requires__ = {('Reproject'): ['gwcs']}


class DiscretePRF(Fittable2DModel):
    """
    A discrete Pixel Response Function (PRF) model.

    The discrete PRF model stores images of the PRF at different
    subpixel positions or offsets as a lookup table. The resolution is
    given by the subsampling parameter, which states in how many
    subpixels a pixel is divided.

    In the typical case of wanting to create a PRF from an image with
    many point sources, use the `~DiscretePRF.create_from_image` method,
    rather than directly initializing this class.

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
        Normalize PRF images to unity.  Equivalent to saying there is
        *no* flux outside the bounds of the PRF images.
    subsampling : int, optional
        Factor of subsampling. Default = 1.

    Notes
    -----
    See :ref:`psf-terminology` for more details on the distinction
    between PSF and PRF as used in this module.
    """

    flux = Parameter('flux')
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
        flux = 1
        super().__init__(n_models=1, x_0=x_0, y_0=y_0, flux=flux,
                         **constraints)
        self.fitter = LevMarLSQFitter()

    @property
    def prf_shape(self):
        """Shape of the PRF image."""
        return self._prf_array.shape[-2:]

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Discrete PRF model evaluation.

        Given a certain position and flux the corresponding image of
        the PSF is chosen and scaled to the flux. If x and y are
        outside the boundaries of the image, zero will be returned.

        Parameters
        ----------
        x : float
            x coordinate array in pixel coordinates.

        y : float
            y coordinate array in pixel coordinates.

        flux : float
            Model flux.

        x_0 : float
            x position of the center of the PRF.

        y_0 : float
            y position of the center of the PRF.
        """
        # Convert x and y to index arrays
        x = (x - x_0 + 0.5 + self.prf_shape[1] // 2).astype('int')
        y = (y - y_0 + 0.5 + self.prf_shape[0] // 2).astype('int')

        # Get subpixel indices
        y_sub, x_sub = subpixel_indices((y_0, x_0), self.subsampling)

        # Out of boundary masks
        x_bound = np.logical_or(x < 0, x >= self.prf_shape[1])
        y_bound = np.logical_or(y < 0, y >= self.prf_shape[0])
        out_of_bounds = np.logical_or(x_bound, y_bound)

        # Set out of boundary indices to zero
        x[x_bound] = 0
        y[y_bound] = 0
        result = flux * self._prf_array[int(y_sub), int(x_sub)][y, x]

        # Set out of boundary values to zero
        result[out_of_bounds] = 0
        return result

    @classmethod
    def create_from_image(cls, imdata, positions, size, fluxes=None,
                          mask=None, mode='mean', subsampling=1,
                          fix_nan=False):
        """
        Create a discrete point response function (PRF) from image data.

        Given a list of positions and size this function estimates an
        image of the PRF by extracting and combining the individual PRFs
        from the given positions.

        NaN values are either ignored by passing a mask or can be
        replaced by the mirrored value with respect to the center of the
        PRF.

        Note that if fluxes are *not* specified explicitly, it will be
        flux estimated from an aperture of the same size as the PRF
        image. This does *not* account for aperture corrections so often
        will *not* be what you want for anything other than quick-look
        needs.

        Parameters
        ----------
        imdata : array
            Data array with the image to extract the PRF from

        positions : List or array or `~astropy.table.Table`
            List of pixel coordinate source positions to use in creating
            the PRF.  If this is a `~astropy.table.Table` it must have
            columns called ``x_0`` and ``y_0``.

        size : odd int
            Size of the quadratic PRF image in pixels.

        mask : bool array, optional
            Boolean array to mask out bad values.

        fluxes : array, optional
            Object fluxes to normalize extracted PRFs. If not given (or
            None), the flux is estimated from an aperture of the same
            size as the PRF image.

        mode : {'mean', 'median'}
            One of the following modes to combine the extracted PRFs:
                * 'mean':  Take the pixelwise mean of the extracted PRFs.
                * 'median':  Take the pixelwise median of the extracted PRFs.

        subsampling : int
            Factor of subsampling of the PRF (default = 1).

        fix_nan : bool
            Fix NaN values in the data by replacing it with the
            mirrored value. Assuming that the PRF is symmetrical.

        Returns
        -------
        prf : `photutils.psf.sandbox.DiscretePRF`
            Discrete PRF model estimated from data.
        """
        # Check input array type and dimension.
        if np.iscomplexobj(imdata):
            raise TypeError('Complex type not supported')
        if imdata.ndim != 2:
            raise ValueError(f'{imdata.ndim}-d array not supported. '
                             'Only 2-d arrays supported.')
        if size % 2 == 0:
            raise TypeError("Size must be odd.")

        if fluxes is not None and len(fluxes) != len(positions):
            raise TypeError('Position and flux arrays must be of equal '
                            'length.')

        if mask is None:
            mask = np.isnan(imdata)

        if isinstance(positions, (list, tuple)):
            positions = np.array(positions)

        if isinstance(positions, Table) or \
                (isinstance(positions, np.ndarray) and
                 positions.dtype.names is not None):
            # One can do clever things like
            # positions['x_0', 'y_0'].as_array().view((positions['x_0'].dtype,
            #                                          2))
            # but that requires positions['x_0'].dtype is
            # positions['y_0'].dtype.
            # Better do something simple to allow type promotion if required.
            pos = np.empty((len(positions), 2))
            pos[:, 0] = positions['x_0']
            pos[:, 1] = positions['y_0']
            positions = pos

        if isinstance(fluxes, (list, tuple)):
            fluxes = np.array(fluxes)

        if mode == 'mean':
            combine = np.ma.mean
        elif mode == 'median':
            combine = np.ma.median
        else:
            raise Exception('Invalid mode to combine prfs.')

        data_internal = np.ma.array(data=imdata, mask=mask)
        prf_model = np.ndarray(shape=(subsampling, subsampling, size, size))
        positions_subpixel_indices = \
            np.array([subpixel_indices(_, subsampling) for _ in positions],
                     dtype=int)

        for i in range(subsampling):
            for j in range(subsampling):
                extracted_sub_prfs = []
                sub_prf_indices = np.all(positions_subpixel_indices == [j, i],
                                         axis=1)
                if not sub_prf_indices.any():
                    raise ValueError('The source coordinates do not sample '
                                     'all sub-pixel positions. Reduce the '
                                     'value of the subsampling parameter.')

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
                                    extracted_prf = mask_to_mirrored_value(
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
        return cls(prf_model, subsampling=subsampling)


class Reproject:
    """
    Class to reproject pixel coordinates between unrectified and
    rectified images.

    Parameters
    ----------
    wcs_original, wcs_rectified : `~astropy.wcs.WCS` or `~gwcs.wcs.WCS`
        The WCS objects for the original (unrectified) and rectified
        images.

    origin : {0, 1}
        Whether to use 0- or 1-based pixel coordinates.
    """

    def __init__(self, wcs_original, wcs_rectified):
        self.wcs_original = wcs_original
        self.wcs_rectified = wcs_rectified

    @staticmethod
    def _reproject(wcs1, wcs2, x, y):
        """
        Perform the forward transformation of ``wcs1`` followed by the
        inverse transformation of ``wcs2``.

        Parameters
        ----------
        wcs1, wcs2 : WCS objects
            World coordinate system (WCS) transformations that
            support the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        x, y : float or array-like of float
            The input pixel coordinates.

        Returns
        -------
        x, y:  float or array-like of float
            The reprojected pixel coordinates.
        """
        try:
            skycoord = wcs1.pixel_to_world(x, y)
            return wcs2.world_to_pixel(skycoord)
        except AttributeError:
            raise ValueError('Input wcs objects do not support the shared '
                             'WCS interface.')

    def to_rectified(self, x, y):
        """
        Convert the input (x, y) positions from the original
        (unrectified) image to the rectified image.

        Parameters
        ----------
        x, y : float or array-like of float
            The zero-index pixel coordinates in the original
            (unrectified) image.

        Returns
        -------
        x, y:  float or array-like
            The zero-index pixel coordinates in the rectified image.
        """
        return self._reproject(self.wcs_original, self.wcs_rectified, x, y)

    def to_original(self, x, y):
        """
        Convert the input (x, y) positions from the rectified image to
        the original (unrectified) image.

        Parameters
        ----------
        x, y : float or array-like of float
            The zero-index pixel coordinates in the rectified image.

        Returns
        -------
        x, y:  float or array-like
            The zero-index pixel coordinates in the original
            (unrectified) image.
        """
        return self._reproject(self.wcs_rectified, self.wcs_original, x, y)
