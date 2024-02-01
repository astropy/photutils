# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the GriddedPSFModel and related tools.
"""

import copy
import io
import itertools
import os
import warnings
from functools import lru_cache

import astropy
import numpy as np
from astropy.io import fits, registry
from astropy.io.fits.verify import VerifyWarning
from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import NDData, reshape_as_blocks
from astropy.utils import minversion
from astropy.visualization import simple_norm

from photutils.utils._parameters import as_pair

__all__ = ['GriddedPSFModel', 'ModelGridPlotMixin', 'stdpsf_reader',
           'webbpsf_reader', 'STDPSFGrid']
__doctest_skip__ = ['GriddedPSFModelRead', 'STDPSFGrid']


class ModelGridPlotMixin:
    """
    Mixin class to plot a grid of ePSF models.
    """

    def _reshape_grid(self, data):
        """
        Reshape the 3D ePSF grid as a 2D array of horizontally and
        vertically stacked ePSFs.
        """
        nypsfs = self._ygrid.shape[0]
        nxpsfs = self._xgrid.shape[0]
        ny, nx = self.data.shape[1:]
        data.shape = (nypsfs, nxpsfs, ny, nx)

        return data.transpose([0, 2, 1, 3]).reshape(nypsfs * ny, nxpsfs * nx)

    def plot_grid(self, *, ax=None, vmax_scale=None, peak_norm=False,
                  deltas=False, cmap=None, dividers=True,
                  divider_color='darkgray', divider_ls='-', figsize=None):
        """
        Plot the grid of ePSF models.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot.  If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        vmax_scale : float, optional
            Scale factor to apply to the image stretch limits. This
            value is multiplied by the peak ePSF value to determine the
            plotting ``vmax``. The defaults are 1.0 for plotting the
            ePSF data and 0.03 for plotting the ePSF difference data
            (``deltas=True``). If ``deltas=True``, the ``vmin`` is set
            to ``-vmax``. If ``deltas=False`` the ``vmin`` is set to
            ``vmax`` / 1e4.

        peak_norm : bool, optional
            Whether to normalize the ePSF data by the peak value. The
            default shows the ePSF flux per pixel.

        deltas : bool, optional
            Set to `True` to show the differences between each ePSF
            and the average ePSF.

        cmap : str or `matplotlib.colors.Colormap`, optional
            The colormap to use. The default is `None`, which uses
            the 'viridis' colormap for plotting ePSF data and the
            'gray_r' colormap for plotting the ePSF difference data
            (``deltas=True``).

        dividers : bool, optional
            Whether to show divider lines between the ePSFs.

        divider_color, divider_ls : str, optional
            Matplotlib color and linestyle options for the divider
            lines between ePSFs. These keywords have no effect unless
            ``show_dividers=True``.

        figsize : (float, float), optional
            The figure (width, height) in inches.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The matplotlib figure object. This will be the current
            figure if ``ax=None``. Use ``fig.savefig()`` to save the
            figure to a file.

            Note that when calling this method in a notebook, if you do
            not store the return value of this function, the figure will
            be displayed twice due to the REPL automatically displaying
            the return value of the last function call. Alternatively,
            you can append a semicolon to the end of the function call
            to suppress the display of the return value.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        data = self.data.copy()
        if deltas:
            # Compute mean ignoring any blank (all zeros) ePSFs.
            # This is the case for MIRI with its non-square FOV.
            mask = np.zeros(data.shape[0], dtype=bool)
            for i, arr in enumerate(data):
                if np.count_nonzero(arr) == 0:
                    mask[i] = True
            data -= np.mean(data[~mask], axis=0)
            data[mask] = 0.0

        data = self._reshape_grid(data)

        if ax is None:
            if figsize is None and self.meta.get('detector', '') == 'NRCSW':
                figsize = (20, 8)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()

        if peak_norm:  # normalize relative to peak
            if data.max() != 0:
                data /= data.max()

        if deltas:
            if cmap is None:
                cmap = cm.gray_r.copy()

            if vmax_scale is None:
                vmax_scale = 0.03
            vmax = data.max() * vmax_scale
            vmin = -vmax
            if minversion(astropy, '6.1.dev'):
                norm = simple_norm(data, 'linear', vmin=vmin, vmax=vmax)
            else:
                norm = simple_norm(data, 'linear', min_cut=vmin, max_cut=vmax)
        else:
            if cmap is None:
                cmap = cm.viridis.copy()

            if vmax_scale is None:
                vmax_scale = 1.0
            vmax = data.max() * vmax_scale
            vmin = vmax / 1.0e4
            if minversion(astropy, '6.1.dev'):
                norm = simple_norm(data, 'log', vmin=vmin, vmax=vmax,
                                   log_a=1.0e4)
            else:
                norm = simple_norm(data, 'log', min_cut=vmin, max_cut=vmax,
                                   log_a=1.0e4)

        # Set up the coordinate axes to later set tick labels based on
        # detector ePSF coordinates. This sets up axes to have, behind the
        # scenes, the ePSFs centered at integer coords 0, 1, 2, 3 etc.
        # extent = (left, right, bottom, top)
        nypsfs = self._ygrid.shape[0]
        nxpsfs = self._xgrid.shape[0]
        extent = [-0.5, nxpsfs - 0.5, -0.5, nypsfs - 0.5]

        ax.imshow(data, extent=extent, norm=norm, cmap=cmap, origin='lower')

        # Use the axes set up above to set appropriate tick labels
        xticklabels = self._xgrid.astype(int)
        yticklabels = self._ygrid.astype(int)
        if self.meta.get('detector', '') == 'NRCSW':
            xticklabels = list(xticklabels[0:5]) * 4
            yticklabels = list(yticklabels[0:5]) * 2
        ax.set_xticks(np.arange(nxpsfs))
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('ePSF location in detector X pixels')
        ax.set_yticks(np.arange(nypsfs))
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel('ePSF location in detector Y pixels')

        if dividers:
            for ix in range(nxpsfs):
                ax.axvline(ix + 0.5, color=divider_color, ls=divider_ls)
            for iy in range(nypsfs):
                ax.axhline(iy + 0.5, color=divider_color, ls=divider_ls)

        instrument = self.meta.get('instrument', '')
        if not instrument:
            # WebbPSF output
            instrument = self.meta.get('instrume', '')
        detector = self.meta.get('detector', '')
        filtername = self.meta.get('filter', '')

        # WebbPSF outputs a tuple with the comment in the second element
        if isinstance(instrument, (tuple, list, np.ndarray)):
            instrument = instrument[0]
        if isinstance(detector, (tuple, list, np.ndarray)):
            detector = detector[0]
        if isinstance(filtername, (tuple, list, np.ndarray)):
            filtername = filtername[0]

        title = f'{instrument} {detector} {filtername}'
        if title != '':
            # add extra space at end
            title += ' '

        if deltas:
            ax.set_title(f'{title}(ePSFs − <ePSF>)')
            if peak_norm:
                label = 'Difference relative to average ePSF peak'
            else:
                label = 'Difference relative to average ePSF values'
        else:
            ax.set_title(f'{title}ePSFs')
            if peak_norm:
                label = 'Scale relative to ePSF peak pixel'
            else:
                label = 'ePSF flux per pixel'

        cbar = plt.colorbar(label=label, mappable=ax.images[0])
        if not deltas:
            cbar.ax.set_yscale('log')

        if self.meta.get('detector', '') == 'NRCSW':
            # NIRCam NRCSW STDPSF files contain all detectors.
            # The plot gets extra divider lines and SCA name labels.
            nxpsfs = len(self._xgrid)
            nypsfs = len(self._ygrid)
            plt.axhline(nypsfs / 2 - 0.5, color='orange')
            for i in range(1, 4):
                ax.axvline(nxpsfs / 4 * i - 0.5, color='orange')

            det_labels = [['A1', 'A3', 'B4', 'B2'], ['A2', 'A4', 'B3', 'B1']]
            for i in range(2):
                for j in range(4):
                    ax.text(j * nxpsfs / 4 - 0.45,
                            (i + 1) * nypsfs / 2 - 0.55,
                            det_labels[i][j], color='orange',
                            verticalalignment='top', fontsize=12)

        fig.tight_layout()

        return fig


class GriddedPSFModelRead(registry.UnifiedReadWrite):
    """
    Read and parse a FITS file into a `GriddedPSFModel` instance.

    This class enables the astropy unified I/O layer for
    `GriddedPSFModel`. This allows easily reading a file in different
    supported data formats using syntax such as::

      >>> from photutils.psf import GriddedPSFModel
      >>> psf_model = GriddedPSFModel.read('filename.fits', format=format)

    Get help on the available readers for `GriddedPSFModel` using the
    ``help()`` method::

      >>> # Get help reading Table and list supported formats
      >>> GriddedPSFModel.read.help()

      >>> # Get detailed help on the STSPSF FITS reader
      >>> GriddedPSFModel.read.help('stdpsf')

      >>> # Get detailed help on the WebbPSF FITS reader
      >>> GriddedPSFModel.read.help('webbpsf')

      >>> # Print list of available formats
      >>> GriddedPSFModel.read.list_formats()

    Parameters
    ----------
    *args : tuple, optional
        Positional arguments passed through to data reader. If supplied
        the first argument is typically the input filename.
    format : str
        File format specifier.
    **kwargs : dict, optional
        Keyword arguments passed through to data reader.

    Returns
    -------
    out : `~photutils.psf.GriddedPSFModel`
        A gridded ePSF model corresponding to FITS file contents.
    """
    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'read', registry=None)
        # uses default global registry

    def __call__(self, *args, **kwargs):
        return self.registry.read(self._cls, *args, **kwargs)


class GriddedPSFModel(ModelGridPlotMixin, Fittable2DModel):
    """
    A fittable 2D model containing a grid ePSF models.

    The ePSF models are defined at fiducial detector locations and are
    bilinearly interpolated to calculate an ePSF model at an arbitrary
    (x, y) detector position.

    When evaluating this model, it cannot be called with x and y arrays
    that have greater than 2 dimensions.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object containing the grid of
        reference ePSF arrays. The data attribute must contain a 3D
        `~numpy.ndarray` containing a stack of the 2D ePSFs with a shape
        of ``(N_psf, ePSF_ny, ePSF_nx)``.

        The meta attribute must be `dict` containing the following:

            * ``'grid_xypos'``: A list of the (x, y) grid positions of
              each reference ePSF. The order of positions should match the
              first axis of the 3D `~numpy.ndarray` of ePSFs. In other
              words, ``grid_xypos[i]`` should be the (x, y) position of
              the reference ePSF defined in ``data[i]``.

            * ``'oversampling'``: The integer oversampling factor(s) of
              the ePSF. If ``oversampling`` is a scalar then it will be
              used for both axes. If ``oversampling`` has two elements,
              they must be in ``(y, x)`` order.

        The meta attribute may contain other properties such as the
        telescope, instrument, detector, and filter of the ePSF.

    Methods
    -------
    read(\\*args, \\**kwargs)
        Class method to create a `GriddedPSFModel` instance from a
        STDPSF FITS file. This method uses :func:`stdpsf_reader` with
        the provided parameters.

    Notes
    -----
    Internally, the grid of ePSFs will be arranged and stored such that
    it is sorted first by y and then by x.
    """

    flux = Parameter(description='Intensity scaling factor for the ePSF '
                     'model.', default=1.0)
    x_0 = Parameter(description='x position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)
    y_0 = Parameter(description='y position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)

    read = registry.UnifiedReadWriteMethod(GriddedPSFModelRead)

    def __init__(self, nddata, *, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, fill_value=0.0):

        self._validate_data(nddata)
        self.data, self.grid_xypos = self._define_grid(nddata)
        # use _meta to avoid the meta descriptor
        self._meta = nddata.meta.copy()
        self.oversampling = as_pair('oversampling',
                                    nddata.meta['oversampling'],
                                    lower_bound=(0, 1))

        self.fill_value = fill_value

        self._grid_xpos, self._grid_ypos = np.transpose(self.grid_xypos)
        self._xgrid = np.unique(self._grid_xpos)  # also sorts values
        self._ygrid = np.unique(self._grid_ypos)  # also sorts values
        self.meta['grid_shape'] = (len(self._ygrid), len(self._xgrid))
        if (len(list(itertools.product(self._xgrid, self._ygrid)))
                != len(self.grid_xypos)):
            raise ValueError('"grid_xypos" must form a regular grid.')

        self._xidx = np.arange(self.data.shape[2], dtype=float)
        self._yidx = np.arange(self.data.shape[1], dtype=float)

        # Here we avoid decorating the instance method with @lru_cache
        # to prevent memory leaks; we set maxsize=128 to prevent the
        # cache from growing too large.
        self._calc_interpolator = lru_cache(maxsize=128)(
            self._calc_interpolator_uncached)

        super().__init__(flux, x_0, y_0)

    @staticmethod
    def _validate_data(data):
        if not isinstance(data, NDData):
            raise TypeError('data must be an NDData instance.')

        if data.data.ndim != 3:
            raise ValueError('The NDData data attribute must be a 3D numpy '
                             'ndarray')

        if 'grid_xypos' not in data.meta:
            raise ValueError('"grid_xypos" must be in the nddata meta '
                             'dictionary.')
        if len(data.meta['grid_xypos']) != data.data.shape[0]:
            raise ValueError('The length of grid_xypos must match the number '
                             'of input ePSFs.')

        if 'oversampling' not in data.meta:
            raise ValueError('"oversampling" must be in the nddata meta '
                             'dictionary.')

    def _define_grid(self, nddata):
        """
        Sort the input ePSF data into a regular grid where the ePSFs are
        sorted first by y and then by x.

        Parameters
        ----------
        nddata : `~astropy.nddata.NDData`
            The input NDData object containing the ePSF data.

        Returns
        -------
        data : 3D `~numpy.ndarray`
            The 3D array of ePSFs.
        grid_xypos : array of (x, y) pairs
            The (x, y) positions of the ePSFs, sorted first by y and
            then by x.
        """
        grid_xypos = np.array(nddata.meta['grid_xypos'])
        # sort by y and then by x
        idx = np.lexsort((grid_xypos[:, 0], grid_xypos[:, 1]))
        grid_xypos = grid_xypos[idx]
        data = nddata.data[idx]

        return data, grid_xypos

    def _cls_info(self):
        cls_info = []

        keys = ('STDPSF', 'instrument', 'detector', 'filter', 'grid_shape')
        for key in keys:
            if key in self.meta:
                name = key.capitalize() if key != 'STDPSF' else key
                cls_info.append((name, self.meta[key]))

        cls_info.extend([('Number of ePSFs', len(self.grid_xypos)),
                         ('ePSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', tuple(self.oversampling))])
        return cls_info

    def __str__(self):
        return self._format_str(keywords=self._cls_info())

    def copy(self):
        """
        Return a copy of this model where only the model parameters are
        copied.

        All other copied model attributes are references to the
        original model. This prevents copying the ePSF grid data, which
        may contain a large array.

        This method is useful if one is interested in only changing
        the model parameters in a model copy. It is used in the PSF
        photometry classes during model fitting.

        Use the `deepcopy` method if you want to copy all of the model
        attributes, including the ePSF grid data.
        """
        newcls = object.__new__(self.__class__)

        for key, val in self.__dict__.items():
            if key in self.param_names:  # copy only the parameter values
                newcls.__dict__[key] = copy.copy(val)
            else:
                newcls.__dict__[key] = val

        return newcls

    def deepcopy(self):
        """
        Return a deep copy of this model.
        """
        return copy.deepcopy(self)

    def clear_cache(self):
        """
        Clear the internal cache.
        """
        self._calc_interpolator.cache_clear()

    def _cache_info(self):
        """
        Return information about the internal cache.
        """
        return self._calc_interpolator.cache_info()

    @staticmethod
    def _find_start_idx(data, x):
        """
        Find the index of the lower bound where ``x`` should be inserted
        into ``a`` to maintain order.

        The index of the upper bound is the index of the lower bound
        plus 2.  Both bound indices must be within the array.

        Parameters
        ----------
        data : 1D `~numpy.ndarray`
            The 1D array to search.

        x : float
            The value to insert.

        Returns
        -------
        index : int
            The index of the lower bound.
        """
        idx = np.searchsorted(data, x)
        if idx == 0:
            idx0 = 0
        elif idx == len(data):  # pragma: no cover
            idx0 = idx - 2
        else:
            idx0 = idx - 1
        return idx0

    def _find_bounding_points(self, x, y):
        """
        Find the indices of the grid points that bound the input
        ``(x, y)`` position.

        Parameters
        ----------
        x, y : float
            The ``(x, y)`` position where the ePSF is to be evaluated.
            The position must be inside the region defined by the grid
            of ePSF positions.

        Returns
        -------
        indices : list of int
            A list of indices of the bounding grid points.
        """
        x0 = self._find_start_idx(self._xgrid, x)
        y0 = self._find_start_idx(self._ygrid, y)
        xypoints = list(itertools.product(self._xgrid[x0:x0 + 2],
                                          self._ygrid[y0:y0 + 2]))

        # find the grid_xypos indices of the reference xypoints
        indices = []
        for xx, yy in xypoints:
            indices.append(np.argsort(np.hypot(self._grid_xpos - xx,
                                               self._grid_ypos - yy))[0])

        return indices

    @staticmethod
    def _bilinear_interp(xyref, zref, xi, yi):
        """
        Perform bilinear interpolation of four 2D arrays located at
        points on a regular grid.

        Parameters
        ----------
        xyref : list of 4 (x, y) pairs
            A list of 4 ``(x, y)`` pairs that form a rectangle.

        zref : 3D `~numpy.ndarray`
            A 3D `~numpy.ndarray` of shape ``(4, nx, ny)``. The first
            axis corresponds to ``xyref``, i.e., ``refdata[0, :, :]`` is
            the 2D array located at ``xyref[0]``.

        xi, yi : float
            The ``(xi, yi)`` point at which to perform the
            interpolation.  The ``(xi, yi)`` point must lie within the
            rectangle defined by ``xyref``.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The 2D interpolated array.
        """
        xyref = [tuple(i) for i in xyref]
        idx = sorted(range(len(xyref)), key=xyref.__getitem__)
        xyref = sorted(xyref)  # sort by x, then y
        (x0, y0), (_x0, y1), (x1, _y0), (_x1, _y1) = xyref

        if x0 != _x0 or x1 != _x1 or y0 != _y0 or y1 != _y1:
            raise ValueError('The refxy points do not form a rectangle.')

        if not np.isscalar(xi):
            xi = xi[0]
        if not np.isscalar(yi):
            yi = yi[0]

        if not x0 <= xi <= x1 or not y0 <= yi <= y1:
            raise ValueError('The (x, y) input is not within the rectangle '
                             'defined by xyref.')

        data = np.asarray(zref)[idx]
        weights = np.array([(x1 - xi) * (y1 - yi), (x1 - xi) * (yi - y0),
                            (xi - x0) * (y1 - yi), (xi - x0) * (yi - y0)])
        norm = (x1 - x0) * (y1 - y0)

        return np.sum(data * weights[:, None, None], axis=0) / norm

    def _calc_interpolator_uncached(self, x_0, y_0):
        """
        Return the local interpolation function for the ePSF model at
        (x_0, y_0).

        Note that the interpolator will be cached by _calc_interpolator.
        It can be cleared by calling the clear_cache method.
        """
        from scipy.interpolate import RectBivariateSpline

        if (x_0 < self._xgrid[0] or x_0 > self._xgrid[-1]
                or y_0 < self._ygrid[0] or y_0 > self._ygrid[-1]):
            # position is outside of the grid, so simply use the
            # closest reference ePSF
            ref_index = np.argsort(np.hypot(self._grid_xpos - x_0,
                                            self._grid_ypos - y_0))[0]
            psf_image = self.data[ref_index, :, :]
        else:
            # find the four bounding reference ePSFs and interpolate
            ref_indices = self._find_bounding_points(x_0, y_0)
            xyref = self.grid_xypos[ref_indices]
            psfs = self.data[ref_indices, :, :]

            psf_image = self._bilinear_interp(xyref, psfs, x_0, y_0)

        interpolator = RectBivariateSpline(self._xidx, self._yidx,
                                           psf_image.T, kx=3, ky=3, s=0)

        return interpolator

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Evaluate the `GriddedPSFModel` for the input parameters.
        """
        if x.ndim > 2:
            raise ValueError('x and y must be 1D or 2D.')

        # NOTE: the astropy base Model.__call__() method converts scalar
        # inputs to size-1 arrays before calling evaluate().
        if not np.isscalar(flux):
            flux = flux[0]
        if not np.isscalar(x_0):
            x_0 = x_0[0]
        if not np.isscalar(y_0):
            y_0 = y_0[0]

        # Calculate the local interpolation function for the ePSF at
        # (x_0, y_0). Only the integer part of the position is input in
        # order to have effective caching.
        interpolator = self._calc_interpolator(int(x_0), int(y_0))

        # now evaluate the ePSF at the (x_0, y_0) subpixel position on
        # the input (x, y) values
        xi = self.oversampling[1] * (np.asarray(x, dtype=float) - x_0)
        yi = self.oversampling[0] * (np.asarray(y, dtype=float) - y_0)

        # define origin at the ePSF image center
        ny, nx = self.data.shape[1:]
        xi += (nx - 1) / 2
        yi += (ny - 1) / 2

        evaluated_model = flux * interpolator.ev(xi, yi)

        if self.fill_value is not None:
            # find indices of pixels that are outside the input pixel
            # grid and set these pixels to the fill_value
            invalid = (((xi < 0) | (xi > nx - 1))
                       | ((yi < 0) | (yi > ny - 1)))
            evaluated_model[invalid] = self.fill_value

        return evaluated_model


def _read_stdpsf(filename):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        with fits.open(filename, ignore_missing_end=True) as hdulist:
            header = hdulist[0].header
            data = hdulist[0].data

    try:
        npsfs = header['NAXIS3']
        nxpsfs = header['NXPSFS']
        nypsfs = header['NYPSFS']
    except KeyError as exc:
        raise ValueError('Invalid STDPDF FITS file.') from exc

    if 'IPSFX01' in header:
        xgrid = [header[f'IPSFX{i:02d}'] for i in range(1, nxpsfs + 1)]
        ygrid = [header[f'JPSFY{i:02d}'] for i in range(1, nypsfs + 1)]
    elif 'IPSFXA5' in header:
        xgrid = []
        ygrid = []
        xkeys = ('IPSFXA5', 'IPSFXB5', 'IPSFXC5', 'IPSFXD5')
        for xkey in xkeys:
            xgrid.extend([int(n) for n in header[xkey].split()])
        ykeys = ('JPSFYA5', 'JPSFYB5')
        for ykey in ykeys:
            ygrid.extend([int(n) for n in header[ykey].split()])
    else:
        raise ValueError('Unknown STDPSF FITS file.')

    # STDPDF FITS positions are 1-indexed
    xgrid = np.array(xgrid) - 1
    ygrid = np.array(ygrid) - 1

    # (nypsfs, nxpsfs)
    # (6, 6)   # WFPC2, 4 det
    # (1, 1)   # ACS/HRC
    # (10, 9)  # ACS/WFC, 2 det
    # (3, 3)   # WFC3/IR
    # (8, 7)   # WFC3/UVIS, 2 det
    # (5, 5)   # NIRISS
    # (5, 5)   # NIRCam SW
    # (10, 20) # NIRCam SW (NRCSW), 8 det
    # (5, 5)   # NIRCam LW
    # (3, 3)   # MIRI

    grid_data = {'data': data,
                 'npsfs': npsfs,
                 'nxpsfs': nxpsfs,
                 'nypsfs': nypsfs,
                 'xgrid': xgrid,
                 'ygrid': ygrid}

    return grid_data


def _split_detectors(grid_data, detector_data, detector_id):
    """
    Split an ePSF array into individual detectors.

    In particular::

        * HST WFPC2 STDPSF file contains 4 detectors
        * HST ACS/WFC STDPSF file contains 2 detectors
        * HST WFC3/UVIS STDPSF file contains 2 detectors
        * JWST NIRCam "NRCSW" STDPSF file contains 8 detectors
    """
    data = grid_data['data']
    npsfs = grid_data['npsfs']
    nxpsfs = grid_data['nxpsfs']
    nypsfs = grid_data['nypsfs']
    xgrid = grid_data['xgrid']
    ygrid = grid_data['ygrid']
    nxdet = detector_data['nxdet']
    nydet = detector_data['nydet']
    det_map = detector_data['det_map']
    det_size = detector_data['det_size']

    ii = np.arange(npsfs).reshape((nypsfs, nxpsfs))
    nxpsfs //= nxdet
    nypsfs //= nydet
    ndet = nxdet * nydet
    ii = reshape_as_blocks(ii, (nypsfs, nxpsfs))
    ii = ii.reshape(ndet, npsfs // ndet)

    # detector_id -> index
    det_idx = det_map[detector_id]
    idx = ii[det_idx]
    data = data[idx]

    xp = det_idx % nxdet
    i0 = xp * nxpsfs
    i1 = i0 + nxpsfs
    xgrid = xgrid[i0:i1] - xp * det_size

    if det_idx < nxdet:
        ygrid = ygrid[:nypsfs]
    else:
        ygrid = ygrid[nypsfs:] - det_size

    return data, xgrid, ygrid


def _split_wfc_uvis(grid_data, detector_id):
    if detector_id is None:
        raise ValueError('detector_id must be specified for ACS/WFC and '
                         'WFC3/UVIS ePSFs.')
    if detector_id not in (1, 2):
        raise ValueError('detector_id must be 1 or 2.')

    # ACS/WFC1 and WFC3/UVIS1 chip1 (sci, 2) are above chip2 (sci, 1)
    # in y-pixel coordinates
    xgrid = grid_data['xgrid']
    ygrid = grid_data['ygrid']
    ygrid = ygrid.reshape((2, ygrid.shape[0] // 2))[detector_id - 1]
    if detector_id == 2:
        ygrid -= 2048

    npsfs = grid_data['npsfs']
    data = grid_data['data']
    data_ny, data_nx = data.shape[1:]
    data = data.reshape((2, npsfs // 2, data_ny, data_nx))[detector_id - 1]

    return data, xgrid, ygrid


def _split_wfpc2(grid_data, detector_id):
    if detector_id is None:
        raise ValueError('detector_id must be specified for WFPC2 ePSFs')
    if detector_id not in range(1, 5):
        raise ValueError('detector_id must be between 1 and 4, inclusive')

    nxdet = 2
    nydet = 2
    det_size = 800

    # det (exten:idx)
    # WF2 (2:2)  PC (1:3)
    # WF3 (3:0)  WF4 (4:1)
    det_map = {1: 3, 2: 2, 3: 0, 4: 1}

    detector_data = {'nxdet': nxdet,
                     'nydet': nydet,
                     'det_size': det_size,
                     'det_map': det_map}

    return _split_detectors(grid_data, detector_data, detector_id)


def _split_nrcsw(grid_data, detector_id):
    if detector_id is None:
        raise ValueError('detector_id must be specified for NRCSW ePSFs')
    if detector_id not in range(1, 9):
        raise ValueError('detector_id must be between 1 and 8, inclusive')

    nxdet = 4
    nydet = 2
    det_size = 2048

    # det (ext:idx)
    # A2 (2:4)  A4 (4:5)  B3 (7:6)  B1 (5:7)
    # A1 (1:0)  A3 (3:1)  B4 (8:2)  B2 (6:3)
    det_map = {1: 0, 3: 1, 8: 2, 6: 3, 2: 4, 4: 5, 7: 6, 5: 7}

    detector_data = {'nxdet': nxdet,
                     'nydet': nydet,
                     'det_size': det_size,
                     'det_map': det_map}

    return _split_detectors(grid_data, detector_data, detector_id)


def _get_metadata(filename, detector_id):
    """
    Get metadata from the filename and ``detector_id``.
    """
    if isinstance(filename, io.FileIO):
        filename = filename.name

    parts = os.path.basename(filename).strip('.fits').split('_')
    if len(parts) not in (3, 4):
        return None  # filename from astropy download_file

    detector, filter_name = parts[1:3]
    meta = {'STDPSF': filename,
            'detector': detector,
            'filter': filter_name}

    if detector_id is not None:
        detector_map = {'WFPC2': ['HST/WFPC2', 'WFPC2'],
                        'ACSHRC': ['HST/ACS', 'HRC'],
                        'ACSWFC': ['HST/ACS', 'WFC'],
                        'WFC3UV': ['HST/WFC3', 'UVIS'],
                        'WFC3IR': ['HST/WFC3', 'IR'],
                        'NRCSW': ['JWST/NIRCam', 'NRCSW'],
                        'NRCA1': ['JWST/NIRCam', 'A1'],
                        'NRCA2': ['JWST/NIRCam', 'A2'],
                        'NRCA3': ['JWST/NIRCam', 'A3'],
                        'NRCA4': ['JWST/NIRCam', 'A4'],
                        'NRCB1': ['JWST/NIRCam', 'B1'],
                        'NRCB2': ['JWST/NIRCam', 'B2'],
                        'NRCB3': ['JWST/NIRCam', 'B3'],
                        'NRCB4': ['JWST/NIRCam', 'B4'],
                        'NRCAL': ['JWST/NIRCam', 'A5'],
                        'NRCBL': ['JWST/NIRCam', 'B5'],
                        'NIRISS': ['JWST/NIRISS', 'NIRISS'],
                        'MIRI': ['JWST/MIRI', 'MIRIM']}

        try:
            inst_det = detector_map[detector]
        except KeyError as exc:
            raise ValueError(f'Unknown detector {detector}.') from exc

        if inst_det[1] == 'WFPC2':
            wfpc2_map = {1: 'PC', 2: 'WF2', 3: 'WF3', 4: 'WF4'}
            inst_det[1] = wfpc2_map[detector_id]

        if inst_det[1] in ('WFC', 'UVIS'):
            chip = 2 if detector_id == 1 else 1
            inst_det[1] = f'{inst_det[1]}{chip}'

        if inst_det[1] == 'NRCSW':
            sw_map = {1: 'A1', 2: 'A2', 3: 'A3', 4: 'A4',
                      5: 'B1', 6: 'B2', 7: 'B3', 8: 'B4'}
            inst_det[1] = sw_map[detector_id]

        meta['instrument'] = inst_det[0]
        meta['detector'] = inst_det[1]

    return meta


def stdpsf_reader(filename, detector_id=None):
    """
    Generate a `~photutils.psf.GriddedPSFModel` from a STScI
    standard-format ePSF (STDPSF) FITS file.

    .. note::
        Instead of being used directly, this function is intended to be
        used via the `GriddedPSFModel` ``read`` method, e.g.,
        ``model = GriddedPSFModel.read(filename, format='stdpsf')``.

    STDPSF files are FITS files that contain a 3D array of ePSFs with
    the header detailing where the fiducial ePSFs are located in the
    detector coordinate frame.

    The oversampling factor for STDPSF FITS files is assumed to be 4.

    Parameters
    ----------
    filename : str
        The name of the STDPDF FITS file. A URL can also be used.

    detector_id : `None` or int, optional
        For STDPSF files that contain ePSF grids for multiple detectors,
        one will need to identify the detector for which to extract the
        ePSF grid. This keyword is ignored for STDPSF files that do not
        contain ePSF grids for multiple detectors.

        For WFPC2, the detector value (int) should be:

            - 1: PC, 2: WF2, 3: WF3, 4: WF4

        For ACS/WFC and WFC3/UVIS, the detector value should be:

            - 1: WFC2, UVIS2 (sci, 1)
            - 2: WFC1, UVIS1 (sci, 2)

        Note that for these two instruments, detector 1 is above
        detector 2 in the y direction. However, in the FLT FITS files,
        the (sci, 1) extension corresponds to detector 2 (WFC2, UVIS2)
        and the (sci, 2) extension corresponds to detector 1 (WFC1,
        UVIS1).

        For NIRCam NRCSW files that contain ePSF grids for all 8 SW
        detectors, the detector value should be:

            * 1: A1, 2: A2, 3: A3, 4: A4
            * 5: B1, 6: B2, 7: B3, 8: B4

    Returns
    -------
    model : `~photutils.psf.GriddedPSFModel`
        The gridded ePSF model.
    """
    grid_data = _read_stdpsf(filename)

    npsfs = grid_data['npsfs']
    if npsfs in (90, 56, 36, 200):
        if npsfs in (90, 56):  # ACS/WFC or WFC3/UVIS data (2 chips)
            data, xgrid, ygrid = _split_wfc_uvis(grid_data, detector_id)
        elif npsfs == 36:  # WFPC2 data (4 chips)
            data, xgrid, ygrid = _split_wfpc2(grid_data, detector_id)
        elif npsfs == 200:  # NIRCam SW data (8 chips)
            data, xgrid, ygrid = _split_nrcsw(grid_data, detector_id)
        else:
            raise ValueError('Unknown detector or STDPSF format')
    else:
        data = grid_data['data']
        xgrid = grid_data['xgrid']
        ygrid = grid_data['ygrid']

    # itertools.product iterates over the last input first
    xy_grid = [yx[::-1] for yx in itertools.product(ygrid, xgrid)]

    oversampling = 4  # assumption for STDPSF files
    nxpsfs = xgrid.shape[0]
    nypsfs = ygrid.shape[0]
    meta = {'grid_xypos': xy_grid,
            'oversampling': oversampling,
            'nxpsfs': nxpsfs,
            'nypsfs': nypsfs}

    # try to get additional metadata from the filename because this
    # information is not currently available in the FITS headers
    file_meta = _get_metadata(filename, detector_id)
    if file_meta is not None:
        meta.update(file_meta)

    return GriddedPSFModel(NDData(data, meta=meta))


def webbpsf_reader(filename):
    """
    Generate a `~photutils.psf.GriddedPSFModel` from a WebbPSF
    FITS file containing a PSF grid.

    .. note::
        Instead of being used directly, this function is intended to be
        used via the `GriddedPSFModel` ``read`` method, e.g., ``model =
        GriddedPSFModel.read(filename, format='webbpsf')``.

    The WebbPSF FITS file contain a 3D array of ePSFs with the header
    detailing where the fiducial ePSFs are located in the detector
    coordinate frame.

    Parameters
    ----------
    filename : str
        The name of the WebbPSF FITS file. A URL can also be used.

    Returns
    -------
    model : `~photutils.psf.GriddedPSFModel`
        The gridded ePSF model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        with fits.open(filename, ignore_missing_end=True) as hdulist:
            header = hdulist[0].header
            data = hdulist[0].data

    # handle the case of only one 2D PSF
    data = np.atleast_3d(data)

    if not any('DET_YX' in key for key in header.keys()):
        raise ValueError('Invalid WebbPSF FITS file; missing "DET_YX{}" '
                         'header keys.')
    if 'OVERSAMP' not in header.keys():
        raise ValueError('Invalid WebbPSF FITS file; missing "OVERSAMP" '
                         'header key.')

    # convert header to meta dict
    header = header.copy(strip=True)
    header.pop('HISTORY', None)
    header.pop('COMMENT', None)
    header.pop('', None)
    meta = dict(header)
    meta = {key.lower(): meta[key] for key in meta}  # user lower-case keys

    # define grid_xypos from DET_YX{} FITS header keywords
    xypos = []
    for key in meta.keys():
        if 'det_yx' in key:
            vals = header[key].lstrip('(').rstrip(')').split(',')
            xypos.append((float(vals[0]), float(vals[1])))
    meta['grid_xypos'] = xypos

    if 'oversampling' not in meta:
        meta['oversampling'] = meta['oversamp']

    ndd = NDData(data, meta=meta)

    return GriddedPSFModel(ndd)


def is_stdpsf(origin, filepath, fileobj, *args, **kwargs):
    """
    Determine whether `origin` is a STDPSF FITS file.

    Parameters
    ----------
    origin : str or readable file-like
        Path or file object containing a potential FITS file.

    Returns
    -------
    is_stdpsf : bool
        Returns `True` if the given file is a STDPSF FITS file.
    """
    if filepath is not None:
        extens = ('.fits', '.fits.gz', '.fit', '.fit.gz', '.fts', '.fts.gz')
        isfits = filepath.lower().endswith(extens)
        if isfits:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', VerifyWarning)
                header = fits.getheader(filepath)
            keys = ('NAXIS3', 'NXPSFS', 'NYPSFS')
            for key in keys:
                if key not in header:
                    return False
            return True
    return False


def is_webbpsf(origin, filepath, fileobj, *args, **kwargs):
    """
    Determine whether `origin` is a WebbPSF FITS file.

    Parameters
    ----------
    origin : str or readable file-like
        Path or file object containing a potential FITS file.

    Returns
    -------
    is_webbpsf : bool
        Returns `True` if the given file is a WebbPSF FITS file.
    """
    if filepath is not None:
        extens = ('.fits', '.fits.gz', '.fit', '.fit.gz', '.fts', '.fts.gz')
        isfits = filepath.lower().endswith(extens)
        if isfits:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', VerifyWarning)
                header = fits.getheader(filepath)
            keys = ('NAXIS3', 'OVERSAMP', 'DET_YX0')
            for key in keys:
                if key not in header:
                    return False
            return True
    return False


class STDPSFGrid(ModelGridPlotMixin):
    """
    Class to read and plot "STDPSF" format ePSF model grids.

    STDPSF files are FITS files that contain a 3D array of ePSFs with
    the header detailing where the fiducial ePSFs are located in the
    detector coordinate frame.

    The oversampling factor for STDPSF FITS files is assumed to be 4.

    Parameters
    ----------
    filename : str
        The name of the STDPDF FITS file. A URL can also be used.

    Examples
    --------
    >>> psfgrid = STDPSFGrid.read('STDPSF_ACSWFC_F814W.fits')
    >>> psfgrid.plot_grid()
    """

    def __init__(self, filename):
        grid_data = _read_stdpsf(filename)
        self.data = grid_data['data']
        self._xgrid = grid_data['xgrid']
        self._ygrid = grid_data['ygrid']
        xy_grid = [yx[::-1] for yx in itertools.product(self._ygrid,
                                                        self._xgrid)]
        oversampling = 4  # assumption for STDPSF files
        self.grid_xypos = xy_grid
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        meta = {'grid_shape': (len(self._ygrid), len(self._xgrid)),
                'grid_xypos': xy_grid,
                'oversampling': oversampling}

        # try to get additional metadata from the filename because this
        # information is not currently available in the FITS headers
        file_meta = _get_metadata(filename, None)
        if file_meta is not None:
            meta.update(file_meta)

        self.meta = meta

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        cls_info = []

        keys = ('STDPSF', 'detector', 'filter', 'grid_shape')
        for key in keys:
            if key in self.meta:
                name = key.capitalize() if key != 'STDPSF' else key
                cls_info.append((name, self.meta[key]))

        cls_info.extend([('Number of ePSFs', len(self.grid_xypos)),
                         ('ePSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', self.oversampling)])

        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'{key}: {val}' for key, val in cls_info]

        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()


with registry.delay_doc_updates(GriddedPSFModel):
    registry.register_reader('stdpsf', GriddedPSFModel, stdpsf_reader)
    registry.register_identifier('stdpsf', GriddedPSFModel, is_stdpsf)
    registry.register_reader('webbpsf', GriddedPSFModel, webbpsf_reader)
    registry.register_identifier('webbpsf', GriddedPSFModel, is_webbpsf)
