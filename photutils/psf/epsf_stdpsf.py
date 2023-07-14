# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to load ePSFs in "STDPSF" format, as
created for HST and JWST by J. Anderson et al., and convert them into
GriddedPSFModel instances.
"""

import os

import astropy
import numpy as np
from astropy.io import fits

from photutils.psf.models import GriddedPSFModel

__all__ = ['STDPSFGrid']


class STDPSFGrid:
    """
    Pythonic wrapper for "STDPSF" format ePSF model grids, from Anderson
    et al.

    See for instance files available at
    https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs/NIRISS/
    from Libralatto et al. 2023

    This class includes support for special "NRCSW"-format grids that
    combine all the SW detectors together.

    Note that STDPSF format files contain minimal FITS header metadata,
    so the filenames are treated as semantically significant for
    inference about instrument and filter.

    Parameters
    ----------
    filename : str
        Filename to load
    oversample : int, optional
        Oversampling factor relative to native detector pixels. This is
        not available from FITS headers, so if the value differs from
        the default, assumed to be 4, then it must be provided by the
        user.
    filter : str, optional
        Bandpass filter name. Set this if it cannot be inferred from the
        filename.
    verbose : bool, optional
        Whether to be more verbose in output.
    """

    def __init__(self, filename, oversample=4, filter=None, verbose=True):
        self.filename = filename
        self.hdu = fits.open(filename)

        # Read header keywords
        self._nxpsfs = self.hdu[0].header['NXPSFS']
        self._nypsfs = self.hdu[0].header['NYPSFS']
        self._nx = self.hdu[0].header['NAXIS1']
        self._ny = self.hdu[0].header['NAXIS2']
        self._npsfs = self.hdu[0].header['NAXIS3']

        # This is not recorded in the header anywhere
        self.oversample = oversample

        # STD-format PSFs from Fortran have minimal metadata in the FITS
        # headers. Therefore attempt to infer the instrument and filter
        # from filename
        fn_base = os.path.basename(filename)
        fn_middle = fn_base.split('_')[1]

        if 'NRCSW' in fn_base:
            # special case file format for
            # All the NIRCam SW detectors stacked together.
            self.instrument = 'NIRCam'
            self.detector = 'NRCSW'
            self._ndetectors = 8
            # convert to N of PSFs **per detector**
            self._npsfs //= self._ndetectors
        elif 'NRC' in fn_middle:
            self.instrument = 'NIRCam'
            self.detector = fn_middle
            self._ndetectors = 1
        else:
            instruments = ['NIRISS', 'MIRI', 'WFC3UV', 'WFC3IR']

            for instname in instruments:
                self.instrument = instname
                self.detector = 'MIRIM' if instname == 'MIRI' else instname

                # special case: WFC3UV has 2 detectors but, unlike for
                # NIRCam, these sipmly get put in one grid
                self._ndetectors = 1

                break
            else:
                raise NotImplementedError('Not sure what instrument this '
                                          'is...')

        if filter:
            self.filter = filter
        else:
            # attempt to infer filter
            fn_parts = os.path.splitext(fn_base)[0].split('_')
            for part in fn_parts:
                if part.startswith('F'):
                    self.filter = part
                    break
            else:
                raise RuntimeError('Could not infer filter from filename.')

        if verbose:
            print(f'Loading STD format ePSF grid for {self.instrument}, '
                  f'filter={self.filter}')
            print(f'   Using oversample = {self.oversample}.  \n')

        if self._ndetectors == 1:
            # Read in PSF locations from IPSFXnn, JPSFYnn keywords
            self._ipsfx = [self.hdu[0].header[f'IPSFX{i + 1:02d}']
                           for i in range(self._nxpsfs)]
            self._jpsfy = [self.hdu[0].header[f'JPSFY{j + 1:02d}']
                           for j in range(self._nypsfs)]
        elif self._ndetectors == 8:
            self._ipsfx = [int(n)
                           for n in self.hdu[0].header['IPSFXA5'].split()] * 4
            self._jpsfy = [int(n)
                           for n in self.hdu[0].header['JPSFYA5'].split()] * 2

        else:
            raise NotImplementedError('Not sure how to handle '
                                      f'{self._ndetectors} detectors')

        if verbose:
            print(f'   Found {self._npsfs * self._ndetectors} ePSFs on '
                  f'{self._ndetectors} detector(s)')
            print(f'   Each PSF is {self._nx} x {self._ny} oversampled '
                  f'pixels, or {self._nx/self.oversample} x '
                  f'{self._ny / self.oversample} real pixels.')

        # Load data
        self.data = self.hdu[0].data.copy()
        self.data.shape = (self._ndetectors, self._npsfs, self._ny, self._nx)

        # special case: determine if any ePSFs are invalid/blank (i.e.
        # all zeros). This can be the case for MIRI with its non-square
        # FOV, in work in progress by Libralatto. We create a bool mask
        # for whether the ePSF at each position is nonzero and valid.
        self._valid_mask = np.ones((self._ndetectors, self._npsfs), bool)
        self._valid_mask = np.any(np.any(self.data[self._valid_mask] != 0,
                                         axis=-1), axis=-1)
        self._valid_mask.shape = (self._ndetectors, self._npsfs)

        # Compute mean PSF and deltas relative to that, using only valid PSFs
        self.psf_mean = self.data[self._valid_mask].mean(axis=0)
        self.psf_deltas = self.data - self.psf_mean
        self.psf_deltas[~self._valid_mask] = 0

    def as_2d(self, deltas=False):
        """
        Return the ePSF grid unpacked and stacked into a 2D array,
        arranged NxN.

        Parameters
        ----------
        deltas : bool
            If true, return the differences relative to the mean ePSF.
            Otherwise by default return the actual ePSFs.
        """
        if deltas:
            gridcopy = self.psf_deltas.copy()
        else:
            gridcopy = self.data.copy()

        # note, the following works regardless of self._ndetectors,
        # since we adjust the array shape directly
        gridcopy.shape = (self._nypsfs, self._nxpsfs, self._ny, self._nx)
        array2d = gridcopy.transpose([0, 2, 1, 3]).reshape(
            (self._nypsfs * self._ny, self._nxpsfs * self._nx))

        return array2d

    def display_grid(self, show_deltas=False, vmax_scale=None,
                     show_dividers=True, divider_color='darkgray',
                     divider_ls='-', peak_norm=False, percent_norm=False,
                     ax=None):
        """
        Plot the PSF grid.

        Parameters
        ----------
        show_deltas : bool, optional
            Set to `True` to show the differences between each ePSF and
            the average ePSF.
        ax : optional matplotlib Axes, optional
            Provide this to plot into an already-existing axes.
        show_dividers : bool, optional
            Whether to show divider lines between ePSFs.
        divider_color, divider_ls : str, optional
            Matplotlib display options for the divider lines between
            ePSFs.
        peak_norm, percent_norm : bool, optional
            Options for how to normalize the ePSFs. Default shows flux
            per pixel.
        vmax_scale : float, optional
            Scale factor to increase or decrease the display stretch
            limits.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import LogNorm, Normalize

        if ax is None:
            plt.figure(figsize=(12 if self._ndetectors == 1 else 20, 8))
            ax = plt.gca()

        grid2d = self.as_2d(deltas=show_deltas)

        if percent_norm and not show_deltas:
            raise RuntimeError('The percent normalization is only available '
                               'if showing deltas')

        if peak_norm:  # normalize relative to peak
            grid2d /= self.data.max()
        elif percent_norm:
            # normalize relative to peak, then convert to percentages
            grid2d /= self.psf_mean.max() / 100

        if show_deltas:
            # To match Librallato 2023 Fig 4 plot
            cm = cm.gray_r.copy()
            vmax = 100 if percent_norm else grid2d.max()
            if vmax_scale is None:
                vmax_scale = 0.03

            norm = Normalize(vmin=-vmax * vmax_scale, vmax=vmax * vmax_scale)
        else:
            vmax = grid2d.max()
            if vmax_scale is None:
                vmax_scale = 1.0

            norm = LogNorm(vmin=vmax / 1e4 * vmax_scale,
                           vmax=vmax * vmax_scale)
            cm = cm.viridis.copy()

        cm.set_bad(cm(0))

        # Coordinate axes setup to later set tick labels based on
        # detector psf coords. This sets up axes to have, behind the
        # scenes, the ePSFs centered at integer coords 0, 1, 2,3 etc.
        # extent = (left, right, bottom, top)
        extent = [-0.5, self._nxpsfs - 0.5, -0.5, self._nypsfs - 0.5]

        ax.imshow(grid2d, extent=extent, norm=norm, cmap=cm, origin='lower')

        # Use the axes set up above to set appropriate tick labels
        ax.set_xticks(np.arange(self._nxpsfs))
        ax.set_xticklabels(self._ipsfx)
        ax.set_xlabel('ePSF location in detector X pixels')
        ax.set_yticks(np.arange(self._nypsfs))
        ax.set_yticklabels(self._jpsfy)
        ax.set_ylabel('ePSF location in detector Y pixels')

        if show_dividers:
            for ix in range(self._nxpsfs):
                ax.axvline(ix + 0.5, color=divider_color, ls=divider_ls)
            for iy in range(self._nypsfs):
                ax.axhline(iy + 0.5, color=divider_color, ls=divider_ls)

        if self.detector == 'NRCSW':
            # NIRCam SW all detectors gets extra divider lines and SCA
            # name labels
            plt.axhline(self._nypsfs / 2 - 0.5, color='orange')
            for i in range(1, 4):
                ax.axvline(self._nxpsfs / 4 * i - 0.5, color='orange')

            det_labels = [['A1', 'A3', 'B4', 'B2'], ['A2', 'A4', 'B3', 'B1']]
            for i in range(2):
                for j in range(4):
                    ax.text(j * self._nxpsfs / 4 - 0.45,
                            (i + 1) * self._nypsfs / 2 - 0.55,
                            det_labels[i][j], color='orange',
                            verticalalignment='top', fontsize=18)

        if show_deltas:
            ax.set_title(f'{os.path.basename(self.filename)} ePSFs - '
                         'average ePSF')
        else:
            ax.set_title(f'{os.path.basename(self.filename)} ePSFs')

        if peak_norm:
            plt.colorbar(label='Scale relative to ePSF peak',
                         mappable=ax.images[0])
        elif percent_norm:
            plt.colorbar(label='Difference relative to average ePSF peak [%]',
                         mappable=ax.images[0])
        else:
            plt.colorbar(label='ePSF flux per pixel', mappable=ax.images[0])

    def to_griddedpsfmodel(self):
        """
        Convert this PSF grid to a `~photutils.psf.GriddedPSFModel`.

        This is a format conversion only; there are no modifications to
        the pixel data values.

        Returns
        -------
        result : `~photutils.psf.GriddedPSFModel` instance
            The gridded PSF model.
        """
        if self._ndetectors > 1:
            raise RuntimeError('Cannot convert a STDPSF grid with multiple '
                               'detectors into a GriddedPSFModel')

        psfdata = self.data.reshape(self._ndetectors * self._npsfs, self._nx,
                                    self._ny)

        meta = {}
        meta['instrume'] = (self.instrument, 'Instrument name')
        meta['detector'] = (self.detector, 'Detector')
        meta['filter'] = (self.filter, 'Filter')
        meta['source'] = (self.filename, 'PSF data taken from this file.')
        meta['oversampling'] = self.oversample

        meta['grid_xypos'] = []
        k = 0
        for j in range(self._nypsfs):
            for i in range(self._nxpsfs):
                meta['grid_xypos'].append((self._ipsfx[i], self._jpsfy[j]))
                k += 1

        psfdata_with_meta = astropy.nddata.NDData(psfdata, meta=meta)

        return GriddedPSFModel(psfdata_with_meta)
