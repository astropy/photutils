# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines tools to plot Gridded PSF models.
"""

import astropy
import numpy as np
from astropy.utils import minversion
from astropy.visualization import simple_norm

__all__ = ['ModelGridPlotMixin']


class ModelGridPlotMixin:
    """
    Mixin class to plot a grid of ePSF models.
    """

    def _reshape_grid(self, data):
        """
        Reshape the 3D ePSF grid as a 2D array of horizontally and
        vertically stacked ePSFs.

        Parameters
        ----------
        data : `numpy.ndarray`
            The 3D array of ePSF data.

        Returns
        -------
        reshaped_data : `numpy.ndarray`
            The 2D array of ePSF data.
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
            The matplotlib axes on which to plot. If `None`, then the
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

        Notes
        -----
        This method returns a figure object. If you are using this
        method in a script, you will need to call ``plt.show()`` to
        display the figure. If you are using this method in a Jupyter
        notebook, the figure will be displayed automatically.

        When in a notebook, if you do not store the return value of this
        function, the figure will be displayed twice due to the REPL
        automatically displaying the return value of the last function
        call. Alternatively, you can append a semicolon to the end of
        the function call to suppress the display of the return value.
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

        if peak_norm and data.max() != 0:
            # normalize relative to peak
            data /= data.max()

        if deltas:
            if cmap is None:
                cmap = cm.gray_r.copy()

            if vmax_scale is None:
                vmax_scale = 0.03
            vmax = data.max() * vmax_scale
            vmin = -vmax
            if minversion(astropy, '6.1'):
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
            if minversion(astropy, '6.1'):
                norm = simple_norm(data, 'log', vmin=vmin, vmax=vmax,
                                   log_a=1.0e4)
            else:
                norm = simple_norm(data, 'log', min_cut=vmin, max_cut=vmax,
                                   log_a=1.0e4)

        # Set up the coordinate axes to later set tick labels based on
        # detector ePSF coordinates. This sets up axes to have, behind the
        # scenes, the ePSFs centered at integer coords 0, 1, 2, 3 etc.
        # extent order: left, right, bottom, top
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
            minus = '\u2212'
            ax.set_title(f'{title}(ePSFs {minus} <ePSF>)')
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
