# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model_plotting module.
"""

from itertools import product

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NDData

from photutils.psf import GriddedPSFModel, STDPSFGrid
from photutils.psf.tests.test_model_io import (STDPSF_FILENAMES,
                                               WEBBPSF_FILENAMES,
                                               get_data_filename)
from photutils.utils._optional_deps import HAS_MATPLOTLIB

pytestmark = pytest.mark.skipif(not HAS_MATPLOTLIB,
                                reason='matplotlib is required')


@pytest.fixture(autouse=True)
def close_figures():
    """
    Close all matplotlib figures after each test so that the open
    figure limit is not exceeded.
    """
    yield

    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture(name='psfmodel')
def fixture_griddedpsf_data():
    psfs = []
    yy, xx = np.mgrid[0:101, 0:101]
    for i in range(16):
        theta = np.deg2rad(i * 10.0)
        gmodel = Gaussian2D(1, 50, 50, 10, 5, theta=theta)
        psfs.append(gmodel(xx, yy))

    xgrid = [0, 40, 160, 200]
    ygrid = [0, 60, 140, 200]
    meta = {'grid_xypos': list(product(xgrid, ygrid)),
            'oversampling': 4}

    return GriddedPSFModel(NDData(psfs, meta=meta))


class TestPlotGrid:
    """
    Tests for the GriddedPSFModel and STDPSFGrid plot_grid methods.
    """

    def test_plot_grid(self, psfmodel):
        psfmodel.plot_grid()
        psfmodel.plot_grid(peak_norm=True, cmap='Blues', vmax_scale=0.9)
        psfmodel.plot_grid(deltas=True)
        psfmodel.plot_grid(deltas=True, peak_norm=True)

    def test_plot_grid_blank_epsfs(self, psfmodel):
        """
        Test a grid where one or more ePSFs are blank (all zeros), as
        is the case for MIRI with its non-square FOV.
        """
        model = psfmodel.deepcopy()
        model._data[0] = 0.0
        model.plot_grid(deltas=True)

    def test_plot_grid_input_axes(self, psfmodel):
        """
        Test that a user-supplied axes is used and that its figure is
        returned.
        """
        import matplotlib.pyplot as plt

        fig_in, ax = plt.subplots()
        fig_out = psfmodel.plot_grid(ax=ax)
        assert fig_out is fig_in

    def test_plot_grid_dividers(self, psfmodel):
        fig = psfmodel.plot_grid(dividers=False)
        ax = fig.axes[0]
        assert len(ax.lines) == 0

        fig = psfmodel.plot_grid(dividers=True, divider_color='red',
                                 divider_ls=':')
        ax = fig.axes[0]
        # 3 vertical and 3 horizontal dividers for a 4x4 grid
        assert len(ax.lines) == 6

    def test_plot_grid_figsize(self, psfmodel):
        fig = psfmodel.plot_grid(figsize=(6, 5))
        assert tuple(fig.get_size_inches()) == (6.0, 5.0)

    @pytest.mark.parametrize('filename', WEBBPSF_FILENAMES)
    def test_plot_grid_webbpsf(self, filename):
        psfmodel = GriddedPSFModel.read(get_data_filename(filename),
                                        format='webbpsf')
        psfmodel.plot_grid()

    @pytest.mark.parametrize('filename', STDPSF_FILENAMES)
    def test_plot_grid_stdpsfgrid(self, filename):
        psfgrid = STDPSFGrid(get_data_filename(filename))
        psfgrid.plot_grid()

    def test_plot_grid_tuple_metadata(self, psfmodel):
        """
        Test that tuple-valued metadata is unwrapped for the plot title.

        WebbPSF stores each header value as a ``(value, comment)``
        tuple.
        """
        model = psfmodel.deepcopy()
        model.meta['instrument'] = ('NIRCam', 'instrument name')
        model.meta['detector'] = ['NRCA1', 'detector name']
        model.meta['filter'] = np.array(['F200W', 'filter name'])

        fig = model.plot_grid()
        assert fig.axes[0].get_title() == 'NIRCam NRCA1 F200W ePSFs'

    def test_plot_grid_instrume_metadata(self, psfmodel):
        """
        Test the WebbPSF 'instrume' metadata key fallback.
        """
        model = psfmodel.deepcopy()
        model.meta['instrume'] = 'NIRCam'

        fig = model.plot_grid()
        assert fig.axes[0].get_title().startswith('NIRCam')

    def test_plot_grid_no_metadata(self, psfmodel):
        """
        Test that no leading space is added when there is no
        instrument, detector, or filter metadata.
        """
        fig = psfmodel.plot_grid()
        assert fig.axes[0].get_title() == 'ePSFs'

    def test_plot_grid_nrcsw(self):
        """
        Test the NIRCam NRCSW special case, which adds detector labels
        and extra divider lines.
        """
        psfgrid = STDPSFGrid(get_data_filename('STDPSF_NRCSW_F150W_mock.fits'))
        assert psfgrid.meta['detector'] == 'NRCSW'

        fig = psfgrid.plot_grid()
        texts = [text.get_text() for text in fig.axes[0].texts]
        assert set(texts) == {'A1', 'A2', 'A3', 'A4',
                              'B1', 'B2', 'B3', 'B4'}
        assert tuple(fig.get_size_inches()) == (20.0, 8.0)

    @pytest.mark.parametrize('deltas', [False, True])
    def test_plot_grid_colorbar_label(self, psfmodel, deltas):
        for peak_norm in (False, True):
            fig = psfmodel.plot_grid(deltas=deltas, peak_norm=peak_norm)
            # The colorbar is the second axes
            assert fig.axes[1].get_ylabel() != ''
