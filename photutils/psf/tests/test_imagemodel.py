# Licensed under a 3-clause BSD style license - see LICENSE.rst

from itertools import product
import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NDData
import astropy.units as u

from .. import FittableImageModel, GriddedPSFModel
from ...segmentation import detect_sources, source_properties

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_image_model():
    gm = Gaussian2D(x_stddev=3, y_stddev=3)
    xg, yg = np.mgrid[-2:3, -2:3]

    imod_nonorm = FittableImageModel(gm(xg, yg))
    assert np.allclose(imod_nonorm(0, 0), gm(0, 0))
    assert np.allclose(imod_nonorm(1, 1), gm(1, 1))
    assert np.allclose(imod_nonorm(-2, 1), gm(-2, 1))

    # now sub-pixel should *not* match, but be reasonably close
    assert not np.allclose(imod_nonorm(0.5, 0.5), gm(0.5, 0.5))
    # in this case good to ~0.1% seems to be fine
    assert np.allclose(imod_nonorm(0.5, 0.5), gm(0.5, 0.5), rtol=.001)
    assert np.allclose(imod_nonorm(-0.5, 1.75), gm(-0.5, 1.75), rtol=.001)

    imod_norm = FittableImageModel(gm(xg, yg), normalize=True)
    assert not np.allclose(imod_norm(0, 0), gm(0, 0))
    assert np.allclose(np.sum(imod_norm(xg, yg)), 1)

    imod_norm2 = FittableImageModel(gm(xg, yg), normalize=True,
                                    normalization_correction=2)
    assert not np.allclose(imod_norm2(0, 0), gm(0, 0))
    assert np.allclose(imod_norm(0, 0), imod_norm2(0, 0)*2)
    assert np.allclose(np.sum(imod_norm2(xg, yg)), 0.5)


@pytest.mark.skipif('not HAS_SCIPY')
def test_image_model_oversampling():
    gm = Gaussian2D(x_stddev=3, y_stddev=3)

    osa = 3  # oversampling factor
    xg, yg = np.mgrid[-3:3.00001:(1/osa), -3:3.00001:(1/osa)]

    im = gm(xg, yg)
    # should be obvious, but at least ensures the test is right:
    assert im.shape[0] > 7
    imod_oversampled = FittableImageModel(im, oversampling=osa)

    assert np.allclose(imod_oversampled(0, 0), gm(0, 0))
    assert np.allclose(imod_oversampled(1, 1), gm(1, 1))
    assert np.allclose(imod_oversampled(-2, 1), gm(-2, 1))
    assert np.allclose(imod_oversampled(0.5, 0.5), gm(0.5, 0.5), rtol=.001)
    assert np.allclose(imod_oversampled(-0.5, 1.75), gm(-0.5, 1.75), rtol=.001)

    imod_wrongsampled = FittableImageModel(im)

    # now make sure that all *fails* without the oversampling
    # except for at the origin
    assert np.allclose(imod_wrongsampled(0, 0), gm(0, 0))
    assert not np.allclose(imod_wrongsampled(1, 1), gm(1, 1))
    assert not np.allclose(imod_wrongsampled(-2, 1), gm(-2, 1))
    assert not np.allclose(imod_wrongsampled(0.5, 0.5), gm(0.5, 0.5),
                           rtol=.001)
    assert not np.allclose(imod_wrongsampled(-0.5, 1.75), gm(-0.5, 1.75),
                           rtol=.001)


@pytest.mark.skipif('not HAS_SCIPY')
def test_centering_oversampled():
    gm = Gaussian2D(x_stddev=2, y_stddev=3)

    osa = 3  # oversampling factor
    xg, yg = np.mgrid[-3:3.00001:(1/osa), -3:3.00001:(1/osa)]

    imod_oversampled = FittableImageModel(gm(xg, yg), oversampling=osa)

    valcen = gm(0, 0)
    val36 = gm(0.66, 0.66)

    assert np.allclose(valcen, imod_oversampled(0, 0))
    assert np.allclose(val36, imod_oversampled(0.66, 0.66))

    imod_oversampled.x_0 = 2.5
    imod_oversampled.y_0 = -3.5

    assert np.allclose(valcen, imod_oversampled(2.5, -3.5))
    assert np.allclose(val36, imod_oversampled(2.5 + 0.66, -3.5 + 0.66))


class TestGriddedPSFModel:
    def setup_class(self):
        psfs = []
        y, x = np.mgrid[0:101, 0:101]
        for i in range(16):
            theta = i * 10. * np.pi / 180.
            g = Gaussian2D(1, 50, 50, 10, 5, theta=theta)
            m = g(x, y)
            psfs.append(m)

        xgrid = [0, 40, 160, 200]
        ygrid = [0, 60, 140, 200]
        grid_xypos = list(product(xgrid, ygrid))

        meta = {}
        meta['grid_xypos'] = grid_xypos
        meta['oversampling'] = 4
        self.nddata = NDData(psfs, meta=meta)
        self.psfmodel = GriddedPSFModel(self.nddata)

    def test_gridded_psf_model(self):
        keys = ['grid_xypos', 'oversampling']
        for key in keys:
            assert key in self.psfmodel.meta
        assert len(self.psfmodel.meta) == 2
        assert len(self.psfmodel.meta['grid_xypos']) == 16
        assert self.psfmodel.oversampling == 4
        assert (self.psfmodel.meta['oversampling'] ==
                self.psfmodel.oversampling)
        assert self.psfmodel.data.shape == (16, 101, 101)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_gridded_psf_model_basic_eval(self):
        y, x = np.mgrid[0:100, 0:100]
        psf = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=40, y_0=60)
        assert psf.shape == (100, 100)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_gridded_psf_model_eval_outside_grid(self):
        y, x = np.mgrid[-50:50, -50:50]
        psf1 = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=0, y_0=0)
        y, x = np.mgrid[-60:40, -60:40]
        psf2 = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=-10, y_0=-10)
        assert_allclose(psf1, psf2)

        y, x = np.mgrid[150:250, 150:250]
        psf3 = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=200, y_0=200)
        y, x = np.mgrid[170:270, 170:270]
        psf4 = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=220, y_0=220)
        assert_allclose(psf3, psf4)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_gridded_psf_model_interp(self):
        # test xyref length
        with pytest.raises(ValueError):
            self.psfmodel._bilinear_interp([1, 1], 1, 1, 1)

        # test zref shape
        with pytest.raises(ValueError):
            xyref = [[0, 0], [0, 1], [1, 0], [1, 1]]
            zref = np.ones((3, 4, 4))
            self.psfmodel._bilinear_interp(xyref, zref, 1, 1)

        # test if refxy points form a rectangle
        with pytest.raises(ValueError):
            xyref = [[0, 0], [0, 1], [1, 0], [2, 2]]
            zref = np.ones((4, 4, 4))
            self.psfmodel._bilinear_interp(xyref, zref, 1, 1)

        # test if xi and yi are outside of xyref
        xyref = [[0, 0], [0, 1], [1, 0], [1, 1]]
        zref = np.ones((4, 4, 4))
        with pytest.raises(ValueError):
            self.psfmodel._bilinear_interp(xyref, zref, 100, 1)
        with pytest.raises(ValueError):
            self.psfmodel._bilinear_interp(xyref, zref, 1, 100)

        # test non-scalar xi and yi
        idx = [0, 1, 4, 5]
        xyref = np.array(self.psfmodel.grid_xypos)[idx]
        psfs = self.psfmodel.data[idx, :, :]
        val1 = self.psfmodel._bilinear_interp(xyref, psfs, 10, 20)
        val2 = self.psfmodel._bilinear_interp(xyref, psfs, [10], [20])
        assert_allclose(val1, val2)

    def test_gridded_psf_model_invalid_inputs(self):
        data = np.ones((4, 3, 3))

        # check if NDData
        with pytest.raises(TypeError):
            GriddedPSFModel(data)

        # check PSF data dimension
        with pytest.raises(ValueError):
            GriddedPSFModel(NDData(np.ones((3, 3))))

        # check that grid_xypos is in meta
        meta = {'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check grid_xypos length
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0]],
                'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check if grid_xypos is a regular grid
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0], [3, 4]],
                'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check that oversampling is in meta
        meta = {'grid_xypos': [[0, 0], [0, 1], [1, 0], [1, 1]]}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check oversampling is a scalar
        meta = {'grid_xypos': [[0, 0], [0, 1], [1, 0], [1, 1]],
                'oversampling': [4, 4]}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_gridded_psf_model_eval(self):
        """
        Create a simulated image using GriddedPSFModel and test
        the properties of the generated sources.
        """

        shape = (200, 200)
        data = np.zeros(shape)
        eval_xshape = np.int(np.ceil(self.psfmodel.data.shape[2] /
                                     self.psfmodel.oversampling))
        eval_yshape = np.int(np.ceil(self.psfmodel.data.shape[1] /
                                     self.psfmodel.oversampling))

        xx = [40, 50, 160, 160]
        yy = [60, 150, 50, 140]
        zz = [100, 100, 100, 100]
        for xxi, yyi, zzi in zip(xx, yy, zz):
            x0 = np.int(np.floor(xxi - (eval_xshape - 1) / 2.))
            y0 = np.int(np.floor(yyi - (eval_yshape - 1) / 2.))
            x1 = x0 + eval_xshape
            y1 = y0 + eval_yshape

            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > shape[1]:
                x1 = shape[1]
            if y1 > shape[0]:
                y1 = shape[0]

            y, x = np.mgrid[y0:y1, x0:x1]
            data[y, x] += self.psfmodel.evaluate(x=x, y=y, flux=zzi, x_0=xxi,
                                                 y_0=yyi)

        segm = detect_sources(data, 0., 5)
        props = source_properties(data, segm)
        tbl = props.to_table()
        orients = tbl['orientation'].to(u.deg)
        assert_allclose(orients[1].value, 50., rtol=1.e-5)
        assert_allclose(orients[2].value, -80., rtol=1.e-5)
        assert 88.3 < orients[0].value < 88.4
        assert 64. < orients[3].value < 64.2
