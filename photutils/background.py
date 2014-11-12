# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.stats import sigma_clip
from astropy.utils import lazyproperty


__all__ = ['Background']


__doctest_requires__ = {('Background'): ['scipy']}


class Background(object):
    def __init__(self, data, box_shape, filter_shape, mask=None,
                 method='sextractor', sigclip_sigma=3., sigclip_iters=None):
        """
        filter_shape == (1, 1) -> no filtering
        """
        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask shape must match data shape')
        self.box_shape = box_shape
        self.filter_shape = filter_shape
        self.method = method
        self.sigclip_sigma = sigclip_sigma
        self.sigclip_iters = sigclip_iters
        self.yextra = data.shape[0] % box_shape[0]
        self.xextra = data.shape[1] % box_shape[1]
        self.data_shape_orig = data.shape
        if (self.yextra > 0) or (self.xextra > 0):
            self.padded = True
            self.data = self._pad_data(data, mask)
        else:
            self.padded = False
            self.data = np.ma.masked_array(data, mask=mask)
        self._sigclip_data()

    def _pad_data(self, data, mask=None):
        """
        Pad the ``data`` and ``mask`` on the right and top with zeros if
        necessary to have a integer number of background meshes of size
        ``box_shape``.
        """
        ypad, xpad = 0, 0
        if self.yextra > 0:
            ypad = self.box_shape[0] - self.yextra
        if self.xextra > 0:
            xpad = self.box_shape[1] - self.xextra
        pad_width = ((0, ypad), (0, xpad))
        mode = str('constant')
        padded_data = np.pad(data, pad_width, mode=mode,
                             constant_values=[np.nan])
        padded_mask = np.isnan(padded_data)
        if mask is not None:
            padded_mask = np.pad(np.logical_or(mask, padded_mask), pad_width,
                                 mode=mode, constant_values=[False])
        return np.ma.masked_array(padded_data, mask=padded_mask)

    def _sigclip_data(self):
        """
        Perform sigma clipping on the data in regions of size
        ``box_shape``.
        """
        ny, nx = self.data.shape
        ny_box, nx_box = self.box_shape
        y_nbins = ny / ny_box     # always integer because data were padded
        x_nbins = nx / nx_box     # always integer because data were padded
        data_rebin = np.ma.swapaxes(self.data.reshape(
            y_nbins, ny_box, x_nbins, nx_box), 1, 2).reshape(y_nbins, x_nbins,
                                                             ny_box * nx_box)
        self.data_sigclip = sigma_clip(
            data_rebin, sig=self.sigclip_sigma, axis=2,
            iters=self.sigclip_iters, cenfunc=np.ma.median, varfunc=np.ma.var)

    def _filter_meshes(self, mesh):
        """
        Apply a 2d median filter to the background meshes, including
        only pixels inside the image at the borders.
        """
        from scipy.ndimage import generic_filter
        return generic_filter(mesh, np.nanmedian, size=self.filter_size,
                              mode='constant', cval=np.nan)

    def _resize_meshes(self, mesh):
        """
        Resize the background meshes to the original data size using
        bicubic interpolation.
        """
        from scipy.interpolate import RectBivariateSpline
        ny, nx = mesh.shape
        bbox = [-0.5, ny + 0.5, -0.5, nx + 0.5]
        x = np.arange(nx)
        y = np.arange(ny)
        print(x, y, bbox)

        # xx = np.linspace(x.min(), x.max(), self.data.shape[1])
        # yy = np.linspace(y.min(), y.max(), self.data.shape[0])
        xx = np.linspace(x.min() - 0.5, x.max() + 0.5, self.data.shape[1])
        yy = np.linspace(y.min() - 0.5, y.max() + 0.5, self.data.shape[0])
        # xx = np.linspace(bbox[2], bbox[3], self.data.shape[1])
        # yy = np.linspace(bbox[0], bbox[1], self.data.shape[0])
        # return RectBivariateSpline(y, x, mesh, kx=3, ky=3, s=0,
        #                           bbox=bbox)(yy, xx)
        return RectBivariateSpline(y, x, mesh, kx=3, ky=3, s=0)(yy, xx)

    def _resize_meshes2(self, mesh):
        from scipy.misc import imresize
        return imresize(mesh, self.data.shape, interp='bicubic', mode='F')

    def _resize_meshes3(self, mesh):
        from scipy.ndimage.interpolation import zoom
        zoom_factor = self.box_shape
        return zoom(mesh, zoom_factor, order=3)

    def _resize_meshes4(self, mesh):
        from scipy.ndimage import map_coordinates

        # sry = np.linspace(-0.48, 5.5-0.02, 300.)
        # srx = np.linspace(-0.48, 9.5-0.02, 500.)
        # sry = np.linspace(0, 6., 300.)
        # srx = np.linspace(0, 10., 500.)
        # sry = np.linspace(0.5, 5.0, 300.)
        # srx = np.linspace(0.5, 9.0, 500.)

        m1 = 0.  # minus_one
        halfx = -0.5 + (mesh.shape[1] / self.data.shape[1])
        halfy = -0.5 + (mesh.shape[0] / self.data.shape[0])
        y_zoom = (mesh.shape[0] - m1) / (self.data.shape[0] - m1)
        x_zoom = (mesh.shape[1] - m1) / (self.data.shape[1] - m1)
        sry = (y_zoom * np.arange(self.data.shape[0])) + halfy
        srx = (x_zoom * np.arange(self.data.shape[1])) + halfx
        yy, xx = np.meshgrid(sry, srx)
        coords = np.array([yy.T, xx.T])
        # return map_coordinates(mesh, coords, mode='reflect')
        return map_coordinates(mesh, coords)

    @lazyproperty
    def background_mesh(self):
        if self.method == 'mean':
            bkg_mesh = np.ma.mean(self.data_sigclip, axis=2)
        elif self.method == 'median':
            bkg_mesh = np.ma.median(self.data_sigclip, axis=2)
        elif self.method == 'sextractor':
            bkg_mesh = (2.5 * np.ma.median(self.data_sigclip, axis=2) -
                        1.5 * np.ma.mean(self.data_sigclip, axis=2))
        elif self.method == 'mode_estimate':
            bkg_mesh = (3. * np.ma.median(self.data_sigclip, axis=2) -
                        2. * np.ma.mean(self.data_sigclip, axis=2))
        else:
            raise ValueError('method "{0}" is not '
                             'defined'.format(self.method))
        if self.filter_shape != (1, 1):
            return self._filter_meshes(self, bkg_mesh)
        else:
            return bkg_mesh

    @lazyproperty
    def background_rms_mesh(self):
        bkgrms_mesh = np.ma.std(self.data_sigclip, axis=2)
        if self.filter_shape != (1, 1):
            return self._filter_meshes(self, bkgrms_mesh)
        else:
            return bkgrms_mesh

    @lazyproperty
    def background(self):
        if self.padded:
            y0 = self.data_shape_orig[0]
            x0 = self.data_shape_orig[1]
            return self._resize_meshes(self.background_mesh)[0:y0, 0:x0]
        else:
            return self._resize_meshes(self.background_mesh)

    @lazyproperty
    def background_rms(self):
        if self.padded:
            return self._resize_meshes(self.background_rms_mesh)[0:y0, 0:x0]
        else:
            return self._resize_meshes(self.background_rms_mesh)
