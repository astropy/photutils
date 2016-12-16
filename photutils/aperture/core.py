# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import abc
import copy
import warnings
from collections import OrderedDict

import numpy as np
from astropy.extern import six
from astropy.io import fits
from astropy.nddata import support_nddata
from astropy.table import QTable
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.misc import InheritDocstrings
from astropy.wcs import WCS


__all__ = ['Aperture', 'SkyAperture', 'PixelAperture', 'aperture_photometry']


def _get_version_info():
    from astropy import __version__
    astropy_version = __version__

    from photutils import __version__
    photutils_version = __version__

    return 'astropy: {0}, photutils: {1}'.format(astropy_version,
                                                 photutils_version)


class _ABCMetaAndInheritDocstrings(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(_ABCMetaAndInheritDocstrings)
class Aperture(object):
    """
    Abstract base class for all apertures.
    """

    def __len__(self):
        if isinstance(self, SkyAperture) and self.positions.isscalar:
            return 1
        return len(self.positions)


class PixelAperture(Aperture):
    """
    Abstract base class for apertures defined in pixel coordinates.
    """

    @staticmethod
    def _sanitize_positions(positions):
        if isinstance(positions, u.Quantity):
            if positions.unit is u.pixel:
                positions = np.atleast_2d(positions.value)
            else:
                raise u.UnitsError('positions should be in pixel units')
        elif isinstance(positions, (list, tuple, np.ndarray)):
            positions = np.atleast_2d(positions)
            if positions.shape[1] != 2:
                if positions.shape[0] == 2:
                    positions = np.transpose(positions)
                else:
                    raise TypeError('List or array of (x, y) pixel '
                                    'coordinates is expected, got "{0}".'
                                    .format(positions))
        elif isinstance(positions, zip):
            # This is needed for zip to work seamlessly in Python 3
            positions = np.atleast_2d(list(positions))
        else:
            raise TypeError('List or array of (x, y) pixel coordinates '
                            'is expected, got "{0}".'.format(positions))

        if positions.ndim > 2:
            raise ValueError('{0}D position array is not supported. Only 2D '
                             'arrays are supported.'.format(positions.ndim))

        return positions

    @staticmethod
    def _translate_mask_mode(mode, subpixels, rectangle=False):
        if mode not in ('center', 'subpixel', 'exact'):
            raise ValueError('Invalid mask mode: {0}'.format(mode))

        if rectangle and mode == 'exact':
            warnings.warn('The "exact" method is not yet implemented for '
                          'rectangular apertures -- using "subpixel" method '
                          'with "subpixels=32"', AstropyUserWarning)
            mode = 'subpixel'
            subpixels = 32

        if mode == 'subpixels':
            if not isinstance(subpixels, int) or subpixels <= 0:
                raise ValueError('subpixels must be a strictly positive '
                                 'integer'.format(subpixels))

        if mode == 'center':
            use_exact = 0
            subpixels = 1
        elif mode == 'subpixel':
            use_exact = 0
        elif mode == 'exact':
            use_exact = 1
            subpixels = 1

        return use_exact, subpixels

    @abc.abstractproperty
    def bounding_boxes(self):
        """
        A list of minimal bounding boxes (`~photutils.BoundingBox`), one
        for each position, for the aperture.
        """

        raise NotImplementedError('Needs to be implemented in a '
                                  'PixelAperture subclass.')

    @property
    def _centered_edges(self):
        """
        A list of ``(xmin, xmax, ymin, ymax)`` tuples, one for each
        position, of the pixel edges after recentering the aperture at
        the origin.

        These pixel edges are used by the low-level `photutils.geometry`
        functions.
        """

        edges = []
        for position, bbox in zip(self.positions, self.bounding_boxes):
            xmin = bbox.ixmin - 0.5 - position[0]
            xmax = bbox.ixmax - 0.5 - position[0]
            ymin = bbox.iymin - 0.5 - position[1]
            ymax = bbox.iymax - 0.5 - position[1]
            edges.append((xmin, xmax, ymin, ymax))

        return edges

    def area(self):
        """
        Return the exact area of the aperture shape.

        Returns
        -------
        area : float
            The aperture area.
        """

        raise NotImplementedError('Needs to be implemented in a '
                                  'PixelAperture subclass.')

    def mask_area(self, method='exact', subpixels=5):
        """
        Return the area of the aperture(s) mask.

        For ``method`` other than ``'exact'``, this area will be less
        than the exact analytical area (e.g. the ``area`` method).  Note
        that for these methods, the values can also differ because of
        fractional pixel positions.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        area : float
            A list of the mask area of the aperture(s).
        """

        mask = self.to_mask(method=method, subpixels=subpixels)
        return [np.sum(m.data) for m in mask]

    @abc.abstractmethod
    def to_mask(self, method='exact', subpixels=5):
        """
        Return a list of `~photutils.ApertureMask` objects, one for each
        aperture position.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        mask : list of `~photutils.ApertureMask`
            A list of aperture mask objects.
        """

        raise NotImplementedError('Needs to be implemented in a '
                                  'PixelAperture subclass.')

    @staticmethod
    def _prepare_photometry_output(_list, unit=None):
        if len(_list) == 0:   # if error is not input
            return _list

        if unit is not None:
            unit = u.Unit(unit, parse_strict='warn')
            if isinstance(unit, u.UnrecognizedUnit):
                warnings.warn('The input unit is not parseable as a valid '
                              'unit.', AstropyUserWarning)
                unit = None

        if isinstance(_list[0], u.Quantity):
            # list of Quantity -> Quantity array
            output = u.Quantity(_list)

            if unit is not None:
                if output.unit != unit:
                    warnings.warn('The input unit does not agree with the '
                                  'data and/or error unit.',
                                  AstropyUserWarning)
        else:
            if unit is not None:
                output = u.Quantity(_list, unit=unit)
            else:
                output = np.array(_list)

        return output

    def do_photometry(self, data, error=None, pixelwise_error=True,
                      mask=None, method='exact', subpixels=5, unit=None):
        """
        Perform aperture photometry on the input data.

        Parameters
        ----------
        data : array_like or `~astropy.units.Quantity` instance
            The 2D array on which to perform photometry.  ``data``
            should be background subtracted.

        error : array_like or `~astropy.units.Quantity`, optional
            The pixel-wise Gaussian 1-sigma errors of the input
            ``data``.  ``error`` is assumed to include *all* sources of
            error, including the Poisson error of the sources (see
            `~photutils.utils.calc_total_error`) .  ``error`` must have
            the same shape as the input ``data``.

        pixelwise_error : bool, optional
            If `True` (default), the photometric error is calculated
            using the ``error`` values from each pixel within the
            aperture.  If `False`, the ``error`` value at the center of
            the aperture is used for the entire aperture.

        mask : array_like (bool), optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked.  Masked data are excluded from all calculations.

        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        unit : `~astropy.units.UnitBase` object or str, optional
            An object that represents the unit associated with the input
            ``data`` and ``error`` arrays.  Must be a
            `~astropy.units.UnitBase` object or a string parseable by
            the :mod:`~astropy.units` package.  If ``data`` or ``error``
            already have a different unit, the input ``unit`` will not
            be used and a warning will be raised.

        Returns
        -------
        aperture_sums : `~numpy.ndarray` or `~astropy.units.Quantity`
            The sums within each aperture.

        aperture_sum_errs : `~numpy.ndarray` or `~astropy.units.Quantity`
            The errors on the sums within each aperture.
        """

        data = np.asanyarray(data)

        if mask is not None:
            mask = np.asanyarray(mask)

            data = copy.deepcopy(data)    # do not modify input data
            data[mask] = 0

            if error is not None:
                # do not modify input data
                error = copy.deepcopy(np.asanyarray(error))
                error[mask] = 0.

        aperture_sums = []
        aperture_sum_errs = []
        for mask in self.to_mask(method=method, subpixels=subpixels):
            data_cutout = mask.cutout(data)

            if data_cutout is None:
                aperture_sums.append(np.nan)
            else:
                aperture_sums.append(np.sum(data_cutout * mask.data))

            if error is not None:
                error_cutout = mask.cutout(error)

                if error_cutout is None:
                    aperture_sum_errs.append(np.nan)
                else:
                    if pixelwise_error:
                        aperture_var = np.sum(error_cutout ** 2 * mask.data)
                    else:
                        # use central value (shifted for partial overlap)
                        _, slc_sm = mask._overlap_slices(error.shape)
                        yidx = int((slc_sm[0].start + slc_sm[0].stop - 1) /
                                   2. + 0.5)
                        xidx = int((slc_sm[1].start + slc_sm[1].stop - 1) /
                                   2. + 0.5)
                        error_value = error_cutout[yidx, xidx]

                        aperture_var = np.sum(error_value ** 2 *
                                              np.sum(mask.data))

                    aperture_sum_errs.append(np.sqrt(aperture_var))

        # handle Quantity objects and input units
        aperture_sums = self._prepare_photometry_output(aperture_sums,
                                                        unit=unit)
        aperture_sum_errs = self._prepare_photometry_output(aperture_sum_errs,
                                                            unit=unit)

        return aperture_sums, aperture_sum_errs

    @staticmethod
    def _make_annulus_path(patch_inner, patch_outer):
        import matplotlib.path as mpath

        verts_inner = patch_inner.get_verts()
        verts_outer = patch_outer.get_verts()
        codes_inner = (np.ones(len(verts_inner), dtype=mpath.Path.code_type) *
                       mpath.Path.LINETO)
        codes_inner[0] = mpath.Path.MOVETO
        codes_outer = (np.ones(len(verts_outer), dtype=mpath.Path.code_type) *
                       mpath.Path.LINETO)
        codes_outer[0] = mpath.Path.MOVETO
        codes = np.concatenate((codes_inner, codes_outer))
        verts = np.concatenate((verts_inner, verts_outer[::-1]))

        return mpath.Path(verts, codes)

    def _prepare_plot(self, origin=(0, 0), indices=None, ax=None,
                      fill=False, **kwargs):
        """
        Prepare to plot the aperture(s) on a matplotlib
        `~matplotlib.axes.Axes` instance.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        indices : int or array of int, optional
            The indices of the aperture(s) to plot.

        ax : `matplotlib.axes.Axes` instance, optional
            If `None`, then the current `~matplotlib.axes.Axes` instance
            is used.

        fill : bool, optional
            Set whether to fill the aperture patch.  The default is
            `False`.

        kwargs
            Any keyword arguments accepted by `matplotlib.patches.Patch`.

        Returns
        -------
        plot_positions : `~numpy.ndarray`
            The positions of the apertures to plot, after any
            ``indices`` slicing and ``origin`` shift.

        ax : `matplotlib.axes.Axes` instance, optional
            The `~matplotlib.axes.Axes` on which to plot.

        kwargs
            Any keyword arguments accepted by `matplotlib.patches.Patch`.
        """

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        # This is necessary because the `matplotlib.patches.Patch` default
        # is ``fill=True``.  Here we make the default ``fill=False``.
        kwargs['fill'] = fill

        plot_positions = copy.deepcopy(self.positions)
        if indices is not None:
            plot_positions = plot_positions[np.atleast_1d(indices)]

        plot_positions[:, 0] -= origin[0]
        plot_positions[:, 1] -= origin[1]

        return plot_positions, ax, kwargs

    @abc.abstractmethod
    def plot(self, origin=(0, 0), indices=None, ax=None, fill=False,
             **kwargs):
        """
        Plot the aperture(s) on a matplotlib `~matplotlib.axes.Axes`
        instance.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        indices : int or array of int, optional
            The indices of the aperture(s) to plot.

        ax : `matplotlib.axes.Axes` instance, optional
            If `None`, then the current `~matplotlib.axes.Axes` instance
            is used.

        fill : bool, optional
            Set whether to fill the aperture patch.  The default is
            `False`.

        kwargs
            Any keyword arguments accepted by `matplotlib.patches.Patch`.
        """

        raise NotImplementedError('Needs to be implemented in a '
                                  'PixelAperture subclass.')


class SkyAperture(Aperture):
    """
    Abstract base class for all apertures defined in celestial
    coordinates.
    """

    @abc.abstractmethod
    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to a `PixelAperture` object in pixel
        coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS` object
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `PixelAperture` object
            A `PixelAperture` object.
        """

        raise NotImplementedError('Needs to be implemented in a '
                                  'SkyAperture subclass.')


def _prepare_photometry_input(data, error, pixelwise_error, mask, wcs, unit):
    """
    Parse the inputs to `aperture_photometry`.

    `aperture_photometry` accepts a wide range of inputs, e.g. ``data``
    could be a numpy array, a Quantity array, or a fits HDU.  This
    requires some parsing and validation to ensure that all inputs are
    complete and consistent.  For example, the data could carry a unit
    and the wcs itself, so we need to check that it is consistent with
    the unit and wcs given as input parameters.
    """

    if isinstance(data, fits.HDUList):
        for i in range(len(data)):
            if data[i].data is not None:
                warnings.warn("Input data is a HDUList object, photometry is "
                              "run only for the {0} HDU."
                              .format(i), AstropyUserWarning)
                data = data[i]
                break

    if isinstance(data, (fits.PrimaryHDU, fits.ImageHDU)):
        header = data.header
        data = data.data

        if 'BUNIT' in header:
            bunit = u.Unit(header['BUNIT'], parse_strict='warn')
            if isinstance(bunit, u.UnrecognizedUnit):
                warnings.warn('The BUNIT in the header of the input data is '
                              'not parseable as a valid unit.',
                              AstropyUserWarning)
            else:
                data = u.Quantity(data, unit=bunit)

    if wcs is None:
        try:
            wcs = WCS(header)
        except:
            # A valid WCS was not found in the header.  Let the calling
            # application raise an exception if it needs a WCS.
            pass

    data = np.asanyarray(data)
    if data.ndim != 2:
        raise ValueError('data must be a 2D array.')

    if unit is not None:
        unit = u.Unit(unit, parse_strict='warn')
        if isinstance(unit, u.UnrecognizedUnit):
            warnings.warn('The input unit is not parseable as a valid '
                          'unit.', AstropyUserWarning)
            unit = None

    if isinstance(data, u.Quantity):
        if unit is not None and data.unit != unit:
            warnings.warn('The input unit does not agree with the data '
                          'unit.', AstropyUserWarning)
    else:
        if unit is not None:
            data = u.Quantity(data, unit=unit)

    if error is not None:
        if isinstance(error, u.Quantity):
            if unit is not None and error.unit != unit:
                warnings.warn('The input unit does not agree with the error '
                              'unit.', AstropyUserWarning)

            if np.isscalar(error.value):
                error = u.Quantity(np.broadcast_arrays(error, data),
                                   unit=error.unit)[0]
                pixelwise_error = False
                warnings.warn('Because input error was a scalar, '
                              'pixelwise_error=False is being used.',
                              AstropyUserWarning)
        else:
            if np.isscalar(error):
                error = np.broadcast_arrays(error, data)[0]
                pixelwise_error = False
                warnings.warn('Because input error was a scalar, '
                              'pixelwise_error=False is being used.',
                              AstropyUserWarning)

            if unit is not None:
                error = u.Quantity(error, unit=unit)

            error = np.asanyarray(error)

        if error.shape != data.shape:
            raise ValueError('error and data must have the same shape.')

    if mask is not None:
        mask = np.asanyarray(mask)
        if mask.shape != data.shape:
            raise ValueError('mask and data must have the same shape.')

    return data, error, pixelwise_error, mask, wcs


@support_nddata
def aperture_photometry(data, apertures, error=None, pixelwise_error=True,
                        mask=None, method='exact', subpixels=5, unit=None,
                        wcs=None):
    """
    Perform aperture photometry on the input data by summing the flux
    within the given aperture(s).

    Parameters
    ----------
    data : array_like, `~astropy.units.Quantity`, `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.HDUList`
        The 2D array on which to perform photometry. ``data`` should be
        background-subtracted.  Units can be used during the photometry,
        either provided with the data (i.e. a `~astropy.units.Quantity`
        array) or the ``unit`` keyword.  If ``data`` is an
        `~astropy.io.fits.ImageHDU` or `~astropy.io.fits.HDUList`, the
        unit is determined from the ``'BUNIT'`` header keyword.

    apertures : `~photutils.Aperture`
        The aperture(s) to use for the photometry.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.

    pixelwise_error : bool, optional
        If `True` (default), the photometric error is calculated using
        the ``error`` values from each pixel within the aperture.  If
        `False`, the ``error`` value at the center of the aperture is
        used for the entire aperture.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    method : {'exact', 'center', 'subpixel'}, optional
        The method used to determine the overlap of the aperture on the
        pixel grid.  Not all options are available for all aperture
        types.  Note that the more precise methods are generally slower.
        The following methods are available:

            * ``'exact'`` (default):
                The the exact fractional overlap of the aperture and
                each pixel is calculated.  The returned mask will
                contain values between 0 and 1.

            * ``'center'``:
                A pixel is considered to be entirely in or out of the
                aperture depending on whether its center is in or out of
                the aperture.  The returned mask will contain values
                only of 0 (out) and 1 (in).

            * ``'subpixel'``:
                A pixel is divided into subpixels (see the ``subpixels``
                keyword), each of which are considered to be entirely in
                or out of the aperture depending on whether its center
                is in or out of the aperture.  If ``subpixels=1``, this
                method is equivalent to ``'center'``.  The returned mask
                will contain values between 0 and 1.

    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor in
        each dimension.  That is, each pixel is divided into ``subpixels
        ** 2`` subpixels.

    unit : `~astropy.units.UnitBase` object or str, optional
        An object that represents the unit associated with the input
        ``data`` and ``error`` arrays.  Must be a
        `~astropy.units.UnitBase` object or a string parseable by the
        :mod:`~astropy.units` package.  If ``data`` or ``error`` already
        have a different unit, the input ``unit`` will not be used and a
        warning will be raised.  If ``data`` is an
        `~astropy.io.fits.ImageHDU` or `~astropy.io.fits.HDUList`,
        ``unit`` will override the ``'BUNIT'`` header keyword.

    wcs : `~astropy.wcs.WCS`, optional
        The WCS transformation to use if the input ``apertures`` is a
        `SkyAperture` object.  If ``data`` is an
        `~astropy.io.fits.ImageHDU` or `~astropy.io.fits.HDUList`,
        ``wcs`` overrides any WCS transformation present in the header.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of the photometry with the following columns:

            * ``'id'``:
              The source ID.

            * ``'xcenter'``, ``'ycenter'``:
              The ``x`` and ``y`` pixel coordinates of the input
              aperture center(s).

            * ``'celestial_center'``:
              The celestial coordinates of the input aperture center(s).
              Returned only if the input ``apertures`` is a
              `SkyAperture` object.

            * ``'aperture_sum'``:
              The sum of the values within the aperture.

            * ``'aperture_sum_err'``:
              The corresponding uncertainty in the ``'aperture_sum'``
              values.  Returned only if the input ``error`` is not
              `None`.

        The table metadata includes the Astropy and Photutils version
        numbers and the `aperture_photometry` calling arguments.

    Notes
    -----
    This function is decorated with `~astropy.nddata.support_nddata` and
    thus supports `~astropy.nddata.NDData` objects as input.
    """

    data, error, pixelwise_error, mask, wcs = \
        _prepare_photometry_input(data, error, pixelwise_error, mask, wcs,
                                  unit)

    if method == 'subpixel':
        if (int(subpixels) != subpixels) or (subpixels <= 0):
            raise ValueError('subpixels must be a positive integer.')

    apertures = np.atleast_1d(apertures)

    # convert sky to pixel apertures
    skyaper = False
    if isinstance(apertures[0], SkyAperture):
        if wcs is None:
            raise ValueError('A WCS transform must be defined by the input '
                             'data or the wcs keyword when using a '
                             'SkyAperture object.')

        skyaper = True
        skycoord_pos = apertures[0].positions

        pix_aper = [aper.to_pixel(wcs) for aper in apertures]
        apertures = pix_aper

    # do comparison in pixels to avoid comparing SkyCoord objects
    positions = apertures[0].positions
    for aper in apertures[1:]:
        if not np.array_equal(aper.positions, positions):
            raise ValueError('Input apertures must all have identical '
                             'positions.')

    calling_args = ("method='{0}', subpixels={1}, pixelwise_error={2}"
                    .format(method, subpixels, pixelwise_error))
    meta = OrderedDict()
    meta['name'] = 'Aperture photometry results'
    meta['version'] = _get_version_info()
    meta['aperture_photometry_args'] = calling_args

    tbl = QTable(meta=meta)
    tbl['id'] = np.arange(len(apertures[0]), dtype=int) + 1

    xypos_pixel = np.transpose(apertures[0].positions) * u.pixel
    tbl['xcenter'] = xypos_pixel[0]
    tbl['ycenter'] = xypos_pixel[1]

    if skyaper:
        if skycoord_pos.isscalar:
            tbl['celestial_center'] = (skycoord_pos,)
        else:
            tbl['celestial_center'] = skycoord_pos

    for i, aper in enumerate(apertures):
        aper_sum, aper_sum_err = aper.do_photometry(
            data, error=error, pixelwise_error=pixelwise_error, mask=mask,
            method=method, subpixels=subpixels)

        sum_key = 'aperture_sum'
        sum_err_key = 'aperture_sum_err'
        if len(apertures) > 1:
            sum_key += '_{}'.format(i)
            sum_err_key += '_{}'.format(i)

        tbl[sum_key] = aper_sum
        if error is not None:
            tbl[sum_err_key] = aper_sum_err

    return tbl
