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


def _sanitize_pixel_positions(positions):

    if isinstance(positions, u.Quantity):
        if positions.unit is u.pixel:
            positions = np.atleast_2d(positions.value)
        else:
            raise u.UnitsError("positions should be in pixel units")
    elif isinstance(positions, (list, tuple, np.ndarray)):
        positions = np.atleast_2d(positions)
        if positions.shape[1] != 2:
            if positions.shape[0] == 2:
                positions = np.transpose(positions)
            else:
                raise TypeError("List or array of (x, y) pixel coordinates "
                                "is expected got '{0}'.".format(positions))
    elif isinstance(positions, zip):
        # This is needed for zip to work seamlessly in Python 3
        positions = np.atleast_2d(list(positions))
    else:
        raise TypeError("List or array of (x, y) pixel coordinates "
                        "is expected got '{0}'.".format(positions))

    if positions.ndim > 2:
        raise ValueError('{0}-d position array not supported. Only 2-d '
                         'arrays supported.'.format(positions.ndim))

    return positions


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


def _get_phot_extents(data, positions, extents):
    """
    Get the photometry extents and check if the apertures is fully out of data.

    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.

    Returns
    -------
    extents : dict
        The ``extents`` dictionary contains 3 elements:

        * ``'ood_filter'``
            A boolean array with `True` elements where the aperture is
            falling out of the data region.
        * ``'pixel_extent'``
            x_min, x_max, y_min, y_max : Refined extent of apertures with
            data coverage.
        * ``'phot_extent'``
            x_pmin, x_pmax, y_pmin, y_pmax: Extent centered to the 0, 0
            positions as required by the `~photutils.geometry` functions.
    """

    # Check if an aperture is fully out of data
    ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                               extents[:, 1] <= 0)
    np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                  out=ood_filter)
    np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

    # TODO check whether it makes sense to have negative pixel
    # coordinate, one could imagine a stackes image where the reference
    # was a bit offset from some of the images? Or in those cases just
    # give Skycoord to the Aperture and it should deal with the
    # conversion for the actual case?
    x_min = np.maximum(extents[:, 0], 0)
    x_max = np.minimum(extents[:, 1], data.shape[1])
    y_min = np.maximum(extents[:, 2], 0)
    y_max = np.minimum(extents[:, 3], data.shape[0])

    x_pmin = x_min - positions[:, 0] - 0.5
    x_pmax = x_max - positions[:, 0] - 0.5
    y_pmin = y_min - positions[:, 1] - 0.5
    y_pmax = y_max - positions[:, 1] - 0.5

    # TODO: check whether any pixel is nan in data[y_min[i]:y_max[i],
    # x_min[i]:x_max[i])), if yes return something valid rather than nan

    pixel_extent = [x_min, x_max, y_min, y_max]
    phot_extent = [x_pmin, x_pmax, y_pmin, y_pmax]

    return ood_filter, pixel_extent, phot_extent


def _calc_aperture_var(data, fraction, error, flux, xmin, xmax, ymin, ymax,
                       pixelwise_error):

    if isinstance(error, u.Quantity):
        zero_variance = 0 * error.unit**2
    else:
        zero_variance = 0

    if pixelwise_error:
        subvariance = error[ymin:ymax, xmin:xmax] ** 2

        # Make sure variance is > 0
        fluxvar = np.maximum(np.sum(subvariance * fraction), zero_variance)
    else:
        local_error = error[int((ymin + ymax) / 2 + 0.5),
                            int((xmin + xmax) / 2 + 0.5)]
        fluxvar = np.maximum(local_error ** 2 * np.sum(fraction),
                             zero_variance)

    return fluxvar


class _ABCMetaAndInheritDocstrings(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(_ABCMetaAndInheritDocstrings)
class Aperture(object):
    """
    Abstract base class for all apertures.
    """


class SkyAperture(Aperture):
    """
    Abstract base class for 2-d apertures defined in celestial coordinates.
    """


class PixelAperture(Aperture):
    """
    Abstract base class for 2-d apertures defined in pixel coordinates.

    Derived classes should contain whatever internal data is needed to
    define the aperture, and provide the method `do_photometry` (and
    optionally, ``area``).
    """

    def _prepare_plot(self, origin=(0, 0), source_id=None, ax=None,
                      fill=False, **kwargs):
        """
        Prepare to plot the aperture(s) on a matplotlib Axes instance.

        Parameters
        ----------
        origin : array-like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        source_id : int or array of int, optional
            The source ID(s) of the aperture(s) to plot.

        ax : `matplotlib.axes.Axes` instance, optional
            If `None`, then the current ``Axes`` instance is used.

        fill : bool, optional
            Set whether to fill the aperture patch.  The default is
            `False`.

        kwargs
            Any keyword arguments accepted by `matplotlib.patches.Patch`.

        Returns
        -------
        plot_positions : `~numpy.ndarray`
            The positions of the apertures to plot, after any
            ``source_id`` slicing and origin shift.

        ax : `matplotlib.axes.Axes` instance, optional
            The `matplotlib.axes.Axes` on which to plot.

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
        if source_id is not None:
            plot_positions = plot_positions[np.atleast_1d(source_id)]

        plot_positions[:, 0] -= origin[0]
        plot_positions[:, 1] -= origin[1]

        return plot_positions, ax, kwargs

    @abc.abstractmethod
    def plot(self, origin=(0, 0), source_id=None, ax=None, fill=False,
             **kwargs):
        """
        Plot the aperture(s) on a matplotlib Axes instance.

        Parameters
        ----------
        origin : array-like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        source_id : int or array of int, optional
            The source ID(s) of the aperture(s) to plot.

        ax : `matplotlib.axes.Axes` instance, optional
            If `None`, then the current ``Axes`` instance is used.

        fill : bool, optional
            Set whether to fill the aperture patch.  The default is
            `False`.

        kwargs
            Any keyword arguments accepted by `matplotlib.patches.Patch`.
        """

    @abc.abstractmethod
    def do_photometry(self, data, error=None, pixelwise_error=True,
                      method='exact', subpixels=5):
        """Sum flux within aperture(s).

        Parameters
        ----------
        data : array_like
            The 2-d array on which to perform photometry.
        error : array_like, optional
            Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
            ``error`` has to have the same shape as ``data``.
        pixelwise_error : bool, optional
            If `True`, assume ``error`` varies significantly within an
            aperture and sum the contribution from each pixel. If
            `False`, assume ``error`` does not vary significantly within
            an aperture and use the single value of ``error`` at the
            center of each aperture as the value for the entire
            aperture.  Default is `True`.
        method : str, optional
            Method to use for determining overlap between the aperture
            and pixels.  Options include ['center', 'subpixel',
            'exact'], but not all options are available for all types of
            apertures. More precise methods will generally be slower.

            * ``'center'``
                A pixel is considered to be entirely in or out of the
                aperture depending on whether its center is in or out of
                the aperture.

            * ``'subpixel'``
                A pixel is divided into subpixels and the center of each
                subpixel is tested (as above). With ``subpixels`` set to
                1, this method is equivalent to 'center'. Note that for
                subpixel sampling, the input array is only resampled
                once for each object.

            * ``'exact'`` (default)
                The exact overlap between the aperture and each pixel is
                calculated.
        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor (in
            each dimension). That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Sum of the values withint the aperture(s). Unit is kept to be the
            unit of the input ``data``.

        fluxvar :  `~astropy.units.Quantity`
            Corresponting uncertainity in the ``flux`` values. Returned only
            if input ``error`` is not `None`.
        """

    @abc.abstractmethod
    def get_fractions(self, data, method='exact', subpixels=5):
        """Weight of pixels in data, within aperture(s).

        Parameters
        ----------
        data : array_like
            The 2-d array on which to perform photometry.
        method : str, optional
            Method to use for determining overlap between the aperture
            and pixels.  Options include ['center', 'subpixel',
            'exact'], but not all options are available for all types of
            apertures. More precise methods will generally be slower.

            * ``'center'``
                A pixel is considered to be entirely in or out of the
                aperture depending on whether its center is in or out of
                the aperture.

            * ``'subpixel'``
                A pixel is divided into subpixels and the center of each
                subpixel is tested (as above). With ``subpixels`` set to
                1, this method is equivalent to 'center'. Note that for
                subpixel sampling, the input array is only resampled
                once for each object.

            * ``'exact'`` (default)
                The exact overlap between the aperture and each pixel is
                calculated.
        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor (in
            each dimension). That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        fraction : `numpy.array`
            Array with the same shape as ``data``. Each element is the
            fraction of the corresponding ``data`` pixel that falls
            within the aperture.
        """

    def area():
        """
        Area of aperture.

        Returns
        -------
        area : float
            Area of aperture.
        """


def _prepare_photometry_input(data, unit, wcs, mask, error, pixelwise_error):
    """
    Parse photometry input.

    Photometry routines accept a wide range of inputs, e.g. ``data``
    could be (among others)  a numpy array, or a fits HDU.
    This requires some parsing and bookkeping to ensure that all inputs
    are complete and consistent.
    For example, the data could carry a unit and the wcs itself, so we need to
    check that it is consistent with the unit and wcs given as explicit
    parameters.

    Note that this function is meant to be used in addition to, not instead
    of, the `~astropy.nddata.support_nddata` decorator, i. e. ``data`` is
    never an `~astropy.nddata.NDData` object, because that will be split up
    in data, wcs, mask, ... keywords by the decorator already.

    See `~photutils.aperture_photometry` for a description of all
    possible input values.

    Returns
    -------
    data : `~astropy.units.Quantity` instance
    wcs_transformation : `~astropy.wcs.WCS` instance or None
    mask : np.array or None
    error : `~astropy.units.Quantity` instance or None
    """

    dataunit = None
    datamask = None
    wcs_transformation = wcs

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
            dataunit = header['BUNIT']

    if wcs_transformation is None:
        try:
            wcs_transformation = WCS(header)
        except:
            # data was not fits so header is not defined or header is invalid
            # Let the calling application raise an error is it needs a WCS.
            pass

    if hasattr(data, 'unit'):
        dataunit = data.unit

    if unit is not None and dataunit is not None:
        dataunit = u.Unit(dataunit, parse_strict='warn')
        unit = u.Unit(unit, parse_strict='warn')

        if not isinstance(unit, u.UnrecognizedUnit):
            data = u.Quantity(data, unit=unit, copy=False)
            if not isinstance(dataunit, u.UnrecognizedUnit):
                if unit != dataunit:
                    warnings.warn('Unit of input data ({0}) and unit given '
                                  'by unit argument ({1}) are not identical.'
                                  .format(dataunit, unit))
        else:
            if not isinstance(dataunit, u.UnrecognizedUnit):
                data = u.Quantity(data, unit=dataunit, copy=False)
            else:
                warnings.warn('Neither the unit of the input data ({0}), nor '
                              'the unit given by the unit argument ({1}) is '
                              'parseable as a valid unit'
                              .format(dataunit, unit))

    elif unit is None:
        if dataunit is not None:
            dataunit = u.Unit(dataunit, parse_strict='warn')
            data = u.Quantity(data, unit=dataunit, copy=False)
        else:
            data = u.Quantity(data, copy=False)
    else:
        unit = u.Unit(unit, parse_strict='warn')
        data = u.Quantity(data, unit=unit, copy=False)

    # Check input array type and dimension.
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))

    # Deal with the mask if it exists
    if mask is not None or datamask is not None:
        if mask is None:
            mask = datamask
        else:
            mask = np.asarray(mask)
            if np.iscomplexobj(mask):
                raise TypeError('Complex type not supported')
            if mask.shape != data.shape:
                raise ValueError('Shapes of mask array and data array '
                                 'must match')

            if datamask is not None:
                # combine the masks
                mask = np.logical_or(mask, datamask)

    # Check whether we really need to calculate pixelwise errors, even if
    # requested.  If error is not an array, then it's not needed.
    if (error is None) or (np.isscalar(error)):
        pixelwise_error = False

    # Check error shape.
    if error is not None:
        if isinstance(error, u.Quantity):
            if np.isscalar(error.value):
                error = u.Quantity(np.broadcast_arrays(error, data),
                                   unit=error.unit)[0]
        elif np.isscalar(error):
            error = u.Quantity(np.broadcast_arrays(error, data),
                               unit=data.unit)[0]
        else:
            error = u.Quantity(error, unit=data.unit, copy=False)

        if error.shape != data.shape:
            raise ValueError('shapes of error array and data array must'
                             ' match')

    return data, wcs_transformation, mask, error, pixelwise_error


@support_nddata
def aperture_photometry(data, apertures, unit=None, wcs=None, error=None,
                        mask=None, method='exact', subpixels=5,
                        pixelwise_error=True):
    """
    Sum flux within an aperture at the given position(s).

    Parameters
    ----------
    data : array_like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
        The 2-d array on which to perform photometry. ``data`` should be
        background-subtracted.  Units are used during the photometry,
        either provided along with the data array, or stored in the
        header keyword ``'BUNIT'``.
    apertures : `~photutils.Aperture` instance
        The apertures to use for the photometry.
    unit : `~astropy.units.UnitBase` instance, str
        An object that represents the unit associated with ``data``.  Must
        be an `~astropy.units.UnitBase` object or a string parseable by the
        :mod:`~astropy.units` package. It overrides the ``data`` unit from
        the ``'BUNIT'`` header keyword and issues a warning if
        different. However an error is raised if ``data`` as an array
        already has a different unit.
    wcs : `~astropy.wcs.WCS`, optional
        Use this as the wcs transformation. It overrides any wcs transformation
        passed along with ``data`` either in the header or in an attribute.
    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.
    mask : array_like (bool), optional
        Mask to apply to the data.  Masked pixels are excluded/ignored.
    method : str, optional
        Method to use for determining overlap between the aperture and pixels.
        Options include ['center', 'subpixel', 'exact'], but not all options
        are available for all types of apertures. More precise methods will
        generally be slower.

        * ``'center'``
            A pixel is considered to be entirely in or out of the aperture
            depending on whether its center is in or out of the aperture.
        * ``'subpixel'``
            A pixel is divided into subpixels and the center of each
            subpixel is tested (as above). With ``subpixels`` set to 1, this
            method is equivalent to 'center'. Note that for subpixel
            sampling, the input array is only resampled once for each
            object.
        * ``'exact'`` (default)
            The exact overlap between the aperture and each pixel is
            calculated.
    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor (in
        each dimension). That is, each pixel is divided into
        ``subpixels ** 2`` subpixels.
    pixelwise_error : bool, optional
        If `True`, assume ``error`` varies significantly within an
        aperture and sum the contribution from each pixel. If `False`,
        assume ``error`` does not vary significantly within an aperture
        and use the single value of ``error`` at the center of each
        aperture as the value for the entire aperture.  Default is
        `True`.

    Returns
    -------
    phot_table : `~astropy.table.QTable`
        A table of the photometry with the following columns:

        * ``'aperture_sum'``: Sum of the values within the aperture.
        * ``'aperture_sum_err'``: Corresponding uncertainty in
          ``'aperture_sum'`` values.  Returned only if input ``error``
          is not `None`.
        * ``'xcenter'``, ``'ycenter'``: x and y pixel coordinates of the
          center of the apertures. Unit is pixel.
        * ``'xcenter_input'``, ``'ycenter_input'``: input x and y
          coordinates as they were given in the input ``positions``
          parameter.

        The metadata of the table stores the version numbers of both astropy
        and photutils, as well as the calling arguments.

    Notes
    -----
    This function is decorated with `~astropy.nddata.support_nddata` and
    thus supports `~astropy.nddata.NDData` objects as input.
    """

    data, wcs, mask, error, pixelwise_error = \
        _prepare_photometry_input(data, unit, wcs, mask, error,
                                  pixelwise_error)

    if mask is not None:
        data = copy.deepcopy(data)    # do not modify input data
        data[mask] = 0

        if error is not None:
            error = copy.deepcopy(error)    # do not modify input data
            error[mask] = 0.

    if method == 'subpixel':
        if (int(subpixels) != subpixels) or (subpixels <= 0):
            raise ValueError('subpixels must be a positive integer.')

    skyaper = False
    if isinstance(apertures, SkyAperture):
        if wcs is None:
            raise ValueError('A WCS transform must be defined by the input '
                             'data or the wcs keyword.')
        skyaper = True
        skycoord_pos = apertures.positions
        apertures = apertures.to_pixel(wcs)

    xypos_pixel = np.transpose(apertures.positions) * u.pixel

    photometry_result = apertures.do_photometry(
        data, method=method, subpixels=subpixels, error=error,
        pixelwise_error=pixelwise_error)
    if error is None:
        aper_sum = photometry_result
    else:
        aper_sum, aper_err = photometry_result

    calling_args = ('method={0}, subpixels={1}, pixelwise_error={2}'
                    .format(method, subpixels, pixelwise_error))
    meta = OrderedDict()
    meta['name'] = 'Aperture photometry results'
    meta['version'] = _get_version_info()
    meta['aperture_photometry_args'] = calling_args

    tbl = QTable(meta=meta)
    tbl['xcenter'] = xypos_pixel[0]
    tbl['ycenter'] = xypos_pixel[1]
    if skyaper:
        if skycoord_pos.isscalar:
            tbl['input_center'] = (skycoord_pos,)
        else:
            tbl['input_center'] = skycoord_pos
    tbl['aperture_sum'] = aper_sum
    if error is not None:
        tbl['aperture_sum_err'] = aper_err

    return tbl
