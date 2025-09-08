# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provide tools for making example WCS objects.
"""

import astropy.units as u
import numpy as np
from astropy import coordinates as coord
from astropy.modeling import models
from astropy.wcs import WCS

__all__ = ['make_gwcs', 'make_wcs']

__doctest_requires__ = {'make_gwcs': ['gwcs']}


def make_wcs(shape, galactic=False):
    """
    Create a simple celestial `~astropy.wcs.WCS` object in either the
    ICRS or Galactic coordinate frame.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the 2D array to be used with the output
        `~astropy.wcs.WCS` object.

    galactic : bool, optional
        If `True`, then the output WCS will be in the Galactic
        coordinate frame. If `False` (default), then the output WCS will
        be in the ICRS coordinate frame.

    Returns
    -------
    wcs : `astropy.wcs.WCS` object
        The world coordinate system (WCS) transformation.

    See Also
    --------
    make_gwcs

    Notes
    -----
    The `make_gwcs` function returns an equivalent WCS transformation to
    this one, but as a `gwcs.wcs.WCS` object.

    Examples
    --------
    >>> from photutils.datasets import make_wcs
    >>> shape = (100, 100)
    >>> wcs = make_wcs(shape)
    >>> print(wcs.wcs.crpix)  # doctest: +FLOAT_CMP
    [50. 50.]
    >>> print(wcs.wcs.crval)  # doctest: +FLOAT_CMP
    [197.8925      -1.36555556]
    >>> skycoord = wcs.pixel_to_world(42, 57)
    >>> print(skycoord)  # doctest: +FLOAT_CMP
    <SkyCoord (ICRS): (ra, dec) in deg
        (197.89278975, -1.36561284)>
    """
    wcs = WCS(naxis=2)
    rho = np.pi / 3.0
    scale = 0.1 / 3600.0  # 0.1 arcsec/pixel in deg/pix

    wcs.pixel_shape = shape
    wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]  # 1-indexed (x, y)
    wcs.wcs.crval = [197.8925, -1.36555556]
    wcs.wcs.cunit = ['deg', 'deg']
    wcs.wcs.cd = [[-scale * np.cos(rho), scale * np.sin(rho)],
                  [scale * np.sin(rho), scale * np.cos(rho)]]
    if not galactic:
        wcs.wcs.radesys = 'ICRS'
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    else:
        wcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']

    return wcs


def make_gwcs(shape, galactic=False):
    """
    Create a simple celestial gWCS object in the ICRS coordinate frame.

    This function requires the `gwcs
    <https://github.com/spacetelescope/gwcs>`_ package.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the 2D array to be used with the output
        `~gwcs.wcs.WCS` object.

    galactic : bool, optional
        If `True`, then the output WCS will be in the Galactic
        coordinate frame. If `False` (default), then the output WCS will
        be in the ICRS coordinate frame.

    Returns
    -------
    wcs : `gwcs.wcs.WCS` object
        The generalized world coordinate system (WCS) transformation.

    See Also
    --------
    make_wcs

    Notes
    -----
    The `make_wcs` function returns an equivalent WCS transformation to
    this one, but as an `astropy.wcs.WCS` object.

    Examples
    --------
    >>> from photutils.datasets import make_gwcs
    >>> shape = (100, 100)
    >>> gwcs = make_gwcs(shape)
    >>> print(gwcs)
      From      Transform
    -------- ----------------
    detector linear_transform
        icrs             None
    >>> skycoord = gwcs.pixel_to_world(42, 57)
    >>> print(skycoord)  # doctest: +FLOAT_CMP
    <SkyCoord (ICRS): (ra, dec) in deg
        (197.89278975, -1.36561284)>
    """
    from gwcs import coordinate_frames as cf
    from gwcs import wcs as gwcs_wcs

    rho = np.pi / 3.0
    scale = 0.1 / 3600.0  # 0.1 arcsec/pixel in deg/pix

    shift_by_crpix = (models.Shift((-shape[1] / 2) + 1)
                      & models.Shift((-shape[0] / 2) + 1))

    cd_matrix = np.array([[-scale * np.cos(rho), scale * np.sin(rho)],
                          [scale * np.sin(rho), scale * np.cos(rho)]])

    rotation = models.AffineTransformation2D(cd_matrix, translation=[0, 0])
    rotation.inverse = models.AffineTransformation2D(
        np.linalg.inv(cd_matrix), translation=[0, 0])

    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(197.8925, -1.36555556,
                                                       180.0)

    det2sky = shift_by_crpix | rotation | tan | celestial_rotation
    det2sky.name = 'linear_transform'

    detector_frame = cf.Frame2D(name='detector', axes_names=('x', 'y'),
                                unit=(u.pix, u.pix))

    if galactic:
        sky_frame = cf.CelestialFrame(reference_frame=coord.Galactic(),
                                      name='galactic', unit=(u.deg, u.deg))
    else:
        sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(),
                                      name='icrs', unit=(u.deg, u.deg))

    pipeline = [(detector_frame, det2sky), (sky_frame, None)]

    return gwcs_wcs.WCS(pipeline)
