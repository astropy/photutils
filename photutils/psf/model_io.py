# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for reading and writing PSF models.
"""

import io
import itertools
import os
import re
import warnings

import numpy as np
from astropy.io import fits, registry
from astropy.io.fits.verify import VerifyWarning
from astropy.nddata import NDData, reshape_as_blocks

from photutils.utils._deprecation import deprecated_positional_kwargs

__all__ = ['GriddedPSFModelRead', 'stdpsf_reader', 'webbpsf_reader']
__doctest_skip__ = ['GriddedPSFModelRead']

# Filename extensions recognized as FITS files
_FITS_EXTENSIONS = ('.fits', '.fits.gz', '.fit', '.fit.gz', '.fts',
                    '.fts.gz')

# Pattern matching the WebbPSF detector-position header keywords
_DET_YX_PATTERN = re.compile(r'DET_YX(\d+)$')

# Mapping of the STDPSF filename detector field to the (instrument,
# detector) metadata values
_STDPSF_DETECTOR_MAP = {'WFPC2': ('HST/WFPC2', 'WFPC2'),
                        'ACSHRC': ('HST/ACS', 'HRC'),
                        'ACSWFC': ('HST/ACS', 'WFC'),
                        'WFC3UV': ('HST/WFC3', 'UVIS'),
                        'WFC3IR': ('HST/WFC3', 'IR'),
                        'NRCSW': ('JWST/NIRCam', 'NRCSW'),
                        'NRCA1': ('JWST/NIRCam', 'A1'),
                        'NRCA2': ('JWST/NIRCam', 'A2'),
                        'NRCA3': ('JWST/NIRCam', 'A3'),
                        'NRCA4': ('JWST/NIRCam', 'A4'),
                        'NRCB1': ('JWST/NIRCam', 'B1'),
                        'NRCB2': ('JWST/NIRCam', 'B2'),
                        'NRCB3': ('JWST/NIRCam', 'B3'),
                        'NRCB4': ('JWST/NIRCam', 'B4'),
                        'NRCAL': ('JWST/NIRCam', 'A5'),
                        'NRCBL': ('JWST/NIRCam', 'B5'),
                        'NIRISS': ('JWST/NIRISS', 'NIRISS'),
                        'MIRI': ('JWST/MIRI', 'MIRIM')}


class GriddedPSFModelRead(registry.UnifiedReadWrite):
    """
    Read and parse a FITS file into a `GriddedPSFModel` instance.

    This class enables the astropy unified I/O layer for
    `~photutils.psf.GriddedPSFModel`. This allows easily reading a file
    in different supported data formats using syntax such as::

      >>> from photutils.psf import GriddedPSFModel
      >>> psf_model = GriddedPSFModel.read('filename.fits', format=format)

    Get help on the available readers for
    `~photutils.psf.GriddedPSFModel` using the ``help()`` method::

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
    instance : object
        Descriptor calling instance or `None` if no instance.

    cls : type
        Descriptor calling class (either owner class or instance class).
    """

    def __init__(self, instance, cls):
        # Use default global registry
        super().__init__(instance, cls, 'read', registry=None)

    def __call__(self, *args, **kwargs):
        """
        Read and parse a FITS file into a `GriddedPSFModel` instance
        using the registered "read" function.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed through to data reader. The
            first argument is typically the input filename.

        **kwargs : dict, optional
            Keyword arguments passed through to data reader. This
            includes the ``format`` keyword argument.

        Returns
        -------
        out : `~photutils.psf.GriddedPSFModel`
            A gridded ePSF model corresponding to FITS file contents.
        """
        return self.registry.read(self._cls, *args, **kwargs)


def _read_stdpsf(filename):
    """
    Read a STScI standard-format ePSF (STDPSF) FITS file.

    Parameters
    ----------
    filename : str
        The name of the STDPSF FITS file.

    Returns
    -------
    data : dict
        A dictionary containing the ePSF data and metadata.
    """
    is_hdulist = isinstance(filename, fits.HDUList)
    is_fileobj = (isinstance(filename, io.FileIO)
                  and filename.name.lower().endswith(_FITS_EXTENSIONS))
    is_fits_ext = (isinstance(filename, str)
                   and filename.lower().endswith(_FITS_EXTENSIONS))
    if is_hdulist or is_fileobj or is_fits_ext:
        return _read_fits_stdpsf(filename)
    msg = 'This interface supports only FITS files.'
    raise TypeError(msg)


def _read_fits_stdpsf(filename):
    """
    Read a STScI standard-format ePSF (STDPSF) FITS file.

    Parameters
    ----------
    filename : str
        The name of the STDPSF FITS file.

    Returns
    -------
    data : dict
        A dictionary containing the ePSF data and metadata.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        with fits.open(filename, ignore_missing_end=True) as hdulist:
            header = hdulist[0].header
            data = hdulist[0].data

    try:
        n_psfs = header['NAXIS3']
        nx_grid = header['NXPSFS']
        ny_grid = header['NYPSFS']
    except KeyError as exc:
        msg = 'Invalid STDPSF FITS file'
        raise ValueError(msg) from exc

    if 'IPSFX01' in header:
        xgrid = [header[f'IPSFX{i:02d}'] for i in range(1, nx_grid + 1)]
        ygrid = [header[f'JPSFY{i:02d}'] for i in range(1, ny_grid + 1)]
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
        msg = 'Unknown STDPSF FITS file'
        raise ValueError(msg)

    # STDPSF FITS positions are 1-indexed
    xgrid = np.array(xgrid) - 1
    ygrid = np.array(ygrid) - 1

    # ny_grid, nx_grid, detector
    # 6, 6     WFPC2, 4 det
    # 1, 1     ACS/HRC
    # 10, 9    ACS/WFC, 2 det
    # 3, 3     WFC3/IR
    # 8, 7     WFC3/UVIS, 2 det
    # 5, 5     NIRISS
    # 5, 5     NIRCam SW
    # 10, 20   NIRCam SW (NRCSW), 8 det
    # 5, 5     NIRCam LW
    # 3, 3     MIRI

    return {'data': data,
            'n_psfs': n_psfs,
            'nx_grid': nx_grid,
            'ny_grid': ny_grid,
            'xgrid': xgrid,
            'ygrid': ygrid}


def _split_detectors(grid_data, detector_data, detector_id):
    """
    Split an ePSF array into individual detectors.

    Parameters
    ----------
    grid_data : dict
        A dictionary containing the ePSF data and metadata.

    detector_data : dict
        A dictionary containing the detector data.

    detector_id : int
        The detector ID.

    Returns
    -------
    data : `~numpy.ndarray`
        The ePSF data for the specified detector.

    xgrid : `~numpy.ndarray`
        The x-grid for the specified detector.

    ygrid : `~numpy.ndarray`
        The y-grid for the specified detector.

    Notes
    -----
    The STDPSF files that contain multiple detectors are:

    * HST WFPC2 STDPSF file contains 4 detectors
    * HST ACS/WFC STDPSF file contains 2 detectors
    * HST WFC3/UVIS STDPSF file contains 2 detectors
    * JWST NIRCam "NRCSW" STDPSF file contains 8 detectors
    """
    data = grid_data['data']
    n_psfs = grid_data['n_psfs']
    nx_grid = grid_data['nx_grid']
    ny_grid = grid_data['ny_grid']
    xgrid = grid_data['xgrid']
    ygrid = grid_data['ygrid']
    nx_det = detector_data['nx_det']
    ny_det = detector_data['ny_det']
    det_map = detector_data['det_map']
    det_size = detector_data['det_size']

    ii = np.arange(n_psfs).reshape((ny_grid, nx_grid))
    nx_grid //= nx_det
    ny_grid //= ny_det
    n_detectors = nx_det * ny_det
    ii = reshape_as_blocks(ii, (ny_grid, nx_grid))
    ii = ii.reshape(n_detectors, n_psfs // n_detectors)

    # Map detector_id to index
    det_idx = det_map[detector_id]
    idx = ii[det_idx]
    data = data[idx]

    xp = det_idx % nx_det
    i0 = xp * nx_grid
    i1 = i0 + nx_grid
    xgrid = xgrid[i0:i1] - xp * det_size
    ygrid = (ygrid[:ny_grid] if det_idx < nx_det
             else ygrid[ny_grid:] - det_size)

    return data, xgrid, ygrid


def _split_wfc_uvis(grid_data, detector_id):
    """
    Split an ePSF array into individual WFC/UVIS detectors.

    Parameters
    ----------
    grid_data : dict
        A dictionary containing the ePSF data and metadata.

    detector_id : int
        The detector ID.

    Returns
    -------
    data : `~numpy.ndarray`
        The ePSF data for the specified detector.

    xgrid : `~numpy.ndarray`
        The x-grid for the specified detector.

    ygrid : `~numpy.ndarray`
        The y-grid for the specified detector.
    """
    if detector_id is None:
        msg = 'detector_id must be specified for ACS/WFC and WFC3/UVIS ePSFs'
        raise ValueError(msg)
    if detector_id not in (1, 2):
        msg = 'detector_id must be 1 or 2'
        raise ValueError(msg)

    # ACS/WFC1 and WFC3/UVIS1 chip1 (sci, 2) are above chip2 (sci, 1)
    # in y-pixel coordinates
    xgrid = grid_data['xgrid']
    ygrid = grid_data['ygrid']
    ygrid = ygrid.reshape((2, ygrid.shape[0] // 2))[detector_id - 1]
    if detector_id == 2:
        ygrid -= 2048

    n_psfs = grid_data['n_psfs']
    data = grid_data['data']
    data_ny, data_nx = data.shape[1:]
    data = data.reshape((2, n_psfs // 2, data_ny, data_nx))[detector_id - 1]

    return data, xgrid, ygrid


def _split_wfpc2(grid_data, detector_id):
    """
    Split an ePSF array into individual WFPC2 detectors.

    Parameters
    ----------
    grid_data : dict
        A dictionary containing the ePSF data and metadata.

    detector_id : int
        The detector ID.

    Returns
    -------
    data : `~numpy.ndarray`
        The ePSF data for the specified detector.

    xgrid : `~numpy.ndarray`
        The x-grid for the specified detector.

    ygrid : `~numpy.ndarray`
        The y-grid for the specified detector.
    """
    if detector_id is None:
        msg = 'detector_id must be specified for WFPC2 ePSFs'
        raise ValueError(msg)
    if detector_id not in range(1, 5):
        msg = 'detector_id must be between 1 and 4, inclusive'
        raise ValueError(msg)

    nx_det = 2
    ny_det = 2
    det_size = 800

    # Map of detector ID to index in the 2x2 grid of detectors. The
    # detector IDs are defined in the STDPSF filenames as follows:
    # det (exten:idx)
    # WF2 (2:2)  PC (1:3)
    # WF3 (3:0)  WF4 (4:1)
    det_map = {1: 3, 2: 2, 3: 0, 4: 1}

    detector_data = {'nx_det': nx_det,
                     'ny_det': ny_det,
                     'det_size': det_size,
                     'det_map': det_map}

    return _split_detectors(grid_data, detector_data, detector_id)


def _split_nrcsw(grid_data, detector_id):
    """
    Split an ePSF array into individual NIRCam SW detectors.

    Parameters
    ----------
    grid_data : dict
        A dictionary containing the ePSF data and metadata.

    detector_id : int
        The detector ID.

    Returns
    -------
    data : `~numpy.ndarray`
        The ePSF data for the specified detector.

    xgrid : `~numpy.ndarray`
        The x-grid for the specified detector.

    ygrid : `~numpy.ndarray`
        The y-grid for the specified detector.
    """
    if detector_id is None:
        msg = 'detector_id must be specified for NRCSW ePSFs'
        raise ValueError(msg)
    if detector_id not in range(1, 9):
        msg = 'detector_id must be between 1 and 8, inclusive'
        raise ValueError(msg)

    nx_det = 4
    ny_det = 2
    det_size = 2048

    # Map of detector ID to index in the 4x2 grid of detectors. The
    # detector IDs are defined in the STDPSF filenames as follows:
    # det (ext:idx)
    # A2 (2:4)  A4 (4:5)  B3 (7:6)  B1 (5:7)
    # A1 (1:0)  A3 (3:1)  B4 (8:2)  B2 (6:3)
    det_map = {1: 0, 3: 1, 8: 2, 6: 3, 2: 4, 4: 5, 7: 6, 5: 7}

    detector_data = {'nx_det': nx_det,
                     'ny_det': ny_det,
                     'det_size': det_size,
                     'det_map': det_map}

    return _split_detectors(grid_data, detector_data, detector_id)


def _get_metadata(filename, detector_id):
    """
    Get metadata from the filename and ``detector_id``.

    Parameters
    ----------
    filename : str
        The name of the STDPSF FITS file.

    detector_id : int
        The detector ID.

    Returns
    -------
    meta : dict or `None`
        A dictionary containing the metadata.
    """
    if isinstance(filename, io.FileIO):
        filename = filename.name

    # Strip the file extension (e.g., '.fits' or '.fits.gz') before
    # splitting the filename into its underscore-separated fields.
    basename = os.path.basename(filename).split('.')[0]
    parts = basename.split('_')
    if len(parts) not in (3, 4):
        return None  # filename from astropy download_file

    detector, filter_name = parts[1:3]
    meta = {'STDPSF': filename,
            'detector': detector,
            'filter': filter_name}

    if detector_id is not None:
        try:
            # Copy so that the module-level map is never mutated
            inst_det = list(_STDPSF_DETECTOR_MAP[detector])
        except KeyError as exc:
            msg = f'Unknown detector {detector}'
            raise ValueError(msg) from exc

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


@deprecated_positional_kwargs(since='3.0', until='4.0')
def stdpsf_reader(filename, detector_id=None):
    """
    Generate a `~photutils.psf.GriddedPSFModel` from a STScI standard-
    format ePSF (STDPSF) FITS file.

    .. note::
        Instead of being used directly, this function is intended to
        be used via the `~photutils.psf.GriddedPSFModel` ``read``
        method, e.g., ``model = GriddedPSFModel.read(filename,
        format='stdpsf')``.

    STDPSF files are FITS files that contain a 3D array of ePSFs with
    the header detailing where the fiducial ePSFs are located in the
    detector coordinate frame.

    The oversampling factor for STDPSF FITS files is assumed to be 4.

    Parameters
    ----------
    filename : str
        The name of the STDPSF FITS file. A URL can also be used.

    detector_id : `None` or int, optional
        For STDPSF files that contain ePSF grids for multiple detectors,
        one will need to identify the detector for which to extract the
        ePSF grid. This keyword is ignored for STDPSF files that do not
        contain ePSF grids for multiple detectors.

        For WFPC2, the detector value (int) should be:

        * 1: PC, 2: WF2, 3: WF3, 4: WF4

        For ACS/WFC and WFC3/UVIS, the detector value should be:

        * 1: WFC2, UVIS2 (sci, 1)
        * 2: WFC1, UVIS1 (sci, 2)

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
    from photutils.psf import GriddedPSFModel  # prevent circular import

    grid_data = _read_stdpsf(filename)

    # Number of ePSFs in the STDPSF files that contain grids for
    # multiple detectors, mapped to the function that extracts a single
    # detector from the grid
    splitters = {90: _split_wfc_uvis,  # ACS/WFC or WFC3/UVIS (2 chips)
                 56: _split_wfc_uvis,  # ACS/WFC or WFC3/UVIS (2 chips)
                 36: _split_wfpc2,  # WFPC2 (4 chips)
                 200: _split_nrcsw}  # NIRCam SW (8 chips)

    splitter = splitters.get(grid_data['n_psfs'])
    if splitter is not None:
        data, xgrid, ygrid = splitter(grid_data, detector_id)
    else:
        data = grid_data['data']
        xgrid = grid_data['xgrid']
        ygrid = grid_data['ygrid']

    # itertools.product iterates over the last input first
    xy_grid = np.array([yx[::-1] for yx in itertools.product(ygrid, xgrid)])

    oversampling = 4  # assumption for STDPSF files
    meta = {'grid_xypos': xy_grid,
            'oversampling': oversampling}

    # Try to get additional metadata from the filename because this
    # information is not currently available in the FITS headers.
    file_meta = _get_metadata(filename, detector_id)
    if file_meta is not None:
        meta.update(file_meta)

    return GriddedPSFModel(NDData(data, meta=meta))


def webbpsf_reader(filename):
    """
    Generate a `~photutils.psf.GriddedPSFModel` from a WebbPSF FITS file
    containing a PSF grid.

    .. note::
        Instead of being used directly, this function is intended to
        be used via the `~photutils.psf.GriddedPSFModel` ``read``
        method, e.g., ``model = GriddedPSFModel.read(filename,
        format='webbpsf')``.

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
    from photutils.psf import GriddedPSFModel  # prevent circular import

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        with fits.open(filename, ignore_missing_end=True) as hdulist:
            header = hdulist[0].header
            data = hdulist[0].data

    # Handle the case of only one 2D PSF
    data = np.atleast_3d(data)

    if not any('DET_YX' in key for key in header):
        msg = 'Invalid WebbPSF FITS file; missing "DET_YX{}" header keys'
        raise ValueError(msg)
    if 'OVERSAMP' not in header:
        msg = 'Invalid WebbPSF FITS file; missing "OVERSAMP" header key'
        raise ValueError(msg)

    # Convert header to meta dict
    header = header.copy(strip=True)
    header.pop('HISTORY', None)
    header.pop('COMMENT', None)
    header.pop('', None)
    meta = dict(header)
    meta = {key.lower(): meta[key] for key in meta}  # user lower-case keys

    # Define grid_xypos from the DET_YX{i} FITS header keywords. The
    # keywords are sorted by their numeric index so that the positions
    # always match the order of the ePSF planes in the data array. The
    # header values are the '(y, x)' detector positions, but grid_xypos
    # is defined in (x, y) order.
    det_yx_keys = {}
    for key in header:
        match = _DET_YX_PATTERN.match(key)
        if match is not None:
            det_yx_keys[int(match.group(1))] = key

    xypos = []
    for index in sorted(det_yx_keys):
        vals = header[det_yx_keys[index]].lstrip('(').rstrip(')').split(',')
        xypos.append((float(vals[1]), float(vals[0])))
    meta['grid_xypos'] = xypos

    if 'oversampling' not in meta:
        meta['oversampling'] = meta['oversamp']

    ndd = NDData(data, meta=meta)

    return GriddedPSFModel(ndd)


def _has_fits_header_keys(filepath, keys):
    """
    Determine whether a file is a FITS file whose primary header
    contains all of the given keywords.

    Parameters
    ----------
    filepath : str or `None`
        The file path of the FITS file.

    keys : tuple of str
        The FITS header keywords that must all be present.

    Returns
    -------
    result : bool
        Returns `True` if the file is a FITS file containing all of the
        input keywords.
    """
    if filepath is None or not filepath.lower().endswith(_FITS_EXTENSIONS):
        return False

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        header = fits.getheader(filepath)

    return all(key in header for key in keys)


def is_stdpsf(origin, filepath, fileobj, *args, **kwargs):
    """
    Determine whether a file is a STDPSF FITS file.

    Parameters
    ----------
    origin : {'read', 'write'}
        A string indicating whether the file is to be opened for reading
        or writing.

    filepath : str
        The file path of the FITS file.

    fileobj : file-like object
        An open file object to read the file's contents, or `None` if
        the file could not be opened.

    *args, **kwargs
        Any additional positional or keyword arguments for the read or
        write function.

    Returns
    -------
    result : bool
        Returns `True` if the given file is a STDPSF FITS file.
    """
    return _has_fits_header_keys(filepath, ('NAXIS3', 'NXPSFS', 'NYPSFS'))


def is_webbpsf(origin, filepath, fileobj, *args, **kwargs):
    """
    Determine whether a file is a WebbPSF FITS file.

    Parameters
    ----------
    origin : {'read', 'write'}
        A string indicating whether the file is to be opened for reading
        or writing.

    filepath : str
        The file path of the FITS file.

    fileobj : file-like object
        An open file object to read the file's contents, or `None` if
        the file could not be opened.

    *args, **kwargs
        Any additional positional or keyword arguments for the read or
        write function.

    Returns
    -------
    result : bool
        Returns `True` if the given file is a WebbPSF FITS file.
    """
    return _has_fits_header_keys(filepath, ('NAXIS3', 'OVERSAMP', 'DET_YX0'))
