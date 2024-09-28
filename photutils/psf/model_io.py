# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines tools for reading and writing PSF models.
"""

import io
import itertools
import os
import warnings

import numpy as np
from astropy.io import fits, registry
from astropy.io.fits.verify import VerifyWarning
from astropy.nddata import NDData, reshape_as_blocks

__all__ = ['GriddedPSFModelRead', 'stdpsf_reader', 'webbpsf_reader']
__doctest_skip__ = ['GriddedPSFModelRead']


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
        # uses default global registry
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
        The name of the STDPDF FITS file.

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

    # nypsfs, nxpsfs, detector
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
            'npsfs': npsfs,
            'nxpsfs': nxpsfs,
            'nypsfs': nypsfs,
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

    ygrid = ygrid[:nypsfs] if det_idx < nxdet else ygrid[nypsfs:] - det_size

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

    Parameters
    ----------
    filename : str
        The name of the STDPDF FITS file.

    detector_id : int
        The detector ID.

    Returns
    -------
    meta : dict or `None`
        A dictionary containing the metadata.
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
    from photutils.psf import GriddedPSFModel  # prevent circular import

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

    # handle the case of only one 2D PSF
    data = np.atleast_3d(data)

    if not any('DET_YX' in key for key in header):
        raise ValueError('Invalid WebbPSF FITS file; missing "DET_YX{}" '
                         'header keys.')
    if 'OVERSAMP' not in header:
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
    for key in meta:
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
    if filepath is not None:
        extens = ('.fits', '.fits.gz', '.fit', '.fit.gz', '.fts', '.fts.gz')
        isfits = filepath.lower().endswith(extens)
        if isfits:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', VerifyWarning)
                header = fits.getheader(filepath)
            keys = ('NAXIS3', 'NXPSFS', 'NYPSFS')
            return all(key in header for key in keys)

    return False


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
    if filepath is not None:
        extens = ('.fits', '.fits.gz', '.fit', '.fit.gz', '.fts', '.fts.gz')
        isfits = filepath.lower().endswith(extens)
        if isfits:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', VerifyWarning)
                header = fits.getheader(filepath)
            keys = ('NAXIS3', 'OVERSAMP', 'DET_YX0')
            return all(key in header for key in keys)

    return False
