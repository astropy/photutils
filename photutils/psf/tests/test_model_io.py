# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model_io module.
"""

import io
import os.path as op
from itertools import product

import numpy as np
import pytest
from astropy.io import fits
from numpy.testing import assert_allclose, assert_equal

from photutils.psf import GriddedPSFModel
from photutils.psf.model_io import (_get_metadata, _has_fits_header_keys,
                                    _read_stdpsf, is_stdpsf, is_webbpsf)

# The first file has a single detector, the rest have multiple detectors
STDPSF_FILENAMES = ('STDPSF_NRCA1_F150W_mock.fits',
                    'STDPSF_ACSWFC_F814W_mock.fits',
                    'STDPSF_NRCSW_F150W_mock.fits',
                    'STDPSF_WFC3UV_F814W_mock.fits',
                    'STDPSF_WFPC2_F814W_mock.fits')

WEBBPSF_FILENAMES = ('nircam_nrca1_f200w_fovp101_samp4_npsf16_mock.fits',
                     'nircam_nrca1_f200w_fovp101_samp4_npsf4_mock.fits',
                     'nircam_nrca5_f444w_fovp101_samp4_npsf4_mock.fits',
                     'nircam_nrcb4_f150w_fovp101_samp4_npsf1_mock.fits')


def get_data_filename(filename):
    """
    Get the full path of a test data file.

    Parameters
    ----------
    filename : str
        The name of the file in the test data directory.

    Returns
    -------
    result : str
        The full path of the test data file.
    """
    return op.join(op.dirname(op.abspath(__file__)), 'data', filename)


def encode_position(x, y):
    """
    Encode an ``(x, y)`` grid position as a single unique value.

    This is used to fill each plane of a mock ePSF grid so that the
    grid position a plane was read back at can be verified.

    Parameters
    ----------
    x, y : float
        The grid position.

    Returns
    -------
    result : float
        The encoded position.
    """
    return float(x) + 10000.0 * float(y)


def make_webbpsf_file(filename, xgrid, ygrid, psf_shape=(8, 8),
                      oversampling=4, keys_to_remove=()):
    """
    Write a mock WebbPSF gridded-ePSF FITS file.

    WebbPSF orders the ePSF planes with the y position varying fastest
    and records each position as a ``"(y, x)"`` string in the
    ``DET_YX{i}`` header keywords. Each plane is filled with its
    `encode_position` value.

    Parameters
    ----------
    filename : str
        The output filename.

    xgrid, ygrid : array_like
        The x and y detector grid positions.

    psf_shape : tuple of int, optional
        The ``(ny, nx)`` shape of each ePSF image.

    oversampling : int, optional
        The oversampling factor written to the ``OVERSAMP`` keyword.

    keys_to_remove : tuple of str, optional
        FITS header keywords to omit from the output file. This is used
        to create invalid files.
    """
    header = fits.Header()
    header['OVERSAMP'] = oversampling
    header['INSTRUME'] = 'NIRCam'
    header['DETECTOR'] = 'NRCA1'
    header['FILTER'] = 'F200W'

    planes = []
    for index, (x, y) in enumerate(product(xgrid, ygrid)):
        planes.append(np.full(psf_shape, encode_position(x, y)))
        header[f'DET_YX{index}'] = (
            f'({float(y)}, {float(x)})',
            f"The #{index} PSF's (y,x) detector pixel position")

    for key in keys_to_remove:
        del header[key]

    hdu = fits.PrimaryHDU(np.array(planes), header=header)
    hdu.writeto(filename, overwrite=True)


class TestSTDPSFReader:
    """
    Tests for reading STDPSF FITS files.
    """

    def test_read_stdpsf(self):
        """
        Test STDPSF read for a single detector.
        """
        filename = get_data_filename('STDPSF_NRCA1_F150W_mock.fits')
        psfmodel = GriddedPSFModel.read(filename)
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])
        assert isinstance(psfmodel.meta['grid_xypos'], np.ndarray)
        assert_equal(psfmodel.oversampling, [4, 4])
        assert_equal(psfmodel.meta['oversampling'], psfmodel.oversampling)

    @pytest.mark.parametrize('filename', STDPSF_FILENAMES[1:])
    @pytest.mark.parametrize('detector_id', [1, 2])
    def test_read_stdpsf_multi_detector(self, filename, detector_id):
        """
        Test STDPSF read for multiple detectors.
        """
        filename = get_data_filename(filename)
        psfmodel = GriddedPSFModel.read(filename, detector_id=detector_id,
                                        format='stdpsf')
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])
        assert_equal(psfmodel.oversampling, [4, 4])
        assert_equal(psfmodel.meta['oversampling'], psfmodel.oversampling)

        # Test format auto-detect
        psfmodel = GriddedPSFModel.read(filename, detector_id=detector_id)
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])

        match = 'detector_id must be specified'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel.read(filename, detector_id=None)

        match = 'detector_id must be '
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel.read(filename, detector_id=-1)

    def test_read_stdpsf_hdulist(self):
        """
        Test that an open HDUList is accepted.
        """
        filename = get_data_filename('STDPSF_NRCA1_F150W_mock.fits')
        with fits.open(filename) as hdulist:
            grid_data = _read_stdpsf(hdulist)
        assert grid_data['data'].shape[0] == grid_data['n_psfs']

    def test_read_stdpsf_fileobj(self):
        """
        Test that an open file object is accepted.
        """
        filename = get_data_filename('STDPSF_NRCA1_F150W_mock.fits')
        with io.FileIO(filename) as fileobj:
            grid_data = _read_stdpsf(fileobj)
        assert grid_data['data'].shape[0] == grid_data['n_psfs']

    def test_read_stdpsf_not_fits(self):
        match = 'This interface supports only FITS files'
        with pytest.raises(TypeError, match=match):
            _read_stdpsf('filename.txt')
        with pytest.raises(TypeError, match=match):
            _read_stdpsf(42)

    def test_read_stdpsf_missing_keys(self, tmp_path):
        filename = str(tmp_path / 'bad_stdpsf.fits')
        fits.PrimaryHDU(np.ones((4, 5, 5))).writeto(filename)

        match = 'Invalid STDPSF FITS file'
        with pytest.raises(ValueError, match=match):
            _read_stdpsf(filename)

    def test_read_stdpsf_unknown_grid_keys(self, tmp_path):
        header = fits.Header()
        header['NXPSFS'] = 2
        header['NYPSFS'] = 2
        filename = str(tmp_path / 'unknown_stdpsf.fits')
        fits.PrimaryHDU(np.ones((4, 5, 5)), header=header).writeto(filename)

        match = 'Unknown STDPSF FITS file'
        with pytest.raises(ValueError, match=match):
            _read_stdpsf(filename)


class TestWebbPSFReader:
    """
    Tests for reading WebbPSF FITS files.
    """

    @pytest.mark.parametrize('filename', WEBBPSF_FILENAMES)
    def test_read_webbpsf(self, filename):
        filename = get_data_filename(filename)
        psfmodel = GriddedPSFModel.read(filename, format='webbpsf')
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])
        assert_equal(psfmodel.oversampling, [4, 4])
        assert_equal(psfmodel.meta['oversampling'], psfmodel.oversampling)

        # Test format auto-detect
        psfmodel = GriddedPSFModel.read(filename)
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])

    def test_read_webbpsf_asymmetric_grid(self, tmp_path):
        """
        Test that each ePSF plane of a WebbPSF file is placed at the
        detector position given by its DET_YX header keyword.

        The grid is deliberately non-square so that an x/y transpose
        cannot go unnoticed.
        """
        xgrid = (0.0, 1000.0, 2047.0)
        ygrid = (0.0, 500.0, 1000.0, 1500.0, 2047.0)
        filename = str(tmp_path / 'webbpsf_asymmetric_mock.fits')
        make_webbpsf_file(filename, xgrid, ygrid)

        psfmodel = GriddedPSFModel.read(filename, format='webbpsf')
        assert_equal(np.unique(psfmodel.grid_xypos[:, 0]), xgrid)
        assert_equal(np.unique(psfmodel.grid_xypos[:, 1]), ygrid)
        assert psfmodel.meta['grid_shape'] == (len(ygrid), len(xgrid))

        for (x, y), plane in zip(psfmodel.grid_xypos, psfmodel.data,
                                 strict=True):
            assert_allclose(plane, encode_position(x, y))

    def test_read_webbpsf_key_order(self, tmp_path):
        """
        Test that the DET_YX keywords are ordered by their numeric
        index and not by their string ordering (e.g., DET_YX10 must
        follow DET_YX9).
        """
        xgrid = (0.0, 1000.0)
        ygrid = (0.0, 400.0, 800.0, 1200.0, 1600.0, 2047.0)
        filename = str(tmp_path / 'webbpsf_key_order_mock.fits')
        make_webbpsf_file(filename, xgrid, ygrid)

        psfmodel = GriddedPSFModel.read(filename, format='webbpsf')
        assert len(psfmodel.grid_xypos) == 12
        for (x, y), plane in zip(psfmodel.grid_xypos, psfmodel.data,
                                 strict=True):
            assert_allclose(plane, encode_position(x, y))

    def test_read_webbpsf_missing_det_yx(self, tmp_path):
        filename = str(tmp_path / 'webbpsf_no_det_yx.fits')
        make_webbpsf_file(filename, (0.0, 10.0), (0.0, 10.0),
                          keys_to_remove=('DET_YX0', 'DET_YX1', 'DET_YX2',
                                          'DET_YX3'))

        match = 'missing "DET_YX'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel.read(filename, format='webbpsf')

    def test_read_webbpsf_missing_oversamp(self, tmp_path):
        filename = str(tmp_path / 'webbpsf_no_oversamp.fits')
        make_webbpsf_file(filename, (0.0, 10.0), (0.0, 10.0),
                          keys_to_remove=('OVERSAMP',))

        match = 'missing "OVERSAMP" header key'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel.read(filename, format='webbpsf')


class TestGetMetadata:
    """
    Tests for parsing metadata from the STDPSF filename.
    """

    @pytest.mark.parametrize(('filename', 'detector', 'filter_name'),
                             [('STDPSF_ACSWFC_F814W.fits', 'ACSWFC', 'F814W'),
                              ('STDPSF_MIRI_F560W.fits.gz', 'MIRI', 'F560W'),
                              ('STDPSF_NRCA1_F150W_mock.fits', 'NRCA1',
                               'F150W'),
                              ('/a/b/STDPSF_WFC3IR_F160W.fits', 'WFC3IR',
                               'F160W')])
    def test_filename_fields(self, filename, detector, filter_name):
        meta = _get_metadata(filename, None)
        assert meta['STDPSF'] == filename
        assert meta['detector'] == detector
        assert meta['filter'] == filter_name

    def test_fileobj(self, tmp_path):
        path = tmp_path / 'STDPSF_ACSHRC_F606W.fits'
        path.write_bytes(b'')
        with io.FileIO(str(path)) as fileobj:
            meta = _get_metadata(fileobj, None)
        assert meta['detector'] == 'ACSHRC'
        assert meta['filter'] == 'F606W'

    @pytest.mark.parametrize('filename', ['abcdef123456', 'a_b', 'a_b_c_d_e'])
    def test_unparsable_filename(self, filename):
        # Test a cache filename from astropy download_file
        assert _get_metadata(filename, None) is None

    @pytest.mark.parametrize(('detector_id', 'detector'),
                             [(1, 'PC'), (2, 'WF2'), (3, 'WF3'), (4, 'WF4')])
    def test_wfpc2_detectors(self, detector_id, detector):
        meta = _get_metadata('STDPSF_WFPC2_F814W.fits', detector_id)
        assert meta['instrument'] == 'HST/WFPC2'
        assert meta['detector'] == detector

    @pytest.mark.parametrize(('detector_id', 'detector'),
                             [(1, 'WFC2'), (2, 'WFC1')])
    def test_wfc_chips(self, detector_id, detector):
        meta = _get_metadata('STDPSF_ACSWFC_F814W.fits', detector_id)
        assert meta['instrument'] == 'HST/ACS'
        assert meta['detector'] == detector

    def test_nrcsw_detectors(self):
        detectors = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4']
        for detector_id, detector in enumerate(detectors, start=1):
            meta = _get_metadata('STDPSF_NRCSW_F150W.fits', detector_id)
            assert meta['instrument'] == 'JWST/NIRCam'
            assert meta['detector'] == detector

    def test_unknown_detector(self):
        match = 'Unknown detector BADDET'
        with pytest.raises(ValueError, match=match):
            _get_metadata('STDPSF_BADDET_F814W.fits', 1)

    def test_detector_map_not_mutated(self):
        """
        Repeated calls must not corrupt the module-level lookup table.
        """
        first = _get_metadata('STDPSF_ACSWFC_F814W.fits', 1)['detector']
        _get_metadata('STDPSF_ACSWFC_F814W.fits', 2)
        third = _get_metadata('STDPSF_ACSWFC_F814W.fits', 1)['detector']
        assert first == third == 'WFC2'


class TestFormatIdentifiers:
    """
    Tests for the unified-I/O format identifier functions.
    """

    def test_is_stdpsf(self):
        filename = get_data_filename('STDPSF_NRCA1_F150W_mock.fits')
        assert is_stdpsf('read', filename, None)
        assert not is_webbpsf('read', filename, None)

    def test_is_webbpsf(self):
        filename = get_data_filename(WEBBPSF_FILENAMES[0])
        assert is_webbpsf('read', filename, None)
        assert not is_stdpsf('read', filename, None)

    @pytest.mark.parametrize('filepath', [None, 'filename.txt'])
    def test_not_a_fits_file(self, filepath):
        assert not is_stdpsf('read', filepath, None)
        assert not is_webbpsf('read', filepath, None)

    def test_has_fits_header_keys(self):
        filename = get_data_filename('STDPSF_NRCA1_F150W_mock.fits')
        assert _has_fits_header_keys(filename, ('NAXIS3',))
        assert not _has_fits_header_keys(filename, ('NAXIS3', 'NOSUCHKEY'))
