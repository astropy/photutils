# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the load module.
"""

from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.datasets import load
from photutils.datasets.load import _get_path, _load_fits_as_imagehdu


def test_get_path():
    """
    Test _get_path with a valid filename and location, and with an
    invalid location.
    """
    fn = '4gaussians_params.ecsv'
    path = _get_path(fn, location='local')
    assert fn in path

    match = 'Invalid location:'
    with pytest.raises(ValueError, match=match):
        _get_path('filename', location='invalid')


def test_get_path_photutils_datasets():
    """
    Test _get_path with location='photutils-datasets'.
    """
    with patch('photutils.datasets.load.download_file') as mock_dl:
        mock_dl.return_value = '/path/to/file.fits'
        result = _get_path('file.fits', location='photutils-datasets',
                           cache=False)
        assert result == '/path/to/file.fits'
        mock_dl.assert_called_once()
        call_args = mock_dl.call_args
        assert 'photutils-datasets' in call_args[0][0]
        assert call_args[1]['cache'] is False


@pytest.fixture
def url_paths():
    """
    Fixture providing URLs for remote location tests.
    """
    filename = 'test_file.fits'
    primary_url = f'http://data.astropy.org/photometry/{filename}'
    datasets_url = (
        'https://github.com/astropy/photutils-datasets/raw/'
        f'main/data/{filename}'
    )
    return {
        'filename': filename,
        'primary_url': primary_url,
        'datasets_url': datasets_url,
    }


class TestGetPathRemote:
    """
    Tests for the behavior of _get_path with location='remote'.

    These tests verify that ``download_file`` is called with both the
    primary astropy data URL and the photutils-datasets fallback URL
    passed as ``sources``, so that astropy handles the cache lookup
    and fallback behavior automatically.
    """

    def test_calls_download_file_with_sources(self, url_paths):
        """
        Test that _get_path calls ``download_file`` with the primary
        URL as the cache key and both URLs in ``sources``.
        """
        with patch('photutils.datasets.load.download_file') as mock_dl:
            mock_dl.return_value = '/cached/path/test_file.fits'

            result = _get_path(url_paths['filename'], location='remote')

            assert result == '/cached/path/test_file.fits'
            mock_dl.assert_called_once_with(
                url_paths['primary_url'],
                cache=True,
                sources=[url_paths['primary_url'],
                         url_paths['datasets_url']],
                show_progress=False,
            )

    def test_passes_cache_and_show_progress(self, url_paths):
        """
        Test that the ``cache`` and ``show_progress`` arguments are
        forwarded to ``download_file``.
        """
        with patch('photutils.datasets.load.download_file') as mock_dl:
            mock_dl.return_value = '/downloaded/path/test_file.fits'

            result = _get_path(url_paths['filename'], location='remote',
                               cache=False, show_progress=True)

            assert result == '/downloaded/path/test_file.fits'
            mock_dl.assert_called_once_with(
                url_paths['primary_url'],
                cache=False,
                sources=[url_paths['primary_url'],
                         url_paths['datasets_url']],
                show_progress=True,
            )


def test_load_irac_psf_invalid_channel():
    """
    Test that load_irac_psf raises a ValueError when an invalid channel
    number is provided.
    """
    match = 'channel must be 1, 2, 3, or 4'
    with pytest.raises(ValueError, match=match):
        load.load_irac_psf(0)
    with pytest.raises(ValueError, match=match):
        load.load_irac_psf(5)


@pytest.mark.remote_data
def test_load_star_image():
    """
    Test that load_star_image returns an HDU with the expected header
    and data shape.
    """
    with pytest.warns(AstropyDeprecationWarning):
        hdu = load.load_star_image()
    assert len(hdu.header) == 106
    assert hdu.data.shape == (1059, 1059)


def test_load_fits_as_imagehdu(tmp_path):
    """
    Test the _load_fits_as_imagehdu helper function.
    """
    data = np.ones((10, 10))
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -32
    header['NAXIS'] = 2
    header['NAXIS1'] = 10
    header['NAXIS2'] = 10
    header['EXTEND'] = True
    header['COMMENT'] = 'Test header'

    primary_hdu = fits.PrimaryHDU(data=data, header=header)
    hdulist = fits.HDUList([primary_hdu])

    filepath = tmp_path / 'test.fits'
    hdulist.writeto(filepath)

    result = _load_fits_as_imagehdu(str(filepath))

    assert isinstance(result, fits.ImageHDU)
    assert np.array_equal(result.data, data)
    assert result.header['COMMENT'] == 'Test header'


class TestLoadFunctionsMocked:
    """
    Tests for the load functions using mocking to avoid remote data
    access.
    """

    def test_load_spitzer_image(self, tmp_path):
        """
        Test the load_spitzer_image function with mocked file download.
        """
        data = np.random.default_rng(seed=0).random((100, 100))
        header = fits.Header()
        header['TELESCOP'] = 'Spitzer'
        header['INSTRUME'] = 'IRAC'

        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([primary_hdu])

        filepath = tmp_path / 'spitzer_example_image.fits'
        hdulist.writeto(filepath)

        with (pytest.warns(AstropyDeprecationWarning),
              patch('photutils.datasets.load._get_path',
                    return_value=str(filepath))):
            hdu = load.load_spitzer_image()

        assert isinstance(hdu, fits.ImageHDU)
        assert np.array_equal(hdu.data, data)
        assert hdu.header['TELESCOP'] == 'Spitzer'

    def test_load_spitzer_image_show_progress(self, tmp_path):
        """
        Test the load_spitzer_image function with show_progress=True.
        """
        data = np.random.default_rng(seed=0).random((100, 100))
        header = fits.Header()

        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([primary_hdu])

        filepath = tmp_path / 'spitzer_example_image.fits'
        hdulist.writeto(filepath)

        with (pytest.warns(AstropyDeprecationWarning),
              patch('photutils.datasets.load._get_path',
                    return_value=str(filepath)) as mock_get_path):
            hdu = load.load_spitzer_image(show_progress=True)

        assert isinstance(hdu, fits.ImageHDU)
        mock_get_path.assert_called_once_with(
            'spitzer_example_image.fits',
            location='remote',
            show_progress=True,
        )

    def test_load_spitzer_catalog(self, tmp_path):
        """
        Test the load_spitzer_catalog function with mocked file
        download.
        """
        catalog_data = Table()
        catalog_data['l'] = [18.23, 18.15, 18.30]
        catalog_data['b'] = [0.20, 0.22, 0.18]

        filepath = tmp_path / 'spitzer_example_catalog.xml'
        catalog_data.write(filepath, format='votable', overwrite=True)

        with (pytest.warns(AstropyDeprecationWarning),
              patch('photutils.datasets.load._get_path',
                    return_value=str(filepath))):
            catalog = load.load_spitzer_catalog()

        assert isinstance(catalog, Table)
        assert 'l' in catalog.colnames
        assert 'b' in catalog.colnames
        assert len(catalog) == 3

    def test_load_spitzer_catalog_show_progress(self, tmp_path):
        """
        Test the load_spitzer_catalog function with show_progress=True.
        """
        catalog_data = Table()
        catalog_data['l'] = [18.23]
        catalog_data['b'] = [0.20]

        filepath = tmp_path / 'spitzer_example_catalog.xml'
        catalog_data.write(filepath, format='votable', overwrite=True)

        with (pytest.warns(AstropyDeprecationWarning),
              patch('photutils.datasets.load._get_path',
                    return_value=str(filepath)) as mock_get_path):
            catalog = load.load_spitzer_catalog(show_progress=True)

        assert isinstance(catalog, Table)
        mock_get_path.assert_called_once_with(
            'spitzer_example_catalog.xml',
            location='remote',
            show_progress=True,
        )

    def test_load_irac_psf(self, tmp_path):
        """
        Test the load_irac_psf function with mocked file download.
        """
        data = np.random.default_rng(seed=0).random((41, 41))
        header = fits.Header()
        header['TELESCOP'] = 'Spitzer'
        header['INSTRUME'] = 'IRAC'
        header['CHNLNUM'] = 1

        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([primary_hdu])

        filepath = tmp_path / 'irac_ch1_flight.fits'
        hdulist.writeto(filepath)

        with patch('photutils.datasets.load._get_path',
                   return_value=str(filepath)):
            hdu = load.load_irac_psf(1)

        assert isinstance(hdu, fits.ImageHDU)
        assert np.array_equal(hdu.data, data)
        assert hdu.header['TELESCOP'] == 'Spitzer'

    def test_load_irac_psf_all_channels(self, tmp_path):
        """
        Test the load_irac_psf function for all valid channels.
        """
        for channel in range(1, 5):
            data = np.random.default_rng(seed=0).random((41, 41))
            header = fits.Header()
            header['CHNLNUM'] = channel

            primary_hdu = fits.PrimaryHDU(data=data, header=header)
            hdulist = fits.HDUList([primary_hdu])

            filepath = tmp_path / f'irac_ch{channel}_flight.fits'
            hdulist.writeto(filepath, overwrite=True)

            with patch('photutils.datasets.load._get_path',
                       return_value=str(filepath)):
                hdu = load.load_irac_psf(channel)

            assert isinstance(hdu, fits.ImageHDU)
            assert hdu.header['CHNLNUM'] == channel

    def test_load_irac_psf_show_progress(self, tmp_path):
        """
        Test the load_irac_psf function with show_progress=True.
        """
        data = np.random.default_rng(seed=0).random((41, 41))
        header = fits.Header()

        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([primary_hdu])

        filepath = tmp_path / 'irac_ch2_flight.fits'
        hdulist.writeto(filepath)

        with (
            patch('photutils.datasets.load._get_path',
                  return_value=str(filepath)) as mock_get_path,
        ):
            hdu = load.load_irac_psf(2, show_progress=True)

        assert isinstance(hdu, fits.ImageHDU)
        mock_get_path.assert_called_once_with(
            'irac_ch2_flight.fits',
            location='remote',
            show_progress=True,
        )

    def test_load_star_image_mocked(self, tmp_path):
        """
        Test the load_star_image function with mocked file download.
        """
        data = np.random.default_rng(seed=0).random((200, 200))
        header = fits.Header()
        header['OBJECT'] = 'M67'
        header['TELESCOP'] = 'Palomar'

        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([primary_hdu])

        filepath = tmp_path / 'M6707HH.fits'
        hdulist.writeto(filepath)

        with (pytest.warns(AstropyDeprecationWarning),
              patch('photutils.datasets.load._get_path',
                    return_value=str(filepath))):
            hdu = load.load_star_image()

        assert isinstance(hdu, fits.ImageHDU)
        assert np.array_equal(hdu.data, data)
        assert hdu.header['OBJECT'] == 'M67'

    def test_load_star_image_show_progress(self, tmp_path):
        """
        Test the load_star_image function with show_progress=True.
        """
        data = np.random.default_rng(seed=0).random((200, 200))
        header = fits.Header()

        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([primary_hdu])

        filepath = tmp_path / 'M6707HH.fits'
        hdulist.writeto(filepath)

        with (pytest.warns(AstropyDeprecationWarning),
              patch('photutils.datasets.load._get_path',
                    return_value=str(filepath)) as mock_get_path):
            hdu = load.load_star_image(show_progress=True)

        assert isinstance(hdu, fits.ImageHDU)
        mock_get_path.assert_called_once_with(
            'M6707HH.fits',
            location='remote',
            show_progress=True,
        )

    def test_load_simulated_hst_star_image(self, tmp_path):
        """
        Test the load_simulated_hst_star_image function with mocked file
        download.
        """
        data = np.random.default_rng(seed=0).random((300, 300))
        header = fits.Header()
        header['TELESCOP'] = 'HST'
        header['INSTRUME'] = 'WFC3'
        header['FILTER'] = 'F160W'

        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([primary_hdu])

        filepath = tmp_path / 'hst_wfc3ir_f160w_simulated_starfield.fits'
        hdulist.writeto(filepath)

        with patch('photutils.datasets.load._get_path',
                   return_value=str(filepath)):
            hdu = load.load_simulated_hst_star_image()

        assert isinstance(hdu, fits.ImageHDU)
        assert np.array_equal(hdu.data, data)
        assert hdu.header['TELESCOP'] == 'HST'
        assert hdu.header['INSTRUME'] == 'WFC3'
        assert hdu.header['FILTER'] == 'F160W'

    def test_load_simulated_hst_star_image_show_progress(self, tmp_path):
        """
        Test the load_simulated_hst_star_image function with
        show_progress=True.
        """
        data = np.random.default_rng(seed=0).random((300, 300))
        header = fits.Header()

        primary_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([primary_hdu])

        filepath = tmp_path / 'hst_wfc3ir_f160w_simulated_starfield.fits'
        hdulist.writeto(filepath)

        with (
            patch('photutils.datasets.load._get_path',
                  return_value=str(filepath)) as mock_get_path,
        ):
            hdu = load.load_simulated_hst_star_image(show_progress=True)

        assert isinstance(hdu, fits.ImageHDU)
        mock_get_path.assert_called_once_with(
            'hst_wfc3ir_f160w_simulated_starfield.fits',
            location='photutils-datasets',
            show_progress=True,
        )


def test_deprecated_get_path():
    """
    Test that the deprecated get_path function raises a warning and
    returns the expected path.
    """
    fn = '4gaussians_params.ecsv'
    with pytest.warns(AstropyDeprecationWarning):
        result = load.get_path(fn, location='local')
    assert fn in result
