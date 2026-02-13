# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the load module.
"""

from unittest.mock import patch
from urllib.error import URLError

import pytest

from photutils.datasets import get_path, load


def test_get_path():
    """
    Test get_path with a valid filename and location, and with an
    invalid location.
    """
    fn = '4gaussians_params.ecsv'
    path = get_path(fn, location='local')
    assert fn in path

    match = 'Invalid location:'
    with pytest.raises(ValueError, match=match):
        get_path('filename', location='invalid')


class TestGetPathCache:
    """
    Tests for the caching behavior of get_path.

    Tests that it correctly checks the cache for both the primary and
    fallback URLs, and that it falls back to downloading from the
    datasets URL if the primary URL is not cached and fails to download.
    """

    def setup_method(self):
        self.filename = 'test_file.fits'
        self.primary_url = (
            f'http://data.astropy.org/photometry/{self.filename}'
        )
        self.datasets_url = (
            'https://github.com/astropy/photutils-datasets/raw/'
            f'main/data/{self.filename}'
        )

    def test_cache_hit_primary_url(self):
        """
        Test that get_path uses the cached file when the primary URL is
        already in the cache, without trying the fallback URL.
        """
        with (
            patch('photutils.datasets.load.is_url_in_cache') as mock_cache,
            patch('photutils.datasets.load.download_file') as mock_dl,
        ):
            mock_cache.side_effect = (
                lambda url: url == self.primary_url
            )
            mock_dl.return_value = '/cached/path/test_file.fits'

            result = get_path(self.filename, location='remote')

            assert result == '/cached/path/test_file.fits'
            mock_dl.assert_called_once_with(
                self.primary_url, cache=True, show_progress=False,
            )

    def test_cache_hit_datasets_url(self):
        """
        Test that get_path uses the cached file when only the fallback
        datasets URL is in the cache.
        """
        with (
            patch('photutils.datasets.load.is_url_in_cache') as mock_cache,
            patch('photutils.datasets.load.download_file') as mock_dl,
        ):
            mock_cache.side_effect = (
                lambda url: url == self.datasets_url
            )
            mock_dl.return_value = '/cached/path/test_file.fits'

            result = get_path(self.filename, location='remote')

            assert result == '/cached/path/test_file.fits'
            mock_dl.assert_called_once_with(
                self.datasets_url, cache=True, show_progress=False,
            )

    def test_no_cache_falls_through_to_download(self):
        """
        Test that get_path tries the primary URL and falls back to the
        datasets URL when neither is cached and the primary fails.
        """
        with (
            patch('photutils.datasets.load.is_url_in_cache',
                  return_value=False),
            patch('photutils.datasets.load.download_file') as mock_dl,
        ):
            mock_dl.side_effect = [
                URLError('timeout'),
                '/downloaded/path/test_file.fits',
            ]

            result = get_path(self.filename, location='remote')

            assert result == '/downloaded/path/test_file.fits'
            assert mock_dl.call_count == 2


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
    hdu = load.load_star_image()
    assert len(hdu.header) == 106
    assert hdu.data.shape == (1059, 1059)
