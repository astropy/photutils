# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils ASDF extension.
"""

import yaml
from asdf import get_config
from asdf.extension import Converter, ExtensionProxy, ManifestExtension

from photutils import converters
from photutils.converters import _ASDF_ASTROPY_INSTALLED
from photutils.extension import (PHOTUTILS_APERTURE_CONVERTERS,
                                 PHOTUTILS_CONVERTERS, PHOTUTILS_MANIFEST_URIS,
                                 PHOTUTILS_PSF_CONVERTERS, get_extensions,
                                 get_resource_mappings)


def _manifest_tags():
    """
    Return the set of tag URIs defined by the photutils manifests.
    """
    resource_manager = get_config().resource_manager
    tags = set()
    for manifest_uri in PHOTUTILS_MANIFEST_URIS:
        manifest = yaml.safe_load(resource_manager[manifest_uri])
        tags.update(tag['tag_uri'] for tag in manifest['tags'])
    return tags


def _converter_tag_prefixes(converters_):
    """
    Return the set of unversioned tag URIs handled by ``converters_``.
    """
    return {tag.removesuffix('-*')
            for converter in converters_ for tag in converter.tags}


def test_converters_implement_asdf_interface():
    """
    Test that every registered converter implements the asdf Converter
    interface.

    asdf rejects an entire extension that contains a converter which
    does not, so a single invalid converter would disable all photutils
    ASDF support.
    """
    assert PHOTUTILS_CONVERTERS
    for converter in PHOTUTILS_CONVERTERS:
        assert isinstance(converter, Converter)


def test_extension_loads():
    """
    Test that the photutils extension is accepted by asdf.
    """
    extensions = get_extensions()
    assert len(extensions) == len(PHOTUTILS_MANIFEST_URIS)
    for extension in extensions:
        proxy = ExtensionProxy(extension)
        assert len(proxy.converters) == len(PHOTUTILS_CONVERTERS)


def test_aperture_converters_do_not_need_asdf_astropy():
    """
    Test that the aperture converters alone form a valid extension.

    The aperture converters need only asdf, so they remain available
    when the optional asdf-astropy package is not installed.
    """
    assert PHOTUTILS_APERTURE_CONVERTERS
    extension = ManifestExtension.from_uri(
        PHOTUTILS_MANIFEST_URIS[0], converters=PHOTUTILS_APERTURE_CONVERTERS)
    proxy = ExtensionProxy(extension)
    assert len(proxy.converters) == len(PHOTUTILS_APERTURE_CONVERTERS)


def test_psf_converters_require_asdf_astropy():
    """
    Test that the PSF converters are registered only when asdf-astropy
    is installed.
    """
    if _ASDF_ASTROPY_INSTALLED:
        assert PHOTUTILS_PSF_CONVERTERS
    else:
        assert PHOTUTILS_PSF_CONVERTERS == []


def test_converter_tags_are_defined_by_the_manifest():
    """
    Test that every tag handled by a converter is defined by a manifest.
    """
    assert _converter_tag_prefixes(PHOTUTILS_CONVERTERS)
    manifest_tags = _manifest_tags()
    for prefix in _converter_tag_prefixes(PHOTUTILS_CONVERTERS):
        assert any(tag.startswith(prefix) for tag in manifest_tags), prefix


def test_manifest_tags_all_have_converters():
    """
    Test that every tag defined by a manifest has a converter.
    """
    if not _ASDF_ASTROPY_INSTALLED:
        # the PSF tags are intentionally left without a converter
        return

    prefixes = _converter_tag_prefixes(PHOTUTILS_CONVERTERS)
    for tag in _manifest_tags():
        assert any(tag.startswith(prefix) for prefix in prefixes), tag


def test_resource_mappings_register_all_files():
    """
    Test that every schema and manifest file is registered with asdf and
    resolves to a resource with a matching ``id``.
    """
    resource_manager = get_config().resource_manager

    uris = set()
    for mapping in get_resource_mappings():
        uris.update(mapping)
    assert uris

    for uri in uris:
        assert uri in resource_manager
        assert yaml.safe_load(resource_manager[uri])['id'] == uri


def test_public_names_are_importable():
    """
    Test that every name in ``__all__`` is defined.

    The converters are imported conditionally, so ``__all__`` must be
    built to match.
    """
    assert converters.__all__
    for name in converters.__all__:
        assert hasattr(converters, name)
