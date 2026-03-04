# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ASDF extension for photutils.
"""
import importlib.resources as importlib_resources

from asdf.extension import ManifestExtension
from asdf.resource import DirectoryResourceMapping

from .converters.functional_models import AiryDiskPSFConverter
from .converters.apertures import CircularApertureConverter


__all__ = [
    "PHOTUTILS_PSF_CONVERTERS",
    "PHOTUTILS_APERTURE_CONVERTERS",
    "PHOTUTILS_MANIFEST_URIS",
]

PHOTUTILS_PSF_CONVERTERS = [
    AiryDiskPSFConverter(),
]

PHOTUTILS_APERTURE_CONVERTERS = [
    CircularApertureConverter(),

]

PHOTUTILS_CONVERTERS = PHOTUTILS_PSF_CONVERTERS + PHOTUTILS_APERTURE_CONVERTERS


# The order here is important; asdf will prefer to use extensions
# that occur earlier in the list.
PHOTUTILS_MANIFEST_URIS = [
    "asdf://astropy.org/photutils/manifests/photutils-1.0.0"
]


def get_extensions():
    """
    Get the gwcs.converters extension.
    This method is registered with the asdf.extensions entry point.
    Returns
    -------
    list of asdf.extension.Extension
    """
    return [
        ManifestExtension.from_uri(
            uri,
            converters=PHOTUTILS_CONVERTERS,
        )
        for uri in PHOTUTILS_MANIFEST_URIS
    ]


def get_resource_mappings():
    """
    Get the resource mapping instances for the photutils schemas
    and manifests.  This method is registered with the
    asdf.resource_mappings entry point.

    Returns
    -------
    list of collections.abc.Mapping
    """
    from . import resources

    resources_root = importlib_resources.files(resources)

    return [
        DirectoryResourceMapping(resources_root / "schemas",
                                 "asdf://astropy.org/photutils/schemas",
                                 recursive=True),
        DirectoryResourceMapping(resources_root / "manifests",
                                 "asdf://astropy.org/photutils/manifests"),
    ]
