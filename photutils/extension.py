# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ASDF extension for photutils.
"""
import importlib.resources as importlib_resources

from asdf.extension import ManifestExtension
from asdf.resource import DirectoryResourceMapping

from .converters import apertures, functional_models

__all__ = [
    'PHOTUTILS_APERTURE_CONVERTERS',
    'PHOTUTILS_MANIFEST_URIS',
    'PHOTUTILS_PSF_CONVERTERS',
]

PHOTUTILS_PSF_CONVERTERS = [
    functional_models.AiryDiskPSFConverter(),
    functional_models.CircularGaussianPRFConverter(),
    functional_models.CircularGaussianPSFConverter(),
    functional_models.CircularGaussianSigmaPRFConverter(),
    functional_models.GaussianPRFConverter(),
    functional_models.GaussianPSFConverter(),
    functional_models.MoffatPSFConverter(),
]

PHOTUTILS_APERTURE_CONVERTERS = [
    apertures.CircularApertureConverter(),
]

PHOTUTILS_CONVERTERS = PHOTUTILS_PSF_CONVERTERS + PHOTUTILS_APERTURE_CONVERTERS


# The order here is important; asdf will prefer to use extensions
# that occur earlier in the list.
PHOTUTILS_MANIFEST_URIS = [
    'asdf://astropy.org/photutils/manifests/photutils-1.0.0',
]


def get_extensions():
    """
    Get the gwcs.converters extension.
    This method is registered with the asdf.extensions entry point.

    Returns
    -------
    list
        A list of ASDF extensions.
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
    list
        A list of collections.abc.Mapping of ASDF resource mappings.
    """
    from . import resources

    resources_root = importlib_resources.files(resources)

    return [
        DirectoryResourceMapping(resources_root / 'schemas',
                                 'asdf://astropy.org/photutils/schemas',
                                 recursive=True),
        DirectoryResourceMapping(resources_root / 'manifests',
                                 'asdf://astropy.org/photutils/manifests'),
    ]
