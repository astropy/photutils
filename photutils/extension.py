# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ASDF extension for photutils.
"""
import importlib.resources as importlib_resources

from asdf.extension import ManifestExtension
from asdf.resource import DirectoryResourceMapping

from .converters import _ASDF_ASTROPY_INSTALLED, apertures

__all__ = [
    'PHOTUTILS_APERTURE_CONVERTERS',
    'PHOTUTILS_CONVERTERS',
    'PHOTUTILS_MANIFEST_URIS',
    'PHOTUTILS_PSF_CONVERTERS',
    'get_extensions',
    'get_resource_mappings',
]

PHOTUTILS_APERTURE_CONVERTERS = [
    apertures.CircularApertureConverter(),
    apertures.CircularAnnulusConverter(),
    apertures.EllipticalApertureConverter(),
    apertures.EllipticalAnnulusConverter(),
    apertures.PolygonApertureConverter(),
    apertures.RectangularApertureConverter(),
    apertures.RectangularAnnulusConverter(),
]

# The PSF converters are built on the asdf-astropy transform machinery.
# asdf rejects an entire extension that contains a converter which does
# not implement its interface, so registering the PSF converters when
# asdf-astropy is missing would also disable the aperture converters,
# which need only asdf.
if _ASDF_ASTROPY_INSTALLED:
    from .converters import functional_models, image_models

    PHOTUTILS_PSF_CONVERTERS = [
        functional_models.AiryDiskPSFConverter(),
        functional_models.CircularGaussianPRFConverter(),
        functional_models.CircularGaussianPSFConverter(),
        functional_models.CircularGaussianSigmaPRFConverter(),
        functional_models.GaussianPRFConverter(),
        functional_models.GaussianPSFConverter(),
        functional_models.MoffatPSFConverter(),
        image_models.ImagePSFConverter(),
        image_models.GriddedPSFModelConverter(),
        image_models.STDPSFGridConverter(),
    ]
else:
    PHOTUTILS_PSF_CONVERTERS = []

PHOTUTILS_CONVERTERS = PHOTUTILS_PSF_CONVERTERS + PHOTUTILS_APERTURE_CONVERTERS


# The order here is important; asdf will prefer to use extensions
# that occur earlier in the list.
PHOTUTILS_MANIFEST_URIS = [
    'asdf://astropy.org/photutils/manifests/photutils-1.0.0',
]


def get_extensions():
    """
    Get the photutils extension.

    This method is registered with the asdf.extensions entry point.

    Returns
    -------
    extensions : list
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
    Get the resource mapping instances for the photutils schemas and
    manifests.

    This method is registered with the asdf.resource_mappings entry
    point.

    Returns
    -------
    mappings : list
        A list of `collections.abc.Mapping` of ASDF resource mappings.
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
