# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ASDF converters.
"""

_ASDF_ASTROPY_INSTALLED = True

try:
    import asdf_astropy  # noqa: F401 -- needed to register the converters
except ImportError:
    _ASDF_ASTROPY_INSTALLED = False

if _ASDF_ASTROPY_INSTALLED:
    from .apertures import (
        CircularApertureConverter,
        CircularAnnulusConverter,
        EllipticalApertureConverter,
        EllipticalAnnulusConverter,
        PolygonApertureConverter,
        RectangularApertureConverter,
        RectangularAnnulusConverter,
    )
    from .functional_models import (
        AiryDiskPSFConverter,
        CircularGaussianPRFConverter,
        CircularGaussianPSFConverter,
        CircularGaussianSigmaPRFConverter,
        GaussianPRFConverter,
        GaussianPSFConverter,
        MoffatPSFConverter,
    )
    from .image_models import (
        GriddedPSFConverter,
        ImagePSFConverter,
        STDPSFGridConverter,
    )


__all__ = [
    'AiryDiskPSFConverter',
    'CircularAnnulusConverter',
    'CircularApertureConverter',
    'CircularGaussianPRFConverter',
    'CircularGaussianPSFConverter',
    'CircularGaussianSigmaPRFConverter',
    'EllipticalAnnulusConverter',
    'EllipticalApertureConverter',
    'GaussianPRFConverter',
    'GaussianPSFConverter',
    'GriddedPSFConverter',
    'ImagePSFConverter',
    'MoffatPSFConverter',
    'PolygonApertureConverter',
    'RectangularAnnulusConverter',
    'RectangularApertureConverter',
    'STDPSFGridConverter',
]
