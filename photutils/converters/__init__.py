# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ASDF converters.

The aperture converters need only ``asdf``. The PSF converters are built
on the ``asdf-astropy`` transform machinery, so they are defined only
when that optional dependency is installed.
"""

_ASDF_INSTALLED = True
_ASDF_ASTROPY_INSTALLED = True

try:
    import asdf  # noqa: F401
except ImportError:
    _ASDF_INSTALLED = False

try:
    import asdf_astropy  # noqa: F401
except ImportError:
    _ASDF_ASTROPY_INSTALLED = False

__all__ = []

if _ASDF_INSTALLED:
    from .apertures import (
        CircularAnnulusConverter,
        CircularApertureConverter,
        EllipticalAnnulusConverter,
        EllipticalApertureConverter,
        PolygonApertureConverter,
        RectangularAnnulusConverter,
        RectangularApertureConverter,
    )

    __all__ += [
        'CircularAnnulusConverter',
        'CircularApertureConverter',
        'EllipticalAnnulusConverter',
        'EllipticalApertureConverter',
        'PolygonApertureConverter',
        'RectangularAnnulusConverter',
        'RectangularApertureConverter',
    ]

if _ASDF_ASTROPY_INSTALLED:
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
        GriddedPSFModelConverter,
        ImagePSFConverter,
        STDPSFGridConverter,
    )

    __all__ += [
        'AiryDiskPSFConverter',
        'CircularGaussianPRFConverter',
        'CircularGaussianPSFConverter',
        'CircularGaussianSigmaPRFConverter',
        'GaussianPRFConverter',
        'GaussianPSFConverter',
        'GriddedPSFModelConverter',
        'ImagePSFConverter',
        'MoffatPSFConverter',
        'STDPSFGridConverter',
    ]
