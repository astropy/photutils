# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ASDF converters.

The aperture converters need only ``asdf``. The PSF converters are built
on the ``asdf-astropy`` transform machinery, so they are defined only
when that optional dependency is installed.
"""

from importlib.util import find_spec

_ASDF_INSTALLED = find_spec('asdf') is not None

# Because asdf-astropy imports asdf, it is unusable without it.
_ASDF_ASTROPY_INSTALLED = (_ASDF_INSTALLED
                           and find_spec('asdf_astropy') is not None)

__all__ = []

if _ASDF_INSTALLED:
    from .apertures import (CircularAnnulusConverter,
                            CircularApertureConverter,
                            EllipticalAnnulusConverter,
                            EllipticalApertureConverter,
                            PolygonApertureConverter,
                            RectangularAnnulusConverter,
                            RectangularApertureConverter)

    __all__ += ['CircularAnnulusConverter',
                'CircularApertureConverter',
                'EllipticalAnnulusConverter',
                'EllipticalApertureConverter',
                'PolygonApertureConverter',
                'RectangularAnnulusConverter',
                'RectangularApertureConverter',
                ]

if _ASDF_ASTROPY_INSTALLED:
    from .functional_models import (AiryDiskPSFConverter,
                                    CircularGaussianPRFConverter,
                                    CircularGaussianPSFConverter,
                                    CircularGaussianSigmaPRFConverter,
                                    GaussianPRFConverter, GaussianPSFConverter,
                                    MoffatPSFConverter)
    from .image_models import (GriddedPSFModelConverter, ImagePSFConverter,
                               STDPSFGridConverter)

    __all__ += ['AiryDiskPSFConverter',
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
