# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Converters to and from the ASDF format for photutils.psf.functional_models.
"""

from asdf_astropy.converters.transform.core import (TransformConverterBase,
                                                    parameter_to_value)

__all__ = ['AiryDiskPSFConverter',
           'CircularGaussianPRFConverter',
           'CircularGaussianPSFConverter',
           'CircularGaussianSigmaPRFConverter',
           'GaussianPRFConverter',
           'GaussianPSFConverter',
           'MoffatPSFConverter',
           ]


class AiryDiskPSFConverter(TransformConverterBase):
    """
    Converter for AiryDiskPSF.
    """

    tags = ('tag:astropy.org:photutils/psf/airy_disk_psf-*',)
    types = ('photutils.psf.AiryDiskPSF',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'radius': parameter_to_value(model.radius),
            'bbox_factor': model.bbox_factor,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import AiryDiskPSF

        return AiryDiskPSF(
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            radius=node['radius'],
            bbox_factor=node['bbox_factor'],
        )


class CircularGaussianPRFConverter(TransformConverterBase):
    """
    ASDF converter for CircularGaussianPRF model.
    """

    tags = ('tag:astropy.org:photutils/psf/circular_gaussian_prf-*',)
    types = ('photutils.psf.CircularGaussianPRF',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'fwhm': parameter_to_value(model.fwhm),
            'bbox_factor': model.bbox_factor,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import CircularGaussianPRF

        return CircularGaussianPRF(
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            fwhm=node['fwhm'],
            bbox_factor=node['bbox_factor'],
        )


class CircularGaussianPSFConverter(TransformConverterBase):
    """
    ASDF converter for CircularGaussianPSF model.
    """

    tags = ('tag:astropy.org:photutils/psf/circular_gaussian_psf-*',)
    types = ('photutils.psf.CircularGaussianPSF',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'fwhm': parameter_to_value(model.fwhm),
            'bbox_factor': model.bbox_factor,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import CircularGaussianPSF

        return CircularGaussianPSF(
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            fwhm=node['fwhm'],
            bbox_factor=node['bbox_factor'],
        )


class CircularGaussianSigmaPRFConverter(TransformConverterBase):
    """
    ASDF converter for CircularGaussianSigmaPRF model.
    """

    tags = ('tag:astropy.org:photutils/psf/circular_gaussian_sigma_prf-*',)
    types = ('photutils.psf.CircularGaussianSigmaPRF',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'sigma': parameter_to_value(model.sigma),
            'bbox_factor': model.bbox_factor,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import CircularGaussianSigmaPRF

        return CircularGaussianSigmaPRF(
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            sigma=node['sigma'],
            bbox_factor=node['bbox_factor'],
        )


class GaussianPRFConverter(TransformConverterBase):
    """
    ASDF converter for GaussianPRF model.
    """

    tags = ('tag:astropy.org:photutils/psf/gaussian_prf-*',)
    types = ('photutils.psf.GaussianPRF',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'x_fwhm': parameter_to_value(model.x_fwhm),
            'y_fwhm': parameter_to_value(model.y_fwhm),
            'theta': parameter_to_value(model.theta),
            'bbox_factor': model.bbox_factor,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import GaussianPRF

        return GaussianPRF(
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            x_fwhm=node['x_fwhm'],
            y_fwhm=node['y_fwhm'],
            theta=node['theta'],
            bbox_factor=node['bbox_factor'],
        )


class GaussianPSFConverter(TransformConverterBase):
    """
    ASDF converter for GaussianPSF model.
    """

    tags = ('tag:astropy.org:photutils/psf/gaussian_psf-*',)
    types = ('photutils.psf.GaussianPSF',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'x_fwhm': parameter_to_value(model.x_fwhm),
            'y_fwhm': parameter_to_value(model.y_fwhm),
            'theta': parameter_to_value(model.theta),
            'bbox_factor': model.bbox_factor,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import GaussianPSF

        return GaussianPSF(
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            x_fwhm=node['x_fwhm'],
            y_fwhm=node['y_fwhm'],
            theta=node['theta'],
            bbox_factor=node['bbox_factor'],
        )


class MoffatPSFConverter(TransformConverterBase):
    """
    ASDF converter for MoffatPSF model.
    """

    tags = ('tag:astropy.org:photutils/psf/moffat_psf-*',)
    types = ('photutils.psf.MoffatPSF',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'alpha': parameter_to_value(model.alpha),
            'beta': parameter_to_value(model.beta),
            'bbox_factor': model.bbox_factor,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import MoffatPSF

        return MoffatPSF(
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            alpha=node['alpha'],
            beta=node['beta'],
            bbox_factor=node['bbox_factor'],
        )
