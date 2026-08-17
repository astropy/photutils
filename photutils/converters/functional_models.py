# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Converters to and from the ASDF format for photutils.psf.functional_models.
"""

from asdf_astropy.converters.transform.core import (TransformConverterBase,
                                                    parameter_to_value)

from photutils.converters._utils import optional_params

__all__ = ['AiryDiskPSFConverter',
           'CircularGaussianPRFConverter',
           'CircularGaussianPSFConverter',
           'CircularGaussianSigmaPRFConverter',
           'GaussianPRFConverter',
           'GaussianPSFConverter',
           'MoffatPSFConverter',
           ]


class _PSFModelConverter(TransformConverterBase):
    """
    Base class for the photutils PSF model converters.

    Subclasses name the model parameters that describe the profile;
    ``flux``, ``x_0``, ``y_0``, and ``bbox_factor`` are common to every
    model and are handled here.

    Every parameter is written to the file. Those in
    ``optional_model_params``, and ``bbox_factor``, may be absent when
    reading, in which case the model class supplies its default.
    """

    model_params = ()
    optional_model_params = ()

    def _model_class(self):
        """
        Return the model class handled by this converter.
        """
        from photutils import psf

        return getattr(psf, self.types[0].rsplit('.', 1)[-1])

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        names = ('flux', 'x_0', 'y_0', *self.model_params,
                 *self.optional_model_params)
        node = {name: parameter_to_value(getattr(model, name))
                for name in names}
        node['bbox_factor'] = model.bbox_factor
        return node

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        names = ('flux', 'x_0', 'y_0', *self.model_params)
        params = {name: node[name] for name in names}

        model = self._model_class()(
            **params,
            **optional_params(node, *self.optional_model_params,
                              'bbox_factor'),
        )

        # The file stores only the fixed=True entries and the non-empty
        # bounds, relative to a baseline of fixed=False and bounds of
        # (None, None). The photutils models default their shape
        # parameters to fixed=True with bounds set, so the constraints
        # are reset to that baseline here. The caller then applies the
        # stored constraints on top, which reconstructs the saved state
        # exactly.
        for name in model.param_names:
            model.fixed[name] = False
            model.bounds[name] = (None, None)

        return model


class AiryDiskPSFConverter(_PSFModelConverter):
    """
    ASDF converter for AiryDiskPSF.
    """

    tags = ('tag:astropy.org:photutils/psf/airy_disk_psf-*',)
    types = ('photutils.psf.AiryDiskPSF',)
    model_params = ('radius',)


class CircularGaussianPRFConverter(_PSFModelConverter):
    """
    ASDF converter for CircularGaussianPRF.
    """

    tags = ('tag:astropy.org:photutils/psf/circular_gaussian_prf-*',)
    types = ('photutils.psf.CircularGaussianPRF',)
    model_params = ('fwhm',)


class CircularGaussianPSFConverter(_PSFModelConverter):
    """
    ASDF converter for CircularGaussianPSF.
    """

    tags = ('tag:astropy.org:photutils/psf/circular_gaussian_psf-*',)
    types = ('photutils.psf.CircularGaussianPSF',)
    model_params = ('fwhm',)


class CircularGaussianSigmaPRFConverter(_PSFModelConverter):
    """
    ASDF converter for CircularGaussianSigmaPRF.
    """

    tags = ('tag:astropy.org:photutils/psf/circular_gaussian_sigma_prf-*',)
    types = ('photutils.psf.CircularGaussianSigmaPRF',)
    model_params = ('sigma',)


class GaussianPRFConverter(_PSFModelConverter):
    """
    ASDF converter for GaussianPRF.
    """

    tags = ('tag:astropy.org:photutils/psf/gaussian_prf-*',)
    types = ('photutils.psf.GaussianPRF',)
    model_params = ('x_fwhm', 'y_fwhm')
    optional_model_params = ('theta',)


class GaussianPSFConverter(_PSFModelConverter):
    """
    ASDF converter for GaussianPSF.
    """

    tags = ('tag:astropy.org:photutils/psf/gaussian_psf-*',)
    types = ('photutils.psf.GaussianPSF',)
    model_params = ('x_fwhm', 'y_fwhm')
    optional_model_params = ('theta',)


class MoffatPSFConverter(_PSFModelConverter):
    """
    ASDF converter for MoffatPSF.
    """

    tags = ('tag:astropy.org:photutils/psf/moffat_psf-*',)
    types = ('photutils.psf.MoffatPSF',)
    model_params = ('alpha', 'beta')
