# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module was deprecated in version 1.13.0 and will be removed in
version 1.15.0 (or 2.0.0).
"""

import warnings

from photutils.datasets import examples, images, noise, sources, wcs

__depr__ = {}
__depr__[examples] = ['make_4gaussians_image', 'make_100gaussians_image']
__depr__[images] = ['make_model_sources_image', 'make_gaussian_sources_image',
                    'make_gaussian_prf_sources_image', 'make_test_psf_data']
__depr__[noise] = ['apply_poisson_noise', 'make_noise_image']
__depr__[sources] = ['make_random_models_table', 'make_random_gaussians_table']
__depr__[wcs] = ['make_wcs', 'make_gwcs', 'make_imagehdu']


__depr_mesg__ = ('`photutils.datasets.make.{attr}` is a deprecated alias for '
                 '`{module}.{attr}` and will be removed in the future. '
                 'Instead, please use `from {module} import {attr}` '
                 'to silence this warning.')

__depr_attrs__ = {}
for k, vals in __depr__.items():
    for val in vals:
        __depr_attrs__[val] = (getattr(k, val),
                               __depr_mesg__.format(
                                   module='photutils.datasets',
                                   attr=val))
del k, val, vals  # pylint: disable=W0631


def __getattr__(attr):  # pragma: no cover
    if attr in __depr_attrs__:
        obj, message = __depr_attrs__[attr]
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return obj
    raise AttributeError(f'module {__name__!r} has no attribute {attr!r}')


message = ('photutils.datasets.make is deprecated and will be removed in '
           'a future version. Instead, please import functions from '
           'photutils.datasets')
warnings.warn(message, DeprecationWarning, stacklevel=2)
