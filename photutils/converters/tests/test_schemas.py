# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils ASDF manifests and schemas.
"""

import re
import textwrap

import asdf
import numpy as np
import pytest
import yaml
from asdf import get_config, treeutil
from asdf.exceptions import ValidationError
from asdf.testing.helpers import yaml_to_asdf
from astropy import units as u
from astropy.nddata import NDData
from numpy.testing import assert_array_equal

from photutils.converters import _ASDF_ASTROPY_INSTALLED
from photutils.extension import PHOTUTILS_MANIFEST_URIS
from photutils.psf import ImagePSF

# Matches the top-level YAML tag that opens a schema example.
EXAMPLE_TAG_PATTERN = re.compile(r'!<([^>]+)>')

PIXEL_POSITIONS = '[10.0, 20.0]'

SKY_POSITIONS = """!<tag:astropy.org:astropy/coordinates/skycoord-1.0.0>
      dec: !<tag:astropy.org:astropy/coordinates/latitude-1.2.0>
        {unit: !unit/unit-1.0.0 deg, value: 20.0}
      frame: icrs
      ra: !<tag:astropy.org:astropy/coordinates/longitude-1.2.0>
        {unit: !unit/unit-1.0.0 deg, value: 10.0,
         wrap_angle: !<tag:astropy.org:astropy/coordinates/angle-1.2.0>
           {unit: !unit/unit-1.0.0 deg, value: 360.0}}
      representation_type: spherical"""

VERTICES = '[[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0]]'
PIXEL_OFFSETS = f'!core/ndarray-1.1.0 {VERTICES}'
SKY_OFFSETS = ('!unit/quantity-1.3.0 {unit: !unit/unit-1.0.0 arcsec, '
               f'value: !core/ndarray-1.1.0 {VERTICES}}}')

# The parameters of each aperture that must agree with ``positions``.
# ``theta`` is excluded because the pixel apertures also accept an
# angular quantity.
SIZE_PARAMS = {
    'circular_aperture': ('r',),
    'circular_annulus': ('r_in', 'r_out'),
    'elliptical_aperture': ('a', 'b'),
    'elliptical_annulus': ('a_in', 'a_out', 'b_in', 'b_out'),
    'polygon_aperture': ('vertex_offsets',),
    'rectangular_aperture': ('w', 'h'),
    'rectangular_annulus': ('w_in', 'w_out', 'h_in', 'h_out'),
}


def _load_resource(uri):
    """
    Load a YAML resource registered with ASDF.
    """
    return yaml.safe_load(get_config().resource_manager[uri])


def _iter_examples(schema):
    """
    Yield each entry of every ``examples`` block in a schema.
    """
    for node in treeutil.iter_tree(schema):
        if isinstance(node, dict) and isinstance(node.get('examples'), list):
            yield from node['examples']


def _example_text(example):
    """
    Return the example source from a schema ``examples`` entry.

    Entries are ``[description, text]``, or ``[description, version,
    text]`` when an ASDF Standard version is pinned.
    """
    return example[2] if len(example) > 2 else example[-1]


def _manifest_tags():
    """
    Map each tag URI defined by the photutils manifests to its schema.
    """
    tags = {}
    for manifest_uri in PHOTUTILS_MANIFEST_URIS:
        for tag in _load_resource(manifest_uri)['tags']:
            tags[tag['tag_uri']] = tag['schema_uri']
    return tags


def _schema_example_tags(schema):
    """
    Yield the ``(index, tag)`` of every example in a schema.

    ``tag`` is `None` if the example is not tagged.
    """
    for index, example in enumerate(_iter_examples(schema)):
        text = _example_text(example).strip()
        match = EXAMPLE_TAG_PATTERN.match(text)
        yield index, match[1] if match else None


def _example_tags(manifest_tags):
    """
    Collect the top-level tag of every example in every schema.

    Returns a list of ``(schema_uri, example_index, tag)`` tuples, where
    ``tag`` is `None` if the example is not tagged.
    """
    examples = []
    for schema_uri in dict.fromkeys(manifest_tags.values()):
        schema = _load_resource(schema_uri)
        examples.extend((schema_uri, index, tag)
                        for index, tag in _schema_example_tags(schema))
    return examples


CIRCULAR_APERTURE_URI = ('asdf://astropy.org/photutils/schemas/aperture/'
                         'circular_aperture-1.0.0')
CIRCULAR_ANNULUS_TAG = ('tag:astropy.org:photutils/aperture/'
                        'circular_annulus-1.0.0')


def _circular_aperture_schema(example_tag):
    """
    Build a circular aperture schema whose single example carries
    ``example_tag``.

    ``example_tag`` is `None` for an untagged example.
    """
    body = 'positions: [1.0, 2.0]\nr: 5.0\n'
    if example_tag is not None:
        body = f'!<{example_tag}>\n' + textwrap.indent(body, '  ')
    example = textwrap.indent(body, ' ' * 6).rstrip()
    return f"""%YAML 1.1
---
$schema: "http://stsci.edu/schemas/yaml-schema/draft-01"
id: "{CIRCULAR_APERTURE_URI}"
title: Circular aperture
examples:
  -
    - A CircularAperture with a single position and radius of 5 pixels.
    - |
{example}
type: object
...
"""


MANIFEST_TAGS = _manifest_tags()
EXAMPLE_TAGS = _example_tags(MANIFEST_TAGS)
EXAMPLE_IDS = [f'{uri.rsplit("/", 1)[-1]}-example{index}'
               for uri, index, _ in EXAMPLE_TAGS]


def test_manifest_schemas_are_registered():
    """
    Test that every schema referenced by the manifests is registered.
    """
    assert MANIFEST_TAGS
    for tag_uri, schema_uri in MANIFEST_TAGS.items():
        schema = _load_resource(schema_uri)
        assert schema['id'] == schema_uri, tag_uri


@pytest.mark.parametrize(('schema_uri', 'index', 'tag'), EXAMPLE_TAGS,
                         ids=EXAMPLE_IDS)
def test_example_tag_in_manifest(schema_uri, index, tag):
    """
    Test that each schema example is tagged with the manifest tag of the
    schema that contains it.

    The ASDF schema example self-tests validate a tagged tree, so the
    schema applied to an example is selected by its tag. An unrecognized
    tag selects no schema at all, which makes the example self-test pass
    without validating anything.
    """
    assert tag is not None, f'{schema_uri} example {index} is untagged'
    assert tag in MANIFEST_TAGS, f'{tag} is not defined in the manifest'
    assert MANIFEST_TAGS[tag] == schema_uri, (
        f'{tag} is defined by a different schema')


@pytest.mark.parametrize(('example_tag', 'match'), [
    (None, 'example 0 is untagged'),
    ('tag:astropy.org:photutils/aperture/circular_apperture-1.0.0',
     'is not defined in the manifest'),
    (CIRCULAR_ANNULUS_TAG, 'is defined by a different schema')],
    ids=['untagged', 'misspelled', 'other_schema'])
def test_invalid_example_tag_is_detected(example_tag, match):
    """
    Test that a schema example carrying an invalid tag is rejected.

    The tag is extracted from the schema source, so this exercises the
    same path as the schemas registered by the manifests.
    """
    schema = yaml.safe_load(_circular_aperture_schema(example_tag))
    [(index, tag)] = _schema_example_tags(schema)
    assert tag == example_tag

    with pytest.raises(AssertionError, match=match):
        test_example_tag_in_manifest(schema['id'], index, tag)


def _quantity(value):
    """
    Build the YAML for a scalar angular quantity.
    """
    return ('!unit/quantity-1.3.0 '
            f'{{unit: !unit/unit-1.0.0 arcsec, value: {value}}}')


def _sizes(names, *, sky=False):
    """
    Build the size parameters of an aperture in pixel or sky form.
    """
    values = {}
    for index, name in enumerate(names):
        if name == 'vertex_offsets':
            values[name] = SKY_OFFSETS if sky else PIXEL_OFFSETS
        elif sky:
            values[name] = _quantity(index + 3.0)
        else:
            values[name] = f'{index + 3.0}'
    return values


def _aperture_yaml(stem, positions, sizes):
    """
    Build the tagged YAML for a photutils aperture.
    """
    params = {'positions': positions, **sizes}
    body = '\n'.join(f'  {key}: {value}' for key, value in params.items())
    return f'!<tag:astropy.org:photutils/aperture/{stem}-1.0.0>\n{body}'


def _mixed_apertures():
    """
    Build apertures that mix pixel values with angular quantities.
    """
    mixed = []
    for stem, names in SIZE_PARAMS.items():
        mixed.append((f'{stem}-sky-positions',
                      _aperture_yaml(stem, SKY_POSITIONS,
                                     _sizes(names, sky=False))))
        mixed.append((f'{stem}-sky-sizes',
                      _aperture_yaml(stem, PIXEL_POSITIONS,
                                     _sizes(names, sky=True))))
    return mixed


MIXED_APERTURES = _mixed_apertures()


@pytest.mark.parametrize('example', [case[1] for case in MIXED_APERTURES],
                         ids=[case[0] for case in MIXED_APERTURES])
def test_mixed_pixel_and_sky_rejected(example):
    """
    Test that an aperture mixing pixel values with angular quantities
    fails schema validation when read.

    The pixel and sky variants of each aperture share a tag, so this
    combination is excluded only by the ``oneOf`` in the schema. Valid
    pixel and sky files are covered by the round-trip tests in
    ``test_apertures.py``.
    """
    with pytest.raises(ValidationError), \
            asdf.open(yaml_to_asdf(f'aper: {example}')):
        pass


# The size parameters that each aperture schema requires. The remaining
# parameters of these apertures (``theta`` for all of them, plus the
# inner size of the annuli) are optional.
REQUIRED_SIZE_PARAMS = {
    'elliptical_aperture': ('a', 'b'),
    'elliptical_annulus': ('a_in', 'a_out', 'b_out'),
    'rectangular_aperture': ('w', 'h'),
    'rectangular_annulus': ('w_in', 'w_out', 'h_out'),
}


def _read_minimal_aperture(stem, *, sky):
    """
    Read an aperture from a file containing only its required
    parameters.
    """
    example = _aperture_yaml(stem,
                             SKY_POSITIONS if sky else PIXEL_POSITIONS,
                             _sizes(REQUIRED_SIZE_PARAMS[stem], sky=sky))
    with asdf.open(yaml_to_asdf(f'aper: {example}')) as af:
        return af['aper']


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('sky', [False, True], ids=['pixel', 'sky'])
@pytest.mark.parametrize('stem', list(REQUIRED_SIZE_PARAMS))
def test_optional_theta_uses_default(stem, sky):
    """
    Test that an aperture file that omits ``theta`` is read using the
    default rotation angle of its aperture class.

    ``theta`` is optional in both the aperture classes and the schemas,
    and its default differs between the pixel and sky apertures.
    """
    aperture = _read_minimal_aperture(stem, sky=sky)
    assert aperture.theta == (0.0 * u.deg if sky else 0.0)


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('sky', [False, True], ids=['pixel', 'sky'])
def test_optional_inner_size_uses_default(sky):
    """
    Test that an annulus file that omits its inner size is read using
    the inner size derived by its aperture class.

    ``b_in`` and ``h_in`` are optional in both the aperture classes and
    the schemas.
    """
    ellipse = _read_minimal_aperture('elliptical_annulus', sky=sky)
    assert ellipse.b_in == ellipse.b_out * ellipse.a_in / ellipse.a_out

    rectangle = _read_minimal_aperture('rectangular_annulus', sky=sky)
    assert (rectangle.h_in
            == rectangle.w_in * rectangle.h_out / rectangle.w_out)


@pytest.mark.parametrize('stem', list(REQUIRED_SIZE_PARAMS))
def test_sky_numeric_theta_rejected(stem):
    """
    Test that a sky aperture with a numeric ``theta`` fails schema
    validation when read.

    The sky apertures require an angular ``theta``, so a number is
    excluded by the sky branch of the ``oneOf`` in the schema.
    """
    sizes = _sizes(REQUIRED_SIZE_PARAMS[stem], sky=True)
    example = _aperture_yaml(stem, SKY_POSITIONS,
                             {**sizes, 'theta': '0.5'})
    with pytest.raises(ValidationError), \
            asdf.open(yaml_to_asdf(f'aper: {example}')):
        pass


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('stem', list(REQUIRED_SIZE_PARAMS))
def test_pixel_angular_theta_accepted(stem):
    """
    Test that a pixel aperture with an angular ``theta`` is read
    without error.

    The pixel apertures accept either a number or an angular quantity,
    so ``theta`` is constrained only in the sky branch of the ``oneOf``
    in the schema.
    """
    sizes = _sizes(REQUIRED_SIZE_PARAMS[stem], sky=False)
    example = _aperture_yaml(stem, PIXEL_POSITIONS,
                             {**sizes, 'theta': _quantity(0.5)})
    with asdf.open(yaml_to_asdf(f'aper: {example}')) as af:
        assert af['aper'].theta == 0.5 * u.arcsec


@pytest.mark.parametrize('stem', list(SIZE_PARAMS))
def test_unknown_property_rejected(stem):
    """
    Test that an aperture with an unknown property fails schema
    validation when read.

    Without ``additionalProperties: false`` an unknown key is accepted
    and then silently dropped, so a misspelled parameter would read
    back as an aperture with default values.
    """
    params = _sizes(SIZE_PARAMS[stem], sky=False)
    params['not_a_parameter'] = '999.0'
    example = _aperture_yaml(stem, PIXEL_POSITIONS, params)
    with pytest.raises(ValidationError), \
            asdf.open(yaml_to_asdf(f'aper: {example}')):
        pass


def _ndarray(values):
    """
    Build the inline YAML for an ndarray.
    """
    return f'!core/ndarray-1.1.0 {np.asarray(values).tolist()}'


# A 4x4 image, the smallest accepted by the image-based PSF models,
# four copies of it (one per grid position), and the grid positions.
PSF_IMAGE_VALUES = np.arange(16.0).reshape((4, 4))
PSF_GRID_VALUES = np.repeat(PSF_IMAGE_VALUES[np.newaxis], 4, axis=0)
GRID_XYPOS_VALUES = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

PSF_IMAGE = _ndarray(PSF_IMAGE_VALUES)
PSF_GRID = _ndarray(PSF_GRID_VALUES)
GRID_XYPOS = _ndarray(GRID_XYPOS_VALUES)

# The parameters that each PSF schema requires. The remaining
# parameters are optional; ``bbox_factor`` is optional in every
# functional-model schema, as is ``theta`` for the elliptical Gaussians.
REQUIRED_PSF_PARAMS = {
    'airy_disk_psf': {'flux': 1.0, 'x_0': 2.0, 'y_0': 3.0, 'radius': 4.0},
    'circular_gaussian_prf': {'flux': 1.0, 'x_0': 2.0, 'y_0': 3.0,
                              'fwhm': 4.0},
    'circular_gaussian_psf': {'flux': 1.0, 'x_0': 2.0, 'y_0': 3.0,
                              'fwhm': 4.0},
    'circular_gaussian_sigma_prf': {'flux': 1.0, 'x_0': 2.0, 'y_0': 3.0,
                                    'sigma': 4.0},
    'gaussian_prf': {'flux': 1.0, 'x_0': 2.0, 'y_0': 3.0, 'x_fwhm': 4.0,
                     'y_fwhm': 5.0},
    'gaussian_psf': {'flux': 1.0, 'x_0': 2.0, 'y_0': 3.0, 'x_fwhm': 4.0,
                     'y_fwhm': 5.0},
    'moffat_psf': {'flux': 1.0, 'x_0': 2.0, 'y_0': 3.0, 'alpha': 4.0,
                   'beta': 5.0},
    'image_psf': {'data': PSF_IMAGE},
    'gridded_psf_model': {'data': PSF_GRID,
                          'oversampling': '!core/ndarray-1.1.0 [1, 1]',
                          'grid_xypos': GRID_XYPOS,
                          'meta': '{}'},
    'stdpsf_grid': {'data': PSF_GRID,
                    'grid_xypos': GRID_XYPOS,
                    'grid_shape': '[2, 2]',
                    'oversampling': '[4, 4]',
                    'meta': '{}'},
}

# The models whose parameters are all plain numbers, so that a reference
# model can be built from the same required parameters.
FUNCTIONAL_PSF_STEMS = list(REQUIRED_PSF_PARAMS)[:7]


def _psf_yaml(stem, params):
    """
    Build the tagged YAML for a photutils PSF model.
    """
    body = '\n'.join(f'  {key}: {value}' for key, value in params.items())
    return f'!<tag:astropy.org:photutils/psf/{stem}-1.0.0>\n{body}'


def _read_psf(stem, params):
    """
    Read a PSF model from a file containing only ``params``.
    """
    with asdf.open(yaml_to_asdf(f'psf: {_psf_yaml(stem, params)}')) as af:
        return af['psf']


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('stem', FUNCTIONAL_PSF_STEMS)
def test_optional_psf_params_use_defaults(stem):
    """
    Test that a PSF file containing only the parameters required by its
    schema is read using the defaults of its model class.

    ``bbox_factor`` is optional in every functional-model schema, as is
    ``theta`` for the elliptical Gaussian models, so a schema-valid file
    may omit them.
    """
    params = REQUIRED_PSF_PARAMS[stem]
    model = _read_psf(stem, params)
    reference = type(model)(**params)

    for name, value in params.items():
        assert getattr(model, name) == value
    assert model.bbox_factor == reference.bbox_factor
    if hasattr(model, 'theta'):
        assert model.theta == reference.theta


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_optional_image_psf_params_use_defaults():
    """
    Test that an ImagePSF file containing only ``data`` is read using
    the defaults of the model class.
    """
    model = _read_psf('image_psf', REQUIRED_PSF_PARAMS['image_psf'])
    reference = ImagePSF(data=PSF_IMAGE_VALUES)

    assert_array_equal(model.data, PSF_IMAGE_VALUES)
    for name in ('flux', 'x_0', 'y_0', 'fill_value'):
        assert getattr(model, name) == getattr(reference, name)
    assert_array_equal(model.oversampling, reference.oversampling)
    assert_array_equal(model.origin, reference.origin)


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_optional_gridded_psf_model_params_use_defaults():
    """
    Test that a GriddedPSFModel file containing only the parameters
    required by its schema is read using the defaults of the model
    class.
    """
    from photutils.psf import GriddedPSFModel

    model = _read_psf('gridded_psf_model',
                      REQUIRED_PSF_PARAMS['gridded_psf_model'])
    nddata = NDData(data=PSF_GRID_VALUES,
                    meta={'oversampling': 1, 'grid_xypos': GRID_XYPOS_VALUES})
    reference = GriddedPSFModel(nddata=nddata)

    assert_array_equal(model.data, PSF_GRID_VALUES)
    for name in ('flux', 'x_0', 'y_0', 'fill_value'):
        assert getattr(model, name) == getattr(reference, name)


@pytest.mark.parametrize('stem', list(REQUIRED_PSF_PARAMS))
def test_unknown_psf_property_rejected(stem):
    """
    Test that a PSF model with an unknown property fails schema
    validation when read.

    Without ``additionalProperties: false`` an unknown key is accepted
    and then silently dropped, so a misspelled parameter would read back
    as a model with default values. The properties of the transform
    schema referenced by the PSF models are repeated in each schema so
    that they remain allowed.
    """
    params = {**REQUIRED_PSF_PARAMS[stem], 'not_a_parameter': '999.0'}
    with pytest.raises(ValidationError), \
            asdf.open(yaml_to_asdf(f'psf: {_psf_yaml(stem, params)}')):
        pass


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('stem', FUNCTIONAL_PSF_STEMS)
def test_transform_properties_accepted(stem):
    """
    Test that the properties of the referenced transform schema are
    accepted by the PSF schemas.

    They are repeated in each schema so that ``additionalProperties``
    rejects only unknown keys.
    """
    params = {**REQUIRED_PSF_PARAMS[stem], 'name': 'psf',
              'inputs': '[x, y]', 'outputs': '[z]'}
    model = _read_psf(stem, params)
    assert model.name == 'psf'
    assert model.inputs == ('x', 'y')
    assert model.outputs == ('z',)
