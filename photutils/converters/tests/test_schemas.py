# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils ASDF manifests and schemas.
"""

import re
import textwrap

import asdf
import pytest
import yaml
from asdf import get_config, treeutil
from asdf.exceptions import ValidationError
from asdf.testing.helpers import yaml_to_asdf

from photutils.extension import PHOTUTILS_MANIFEST_URIS

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
