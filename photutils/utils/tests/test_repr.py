# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _repr module.
"""

import pytest

from photutils.utils._repr import make_repr


class ExampleClass:
    def __init__(self, x, y, z=1):
        self.x = x
        self.y = y
        self.z = z


def test_make_repr():
    obj = ExampleClass(1, 2)
    params = ('x', 'y', 'z')
    repr_str = make_repr(obj, params)
    assert repr_str == 'ExampleClass(x=1, y=2, z=1)'

    params = ('x', 'y')
    repr_str = make_repr(obj, params)
    assert repr_str == 'ExampleClass(x=1, y=2)'

    params = 'x'
    repr_str = make_repr(obj, params)
    assert repr_str == 'ExampleClass(x=1)'

    params = ('x', 'y', 'z')
    repr_str = make_repr(obj, params, long=True)
    ref = ('<photutils.utils.tests.test_repr.ExampleClass>\n'
           'x: 1\n'
           'y: 2\n'
           'z: 1')
    assert repr_str == ref

    overrides = {'x': '...'}
    repr_str = make_repr(obj, params, overrides=overrides)
    assert repr_str == "ExampleClass(x='...', y=2, z=1)"

    overrides = {'x': '...', 'z': '...'}
    repr_str = make_repr(obj, params, overrides=overrides)
    assert repr_str == "ExampleClass(x='...', y=2, z='...')"

    params = ('a', 'x', 'y', 'z')
    match = 'not found in instance or overrides'
    with pytest.raises(ValueError, match=match):
        repr_str = make_repr(obj, params)

    params = ('x', 'y', 'z')
    overrides = {'a': '...'}
    match = 'The overrides keys must be a subset of the params list'
    with pytest.raises(ValueError, match=match):
        repr_str = make_repr(obj, params, overrides=overrides)

    params = ('a', 'x', 'y', 'z')
    overrides = {'a': '...'}
    repr_str = make_repr(obj, params, overrides=overrides)
    assert repr_str == "ExampleClass(a='...', x=1, y=2, z=1)"

    params = ('a', 'x', 'y', 'z')
    overrides = {'a': '...', 'z': '...'}
    repr_str = make_repr(obj, params, overrides=overrides)
    assert repr_str == "ExampleClass(a='...', x=1, y=2, z='...')"

    params = ('x', 'y', 'z')
    overrides = {'x': '...'}
    obj = ExampleClass(None, 2)
    repr_str = make_repr(obj, params, overrides=overrides)
    assert repr_str == 'ExampleClass(x=None, y=2, z=1)'
