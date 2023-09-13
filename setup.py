#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup  # noqa: E402

from extension_helpers import get_extensions  # noqa: E402

setup(ext_modules=get_extensions())
