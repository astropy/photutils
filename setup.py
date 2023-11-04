#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup

from extension_helpers import get_extensions

setup(ext_modules=get_extensions())
