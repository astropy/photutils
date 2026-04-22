# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Generate a pip requirements file pinning build and runtime dependencies
to their minimum supported versions, as declared in pyproject.toml.

This script reads [build-system].requires and [project].dependencies
from pyproject.toml and writes a requirements file that pins each
dependency with a >= specifier to its minimum version (using ==).
"""

import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

pyproject = Path(__file__).parent / 'pyproject.toml'
data = tomllib.loads(pyproject.read_text())


def pin_minimum(req_str_sections):
    """
    Return a list of pinned requirement strings with the minimum
    versions for each package.

    Parameters
    ----------
    req_str_sections : list of list of str
        A list of sections, where each section is a list of
        requirement strings (e.g. from [build-system].requires or
        [project].dependencies).

    Returns
    -------
    result : list of str
        A list of requirement strings with minimum versions pinned (e.g.
        'package==1.2.3').
    """
    mins = {}
    for section in req_str_sections:
        for req_str in section:
            req = Requirement(req_str)
            name = req.name.lower()
            for spec in req.specifier:
                if spec.operator == '>=':
                    ver = Version(spec.version)
                    if name not in mins or ver > mins[name][1]:
                        mins[name] = (req.name, ver)
                    break
    return [f'{name}=={ver}' for name, ver in mins.values()]


build_pins = pin_minimum([data['build-system']['requires']])
all_pins = pin_minimum([data['build-system']['requires'],
                        data['project']['dependencies']])

for filename, pins in (
    ('build-oldest-constraints.txt', build_pins),
    ('install-oldest-constraints.txt', all_pins),
):
    output = pyproject.parent / filename
    output.write_text('\n'.join(pins) + '\n')
    sys.stdout.write(f'Wrote {output}:\n')
    for p in pins:
        sys.stdout.write(f'  {p}\n')
