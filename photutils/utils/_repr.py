# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for class __repr__ and __str__ strings.
"""


def make_repr(self, params, ellipsis=(), long=False):
    cls_name = f'{self.__class__.__name__}'
    if long:
        cls_name = f'{self.__class__.__module__}.{cls_name}'

    cls_info = []
    for param in params:
        value = getattr(self, param)
        if param in ellipsis and value is not None:
            value = '...'
        cls_info.append((param, value))

    if long:
        delim = ': '
        join_str = '\n'
    else:
        delim = '='
        join_str = ', '

    fmt = [f'{key}{delim}{val!r}' for key, val in cls_info]
    fmt = f'{join_str}'.join(fmt)

    if long:
        return f'<{cls_name}>\n{fmt}'
    else:
        return f'{cls_name}({fmt})'
