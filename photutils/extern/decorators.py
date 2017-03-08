# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sundry function and class decorators."""

from __future__ import print_function

import functools
import warnings
import inspect

from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)

ASTROPY_LT_1_3 = not minversion('astropy', '1.3')


__all__ = ['deprecated_renamed_argument']


if not ASTROPY_LT_1_3:
    from astropy.utils.decorators import deprecated_renamed_argument
else:
    def deprecated_renamed_argument(old_name, new_name, since,
                                    arg_in_kwargs=False, relax=False):
        """Deprecate a _renamed_ function argument.

        The decorator assumes that the argument with the ``old_name`` was removed
        from the function signature and the ``new_name`` replaced it at the
        **same position** in the signature.  If the ``old_name`` argument is
        given when calling the decorated function the decorator will catch it and
        issue a deprecation warning and pass it on as ``new_name`` argument.

        Parameters
        ----------
        old_name : str or list/tuple thereof
            The old name of the argument.

        new_name : str or list/tuple thereof
            The new name of the argument.

        since : str or number or list/tuple thereof
            The release at which the old argument became deprecated.

        arg_in_kwargs : bool or list/tuple thereof, optional
            If the argument is not a named argument (for example it
            was meant to be consumed by ``**kwargs``) set this to
            ``True``.  Otherwise the decorator will throw an Exception
            if the ``new_name`` cannot be found in the signature of
            the decorated function.
            Default is ``False``.

        relax : bool or list/tuple thereof, optional
            If ``False`` a ``TypeError`` is raised if both ``new_name`` and
            ``old_name`` are given.  If ``True`` the value for ``new_name`` is used
            and a Warning is issued.
            Default is ``False``.

        Raises
        ------
        TypeError
            If the new argument name cannot be found in the function
            signature and arg_in_kwargs was False or if it is used to
            deprecate the name of the ``*args``-, ``**kwargs``-like arguments.
            At runtime such an Error is raised if both the new_name
            and old_name were specified when calling the function and
            "relax=False".

        Notes
        -----
        The decorator should be applied to a function where the **name**
        of an argument was changed but it applies the same logic.

        .. warning::
            If ``old_name`` is a list or tuple the ``new_name`` and ``since`` must
            also be a list or tuple with the same number of entries. ``relax`` and
            ``arg_in_kwarg`` can be a single bool (applied to all) or also a
            list/tuple with the same number of entries like ``new_name``, etc.

        .. warning::
            This decorator needs to access the original signature of the decorated
            function. Therefore this decorator must be the **first** decorator on
            any function if it needs to work for Python before version 3.4.
        """
        cls_iter = (list, tuple)
        if isinstance(old_name, cls_iter):
            n = len(old_name)
            # Assume that new_name and since are correct (tuple/list with the
            # appropriate length) in the spirit of the "consenting adults". But the
            # optional parameters may not be set, so if these are not iterables
            # wrap them.
            if not isinstance(arg_in_kwargs, cls_iter):
                arg_in_kwargs = [arg_in_kwargs] * n
            if not isinstance(relax, cls_iter):
                relax = [relax] * n
        else:
            # To allow a uniform approach later on, wrap all arguments in lists.
            n = 1
            old_name = [old_name]
            new_name = [new_name]
            since = [since]
            arg_in_kwargs = [arg_in_kwargs]
            relax = [relax]

        def decorator(function):
            # Lazy import to avoid cyclic imports
            from astropy.utils.compat.funcsigs import signature

            # The named arguments of the function.
            arguments = signature(function).parameters
            keys = list(arguments.keys())
            position = [None] * n

            for i in range(n):
                # Determine the position of the argument.
                if new_name[i] in arguments:
                    param = arguments[new_name[i]]
                    # There are several possibilities now:

                    # 1.) Positional or keyword argument:
                    if param.kind == param.POSITIONAL_OR_KEYWORD:
                        position[i] = keys.index(new_name[i])

                    # 2.) Keyword only argument (Python 3 only):
                    elif param.kind == param.KEYWORD_ONLY:
                        # These cannot be specified by position.
                        position[i] = None

                    # 3.) positional-only argument, varargs, varkwargs or some
                    #     unknown type:
                    else:
                        raise TypeError('cannot replace argument "{0}" of kind {1}'
                                        '.'.format(new_name[i], repr(param.kind)))

                # In case the argument is not found in the list of arguments
                # the only remaining possibility is that it should be catched
                # by some kind of **kwargs argument.
                # This case has to be explicitly specified, otherwise throw
                # an exception!
                elif arg_in_kwargs[i]:
                    position[i] = None
                else:
                    raise TypeError('"{}" was not specified in the function '
                                    'signature. If it was meant to be part of '
                                    '"**kwargs" then set "arg_in_kwargs" to "True"'
                                    '.'.format(new_name[i]))

            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                for i in range(n):
                    # The only way to have oldkeyword inside the function is
                    # that it is passed as kwarg because the oldkeyword
                    # parameter was renamed to newkeyword.
                    if old_name[i] in kwargs:
                        value = kwargs.pop(old_name[i])
                        warnings.warn('"{0}" was deprecated in version {1} '
                                      'and will be removed in a future version. '
                                      'Use argument "{2}" instead.'
                                      ''.format(old_name[i], since[i],
                                                new_name[i]),
                                      AstropyDeprecationWarning)

                        # Check if the newkeyword was given as well.
                        newarg_in_args = (position[i] is not None and
                                          len(args) > position[i])
                        newarg_in_kwargs = new_name[i] in kwargs

                        if newarg_in_args or newarg_in_kwargs:
                            # If both are given print a Warning if relax is True or
                            # raise an Exception is relax is False.
                            if relax[i]:
                                warnings.warn('"{0}" and "{1}" keywords were set. '
                                              'Using the value of "{1}".'
                                              ''.format(old_name[i], new_name[i]),
                                              AstropyUserWarning)
                            else:
                                raise TypeError('cannot specify both "{}" and "{}"'
                                                '.'.format(old_name[i],
                                                           new_name[i]))
                        else:
                            # If the new argument isn't specified just pass the old
                            # one with the name of the new argument to the function
                            kwargs[new_name[i]] = value
                return function(*args, **kwargs)

            return wrapper
        return decorator
