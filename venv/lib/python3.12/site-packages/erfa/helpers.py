# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Helpers to interact with the ERFA library, in particular for leap seconds.
"""
import functools
import threading
from datetime import datetime, timedelta
from warnings import warn

import numpy as np

from .core import ErfaWarning

from .ufunc import get_leap_seconds, set_leap_seconds, dt_eraLEAPSECOND


NUMPY_LT_2_0 = np.__version__.startswith("1.")

_NotFound = object()


# TODO: This can still be made to work for setters by implementing an
# accompanying metaclass that supports it; we just don't need that right this
# second
class classproperty(property):
    """
    Similar to `property`, but allows class-level properties.  That is,
    a property whose getter is like a `classmethod`.

    The wrapped method may explicitly use the `classmethod` decorator (which
    must become before this decorator), or the `classmethod` may be omitted
    (it is implicit through use of this decorator).

    .. note::

        classproperty only works for *read-only* properties.  It does not
        currently allow writeable/deletable properties, due to subtleties of how
        Python descriptors work.  In order to implement such properties on a class
        a metaclass for that class must be implemented.

    Parameters
    ----------
    fget : callable
        The function that computes the value of this property (in particular,
        the function when this is used as a decorator) a la `property`.

    doc : str, optional
        The docstring for the property--by default inherited from the getter
        function.

    lazy : bool, optional
        If True, caches the value returned by the first call to the getter
        function, so that it is only called once (used for lazy evaluation
        of an attribute).  This is analogous to `lazyproperty`.  The ``lazy``
        argument can also be used when `classproperty` is used as a decorator
        (see the third example below).  When used in the decorator syntax this
        *must* be passed in as a keyword argument.

    Examples
    --------

    ::

        >>> class Foo:
        ...     _bar_internal = 1
        ...     @classproperty
        ...     def bar(cls):
        ...         return cls._bar_internal + 1
        ...
        >>> Foo.bar
        2
        >>> foo_instance = Foo()
        >>> foo_instance.bar
        2
        >>> foo_instance._bar_internal = 2
        >>> foo_instance.bar  # Ignores instance attributes
        2

    As previously noted, a `classproperty` is limited to implementing
    read-only attributes::

        >>> class Foo:
        ...     _bar_internal = 1
        ...     @classproperty
        ...     def bar(cls):
        ...         return cls._bar_internal
        ...     @bar.setter
        ...     def bar(cls, value):
        ...         cls._bar_internal = value
        ...
        Traceback (most recent call last):
        ...
        NotImplementedError: classproperty can only be read-only; use a
        metaclass to implement modifiable class-level properties

    When the ``lazy`` option is used, the getter is only called once::

        >>> class Foo:
        ...     @classproperty(lazy=True)
        ...     def bar(cls):
        ...         print("Performing complicated calculation")
        ...         return 1
        ...
        >>> Foo.bar
        Performing complicated calculation
        1
        >>> Foo.bar
        1

    If a subclass inherits a lazy `classproperty` the property is still
    re-evaluated for the subclass::

        >>> class FooSub(Foo):
        ...     pass
        ...
        >>> FooSub.bar
        Performing complicated calculation
        1
        >>> FooSub.bar
        1
    """

    def __new__(cls, fget=None, doc=None, lazy=False):
        if fget is None:
            # Being used as a decorator--return a wrapper that implements
            # decorator syntax
            def wrapper(func):
                return cls(func, lazy=lazy)

            return wrapper

        return super().__new__(cls)

    def __init__(self, fget, doc=None, lazy=False):
        self._lazy = lazy
        if lazy:
            self._lock = threading.RLock()   # Protects _cache
            self._cache = {}
        fget = self._wrap_fget(fget)

        super().__init__(fget=fget, doc=doc)

        # There is a buglet in Python where self.__doc__ doesn't
        # get set properly on instances of property subclasses if
        # the doc argument was used rather than taking the docstring
        # from fget
        # Related Python issue: https://bugs.python.org/issue24766
        if doc is not None:
            self.__doc__ = doc

    def __get__(self, obj, objtype):
        if self._lazy:
            val = self._cache.get(objtype, _NotFound)
            if val is _NotFound:
                with self._lock:
                    # Check if another thread initialised before we locked.
                    val = self._cache.get(objtype, _NotFound)
                    if val is _NotFound:
                        val = self.fget.__wrapped__(objtype)
                        self._cache[objtype] = val
        else:
            # The base property.__get__ will just return self here;
            # instead we pass objtype through to the original wrapped
            # function (which takes the class as its sole argument)
            val = self.fget.__wrapped__(objtype)
        return val

    def getter(self, fget):
        return super().getter(self._wrap_fget(fget))

    def setter(self, fset):
        raise NotImplementedError(
            "classproperty can only be read-only; use a metaclass to "
            "implement modifiable class-level properties")

    def deleter(self, fdel):
        raise NotImplementedError(
            "classproperty can only be read-only; use a metaclass to "
            "implement modifiable class-level properties")

    @staticmethod
    def _wrap_fget(orig_fget):
        if isinstance(orig_fget, classmethod):
            orig_fget = orig_fget.__func__

        # Using stock functools.wraps instead of the fancier version
        # found later in this module, which is overkill for this purpose

        @functools.wraps(orig_fget)
        def fget(obj):
            return orig_fget(obj.__class__)

        return fget


class leap_seconds:
    """Leap second management.

    This singleton class allows access to ERFA's leap second table,
    using the methods 'get', 'set', and 'update'.

    One can also check expiration with 'expires' and 'expired'.

    Note that usage of the class is similar to a ``ScienceState`` class,
    but it cannot be used as a context manager.
    """
    _expires = None
    """Explicit expiration date inferred from leap-second table."""
    _expiration_days = 180
    """Number of days beyond last leap second at which table expires."""

    def __init__(self):
        raise RuntimeError("This class is a singleton.  Do not instantiate.")

    @classmethod
    def get(cls):
        """Get the current leap-second table used internally."""
        return get_leap_seconds()

    @classmethod
    def validate(cls, table):
        """Validate a leap-second table.

        Parameters
        ----------
        table : array_like
            Must have 'year', 'month', and 'tai_utc' entries.  If a 'day'
            entry is present, it will be checked that it is always 1.
            If ``table`` has an 'expires' attribute, it will be interpreted
            as an expiration date.

        Returns
        -------
        array : `~numpy.ndarray`
            Structures array with 'year', 'month', 'tai_utc'.
        expires: `~datetime.datetime` or None
            Possible expiration date inferred from the table.  `None` if not
            present or if not a `~datetime.datetime` or `~astropy.time.Time`
            instance and not parsable as a 'dd month yyyy' string.

        Raises
        ------
        ValueError
            If the leap seconds in the table are not on the 1st of January or
            July, or if the sorted TAI-UTC do not increase in increments of 1.
        """
        try:
            day = table['day']
        except Exception:
            day = 1

        expires = getattr(table, 'expires', None)
        if expires is not None and not isinstance(expires, datetime):
            # Maybe astropy Time? Cannot go via strftime, since that
            # might need leap-seconds.  If not, try standard string
            # format from leap_seconds.dat and leap_seconds.list
            isot = getattr(expires, 'isot', None)
            try:
                if isot is not None:
                    expires = datetime.strptime(isot.partition('T')[0],
                                                '%Y-%m-%d')
                else:
                    expires = datetime.strptime(expires, '%d %B %Y')

            except Exception as exc:
                warn(f"ignoring non-datetime expiration {expires}; "
                     f"parsing it raised {exc!r}", ErfaWarning)
                expires = None

        # Take care of astropy Table.
        if hasattr(table, '__array__'):
            table = table.__array__()[list(dt_eraLEAPSECOND.names)]

        table = np.array(table, dtype=dt_eraLEAPSECOND, ndmin=1,
                         copy=False if NUMPY_LT_2_0 else None)

        # Simple sanity checks.
        if table.ndim > 1:
            raise ValueError("can only pass in one-dimensional tables.")

        if not np.all(((day == 1) &
                       (table['month'] == 1) | (table['month'] == 7)) |
                      (table['year'] < 1972)):
            raise ValueError("leap seconds inferred that are not on "
                             "1st of January or 1st of July.")

        if np.any((table['year'][:-1] > 1970) &
                  (np.diff(table['tai_utc']) != 1)):
            raise ValueError("jump in TAI-UTC by something else than one.")

        return table, expires

    @classmethod
    def set(cls, table=None):
        """Set the ERFA leap second table.

        Note that it is generally safer to update the leap-second table than
        to set it directly, since most tables do not have the pre-1970 changes
        in TAI-UTC that are part of the built-in ERFA table.

        Parameters
        ----------
        table : array_like or `None`
            Leap-second table that should at least hold columns of 'year',
            'month', and 'tai_utc'.  Only simple validation is done before it
            is being used, so care need to be taken that entries are correct.
            If `None`, reset the ERFA table to its built-in values.

        Raises
        ------
        ValueError
            If the leap seconds in the table are not on the 1st of January or
            July, or if the sorted TAI-UTC do not increase in increments of 1.
        """
        if table is None:
            expires = None
        else:
            table, expires = cls.validate(table)

        set_leap_seconds(table)
        cls._expires = expires

    @classproperty
    def expires(cls):
        """The expiration date of the current ERFA table.

        This is either a date inferred from the last table used to update or
        set the leap-second array, or a number of days beyond the last leap
        second.
        """
        if cls._expires is None:
            last = cls.get()[-1]
            return (datetime(last['year'], last['month'], 1) +
                    timedelta(cls._expiration_days))
        else:
            return cls._expires

    @classproperty
    def expired(cls):
        """Whether the leap second table is valid beyond the present."""
        return cls.expires < datetime.now()

    @classmethod
    def update(cls, table):
        """Add any leap seconds not already present to the ERFA table.

        This method matches leap seconds with those present in the ERFA table,
        and extends the latter as necessary.

        If the ERFA leap seconds file was corrupted, it will be reset.

        If the table is corrupted, the ERFA file will be unchanged.

        Parameters
        ----------
        table : array_like or `~astropy.utils.iers.LeapSeconds`
            Array or table with TAI-UTC from leap seconds.  Should have
            'year', 'month', and 'tai_utc' columns.

        Returns
        -------
        n_update : int
            Number of items updated.

        Raises
        ------
        ValueError
            If the leap seconds in the table are not on the 1st of January or
            July, or if the sorted TAI-UTC do not increase in increments of 1.
        """
        table, expires = cls.validate(table)

        # Get erfa table and check it is OK; if not, reset it.
        try:
            erfa_ls, _ = cls.validate(cls.get())
        except Exception:
            cls.set()
            erfa_ls = cls.get()

        # Create the combined array and use it (validating the combination).
        ls = np.union1d(erfa_ls, table)
        cls.set(ls)

        # If the update table has an expiration beyond that inferred from
        # the new leap second second array, use it (but, now that the new
        # array is set, do not allow exceptions due to misformed expires).
        try:
            if expires is not None and expires > cls.expires:
                cls._expires = expires

        except Exception as exc:
            warn("table 'expires' attribute ignored as comparing it "
                 "with a datetime raised an error:\n" + str(exc),
                 ErfaWarning)

        return len(ls) - len(erfa_ls)
