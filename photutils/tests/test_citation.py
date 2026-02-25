# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for reading the CITATION.rst file.
"""


def test_citation_bibtex():
    """
    Test that __citation__ and __bibtex__ are non-empty strings
    containing the expected BibTeX entry.
    """
    import photutils

    assert isinstance(photutils.__citation__, str)
    assert isinstance(photutils.__bibtex__, str)
    assert len(photutils.__citation__) > 0
    assert '@software' in photutils.__citation__


def test_citation_unicode_regression(monkeypatch):
    """
    Test that importing photutils does not raise a UnicodeDecodeError
    when the system locale encoding is ASCII.

    On some systems (e.g., Linux containers or minimal CI environments),
    the default locale encoding may be ASCII (e.g., when ``LANG=C``).
    If CITATION.rst contains non-ASCII characters (e.g., in author
    names), and if the ``open()`` call in ``_get_bibtex()`` does not
    specify ``encoding='utf-8'``, then importing photutils will raise
    a UnicodeDecodeError when it tries to read CITATION.rst using the
    ASCII encoding.

    This test patches ``builtins.open`` to default to ASCII (simulating
    a ``LANG=C`` environment) and then reloads photutils so that
    ``_get_bibtex()`` is re-executed under those conditions. If there
    are non-ASCII characters in CITATION.rst, the test will fail with
    ``UnicodeDecodeError`` if ``encoding='utf-8'`` is removed from the
    ``open()`` call in ``photutils/__init__.py``.
    """
    import importlib

    import photutils

    real_open = open

    def ascii_default_open(*args, **kwargs):
        mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
        if 'b' not in mode and 'encoding' not in kwargs:
            kwargs['encoding'] = 'ascii'
        return real_open(*args, **kwargs)

    monkeypatch.setattr('builtins.open', ascii_default_open)
    importlib.reload(photutils)

    assert '@software' in photutils.__citation__
