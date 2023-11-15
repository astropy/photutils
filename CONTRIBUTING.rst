Contributing to Photutils
=========================

Reporting Issues
----------------

When opening an issue to report a problem, please try to provide a
minimal code example that reproduces the issue. Also, please include
details of the operating system and the Python, NumPy, Astropy, and
Photutils versions you are using.

Contributing code
-----------------

Contributions to Photutils are done via pull requests
from GitHub users' forks of the `Photutils repository
<https://github.com/astropy/photutils>`_. If you're new to this style of
development, you'll want to read the `Astropy Development documentation
<https://docs.astropy.org/en/latest/index_dev.html>`_.

Once you open a pull request (which should be opened against the
``main`` branch, not against any other branch), please make sure that
you include the following:

- **Code**: the code you are adding, which should follow as much as
  possible the `Astropy coding guidelines <https://docs.astropy.org/en/latest/development/codeguide.html>`_.

- **Tests**: these are either tests to ensure code that previously
  failed now works (regression tests) or tests that cover as
  much as possible of the new functionality to make sure it
  doesn't break in the future. The tests are also used to ensure
  consistent results on all platforms, since we run these tests
  on many platforms/configurations. For more information about
  how to write tests, see the `Astropy testing guidelines
  <https://docs.astropy.org/en/latest/development/testguide.html>`_.

- **Documentation**: if you are adding new functionality, be sure to
  include a description in the main documentation
  (in ``docs/``). For more information, please see
  the detailed `Astropy documentation guidelines
  <https://docs.astropy.org/en/latest/development/docguide.html>`_.

- **Changelog entry**: if you are fixing a bug or adding new
  functionality, you should add an entry to the ``CHANGES.rst`` file
  that includes the PR number and if possible the issue number (if you
  are opening a pull request you may not know this yet, but you can add
  it once the pull request is open). If you're not sure where to put
  the changelog entry, wait until a maintainer has reviewed your PR and
  assigned it to a milestone.

  You do not need to include a changelog entry for fixes to bugs
  introduced in the developer version and therefore are not present
  in the stable releases. In general, you do not need to include
  a changelog entry for minor documentation or test updates. Only
  user-visible changes (new features/API changes, fixed issues) need
  to be mentioned. If in doubt, ask the core maintainer reviewing your
  changes.

Checklist for Contributed Code
------------------------------

A pull request for a new feature will be reviewed to see if it meets the
following requirements. For any pull request, a Photutils maintainer
can help to make sure that the pull request meets the requirements for
inclusion in the package.

**Scientific Quality**
(when applicable)

* Is the submission relevant to this package?
* Are references included to the original source for the algorithm?
* Does the code perform as expected?
* Has the code been tested against previously existing implementations?

**Code Quality**

* Are the `Astropy coding guidelines <https://docs.astropy.org/en/latest/development/codeguide.html>`_ followed?
* Are there dependencies other than the Astropy core, the Python
  Standard Library, and NumPy?

  - Are additional dependencies handled appropriately?
  - Do functions and classes that require additional dependencies raise
    an `ImportError` if they are not present?

**Testing**

* Are the `Astropy testing guidelines <https://docs.astropy.org/en/latest/development/testguide.html>`_ followed?
* Are the inputs to the functions and classes sufficiently tested?
* Are there tests for any exceptions raised?
* Are there tests for the expected performance?
* Are the sources for the tests documented?
* Are the tests that require an `optional dependency <https://docs.astropy.org/en/latest/development/testguide.html#tests-requiring-optional-dependencies>`_ marked as such?
* Does "``tox -e test``" run without failures?

**Documentation**

* Are the `Astropy documentation guidelines <https://docs.astropy.org/en/latest/development/docguide.html>`_ followed?
* Is there a `docstring <https://docs.astropy.org/en/latest/development/docrules.html>`_ in the functions and classes describing:

  - What the code does?
  - The format of the inputs of the function or class?
  - The format of the outputs of the function or class?
  - References to the original algorithms?
  - Any exceptions which are raised?
  - An example of running the code?

* Is there any information needed to be added to the docs to describe the function or class?
* Does the documentation build without errors or warnings?
* If applicable, has an entry been added into the changelog?

**License**

* Is the Photutils license included at the top of the file?
* Are there any conflicts with this code and existing codes?
