.. doctest-skip-all

****************************
Package Release Instructions
****************************

This document outlines the steps for releasing Photutils to `PyPI
<https://pypi.org/project/photutils/>`_. This process requires
admin-level access to the Photutils GitHub repository, as it relies on
the ability to push directly to the ``main`` branch.

These instructions assume the name of the git remote for the main
repository is called ``upstream``.

#. Check out the branch that you are going to release. This will usually
   be the ``main`` branch, unless you are making a bugfix release.

   For a bugfix release, check out the ``A.B.x`` branch. Use ``git
   cherry-pick <hash>`` (or ``git cherry-pick -m1 <hash>`` for merge
   commits) to backport fixes to the bugfix branch. Also, be sure to
   push all changes to the repository so that CI can run on the
   bugfix branch.

#. Ensure that `CI tests <https://github.com/astropy/photutils/actions>`_
   are passing for the branch you are going to
   release. Also, ensure that `Read the Docs builds
   <https://readthedocs.org/projects/photutils/builds/>`_ are passing.

#. As an extra check, run the tests locally using ``tox`` to thoroughly
   test the code in isolated environments::

        tox -e test-alldeps -- --remote-data=any
        tox -e build_docs
        tox -e linkcheck

#. Update the ``CHANGES.rst`` file to make sure that all the changes are
   listed and update the release date from ``unreleased`` to the current
   date in ``yyyy-mm-dd`` format. Then commit the changes::

        git add CHANGES.rst
        git commit -m'Finalizing changelog for version <X.Y.Z>'

#. Create an annotated git tag (optionally signing with the ``-s``
   option) for the version number you are about to release::

        git tag -a <X.Y.Z> -m'<X.Y.Z>'
        git show <X.Y.Z>  # show the tag
        git tag  # show all tags

#. Optionally, :ref:`even more manual tests <manual_tests>` can be run.

#. Push this new tag to the upstream repo::

        git push upstream <X.Y.Z>

   The new tag will trigger the automated `Publish workflow
   <https://github.com/astropy/photutils/actions/workflows/publish.yml>`_
   to build the source distribution and wheels and upload them to `PyPI
   <https://pypi.org/project/photutils/>`_.

#. Create a `GitHub Release
   <https://github.com/astropy/photutils/releases>`_ by clicking on
   "Draft a new release", select the tag of the released version, add
   a release title with the released version, and add the following
   description::

        See the [changelog](https://photutils.readthedocs.io/en/stable/changelog.html) for release notes.

   Then click "Publish release". This step will trigger an automatic
   update of the package on Zenodo (see below).

#. Check that `Zenodo <https://doi.org/10.5281/zenodo.596036>`_
   is updated with the released version. Zenodo is already configured to
   automatically update with a new published GitHub Release (see above).

#. Open a new `GitHub Milestone
   <https://github.com/astropy/photutils/milestones>`_ for the next
   release. If there are any open issues or pull requests for the new
   released version, then move them to the next milestone. After there
   are no remaining open issues or pull requests for the released
   version then close its GitHub Milestone.

#. Go to `Read the Docs
   <https://readthedocs.org/projects/photutils/versions/>`_ and check
   that the "stable" docs correspond to the new released version.
   Deactivate any older released versions (i.e., uncheck "Active").

#. After the release, the conda-forge bot (``regro-cf-autotick-bot``)
   will automatically create a pull request to the `Photutils feedstock
   repository <https://github.com/conda-forge/photutils-feedstock>`_.
   The ``meta.yaml`` recipe may need to be edited to update
   dependencies or versions. Modify (if necessary), review,
   and merge the PR to create the `conda-forge package
   <https://anaconda.org/conda-forge/photutils>`_. The `Astropy conda
   channel <https://anaconda.org/astropy/photutils>`_ will automatically
   mirror the package from conda-forge.

#. Update ``CHANGES.rst``. After releasing a minor (bugfix) version,
   update its release date. After releasing a major version, add a new
   section to ``CHANGES.rst`` for the next ``x.y.z`` version, e.g.,::

       x.y.z (unreleased)
       ------------------

       General
       ^^^^^^^

       New Features
       ^^^^^^^^^^^^

       Bug Fixes
       ^^^^^^^^^

       API Changes
       ^^^^^^^^^^^

   Then commit the changes and push to the upstream repo::

        git add CHANGES.rst
        git commit -m'Add version <x.y.z> to the changelog'
        git push upstream main

#. After releasing a major version, tag this new commit with the
   development version of the next major version and push the tag to
   the upstream repo. This is needed if the latest package release is
   the first bugfix release tagged on a bugfix branch (not the main
   branch)::

        git tag -a <x.y.z.dev> -m'<x.y.z.dev>'
        git push upstream <x.y.z.dev>


.. _manual_tests:

Additional Manual Tests
-----------------------

These additional manual checks can be run before pushing the release tag
to the upstream repository.

#. Remove any untracked files (WARNING: this will permanently remove any
   files that have not been previously committed, so make sure that you
   don't need to keep any of these files)::

        git clean -dfx

#. Check out the release tag::

        git checkout <X.Y.Z>

#. Ensure the `build <https://pypi.org/project/build/>`_ and `twine
   <https://pypi.org/project/twine/>`_ packages are installed and up to
   date::

        pip install build twine --upgrade

#. Generate the source distribution tar file::

        python -m build --sdist .

   and perform a preliminary check of the tar file::

       python -m twine check --strict dist/*

#. Run tests on the generated source distribution by going inside the
   ``dist`` directory, expanding the tar file, going inside the expanded
   directory, and running the tests with::

        cd dist
        tar xvfz <file>.tar.gz
        cd <file>
        tox -e test-alldeps -- --remote-data=any
        tox -e build_docs

   Optionally, install and test the source distribution in a virtual
   environment::

        <install and activate virtual environment>
        pip install -e '.[all,test]'
        pytest --remote-data=any

   or::

        <install and activate virtual environment>
        pip install '../<file>.tar.gz[all,test]'
        cd <any-directory-outside-of-photutils-source>
        python
        >>> import photutils
        >>> photutils.__version__
        >>> photutils.test(remote_data=True)

#. Go back to the package root directory and remove the generated files
   with::

        git clean -dfx
