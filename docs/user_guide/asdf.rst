.. _asdf:

ASDF Serialization
==================

Introduction
------------

Photutils apertures and PSF models can be written to and read from
files in the `Advanced Scientific Data Format (ASDF)
<https://asdf.readthedocs.io/>`_. ASDF stores the object as structured
metadata in a human-readable YAML header, with any array data appended
as binary blocks. Unlike :mod:`pickle`, an ASDF file is described by a
published, versioned schema, so it can be inspected without Photutils
and read by other implementations.

Photutils registers its ASDF support through the ``asdf.extensions``
and ``asdf.resource_mappings`` entry points. No explicit imports are
required to enable this support. Simply installing Photutils alongside
asdf is sufficient.


Requirements
------------

Serialization requires the optional ``asdf`` package. Objects that
store an `~astropy.units.Quantity` or an
`~astropy.coordinates.SkyCoord` additionally require the optional
``asdf-astropy`` package, which provides the converters for those
Astropy types:

* All PSF models require ``asdf-astropy``.

* Sky apertures require ``asdf-astropy``, as do pixel apertures whose
  rotation angle is given as an angular quantity.

* Pixel apertures whose parameters are all plain numbers require only
  ``asdf``.

Both packages are installed by the ``all`` extra::

    pip install photutils[all]

If ``asdf-astropy`` is missing, PSF models raise an
``AsdfSerializationError`` when written, and apertures that store an
angular quantity fail schema validation.


Writing and Reading Files
-------------------------

An object is written by assigning it to a key in an
``asdf.AsdfFile`` tree:

.. doctest-skip::

    >>> import asdf
    >>> from photutils.aperture import CircularAperture
    >>> aperture = CircularAperture((10, 20), r=5)
    >>> with asdf.AsdfFile() as af:
    ...     af['aperture'] = aperture
    ...     af.write_to('aperture.asdf')

Reading returns an equivalent object:

.. doctest-skip::

    >>> with asdf.open('aperture.asdf') as af:
    ...     aperture = af['aperture']
    >>> aperture
    <CircularAperture([10., 20.], r=5.0)>

The tree key is arbitrary, and a tree may hold any number of objects
mixed with other ASDF-serializable data:

.. doctest-skip::

    >>> with asdf.AsdfFile() as af:
    ...     af['apertures'] = [aperture, other_aperture]
    ...     af['psf'] = psf_model
    ...     af['meta'] = {'program': 1234}
    ...     af.write_to('photometry.asdf')


Supported Objects
-----------------

The pixel and sky variant of each aperture share a tag, because they
differ only in whether their parameters are pixel values or angular
quantities.

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - Tag
      - Classes
    * - ``photutils/aperture/circular_aperture-1.0.0``
      - `~photutils.aperture.CircularAperture`,
        `~photutils.aperture.SkyCircularAperture`
    * - ``photutils/aperture/circular_annulus-1.0.0``
      - `~photutils.aperture.CircularAnnulus`,
        `~photutils.aperture.SkyCircularAnnulus`
    * - ``photutils/aperture/elliptical_aperture-1.0.0``
      - `~photutils.aperture.EllipticalAperture`,
        `~photutils.aperture.SkyEllipticalAperture`
    * - ``photutils/aperture/elliptical_annulus-1.0.0``
      - `~photutils.aperture.EllipticalAnnulus`,
        `~photutils.aperture.SkyEllipticalAnnulus`
    * - ``photutils/aperture/polygon_aperture-1.0.0``
      - `~photutils.aperture.PolygonAperture`,
        `~photutils.aperture.SkyPolygonAperture`
    * - ``photutils/aperture/rectangular_aperture-1.0.0``
      - `~photutils.aperture.RectangularAperture`,
        `~photutils.aperture.SkyRectangularAperture`
    * - ``photutils/aperture/rectangular_annulus-1.0.0``
      - `~photutils.aperture.RectangularAnnulus`,
        `~photutils.aperture.SkyRectangularAnnulus`
    * - ``photutils/psf/airy_disk_psf-1.0.0``
      - `~photutils.psf.AiryDiskPSF`
    * - ``photutils/psf/circular_gaussian_prf-1.0.0``
      - `~photutils.psf.CircularGaussianPRF`
    * - ``photutils/psf/circular_gaussian_psf-1.0.0``
      - `~photutils.psf.CircularGaussianPSF`
    * - ``photutils/psf/circular_gaussian_sigma_prf-1.0.0``
      - `~photutils.psf.CircularGaussianSigmaPRF`
    * - ``photutils/psf/gaussian_prf-1.0.0``
      - `~photutils.psf.GaussianPRF`
    * - ``photutils/psf/gaussian_psf-1.0.0``
      - `~photutils.psf.GaussianPSF`
    * - ``photutils/psf/gridded_psf_model-1.0.0``
      - `~photutils.psf.GriddedPSFModel`
    * - ``photutils/psf/image_psf-1.0.0``
      - `~photutils.psf.ImagePSF`
    * - ``photutils/psf/moffat_psf-1.0.0``
      - `~photutils.psf.MoffatPSF`
    * - ``photutils/psf/stdpsf_grid-1.0.0``
      - `~photutils.psf.STDPSFGrid`

A PSF model is serialized as an Astropy transform, so its fitting state
is preserved along with its parameters, including which parameters are
fixed and any parameter bounds.

For `~photutils.psf.GriddedPSFModel` and `~photutils.psf.STDPSFGrid`,
the ePSF images, the grid positions, the grid shape, and the
oversampling factors are stored as separate properties, and any
remaining metadata is stored under a ``meta`` key.


Schemas and Versioning
----------------------

Each tag is backed by a published schema. The tag and schema URIs are
of the form::

    tag:astropy.org:photutils/aperture/circular_aperture-1.0.0
    asdf://astropy.org/photutils/schemas/aperture/circular_aperture-1.0.0

and the tags are collected in the extension manifest
``asdf://astropy.org/photutils/manifests/photutils-1.0.0``. The schemas
and the manifest ship inside Photutils and are registered with ``asdf``
automatically, so ``asdf`` can validate a file without any network
access.

Tags and schemas are versioned independently of Photutils. A change to
what a tag stores requires a new tag version, and the converters for
existing versions are retained, so that files written by an older
version of Photutils remain readable. An ASDF file records the
extension that wrote it, so ``asdf`` warns when a file is opened
without that extension available.

Every tagged object is validated against its schema on both write and
read.
