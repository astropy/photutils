PSF Fitting
===========

.. role::  raw-html(raw)
    :format: html

`photutils` offers a module implementing a series of PSF fitting algorithms. These
fitting algorithms are broken down into a series of submodules, allowing the user
to implement specific versions of each part of the fitting process, implementing
a variety of fitting procedures best suited to the specific science cases. However,
at its most basic, all choices share the same basic functionality: a 2D image is
supplied and a `~astropy.table.Table` of detected sources, with positions and flux
counts, is returned.

This simplified workflow is expanded below for the case of the
`~photutils.psf.IterativelySubtractedPSFPhotometry` class, building upon the simpler
`~photutils.psf.BasicPSFPhotometry` algorithm. While there are additional blocks
used within the workflow, the main `~photutils.psf.BasicPSFPhotometry` flow is
``image`` :raw-html:`&rarr;` ``finder`` :raw-html:`&rarr;` ``group maker``
:raw-html:`&rarr;` ``fitter`` :raw-html:`&rarr;` ``stars``. The extension to more
complex fitting classes, such as `~photutils.psf.IterativelySubtractedPSFPhotometry`,
have the flow ``image`` :raw-html:`&rarr;` ``finder`` :raw-html:`&rarr;` ``group maker``
:raw-html:`&rarr;` ``fitter`` :raw-html:`&rarr;` ``culler and ender`` :raw-html:`&rarr;`
``finder`` :raw-html:`&rarr;` ... :raw-html:`&rarr;` ``stars``.

Each submodule is described briefly here, with context for how it fits into the overall
PSF fitting framework, providing a top-down view of the process. Where available, links
to more detailed descriptions of each block is also provided.

.. graphviz::

    digraph PSFFitting {
        compound = true;
        newrank = true;
        subgraph cluster0 {
            label="IterativelySubtractedPSFPhotometry"
            f [label = "finder"];
            g [label = "group maker"];
            som [label = "single object model"];
            n [label = "noise"];
            ft [label = "fitter"];
            b [label = "background estimator"];
            c [label = "culler and ender"];
            // required to allow finder and guess to combine into group maker
            invis [shape = circle, width = .1, fixedsize = true, style = invis];
        }

        w [label = "wcs"];
        i [label = "input image", shape = box];
        gs [label = "guess at stars", shape = box];
        p [label = "psf model"];
        sm [label = "scene maker"];
        st [label = "stars\n(astropy.table)", shape = box];

        label = <<FONT POINT-SIZE = "20"><I>photutils.psf </I>class structures</FONT>>;
        labelloc = top;

        w -> i [style = dashed];
        // lhead sets the arrow end at the subgraph, rather than at finder itself
        i -> f [lhead = cluster0];
        gs -> invis [style = dashed];

        f -> g;
        f -> invis;
        invis -> g;
        som -> f [style = dashed];
        n -> ft;
        som -> ft;
        g -> ft;
        b -> f [style = dashed];
        b -> g [style = dashed];
        b -> ft [style = dashed];
        ft -> c [style = dashed, label = "Subtract", dir = none];
        c -> f [style = dashed];

        i -> gs [style = invis];
        
        p -> som;
        sm -> g [style = dotted];
        ft -> st;
        // reverse arrow to put psf model above scene maker, saving space with scene
        // maker next to the middle of the main box
        p -> sm [style = dotted, dir = back];

        // puts things that should be at the top of the box at the top for orientation
        // and structuring; requires 'newrank' above to set subgraph items equal
        {rank = same; w; som;}
        {rank = same; p; f; n;}
        {rank = same; gs; g; sm;}
    }


.. _Image:

Input Image
^^^^^^^^^^^

The input image is the most fundamental part of the PSF fitting process, as it is the
product on which all subsequent methodology is applied. The image must be a two-dimensional
array of ``x`` and ``y`` coordinates with ``flux`` represented by array values. If a
WCS -- contained within a FITS file, for instance -- is passed with the two-dimensional
array then the position of sources can be given in sky coordinates, instead of pixel values.
The given image is passed as an input to the chosen PSF fitting class.


.. _Finder:

Finder
^^^^^^

The finder is the first step in the PSF fitting process, as sources must be discovered in
the image before any kind of fit can be applied to them. These accept the input ``image``
and produce a `~astropy.table.Table` of detected sources, by a set of criteria internal to
the given ``Finder`` (see, e.g., `~photutils.detection.DAOStarFinder` using ``roundness``
and ``sharpness`` to determine if sources are point-like). If an initial set of detected
sources is passed to the fitter as ``init_guesses`` then ``Finder`` is not run on the first
pass of an iterative fitting class, instead using the provided positions and fluxes. See
the :doc:`finder <finder>` documentation for more details.


.. _Group Maker:

Group Maker
^^^^^^^^^^^

The second block to run in the fitting process, the various group maker processes, such as
`~photutils.psf.DAOGroup`, allow for the merging of astrometrically overlapping sources
into a set of multiple stars. These stars must be fit simultaneously (see Fitter_) as a
composite model, avoiding the potential issue where blended sources return incorrect fluxes,
as affects aperture photometry. The block accepts the `~astropy.table.Table` output from
Finder_ and returns a second `~astropy.table.Table` containing the same columns as the input
as well as an additional column indicating the group number of each source. More information
on this block can be found in :doc:`Group Maker <group_maker>`.


.. _Fitter:

Fitter
^^^^^^
Once a set of sources has been detected and grouped, each source must have its respective
properties -- position and flux primarily -- determined. For this a ``Fitter`` instance is
required; these are drawn from a separate class or instance of minimization routines, such as
those given under the `~astropy.modeling` umbrella. The fitter must be a two-dimensional fitter,
by nature of the image to be fit, accepted as an input to the instance. The class must then
return the position of the source (or sources, if a composite model was supplied; see
`Group Maker`_) and its total flux, as determined by the PSF model describing the telescope
system -- optics, CCD, etc. -- that created the image. See `below <PSF Model_>`_ for more
details on the PSF model, and the :doc:`fitter <fitter>` documentation for further
information on the ``Fitter`` instance and its properties.


.. _PSF Model:

PSF Model
^^^^^^^^^
Perhaps the most important aspect of PSF fitting, the ``PSF Model`` describes the distribution
of light falling on a given pixel from a source at some position with some total flux, fully
describing the effects of telescope optics, quantized CCD pixels, etc. In this respect it is
more properly described as a PRF than a PSF -- see the :doc:`documentation <psf_model>` for
more details -- but we conflate the two terms here freely. Similar to the ``Fitter``, it is
a separate callable class detailing a specific PSF response. This can be some analytical
function, such as `~photutils.psf.IntegratedGaussianPRF`, or an empirical description of the
PSF, such as that implemented by the effective PSF functionality (see :ref:`build-epsf` for
details). The model must describe a two-dimensional distribution, and must return the total
flux of the source and its sub-pixel position via the ``evaluate`` function, accepted by the
Fitter_ when minimizing the fit between the model and the data.


.. _noise:

Noise Description
^^^^^^^^^^^^^^^^^
The ``noise`` block within the PSF fitting process is, in much the same way as the ``Fitter``
block, a separate ``callable`` function mapping the relationship between the pixel values in
the input ``image`` and the corresponding uncertainty -- in units of standard deviations -- of
the pixel. It must therefore accept a two-dimensional array and return an array of the same
shape as ``image``. Alternatively it can be overloaded to indicate that the uncertainty array
was passed along with ``image`` via an `~astropy.nddata.NDData` instance. A more detailed
description of this block is available :doc:`here <noise_data>`.


.. _Background Estimator:

Background Estimator
^^^^^^^^^^^^^^^^^^^^
The ``image`` on which source extraction and evaluation is to be performed is not necessarily
assumed to have had any pre-processing applied to it, and thus it may be necessary to account
for the counts of the image background when handling PSF fitting. This block handles
the determination of the background levels, avoiding high count peaks to calculate the typical
count of an otherwise empty pixel. The block must therefore accepts the two-dimensional ``image``
and returns either a single value -- the background count of the entire image, as calculated
by the criteria of the specific block implementation -- or a 2D array of values, allowing for
the possibility of varying background counts across the image. More information on this block
can be found :doc:`here <background_estimator>`.


.. _Culler and Ender:

Culler and Ender
^^^^^^^^^^^^^^^^
Only implemented in iterative fitting classes, part of the extension to
`~photutils.psf.BasicPSFPhotometry`, this block is the final step of a single iteration. While
a maximum number of iterations can be specified for, e.g., 
`~photutils.psf.IterativelySubtractedPSFPhotometry`, this block -- specifically, the ender half
-- assesses whether all sources have been found within the ``image``. If no new sources have
been found within an iteration, fitting can be stopped prematurely, avoiding wasteful computational
time. Similarly, the culler aspect of this block examines sources found within an iteration for
quality; it should calculate some goodness-of-fit criterion and reject sources picked up by the
``Finder`` but fall below a given threshold. This block therefore accepts the ``stars``
output `~astropy.table.Table` and returns a new `~astropy.table.Table` with low quality sources
-- such as cosmic ray hits present in the image -- removed as well as a boolean flag indicating
whether the iterative fitting process has reached a conclusion and can be terminated. Please refer
to the :doc:`documentation <culler_and_ender>` for further details.


.. _Single Object Model:

Single Object Model
^^^^^^^^^^^^^^^^^^^
An additional block sitting between the `PSF model`_ and the Fitter_, the ``Single Object Model``
extends the fitting capability of the class from purely point-source objects to extended sources.
This block, effectively, convolves the PSF of the given observation with a physical light
distribution, producing the flux seen in each pixel by the system of such an extended source. This
block therefore requires as its inputs the PSF model and the object type -- a string representing
the type of source (point-like, spiral galaxy, etc.) to create the intrinsic light distribution
for -- and returns a new `~astropy.modeling.Fittable2DModel` instance, the convolution of the PSF
and the true source. Further specifics on this block, including how it will link to `Scene Maker`_
can be found in the :doc:`documentation <single_object_model>`.


.. _Scene Maker:

Scene Maker
^^^^^^^^^^^
The final block within the scope of PSF fitting, the scene maker is also the last to be implemented.
Currently no PSF fitting class extends its functionality to include this block, but it will
eventually generalize and possibly replace the `Group Maker`_ as the method of merging and assigning
non-independent detections within a given image. While current grouping is simply the determining
of point sources with potentially overlapping flux, the *scene* maker will allow for the grouping
of both point sources into an overlapping, simultaneous fit group but also into a single extended
object. Thus each iteration will require a step in which multiple-point-source extended objects are
evaluated for their separation, and newly detected point sources are evaluated for potential
assignment as part of an extended source. The details of this block are available :doc:`here
<scene_maker>`, and will be expanded upon as the API is finalized.