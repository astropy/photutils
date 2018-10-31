PSF Fitting
===========

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
        gs [label = "guess at stars"];
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
