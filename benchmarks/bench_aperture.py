"""Run benchmarks for aperture photometry functions."""

import time
import argparse
import numpy as np
import photutils
from collections import OrderedDict

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-l", "--label", dest="label", default=None, 
                    help="Save results to a pickle with this label, so that"
                    "they can be displayed later")
parser.add_argument("-s", "--show", dest="show", action="store_true",
                    default=False, help="Show all results from previously "
                    "labeled runs.")
args = parser.parse_args()

if args.show and args.label:
    parser.error("--label doesn't do anything when --show is specified.")

c = OrderedDict()

name = "Small data, single small aperture"
c[name] = {}
c[name]['dims']     = (20, 20)
c[name]['pos']      = (10., 10.)
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1000

name = "Big data, single small aperture"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (500., 500.)
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1000

name = "Big data, single big aperture"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (500., 500.)
c[name]['circ']     = (50.,)
c[name]['circ_ann'] = (50., 60.)
c[name]['elli']     = (50., 20., 0.5)
c[name]['elli_ann'] = (20., 50., 40., 0.5)
c[name]['iter']     = 10

name = "Small data, multiple small apertures"
c[name] = {}
c[name]['dims']     = (20, 20)
c[name]['pos']      = (np.random.uniform(5., 15., 1000), np.random.uniform(5., 15., 1000))
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1

name = "Big data, multiple small apertures"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (np.random.uniform(250., 750., 1000), np.random.uniform(250., 750., 1000))
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1

name = "Big data, multiple small apertures, multiple per object"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (np.random.uniform(250., 750., 1000), np.random.uniform(250., 750., 1000))
c[name]['circ']     = (np.linspace(1., 10., 10).reshape((10, 1)),)
c[name]['iter']     = 1

name = "Big data, multiple big apertures"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (np.random.uniform(250., 750., 100), np.random.uniform(250., 750., 100))
c[name]['circ']     = (50.,)
c[name]['circ_ann'] = (50., 60.)
c[name]['elli']     = (50., 20., 0.5)
c[name]['elli_ann'] = (20., 50., 40., 0.5)
c[name]['iter']     = 1


f = {}
f['circ'] = photutils.aperture_circular
f['circ_ann'] = photutils.annulus_circular
f['elli'] = photutils.aperture_elliptical
f['elli_ann'] = photutils.annulus_elliptical


# Select subset of defined tests and functions to run, to save time.
names_to_run = ["Small data, single small aperture",
                "Big data, single small aperture",
                "Big data, single big aperture",
                "Small data, multiple small apertures",
                "Big data, multiple small apertures, multiple per object",
                "Big data, multiple small apertures",
                "Big data, multiple big apertures"]
functions_to_run = ['circ']

if not args.show:

    # Initialize results
    results = OrderedDict()

    for name in names_to_run:

        # Initialize results container for this benchmark
        results[name] = OrderedDict()
        for t in functions_to_run:
            results[name][t] = OrderedDict()

        # Initialize data
        x, y = c[name]['pos']
        data = np.ones(c[name]['dims'])

        # Print header for this benchmark
        print "=" * 79
        print name, "  (milliseconds)"
        print "%30s " % ("subpixels ="),
        for subpixels in [1, 5, 10, 'exact']:
            print str(subpixels).center(10) + " ",
        print ""
        print "-" * 79

        t0 = time.time()

        for t in functions_to_run:

            if t not in c[name]: continue
            print "%30s " % t,

            for subpixels in [1, 5, 10, 'exact']:
                time1 = time.time()
                for i in range(c[name]['iter']):
                    f[t](data, x, y, *c[name][t], subpixels=subpixels)
                time2 = time.time()
                time_sec = (time2 - time1) / c[name]['iter']
                print "%10.5f " % (time_sec * 1000.),
                results[name][t][subpixels] = time_sec
            print ""

        t1 = time.time()

        print "-" * 79
        print 'Real time: %10.4f s' % (t1 - t0)
        print ""

    # If a label was specified, save results to a pickle.
    if args.label is not None:
        import os
        import pickle

        pikname = '_results/aperture/{0}.pik'.format(args.label)
        if not os.path.exists('_results/aperture'):
            os.makedirs('_results/aperture')
        outfile = open(pikname, 'w')
        pickle.dump(results, outfile)
        outfile.close()

if args.show:

    import glob
    import os.path
    import pickle

    # Load pickled results
    results = OrderedDict()
    piknames = glob.glob('_results/aperture/*.pik')
    for pikname in piknames:
        label = os.path.basename(pikname)[:-4]
        infile = open(pikname, 'r')
        results[label] = pickle.load(infile)
        infile.close()
    if len(results) == 0:
        raise RuntimeError('No saved results.')

    # Loop over different cases
    firstlabel = results.keys()[0]
    for name in results[firstlabel]:
   
        # Print header for this case
        print "=" * 79
        print "%-63s (milliseconds)" % name
        print " " * 45, "subpixels"
        print "%15s %20s" % ("function", "label"),
        for subpixels in [1, 5, 10, 'exact']:
            print str(subpixels).center(8) + " ",
        print ""
        print "-" * 79
        
        for t in functions_to_run:
            for label, result in results.iteritems():
                if t not in result[name]: continue
                print "%15s %20s" % (t, label),
                for subpixels in [1, 5, 10, 'exact']:
                    time_sec = result[name][t][subpixels]
                    print "%8.3f " % (time_sec * 1000.),
                print ""

        print "-" * 79
        print ""
