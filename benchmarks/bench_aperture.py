"""Run benchmarks for aperture photometry functions."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
import os
import glob
import argparse
from collections import OrderedDict
import numpy as np
from photutils import (aperture_photometry, CircularAperture, CircularAnnulus,
                       EllipticalAperture, EllipticalAnnulus)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-l", "--label", dest="label", default=None,
                    help="Save results to a pickle with this label, so that"
                    "they can be displayed later. Pickles are save in a "
                    "'_results' directory in the same parent directory as "
                    "this script.")
parser.add_argument("-s", "--show", dest="show", action="store_true",
                    default=False, help="Show all results from previously "
                    "labeled runs")
parser.add_argument("-d", "--delete", dest="delete_label", default=None,
                    help="Delete saved results with the given label "
                    "(or 'all' to delete all results). Do not run any "
                    "benchmarks.")
args = parser.parse_args()

if args.show and args.label:
    parser.error("--label doesn't do anything when --show is specified.")
if args.delete_label and (args.label or args.show):
    parser.error("--label and --show do not do anything when --delete is "
                 "specified.")

resultsdir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '_results', 'aperture'))
piknames = glob.glob(os.path.join(resultsdir, '*.pik'))

# Delete saved results, if requested.
if args.delete_label is not None:
    if args.delete_label.lower() == 'all':
        for pikname in piknames:
            os.remove(pikname)
    else:
        try:
            os.remove(os.path.join(resultsdir,
                                   '{0}.pik'.format(args.delete_label)))
        except OSError:
            raise ValueError('No such label exists: {0}'
                             .format(args.delete_label))
    exit()

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
c[name]['multiap']  = False
c[name]['multipos'] = False
c[name]['error'] = False

name = "Small data with error, single small aperture"
c[name] = {}
c[name]['dims']     = (20, 20)
c[name]['pos']      = (10., 10.)
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1000
c[name]['multiap']  = False
c[name]['multipos'] = False
c[name]['error'] = True

name = "Big data, single small aperture"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (500., 500.)
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1000
c[name]['multiap']  = False
c[name]['multipos'] = False
c[name]['error'] = False

name = "Big data with error, single small aperture"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (500., 500.)
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1000
c[name]['multiap']  = False
c[name]['multipos'] = False
c[name]['error'] = True

name = "Big data, single big aperture"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (500., 500.)
c[name]['circ']     = (50.,)
c[name]['circ_ann'] = (50., 60.)
c[name]['elli']     = (50., 20., 0.5)
c[name]['elli_ann'] = (20., 50., 40., 0.5)
c[name]['iter']     = 10
c[name]['multiap']  = False
c[name]['multipos'] = False
c[name]['error'] = False

name = "Big data with error, single big aperture"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (500., 500.)
c[name]['circ']     = (50.,)
c[name]['circ_ann'] = (50., 60.)
c[name]['elli']     = (50., 20., 0.5)
c[name]['elli_ann'] = (20., 50., 40., 0.5)
c[name]['iter']     = 10
c[name]['multiap']  = False
c[name]['multipos'] = False
c[name]['error'] = True

name = "Small data, multiple small apertures"
c[name] = {}
c[name]['dims']     = (20, 20)
c[name]['pos']      = (zip(np.random.uniform(5., 15., 1000), np.random.uniform(5., 15., 1000)))
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1
c[name]['multiap']  = False
c[name]['multipos'] = True
c[name]['error'] = False

name = "Small data with error, multiple small apertures"
c[name] = {}
c[name]['dims']     = (20, 20)
c[name]['pos']      = (zip(np.random.uniform(5., 15., 1000), np.random.uniform(5., 15., 1000)))
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1
c[name]['multiap']  = False
c[name]['multipos'] = True
c[name]['error'] = True

name = "Big data, multiple small apertures"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (zip(np.random.uniform(250., 750., 1000), np.random.uniform(250., 750., 1000)))
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1
c[name]['multiap']  = False
c[name]['multipos'] = True
c[name]['error'] = False

name = "Big data with error, multiple small apertures"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (zip(np.random.uniform(250., 750., 1000), np.random.uniform(250., 750., 1000)))
c[name]['circ']     = (5.,)
c[name]['circ_ann'] = (5., 6.)
c[name]['elli']     = (5., 2., 0.5)
c[name]['elli_ann'] = (2., 5., 4., 0.5)
c[name]['iter']     = 1
c[name]['multiap']  = False
c[name]['multipos'] = True
c[name]['error'] = True

name = "Big data, multiple small apertures, multiple per object"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (zip(np.random.uniform(250., 750., 1000), np.random.uniform(250., 750., 1000)))
c[name]['circ']     = (np.linspace(1., 10., 10).reshape((10, 1)),)
c[name]['iter']     = 1
c[name]['multiap']  = True
c[name]['multipos'] = True
c[name]['error'] = False

name = "Big data with error, multiple small apertures, multiple per object"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (zip(np.random.uniform(250., 750., 1000), np.random.uniform(250., 750., 1000)))
c[name]['circ']     = (np.linspace(1., 10., 10).reshape((10, 1)),)
c[name]['iter']     = 1
c[name]['multiap']  = True
c[name]['multipos'] = True
c[name]['error'] = True

name = "Big data, multiple big apertures"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (zip(np.random.uniform(250., 750., 100), np.random.uniform(250., 750., 100)))
c[name]['circ']     = (50.,)
c[name]['circ_ann'] = (50., 60.)
c[name]['elli']     = (50., 20., 0.5)
c[name]['elli_ann'] = (20., 50., 40., 0.5)
c[name]['iter']     = 1
c[name]['multiap']  = False
c[name]['multipos'] = True
c[name]['error'] = False

name = "Big data with error, multiple big apertures"
c[name] = {}
c[name]['dims']     = (1000, 1000)
c[name]['pos']      = (zip(np.random.uniform(250., 750., 100), np.random.uniform(250., 750., 100)))
c[name]['circ']     = (50.,)
c[name]['circ_ann'] = (50., 60.)
c[name]['elli']     = (50., 20., 0.5)
c[name]['elli_ann'] = (20., 50., 40., 0.5)
c[name]['iter']     = 1
c[name]['multiap']  = False
c[name]['multipos'] = True
c[name]['error'] = True


f = {}
f['circ'] = CircularAperture
f['circ_ann'] = CircularAnnulus
f['elli'] = EllipticalAperture
f['elli_ann'] = EllipticalAnnulus


# Select subset of defined tests and functions to run, to save time.
names_to_run = ["Small data, single small aperture",
                "Big data, single small aperture",
                "Big data with error, single small aperture",
                "Big data, single big aperture",
                "Big data with error, single big aperture",
                "Small data, multiple small apertures",
                "Small data with error, multiple small apertures",
                "Big data, multiple small apertures",
                "Big data with error, multiple small apertures",
                "Big data, multiple big apertures",
                "Big data with error, multiple big apertures",
                "Big data, multiple small apertures, multiple per object",
                "Big data with error, multiple small apertures, multiple per object"]

functions_to_run = ['circ']

if not args.show:

    # Initialize results
    results = OrderedDict()

    # print version information
    print("=" * 79)
    from astropy import __version__
    print("astropy version:", __version__)
    from photutils import __version__
    print("photutils version:", __version__)
    print("numpy version:", np.__version__)
    print("=" * 79)

    for name in names_to_run:

        # Initialize results container for this benchmark
        results[name] = OrderedDict()
        for t in functions_to_run:
            results[name][t] = OrderedDict()

        # Initialize data
        data = np.ones(c[name]['dims'])

        if c[name]['error'] is True:
            error = np.ones(c[name]['dims'])
        else:
            error = None

        # Print header for this benchmark
        print("=" * 79)
        print(name, "  (milliseconds)")
        print("{0}".format("subpixels ="))
        for subpixels in [1, 5, 10, 'exact']:
            print(str(subpixels).center(10) + " ")
        print("")
        print("-" * 79)

        t0 = time.time()

        for t in functions_to_run:

            if t not in c[name]: continue
            print("{0} ".format(t))

            for subpixels in [1, 5, 10, 'exact']:
                time1 = time.time()
                for i in range(c[name]['iter']):
                    # Check whether it is single or multiple apertures
                    if not c[name]['multiap']:
                        if subpixels == 'exact':
                            aperture_photometry(data, f[t](c[name]['pos'],
                                                           *c[name][t]),
                                                method='exact', error=error)
                        else:
                            aperture_photometry(data, f[t](c[name]['pos'],
                                                           *c[name][t]),
                                                method='subpixel', error=error,
                                                subpixels=subpixels)

                    else:
                        if subpixels == 'exact':
                            for index in range(len(c[name][t][0])):
                                aperture_photometry(data, f[t](c[name]['pos'], *c[name][t][0][index]),
                                                    method='exact', error=error)
                        else:
                            for index in range(len(c[name][t][0])):
                                aperture_photometry(data, f[t](c[name]['pos'], *c[name][t][0][index]),
                                                    method='subpixel',
                                                    error=error,
                                                    subpixels=subpixels)

                time2 = time.time()
                time_sec = (time2 - time1) / c[name]['iter']
                print("{0:10.5f} ".format(time_sec * 1000.))
                results[name][t][subpixels] = time_sec
            print("")

        t1 = time.time()

        print("-" * 79)
        print('Real time: {0:10.4f} s'.format(t1 - t0))
        print("")

    # If a label was specified, save results to a pickle.
    if args.label is not None:
        import pickle
        pikname = os.path.join(resultsdir, '{0}.pik'.format(args.label))
        if not os.path.exists(resultsdir):
            os.makedirs(resultsdir)
        outfile = open(pikname, 'w')
        pickle.dump(results, outfile)
        outfile.close()

if args.show:
    import pickle

    # Load pickled results
    results = OrderedDict()
    piknames = glob.glob(os.path.join(resultsdir, '*.pik'))
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
        print("=" * 79)
        print("{0} (milliseconds)".format(name))
        print(" " * 45, "subpixels")
        print("{0}s {1}s".format("function", "label"))
        for subpixels in [1, 5, 10, 'exact']:
            print(str(subpixels).center(8) + " ")
        print("")
        print("-" * 79)

        for t in functions_to_run:
            for label, result in results.iteritems():
                if t not in result[name]: continue
                print("{0}s {1}s".format(t, label))
                for subpixels in [1, 5, 10, 'exact']:
                    time_sec = result[name][t][subpixels]
                    print("{0:8.3f} " .format(time_sec * 1000.))
                print("")

        print("-" * 79)
        print("")
