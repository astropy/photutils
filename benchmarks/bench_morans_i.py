#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the Moran's I source masking functions.

The timing benchmark covers ``make_morans_i_map`` and ``make_morans_i_mask``
across image sizes.

The memory benchmark records the resident memory of a child process
while it makes a source mask, and plots the memory against the wall
time. Each implementation runs in its own child process, so the traces
are independent. Besides the photutils implementation, the benchmark
can trace an earlier version of ``photutils/background/morans_i.py``
taken from a git revision (e.g., the pure NumPy prototype) and the
original implementation in a clone of the ``moransi_sourcemasking``
repository. All of the implementations flag the same percentage of the
pixels so that they do comparable work.

The memory benchmark requires the optional ``psutil`` and
``matplotlib`` packages.

Run ``python benchmarks/bench_morans_i.py --help`` to see the available
options.
"""

import argparse
import csv
import importlib.util
import multiprocessing
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
from bench_helpers import (add_memory_argument, parse_int_list,
                           print_environment, set_memory_mode, time_best)

from photutils.background import make_morans_i_map, make_morans_i_mask

MB = 1024.0**2
SERIES_COLORS = ['#2a78d6', '#eb6834', '#3d9a5f', '#8a5cc2', '#c2415a',
                 '#6b7280']


def make_source_image(size, *, seed=0):
    """
    Make an image of Gaussian sources on a noisy background.

    Parameters
    ----------
    size : int
        The image is ``(size, size)``.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The image, with about 200 sources per million pixels.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 1.0, (size, size))
    half = 40
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    radius_sq = xx**2 + yy**2
    n_sources = max(1, int(200 * size**2 / 1.0e6))
    for _ in range(n_sources):
        ycen, xcen = rng.integers(half, size - half, 2)
        sigma = rng.uniform(1.5, 8.0)
        amplitude = 10.0**rng.uniform(0.0, 3.0)
        cutout = (slice(ycen - half, ycen + half + 1),
                  slice(xcen - half, xcen + half + 1))
        data[cutout] += amplitude * np.exp(-radius_sq / (2.0 * sigma**2))
    return data


def bench_timing(sizes, *, repeats=3, seed=0):
    """
    Benchmark the map and mask functions.

    Parameters
    ----------
    sizes : list of int
        The image sizes. Each image is ``(size, size)``.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print("\n== Moran's I map and source mask ==")
    print(f'{"function":>20} {"image":>12} {"time":>12}')

    for size in sizes:
        data = make_source_image(size, seed=seed)
        for func in (make_morans_i_map, make_morans_i_mask):
            best = time_best(lambda: func(data),  # noqa: B023
                             repeats=repeats)
            print(f'{func.__name__:>20} {f"{size}x{size}":>12} '
                  f'{best:>11.4f}s')


def load_module(path, name):
    """
    Import a Python source file as a module.

    Parameters
    ----------
    path : `~pathlib.Path`
        The source file.

    name : str
        The name given to the module.

    Returns
    -------
    result : module
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_implementation(spec, percent):
    """
    Return the callable for an implementation specification.

    Parameters
    ----------
    spec : str
        One of ``'photutils'``, ``'git:<revision>'``, or
        ``'upstream:<path>'``.

    percent : float
        The percentage of the pixels to flag as sources.

    Returns
    -------
    result : callable
        A function that takes the image and makes the source mask.
    """
    kind, _, value = spec.partition(':')
    if kind == 'photutils':
        return lambda data: make_morans_i_mask(data, percent=percent)

    if kind == 'git':
        repo = Path(__file__).resolve().parent.parent
        command = ['git', 'show',
                   f'{value}:photutils/background/morans_i.py']
        source = subprocess.run(command, cwd=repo, check=True,  # noqa: S603
                                capture_output=True, text=True).stdout
        tmpdir = Path(tempfile.mkdtemp())
        path = tmpdir / 'morans_i_revision.py'
        path.write_text(source)
        module = load_module(path, 'morans_i_revision')
        return lambda data: module.make_morans_i_mask(data, percent=percent)

    # The upstream module imports two packages that are used only to
    # read configuration files
    for name in ('yaml', 'box'):
        if importlib.util.find_spec(name) is None:
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name].Box = dict
    path = Path(value).expanduser() / 'src/moransi_sourcemasking/moransi.py'
    module = load_module(path, 'moransi_upstream')
    source_filter = module.SlidingMoranSourceFilter(
        threshold_type='percentage', threshold_value=percent)
    return source_filter.flag_sources


def trace_child(spec, size, seed, percent, repeats, conn):
    """
    Run one implementation in a child process.

    The child tells the parent when its inputs are ready, makes the
    mask once while the parent samples its memory, and then repeats
    the call to get the best time.

    Parameters
    ----------
    spec : str
        The implementation specification (see
        `resolve_implementation`).

    size : int
        The image is ``(size, size)``.

    seed : int
        The random number generator seed.

    percent : float
        The percentage of the pixels to flag as sources.

    repeats : int
        The number of calls used to get the best time.

    conn : `multiprocessing.connection.Connection`
        The connection to the parent process.
    """
    data = make_source_image(size, seed=seed)
    func = resolve_implementation(spec, percent)
    conn.send('ready')
    conn.recv()

    t0 = time.perf_counter()
    func(data)
    elapsed = time.perf_counter() - t0
    conn.send(elapsed)
    conn.recv()

    times = [elapsed]
    for _ in range(repeats - 1):
        t0 = time.perf_counter()
        func(data)
        times.append(time.perf_counter() - t0)
    conn.send(min(times))


def trace_memory(spec, *, size, seed, percent, repeats, interval):
    """
    Trace the resident memory of one implementation.

    Parameters
    ----------
    spec : str
        The implementation specification (see
        `resolve_implementation`).

    size : int
        The image is ``(size, size)``.

    seed : int
        The random number generator seed.

    percent : float
        The percentage of the pixels to flag as sources.

    repeats : int
        The number of calls used to get the best time.

    interval : float
        The memory sampling interval in seconds.

    Returns
    -------
    result : dict
        The ``t`` (seconds since the start of the call) and ``rss``
        (bytes above the level before the call) sample arrays, the
        ``peak`` of ``rss``, and the ``best`` time in seconds.
    """
    import psutil

    ctx = multiprocessing.get_context('spawn')
    conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=trace_child,
                       args=(spec, size, seed, percent, repeats, child_conn))
    proc.start()
    try:
        conn.recv()
        child = psutil.Process(proc.pid)
        baseline = child.memory_info().rss
        samples = [(0.0, baseline)]
        conn.send('go')
        t0 = time.perf_counter()
        while not conn.poll(interval):
            samples.append((time.perf_counter() - t0,
                            child.memory_info().rss))
        elapsed = conn.recv()
        samples.append((elapsed, child.memory_info().rss))
        conn.send('continue')
        best = conn.recv()
    finally:
        proc.join()

    t, rss = np.array(samples).T
    rss -= baseline
    return {'t': t, 'rss': rss, 'peak': rss.max(), 'best': best}


def plot_traces(traces, *, size, interval, output):
    """
    Plot the resident memory against the wall time.

    Parameters
    ----------
    traces : dict
        Maps the implementation labels to the results of
        `trace_memory`.

    size : int
        The image is ``(size, size)``.

    interval : float
        The memory sampling interval in seconds.

    output : `~pathlib.Path`
        The PNG file to write.
    """
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # The second panel zooms in on the photutils traces, which are
    # otherwise hidden by much slower implementations
    photutils_labels = [label for label in traces
                        if label.startswith('photutils')]
    panels = [list(traces)]
    if 0 < len(photutils_labels) < len(traces):
        panels.append(photutils_labels)

    colors = {label: SERIES_COLORS[i % len(SERIES_COLORS)]
              for i, label in enumerate(traces)}
    image_mb = size**2 * 8 / MB
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(7.0 * len(panels), 5.5),
                             facecolor='white')
    for ax, labels in zip(axes[0], panels, strict=True):
        for label in labels:
            trace = traces[label]
            ax.plot(trace['t'], trace['rss'] / MB, color=colors[label],
                    lw=1.6, drawstyle='steps-post',
                    label=(f'{label}: {trace["best"]:.3f} s, peak '
                           f'{trace["peak"] / MB:.0f} MB'))
        ax.axhline(image_mb, color='#6b7280', lw=1.0, ls='--',
                   label=f'input image ({image_mb:.0f} MB)')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Wall time since the start of the call (s)')
        ax.set_ylabel('Resident memory above the level before the call '
                      '(MB)')
        ax.grid(visible=True, color='#e5e7eb', lw=0.8)
        ax.set_axisbelow(True)
        for name in ('top', 'right'):
            ax.spines[name].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc='best')
    axes[0, 0].set_title('All implementations', fontsize=10, loc='left')
    if len(panels) > 1:
        axes[0, 1].set_title('photutils only', fontsize=10, loc='left')

    fig.suptitle(f"Moran's I source mask: resident memory vs wall time "
                 f'({size}x{size} float64 image, first call, sampled '
                 f'every {interval * 1000:g} ms)', fontsize=11, x=0.02,
                 ha='left')
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def bench_memory(specs, *, size, seed, percent, repeats, interval, output):
    """
    Trace and plot the memory of the implementations.

    Parameters
    ----------
    specs : dict
        Maps the implementation labels to their specifications (see
        `resolve_implementation`).

    size : int
        The image is ``(size, size)``.

    seed : int
        The random number generator seed.

    percent : float
        The percentage of the pixels to flag as sources.

    repeats : int
        The number of calls used to get the best time.

    interval : float
        The memory sampling interval in seconds.

    output : `~pathlib.Path`
        The PNG file to write. The samples are written to a CSV file
        with the same stem.
    """
    image_mb = size**2 * 8 / MB
    print(f'\n== Source mask memory ({size}x{size} float64 image, '
          f'{image_mb:.0f} MB, percent={percent:g}) ==')
    print(f'{"implementation":>28} {"best time":>12} {"peak memory":>14} '
          f'{"peak / image":>14}')

    traces = {}
    for label, spec in specs.items():
        trace = trace_memory(spec, size=size, seed=seed, percent=percent,
                             repeats=repeats, interval=interval)
        traces[label] = trace
        peak_mb = trace['peak'] / MB
        print(f'{label:>28} {trace["best"] * 1000:>9.1f} ms '
              f'{peak_mb:>11.1f} MB {peak_mb / image_mb:>14.2f}')

    csv_output = output.with_suffix('.csv')
    with open(csv_output, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['implementation', 't_s', 'rss_above_start_bytes'])
        for label, trace in traces.items():
            writer.writerows((label, f'{t:.6f}', int(rss)) for t, rss
                             in zip(trace['t'], trace['rss'], strict=True))

    plot_traces(traces, size=size, interval=interval, output=output)
    print(f'wrote {output} and {csv_output}')


def main():
    """
    Run the Moran's I benchmarks.
    """
    parser = argparse.ArgumentParser(
        description="Benchmarks for the Moran's I source masking "
                    'functions.')
    parser.add_argument('--sizes', default='1024,2048,4096',
                        help='comma-separated image sizes for the timing '
                             'benchmark (default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'timing', 'memory'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    parser.add_argument('--memory-size', type=int, default=4096,
                        help='image size for the memory benchmark '
                             '(default: %(default)s)')
    parser.add_argument('--percent', type=float, default=35.0,
                        help='percentage of the pixels flagged as sources '
                             'in the memory benchmark '
                             '(default: %(default)s)')
    parser.add_argument('--interval', type=float, default=0.002,
                        help='memory sampling interval in seconds '
                             '(default: %(default)s)')
    parser.add_argument('--git-revision', action='append', default=[],
                        metavar='REV',
                        help='also trace photutils/background/morans_i.py '
                             'from this git revision, which must not '
                             'need compiled code (e.g., the NumPy '
                             'prototype commit); can be repeated')
    parser.add_argument('--upstream', default=None, metavar='PATH',
                        help='also trace the original implementation in '
                             'this clone of the moransi_sourcemasking '
                             'repository')
    parser.add_argument('--output', type=Path,
                        default=Path('morans_i_memory.png'),
                        help='the memory plot file (default: %(default)s)')
    add_memory_argument(parser)
    args = parser.parse_args()
    set_memory_mode(args.memory)

    sizes = parse_int_list(args.sizes)

    print_environment()

    if args.which in ('all', 'timing'):
        bench_timing(sizes, repeats=args.repeats, seed=args.seed)

    if args.which in ('all', 'memory'):
        specs = {}
        if args.upstream is not None:
            specs['upstream'] = f'upstream:{args.upstream}'
        for revision in args.git_revision:
            specs[f'git {revision}'] = f'git:{revision}'
        specs['photutils'] = 'photutils'
        bench_memory(specs, size=args.memory_size, seed=args.seed,
                     percent=args.percent, repeats=args.repeats,
                     interval=args.interval, output=args.output)


if __name__ == '__main__':
    main()
