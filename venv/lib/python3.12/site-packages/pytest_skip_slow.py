"""
A pytest plugin to skip `@pytest.mark.slow` tests by default. 
Include the slow tests with `--slow`.
"""

import pytest


__version__ = "1.1.0"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "hugemem: mark test as memory intensive")


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        "--run-slow",
        dest="slow",
        action="store_true",
        help="include tests marked slow",
    )
    parser.addoption(
        "--hugemem",
        "--run-hugemem",
        action="store_true",
        help="include tests marked memory intensive",
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skipper = pytest.mark.skip(reason="need --slow option to run")
        for item in filter(lambda it: "slow" in it.keywords, items):
            item.add_marker(skipper)

    if not config.getoption("--hugemem"):
        skipper = pytest.mark.skip(reason="need --hugemem option to run")
        for item in filter(lambda it: "hugemem" in it.keywords, items):
            item.add_marker(skipper)
