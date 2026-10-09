from __future__ import annotations

import pytest

from pipelineguard.real_benchmark import run_benchmark
from pipelineguard.real_data import load_wellhead_data


@pytest.fixture(scope="session")
def wellheads():
    """The real wellhead fleet, with the corrosion reading kept for the tests."""
    return load_wellhead_data(with_corrosion=True)


@pytest.fixture(scope="session")
def report(wellheads):
    """The full real-data benchmark, run once (it is the expensive fixture)."""
    return run_benchmark(wellheads)
