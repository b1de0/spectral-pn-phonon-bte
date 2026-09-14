"""Shared pytest fixtures for PINN-pBTE tests."""

import numpy as np
import pytest
import torch

from pinn_bte.config.physics import LATTICE_CONSTANT


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: slow tests (deselect with '-m \"not slow\"')")


@pytest.fixture
def device():
    """Get available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_k_values():
    """Sample k-values spanning the Brillouin zone."""
    k_max = np.pi * 2 / LATTICE_CONSTANT
    return np.array([[0.1], [0.3], [0.5], [0.7], [0.9]]) * k_max


@pytest.fixture
def sample_temperature():
    """Standard reference temperature."""
    return 300.0


@pytest.fixture
def ta_branch():
    """TA branch indicator (0)."""
    return np.array([[0.0]])


@pytest.fixture
def la_branch():
    """LA branch indicator (1)."""
    return np.array([[1.0]])


@pytest.fixture
def knudsen_numbers():
    """Range of Knudsen numbers for testing."""
    return [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


@pytest.fixture
def default_float64():
    """Set torch's default dtype to float64 for the duration of ONE test, then
    restore.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


GOLDEN_TORCH_THREADS = 4


@pytest.fixture
def golden_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(GOLDEN_TORCH_THREADS)
    try:
        yield GOLDEN_TORCH_THREADS
    finally:
        torch.set_num_threads(prev)


@pytest.fixture
def torch_dtype():
    """Default torch dtype for tests."""
    return torch.float32


@pytest.fixture
def numpy_seed():
    """Set numpy random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def torch_seed():
    """Set torch random seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture(autouse=True)
def _mode_source_tripwire(request):
    """DIAGNOSTIC (opt-in via KINETIC_SOURCE_TRIPWIRE=1): fail the FIRST test that
    leaks a patched `phonon_modes` past its own scope.
    """
    import os
    yield
    if os.environ.get("KINETIC_SOURCE_TRIPWIRE") != "1":
        return
    import importlib
    td = importlib.import_module("pinn_bte.physics.ttg_dispersion")
    leaked = []
    if getattr(td.phonon_modes, "func", None) is not None:
        leaked.append("ttg_dispersion.phonon_modes is a functools.partial")
    for name in td._MODE_SOURCE_NAMESPACES:
        mod = importlib.import_module(name)
        fn = getattr(mod, "phonon_modes", None)
        if fn is not None and getattr(fn, "func", None) is not None:
            leaked.append(f"{name}.phonon_modes is a functools.partial")
    assert not leaked, (
        f"MODE-SOURCE LEAK after {request.node.nodeid}: " + "; ".join(leaked))
