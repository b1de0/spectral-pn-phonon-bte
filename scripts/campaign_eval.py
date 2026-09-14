"""Run loading and multi-axis battery evaluation for archived ladder arms."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pn_shape_battery import battery, gate  # noqa: F401  (gate re-exported)

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data" / "runs"

GAMMA_SANITY = (0.5, 1.5)


@dataclass(frozen=True)
class RunRef:
    L: float
    geometry: str
    subdir: str
    ts: str
    path: Path


def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _battery(d: dict) -> dict:
    """battery() with its benign empty-band nanmean warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return battery(d)
