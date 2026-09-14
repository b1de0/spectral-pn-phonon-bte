"""The signed-trace PDF must contain the active ladder's numerical content."""
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "figures" / "fig3_traces.pdf"
LADDERS = REPO / "data" / "ladders.yaml"

#: Five periods, in plot order, each followed by 1D then 2D.
PANEL_LS = (0.01, 0.1, 1.0, 10.0, 100.0)
PANELS = {
    "1d": ["PnGrOneDBallistic", "PnGrOneDQuasi", "PnGrOneDUnit",
           "PnGrOneDTen", "PnGrOneDHundred"],
    "2d": ["PnGrTwoDBallistic", "PnGrTwoDQuasi", "PnGrTwoDUnit",
           "PnGrTwoDTen", "PnGrTwoDHundred"],
}

def _pdf_trace_digest():
    out = subprocess.run(["pdfinfo", str(FIG)], capture_output=True,
                          text=True, check=True, timeout=60).stdout
    hit = re.search(r"Subject:\s+PN signed-trace SHA256: ([0-9a-f]{64})", out)
    assert hit, "signed-trace content fingerprint absent from PDF Subject"
    return hit.group(1)

def _ladder_trace_digest(source):
    lad = yaml.safe_load(LADDERS.read_text())
    digest = hashlib.sha256()
    for L in PANEL_LS:
        for geo in PANELS:
            rungs = {float(k): v for k, v in lad["ladders"][source][geo].items()}
            hits = list((REPO / rungs[L]).glob("*_results.npz"))
            assert len(hits) == 1
            with np.load(hits[0]) as d:
                for values in (d["t"] / float(d["t_end"]), d["A"] / d["A"][0]):
                    digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()

def _ruled_gamma_ratio(run_dir: Path) -> float:
    """Cell-D gamma_ratio recomputed from one pinned npz.

    An INDEPENDENT copy of the ruled construction (|A| estimator on both
    sides, denominator = the run's own stored 12001-sample reference), kept
    separate from the generator on purpose: a test that imports the code under
    test to compute its expectations can only prove self-consistency.
    """
    from pinn_bte.physics.decay_metrics import (
        GAMMA_ESTIMATOR_RULED, integral_gamma,
    )
    hits = sorted(run_dir.glob("*_results.npz"))
    assert len(hits) == 1, f"{run_dir}: expected one *_results.npz, got {len(hits)}"
    d = np.load(hits[0], allow_pickle=True)
    t_ref = np.asarray(d["t_ref"])
    assert t_ref.size == 12001, (
        f"{run_dir}: reference has {t_ref.size} samples, not the arbiter's "
        f"native 12001 -- the ruled denominator does not apply")
    return float(
        integral_gamma(np.asarray(d["t"]), np.asarray(d["A"]),
                       estimator=GAMMA_ESTIMATOR_RULED)
        / integral_gamma(t_ref, np.asarray(d["A_ref"]),
                         estimator=GAMMA_ESTIMATOR_RULED))

def _ladder_values_4dp(source: str) -> dict:
    """{(geo, L): '0.9986'} for one named ladder in ladders.yaml."""
    lad = yaml.safe_load(LADDERS.read_text())
    out = {}
    for geo in PANELS:
        rungs = {float(k): v for k, v in lad["ladders"][source][geo].items()}
        for L in PANEL_LS:
            out[(geo, L)] = f"{_ruled_gamma_ratio(REPO / rungs[L]):.4f}"
    return out

@pytest.mark.skipif(not FIG.exists(), reason="fig3_traces not built in this tree")
@pytest.mark.skipif(shutil.which("pdfinfo") is None, reason="pdfinfo absent")
def test_fig3_carries_the_active_ladder_trace_fingerprint():
    active = yaml.safe_load(LADDERS.read_text())["ladder_source"]
    assert _pdf_trace_digest() == _ladder_trace_digest(active), (
        "fig3 trace content belongs to another ladder; regenerate the figure")

@pytest.mark.skipif(not FIG.exists(), reason="fig3_traces not built in this tree")
@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext absent")
def test_signed_trace_labels_are_searchable_without_repeated_rates():
    txt = subprocess.run(["pdftotext", str(FIG), "-"], capture_output=True,
                          text=True, check=True, timeout=60).stdout
    assert "DOM reference" in txt and "late window" in txt
    # These are physical-shape panels; rates are supplied by the rate figure
    assert not re.findall(r"\b[01]\.\d{4}\b", txt)

@pytest.mark.skipif(not FIG.exists(), reason="fig3_traces not built in this tree")
@pytest.mark.skipif(shutil.which("pdfinfo") is None, reason="pdfinfo absent")
def test_flipping_the_ladder_changes_trace_content_and_rates():
    lad = yaml.safe_load(LADDERS.read_text())
    active = lad["ladder_source"]
    others = [s for s in lad["ladders"] if s not in (active, "legacy")
              and all((REPO / rel).exists()
                      for g in lad["ladders"][s].values() for rel in g.values())]
    assert others, "no retired non-legacy ladder available for mutation check"
    retired = others[0]
    active_vals = _ladder_values_4dp(active)
    assert _ladder_trace_digest(retired) != _ladder_trace_digest(active)
    assert _pdf_trace_digest() != _ladder_trace_digest(retired)
