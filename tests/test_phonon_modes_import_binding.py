"""Static census of phonon_modes bindings: every call site is reachable by a
mode-source manager, declares its measure, or is allow-listed with a reason.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_REPO = Path(__file__).resolve().parents[1]

_ZONES = ("pinn_bte", "experiments", "scripts")

_INDIRECT_PROVIDERS = ("holland", "holland_nu", "joint_modes", "fullbz_modes")

_ALLOWED: dict[tuple[str, str], str] = {}

_INDIRECT_IMPORT_TIME: dict[tuple[str, str], str] = {
}


def _python_files() -> list[Path]:
    out: list[Path] = []
    for zone in _ZONES:
        out.extend(sorted((_REPO / zone).rglob("*.py")))
    assert out, "no sources found -- the zone list or the repo layout moved"
    return out


def _dotted(path: Path) -> str:
    return str(path.relative_to(_REPO).with_suffix("")).replace("/", ".")


def _has_module_scope_import(tree: ast.Module, name: str) -> bool:
    """`from ... import <name>` at module scope == a binding no call can rebind."""
    return any(isinstance(n, ast.ImportFrom)
               and any(a.name == name for a in n.names)
               for n in ast.iter_child_nodes(tree))


def _calls(tree: ast.Module, names: tuple[str, ...] | set[str]) -> list[dict]:
    """Every call to `names`, tagged with how and when it resolves."""
    found: list[dict] = []

    def walk(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = in_function or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    name, how = func.id, "name"
                elif isinstance(func, ast.Attribute):
                    name, how = func.attr, "attribute"
                else:
                    name, how = None, None
                if name in names:
                    found.append({
                        "line": child.lineno,
                        "name": name,
                        "how": how,
                        # a call at module scope (including a class body or an
                        # `if __name__ == "__main__"` block) runs at import
                        "import_time": not in_function,
                        "declared": any(kw.arg == "measure"
                                        for kw in child.keywords),
                        "expr": ast.unparse(child),
                    })
            walk(child, nested)

    walk(tree, False)
    return found


def _unreachable_undeclared() -> dict[tuple[str, str], int]:
    """{(relpath, call expression): line} for every hazardous `phonon_modes`."""
    from pinn_bte.physics.ttg_dispersion import _MODE_SOURCE_NAMESPACES

    hazards: dict[tuple[str, str], int] = {}
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:  # a file we cannot read is a file we cannot gate
            raise AssertionError(f"cannot parse {path}: {exc}") from exc
        bound_at_module_scope = _has_module_scope_import(tree, "phonon_modes")
        patched = _dotted(path) in _MODE_SOURCE_NAMESPACES
        for call in _calls(tree, {"phonon_modes"}):
            if call["declared"]:
                continue  # says which comb it wants; a default flip cannot move it
            unreachable = call["import_time"] or (
                call["how"] == "name" and bound_at_module_scope and not patched)
            if unreachable:
                rel = str(path.relative_to(_REPO))
                hazards[(rel, call["expr"])] = call["line"]
    return hazards


def test_no_unreachable_undeclared_phonon_modes_call():
    hazards = _unreachable_undeclared()
    unlisted = {k: v for k, v in hazards.items() if k not in _ALLOWED}
    assert not unlisted, (
        "phonon_modes bound where no mode source can reach it, and without an "
        "explicit `measure=`:\n" + "\n".join(
            f"  {rel}:{line}  {expr}" for (rel, expr), line in
            sorted(unlisted.items(), key=lambda kv: (kv[0][0], kv[1]))) +
        "\nSuch a site silently uses whatever the default comb is on the day it "
        "runs (the default has moved once already). Declare it, route it, or "
        "allow-list it with a reason.")


def test_allow_list_has_no_dead_entries():
    hazards = _unreachable_undeclared()
    dead = sorted(set(_ALLOWED) - set(hazards))
    assert not dead, (
        "allow-listed sites that are no longer unreachable-and-undeclared "
        "(fixed, moved, or re-spelled -- delete the entry or re-argue it):\n" +
        "\n".join(f"  {rel}  {expr}" for rel, expr in dead))


def test_every_allow_list_entry_carries_a_reason():
    """Guard the guard: a blank reason would re-open the defect silently."""
    for key, reason in _ALLOWED.items():
        assert len(reason.strip()) >= 40, f"{key}: reason too thin: {reason!r}"


def test_every_patched_namespace_still_binds_phonon_modes():
    """A listed namespace that stopped importing the name protects nothing."""
    import importlib

    from pinn_bte.physics.ttg_dispersion import _MODE_SOURCE_NAMESPACES

    for dotted in _MODE_SOURCE_NAMESPACES:
        path = _REPO / (dotted.replace(".", "/") + ".py")
        assert path.exists(), f"{dotted} is in the patch list but has no source"
        tree = ast.parse(path.read_text())
        assert _has_module_scope_import(tree, "phonon_modes"), (
            f"{dotted} is in _MODE_SOURCE_NAMESPACES but no longer imports "
            "phonon_modes at module scope -- patching it is a no-op")
        module = importlib.import_module(dotted)
        assert hasattr(module, "phonon_modes"), (
            f"{dotted}.phonon_modes does not exist at run time")


def test_patched_namespaces_have_no_import_time_call():
    """List membership does NOT rescue an import-time call, so there must be none."""
    from pinn_bte.physics.ttg_dispersion import _MODE_SOURCE_NAMESPACES

    for dotted in _MODE_SOURCE_NAMESPACES:
        path = _REPO / (dotted.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text())
        bad = [c for c in _calls(tree, {"phonon_modes"}) if c["import_time"]]
        assert not bad, (
            f"{dotted} calls phonon_modes at import time (lines "
            f"{[c['line'] for c in bad]}) -- being in the patch list does not "
            "help, the call runs before any manager is entered")


def test_indirect_comb_providers_bound_at_import_time_are_the_pinned_census():
    """`holland()` freezes a comb exactly as hard as `phonon_modes()` does."""
    census: dict[tuple[str, str], int] = {}
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for call in _calls(tree, set(_INDIRECT_PROVIDERS)):
            if call["import_time"]:
                census[(str(path.relative_to(_REPO)), call["expr"])] = call["line"]
    new = {k: v for k, v in census.items() if k not in _INDIRECT_IMPORT_TIME}
    gone = sorted(set(_INDIRECT_IMPORT_TIME) - set(census))
    assert not new, (
        "new import-time binding of a derived comb provider:\n" + "\n".join(
            f"  {rel}:{line}  {expr}" for (rel, expr), line in sorted(
                new.items(), key=lambda kv: (kv[0][0], kv[1]))) +
        "\nThese take no `measure=` keyword, so they cannot declare a comb -- "
        "resolve them at call time or argue the invariance here.")
    assert not gone, (
        "pinned indirect-provider sites that no longer exist (delete the "
        f"census entries): {gone}")


@pytest.mark.parametrize("key,reason", sorted(_INDIRECT_IMPORT_TIME.items()))
def test_untriaged_indirect_sites_stay_labelled_untriaged(key, reason):
    """The census must not quietly become a clearance."""
    assert "UNTRIAGED" in reason or "MEASURED" in reason, (
        f"{key}: an indirect-provider entry must say whether its invariance was "
        f"measured; got {reason!r}")


_ARBITER_ENTRY_POINTS = frozenset({
    "ttg_decay_spectral", "ttg_amplitude_curve",
    "ttg_window_crossing", "dominant_gamma",
})
_MODE_MANAGERS = frozenset({"shipped_mode_source", "joint_mode_source"})

#: (relpath, method, entry point) -> reason. Every entry says WHY it is safe, or
#: says UNTRIAGED and names the hypothesis that owns it.
_ARBITER_ALLOWED: dict[tuple[str, str, str], str] = {
    **{("pinn_bte/training/transient/spectral_pn_trainer.py", m, e):
       "MEASURED SAFE for the training path: "
       "experiments/transient/run_spectral_pn.py wraps the entire run in "
       "`mode_source_ctx(args.mode_source, Nk=args.nk)` (joint or shipped); the "
       "comb is declared on the command line and recorded in extra_meta."
       for m, e in (("select_window", "ttg_decay_spectral"),
                    ("select_window", "ttg_window_crossing"),
                    ("build_si_case", "ttg_amplitude_curve"))},
}


def _undeclared_arbiter_calls() -> dict[tuple[str, str, str], list[int]]:
    """{(relpath, enclosing method, entry point): [lines]} for trainer calls to a
    comb-consuming arbiter that are not lexically inside a mode manager.
    """
    hits: dict[tuple[str, str, str], list[int]] = {}
    root = _REPO / "pinn_bte" / "training"
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            raise AssertionError(f"cannot parse {path}: {exc}") from exc
        # annotate parents so a call can ask about its ancestors
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child._census_parent = node  # type: ignore[attr-defined]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name not in _ARBITER_ENTRY_POINTS:
                continue
            wrapped, method, cur = False, "<module>", node
            while (cur := getattr(cur, "_census_parent", None)) is not None:
                if isinstance(cur, ast.With) and any(
                        isinstance(i.context_expr, ast.Call) and
                        getattr(i.context_expr.func, "id",
                                getattr(i.context_expr.func, "attr", None))
                        in _MODE_MANAGERS for i in cur.items):
                    wrapped = True
                    break
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method = cur.name
            if not wrapped:
                key = (str(path.relative_to(_REPO)), method, name)
                hits.setdefault(key, []).append(node.lineno)
    return hits


def test_no_trainer_calls_an_arbiter_without_declaring_the_comb():
    unlisted = {k: v for k, v in _undeclared_arbiter_calls().items()
                if k not in _ARBITER_ALLOWED}
    assert not unlisted, (
        "trainer calls a comb-consuming arbiter with no declared comb:\n" +
        "\n".join(f"  {rel}:{lines}  {meth}() -> {ep}"
                  for (rel, meth, ep), lines in sorted(unlisted.items())) +
        "\nThe library resolves `phonon_modes` internally, so this silently "
        "uses whatever the default is on the day it runs.")


def test_the_arbiter_census_is_not_vacuous():
    """The census must actually be looking at something."""
    found = _undeclared_arbiter_calls()
    assert len(found) >= 3, (
        f"the arbiter census found only {len(found)} sites; it is supposed to "
        "see at least PN's 3 (the runner-wrapped ones). "
        "Did the entry-point names or the trainer layout "
        "move? A census that sees nothing passes for the wrong reason.")


@pytest.mark.parametrize("key,reason", sorted(_ARBITER_ALLOWED.items()))
def test_arbiter_allow_list_entries_state_their_status(key, reason):
    """Every allow-list entry says MEASURED SAFE or UNTRIAGED -- never neither."""
    assert "MEASURED SAFE" in reason or "UNTRIAGED" in reason, (
        f"{key}: an arbiter allow-list entry must say whether it was measured "
        f"safe or is still untriaged; got {reason!r}")


def test_the_arbiter_allow_list_has_no_dead_entries():
    found = _undeclared_arbiter_calls()
    dead = sorted(k for k in _ARBITER_ALLOWED if k not in found)
    assert not dead, (
        "arbiter allow-list entries whose call site is no longer undeclared "
        "(fixed, moved or renamed) -- delete them:\n" +
        "\n".join(f"  {rel}  {meth}() -> {ep}" for rel, meth, ep in dead))
