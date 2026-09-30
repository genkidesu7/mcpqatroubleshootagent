# PB-2 + PB-5: State Safety Verification
# PB-5 applies only after checkpointing is enabled in config/config.yaml and the
# installed agentcore exposes its pre-checkpoint ingress-protection hooks.

from __future__ import annotations

import ast
import pathlib
import re
from collections.abc import Iterator
from typing import Any

import pytest

CREDENTIAL_FIELD_PATTERNS = re.compile(
    r"(jwt|token|api_key|secret|password|credential|connection_string)", re.IGNORECASE
)
PROHIBITED_TYPE_ANNOTATIONS = ["BaseModel", "InvocationContext"]
_RUNTIME_CONFIG_PATH = pathlib.Path(__file__).parents[2] / "config" / "config.yaml"


def _checkpointing_enabled() -> bool:
    """Read runtime checkpoint settings; absent config is a fleet-transition waiver."""
    if not _RUNTIME_CONFIG_PATH.exists():
        return False
    try:
        import yaml

        config = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text()) or {}
    except Exception:
        return False
    return bool(config.get("memory_enabled") or config.get("hitl", {}).get("enabled", False))


def _framework_ingress_protection_available() -> bool:
    """Fail closed to a waiver until both agentcore ingress hooks are importable."""
    try:
        from framework.graph.base_graph import BaseGraph
    except Exception:
        return False
    return all(hasattr(BaseGraph, hook) for hook in ("_sanitize_ingress", "_sanitize_resume_feedback"))


def _scan_state_file(filepath: str) -> list[str]:
    """Scan a state definition file for safety violations."""
    source = pathlib.Path(filepath).read_text()
    tree = ast.parse(source, filename=filepath)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_name = item.target.id
                    if CREDENTIAL_FIELD_PATTERNS.search(field_name):
                        violations.append(f"{filepath}:{item.lineno} — Credential-like field name: {field_name}")
                    if item.annotation:
                        annotation_str = ast.dump(item.annotation)
                        for prohibited in PROHIBITED_TYPE_ANNOTATIONS:
                            if prohibited in annotation_str:
                                violations.append(f"{filepath}:{item.lineno} — Prohibited type in State: {prohibited}")
    return violations


def _walk_checkpoint_surface(value: Any, path: str = "checkpoint") -> Iterator[tuple[str, Any]]:
    """Yield every persisted value/key without lossy object-to-string conversion."""
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_checkpoint_surface(key, f"{path}.<key>")
            yield from _walk_checkpoint_surface(child, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, child in enumerate(value):
            yield from _walk_checkpoint_surface(child, f"{path}[{index}]")


def _assert_raw_ingress_absent(surfaces: dict[str, Any], raw_values: dict[str, Any]) -> None:
    """Fail when raw ingress occurs exactly or embedded in a persisted string leaf/key."""
    for surface, value in surfaces.items():
        for path, persisted in _walk_checkpoint_surface(value, surface):
            for name, raw in raw_values.items():
                if raw in (None, "", {}, []):
                    continue
                if persisted == raw:
                    pytest.fail(f"PB-5 raw {name} found at {path}; ingress crossed persistence boundary")
                if isinstance(raw, str) and isinstance(persisted, str) and raw in persisted:
                    pytest.fail(f"PB-5 raw {name} found within string at {path}; ingress crossed persistence boundary")
                if isinstance(raw, dict):
                    for key, raw_value in raw.items():
                        if isinstance(raw_value, str) and isinstance(persisted, str) and raw_value in persisted:
                            pytest.fail(
                                f"PB-5 raw {name}[{key!r}] found within string at {path}; "
                                "ingress crossed persistence boundary"
                            )


class TestStateSafety:
    """PB-2/PB-5: State must be msgpack-safe with no credentials."""

    def test_state_file_safety(self):
        state_file = pathlib.Path(__file__).parents[2] / "src" / "schemas" / "state.py"
        if not state_file.exists():
            pytest.skip("src/schemas/state.py not found")
        violations = _scan_state_file(str(state_file))
        assert violations == [], "State safety violations found:\n" + "\n".join(violations)


_PB5_APPLICABLE = _checkpointing_enabled() and _framework_ingress_protection_available()
_PB5_WAIVER_REASON = (
    "config/config.yaml enables neither memory_enabled nor hitl.enabled — PB-5 auto-waived"
    if not _checkpointing_enabled()
    else "installed agentcore lacks BaseGraph ingress hooks — PB-5 auto-waived pending framework cutover"
)


@pytest.mark.skipif(not _PB5_APPLICABLE, reason=_PB5_WAIVER_REASON)
def test_pb5_precheckpoint_ingress_not_raw() -> None:
    """PB-5: inspect every persisted surface after unique PII-bearing ingress.

    TODO after agentcore capability cutover: build this template graph with its runtime
    checkpointer; invoke with unique user_input/input_context values; resume with unique
    feedback for HITL; then pass checkpoint, metadata, and pending_writes to the helper.
    Do not use json.dumps(..., default=str) or stringify unknown objects.
    """
    pytest.fail(
        "PB-5 is applicable (checkpointing enabled + framework ingress hooks present) but "
        "this template has not wired a graph/checkpointer fixture — see the TODO above. "
        "Build the fixture and call _assert_raw_ingress_absent() before shipping; once the "
        "capability that makes PB-5 apply is live, this must run the real assertion, not skip."
    )

    # checkpoint_tuple = checkpointer.get_tuple(config)
    # _assert_raw_ingress_absent(
    #     {"checkpoint": checkpoint_tuple.checkpoint,
    #      "metadata": checkpoint_tuple.metadata,
    #      "pending_writes": checkpoint_tuple.pending_writes},
    #     {"user_input": raw_user_input, "input_context": raw_input_context,
    #      "resume_feedback": raw_feedback},
    # )
