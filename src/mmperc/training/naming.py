"""Run-folder naming, shared by scripts/train.py and scripts/run_protocol.py (the protocol relies on it to find its runs)."""
from __future__ import annotations


def run_name(kind: str, mode: str = "learned", gate_inputs: bool = False, mixed: bool = False, tag: str = "") -> str:
    return (f"fusion_{mode}" if kind == "fusion" else kind) + ("_gated" if gate_inputs else "") + ("_mixed" if mixed else "") + tag
