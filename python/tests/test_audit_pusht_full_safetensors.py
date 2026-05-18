from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_pusht_full_safetensors.py"

spec = importlib.util.spec_from_file_location("audit_pusht_full_safetensors", SCRIPT)
assert spec is not None
assert spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def safetensors_prefix(keys: list[str]) -> bytes:
    header = {
        key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        for key in keys
    }
    raw = json.dumps(header).encode("utf-8")
    return len(raw).to_bytes(8, byteorder="little", signed=False) + raw


def test_parse_safetensors_header_counts_tensors() -> None:
    header = audit.parse_safetensors_header(safetensors_prefix(["b", "a"]))

    observed = audit.observed_from_header(header)

    assert observed["format"] == "safetensors"
    assert observed["tensor_count"] == 2
    assert observed["first_tensors"] == ["a", "b"]


def test_legacy_bounded_candidate_is_rejected_with_family_hint() -> None:
    header = audit.parse_safetensors_header(
        safetensors_prefix(sorted(audit.BOUNDED_CORE_KEY_HINTS))
    )
    observed = audit.observed_from_header(header)

    violations = audit.candidate_violations(
        "train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors",
        observed,
    )

    assert observed["inferred_family"] == "bounded_pusht_host_core"
    assert any("legacy bounded prefix" in violation for violation in violations)
    assert any("tensor count 14" in violation for violation in violations)


def test_build_report_marks_ready_candidate_for_contract_check() -> None:
    candidate = {
        "path": "train/pusht-full-burn-jepa-20260518T120000Z/step_0050000.safetensors",
        "size": 123456,
        "download_url": "https://example.invalid/model.safetensors",
        "status": "ready_for_contract_check",
        "reason": "header matches",
        "observed": {
            "format": "safetensors",
            "tensor_count": audit.EXPECTED_DESTINATION_TENSOR_COUNT,
            "first_tensors": ["encoder.blocks.0.attn.proj.bias"],
            "dtypes": ["F32"],
            "metadata_keys": [],
            "inferred_family": "unknown",
        },
        "violations": [],
    }

    report = audit.build_report(
        repo="abdelstark/lewm-rs-pusht",
        revision="main",
        candidates=[candidate],
    )

    assert report["ready_count"] == 1
    assert report["status"] == "ready_for_contract_check"
