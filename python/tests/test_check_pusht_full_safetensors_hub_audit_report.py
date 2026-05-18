from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_pusht_full_safetensors_hub_audit_report.py"


def rejected_candidate() -> dict[str, object]:
    return {
        "path": "train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors",
        "size": 1264,
        "download_url": "https://huggingface.co/example/resolve/main/checkpoint.safetensors",
        "status": "rejected",
        "reason": "train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors: wrong prefix",
        "observed": {
            "format": "safetensors",
            "tensor_count": 14,
            "first_tensors": ["action_encoder.bias"],
            "dtypes": ["F32"],
            "metadata_keys": [],
            "inferred_family": "bounded_pusht_host_core",
        },
        "violations": ["wrong prefix"],
    }


def ready_candidate() -> dict[str, object]:
    return {
        "path": "train/pusht-full-burn-jepa-20260518T120000Z/step_0050000.safetensors",
        "size": 123456,
        "download_url": "https://huggingface.co/example/resolve/main/checkpoint.safetensors",
        "status": "ready_for_contract_check",
        "reason": "header matches",
        "observed": {
            "format": "safetensors",
            "tensor_count": 255,
            "first_tensors": ["encoder.blocks.0.attn.proj.bias"],
            "dtypes": ["F32"],
            "metadata_keys": [],
            "inferred_family": "unknown",
        },
        "violations": [],
    }


def payload(*, ready: bool = False) -> dict[str, object]:
    candidates = [ready_candidate() if ready else rejected_candidate()]
    return {
        "schema_version": "1.0.0",
        "updated": "2026-05-18",
        "repo": "abdelstark/lewm-rs-pusht",
        "revision": "main",
        "source": "https://huggingface.co/api/models/abdelstark/lewm-rs-pusht/tree/main?recursive=1",
        "source_verifier": "scripts/f1_export_pusht_onnx.py --execute",
        "expected": {
            "required_path_pattern": (
                "train/pusht-full-burn-jepa-YYYYMMDDTHHMMSSZ/step_0050000.safetensors"
            ),
            "destination_tensor_count": 255,
            "source_key_count_after_contract_check": 303,
        },
        "candidate_count": len(candidates),
        "ready_count": 1 if ready else 0,
        "status": "ready_for_contract_check" if ready else "blocked",
        "summary": "audit fixture",
        "candidates": candidates,
    }


def run_check(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_blocked_report_passes(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload()), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 0
    assert "PushT full safetensors Hub audit report ok" in result.stdout


def test_valid_ready_report_passes(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload(ready=True)), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 0
    assert "ready=1" in result.stdout


def test_rejects_ready_candidate_with_wrong_tensor_count(tmp_path: Path) -> None:
    body = payload(ready=True)
    candidates = body["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    observed = candidate["observed"]
    assert isinstance(observed, dict)
    observed["tensor_count"] = 14
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 1
    assert "observed.tensor_count must be 255" in result.stderr
