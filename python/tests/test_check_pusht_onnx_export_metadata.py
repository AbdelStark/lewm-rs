from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_pusht_onnx_export_metadata.py"


def metadata(source: str = "/tmp/hub/train/pusht-full-burn-jepa-20260518T120000Z/step_0050000.safetensors"):
    return {
        "schema_version": "1.0.0",
        "source": "burn_safetensors",
        "safetensors_source": source,
        "safetensors_sha256": "a" * 64,
        "step_count": 50_000,
        "export_timestamp": "2026-05-18T12:00:00Z",
        "config": {
            "image_size": 224,
            "history_size": 3,
            "latent_dim": 192,
            "action_dim": 10,
        },
        "variants": {
            "onnxruntime": {
                "opset_version": 18,
                "dynamic_batch": True,
                "encoder": "onnxruntime/encoder.onnx",
                "predictor": "onnxruntime/predictor.onnx",
            },
            "tract-compat": {
                "opset_version": 17,
                "dynamic_batch": False,
                "encoder": "tract-compat/encoder.onnx",
                "predictor": "tract-compat/predictor.onnx",
            },
        },
    }


def write_export_tree(root: Path, payload: dict | None = None) -> None:
    payload = payload or metadata()
    root.mkdir(parents=True)
    (root / "onnx_export.json").write_text(json.dumps(payload), encoding="utf-8")
    for variant in ("onnxruntime", "tract-compat"):
        variant_dir = root / variant
        variant_dir.mkdir()
        (variant_dir / "encoder.onnx").write_bytes(b"encoder")
        (variant_dir / "predictor.onnx").write_bytes(b"predictor")
        (variant_dir / "onnx_export.json").write_text(json.dumps(payload), encoding="utf-8")


def run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_accepts_complete_f1_onnx_export_metadata(tmp_path: Path) -> None:
    root = tmp_path / "onnx-full"
    write_export_tree(root)

    result = run_check(root)

    assert result.returncode == 0
    assert "PushT ONNX export metadata ok" in result.stdout
    assert "step=50000" in result.stdout
    assert "action_dim=10" in result.stdout


def test_rejects_legacy_bounded_source_path(tmp_path: Path) -> None:
    root = tmp_path / "onnx-full"
    write_export_tree(
        root,
        metadata("/tmp/hub/train/pusht-full-lewm-20260515T100908Z/step_0050000.safetensors"),
    )

    result = run_check(root)

    assert result.returncode == 1
    assert "legacy bounded PushT artifact family" in result.stderr


def test_rejects_missing_variant_artifact(tmp_path: Path) -> None:
    root = tmp_path / "onnx-full"
    write_export_tree(root)
    (root / "tract-compat" / "predictor.onnx").unlink()

    result = run_check(root)

    assert result.returncode == 1
    assert "missing tract-compat predictor artifact" in result.stderr


def test_rejects_variant_sidecar_drift(tmp_path: Path) -> None:
    root = tmp_path / "onnx-full"
    write_export_tree(root)
    drifted = metadata()
    drifted["step_count"] = 10
    (root / "onnxruntime" / "onnx_export.json").write_text(
        json.dumps(drifted),
        encoding="utf-8",
    )

    result = run_check(root)

    assert result.returncode == 1
    assert "onnxruntime/onnx_export.json must match root metadata" in result.stderr
