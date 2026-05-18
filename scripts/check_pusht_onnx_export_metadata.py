#!/usr/bin/env python3
"""Validate F1 PushT ONNX export metadata before Hub upload."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_EXPORT_DIR = Path("/tmp/lewm-f1-pusht-onnx/onnx-full")
DEFAULT_STEP = 50_000
DEFAULT_ACTION_DIM = 10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
REQUIRED_SOURCE_FRAGMENT = "train/pusht-full-burn-jepa-"
LEGACY_SOURCE_FRAGMENT = "train/pusht-full-lewm-"
EXPECTED_VARIANTS: dict[str, dict[str, Any]] = {
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
}


class MetadataError(RuntimeError):
    """Raised when the F1 ONNX export metadata is incomplete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help=f"F1 onnx-full export directory ({DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--expected-step",
        type=int,
        default=DEFAULT_STEP,
        help=f"expected PushT checkpoint step ({DEFAULT_STEP})",
    )
    parser.add_argument(
        "--expected-action-dim",
        type=int,
        default=DEFAULT_ACTION_DIM,
        help=f"expected packed PushT action dimension ({DEFAULT_ACTION_DIM})",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MetadataError(f"missing metadata file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MetadataError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MetadataError(f"{path}: metadata root must be a JSON object")
    return payload


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise MetadataError(f"{label} must be {expected!r}, got {actual!r}")


def require_source_path(metadata: dict[str, Any], expected_step: int) -> None:
    source = metadata.get("safetensors_source")
    if not isinstance(source, str) or not source:
        raise MetadataError("safetensors_source must be a non-empty string")
    if LEGACY_SOURCE_FRAGMENT in source:
        raise MetadataError(
            "safetensors_source points at the legacy bounded PushT artifact family"
        )
    if REQUIRED_SOURCE_FRAGMENT not in source:
        raise MetadataError(
            f"safetensors_source must include {REQUIRED_SOURCE_FRAGMENT!r}"
        )
    expected_name = f"step_{expected_step:07d}.safetensors"
    if not source.endswith(expected_name):
        raise MetadataError(f"safetensors_source must end with {expected_name!r}")


def require_common_metadata(metadata: dict[str, Any], *, expected_step: int, action_dim: int) -> None:
    require_equal(metadata.get("schema_version"), "1.0.0", "schema_version")
    require_equal(metadata.get("source"), "burn_safetensors", "source")
    require_equal(metadata.get("step_count"), expected_step, "step_count")
    require_source_path(metadata, expected_step)

    sha256 = metadata.get("safetensors_sha256")
    if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
        raise MetadataError("safetensors_sha256 must be a lowercase 64-character hex digest")

    timestamp = metadata.get("export_timestamp")
    if not isinstance(timestamp, str) or TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise MetadataError("export_timestamp must be an RFC 3339 UTC second timestamp")

    config = metadata.get("config")
    if not isinstance(config, dict):
        raise MetadataError("config must be an object")
    require_equal(config.get("image_size"), 224, "config.image_size")
    require_equal(config.get("history_size"), 3, "config.history_size")
    require_equal(config.get("latent_dim"), 192, "config.latent_dim")
    require_equal(config.get("action_dim"), action_dim, "config.action_dim")

    variants = metadata.get("variants")
    if not isinstance(variants, dict):
        raise MetadataError("variants must be an object")
    require_equal(variants, EXPECTED_VARIANTS, "variants")


def validate_variant_sidecars(root: Path, root_metadata: dict[str, Any]) -> None:
    for variant, info in EXPECTED_VARIANTS.items():
        variant_dir = root / variant
        if not variant_dir.is_dir():
            raise MetadataError(f"missing variant directory: {variant_dir}")
        for artifact_key in ("encoder", "predictor"):
            artifact = root / info[artifact_key]
            if not artifact.exists():
                raise MetadataError(f"missing {variant} {artifact_key} artifact: {artifact}")
        sidecar = load_json(variant_dir / "onnx_export.json")
        if sidecar != root_metadata:
            raise MetadataError(f"{variant_dir / 'onnx_export.json'} must match root metadata")


def validate(root: Path, *, expected_step: int, action_dim: int) -> dict[str, Any]:
    metadata = load_json(root / "onnx_export.json")
    require_common_metadata(metadata, expected_step=expected_step, action_dim=action_dim)
    validate_variant_sidecars(root, metadata)
    return metadata


def main() -> int:
    args = parse_args()
    try:
        metadata = validate(
            args.dir,
            expected_step=args.expected_step,
            action_dim=args.expected_action_dim,
        )
    except MetadataError as exc:
        print(f"check_pusht_onnx_export_metadata.py: {exc}", file=sys.stderr)
        return 1

    print(
        "PushT ONNX export metadata ok: "
        f"dir={args.dir} step={metadata['step_count']} "
        f"action_dim={metadata['config']['action_dim']} "
        f"variants={','.join(sorted(metadata['variants']))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
