#!/usr/bin/env python3
"""Inspect and convert the locked PushT reference checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import param_name_map as pnm


DEFAULT_LOCAL_DIR = Path("/tmp/lewm-rs-reference-model")
DEFAULT_AUDIT = Path("reports/parity/reference-key-audit.json")
STATE_DICT_CANDIDATES = ("state_dict", "model", "model_state_dict", "module")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    audit = subcommands.add_parser(
        "audit",
        help="Validate the reference checkpoint state_dict key set.",
    )
    audit.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Directory containing config.json and weights.pt. Default: {DEFAULT_LOCAL_DIR}",
    )
    audit.add_argument(
        "--download",
        action="store_true",
        help="Use the hf CLI to download the locked config and weights before auditing.",
    )
    audit.add_argument(
        "--audit-out",
        type=Path,
        default=DEFAULT_AUDIT,
        help=f"JSON audit output path. Default: {DEFAULT_AUDIT}",
    )
    audit.add_argument(
        "--include-keys",
        action="store_true",
        help="Include the sorted source key list in the JSON audit.",
    )
    audit.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the 72 MB weights.pt SHA-256 check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "audit":
        return audit_command(args)
    raise AssertionError(f"unhandled command: {args.command}")


def audit_command(args: argparse.Namespace) -> int:
    meta = load_reference_meta()
    source = meta["source_model"]
    local_dir = args.local_dir
    weights_path = local_dir / source["weights_file"]
    config_path = local_dir / source["config_file"]

    if args.download:
        download_reference_files(
            repo_id=source["repo_id"],
            revision=source["revision"],
            local_dir=local_dir,
            filenames=(source["config_file"], source["weights_file"]),
        )
    require_file(config_path, "--download or provide the locked config.json")
    require_file(weights_path, "--download or provide the locked weights.pt")

    weights_sha256 = None
    if not args.skip_sha256:
        weights_sha256 = sha256_file(weights_path)
        if weights_sha256 != source["weights_sha256"]:
            raise SystemExit(
                f"weights sha256 mismatch: got {weights_sha256}, expected {source['weights_sha256']}"
            )

    state_dict = load_torch_state_dict(weights_path)
    audit = audit_state_dict_keys(
        state_dict,
        meta=meta,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        include_keys=args.include_keys,
    )
    write_json(args.audit_out, audit)
    validation = audit["key_validation"]
    print(
        "reference key audit: "
        f"ok={validation['ok']} "
        f"actual={validation['actual_source_tensor_count']} "
        f"expected={validation['expected_source_tensor_count']} "
        f"out={args.audit_out}"
    )
    return 0 if validation["ok"] else 1


def load_reference_meta() -> Mapping[str, Any]:
    return json.loads(pnm.REFERENCE_META_PATH.read_text(encoding="utf-8"))


def download_reference_files(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
    filenames: tuple[str, ...],
) -> None:
    if shutil.which("hf") is None:
        raise SystemExit("hf CLI is required for --download")
    command = [
        "hf",
        "download",
        repo_id,
        *filenames,
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
    ]
    subprocess.run(command, check=True)


def load_torch_state_dict(weights_path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to inspect weights.pt; install torch or run this command "
            "in an environment that already has it."
        ) from exc

    checkpoint = torch.load(weights_path, map_location="cpu")
    return extract_state_dict(checkpoint)


def extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"expected checkpoint mapping, got {type(checkpoint).__name__}")
    if looks_like_state_dict(checkpoint):
        return checkpoint
    for key in STATE_DICT_CANDIDATES:
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and looks_like_state_dict(value):
            return value
    available = ", ".join(str(key) for key in sorted(checkpoint.keys()))
    raise ValueError(f"could not locate tensor state_dict; top-level keys: {available}")


def looks_like_state_dict(value: Mapping[str, Any]) -> bool:
    if not value:
        return False
    return all(is_tensor_like(tensor) for tensor in value.values())


def is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") or hasattr(value, "size")


def audit_state_dict_keys(
    state_dict: Mapping[str, Any],
    *,
    meta: Mapping[str, Any] | None = None,
    weights_path: Path | None = None,
    weights_sha256: str | None = None,
    include_keys: bool = False,
) -> dict[str, Any]:
    reference_meta = load_reference_meta() if meta is None else meta
    source = reference_meta["source_model"]
    keys = tuple(sorted(state_dict.keys()))
    validation = pnm.validate_state_dict_keys(keys)
    audit: dict[str, Any] = {
        "schema_version": "1.0",
        "source_model": {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "weights_file": source["weights_file"],
            "weights_sha256": weights_sha256,
            "weights_path": str(weights_path) if weights_path is not None else None,
        },
        "key_validation": {
            "ok": validation.ok,
            "actual_source_tensor_count": len(keys),
            "expected_source_tensor_count": len(pnm.expected_source_keys()),
            "expected_destination_tensor_count": len(pnm.expected_destination_keys()),
            "missing": list(validation.missing),
            "extra": list(validation.extra),
        },
    }
    if include_keys:
        audit["source_keys"] = list(keys)
    return audit


def require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}; {hint}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
