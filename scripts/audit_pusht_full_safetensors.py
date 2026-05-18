#!/usr/bin/env python3
"""Audit public PushT Hub safetensors files for F1 full-checkpoint candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "abdelstark/lewm-rs-pusht"
DEFAULT_REVISION = "main"
DEFAULT_REPORT = Path("reports/pusht_full_safetensors_hub_audit.json")
SCHEMA_VERSION = "1.0.0"
USER_AGENT = "lewm-rs-pusht-full-safetensors-audit/1.0"
REQUIRED_PREFIX = "train/pusht-full-burn-jepa-"
LEGACY_BOUNDED_PREFIX = "train/pusht-full-lewm-"
REQUIRED_PATH_RE = re.compile(
    r"^train/pusht-full-burn-jepa-\d{8}T\d{6}Z/step_0050000\.safetensors$"
)
EXPECTED_DESTINATION_TENSOR_COUNT = 255
EXPECTED_SOURCE_KEY_COUNT = 303
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
BOUNDED_CORE_KEY_HINTS = {
    "action_encoder.bias",
    "action_encoder.x.weight",
    "action_encoder.y.weight",
    "encoder.bias",
    "encoder.energy.weight",
    "encoder.pixel.weight",
    "encoder.time.weight",
    "pred_proj.bias",
    "pred_proj.weight",
    "predictor.action.weight",
    "predictor.bias",
    "predictor.latent.weight",
    "projector.bias",
    "projector.weight",
}


class HubAuditError(RuntimeError):
    """Raised when the Hub safetensors audit cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Hub model repo ({DEFAULT_REPO})")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help=f"Hub revision ({DEFAULT_REVISION})")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"JSON report path ({DEFAULT_REPORT})",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit non-zero when no ready full-checkpoint candidate is found",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def tree_url(repo: str, revision: str) -> str:
    quoted_revision = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{repo}/tree/{quoted_revision}?recursive=1"


def resolve_url(repo: str, revision: str, path: str) -> str:
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{quoted_revision}/{quoted_path}"


def fetch_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_prefix(url: str, length: int, timeout: float) -> bytes:
    if length <= 0:
        raise HubAuditError("prefix length must be positive")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Range": f"bytes=0-{length - 1}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(length)
    if len(data) < length:
        raise HubAuditError(f"safetensors header response too short: got {len(data)} bytes")
    return data


def safetensors_candidates(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise HubAuditError("Hub tree response must be a list")

    candidates: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith(".safetensors"):
            continue
        kind = item.get("type")
        if kind is not None and kind != "file":
            continue
        size = item.get("size")
        candidates.append(
            {
                "path": path,
                "size": size if isinstance(size, int) else None,
            }
        )
    return sorted(candidates, key=lambda candidate: candidate["path"])


def parse_safetensors_header(prefix: bytes) -> dict[str, Any]:
    if len(prefix) < 8:
        raise HubAuditError("safetensors header is missing the 8-byte length prefix")
    header_len = int.from_bytes(prefix[:8], byteorder="little", signed=False)
    if header_len <= 0:
        raise HubAuditError("safetensors header length must be positive")
    if header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise HubAuditError(
            f"safetensors header length {header_len} exceeds "
            f"{MAX_SAFETENSORS_HEADER_BYTES} bytes"
        )
    total_len = 8 + header_len
    if len(prefix) < total_len:
        raise HubAuditError("safetensors header bytes are incomplete")
    try:
        header = json.loads(prefix[8:total_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HubAuditError(f"safetensors header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise HubAuditError("safetensors header must be a JSON object")
    return header


def fetch_safetensors_header(url: str, timeout: float) -> dict[str, Any]:
    first = fetch_prefix(url, 8, timeout)
    header_len = int.from_bytes(first, byteorder="little", signed=False)
    if header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise HubAuditError(
            f"safetensors header length {header_len} exceeds "
            f"{MAX_SAFETENSORS_HEADER_BYTES} bytes"
        )
    return parse_safetensors_header(fetch_prefix(url, 8 + header_len, timeout))


def observed_from_header(header: dict[str, Any]) -> dict[str, Any]:
    tensor_names = sorted(key for key in header if key != "__metadata__")
    dtypes = sorted(
        {
            value.get("dtype")
            for key, value in header.items()
            if key != "__metadata__" and isinstance(value, dict) and isinstance(value.get("dtype"), str)
        }
    )
    metadata = header.get("__metadata__", {})
    return {
        "format": "safetensors",
        "tensor_count": len(tensor_names),
        "first_tensors": tensor_names[:10],
        "dtypes": dtypes,
        "metadata_keys": sorted(metadata) if isinstance(metadata, dict) else [],
        "inferred_family": (
            "bounded_pusht_host_core" if BOUNDED_CORE_KEY_HINTS.issubset(tensor_names) else "unknown"
        ),
    }


def unsupported_observed(error: Exception) -> dict[str, Any]:
    return {
        "format": "unsupported",
        "error": str(error),
    }


def candidate_violations(path: str, observed: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if path.startswith(LEGACY_BOUNDED_PREFIX):
        violations.append(
            f"path uses legacy bounded prefix {LEGACY_BOUNDED_PREFIX!r}; "
            f"F1 requires {REQUIRED_PREFIX!r}"
        )
    elif not path.startswith(REQUIRED_PREFIX):
        violations.append(f"path must start with {REQUIRED_PREFIX!r}")
    elif REQUIRED_PATH_RE.fullmatch(path) is None:
        violations.append(
            "path must match "
            "'train/pusht-full-burn-jepa-YYYYMMDDTHHMMSSZ/step_0050000.safetensors'"
        )

    if observed.get("format") != "safetensors":
        violations.append("file must have a readable safetensors header")
        return violations

    tensor_count = observed.get("tensor_count")
    if tensor_count != EXPECTED_DESTINATION_TENSOR_COUNT:
        violations.append(
            f"tensor count {tensor_count} does not match expected full Burn/Jepa "
            f"destination tensor count {EXPECTED_DESTINATION_TENSOR_COUNT}"
        )
    return violations


def audit_candidate(
    *,
    repo: str,
    revision: str,
    candidate: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    path = str(candidate["path"])
    download_url = resolve_url(repo, revision, path)
    try:
        observed = observed_from_header(fetch_safetensors_header(download_url, timeout))
    except (HubAuditError, OSError, ValueError) as exc:
        observed = unsupported_observed(exc)
    violations = candidate_violations(path, observed)
    if violations:
        status = "rejected"
        reason = f"{path}: {'; '.join(violations)}"
    else:
        status = "ready_for_contract_check"
        reason = (
            "header matches the F1 full Burn/Jepa path and tensor-count preflight; "
            "run scripts/f1_export_pusht_onnx.py before any upload"
        )
    return {
        "path": path,
        "size": candidate["size"],
        "download_url": download_url,
        "status": status,
        "reason": reason,
        "observed": observed,
        "violations": violations,
    }


def build_report(
    *,
    repo: str,
    revision: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    ready = [candidate for candidate in candidates if candidate["status"] == "ready_for_contract_check"]
    return {
        "schema_version": SCHEMA_VERSION,
        "updated": datetime.now(UTC).date().isoformat(),
        "repo": repo,
        "revision": revision,
        "source": tree_url(repo, revision),
        "source_verifier": "scripts/f1_export_pusht_onnx.py --execute",
        "expected": {
            "required_path_pattern": (
                "train/pusht-full-burn-jepa-YYYYMMDDTHHMMSSZ/step_0050000.safetensors"
            ),
            "destination_tensor_count": EXPECTED_DESTINATION_TENSOR_COUNT,
            "source_key_count_after_contract_check": EXPECTED_SOURCE_KEY_COUNT,
        },
        "candidate_count": len(candidates),
        "ready_count": len(ready),
        "status": "ready_for_contract_check" if ready else "blocked",
        "summary": (
            "At least one PushT full Burn/Jepa safetensors candidate is ready for the F1 "
            "contract-check/export wrapper."
            if ready
            else "No public PushT safetensors file currently satisfies the F1 full-run "
            "source preflight."
        ),
        "candidates": candidates,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_path = resolve_path(args.report)

    try:
        payload = fetch_json(tree_url(args.repo, args.revision), args.timeout)
        candidates = [
            audit_candidate(
                repo=args.repo,
                revision=args.revision,
                candidate=candidate,
                timeout=args.timeout,
            )
            for candidate in safetensors_candidates(payload)
        ]
        report = build_report(repo=args.repo, revision=args.revision, candidates=candidates)
        write_report(report_path, report)
    except (HubAuditError, OSError, ValueError) as exc:
        print(f"audit_pusht_full_safetensors.py: {exc}", file=sys.stderr)
        return 1

    print(
        "PushT full safetensors Hub audit: "
        f"candidates={report['candidate_count']} ready={report['ready_count']} "
        f"status={report['status']} report={report_path.relative_to(ROOT)}"
    )
    if args.require_ready and report["ready_count"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
