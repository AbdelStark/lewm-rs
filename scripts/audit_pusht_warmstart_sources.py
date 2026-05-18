#!/usr/bin/env python3
"""Audit public PushT Hub ``.mpk`` files for SO-100 warm-start compatibility."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "abdelstark/lewm-rs-pusht"
DEFAULT_REVISION = "main"
DEFAULT_REPORT = Path("reports/pusht_warmstart_hub_audit.json")
DEFAULT_CONFIG = Path("configs/pusht.toml")
SCHEMA_VERSION = "1.0.0"
USER_AGENT = "lewm-rs-warmstart-source-audit/1.0"


class HubAuditError(RuntimeError):
    """Raised when the Hub warm-start source audit cannot continue."""


def load_warmstart_checker() -> Any:
    """Load ``scripts/check_warmstart_source.py`` without packaging the scripts dir."""
    checker_path = ROOT / "scripts" / "check_warmstart_source.py"
    spec = importlib.util.spec_from_file_location("check_warmstart_source", checker_path)
    if spec is None or spec.loader is None:
        raise HubAuditError(f"could not load {checker_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Hub model repo ({DEFAULT_REPO})")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help=f"Hub revision ({DEFAULT_REVISION})")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"PushT config for bounded-core parameter count ({DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"JSON report path ({DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Optional directory to keep downloaded candidates for inspection.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--require-compatible",
        action="store_true",
        help="exit non-zero when no compatible warm-start source is found",
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


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def mpk_candidates(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise HubAuditError("Hub tree response must be a list")

    candidates: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith(".mpk"):
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


def write_candidate(repo: str, revision: str, candidate: dict[str, Any], root: Path, timeout: float) -> Path:
    path = str(candidate["path"])
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(fetch_bytes(resolve_url(repo, revision, path), timeout))
    return destination


def expected_param_count(checker: Any, config_path: Path) -> int:
    return int(checker.expected_bounded_param_count(checker.load_config(config_path)))


def observed_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return compact record metadata useful for diagnosing rejected sources."""
    params = payload.get("params")
    adamw_params = payload.get("adamw_params")
    return {
        "format": "json",
        "schema_version": payload.get("schema_version"),
        "kind": payload.get("kind"),
        "step": payload.get("step"),
        "param_count": len(params) if isinstance(params, list) else None,
        "adamw_param_count": len(adamw_params) if isinstance(adamw_params, list) else None,
    }


def record_violations(checker: Any, payload: dict[str, Any], expected_params: int) -> list[str]:
    """Collect all warm-start source contract failures instead of the first one."""
    violations: list[str] = []
    schema_version = payload.get("schema_version")
    if schema_version != checker.EXPECTED_SCHEMA_VERSION:
        violations.append(
            f"schema_version must be {checker.EXPECTED_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    kind = payload.get("kind")
    if kind != checker.EXPECTED_KIND:
        violations.append(f"kind must be {checker.EXPECTED_KIND!r}, got {kind!r}")

    step = payload.get("step")
    if not isinstance(step, int) or step <= 0:
        violations.append("step must be a positive integer")

    params = payload.get("params")
    if not isinstance(params, list):
        violations.append("params must be a list")
    elif len(params) != expected_params:
        violations.append(
            f"params length {len(params)} does not match expected bounded-core "
            f"parameter count {expected_params}"
        )
    elif any(not isinstance(value, int | float) or not math.isfinite(float(value)) for value in params):
        violations.append("params must contain only finite numbers")

    adamw_params = payload.get("adamw_params", [])
    if not isinstance(adamw_params, list):
        violations.append("adamw_params must be a list when present")
    elif adamw_params and len(adamw_params) != expected_params:
        violations.append(
            f"adamw_params length {len(adamw_params)} does not match expected bounded-core "
            f"parameter count {expected_params}"
        )

    return violations


def rejected_result(local_path: Path, observed: dict[str, Any], violations: list[str]) -> dict[str, Any]:
    return {
        "status": "rejected",
        "reason": f"{local_path}: {'; '.join(violations)}",
        "observed": observed,
        "violations": violations,
    }


def validate_candidate(checker: Any, local_path: Path, expected_params: int) -> dict[str, Any]:
    try:
        payload = checker.load_json(local_path)
    except checker.WarmstartSourceError as exc:
        reason = str(exc)
        return {
            "status": "rejected",
            "reason": reason,
            "observed": {
                "format": "unsupported",
                "error": reason.replace(f"{local_path}: ", ""),
            },
            "violations": [reason],
        }

    observed = observed_record(payload)
    violations = record_violations(checker, payload, expected_params)
    if violations:
        return rejected_result(local_path, observed, violations)

    try:
        checker.validate_record(local_path, payload, expected_params)
    except checker.WarmstartSourceError as exc:
        return rejected_result(local_path, observed, [str(exc).replace(f"{local_path}: ", "")])
    return {
        "status": "compatible",
        "reason": "accepted by scripts/check_warmstart_source.py",
        "observed": observed,
        "violations": [],
    }


def audit_candidates(
    *,
    repo: str,
    revision: str,
    candidates: list[dict[str, Any]],
    download_root: Path,
    timeout: float,
    checker: Any,
    expected_params: int,
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for candidate in candidates:
        path = str(candidate["path"])
        local_path = write_candidate(repo, revision, candidate, download_root, timeout)
        result = validate_candidate(checker, local_path, expected_params)
        if "reason" in result:
            result["reason"] = result["reason"].replace(str(local_path), path)
        violations = result.get("violations")
        if isinstance(violations, list):
            result["violations"] = [
                violation.replace(str(local_path), path) if isinstance(violation, str) else violation
                for violation in violations
            ]
        observed = result.get("observed")
        if isinstance(observed, dict) and isinstance(observed.get("error"), str):
            observed["error"] = observed["error"].replace(str(local_path), path)
        audited.append(
            {
                "path": path,
                "size": candidate["size"],
                "download_url": resolve_url(repo, revision, path),
                **result,
            }
        )
    return audited


def build_report(
    *,
    repo: str,
    revision: str,
    candidates: list[dict[str, Any]],
    expected_params: int,
) -> dict[str, Any]:
    compatible = [candidate for candidate in candidates if candidate["status"] == "compatible"]
    return {
        "schema_version": SCHEMA_VERSION,
        "updated": datetime.now(UTC).date().isoformat(),
        "repo": repo,
        "revision": revision,
        "source": tree_url(repo, revision),
        "source_verifier": "scripts/check_warmstart_source.py",
        "expected": {
            "schema_version": "1.1.0",
            "kind": "lewm-rs-pusht-bounded-module-lewm-record",
            "param_count": expected_params,
        },
        "candidate_count": len(candidates),
        "compatible_count": len(compatible),
        "status": "ready" if compatible else "blocked",
        "summary": (
            "Compatible PushT warm-start source found."
            if compatible
            else "No public PushT .mpk currently satisfies the SO-100 warm-start source contract."
        ),
        "candidates": candidates,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_with_download_root(args: argparse.Namespace, download_root: Path) -> dict[str, Any]:
    checker = load_warmstart_checker()
    config_path = resolve_path(args.config)
    expected_params = expected_param_count(checker, config_path)
    payload = fetch_json(tree_url(args.repo, args.revision), args.timeout)
    candidates = mpk_candidates(payload)
    audited = audit_candidates(
        repo=args.repo,
        revision=args.revision,
        candidates=candidates,
        download_root=download_root,
        timeout=args.timeout,
        checker=checker,
        expected_params=expected_params,
    )
    return build_report(
        repo=args.repo,
        revision=args.revision,
        candidates=audited,
        expected_params=expected_params,
    )


def main() -> int:
    args = parse_args()
    report_path = resolve_path(args.report)

    try:
        if args.download_dir is None:
            with tempfile.TemporaryDirectory(prefix="lewm-pusht-warmstart-hub-audit-") as tmp:
                report = run_with_download_root(args, Path(tmp))
        else:
            download_root = resolve_path(args.download_dir)
            download_root.mkdir(parents=True, exist_ok=True)
            report = run_with_download_root(args, download_root)
        write_report(report_path, report)
    except (HubAuditError, OSError, ValueError) as exc:
        print(f"audit_pusht_warmstart_sources.py: {exc}", file=sys.stderr)
        return 1

    print(
        "PushT warm-start Hub audit: "
        f"candidates={report['candidate_count']} compatible={report['compatible_count']} "
        f"status={report['status']} report={report_path.relative_to(ROOT)}"
    )
    if args.require_compatible and report["compatible_count"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
