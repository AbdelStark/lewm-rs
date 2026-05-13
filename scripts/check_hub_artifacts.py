#!/usr/bin/env python3
"""Verify published Hub artifacts against an expected SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("conformance/hub_artifacts.json")
HF_HOST = "huggingface.co"
VALID_REPO_TYPES = {"model", "dataset", "space"}


class ManifestError(Exception):
    """Raised when the artifact manifest is malformed."""


@dataclass(frozen=True)
class HubArtifact:
    name: str
    source: str
    expected_sha256: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Hugging Face Hub artifacts listed in a JSON manifest. "
            "When the default manifest is absent, the check reports an explicit skip."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"artifact manifest path, relative to the repo root by default ({DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="fail when the manifest file is missing",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="download timeout per artifact in seconds",
    )
    return parser.parse_args()


def manifest_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestError(f"{path}: manifest root must be a JSON object")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError(f"{path}: manifest must contain a non-empty 'artifacts' list")
    return data


def require_str(entry: dict[str, Any], key: str, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"artifacts[{index}].{key} must be a non-empty string")
    return value


def validate_sha256(value: str, index: int) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ManifestError(f"artifacts[{index}].sha256 must be a 64-character lowercase hex digest")
    return normalized


def hub_source(entry: dict[str, Any], index: int) -> str:
    explicit_url = entry.get("url")
    if isinstance(explicit_url, str) and explicit_url:
        return explicit_url

    repo = require_str(entry, "repo", index)
    artifact_path = require_str(entry, "path", index)
    revision = entry.get("revision", "main")
    repo_type = entry.get("repo_type", "model")

    if not isinstance(revision, str) or not revision:
        raise ManifestError(f"artifacts[{index}].revision must be a non-empty string")
    if not isinstance(repo_type, str) or repo_type not in VALID_REPO_TYPES:
        raise ManifestError(
            f"artifacts[{index}].repo_type must be one of {sorted(VALID_REPO_TYPES)}"
        )

    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = urllib.parse.quote(artifact_path.lstrip("/"), safe="/")
    if repo_type == "model":
        return f"https://{HF_HOST}/{repo}/resolve/{quoted_revision}/{quoted_path}"
    return f"https://{HF_HOST}/{repo_type}s/{repo}/resolve/{quoted_revision}/{quoted_path}"


def parse_artifacts(data: dict[str, Any]) -> list[HubArtifact]:
    artifacts: list[HubArtifact] = []
    for index, raw_entry in enumerate(data["artifacts"]):
        if not isinstance(raw_entry, dict):
            raise ManifestError(f"artifacts[{index}] must be a JSON object")

        expected_sha256 = validate_sha256(require_str(raw_entry, "sha256", index), index)
        source = hub_source(raw_entry, index)
        name = raw_entry.get("name")
        if not isinstance(name, str) or not name:
            name = source
        artifacts.append(HubArtifact(name=name, source=source, expected_sha256=expected_sha256))
    return artifacts


def local_path_from_source(source: str) -> Path | None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme == "file":
        return Path(urllib.request.url2pathname(parsed.path))
    if parsed.scheme == "":
        path = Path(source)
        if path.is_absolute():
            return path
        return repo_root() / path
    return None


def read_artifact_bytes(source: str, timeout: float) -> bytes:
    local_path = local_path_from_source(source)
    if local_path is not None:
        return local_path.read_bytes()

    request = urllib.request.Request(
        source,
        headers={"User-Agent": "lewm-rs-conformance/1.0"},
    )
    parsed = urllib.parse.urlparse(source)
    token = os.environ.get("HF_TOKEN")
    if token and parsed.scheme == "https" and parsed.netloc == HF_HOST:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def verify_artifact(artifact: HubArtifact, timeout: float) -> str | None:
    try:
        payload = read_artifact_bytes(artifact.source, timeout)
    except (OSError, urllib.error.URLError) as exc:
        return f"{artifact.name}: could not read {artifact.source}: {exc}"

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != artifact.expected_sha256:
        return (
            f"{artifact.name}: sha256 mismatch for {artifact.source}: "
            f"expected {artifact.expected_sha256}, got {actual_sha256}"
        )

    print(f"ok: {artifact.name} sha256={actual_sha256}")
    return None


def main() -> int:
    args = parse_args()
    path = manifest_path(args.manifest)

    if not path.exists():
        message = f"hub artifact check skipped: expected manifest not present at {path}"
        if args.require_manifest:
            print(message, file=sys.stderr)
            return 1
        print(message)
        return 0

    try:
        artifacts = parse_artifacts(load_manifest(path))
    except ManifestError as exc:
        print(f"Hub artifact manifest error: {exc}", file=sys.stderr)
        return 1

    failures = [
        failure
        for artifact in artifacts
        if (failure := verify_artifact(artifact, args.timeout)) is not None
    ]
    if failures:
        print("Hub artifact check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Hub artifact check passed: {len(artifacts)} artifact(s) verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
