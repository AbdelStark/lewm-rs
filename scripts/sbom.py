#!/usr/bin/env python3
"""Generate a deterministic CycloneDX SBOM from Cargo.lock."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def parse_lockfile(path: Path) -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[[package]]":
            if current is not None:
                packages.append(current)
            current = {}
            continue
        if current is None or "=" not in line:
            continue
        key, raw_value = [part.strip() for part in line.split("=", 1)]
        if key in {"name", "version", "source", "checksum"} and raw_value.startswith('"'):
            current[key] = json.loads(raw_value)

    if current is not None:
        packages.append(current)

    return packages


def workspace_version(manifest: Path) -> str:
    in_workspace_package = False
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[workspace.package]":
            in_workspace_package = True
            continue
        if in_workspace_package and line.startswith("["):
            return "0.0.0"
        if in_workspace_package and line.startswith("version"):
            _, raw_value = [part.strip() for part in line.split("=", 1)]
            return json.loads(raw_value)
    return "0.0.0"


def git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sbom_timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        timestamp = dt.datetime.fromtimestamp(int(epoch), tz=dt.UTC)
    else:
        timestamp = dt.datetime.now(tz=dt.UTC)
    return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def component_from_package(package: dict[str, str]) -> dict[str, Any]:
    name = package["name"]
    version = package["version"]
    purl = f"pkg:cargo/{name}@{version}"
    component: dict[str, Any] = {
        "type": "library",
        "bom-ref": purl,
        "name": name,
        "version": version,
        "purl": purl,
    }
    if source := package.get("source"):
        source_url = source.removeprefix("registry+").removeprefix("git+")
        component["externalReferences"] = [
            {
                "type": "distribution",
                "url": source_url,
            }
        ]
    if checksum := package.get("checksum"):
        component["hashes"] = [
            {
                "alg": "SHA-256",
                "content": checksum,
            }
        ]
    return component


def build_sbom(repo: Path) -> dict[str, Any]:
    packages = parse_lockfile(repo / "Cargo.lock")
    version = workspace_version(repo / "Cargo.toml")
    sha = git_sha(repo)
    properties = [{"name": "source_date_epoch", "value": os.environ.get("SOURCE_DATE_EPOCH", "")}]
    if sha is not None:
        properties.append({"name": "git.sha", "value": sha})

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:00000000-0000-0000-0000-{(sha or '0' * 12)[:12]}",
        "version": 1,
        "metadata": {
            "timestamp": sbom_timestamp(),
            "tools": [
                {
                    "type": "application",
                    "name": "lewm-rs scripts/sbom.py",
                    "version": version,
                }
            ],
            "component": {
                "type": "application",
                "name": "lewm-rs",
                "version": version,
                "purl": f"pkg:github/AbdelStark/lewm-rs@{version}",
            },
            "properties": properties,
        },
        "components": [
            component_from_package(package)
            for package in sorted(
                packages,
                key=lambda item: (
                    item.get("name", ""),
                    item.get("version", ""),
                    item.get("source", ""),
                ),
            )
            if "name" in package and "version" in package
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/sbom.cdx.json"),
        help="Path for the CycloneDX JSON output.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sbom = build_sbom(repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
