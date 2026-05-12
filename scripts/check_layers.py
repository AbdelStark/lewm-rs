#!/usr/bin/env python3
"""Validate lewm-rs crate dependency layer invariants."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any


EXPECTED_MEMBERS = [
    "crates/lewm-core",
    "crates/lewm-data",
    "crates/lewm-train",
    "crates/lewm-plan",
    "crates/lewm-infer",
    "crates/lewm-telemetry",
    "crates/lewm-hub",
]

ALLOWED_DEPS = {
    "lewm-core": set(),
    "lewm-data": {"lewm-core"},
    "lewm-hub": {"lewm-core"},
    "lewm-telemetry": {"lewm-core"},
    "lewm-plan": {"lewm-core", "lewm-data", "lewm-telemetry"},
    "lewm-train": {
        "lewm-core",
        "lewm-data",
        "lewm-telemetry",
        "lewm-hub",
        "lewm-plan",
    },
    "lewm-infer": {"lewm-core", "lewm-telemetry"},
}

INFER_BANNED_DEPS = {"burn-cuda", "burn-autodiff"}
PYTHON_BINDING_DEPS = {"pyo3", "pyo3-build-config", "pyo3-ffi", "pyo3-macros"}


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def normalize_dep_name(name: str, spec: Any) -> str:
    if isinstance(spec, dict):
        package = spec.get("package")
        if isinstance(package, str):
            return package
    return name


def collect_deps(manifest: dict[str, Any]) -> set[str]:
    deps: set[str] = set()

    for table_name in ("dependencies", "dev-dependencies", "build-dependencies"):
        table = manifest.get(table_name, {})
        if isinstance(table, dict):
            deps.update(normalize_dep_name(name, spec) for name, spec in table.items())

    target_tables = manifest.get("target", {})
    if isinstance(target_tables, dict):
        for target in target_tables.values():
            if not isinstance(target, dict):
                continue
            for table_name in ("dependencies", "dev-dependencies", "build-dependencies"):
                table = target.get(table_name, {})
                if isinstance(table, dict):
                    deps.update(normalize_dep_name(name, spec) for name, spec in table.items())

    return deps


def python_binding_deps(deps: set[str]) -> list[str]:
    return sorted(dep for dep in deps if dep in PYTHON_BINDING_DEPS or dep.startswith("pyo3-"))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    root_manifest = load_toml(repo_root / "Cargo.toml")
    workspace = root_manifest.get("workspace", {})
    members = workspace.get("members", [])
    if members != EXPECTED_MEMBERS:
        failures.append(
            "workspace members differ from RFC 0001:\n"
            f"  expected: {EXPECTED_MEMBERS}\n"
            f"  actual:   {members}"
        )

    for crate, allowed in ALLOWED_DEPS.items():
        manifest_path = repo_root / "crates" / crate / "Cargo.toml"
        if not manifest_path.exists():
            failures.append(f"{crate}: missing manifest at {manifest_path.relative_to(repo_root)}")
            continue

        manifest = load_toml(manifest_path)
        deps = collect_deps(manifest)
        internal_deps = {dep for dep in deps if dep in ALLOWED_DEPS}
        disallowed = sorted(internal_deps - allowed)
        if disallowed:
            failures.append(
                f"{crate}: disallowed internal dependencies {disallowed}; "
                f"allowed: {sorted(allowed)}"
            )

        if crate == "lewm-infer":
            banned = sorted(deps & INFER_BANNED_DEPS)
            if banned:
                failures.append(f"{crate}: forbidden inference dependencies present: {banned}")

        python_deps = python_binding_deps(deps)
        if python_deps:
            failures.append(
                f"{crate}: forbidden Python binding dependencies present: {python_deps}; "
                "INV-004 requires an accepted ADR before adding PyO3"
            )

    if failures:
        print("Layer check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Layer check passed: no crate dependency violations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
