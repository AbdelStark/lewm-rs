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

INFER_BANNED_DEPS = {"burn-cuda", "burn-autodiff", "nvml-wrapper"}
PYTHON_BINDING_DEPS = {"pyo3", "pyo3-build-config", "pyo3-ffi", "pyo3-macros"}
TELEMETRY_NVML_DEP = "nvml-wrapper"
TELEMETRY_NVML_FEATURE = "nvml"


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


def dependency_spec(manifest: dict[str, Any], dep_name: str) -> Any:
    for table_name in ("dependencies", "dev-dependencies", "build-dependencies"):
        table = manifest.get(table_name, {})
        if isinstance(table, dict) and dep_name in table:
            return table[dep_name]
    return None


def feature_list(manifest: dict[str, Any], feature_name: str) -> list[str]:
    features = manifest.get("features", {})
    if not isinstance(features, dict):
        return []
    values = features.get(feature_name, [])
    if isinstance(values, list) and all(isinstance(value, str) for value in values):
        return values
    return []


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

            telemetry_dep = dependency_spec(manifest, "lewm-telemetry")
            if isinstance(telemetry_dep, dict):
                enabled_features = telemetry_dep.get("features", [])
                if (
                    isinstance(enabled_features, list)
                    and TELEMETRY_NVML_FEATURE in enabled_features
                ):
                    failures.append(
                        f"{crate}: lewm-telemetry enables forbidden feature "
                        f"{TELEMETRY_NVML_FEATURE!r}"
                    )

        if crate == "lewm-telemetry":
            nvml_dep = dependency_spec(manifest, TELEMETRY_NVML_DEP)
            if not isinstance(nvml_dep, dict) or nvml_dep.get("optional") is not True:
                failures.append(
                    f"{crate}: {TELEMETRY_NVML_DEP} must remain an optional dependency"
                )

            if TELEMETRY_NVML_FEATURE in feature_list(manifest, "default"):
                failures.append(
                    f"{crate}: default features must not enable {TELEMETRY_NVML_FEATURE!r}"
                )

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
