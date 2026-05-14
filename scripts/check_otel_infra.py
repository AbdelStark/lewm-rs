#!/usr/bin/env python3
"""Validate the optional self-hosted OpenTelemetry stack."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STACK = ROOT / "infra" / "otel"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def read_required(path: Path, failures: list[str]) -> str:
    if not path.is_file():
        failures.append(f"{path.relative_to(ROOT)} is missing")
        return ""
    return path.read_text(encoding="utf-8")


def validate_compose(failures: list[str]) -> None:
    text = read_required(STACK / "docker-compose.yml", failures)
    if not text:
        return

    for needle in (
        "otel/opentelemetry-collector-contrib:",
        "grafana/tempo:",
        "prom/prometheus:",
        "grafana/grafana:",
        "127.0.0.1:4317:4317",
        "127.0.0.1:4318:4318",
        "127.0.0.1:3000:3000",
    ):
        require(needle in text, f"docker-compose.yml missing {needle!r}", failures)

    for forbidden in ("HF_TOKEN", "INTERN_AUDIT_HF_TOKEN", "OTEL_EXPORTER_OTLP_ENDPOINT_AUTH"):
        require(forbidden not in text, f"docker-compose.yml must not reference {forbidden}", failures)


def validate_collector(failures: list[str]) -> None:
    text = read_required(STACK / "collector.yaml", failures)
    if not text:
        return

    for needle in (
        "endpoint: 0.0.0.0:4317",
        "endpoint: 0.0.0.0:4318",
        "otlp/tempo:",
        "endpoint: tempo:4317",
        "processors: [memory_limiter, resource, batch]",
        "exporters: [otlp/tempo, debug]",
        "exporters: [prometheus, debug]",
    ):
        require(needle in text, f"collector.yaml missing {needle!r}", failures)


def validate_docs(failures: list[str]) -> None:
    text = read_required(STACK / "README.md", failures)
    if not text:
        return

    require(
        "Smoke training and CI do not require this stack" in text,
        "README.md must state the stack is optional for smoke/CI",
        failures,
    )
    require(
        "OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317" in text,
        "README.md must document the local OTLP endpoint",
        failures,
    )


def validate_dashboard(failures: list[str]) -> None:
    path = STACK / "grafana" / "dashboards" / "lewm-rs-training.json"
    text = read_required(path, failures)
    if not text:
        return

    try:
        dashboard = json.loads(text)
    except json.JSONDecodeError as exc:
        failures.append(f"{path.relative_to(ROOT)} invalid JSON: {exc}")
        return

    require(dashboard.get("uid") == "lewm-rs-training", "dashboard uid mismatch", failures)
    panels = dashboard.get("panels")
    require(isinstance(panels, list) and len(panels) >= 2, "dashboard needs at least two panels", failures)
    require("otel" in dashboard.get("tags", []), "dashboard must include otel tag", failures)


def main() -> int:
    failures: list[str] = []
    validate_compose(failures)
    validate_collector(failures)
    validate_docs(failures)
    validate_dashboard(failures)

    if failures:
        for failure in failures:
            print(f"check_otel_infra: {failure}", file=sys.stderr)
        return 1

    print("check_otel_infra: optional self-hosted OTEL stack ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
