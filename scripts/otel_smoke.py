#!/usr/bin/env python3
"""Run a local OTLP smoke check against the self-hosted collector stack."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STACK = ROOT / "infra" / "otel"
ENDPOINT = "http://127.0.0.1:4317"
HEALTH_URL = "http://127.0.0.1:13133/"
METRICS_URL = "http://127.0.0.1:8888/metrics"
SPAN_ACCEPTED_METRICS = (
    "otelcol_receiver_accepted_spans_total",
    "otelcol_receiver_accepted_spans",
)
SPAN_SENT_METRICS = (
    "otelcol_exporter_sent_spans_total",
    "otelcol_exporter_sent_spans",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start infra/otel and prove the collector receives a Rust OTLP span."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for the stack and ingest metrics.",
    )
    parser.add_argument(
        "--down-after",
        action="store_true",
        help="Run docker compose down after the smoke check completes.",
    )
    args = parser.parse_args()

    compose = docker_compose_command()
    if compose is None:
        print("otel_smoke: docker compose is not available", file=sys.stderr)
        return 1

    try:
        run_checked(compose + ["up", "-d"])
        wait_for_url(HEALTH_URL, args.timeout)
        wait_for_metrics(args.timeout)

        accepted_before = read_metric_sum(SPAN_ACCEPTED_METRICS)
        sent_before = read_metric_sum(SPAN_SENT_METRICS)
        run_otel_test()
        accepted_after = wait_for_increment(
            SPAN_ACCEPTED_METRICS,
            accepted_before,
            args.timeout,
        )
        sent_after = wait_for_increment(SPAN_SENT_METRICS, sent_before, args.timeout)
    finally:
        if args.down_after:
            run_checked(compose + ["down"])

    print(
        "otel_smoke: span accepted "
        f"{accepted_before:g}->{accepted_after:g}; "
        f"span exported {sent_before:g}->{sent_after:g}"
    )
    return 0


def docker_compose_command() -> list[str] | None:
    command = [
        "docker",
        "compose",
        "-f",
        str(STACK / "docker-compose.yml"),
        "--env-file",
        str(STACK / "env.example"),
    ]
    try:
        subprocess.run(
            command + ["version"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return command


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_otel_test() -> None:
    env = os.environ.copy()
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = ENDPOINT
    env.setdefault("RUST_LOG", "warn")
    run_checked(
        [
            "cargo",
            "test",
            "-p",
            "lewm-telemetry",
            "--test",
            "otlp_endpoint_smoke",
            "--",
            "--ignored",
            "--nocapture",
        ],
        env=env,
    )


def wait_for_url(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


def wait_for_metrics(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = read_url(METRICS_URL)
        if "otelcol_" in text:
            return
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for collector metrics at {METRICS_URL}")


def wait_for_increment(names: tuple[str, ...], before: float, timeout: float) -> float:
    deadline = time.monotonic() + timeout
    last_value = before
    while time.monotonic() < deadline:
        last_value = read_metric_sum(names)
        if last_value > before:
            return last_value
        time.sleep(1)
    joined = ", ".join(names)
    raise RuntimeError(f"timed out waiting for {joined} to increment above {before:g}")


def read_metric_sum(names: tuple[str, ...]) -> float:
    text = read_url(METRICS_URL)
    total = 0.0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if match and match.group(1) in names:
            total += float(match.group(2))
    return total


def read_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
