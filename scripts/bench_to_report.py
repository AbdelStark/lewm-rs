#!/usr/bin/env python3
"""Convert Criterion output into committed baselines and regression reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_BASELINE = Path("bench-baselines/baselines.json")
DEFAULT_CRITERION_DIR = Path("target/criterion")
DEFAULT_REPORT = Path("reports/bench-regression.md")


@dataclass(frozen=True)
class BenchPoint:
    name: str
    mean_ns: float
    median_ns: float
    source_path: Path | None = None
    grace_started_at: date | None = None


@dataclass(frozen=True)
class Comparison:
    name: str
    baseline_ns: float | None
    current_ns: float | None
    change_pct: float | None
    status: str
    grace_started_at: date | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--criterion-dir", type=Path, default=DEFAULT_CRITERION_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--comment-file", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--grace-days", type=int, default=7)
    parser.add_argument("--default-grace-start-date")
    parser.add_argument("--hardware", default="local-bootstrap")
    parser.add_argument("--generated-at")
    parser.add_argument("--update-baseline", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generated_at(value: str | None) -> str:
    if value:
        return value
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"grace_started_at must be a string date, got {value!r}")
    if "T" in value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    return date.fromisoformat(value)


def criterion_bench_name(estimates_path: Path, criterion_dir: Path) -> str:
    relative = estimates_path.relative_to(criterion_dir)
    parts = relative.parts
    if len(parts) < 3 or parts[-2] != "new" or parts[-1] != "estimates.json":
        raise ValueError(f"unexpected Criterion estimates path: {estimates_path}")
    return "/".join(parts[:-2])


def discover_criterion_points(criterion_dir: Path) -> dict[str, BenchPoint]:
    points: dict[str, BenchPoint] = {}
    for estimates_path in sorted(criterion_dir.glob("**/new/estimates.json")):
        data = read_json(estimates_path)
        name = criterion_bench_name(estimates_path, criterion_dir)
        try:
            mean_ns = float(data["mean"]["point_estimate"])
            median_ns = float(data["median"]["point_estimate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Criterion estimates file: {estimates_path}") from error
        points[name] = BenchPoint(
            name=name,
            mean_ns=mean_ns,
            median_ns=median_ns,
            source_path=estimates_path,
        )
    return points


def load_baselines(path: Path) -> dict[str, BenchPoint]:
    payload = read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} has unsupported schema_version")

    benches = payload.get("benches")
    if not isinstance(benches, list):
        raise ValueError(f"{path} must contain a benches array")

    points: dict[str, BenchPoint] = {}
    for entry in benches:
        if not isinstance(entry, dict):
            raise ValueError(f"{path} contains a non-object bench entry")
        name = str(entry["name"])
        points[name] = BenchPoint(
            name=name,
            mean_ns=float(entry["mean_ns"]),
            median_ns=float(entry["median_ns"]),
            grace_started_at=parse_date(entry.get("grace_started_at")),
        )
    return points


def baseline_payload(
    current: dict[str, BenchPoint],
    old_baselines: dict[str, BenchPoint],
    hardware: str,
    generated_at_value: str,
) -> dict[str, Any]:
    benches: list[dict[str, Any]] = []
    for name, point in sorted(current.items()):
        entry: dict[str, Any] = {
            "mean_ns": point.mean_ns,
            "median_ns": point.median_ns,
            "name": name,
            "unit": "ns",
        }
        old_grace = old_baselines.get(name, BenchPoint(name, 0.0, 0.0)).grace_started_at
        if old_grace is not None:
            entry["grace_started_at"] = old_grace.isoformat()
        benches.append(entry)

    return {
        "benches": benches,
        "generated_at": generated_at_value,
        "hardware": hardware,
        "schema_version": SCHEMA_VERSION,
        "source": "criterion",
        "threshold": 0.05,
    }


def compare_points(
    baseline: dict[str, BenchPoint],
    current: dict[str, BenchPoint],
    threshold: float,
    grace_days: int,
    today: date,
    default_grace_start_date: date | None,
) -> tuple[list[Comparison], list[BenchPoint]]:
    comparisons: list[Comparison] = []

    for name, baseline_point in sorted(baseline.items()):
        current_point = current.get(name)
        if current_point is None:
            comparisons.append(
                Comparison(
                    name=name,
                    baseline_ns=baseline_point.mean_ns,
                    current_ns=None,
                    change_pct=None,
                    status="missing",
                )
            )
            continue

        if baseline_point.mean_ns <= 0.0:
            comparisons.append(
                Comparison(
                    name=name,
                    baseline_ns=baseline_point.mean_ns,
                    current_ns=current_point.mean_ns,
                    change_pct=None,
                    status="invalid-baseline",
                )
            )
            continue

        change_pct = (current_point.mean_ns - baseline_point.mean_ns) / baseline_point.mean_ns
        status = "pass"
        grace_started_at = baseline_point.grace_started_at or default_grace_start_date
        if change_pct > threshold:
            status = "grace"
            if grace_started_at is not None:
                age_days = (today - grace_started_at).days
                if age_days > grace_days:
                    status = "blocking"
        elif change_pct < -threshold:
            status = "improved"

        comparisons.append(
            Comparison(
                name=name,
                baseline_ns=baseline_point.mean_ns,
                current_ns=current_point.mean_ns,
                change_pct=change_pct,
                status=status,
                grace_started_at=grace_started_at,
            )
        )

    new_points = [point for name, point in sorted(current.items()) if name not in baseline]
    return comparisons, new_points


def format_ns(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1_000_000.0:
        return f"{value / 1_000_000.0:.3f} ms"
    if value >= 1_000.0:
        return f"{value / 1_000.0:.3f} us"
    return f"{value:.3f} ns"


def format_change(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:+.2f}%"


def status_label(comparison: Comparison) -> str:
    labels = {
        "blocking": "regression outside grace",
        "grace": "regression in grace",
        "improved": "improved",
        "invalid-baseline": "invalid baseline",
        "missing": "missing current run",
        "pass": "pass",
    }
    label = labels.get(comparison.status, comparison.status)
    if comparison.status == "grace" and comparison.grace_started_at is not None:
        return f"{label} since {comparison.grace_started_at.isoformat()}"
    return label


def render_report(
    comparisons: list[Comparison],
    new_points: list[BenchPoint],
    threshold: float,
    grace_days: int,
) -> str:
    regressions = [item for item in comparisons if item.status in {"blocking", "grace"}]
    blocking = [item for item in comparisons if item.status in {"blocking", "missing", "invalid-baseline"}]

    lines = [
        "# Benchmark Regression Report",
        "",
        f"Threshold: > {threshold * 100.0:.1f}% mean-time regression.",
        f"Grace period: {grace_days} days for entries with a fresh grace date.",
        "",
        f"Regressions: {len(regressions)}. Blocking findings: {len(blocking)}.",
        "",
        "| Bench | Baseline mean | Current mean | Change | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for comparison in comparisons:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{comparison.name}`",
                    format_ns(comparison.baseline_ns),
                    format_ns(comparison.current_ns),
                    format_change(comparison.change_pct),
                    status_label(comparison),
                ]
            )
            + " |"
        )

    if new_points:
        lines.extend(["", "## Unbaselined Benches", ""])
        for point in new_points:
            lines.append(f"- `{point.name}` current mean {format_ns(point.mean_ns)}")

    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_github_output(
    path: Path | None,
    comparisons: list[Comparison],
    new_points: list[BenchPoint],
    exit_code: int,
    report_path: Path,
) -> None:
    if path is None:
        return
    has_regressions = any(item.status in {"blocking", "grace"} for item in comparisons)
    blocking = any(item.status in {"blocking", "missing", "invalid-baseline"} for item in comparisons)
    lines = {
        "blocking": str(blocking).lower(),
        "blocking_count": str(sum(item.status in {"blocking", "missing", "invalid-baseline"} for item in comparisons)),
        "exit_code": str(exit_code),
        "has_regressions": str(has_regressions).lower(),
        "new_bench_count": str(len(new_points)),
        "regression_count": str(sum(item.status in {"blocking", "grace"} for item in comparisons)),
        "report_path": str(report_path),
    }
    with path.open("a", encoding="utf-8") as handle:
        for key, value in lines.items():
            handle.write(f"{key}={value}\n")


def compare_command(args: argparse.Namespace, current: dict[str, BenchPoint]) -> int:
    if not current:
        raise ValueError(f"no Criterion estimates found under {args.criterion_dir}")
    baseline = load_baselines(args.baseline_file)
    comparisons, new_points = compare_points(
        baseline=baseline,
        current=current,
        threshold=args.threshold,
        grace_days=args.grace_days,
        today=datetime.now(tz=UTC).date(),
        default_grace_start_date=parse_date(args.default_grace_start_date),
    )

    report = render_report(comparisons, new_points, args.threshold, args.grace_days)
    write_text(args.report, report)
    if args.comment_file is not None:
        write_text(args.comment_file, report)

    exit_code = 0
    if any(item.status in {"blocking", "missing", "invalid-baseline"} for item in comparisons):
        exit_code = 1
    append_github_output(args.github_output, comparisons, new_points, exit_code, args.report)
    return exit_code


def update_baseline_command(args: argparse.Namespace, current: dict[str, BenchPoint]) -> int:
    if not current:
        raise ValueError(f"no Criterion estimates found under {args.criterion_dir}")
    old_baselines: dict[str, BenchPoint] = {}
    if args.baseline_file.exists():
        old_baselines = load_baselines(args.baseline_file)
    payload = baseline_payload(
        current=current,
        old_baselines=old_baselines,
        hardware=args.hardware,
        generated_at_value=generated_at(args.generated_at),
    )
    write_json(args.baseline_file, payload)
    return 0


def main() -> int:
    args = parse_args()
    try:
        current = discover_criterion_points(args.criterion_dir)
        if args.update_baseline:
            return update_baseline_command(args, current)
        return compare_command(args, current)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"bench_to_report.py: {error}", file=sys.stderr)
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as handle:
                handle.write("blocking=true\n")
                handle.write("exit_code=1\n")
                handle.write("has_regressions=false\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
