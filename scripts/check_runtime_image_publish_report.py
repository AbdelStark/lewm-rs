#!/usr/bin/env python3
"""Validate the runtime-image publication blocker report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPORT = Path("reports/runtime_image_publish.md")


class RuntimeImagePublishReportError(RuntimeError):
    """Raised when the runtime-image publication report is stale or malformed."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"runtime image report path ({DEFAULT_REPORT})",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def require_token(text: str, token: str, path: Path) -> None:
    if token not in text:
        raise RuntimeImagePublishReportError(f"{path}: missing {token!r}")


def extract_section(text: str, start: str, end: str, path: Path) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise RuntimeImagePublishReportError(f"{path}: missing section start {start!r}")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise RuntimeImagePublishReportError(f"{path}: missing section end {end!r}")
    return text[start_index:end_index]


def validate_report(text: str, path: Path) -> None:
    for token in (
        "denied: permission_denied: write_package",
        'image_tag="f1-runtime-$(git rev-parse --short HEAD)"',
        'gh workflow run runtime-image.yml --ref main -f image_tag="${image_tag}"',
        'python3 scripts/verify_runtime_image.py --image-tag "${image_tag}"',
        "reports/f1_source_build_dry_run.json",
        '["source_revision"]',
        'LEWM_SOURCE_REVISION="${source_revision}"',
        "does not resolve F11",
    ):
        require_token(text, token, path)
    normalized_text = " ".join(text.split())
    if "must not be launched without explicit human approval" not in normalized_text:
        raise RuntimeImagePublishReportError(
            f"{path}: missing explicit human approval warning"
        )

    rerun_section = extract_section(
        text,
        "After that user action, rerun:",
        "F1 must not launch",
        path,
    )
    if "f1-runtime-97880d0" in rerun_section:
        raise RuntimeImagePublishReportError(
            f"{path}: rerun command must derive the runtime tag from current HEAD"
        )

    fallback_section = extract_section(
        text,
        "Dry-run preflight:",
        "The rendered command is still a paid",
        path,
    )
    if "git rev-parse HEAD" in fallback_section:
        raise RuntimeImagePublishReportError(
            f"{path}: source-build fallback must use the preflighted source revision report"
        )


def main() -> int:
    path = resolve_path(parse_args().path)
    try:
        text = path.read_text(encoding="utf-8")
        validate_report(text, path)
    except FileNotFoundError:
        print(f"check_runtime_image_publish_report.py: missing report: {path}", file=sys.stderr)
        return 1
    except RuntimeImagePublishReportError as exc:
        print(f"check_runtime_image_publish_report.py: {exc}", file=sys.stderr)
        return 1

    print("runtime image publication report ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
