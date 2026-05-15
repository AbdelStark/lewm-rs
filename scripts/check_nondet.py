#!/usr/bin/env python3
"""Reject known nondeterminism hazards in Rust sources."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ALLOW_MARKER = "determinism-lint: allow"
SKIP_DIRS = {".git", "target"}


@dataclass(frozen=True)
class Rule:
    """One forbidden source pattern."""

    name: str
    pattern: re.Pattern[str]
    message: str


RULES = [
    Rule(
        name="thread_rng",
        pattern=re.compile(r"\bthread_rng\s*\("),
        message="use an RFC 0013 ChaCha20Rng sub-stream instead of OS randomness",
    ),
    Rule(
        name="HashMap::iter",
        pattern=re.compile(r"\bHashMap\s*::\s*iter\b"),
        message="use BTreeMap or sort keys before iteration",
    ),
    Rule(
        name="Instant::now",
        pattern=re.compile(r"\bInstant\s*::\s*now\s*\("),
        message="inject a deterministic Clock in core logic",
    ),
]


def rust_files(root: Path, include_tests: bool) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.rs"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root)
        if not include_tests and ("tests" in rel.parts or rel.name.endswith("_test.rs")):
            continue
        files.append(path)
    return sorted(files)


def line_allowed(line: str, rule: Rule) -> bool:
    return ALLOW_MARKER in line and rule.name in line


def check_file(path: Path, root: Path) -> list[str]:
    failures: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for rule in RULES:
            if rule.pattern.search(line) and not line_allowed(line, rule):
                rel = path.relative_to(root)
                failures.append(f"{rel}:{lineno}: {rule.name}: {rule.message}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="also scan Rust integration-test directories",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for path in rust_files(root, args.include_tests):
        failures.extend(check_file(path, root))

    if failures:
        print("Nondeterminism lint failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Nondeterminism lint passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
