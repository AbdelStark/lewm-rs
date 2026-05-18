#!/usr/bin/env python3
"""Guard release docs against stale PushT full-checkpoint claims."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

DEFAULT_SCAN_PATHS = (
    Path("README.md"),
    Path("RELEASE.md"),
    Path("ROADMAP.md"),
    Path("CHANGELOG.md"),
    Path("reports"),
    Path("docs/src"),
    Path("paper"),
    Path("python/model_cards"),
)
FORBIDDEN_PHRASES = (
    "Full PushT training DONE",
    "PushT 50 k-step full run",
    "50 k-step PushT full run",
    "50k-step PushT full run",
    "PushT full training",
    "full PushT training",
    "Full 50k-step PushT training",
    "Full 50 k-step PushT training",
    "PushT full run",
    "Full Burn-Jepa end-to-end training",
    "from PushT epoch-10",
)
REQUIRED_TOKENS_BY_FILE = {
    Path("ROADMAP.md"): (
        "Historical bounded-core PushT training",
        "F1 full Burn/Jepa PushT release checkpoint is still pending",
        "train/pusht-full-burn-jepa-*",
        "all public PushT `.mpk` sources currently fail",
        "compatible current bounded-core PushT `.mpk` source",
    ),
    Path("docs/src/results/cost.md"): (
        "PushT 50 k-step bounded-core run",
        "F1 full Burn/Jepa source-build run",
        "F3 warm-start SO-100 training",
        "combined \\$27",
        "\\$20 session cap",
    ),
    Path("docs/src/results/discussion.md"): (
        "working bounded-core training pipeline",
        "Full Burn/Jepa PushT checkpoint",
        "F1 still needs",
        "release `onnx-full/` artifacts do not exist",
    ),
    Path("docs/src/status.md"): (
        "50 k-step historical PushT bounded-core run",
        "train/pusht-full-burn-jepa-*",
    ),
    Path("docs/src/training/observability.md"): (
        "PushT bounded-core training report",
    ),
    Path("paper/lewm-rs.md"): (
        "Historical 50k-step bounded-core PushT training",
        "F1 full Burn/Jepa",
    ),
    Path("reports/release_checklist.md"): (
        "Bounded-core only",
        "zero ready `train/pusht-full-burn-jepa-*` candidates",
        "all six public PushT `.mpk` candidates are incompatible",
    ),
    Path("reports/so100_training.md"): (
        "Provide a compatible current bounded-core PushT `.mpk` source",
        "approval-gated SO-100 warm-start job",
        "Evaluate from-scratch vs. warm-start once both checkpoints exist",
    ),
    Path("python/model_cards/README_so100.md"): (
        "Blocked pending compatible PushT `.mpk` source",
        "Launch is blocked until a compatible current bounded-core PushT `.mpk` source exists",
    ),
}


class PushTReleaseWordingError(RuntimeError):
    """Raised when release docs drift back to stale PushT wording."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root(),
        help="repository root to validate",
    )
    return parser.parse_args()


def iter_markdown_files(root: Path, relative_paths: Iterable[Path]) -> Iterable[Path]:
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.exists():
            continue
        if path.is_file():
            yield path
            continue
        for candidate in sorted(path.rglob("*.md")):
            if candidate.is_file():
                yield candidate


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def validate_forbidden_phrases(root: Path) -> None:
    failures: list[str] = []
    lowered_phrases = [(phrase, phrase.lower()) for phrase in FORBIDDEN_PHRASES]
    for path in iter_markdown_files(root, DEFAULT_SCAN_PATHS):
        text = path.read_text(encoding="utf-8")
        lowered_text = text.lower()
        for phrase, lowered_phrase in lowered_phrases:
            index = lowered_text.find(lowered_phrase)
            if index >= 0:
                relative = path.relative_to(root)
                failures.append(
                    f"{relative}:{line_number(text, index)}: stale PushT wording {phrase!r}"
                )
    if failures:
        raise PushTReleaseWordingError("\n".join(failures))


def validate_required_tokens(root: Path) -> None:
    failures: list[str] = []
    for relative_path, tokens in REQUIRED_TOKENS_BY_FILE.items():
        path = root / relative_path
        if not path.exists():
            failures.append(f"{relative_path}: missing required release doc")
            continue
        text = path.read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        for token in tokens:
            normalized_token = " ".join(token.split())
            if token not in text and normalized_token not in normalized_text:
                failures.append(f"{relative_path}: missing required wording {token!r}")
    if failures:
        raise PushTReleaseWordingError("\n".join(failures))


def main() -> int:
    root = parse_args().root.resolve()
    try:
        validate_forbidden_phrases(root)
        validate_required_tokens(root)
    except (OSError, PushTReleaseWordingError) as exc:
        print(f"check_pusht_release_wording.py: {exc}", file=sys.stderr)
        return 1

    print("PushT release wording ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
