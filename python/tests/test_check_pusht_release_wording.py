from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_pusht_release_wording.py"


def write_valid_docs(root: Path) -> None:
    docs = {
        "README.md": "lewm-rs release docs\n",
        "RELEASE.md": "release process\n",
        "ROADMAP.md": (
            "Historical bounded-core PushT training\n"
            "F1 full Burn/Jepa PushT release checkpoint is still pending\n"
            "train/pusht-full-burn-jepa-*\n"
            "all public PushT `.mpk` sources currently fail\n"
            "compatible current bounded-core PushT `.mpk` source\n"
        ),
        "CHANGELOG.md": "PushT bounded-core training job submitted\n",
        "docs/src/results/cost.md": (
            "PushT 50 k-step bounded-core run\n"
            "F1 full Burn/Jepa source-build run\n"
            "F3 warm-start SO-100 training\n"
            "combined \\$27\n"
            "\\$20 session cap\n"
        ),
        "docs/src/results/discussion.md": (
            "working bounded-core training pipeline\n"
            "Full Burn/Jepa PushT checkpoint\n"
            "F1 still needs\n"
            "release `onnx-full/` artifacts do not exist\n"
        ),
        "docs/src/status.md": (
            "50 k-step historical PushT bounded-core run\n"
            "train/pusht-full-burn-jepa-*\n"
        ),
        "docs/src/training/observability.md": "PushT bounded-core training report\n",
        "paper/lewm-rs.md": (
            "Historical 50k-step bounded-core PushT training\n"
            "F1 full Burn/Jepa\n"
        ),
        "reports/release_checklist.md": (
            "Bounded-core only\n"
            "zero ready `train/pusht-full-burn-jepa-*` candidates\n"
            "all six public PushT `.mpk` candidates are incompatible\n"
        ),
        "reports/so100_training.md": (
            "Provide a compatible current bounded-core PushT `.mpk` source\n"
            "approval-gated SO-100 warm-start job\n"
            "Evaluate from-scratch vs. warm-start once both checkpoints exist\n"
        ),
        "python/model_cards/README_so100.md": (
            "Blocked pending compatible PushT `.mpk` source\n"
            "Launch is blocked until a compatible current bounded-core PushT `.mpk` source exists\n"
        ),
    }
    for relative_path, text in docs.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_release_wording_passes(tmp_path: Path) -> None:
    write_valid_docs(tmp_path)

    result = run_check(tmp_path)

    assert result.returncode == 0
    assert "PushT release wording ok" in result.stdout


def test_rejects_stale_full_pusht_claim(tmp_path: Path) -> None:
    write_valid_docs(tmp_path)
    (tmp_path / "docs/src/status.md").write_text(
        "Full 50k-step PushT training\ntrain/pusht-full-burn-jepa-*\n",
        encoding="utf-8",
    )

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "stale PushT wording" in result.stderr
    assert "Full 50k-step PushT training" in result.stderr


def test_rejects_missing_bounded_core_correction(tmp_path: Path) -> None:
    write_valid_docs(tmp_path)
    (tmp_path / "ROADMAP.md").write_text(
        "F1 full Burn/Jepa PushT release checkpoint is still pending\n"
        "train/pusht-full-burn-jepa-*\n",
        encoding="utf-8",
    )

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "ROADMAP.md: missing required wording" in result.stderr
    assert "Historical bounded-core PushT training" in result.stderr


def test_rejects_stale_warmstart_epoch_source_claim(tmp_path: Path) -> None:
    write_valid_docs(tmp_path)
    (tmp_path / "ROADMAP.md").write_text(
        "Historical bounded-core PushT training\n"
        "F1 full Burn/Jepa PushT release checkpoint is still pending\n"
        "train/pusht-full-burn-jepa-*\n"
        "all public PushT `.mpk` sources currently fail\n"
        "compatible current bounded-core PushT `.mpk` source\n"
        "SO-100 warm-start from PushT epoch-10\n",
        encoding="utf-8",
    )

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "from PushT epoch-10" in result.stderr
