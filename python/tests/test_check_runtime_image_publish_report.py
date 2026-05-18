from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_runtime_image_publish_report.py"


def report_text(**replacements: str) -> str:
    text = """# Runtime Image Publication Attempt

denied: permission_denied: write_package

After that user action, rerun:

```bash
image_tag="f1-runtime-$(git rev-parse --short HEAD)"
gh workflow run runtime-image.yml --ref main -f image_tag="${image_tag}"
gh run watch
python3 scripts/verify_runtime_image.py --image-tag "${image_tag}"
```

F1 must not launch `jobs/train_pusht.yaml` until the verifier passes.

Dry-run preflight:

```bash
source_revision="$(python3 -c 'import json; print(json.load(open("reports/f1_source_build_dry_run.json", encoding="utf-8"))["source_revision"])')"
LEWM_SOURCE_REVISION="${source_revision}" \\
  python3 scripts/launch_hf_job.py jobs/train_pusht_source.yaml \\
    --dry-run \\
    --allow-approval-required
```

The rendered command is still a paid 12h A10G-large job and must not be launched
without explicit human approval. This fallback does not resolve F11.
"""
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def run_check(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_runtime_image_report_passes(tmp_path: Path) -> None:
    report = tmp_path / "runtime_image_publish.md"
    report.write_text(report_text(), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 0
    assert "runtime image publication report ok" in result.stdout


def test_rejects_hardcoded_retry_tag(tmp_path: Path) -> None:
    report = tmp_path / "runtime_image_publish.md"
    report.write_text(
        report_text(
            **{
                'image_tag="f1-runtime-$(git rev-parse --short HEAD)"': (
                    "gh workflow run runtime-image.yml --ref main "
                    "-f image_tag=f1-runtime-97880d0"
                )
            }
        ),
        encoding="utf-8",
    )

    result = run_check(report)

    assert result.returncode == 1
    assert "missing 'image_tag=\"f1-runtime-$(git rev-parse --short HEAD)\"'" in result.stderr


def test_rejects_source_build_current_head_shortcut(tmp_path: Path) -> None:
    report = tmp_path / "runtime_image_publish.md"
    report.write_text(
        report_text(
            **{
                'source_revision="$(python3 -c \'import json; print(json.load(open("reports/f1_source_build_dry_run.json", encoding="utf-8"))["source_revision"])\')"': (
                    'source_revision="$(git rev-parse HEAD)"'
                )
            }
        ),
        encoding="utf-8",
    )

    result = run_check(report)

    assert result.returncode == 1
    assert "reports/f1_source_build_dry_run.json" in result.stderr
