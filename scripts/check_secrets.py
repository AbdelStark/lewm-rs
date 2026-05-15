#!/usr/bin/env python3
"""Run the project secret scan through gitleaks."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

DEFAULT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class ScanOptions:
    source: Path
    config: Path
    gitleaks_bin: str
    report_path: Path | None
    no_git: bool
    timeout: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: list[str]) -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Run gitleaks with the lewm-rs allowlist and redacted reporting."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root,
        help="Repository or directory to scan (default: repo root).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / ".gitleaks.toml",
        help="Gitleaks config with project allowlists (default: .gitleaks.toml).",
    )
    parser.add_argument(
        "--gitleaks-bin",
        default=os.environ.get("GITLEAKS_BIN", "gitleaks"),
        help="Path to the gitleaks executable (default: GITLEAKS_BIN or gitleaks).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report output path. A temporary report is used by default.",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Scan the working tree as files instead of traversing git history.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Gitleaks timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run wrapper unit tests without requiring gitleaks.",
    )
    return parser.parse_args(argv)


def build_command(options: ScanOptions, report_path: Path) -> list[str]:
    command = [
        options.gitleaks_bin,
        "detect",
        "--source",
        str(options.source),
        "--config",
        str(options.config),
        "--redact=100",
        "--no-banner",
        "--no-color",
        "--report-format",
        "json",
        "--report-path",
        str(report_path),
        "--exit-code",
        "1",
        "--timeout",
        str(options.timeout),
    ]
    if options.no_git:
        command.append("--no-git")
    return command


def load_findings(report_path: Path) -> list[dict[str, Any]]:
    if not report_path.exists() or report_path.stat().st_size == 0:
        return []

    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"expected gitleaks JSON report list, got {type(payload).__name__}")

    findings: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            findings.append(item)
    return findings


def print_finding_summary(findings: list[dict[str, Any]], stream: TextIO) -> None:
    print(f"Secret scan failed: {len(findings)} finding(s) after project allowlists.", file=stream)
    for index, finding in enumerate(findings[:20], start=1):
        rule = finding.get("RuleID") or finding.get("Rule") or "unknown-rule"
        path = finding.get("File") or finding.get("Filename") or "unknown-path"
        line = finding.get("StartLine") or finding.get("Line") or "?"
        fingerprint = finding.get("Fingerprint") or "no-fingerprint"
        print(f"- {index}. {rule} at {path}:{line} ({fingerprint})", file=stream)
    if len(findings) > 20:
        print(f"- ... {len(findings) - 20} more finding(s) omitted.", file=stream)


def resolve_tool(binary: str) -> str | None:
    if os.sep in binary or (os.altsep and os.altsep in binary):
        return binary if Path(binary).exists() else None
    return shutil.which(binary)


def run_scan(options: ScanOptions, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    if options.timeout <= 0:
        print("--timeout must be a positive integer.", file=stderr)
        return 64

    if not options.config.is_file():
        print(f"Secret scan config not found: {options.config}", file=stderr)
        return 66

    if not options.source.exists():
        print(f"Secret scan source not found: {options.source}", file=stderr)
        return 66

    resolved = resolve_tool(options.gitleaks_bin)
    if resolved is None:
        print(
            "gitleaks is required for TST-0016-SECRETS-001. "
            "Install gitleaks or set GITLEAKS_BIN to the executable path.",
            file=stderr,
        )
        return 127

    report_context = (
        tempfile.TemporaryDirectory(prefix="lewm-gitleaks-")
        if options.report_path is None
        else None
    )
    try:
        if options.report_path is None:
            assert report_context is not None
            report_path = Path(report_context.name) / "gitleaks-report.json"
        else:
            report_path = options.report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)

        effective_options = ScanOptions(
            source=options.source,
            config=options.config,
            gitleaks_bin=resolved,
            report_path=options.report_path,
            no_git=options.no_git,
            timeout=options.timeout,
        )
        command = build_command(effective_options, report_path)
        result = subprocess.run(command, text=True, capture_output=True, check=False)

        if result.returncode == 0:
            print("Secret scan passed: gitleaks reported no findings.", file=stdout)
            return 0

        if result.returncode == 1:
            try:
                findings = load_findings(report_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Secret scan failed and report could not be parsed: {exc}", file=stderr)
                if result.stderr:
                    print(result.stderr, file=stderr, end="" if result.stderr.endswith("\n") else "\n")
                return 1
            print_finding_summary(findings, stderr)
            return 1

        print(f"gitleaks exited with status {result.returncode}.", file=stderr)
        if result.stdout:
            print(result.stdout, file=stderr, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=stderr, end="" if result.stderr.endswith("\n") else "\n")
        return result.returncode
    finally:
        if report_context is not None:
            report_context.cleanup()


class CheckSecretsTests(unittest.TestCase):
    def write_fake_gitleaks(self, directory: Path, body: str) -> Path:
        binary = directory / "gitleaks"
        binary.write_text(body, encoding="utf-8")
        binary.chmod(0o755)
        return binary

    def run_fake_scan(self, fake_body: str) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source = tmp / "repo"
            source.mkdir()
            config = tmp / ".gitleaks.toml"
            config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
            fake_bin = self.write_fake_gitleaks(tmp, fake_body)
            stdout = io.StringIO()
            stderr = io.StringIO()
            options = ScanOptions(
                source=source,
                config=config,
                gitleaks_bin=str(fake_bin),
                report_path=None,
                no_git=True,
                timeout=30,
            )
            code = run_scan(options, stdout=stdout, stderr=stderr)
            return code, stdout.getvalue(), stderr.getvalue()

    def test_clean_report_passes(self) -> None:
        body = textwrap.dedent(
            """\
            #!/usr/bin/env sh
            report=""
            while [ "$#" -gt 0 ]; do
              if [ "$1" = "--report-path" ]; then report="$2"; shift 2; else shift; fi
            done
            printf '[]' > "$report"
            exit 0
            """
        )
        code, stdout, stderr = self.run_fake_scan(body)
        self.assertEqual(code, 0, stderr)
        self.assertIn("Secret scan passed", stdout)

    def test_fake_token_blocks_scan(self) -> None:
        body = textwrap.dedent(
            """\
            #!/usr/bin/env sh
            report=""
            while [ "$#" -gt 0 ]; do
              if [ "$1" = "--report-path" ]; then report="$2"; shift 2; else shift; fi
            done
            cat > "$report" <<'JSON'
            [{"RuleID":"generic-api-key","File":"fixtures/leak.txt","StartLine":7,"Fingerprint":"fixture:fingerprint"}]
            JSON
            exit 1
            """
        )
        code, _stdout, stderr = self.run_fake_scan(body)
        self.assertEqual(code, 1)
        self.assertIn("Secret scan failed: 1 finding", stderr)
        self.assertIn("generic-api-key at fixtures/leak.txt:7", stderr)

    def test_missing_binary_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source = tmp / "repo"
            source.mkdir()
            config = tmp / ".gitleaks.toml"
            config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
            stderr = io.StringIO()
            code = run_scan(
                ScanOptions(
                    source=source,
                    config=config,
                    gitleaks_bin=str(tmp / "missing-gitleaks"),
                    report_path=None,
                    no_git=True,
                    timeout=30,
                ),
                stdout=io.StringIO(),
                stderr=stderr,
            )
            self.assertEqual(code, 127)
            self.assertIn("gitleaks is required", stderr.getvalue())

    def test_command_uses_project_config_and_redaction(self) -> None:
        options = ScanOptions(
            source=Path("repo"),
            config=Path(".gitleaks.toml"),
            gitleaks_bin="gitleaks",
            report_path=None,
            no_git=True,
            timeout=12,
        )
        command = build_command(options, Path("report.json"))
        self.assertIn("--config", command)
        self.assertIn(".gitleaks.toml", command)
        self.assertIn("--redact=100", command)
        self.assertIn("--no-git", command)
        self.assertEqual(command[command.index("--timeout") + 1], "12")


def run_self_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CheckSecretsTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_tests()

    options = ScanOptions(
        source=args.source,
        config=args.config,
        gitleaks_bin=args.gitleaks_bin,
        report_path=args.report_path,
        no_git=args.no_git,
        timeout=args.timeout,
    )
    return run_scan(options)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
