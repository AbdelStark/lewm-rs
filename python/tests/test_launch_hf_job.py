"""Tests for the HF Jobs launcher safety guards.

The launcher lives at `scripts/launch_hf_job.py` and is the single
gateway through which the workspace submits paid HF Jobs. Regressions in
the cost guard or the image-tag rewriter would be expensive
($1.50/hour minimum on a10g-large), so the safety paths get full
unit coverage here.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON_DIR = PROJECT_ROOT / "python"
for candidate in (SCRIPTS_DIR, PYTHON_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import launch_hf_job  # noqa: E402


class TestParseTimeoutHours:
    """`parse_timeout_hours` accepts the HF Jobs YAML timeout grammar."""

    def test_hours(self) -> None:
        assert launch_hf_job.parse_timeout_hours("12h") == Decimal("12")
        assert launch_hf_job.parse_timeout_hours("1h") == Decimal("1")

    def test_minutes_round_up_to_cent(self) -> None:
        assert launch_hf_job.parse_timeout_hours("30m") == Decimal("0.50")
        assert launch_hf_job.parse_timeout_hours("45m") == Decimal("0.75")

    def test_seconds_round_up_to_basis_point(self) -> None:
        assert launch_hf_job.parse_timeout_hours("3600s") == Decimal("1.0000")
        assert launch_hf_job.parse_timeout_hours("60s") == Decimal("0.0167")

    def test_rejects_empty(self) -> None:
        with pytest.raises(launch_hf_job.LaunchError):
            launch_hf_job.parse_timeout_hours("")

    def test_rejects_unknown_suffix(self) -> None:
        with pytest.raises(launch_hf_job.LaunchError):
            launch_hf_job.parse_timeout_hours("5d")

    def test_rejects_unparseable_amount(self) -> None:
        with pytest.raises(launch_hf_job.LaunchError):
            launch_hf_job.parse_timeout_hours("xxh")


class TestRewriteImageTag:
    """`rewrite_image_tag` swaps the `:tag` suffix safely."""

    def test_replaces_latest(self) -> None:
        result = launch_hf_job.rewrite_image_tag(
            "ghcr.io/abdelstark/lewm-rs:latest", "v0.1.0"
        )
        assert result == "ghcr.io/abdelstark/lewm-rs:v0.1.0"

    def test_replaces_existing_version(self) -> None:
        result = launch_hf_job.rewrite_image_tag(
            "ghcr.io/abdelstark/lewm-rs:v0.0.9", "v0.1.0"
        )
        assert result == "ghcr.io/abdelstark/lewm-rs:v0.1.0"

    def test_refuses_to_overwrite_digest(self) -> None:
        with pytest.raises(launch_hf_job.LaunchError, match="digest"):
            launch_hf_job.rewrite_image_tag(
                "ghcr.io/abdelstark/lewm-rs:v0.1.0@sha256:deadbeef",
                "v0.2.0",
            )

    def test_rejects_invalid_tag_characters(self) -> None:
        for bad in ("v0.1.0 ", "tag/with/slash", "with space"):
            with pytest.raises(launch_hf_job.LaunchError, match="invalid image tag"):
                launch_hf_job.rewrite_image_tag(
                    "ghcr.io/abdelstark/lewm-rs:latest", bad
                )


class TestCheckCostCap:
    """`check_cost_cap` is the last line of defence against runaway spend."""

    def _job(self, hardware: str, timeout: str) -> dict[str, object]:
        return {"hardware": hardware, "timeout": timeout}

    # Use an existing in-tree YAML so `path.relative_to(ROOT)` succeeds.
    JOB_PATH = PROJECT_ROOT / "jobs" / "train_pusht.yaml"

    def test_allows_within_cap(self) -> None:
        launch_hf_job.check_cost_cap(
            self.JOB_PATH,
            self._job("a10g-large", "12h"),
            Decimal("20.00"),
        )

    def test_rejects_above_cap(self) -> None:
        with pytest.raises(launch_hf_job.LaunchError, match="exceeds cap"):
            launch_hf_job.check_cost_cap(
                self.JOB_PATH,
                self._job("a10g-large", "12h"),
                Decimal("5.00"),
            )

    def test_cap_zero_disables_guard(self) -> None:
        # h100 * 24h would normally be $192 — guard is disabled here.
        launch_hf_job.check_cost_cap(
            self.JOB_PATH,
            self._job("h100", "24h"),
            Decimal("0"),
        )

    def test_unknown_hardware_refuses(self) -> None:
        with pytest.raises(launch_hf_job.LaunchError, match="unknown hardware flavour"):
            launch_hf_job.check_cost_cap(
                self.JOB_PATH,
                self._job("not-a-real-flavour", "12h"),
                Decimal("20.00"),
            )

    def test_includes_breakdown_in_error_message(self) -> None:
        with pytest.raises(launch_hf_job.LaunchError) as info:
            launch_hf_job.check_cost_cap(
                self.JOB_PATH,
                self._job("a10g-large", "12h"),
                Decimal("5.00"),
            )
        message = str(info.value)
        assert "a10g-large" in message
        assert "12h" in message
        assert "1.50 USD/h" in message
        assert "$18.00" in message
        assert "$5.00" in message


class TestParseCostCap:
    """The `argparse` `type=` hook accepts well-formed USD strings."""

    def test_parses_decimal(self) -> None:
        assert launch_hf_job.parse_cost_cap("20") == Decimal("20.00")
        assert launch_hf_job.parse_cost_cap("0.5") == Decimal("0.50")
        assert launch_hf_job.parse_cost_cap("199.99") == Decimal("199.99")

    def test_zero_allowed(self) -> None:
        assert launch_hf_job.parse_cost_cap("0") == Decimal("0.00")

    def test_negative_rejected(self) -> None:
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            launch_hf_job.parse_cost_cap("-1.00")
