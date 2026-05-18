from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_pusht_warmstart_sources.py"
EXPECTED_PARAM_COUNT = 41_856

spec = importlib.util.spec_from_file_location("audit_pusht_warmstart_sources", SCRIPT)
assert spec is not None
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def write_record(path: Path, **updates: object) -> None:
    payload: dict[str, object] = {
        "schema_version": "1.1.0",
        "kind": "lewm-rs-pusht-bounded-module-lewm-record",
        "step": 50_000,
        "params": [0.0] * EXPECTED_PARAM_COUNT,
        "adamw_params": [],
    }
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_mpk_candidates_filters_tree_payload() -> None:
    payload = [
        {"type": "directory", "path": "train"},
        {"type": "file", "path": "train/run/step_0050000.safetensors", "size": 123},
        {"type": "file", "path": "train/run/step_0050000.mpk", "size": 456},
        {"path": "smoke/run/step_0000050.mpk", "size": "unknown"},
    ]

    assert audit.mpk_candidates(payload) == [
        {"path": "smoke/run/step_0000050.mpk", "size": None},
        {"path": "train/run/step_0050000.mpk", "size": 456},
    ]


def test_validate_candidate_accepts_current_record(tmp_path: Path) -> None:
    checker = audit.load_warmstart_checker()
    source = tmp_path / "step_0050000.mpk"
    write_record(source)

    result = audit.validate_candidate(checker, source, EXPECTED_PARAM_COUNT)

    assert result == {
        "status": "compatible",
        "reason": "accepted by scripts/check_warmstart_source.py",
    }


def test_validate_candidate_records_rejection_reason(tmp_path: Path) -> None:
    checker = audit.load_warmstart_checker()
    source = tmp_path / "step_0050000.mpk"
    write_record(source, schema_version="1.0.0")

    result = audit.validate_candidate(checker, source, EXPECTED_PARAM_COUNT)

    assert result["status"] == "rejected"
    assert "schema_version must be '1.1.0'" in result["reason"]


def test_build_report_marks_blocked_without_compatible_candidates() -> None:
    report = audit.build_report(
        repo="abdelstark/lewm-rs-pusht",
        revision="main",
        expected_params=EXPECTED_PARAM_COUNT,
        candidates=[
            {
                "path": "train/run/step_0050000.mpk",
                "size": 1266,
                "download_url": "https://example.invalid/step_0050000.mpk",
                "status": "rejected",
                "reason": "schema_version must be '1.1.0'",
            }
        ],
    )

    assert report["status"] == "blocked"
    assert report["candidate_count"] == 1
    assert report["compatible_count"] == 0
    assert report["expected"]["param_count"] == EXPECTED_PARAM_COUNT


def test_audit_candidates_normalizes_download_paths(tmp_path: Path, monkeypatch: object) -> None:
    checker = audit.load_warmstart_checker()
    candidate = {"path": "train/run/step_0050000.mpk", "size": 1266}

    def fake_write_candidate(
        repo: str,
        revision: str,
        candidate: dict[str, object],
        root: Path,
        timeout: float,
    ) -> Path:
        del repo, revision, candidate, timeout
        path = root / "train" / "run" / "step_0050000.mpk"
        path.parent.mkdir(parents=True)
        write_record(path, schema_version="1.0.0")
        return path

    monkeypatch.setattr(audit, "write_candidate", fake_write_candidate)

    results = audit.audit_candidates(
        repo="abdelstark/lewm-rs-pusht",
        revision="main",
        candidates=[candidate],
        download_root=tmp_path,
        timeout=1.0,
        checker=checker,
        expected_params=EXPECTED_PARAM_COUNT,
    )

    assert len(results) == 1
    assert results[0]["status"] == "rejected"
    assert str(tmp_path) not in results[0]["reason"]
    assert "train/run/step_0050000.mpk: schema_version must be '1.1.0'" in results[0]["reason"]
