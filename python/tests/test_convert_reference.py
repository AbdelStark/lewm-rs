from __future__ import annotations

import hashlib
import sys
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import convert_reference as convert  # noqa: E402
import param_name_map as pnm  # noqa: E402


class FakeTensor:
    shape = (1,)


def state_dict_with_expected_keys() -> dict[str, FakeTensor]:
    return {key: FakeTensor() for key in pnm.expected_source_keys()}


def test_audit_state_dict_keys_accepts_exact_locked_contract() -> None:
    audit = convert.audit_state_dict_keys(
        state_dict_with_expected_keys(),
        weights_path=Path("/tmp/reference/weights.pt"),
        weights_sha256="abc123",
    )

    validation = audit["key_validation"]
    assert validation["ok"] is True
    assert validation["actual_source_tensor_count"] == pnm.REFERENCE_SOURCE_TENSOR_COUNT
    assert validation["expected_source_tensor_count"] == pnm.REFERENCE_SOURCE_TENSOR_COUNT
    assert validation["expected_destination_tensor_count"] == pnm.REFERENCE_DESTINATION_TENSOR_COUNT
    assert validation["missing"] == []
    assert validation["extra"] == []
    assert audit["source_model"]["weights_path"] == "/tmp/reference/weights.pt"
    assert audit["source_model"]["weights_sha256"] == "abc123"


def test_audit_state_dict_keys_reports_missing_and_extra() -> None:
    state_dict = state_dict_with_expected_keys()
    state_dict.pop("predictor.transformer.norm.weight")
    state_dict["unexpected.weight"] = FakeTensor()

    audit = convert.audit_state_dict_keys(state_dict)

    validation = audit["key_validation"]
    assert validation["ok"] is False
    assert validation["missing"] == ["predictor.transformer.norm.weight"]
    assert validation["extra"] == ["unexpected.weight"]


def test_audit_state_dict_keys_can_include_sorted_source_keys() -> None:
    audit = convert.audit_state_dict_keys(state_dict_with_expected_keys(), include_keys=True)

    keys = audit["source_keys"]
    assert keys == sorted(keys)
    assert keys[0] == "action_encoder.embed.0.bias"
    assert len(keys) == pnm.REFERENCE_SOURCE_TENSOR_COUNT


def test_extract_state_dict_accepts_direct_and_nested_mappings() -> None:
    direct = {"encoder.layernorm.weight": FakeTensor()}
    nested = {"epoch": 1, "state_dict": direct}

    assert convert.extract_state_dict(direct) is direct
    assert convert.extract_state_dict(nested) is direct


def test_extract_state_dict_rejects_non_tensor_mapping() -> None:
    try:
        convert.extract_state_dict({"epoch": 1, "metrics": {"loss": 0.1}})
    except ValueError as exc:
        assert "could not locate tensor state_dict" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_sha256_file_streams_file_contents(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    payload = b"reference-weights"
    path.write_bytes(payload)

    assert convert.sha256_file(path) == hashlib.sha256(payload).hexdigest()
