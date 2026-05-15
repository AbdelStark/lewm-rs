from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

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


def test_write_safetensors_emits_deterministic_header_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "reference.safetensors"
    tensors = {
        "z_float": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "a_int": np.array(7, dtype=np.int64),
    }

    infos = convert.write_safetensors(path, tensors)
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])

    assert infos == {
        "a_int": {"dtype": "I64", "shape": [1], "element_count": 1},
        "z_float": {"dtype": "F32", "shape": [2, 2], "element_count": 4},
    }
    assert header["__metadata__"]["producer"] == "lewm-rs python/convert_reference.py"
    assert header["a_int"]["dtype"] == "I64"
    assert header["z_float"]["dtype"] == "F32"
    assert header["z_float"]["shape"] == [2, 2]
    assert header["z_float"]["data_offsets"] == [8, 24]


def test_build_conversion_meta_links_rules_and_artifacts() -> None:
    rule = pnm.ParamRule.single(
        "encoder.layernorm.weight",
        "encoder.norm.gamma",
        pnm.Transform.IDENTITY,
    )
    audit = convert.audit_state_dict_keys(state_dict_with_expected_keys(), weights_sha256="abc")
    tensor_infos = {
        rule.destination: {
            "dtype": "F32",
            "shape": [192],
            "element_count": 192,
        }
    }

    meta = convert.build_conversion_meta(
        reference_meta=convert.load_reference_meta(),
        audit=audit,
        config_path=Path("/tmp/reference/config.json"),
        safetensors_out=Path("/tmp/reference/reference.safetensors"),
        safetensors_sha256="safe123",
        burn_record_out=Path("/tmp/reference/reference.mpk"),
        burn_record_sha256="mpk123",
        tensor_infos=tensor_infos,
        helper_command=["cargo", "run"],
    )

    assert meta["schema_version"] == "1.0"
    assert meta["artifacts"]["safetensors_sha256"] == "safe123"
    assert meta["artifacts"]["burn_record_sha256"] == "mpk123"
    assert meta["conversion"]["destination_tensor_count"] == 1
    assert meta["conversion"]["transform_counts"] == {"identity": 1}
    assert meta["tensors"]["encoder.norm.gamma"]["sources"] == ["encoder.layernorm.weight"]
