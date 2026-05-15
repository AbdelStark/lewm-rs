from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import param_name_map as pnm  # noqa: E402


def test_expected_source_key_count_matches_reference_metadata() -> None:
    meta = json.loads(pnm.REFERENCE_META_PATH.read_text())
    source_keys = pnm.expected_source_keys()
    destination_keys = pnm.expected_destination_keys()

    assert len(source_keys) == pnm.REFERENCE_SOURCE_TENSOR_COUNT
    assert len(source_keys) == meta["source_model"]["state_dict_tensor_count"]
    assert len(set(source_keys)) == len(source_keys)
    assert len(destination_keys) == pnm.REFERENCE_DESTINATION_TENSOR_COUNT
    assert len(set(destination_keys)) == len(destination_keys)


def test_representative_rules_match_locked_destination_names() -> None:
    by_destination = {rule.destination: rule for rule in pnm.parameter_rules()}

    encoder_qkv = by_destination["encoder.blocks.0.attn.qkv.weight"]
    assert encoder_qkv.sources == (
        "encoder.encoder.layer.0.attention.attention.query.weight",
        "encoder.encoder.layer.0.attention.attention.key.weight",
        "encoder.encoder.layer.0.attention.attention.value.weight",
    )
    assert encoder_qkv.transform == pnm.Transform.QKV_LINEAR_CONCAT_TRANSPOSE

    predictor_qkv = by_destination["predictor.blocks.0.attn.qkv.weight"]
    assert predictor_qkv.sources == ("predictor.transformer.layers.0.attn.to_qkv.weight",)
    assert predictor_qkv.transform == pnm.Transform.LINEAR_TRANSPOSE

    assert (
        by_destination["projector.norm.num_batches_tracked"].sources
        == ("projector.net.1.num_batches_tracked",)
    )
    assert (
        by_destination["pred_proj.fc2.weight"].sources
        == ("pred_proj.net.3.weight",)
    )


def test_encoder_qkv_weight_and_bias_transforms() -> None:
    by_destination = {rule.destination: rule for rule in pnm.parameter_rules()}
    weight_rule = by_destination["encoder.blocks.0.attn.qkv.weight"]
    bias_rule = by_destination["encoder.blocks.0.attn.qkv.bias"]
    state_dict = {
        "encoder.encoder.layer.0.attention.attention.query.weight": np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
        ),
        "encoder.encoder.layer.0.attention.attention.key.weight": np.array(
            [[10.0, 20.0], [30.0, 40.0]], dtype=np.float32
        ),
        "encoder.encoder.layer.0.attention.attention.value.weight": np.array(
            [[100.0, 200.0], [300.0, 400.0]], dtype=np.float32
        ),
        "encoder.encoder.layer.0.attention.attention.query.bias": np.array(
            [1.0, 2.0], dtype=np.float32
        ),
        "encoder.encoder.layer.0.attention.attention.key.bias": np.array(
            [3.0, 4.0], dtype=np.float32
        ),
        "encoder.encoder.layer.0.attention.attention.value.bias": np.array(
            [5.0, 6.0], dtype=np.float32
        ),
    }

    np.testing.assert_array_equal(
        pnm.apply_rule(weight_rule, state_dict),
        np.array(
            [
                [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
                [2.0, 4.0, 20.0, 40.0, 200.0, 400.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        pnm.apply_rule(bias_rule, state_dict),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    )


def test_predictor_qkv_bias_is_not_part_of_locked_contract() -> None:
    assert "predictor.transformer.layers.0.attn.to_qkv.bias" not in pnm.expected_source_keys()
    assert "predictor.blocks.0.attn.qkv.bias" not in pnm.expected_destination_keys()


def test_batch_norm_counter_is_reshaped_for_burn_record() -> None:
    rule = {
        rule.destination: rule for rule in pnm.parameter_rules()
    }["projector.norm.num_batches_tracked"]

    converted = pnm.apply_rule(
        rule,
        {"projector.net.1.num_batches_tracked": np.array(17, dtype=np.int64)},
    )

    assert converted.shape == (1,)
    assert converted.dtype == np.int64
    assert converted.tolist() == [17]


def test_state_dict_key_validation_reports_missing_and_extra_keys() -> None:
    expected = set(pnm.expected_source_keys())
    missing = "encoder.layernorm.weight"
    keys = (expected - {missing}) | {"unexpected.weight"}

    validation = pnm.validate_state_dict_keys(keys)

    assert validation.ok is False
    assert validation.missing == (missing,)
    assert validation.extra == ("unexpected.weight",)
    assert "missing=encoder.layernorm.weight" in validation.format_error()
    assert "extra=unexpected.weight" in validation.format_error()
