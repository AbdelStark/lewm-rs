#!/usr/bin/env python3
"""Parameter-name map for the locked PushT reference checkpoint.

This module is intentionally dependency-light. It records the source PyTorch
state-dict keys and the destination Burn record keys needed by the conversion
pipeline without requiring PyTorch, Transformers, or Safetensors to be present
for preflight checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REFERENCE_META_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "reference_model.meta.json"
)
REFERENCE_SOURCE_TENSOR_COUNT = 303
REFERENCE_DESTINATION_TENSOR_COUNT = 255


class Transform(StrEnum):
    """Array transform required when moving a source tensor into Burn layout."""

    IDENTITY = "identity"
    LINEAR_TRANSPOSE = "linear_transpose"
    QKV_LINEAR_CONCAT_TRANSPOSE = "qkv_linear_concat_transpose"
    QKV_BIAS_CONCAT = "qkv_bias_concat"
    SCALAR_TO_LEN1 = "scalar_to_len1"


@dataclass(frozen=True)
class ParamRule:
    """One destination tensor and the source tensor(s) required to build it."""

    sources: tuple[str, ...]
    destination: str
    transform: Transform

    @classmethod
    def single(cls, source: str, destination: str, transform: Transform) -> "ParamRule":
        return cls((source,), destination, transform)


@dataclass(frozen=True)
class KeyValidation:
    """Exact-key preflight result for a reference checkpoint state dict."""

    missing: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.extra

    def format_error(self) -> str:
        parts: list[str] = []
        if self.missing:
            parts.append("missing=" + ", ".join(self.missing))
        if self.extra:
            parts.append("extra=" + ", ".join(self.extra))
        return "; ".join(parts)


def parameter_rules() -> tuple[ParamRule, ...]:
    """Return the complete locked mapping for `quentinll/lewm-pusht`."""

    rules: list[ParamRule] = []
    rules.extend(_encoder_rules())
    rules.extend(_action_encoder_rules())
    rules.extend(_predictor_rules())
    rules.extend(_mlp_rules("projector"))
    rules.extend(_mlp_rules("pred_proj"))
    return tuple(rules)


def expected_source_keys() -> tuple[str, ...]:
    """Return all source PyTorch state-dict keys expected by the converter."""

    keys = {source for rule in parameter_rules() for source in rule.sources}
    return tuple(sorted(keys))


def expected_destination_keys() -> tuple[str, ...]:
    """Return all destination Burn record keys emitted by the converter."""

    return tuple(sorted(rule.destination for rule in parameter_rules()))


def validate_state_dict_keys(keys: Iterable[str]) -> KeyValidation:
    """Check that a source checkpoint key set exactly matches the locked map."""

    found = set(keys)
    expected = set(expected_source_keys())
    return KeyValidation(
        missing=tuple(sorted(expected - found)),
        extra=tuple(sorted(found - expected)),
    )


def ensure_expected_state_dict_keys(keys: Iterable[str]) -> None:
    """Raise a clear error if source checkpoint keys drift from the locked map."""

    validation = validate_state_dict_keys(keys)
    if not validation.ok:
        raise ValueError(f"reference state_dict key mismatch: {validation.format_error()}")


def map_numpy_state_dict(
    state_dict: Mapping[str, Any],
    rules: Iterable[ParamRule] | None = None,
) -> dict[str, np.ndarray]:
    """Apply conversion rules to numpy-compatible tensors.

    The production converter can call this after loading CPU tensors from the
    reference checkpoint, or use these same rules with a framework-native array
    backend. This helper is kept small so tests can lock transform semantics
    without importing PyTorch.
    """

    selected_rules = tuple(rules) if rules is not None else parameter_rules()
    ensure_expected_state_dict_keys(state_dict.keys())
    return {rule.destination: apply_rule(rule, state_dict) for rule in selected_rules}


def apply_rule(rule: ParamRule, state_dict: Mapping[str, Any]) -> np.ndarray:
    """Apply a single mapping rule to numpy-compatible source tensors."""

    values = [_as_array(state_dict[source]) for source in rule.sources]
    if rule.transform == Transform.IDENTITY:
        return values[0]
    if rule.transform == Transform.LINEAR_TRANSPOSE:
        return values[0].T
    if rule.transform == Transform.QKV_LINEAR_CONCAT_TRANSPOSE:
        return np.concatenate(values, axis=0).T
    if rule.transform == Transform.QKV_BIAS_CONCAT:
        return np.concatenate(values, axis=0)
    if rule.transform == Transform.SCALAR_TO_LEN1:
        return values[0].reshape(1)
    raise ValueError(f"unsupported transform: {rule.transform}")


def _encoder_rules() -> tuple[ParamRule, ...]:
    rules: list[ParamRule] = [
        ParamRule.single(
            "encoder.embeddings.patch_embeddings.projection.weight",
            "encoder.embeddings.patch_embed.proj.weight",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            "encoder.embeddings.patch_embeddings.projection.bias",
            "encoder.embeddings.patch_embed.proj.bias",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            "encoder.embeddings.cls_token",
            "encoder.embeddings.cls_token",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            "encoder.embeddings.position_embeddings",
            "encoder.embeddings.pos_embed",
            Transform.IDENTITY,
        ),
    ]

    for layer in range(12):
        src = f"encoder.encoder.layer.{layer}"
        dst = f"encoder.blocks.{layer}"
        rules.extend(
            [
                ParamRule.single(
                    f"{src}.layernorm_before.weight",
                    f"{dst}.norm1.gamma",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.layernorm_before.bias",
                    f"{dst}.norm1.beta",
                    Transform.IDENTITY,
                ),
                ParamRule(
                    (
                        f"{src}.attention.attention.query.weight",
                        f"{src}.attention.attention.key.weight",
                        f"{src}.attention.attention.value.weight",
                    ),
                    f"{dst}.attn.qkv.weight",
                    Transform.QKV_LINEAR_CONCAT_TRANSPOSE,
                ),
                ParamRule(
                    (
                        f"{src}.attention.attention.query.bias",
                        f"{src}.attention.attention.key.bias",
                        f"{src}.attention.attention.value.bias",
                    ),
                    f"{dst}.attn.qkv.bias",
                    Transform.QKV_BIAS_CONCAT,
                ),
                ParamRule.single(
                    f"{src}.attention.output.dense.weight",
                    f"{dst}.attn.proj.weight",
                    Transform.LINEAR_TRANSPOSE,
                ),
                ParamRule.single(
                    f"{src}.attention.output.dense.bias",
                    f"{dst}.attn.proj.bias",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.layernorm_after.weight",
                    f"{dst}.norm2.gamma",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.layernorm_after.bias",
                    f"{dst}.norm2.beta",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.intermediate.dense.weight",
                    f"{dst}.mlp.fc1.weight",
                    Transform.LINEAR_TRANSPOSE,
                ),
                ParamRule.single(
                    f"{src}.intermediate.dense.bias",
                    f"{dst}.mlp.fc1.bias",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.output.dense.weight",
                    f"{dst}.mlp.fc2.weight",
                    Transform.LINEAR_TRANSPOSE,
                ),
                ParamRule.single(
                    f"{src}.output.dense.bias",
                    f"{dst}.mlp.fc2.bias",
                    Transform.IDENTITY,
                ),
            ]
        )

    rules.extend(
        [
            ParamRule.single(
                "encoder.layernorm.weight",
                "encoder.norm.gamma",
                Transform.IDENTITY,
            ),
            ParamRule.single(
                "encoder.layernorm.bias",
                "encoder.norm.beta",
                Transform.IDENTITY,
            ),
        ]
    )
    return tuple(rules)


def _action_encoder_rules() -> tuple[ParamRule, ...]:
    return (
        ParamRule.single(
            "action_encoder.patch_embed.weight",
            "action_encoder.smoother.weight",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            "action_encoder.patch_embed.bias",
            "action_encoder.smoother.bias",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            "action_encoder.embed.0.weight",
            "action_encoder.fc1.weight",
            Transform.LINEAR_TRANSPOSE,
        ),
        ParamRule.single(
            "action_encoder.embed.0.bias",
            "action_encoder.fc1.bias",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            "action_encoder.embed.2.weight",
            "action_encoder.fc2.weight",
            Transform.LINEAR_TRANSPOSE,
        ),
        ParamRule.single(
            "action_encoder.embed.2.bias",
            "action_encoder.fc2.bias",
            Transform.IDENTITY,
        ),
    )


def _predictor_rules() -> tuple[ParamRule, ...]:
    rules: list[ParamRule] = [
        ParamRule.single(
            "predictor.pos_embedding",
            "predictor.pos_embed",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            "predictor.transformer.norm.weight",
            "predictor.norm.gamma",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            "predictor.transformer.norm.bias",
            "predictor.norm.beta",
            Transform.IDENTITY,
        ),
    ]

    for layer in range(6):
        src = f"predictor.transformer.layers.{layer}"
        dst = f"predictor.blocks.{layer}"
        rules.extend(
            [
                ParamRule.single(
                    f"{src}.attn.norm.weight",
                    f"{dst}.attn.norm.gamma",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.attn.norm.bias",
                    f"{dst}.attn.norm.beta",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.attn.to_qkv.weight",
                    f"{dst}.attn.qkv.weight",
                    Transform.LINEAR_TRANSPOSE,
                ),
                ParamRule.single(
                    f"{src}.attn.to_out.0.weight",
                    f"{dst}.attn.proj.weight",
                    Transform.LINEAR_TRANSPOSE,
                ),
                ParamRule.single(
                    f"{src}.attn.to_out.0.bias",
                    f"{dst}.attn.proj.bias",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.mlp.net.0.weight",
                    f"{dst}.mlp.norm.gamma",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.mlp.net.0.bias",
                    f"{dst}.mlp.norm.beta",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.mlp.net.1.weight",
                    f"{dst}.mlp.fc1.weight",
                    Transform.LINEAR_TRANSPOSE,
                ),
                ParamRule.single(
                    f"{src}.mlp.net.1.bias",
                    f"{dst}.mlp.fc1.bias",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.mlp.net.4.weight",
                    f"{dst}.mlp.fc2.weight",
                    Transform.LINEAR_TRANSPOSE,
                ),
                ParamRule.single(
                    f"{src}.mlp.net.4.bias",
                    f"{dst}.mlp.fc2.bias",
                    Transform.IDENTITY,
                ),
                ParamRule.single(
                    f"{src}.adaLN_modulation.1.weight",
                    f"{dst}.adaln.linear.weight",
                    Transform.LINEAR_TRANSPOSE,
                ),
                ParamRule.single(
                    f"{src}.adaLN_modulation.1.bias",
                    f"{dst}.adaln.linear.bias",
                    Transform.IDENTITY,
                ),
            ]
        )

    return tuple(rules)


def _mlp_rules(prefix: str) -> tuple[ParamRule, ...]:
    return (
        ParamRule.single(
            f"{prefix}.net.0.weight",
            f"{prefix}.fc1.weight",
            Transform.LINEAR_TRANSPOSE,
        ),
        ParamRule.single(
            f"{prefix}.net.0.bias",
            f"{prefix}.fc1.bias",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            f"{prefix}.net.1.weight",
            f"{prefix}.norm.weight",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            f"{prefix}.net.1.bias",
            f"{prefix}.norm.bias",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            f"{prefix}.net.1.running_mean",
            f"{prefix}.norm.running_mean",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            f"{prefix}.net.1.running_var",
            f"{prefix}.norm.running_var",
            Transform.IDENTITY,
        ),
        ParamRule.single(
            f"{prefix}.net.1.num_batches_tracked",
            f"{prefix}.norm.num_batches_tracked",
            Transform.SCALAR_TO_LEN1,
        ),
        ParamRule.single(
            f"{prefix}.net.3.weight",
            f"{prefix}.fc2.weight",
            Transform.LINEAR_TRANSPOSE,
        ),
        ParamRule.single(
            f"{prefix}.net.3.bias",
            f"{prefix}.fc2.bias",
            Transform.IDENTITY,
        ),
    )


def _as_array(value: Any) -> np.ndarray:
    return to_numpy_array(value)


def to_numpy_array(value: Any) -> np.ndarray:
    """Return a CPU NumPy view/copy for PyTorch or NumPy-compatible tensors."""

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
