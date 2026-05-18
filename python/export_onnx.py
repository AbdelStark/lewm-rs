#!/usr/bin/env python3
"""Export a trained Burn LeWM checkpoint to ONNX.

Usage:
    uv run python export_onnx.py \\
        --safetensors /path/to/checkpoint.safetensors \\
        --output-dir /path/to/onnx_out

The script:
1. Loads the Burn safetensors file (produced by lewm_core::export::to_safetensors).
2. Inverts the param_name_map to reconstruct PyTorch state-dict layout.
3. Wraps encoder/projector and action-encoder/predictor/pred-proj in nn.Module.
4. Exports ONNX Runtime and/or Tract-compatible graph pairs.
5. Writes encoder.onnx, predictor.onnx, and onnx_export.json sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import param_name_map as pnm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_OK = True
except ImportError:
    torch = None
    F = None

    class _MissingTorchModule:
        pass

    class _MissingTorchNN:
        Module = _MissingTorchModule

    nn = _MissingTorchNN()
    _TORCH_OK = False

try:
    from safetensors.numpy import load_file as st_load

    _ST_OK = True
except ImportError:
    _ST_OK = False


# ---------------------------------------------------------------------------
# Inverse-transform helpers
# ---------------------------------------------------------------------------

BOUNDED_CORE_KEY_HINTS = frozenset(
    {
        "action_encoder.x.weight",
        "action_encoder.y.weight",
        "encoder.pixel.weight",
        "predictor.latent.weight",
        "projector.weight",
        "pred_proj.weight",
    }
)


class CheckpointContractError(RuntimeError):
    """Raised when a checkpoint cannot satisfy the full ONNX export contract."""


def _invert_identity(burn_value: np.ndarray) -> np.ndarray:
    return burn_value


def _invert_linear_transpose(burn_value: np.ndarray) -> np.ndarray:
    return burn_value.T


def _invert_scalar_to_len1(burn_value: np.ndarray) -> np.ndarray:
    return burn_value.squeeze()


def invert_rule(rule: pnm.ParamRule, burn_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return {src_key: np.array} reconstructed from a Burn tensor."""
    burn_val = burn_dict.get(rule.destination)
    if burn_val is None:
        return {}

    t = rule.transform

    if t == pnm.Transform.IDENTITY:
        assert len(rule.sources) == 1
        return {rule.sources[0]: _invert_identity(burn_val)}

    if t == pnm.Transform.LINEAR_TRANSPOSE:
        assert len(rule.sources) == 1
        return {rule.sources[0]: _invert_linear_transpose(burn_val)}

    if t == pnm.Transform.SCALAR_TO_LEN1:
        assert len(rule.sources) == 1
        return {rule.sources[0]: _invert_scalar_to_len1(burn_val)}

    if t == pnm.Transform.QKV_LINEAR_CONCAT_TRANSPOSE:
        n = len(rule.sources)
        assert n == 3
        stacked = burn_val.T  # [3*hidden, in]
        parts = np.split(stacked, n, axis=0)
        return dict(zip(rule.sources, parts, strict=True))

    if t == pnm.Transform.QKV_BIAS_CONCAT:
        n = len(rule.sources)
        assert n == 3
        parts = np.split(burn_val, n, axis=0)
        return dict(zip(rule.sources, parts, strict=True))

    raise ValueError(f"Unknown transform: {t}")


def recover_pytorch_numpy_from_burn(burn_path: Path) -> dict[str, np.ndarray]:
    """Load a Burn safetensors file and invert it back to PyTorch-keyed arrays."""
    if not _ST_OK:
        raise RuntimeError("safetensors not installed; run: pip install safetensors")

    burn_dict: dict[str, np.ndarray] = st_load(str(burn_path))
    rules = pnm.parameter_rules()

    pt_numpy: dict[str, np.ndarray] = {}
    for rule in rules:
        recovered = invert_rule(rule, burn_dict)
        pt_numpy.update(recovered)

    contract_error = checkpoint_contract_error(set(burn_dict), set(pt_numpy))
    if contract_error is not None:
        raise CheckpointContractError(contract_error)

    return {k: v.astype(np.float32) for k, v in pt_numpy.items()}


def burn_safetensors_to_state_dict(
    burn_path: Path,
) -> dict[str, torch.Tensor]:
    """Load a Burn safetensors file and invert it back to a PyTorch state dict."""
    if not _TORCH_OK or torch is None:
        raise RuntimeError("torch not installed")

    pt_numpy = recover_pytorch_numpy_from_burn(burn_path)
    return {k: torch.from_numpy(v.astype(np.float32)) for k, v in pt_numpy.items()}


def print_contract_summary(burn_path: Path, recovered: dict[str, np.ndarray]) -> None:
    """Print a concise full-checkpoint contract summary."""
    print(
        "Checkpoint contract ok: "
        f"recovered {len(recovered)} of {len(pnm.expected_source_keys())} "
        f"expected PyTorch keys from {burn_path}"
    )
    print(f"Safetensors SHA-256: {sha256_file(burn_path)}")


def checkpoint_contract_error(burn_keys: set[str], recovered_keys: set[str]) -> str | None:
    """Return a human-readable contract error, or None when export can proceed."""
    expected_keys = set(pnm.expected_source_keys())
    missing = expected_keys - recovered_keys
    if not missing:
        return None

    lines = [
        "checkpoint does not match the full Burn/Jepa ONNX export contract",
        f"recovered {len(recovered_keys)} of {len(expected_keys)} expected PyTorch keys",
        f"source safetensors tensor count: {len(burn_keys)}",
    ]
    if BOUNDED_CORE_KEY_HINTS.issubset(burn_keys):
        lines.append(
            "the tensor names match the bounded PushtFullLewmCore training artifact, "
            "not the full 303-tensor lewm_core::Jepa checkpoint"
        )
    lines.append("first missing keys:")
    lines.extend(f"  - {key}" for key in sorted(missing)[:10])
    lines.append(
        "provide a full Burn/Jepa safetensors checkpoint or use a separate exporter "
        "for the bounded-core artifact"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# nn.Module wrappers using raw state-dict ops (no transformers dependency)
# ---------------------------------------------------------------------------

class LeWMEncoderModule(nn.Module):
    """ViT encoder + projector MLP forward for ONNX export.

    Inputs:  pixels (B, C, H, W)
    Outputs: projected embedding (B, output_dim)
    """

    def __init__(self, state: dict[str, torch.Tensor], arch: dict[str, Any]) -> None:
        super().__init__()
        enc_cfg = arch["encoder"]
        self.patch_size: int = enc_cfg["patch_size"]
        self.num_heads: int = enc_cfg["num_attention_heads"]
        self.num_layers: int = enc_cfg["num_hidden_layers"]

        for k, v in state.items():
            if k.startswith("encoder.") or k.startswith("projector."):
                self.register_buffer(k.replace(".", "_"), v)

        self._state = state

    def _s(self, key: str) -> torch.Tensor:
        return self._state[key]

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        patch_size = self.patch_size
        num_heads = self.num_heads
        num_layers = self.num_layers

        patch_w = self._s("encoder.embeddings.patch_embeddings.projection.weight")
        patch_b = self._s("encoder.embeddings.patch_embeddings.projection.bias")
        D = patch_w.shape[0]

        x = F.conv2d(pixels, patch_w, patch_b, stride=patch_size)
        BT = x.shape[0]
        x = x.flatten(2).transpose(1, 2)

        cls_token = self._s("encoder.embeddings.cls_token")
        x = torch.cat([cls_token.expand(BT, -1, -1), x], dim=1)

        pos_embed = self._s("encoder.embeddings.position_embeddings")
        x = x + pos_embed

        head_dim = D // num_heads
        for i in range(num_layers):
            src = f"encoder.encoder.layer.{i}"
            ln1_w = self._s(f"{src}.layernorm_before.weight")
            ln1_b = self._s(f"{src}.layernorm_before.bias")
            normed = F.layer_norm(x, [D], ln1_w, ln1_b, eps=1e-12)

            q = F.linear(normed, self._s(f"{src}.attention.attention.query.weight"),
                         self._s(f"{src}.attention.attention.query.bias"))
            k = F.linear(normed, self._s(f"{src}.attention.attention.key.weight"),
                         self._s(f"{src}.attention.attention.key.bias"))
            v = F.linear(normed, self._s(f"{src}.attention.attention.value.weight"),
                         self._s(f"{src}.attention.attention.value.bias"))

            N = q.shape[1]
            q = q.reshape(BT, N, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(BT, N, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(BT, N, num_heads, head_dim).transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            attn_out = (attn @ v).transpose(1, 2).reshape(BT, N, D)

            out_w = self._s(f"{src}.attention.output.dense.weight")
            out_b = self._s(f"{src}.attention.output.dense.bias")
            x = x + F.linear(attn_out, out_w, out_b)

            ln2_w = self._s(f"{src}.layernorm_after.weight")
            ln2_b = self._s(f"{src}.layernorm_after.bias")
            normed2 = F.layer_norm(x, [D], ln2_w, ln2_b, eps=1e-12)

            fc1_w = self._s(f"{src}.intermediate.dense.weight")
            fc1_b = self._s(f"{src}.intermediate.dense.bias")
            fc2_w = self._s(f"{src}.output.dense.weight")
            fc2_b = self._s(f"{src}.output.dense.bias")
            x = x + F.linear(F.gelu(F.linear(normed2, fc1_w, fc1_b)), fc2_w, fc2_b)

        fn_w = self._s("encoder.layernorm.weight")
        fn_b = self._s("encoder.layernorm.bias")
        x = F.layer_norm(x, [D], fn_w, fn_b, eps=1e-12)
        cls = x[:, 0, :]

        # Projector MLP (BatchNorm1d in eval mode)
        fc1_w = self._s("projector.net.0.weight")
        fc1_b = self._s("projector.net.0.bias")
        bn_w = self._s("projector.net.1.weight")
        bn_b = self._s("projector.net.1.bias")
        bn_mean = self._s("projector.net.1.running_mean")
        bn_var = self._s("projector.net.1.running_var")
        fc2_w = self._s("projector.net.3.weight")
        fc2_b = self._s("projector.net.3.bias")

        proj = F.gelu(F.batch_norm(F.linear(cls, fc1_w, fc1_b), bn_mean, bn_var, bn_w, bn_b, training=False))
        return F.linear(proj, fc2_w, fc2_b)


class LeWMPredictorModule(nn.Module):
    """Action encoder + predictor transformer + pred_proj MLP for ONNX export.

    Inputs:  history (B, T, D), actions (B, T, A)
    Outputs: predicted embedding (B, T, output_dim)
    """

    def __init__(self, state: dict[str, torch.Tensor], arch: dict[str, Any]) -> None:
        super().__init__()
        pred_cfg = arch["predictor"]
        self.num_layers: int = pred_cfg["depth"]
        self.num_heads: int = pred_cfg["heads"]
        self.head_dim: int = pred_cfg["dim_head"]
        self._inner_dim = self.num_heads * self.head_dim
        self._T: int = pred_cfg.get("num_frames", 3)
        self._D: int = pred_cfg.get("input_dim", 192)
        self._state = state
        # Pre-register the causal mask as a buffer so ONNX doesn't see a dynamic
        # torch.ones(T, T) call – using a fixed T from the config is safe because
        # the predictor always runs with exactly num_frames history steps.
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(self._T, self._T, dtype=torch.bool), diagonal=1),
        )

    def _s(self, key: str) -> torch.Tensor:
        return self._state[key]

    def forward(
        self, history: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Action encoder
        smoother_w = self._s("action_encoder.patch_embed.weight")
        smoother_b = self._s("action_encoder.patch_embed.bias")
        fc1_w = self._s("action_encoder.embed.0.weight")
        fc1_b = self._s("action_encoder.embed.0.bias")
        fc2_w = self._s("action_encoder.embed.2.weight")
        fc2_b = self._s("action_encoder.embed.2.bias")

        ae = F.conv1d(actions.permute(0, 2, 1), smoother_w, smoother_b).permute(0, 2, 1)
        ae = F.linear(F.silu(F.linear(ae, fc1_w, fc1_b)), fc2_w, fc2_b)

        # Predictor transformer
        num_layers = self.num_layers
        num_heads = self.num_heads
        head_dim = self.head_dim
        inner_dim = self._inner_dim
        T = self._T
        D = self._D
        B = history.shape[0]

        pos_embed = self._s("predictor.pos_embedding")
        tokens = history + pos_embed[:, :T, :]

        causal_mask = self._causal_mask  # pre-registered buffer, no dynamic creation

        for i in range(num_layers):
            src = f"predictor.transformer.layers.{i}"

            adaln_w = self._s(f"{src}.adaLN_modulation.1.weight")
            adaln_b = self._s(f"{src}.adaLN_modulation.1.bias")
            mods = F.linear(F.silu(ae), adaln_w, adaln_b)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

            normed = F.layer_norm(tokens, [D])
            attn_input = normed * (1.0 + scale_msa) + shift_msa

            attn_norm_w = self._s(f"{src}.attn.norm.weight")
            attn_norm_b = self._s(f"{src}.attn.norm.bias")
            x = F.layer_norm(attn_input, [D], attn_norm_w, attn_norm_b)

            qkv_w = self._s(f"{src}.attn.to_qkv.weight")
            qkv = F.linear(x, qkv_w)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)

            attn_w = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn_w = attn_w.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_w = F.softmax(attn_w, dim=-1)
            attn_out = (attn_w @ v).transpose(1, 2).reshape(B, T, inner_dim)

            proj_w = self._s(f"{src}.attn.to_out.0.weight")
            proj_b = self._s(f"{src}.attn.to_out.0.bias")
            attn_out = F.linear(attn_out, proj_w, proj_b)
            tokens = tokens + gate_msa * attn_out

            normed2 = F.layer_norm(tokens, [D])
            mlp_input = normed2 * (1.0 + scale_mlp) + shift_mlp

            mlp_norm_w = self._s(f"{src}.mlp.net.0.weight")
            mlp_norm_b = self._s(f"{src}.mlp.net.0.bias")
            x_mlp = F.layer_norm(mlp_input, [D], mlp_norm_w, mlp_norm_b)

            fc1_w = self._s(f"{src}.mlp.net.1.weight")
            fc1_b = self._s(f"{src}.mlp.net.1.bias")
            fc2_w = self._s(f"{src}.mlp.net.4.weight")
            fc2_b = self._s(f"{src}.mlp.net.4.bias")
            x_mlp = F.linear(F.gelu(F.linear(x_mlp, fc1_w, fc1_b)), fc2_w, fc2_b)
            tokens = tokens + gate_mlp * x_mlp

        final_norm_w = self._s("predictor.transformer.norm.weight")
        final_norm_b = self._s("predictor.transformer.norm.bias")
        output = F.layer_norm(tokens, [D], final_norm_w, final_norm_b)

        # pred_proj MLP
        fc1_w = self._s("pred_proj.net.0.weight")
        fc1_b = self._s("pred_proj.net.0.bias")
        bn_w = self._s("pred_proj.net.1.weight")
        bn_b = self._s("pred_proj.net.1.bias")
        bn_mean = self._s("pred_proj.net.1.running_mean")
        bn_var = self._s("pred_proj.net.1.running_var")
        fc2_w = self._s("pred_proj.net.3.weight")
        fc2_b = self._s("pred_proj.net.3.bias")

        # Reshape to (B*T, D), project, then restore batch/time.
        # Use -1 for B*T to stay compatible with dynamic batch sizes.
        output_2d = output.reshape(-1, D)
        pp = F.gelu(F.batch_norm(F.linear(output_2d, fc1_w, fc1_b), bn_mean, bn_var, bn_w, bn_b, training=False))
        out = F.linear(pp, fc2_w, fc2_b)
        return out.reshape(output.shape[0], T, -1)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

VARIANT_ONNXRUNTIME = "onnxruntime"
VARIANT_TRACT_COMPAT = "tract-compat"
VARIANTS = (VARIANT_ONNXRUNTIME, VARIANT_TRACT_COMPAT)
STEP_RE = re.compile(r"step_(\d+)\.safetensors$")


def infer_action_dim(state: dict[str, torch.Tensor], fallback: int = 2) -> int:
    """Infer the exported predictor action dimension from checkpoint tensors."""
    smoother_w = state.get("action_encoder.patch_embed.weight")
    if smoother_w is None:
        return fallback
    return int(smoother_w.shape[1])


def export_encoder_onnx(
    state: dict[str, torch.Tensor],
    arch: dict[str, Any],
    output_path: Path,
    *,
    opset_version: int,
    dynamic_batch: bool,
) -> None:
    enc_cfg = arch["encoder"]
    image_size = enc_cfg.get("image_size", 224)
    channels = enc_cfg.get("num_channels", 3)
    dummy = torch.zeros(1, channels, image_size, image_size)
    module = LeWMEncoderModule(state, arch)
    module.eval()
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"pixels": {0: "batch"}, "embedding": {0: "batch"}}
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy,
            str(output_path),
            opset_version=opset_version,
            input_names=["pixels"],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
            verbose=False,
        )
    print(
        "Encoder ONNX written: "
        f"{output_path} (opset={opset_version}, dynamic_batch={dynamic_batch})"
    )


def export_predictor_onnx(
    state: dict[str, torch.Tensor],
    arch: dict[str, Any],
    output_path: Path,
    *,
    action_dim: int,
    opset_version: int,
    dynamic_batch: bool,
) -> None:
    pred_cfg = arch["predictor"]
    history_size = pred_cfg.get("num_frames", 3)
    latent_dim = pred_cfg.get("input_dim", 192)
    dummy_history = torch.zeros(1, history_size, latent_dim)
    dummy_actions = torch.zeros(1, history_size, action_dim)
    module = LeWMPredictorModule(state, arch)
    module.eval()
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "history": {0: "batch"},
            "actions": {0: "batch"},
            "predicted_embedding": {0: "batch"},
        }
    with torch.no_grad():
        torch.onnx.export(
            module,
            (dummy_history, dummy_actions),
            str(output_path),
            opset_version=opset_version,
            input_names=["history", "actions"],
            output_names=["predicted_embedding"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
            verbose=False,
        )
    print(
        "Predictor ONNX written: "
        f"{output_path} (action_dim={action_dim}, opset={opset_version}, "
        f"dynamic_batch={dynamic_batch})"
    )


def selected_variants(name: str) -> tuple[str, ...]:
    """Return the concrete ONNX variants requested by the CLI."""
    if name == "both":
        return VARIANTS
    return (name,)


def variant_output_dir(output_dir: Path, variant: str, variant_count: int) -> Path:
    """Return the output directory for a variant.

    A single-variant export preserves the historical flat layout. Multi-variant
    export writes explicit `onnxruntime/` and `tract-compat/` directories.
    """
    if variant_count == 1:
        return output_dir
    return output_dir / variant


def variant_export_options(variant: str) -> dict[str, Any]:
    """Return export options for a named variant."""
    if variant == VARIANT_ONNXRUNTIME:
        return {"opset_version": 18, "dynamic_batch": True}
    if variant == VARIANT_TRACT_COMPAT:
        return {"opset_version": 17, "dynamic_batch": False}
    raise ValueError(f"unknown ONNX variant: {variant}")


def parse_step_count(path: Path) -> int | None:
    """Parse `step_0050000.safetensors` into an integer step count."""
    match = STEP_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_metadata(
    *,
    safetensors: Path,
    output_dir: Path,
    arch: dict[str, Any],
    action_dim: int,
    variants: tuple[str, ...],
    export_timestamp: str,
) -> dict[str, Any]:
    """Build the ONNX export sidecar payload."""
    enc_cfg = arch["encoder"]
    pred_cfg = arch["predictor"]
    variant_count = len(variants)
    variant_info = {}
    for variant in variants:
        variant_dir = variant_output_dir(output_dir, variant, variant_count)
        options = variant_export_options(variant)
        variant_info[variant] = {
            "opset_version": options["opset_version"],
            "dynamic_batch": options["dynamic_batch"],
            "encoder": str((variant_dir / "encoder.onnx").relative_to(output_dir)),
            "predictor": str((variant_dir / "predictor.onnx").relative_to(output_dir)),
        }

    return {
        "schema_version": "1.0.0",
        "source": "burn_safetensors",
        "safetensors_source": str(safetensors),
        "safetensors_sha256": sha256_file(safetensors),
        "step_count": parse_step_count(safetensors),
        "export_timestamp": export_timestamp,
        "config": {
            "image_size": enc_cfg.get("image_size", 224),
            "history_size": pred_cfg.get("num_frames", 3),
            "latent_dim": pred_cfg.get("input_dim", 192),
            "action_dim": action_dim,
        },
        "variants": variant_info,
    }


def write_metadata(output_dir: Path, info: dict) -> None:
    meta_path = output_dir / "onnx_export.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    print(f"Export metadata: {meta_path}")


def load_arch_from_meta(meta_path: Path) -> dict[str, Any]:
    """Load the locked architecture dict from reference_model.meta.json."""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta["locked_architecture"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--safetensors",
        type=Path,
        required=True,
        help="Path to Burn safetensors checkpoint (from lewm_core::export::to_safetensors).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Path to reference_model.meta.json for locked architecture. "
             "Defaults to tests/fixtures/reference_model.meta.json relative to this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write encoder.onnx, predictor.onnx, and onnx_export.json.",
    )
    parser.add_argument(
        "--check-contract-only",
        action="store_true",
        help=(
            "Validate that --safetensors satisfies the full Burn/Jepa ONNX "
            "export contract and exit before requiring torch or writing ONNX."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("both", VARIANT_ONNXRUNTIME, VARIANT_TRACT_COMPAT),
        default="both",
        help=(
            "ONNX variant to export. Default: both, written under onnxruntime/ "
            "and tract-compat/. Single variants preserve the historical flat layout."
        ),
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=2,
        help="Action dimension. Default: 2 (PushT). Use 6 for SO-100.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not _ST_OK:
        print("ERROR: 'safetensors' is not installed.", file=sys.stderr)
        return 1

    if not args.safetensors.exists():
        print(f"ERROR: safetensors file not found: {args.safetensors}", file=sys.stderr)
        return 1

    if not args.check_contract_only and args.output_dir is None:
        print("ERROR: --output-dir is required unless --check-contract-only is set.", file=sys.stderr)
        return 1

    meta_path = args.meta
    if meta_path is None:
        script_dir = Path(__file__).resolve().parent
        meta_path = script_dir.parent / "tests" / "fixtures" / "reference_model.meta.json"
    if not args.check_contract_only and not meta_path.exists():
        print(f"ERROR: reference_model.meta.json not found: {meta_path}", file=sys.stderr)
        return 1

    print(f"Loading Burn safetensors: {args.safetensors}", flush=True)
    try:
        recovered_numpy = recover_pytorch_numpy_from_burn(args.safetensors)
    except CheckpointContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.check_contract_only:
        print_contract_summary(args.safetensors, recovered_numpy)
        return 0

    if not _TORCH_OK or torch is None:
        print("ERROR: 'torch' is not installed.", file=sys.stderr)
        return 1

    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = {k: torch.from_numpy(v) for k, v in recovered_numpy.items()}
    print(f"Recovered {len(state)} PyTorch keys from Burn checkpoint.")

    arch = load_arch_from_meta(meta_path)
    actual_action_dim = infer_action_dim(state, fallback=args.action_dim)
    variants = selected_variants(args.variant)

    for variant in variants:
        variant_dir = variant_output_dir(args.output_dir, variant, len(variants))
        variant_dir.mkdir(parents=True, exist_ok=True)
        options = variant_export_options(variant)
        encoder_path = variant_dir / "encoder.onnx"
        predictor_path = variant_dir / "predictor.onnx"

        print(f"Exporting {variant} encoder ONNX...")
        export_encoder_onnx(state, arch, encoder_path, **options)

        print(f"Exporting {variant} predictor ONNX...")
        export_predictor_onnx(
            state,
            arch,
            predictor_path,
            action_dim=actual_action_dim,
            **options,
        )

    metadata = build_metadata(
        safetensors=args.safetensors,
        output_dir=args.output_dir,
        arch=arch,
        action_dim=actual_action_dim,
        variants=variants,
        export_timestamp=utc_timestamp(),
    )
    write_metadata(args.output_dir, metadata)
    if len(variants) > 1:
        for variant in variants:
            variant_dir = variant_output_dir(args.output_dir, variant, len(variants))
            write_metadata(variant_dir, metadata)

    print("\nONNX export complete. Run lewm-infer with:")
    if VARIANT_TRACT_COMPAT in variants:
        tract_dir = variant_output_dir(args.output_dir, VARIANT_TRACT_COMPAT, len(variants))
    else:
        tract_dir = args.output_dir
    print(f"  lewm-infer --checkpoint-dir {tract_dir} bench --image /path/to/img.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
