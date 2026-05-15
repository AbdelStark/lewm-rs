#!/usr/bin/env python3
"""Export a trained Burn LeWM checkpoint to ONNX for Tract CPU inference.

Usage:
    uv run python export_onnx.py \\
        --safetensors /path/to/checkpoint.safetensors \\
        --config /path/to/model_config.json \\
        --output-dir /path/to/onnx_out

The script:
1. Loads the Burn safetensors file (produced by lewm_core::export::to_safetensors).
2. Inverts the param_name_map to reconstruct PyTorch state-dict layout.
3. Loads the reference PyTorch model with those weights.
4. Exports encoder and predictor subgraphs to ONNX opset 18.
5. Writes encoder.onnx, predictor.onnx, and onnx_export.json beside them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import param_name_map as pnm

# Detect optional deps at import time; error at call time.
try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    from safetensors.numpy import load_file as st_load
    _ST_OK = True
except ImportError:
    _ST_OK = False

try:
    import transformers
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False


# ---------------------------------------------------------------------------
# Inverse-transform helpers
# ---------------------------------------------------------------------------

def _invert_identity(burn_value: np.ndarray) -> np.ndarray:
    return burn_value


def _invert_linear_transpose(burn_value: np.ndarray) -> np.ndarray:
    # Burn stores nn.Linear.weight as [in, out] (transposed from PyTorch's [out, in]).
    return burn_value.T


def _invert_scalar_to_len1(burn_value: np.ndarray) -> np.ndarray:
    # SCALAR_TO_LEN1 expanded a scalar to shape [1]; invert by squeezing.
    return burn_value.squeeze()


def invert_rule(rule: pnm.ParamRule, burn_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return {src_key: np.array} reconstructed from a Burn tensor.

    For multi-source rules (QKV concat), split the burn tensor back into the
    original source keys.
    """
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
        # burn_val shape: [3*hidden, in] (concatenated, transposed)
        # original Q/K/V weight shapes: [hidden, in] each in PyTorch
        n = len(rule.sources)
        assert n == 3, f"QKV concat expects 3 sources, got {n}"
        # Un-transpose first: [3*hidden, in] -> [in, 3*hidden]
        un_transposed = burn_val.T
        # Split along axis 1 (last dim after un-transpose = 3*hidden)
        # Wait: after linear_transpose inverse, shape is [in, 3*hidden]?
        # Actually the concat_transpose means: Q_w, K_w, V_w each [hidden, in]
        # are concatenated along axis 0 -> [3*hidden, in], then transposed -> [in, 3*hidden]
        # Burn stores it as [in, 3*hidden].
        # To invert: transpose [in, 3*hidden] -> [3*hidden, in], split axis 0 into 3.
        stacked = burn_val.T  # [3*hidden, in]
        parts = np.split(stacked, n, axis=0)  # 3 x [hidden, in]
        # PyTorch Linear weight is [out, in], so these are already [hidden, in].
        return {src: part for src, part in zip(rule.sources, parts)}

    if t == pnm.Transform.QKV_BIAS_CONCAT:
        # burn_val: [3*hidden] - concatenated biases
        n = len(rule.sources)
        assert n == 3
        parts = np.split(burn_val, n, axis=0)
        return {src: part for src, part in zip(rule.sources, parts)}

    raise ValueError(f"Unknown transform: {t}")


def burn_safetensors_to_pytorch_state_dict(
    burn_path: Path,
) -> dict[str, "torch.Tensor"]:
    """Load a Burn safetensors file and invert it back to a PyTorch state dict."""
    if not _ST_OK:
        raise RuntimeError("safetensors not installed; run: pip install safetensors")
    if not _TORCH_OK:
        raise RuntimeError("torch not installed")

    burn_dict: dict[str, np.ndarray] = st_load(str(burn_path))
    rules = pnm.parameter_rules()

    pt_numpy: dict[str, np.ndarray] = {}
    for rule in rules:
        recovered = invert_rule(rule, burn_dict)
        pt_numpy.update(recovered)

    # Check coverage
    missing_pt = set(pnm.expected_source_keys()) - set(pt_numpy)
    if missing_pt:
        print(f"WARNING: {len(missing_pt)} PyTorch keys not recovered from Burn checkpoint.", file=sys.stderr)
        for k in sorted(missing_pt)[:10]:
            print(f"  missing: {k}", file=sys.stderr)

    return {k: torch.from_numpy(v.astype(np.float32)) for k, v in pt_numpy.items()}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_reference_model(
    config_path: Path,
    state_dict: dict[str, "torch.Tensor"],
) -> Any:
    """Load the PyTorch reference LeWM model from config + inverted state dict."""
    if not _TRANSFORMERS_OK:
        raise RuntimeError("transformers not installed")

    # Import reference model from the script directory.
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    try:
        from convert_reference import _load_reference_model as _load
    except ImportError:
        raise RuntimeError(
            "Cannot import _load_reference_model from convert_reference.py; "
            "ensure you run from the python/ directory."
        )

    with open(config_path) as f:
        config_data = json.load(f)

    model = _load(config_data)
    # Load inverted weights - allow missing/unexpected for robustness.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} keys missing when loading into PyTorch model.", file=sys.stderr)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys ignored.", file=sys.stderr)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# ONNX export wrappers
# ---------------------------------------------------------------------------

class EncoderWrapper(nn.Module):
    """Wrap the encoder + projector for single-image ONNX export."""

    def __init__(self, full_model: Any) -> None:
        super().__init__()
        self.model = full_model

    def forward(self, pixels: "torch.Tensor") -> "torch.Tensor":
        # pixels: [B, C, H, W]
        with torch.no_grad():
            embedding = self.model.encode_single(pixels)
        return embedding


class PredictorWrapper(nn.Module):
    """Wrap the predictor for ONNX export."""

    def __init__(self, full_model: Any) -> None:
        super().__init__()
        self.model = full_model

    def forward(
        self, history: "torch.Tensor", actions: "torch.Tensor"
    ) -> "torch.Tensor":
        # history: [B, H, D], actions: [B, H, A]
        with torch.no_grad():
            pred = self.model.predict(history, actions)
        return pred


def export_encoder_onnx(model: Any, output_path: Path, image_size: int = 224, channels: int = 3) -> None:
    dummy = torch.zeros(1, channels, image_size, image_size)
    wrapper = EncoderWrapper(model)
    wrapper.eval()
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        opset_version=18,
        input_names=["pixels"],
        output_names=["embedding"],
        dynamic_axes={"pixels": {0: "batch"}, "embedding": {0: "batch"}},
        verbose=False,
    )
    print(f"Encoder ONNX written: {output_path}")


def export_predictor_onnx(
    model: Any,
    output_path: Path,
    history_size: int = 3,
    latent_dim: int = 192,
    action_dim: int = 10,
) -> None:
    dummy_history = torch.zeros(1, history_size, latent_dim)
    dummy_actions = torch.zeros(1, history_size, action_dim)
    wrapper = PredictorWrapper(model)
    wrapper.eval()
    torch.onnx.export(
        wrapper,
        (dummy_history, dummy_actions),
        str(output_path),
        opset_version=18,
        input_names=["history", "actions"],
        output_names=["predicted_embedding"],
        dynamic_axes={
            "history": {0: "batch", 1: "history"},
            "actions": {0: "batch", 1: "history"},
            "predicted_embedding": {0: "batch", 1: "history"},
        },
        verbose=False,
    )
    print(f"Predictor ONNX written: {output_path}")


def write_metadata(output_dir: Path, info: dict) -> None:
    meta_path = output_dir / "onnx_export.json"
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Export metadata: {meta_path}")


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
        "--config",
        type=Path,
        required=False,
        default=None,
        help="Path to HF-format config.json for the reference model architecture. "
             "If not provided, attempts to load from /tmp/lewm-rs-reference-model/config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write encoder.onnx, predictor.onnx, and onnx_export.json.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (square) for the encoder. Default: 224.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=3,
        help="Context history window for the predictor. Default: 3.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=192,
        help="Latent embedding dimension. Default: 192.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=10,
        help="Action dimension. Default: 10 (PushT). Use 6 for SO-100.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    for lib, ok, name in [(_TORCH_OK, _TORCH_OK, "torch"), (_ST_OK, _ST_OK, "safetensors"), (_TRANSFORMERS_OK, _TRANSFORMERS_OK, "transformers")]:
        if not ok:
            print(f"ERROR: '{name}' is not installed. Install it and retry.", file=sys.stderr)
            return 1

    if not args.safetensors.exists():
        print(f"ERROR: safetensors file not found: {args.safetensors}", file=sys.stderr)
        return 1

    config_path = args.config
    if config_path is None:
        config_path = Path("/tmp/lewm-rs-reference-model/config.json")
    if not config_path.exists():
        print(f"ERROR: config.json not found: {config_path}", file=sys.stderr)
        print("Download the reference model first: python convert_reference.py convert --download", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Burn safetensors: {args.safetensors}")
    state_dict = burn_safetensors_to_pytorch_state_dict(args.safetensors)
    print(f"Recovered {len(state_dict)} PyTorch keys from Burn checkpoint.")

    print("Loading reference PyTorch model...")
    model = load_reference_model(config_path, state_dict)

    encoder_path = args.output_dir / "encoder.onnx"
    predictor_path = args.output_dir / "predictor.onnx"

    print("Exporting encoder ONNX...")
    export_encoder_onnx(model, encoder_path, image_size=args.image_size)

    print("Exporting predictor ONNX...")
    export_predictor_onnx(
        model,
        predictor_path,
        history_size=args.history_size,
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
    )

    metadata = {
        "safetensors_source": str(args.safetensors),
        "opset_version": 18,
        "encoder": {"kind": "encoder", "path": str(encoder_path)},
        "predictor": {"kind": "predictor", "path": str(predictor_path)},
        "config": {
            "image_size": args.image_size,
            "history_size": args.history_size,
            "latent_dim": args.latent_dim,
            "action_dim": args.action_dim,
        },
    }
    write_metadata(args.output_dir, metadata)

    print("\nONNX export complete. Run lewm-infer with:")
    print(f"  lewm-infer --checkpoint-dir {args.output_dir} bench --image /path/to/img.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
