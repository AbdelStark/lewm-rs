#!/usr/bin/env python3
"""Inspect and convert the locked PushT reference checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import param_name_map as pnm


DEFAULT_LOCAL_DIR = Path("/tmp/lewm-rs-reference-model")
DEFAULT_AUDIT = Path("reports/parity/reference-key-audit.json")
DEFAULT_CONVERSION_META = Path("reports/parity/reference-conversion.meta.json")
DEFAULT_PARITY_FIXTURE = pnm.REFERENCE_META_PATH.parent / "parity_fixture.npz"
STATE_DICT_CANDIDATES = ("state_dict", "model", "model_state_dict", "module")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    audit = subcommands.add_parser(
        "audit",
        help="Validate the reference checkpoint state_dict key set.",
    )
    audit.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Directory containing config.json and weights.pt. Default: {DEFAULT_LOCAL_DIR}",
    )
    audit.add_argument(
        "--download",
        action="store_true",
        help="Use the hf CLI to download the locked config and weights before auditing.",
    )
    audit.add_argument(
        "--audit-out",
        type=Path,
        default=DEFAULT_AUDIT,
        help=f"JSON audit output path. Default: {DEFAULT_AUDIT}",
    )
    audit.add_argument(
        "--include-keys",
        action="store_true",
        help="Include the sorted source key list in the JSON audit.",
    )
    audit.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the 72 MB weights.pt SHA-256 check.",
    )
    convert = subcommands.add_parser(
        "convert",
        help="Convert the locked reference checkpoint to Safetensors and a Burn record.",
    )
    convert.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Directory containing config.json and weights.pt. Default: {DEFAULT_LOCAL_DIR}",
    )
    convert.add_argument(
        "--download",
        action="store_true",
        help="Use the hf CLI to download the locked config and weights before converting.",
    )
    convert.add_argument(
        "--safetensors-out",
        type=Path,
        required=True,
        help="Destination Safetensors mirror path.",
    )
    convert.add_argument(
        "--burn-record-out",
        type=Path,
        required=True,
        help="Destination Burn NamedMpk record path. Must end in .mpk.",
    )
    convert.add_argument(
        "--meta-out",
        type=Path,
        default=DEFAULT_CONVERSION_META,
        help=f"Destination conversion metadata JSON. Default: {DEFAULT_CONVERSION_META}",
    )
    convert.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the 72 MB weights.pt SHA-256 check.",
    )
    convert.add_argument(
        "--skip-burn-record",
        action="store_true",
        help="Write only Safetensors and metadata; intended for unit tests and debugging.",
    )
    dump = subcommands.add_parser(
        "dump",
        help=(
            "Run the locked reference model on the parity fixture and capture "
            "per-layer activations (RFC 0008 §4.2)."
        ),
    )
    dump.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Directory containing config.json and weights.pt. Default: {DEFAULT_LOCAL_DIR}",
    )
    dump.add_argument(
        "--dump-dir",
        type=Path,
        required=True,
        help="Destination directory for per-layer activation dumps.",
    )
    dump.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_PARITY_FIXTURE,
        help=f"Path to parity fixture .npz. Default: {DEFAULT_PARITY_FIXTURE}",
    )
    dump.add_argument(
        "--download",
        action="store_true",
        help="Use the hf CLI to download the locked config and weights before dumping.",
    )
    dump.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the 72 MB weights.pt SHA-256 check.",
    )
    dump.add_argument(
        "--fixture-seed",
        type=int,
        default=0,
        help="RNG seed used to sample the SIGReg projection matrix P. Default: 0",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "audit":
        return audit_command(args)
    if args.command == "convert":
        return convert_command(args)
    if args.command == "dump":
        return dump_command(args)
    raise AssertionError(f"unhandled command: {args.command}")


def audit_command(args: argparse.Namespace) -> int:
    meta = load_reference_meta()
    source = meta["source_model"]
    weights_path, config_path, weights_sha256 = prepare_reference_files(
        local_dir=args.local_dir,
        source=source,
        download=args.download,
        skip_sha256=args.skip_sha256,
    )

    state_dict = load_torch_state_dict(weights_path)
    audit = audit_state_dict_keys(
        state_dict,
        meta=meta,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        include_keys=args.include_keys,
    )
    write_json(args.audit_out, audit)
    validation = audit["key_validation"]
    print(
        "reference key audit: "
        f"ok={validation['ok']} "
        f"actual={validation['actual_source_tensor_count']} "
        f"expected={validation['expected_source_tensor_count']} "
        f"out={args.audit_out}"
    )
    return 0 if validation["ok"] else 1


def convert_command(args: argparse.Namespace) -> int:
    meta = load_reference_meta()
    source = meta["source_model"]
    weights_path, config_path, weights_sha256 = prepare_reference_files(
        local_dir=args.local_dir,
        source=source,
        download=args.download,
        skip_sha256=args.skip_sha256,
    )

    state_dict = load_torch_state_dict(weights_path)
    audit = audit_state_dict_keys(
        state_dict,
        meta=meta,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        include_keys=False,
    )
    validation = audit["key_validation"]
    if not validation["ok"]:
        raise SystemExit(f"reference key audit failed: {validation}")

    converted = pnm.map_numpy_state_dict(state_dict)
    tensor_infos = write_safetensors(args.safetensors_out, converted)
    safetensors_sha256 = sha256_file(args.safetensors_out)

    helper_command = None
    if not args.skip_burn_record:
        helper_command = write_burn_record(
            safetensors_out=args.safetensors_out,
            burn_record_out=args.burn_record_out,
        )
    burn_record_sha256 = (
        sha256_file(args.burn_record_out)
        if not args.skip_burn_record and args.burn_record_out.is_file()
        else None
    )

    conversion_meta = build_conversion_meta(
        reference_meta=meta,
        audit=audit,
        config_path=config_path,
        safetensors_out=args.safetensors_out,
        safetensors_sha256=safetensors_sha256,
        burn_record_out=args.burn_record_out,
        burn_record_sha256=burn_record_sha256,
        tensor_infos=tensor_infos,
        helper_command=helper_command,
    )
    write_json(args.meta_out, conversion_meta)
    print(
        "reference conversion: "
        f"tensors={len(converted)} "
        f"safetensors={args.safetensors_out} "
        f"burn_record={args.burn_record_out if not args.skip_burn_record else 'skipped'} "
        f"meta={args.meta_out}"
    )
    return 0


def prepare_reference_files(
    *,
    local_dir: Path,
    source: Mapping[str, Any],
    download: bool,
    skip_sha256: bool,
) -> tuple[Path, Path, str | None]:
    weights_path = local_dir / source["weights_file"]
    config_path = local_dir / source["config_file"]

    if download:
        download_reference_files(
            repo_id=source["repo_id"],
            revision=source["revision"],
            local_dir=local_dir,
            filenames=(source["config_file"], source["weights_file"]),
        )
    require_file(config_path, "--download or provide the locked config.json")
    require_file(weights_path, "--download or provide the locked weights.pt")

    weights_sha256 = None
    if not skip_sha256:
        weights_sha256 = sha256_file(weights_path)
        if weights_sha256 != source["weights_sha256"]:
            raise SystemExit(
                f"weights sha256 mismatch: got {weights_sha256}, expected {source['weights_sha256']}"
            )
    return weights_path, config_path, weights_sha256


def load_reference_meta() -> Mapping[str, Any]:
    return json.loads(pnm.REFERENCE_META_PATH.read_text(encoding="utf-8"))


def download_reference_files(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
    filenames: tuple[str, ...],
) -> None:
    if shutil.which("hf") is None:
        raise SystemExit("hf CLI is required for --download")
    command = [
        "hf",
        "download",
        repo_id,
        *filenames,
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
    ]
    subprocess.run(command, check=True)


def load_torch_state_dict(weights_path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to inspect weights.pt; install torch or run this command "
            "in an environment that already has it."
        ) from exc

    checkpoint = torch.load(weights_path, map_location="cpu")
    return extract_state_dict(checkpoint)


def extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"expected checkpoint mapping, got {type(checkpoint).__name__}")
    if looks_like_state_dict(checkpoint):
        return checkpoint
    for key in STATE_DICT_CANDIDATES:
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and looks_like_state_dict(value):
            return value
    available = ", ".join(str(key) for key in sorted(checkpoint.keys()))
    raise ValueError(f"could not locate tensor state_dict; top-level keys: {available}")


def looks_like_state_dict(value: Mapping[str, Any]) -> bool:
    if not value:
        return False
    return all(is_tensor_like(tensor) for tensor in value.values())


def is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") or hasattr(value, "size")


def audit_state_dict_keys(
    state_dict: Mapping[str, Any],
    *,
    meta: Mapping[str, Any] | None = None,
    weights_path: Path | None = None,
    weights_sha256: str | None = None,
    include_keys: bool = False,
) -> dict[str, Any]:
    reference_meta = load_reference_meta() if meta is None else meta
    source = reference_meta["source_model"]
    keys = tuple(sorted(state_dict.keys()))
    validation = pnm.validate_state_dict_keys(keys)
    audit: dict[str, Any] = {
        "schema_version": "1.0",
        "source_model": {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "weights_file": source["weights_file"],
            "weights_sha256": weights_sha256,
            "weights_path": str(weights_path) if weights_path is not None else None,
        },
        "key_validation": {
            "ok": validation.ok,
            "actual_source_tensor_count": len(keys),
            "expected_source_tensor_count": len(pnm.expected_source_keys()),
            "expected_destination_tensor_count": len(pnm.expected_destination_keys()),
            "missing": list(validation.missing),
            "extra": list(validation.extra),
        },
    }
    if include_keys:
        audit["source_keys"] = list(keys)
    return audit


def write_safetensors(path: Path, tensors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Write a deterministic F32/I64 Safetensors file without optional deps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "producer": "lewm-rs python/convert_reference.py",
            "schema_version": "1.0",
        }
    }
    data_chunks: list[bytes] = []
    tensor_infos: dict[str, dict[str, Any]] = {}
    offset = 0
    for name in sorted(tensors):
        array = canonical_array(name, tensors[name])
        payload = array.tobytes(order="C")
        start = offset
        offset += len(payload)
        dtype = safetensors_dtype(array)
        shape = list(array.shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, offset],
        }
        tensor_infos[name] = {
            "dtype": dtype,
            "shape": shape,
            "element_count": int(array.size),
        }
        data_chunks.append(payload)

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    payload = struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_chunks)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return tensor_infos


def canonical_array(name: str, value: Any) -> Any:
    array = pnm.to_numpy_array(value)
    if array.dtype.kind == "f":
        return np_as_contiguous(array, "<f4")
    if array.dtype.kind in {"i", "u"}:
        return np_as_contiguous(array, "<i8")
    raise TypeError(f"unsupported tensor dtype for {name}: {array.dtype}")


def np_as_contiguous(array: Any, dtype: str) -> Any:
    import numpy as np

    return np.ascontiguousarray(array.astype(dtype, copy=False))


def safetensors_dtype(array: Any) -> str:
    if array.dtype.kind == "f" and array.dtype.itemsize == 4:
        return "F32"
    if array.dtype.kind == "i" and array.dtype.itemsize == 8:
        return "I64"
    raise TypeError(f"unsupported canonical dtype: {array.dtype}")


def write_burn_record(*, safetensors_out: Path, burn_record_out: Path) -> list[str]:
    if burn_record_out.suffix != ".mpk":
        raise SystemExit(f"--burn-record-out must end in .mpk: {burn_record_out}")
    command = [
        "cargo",
        "run",
        "--locked",
        "-p",
        "lewm-train",
        "--bin",
        "lewm-reference-record",
        "--",
        "--safetensors-in",
        str(safetensors_out),
        "--burn-record-out",
        str(burn_record_out),
    ]
    subprocess.run(command, check=True)
    return command


def build_conversion_meta(
    *,
    reference_meta: Mapping[str, Any],
    audit: Mapping[str, Any],
    config_path: Path,
    safetensors_out: Path,
    safetensors_sha256: str,
    burn_record_out: Path,
    burn_record_sha256: str | None,
    tensor_infos: Mapping[str, Mapping[str, Any]],
    helper_command: list[str] | None,
) -> dict[str, Any]:
    rules = pnm.parameter_rules()
    by_destination = {rule.destination: rule for rule in rules}
    transform_counts: dict[str, int] = {}
    tensors: dict[str, Any] = {}
    for name in sorted(tensor_infos):
        rule = by_destination[name]
        transform_counts[rule.transform.value] = transform_counts.get(rule.transform.value, 0) + 1
        tensors[name] = {
            **tensor_infos[name],
            "sources": list(rule.sources),
            "transform": rule.transform.value,
        }

    source = reference_meta["source_model"]
    return {
        "schema_version": "1.0",
        "source_model": {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "config_file": source["config_file"],
            "config_path": str(config_path),
            "weights_file": source["weights_file"],
            "weights_sha256": audit["source_model"]["weights_sha256"],
            "weights_path": audit["source_model"]["weights_path"],
        },
        "key_validation": audit["key_validation"],
        "conversion": {
            "source_tensor_count": len(pnm.expected_source_keys()),
            "destination_tensor_count": len(tensor_infos),
            "transform_counts": transform_counts,
        },
        "artifacts": {
            "safetensors": str(safetensors_out),
            "safetensors_sha256": safetensors_sha256,
            "burn_record": str(burn_record_out),
            "burn_record_sha256": burn_record_sha256,
            "burn_record_helper": helper_command,
        },
        "tensors": tensors,
    }


def dump_command(args: argparse.Namespace) -> int:
    """Capture per-layer activations per RFC 0008 §4.2."""
    try:
        import platform

        import numpy as np
        import torch
        import torch.nn.functional as F  # noqa: F401 – used in helpers via torch.nn.functional
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and NumPy are required for the dump command; "
            "install torch or activate an environment that has it."
        ) from exc

    meta = load_reference_meta()
    source = meta["source_model"]
    arch = meta["locked_architecture"]

    weights_path, _config_path, weights_sha256 = prepare_reference_files(
        local_dir=args.local_dir,
        source=source,
        download=args.download,
        skip_sha256=args.skip_sha256,
    )
    state = load_torch_state_dict(weights_path)
    state = {k: v.float().detach().cpu() for k, v in state.items()}

    fixture_path: Path = args.fixture
    if not fixture_path.is_file():
        raise SystemExit(
            f"parity fixture not found: {fixture_path}; "
            "generate it with python/build_parity_fixture.py"
        )
    npz = np.load(fixture_path)
    pixels_np = npz["pixels"].astype(np.float32)    # (B, T, C, H, W)
    actions_np = npz["actions"].astype(np.float32)   # (B, T, A)
    fixture_hash = sha256_file(fixture_path)

    pixels = torch.from_numpy(pixels_np)
    actions = torch.from_numpy(actions_np)
    B, T, C, H, W = pixels.shape

    dump_dir: Path = args.dump_dir
    dump_dir.mkdir(parents=True, exist_ok=True)

    _save_dump(dump_dir / "inputs/pixels.safetensors", pixels_np)
    _save_dump(dump_dir / "inputs/actions.safetensors", actions_np)

    enc_cfg = arch["encoder"]
    pixels_flat = pixels.reshape(B * T, C, H, W)
    cls_out = _encoder_forward(state, pixels_flat, enc_cfg, dump_dir)

    proj_out = _mlp_bn_forward(
        state, "projector", cls_out,
        dump_dir / "projector/output.safetensors",
    )

    ae_out = _action_encoder_forward(
        state, actions,
        dump_dir / "action_encoder/output.safetensors",
    )

    pred_cfg = arch["predictor"]
    context = proj_out.reshape(B, T, -1)
    pred_out = _predictor_forward(state, context, ae_out, pred_cfg, dump_dir)

    pred_flat = pred_out.reshape(B * T, -1)
    _mlp_bn_forward(
        state, "pred_proj", pred_flat,
        dump_dir / "pred_proj/output.safetensors",
    )

    _sigreg_forward(
        proj_out.reshape(B, T, -1).numpy(),
        seed=args.fixture_seed,
        dump_dir=dump_dir,
    )

    meta_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "weights_sha256": weights_sha256 or "skipped",
        "torch_version": torch.__version__,
        "transformers_version": None,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "fixture_seed": int(args.fixture_seed),
        "fixture_hash": fixture_hash,
        "dump_dir": str(dump_dir.resolve()),
        "batch": B,
        "time_steps": T,
    }
    write_json(dump_dir / "meta.json", meta_payload)

    print(f"parity dumps written to {dump_dir}")
    return 0


def _save_dump(path: Path, array: Any) -> None:
    """Write a single numpy array as a Safetensors file with key 'data'."""
    import numpy as np

    arr = np.asarray(array, dtype=np.float32)
    write_safetensors(path, {"data": arr})


def _encoder_forward(
    state: dict[str, Any],
    pixels: Any,
    cfg: dict[str, Any],
    dump_dir: Path,
) -> Any:
    """ViT encoder forward with intermediate dumps.

    pixels: (B*T, C, H, W) torch tensor
    Returns: cls (B*T, D) torch tensor
    """
    import torch
    import torch.nn.functional as F

    patch_size: int = cfg["patch_size"]
    num_heads: int = cfg["num_attention_heads"]
    num_layers: int = cfg["num_hidden_layers"]

    patch_w = state["encoder.embeddings.patch_embeddings.projection.weight"]
    patch_b = state["encoder.embeddings.patch_embeddings.projection.bias"]
    D = patch_w.shape[0]

    x = F.conv2d(pixels, patch_w, patch_b, stride=patch_size)  # (BT, D, H/P, W/P)
    x = x.flatten(2).transpose(1, 2)                            # (BT, num_patches, D)
    _save_dump(dump_dir / "encoder/after_patch_embed.safetensors", x.numpy())

    cls_token = state["encoder.embeddings.cls_token"]           # (1, 1, D)
    BT = x.shape[0]
    x = torch.cat([cls_token.expand(BT, -1, -1), x], dim=1)   # (BT, P+1, D)
    _save_dump(dump_dir / "encoder/after_cls_concat.safetensors", x.numpy())

    pos_embed = state["encoder.embeddings.position_embeddings"]  # (1, P+1, D)
    x = x + pos_embed
    _save_dump(dump_dir / "encoder/after_pos_embed.safetensors", x.numpy())

    blocks_dir = dump_dir / "encoder/blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    head_dim = D // num_heads

    for i in range(num_layers):
        src = f"encoder.encoder.layer.{i}"

        ln1_w = state[f"{src}.layernorm_before.weight"]
        ln1_b = state[f"{src}.layernorm_before.bias"]
        normed = F.layer_norm(x, [D], ln1_w, ln1_b)

        q_w = state[f"{src}.attention.attention.query.weight"]
        k_w = state[f"{src}.attention.attention.key.weight"]
        v_w = state[f"{src}.attention.attention.value.weight"]
        q_b = state[f"{src}.attention.attention.query.bias"]
        k_b = state[f"{src}.attention.attention.key.bias"]
        v_b = state[f"{src}.attention.attention.value.bias"]

        q = F.linear(normed, q_w, q_b)
        k = F.linear(normed, k_w, k_b)
        v = F.linear(normed, v_w, v_b)

        N = q.shape[1]
        q = q.reshape(BT, N, num_heads, head_dim).transpose(1, 2)  # (BT, H, N, d)
        k = k.reshape(BT, N, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(BT, N, num_heads, head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(BT, N, D)

        out_w = state[f"{src}.attention.output.dense.weight"]
        out_b = state[f"{src}.attention.output.dense.bias"]
        x = x + F.linear(attn_out, out_w, out_b)
        _save_dump(blocks_dir / f"{i:02d}_after_attn.safetensors", x.numpy())

        ln2_w = state[f"{src}.layernorm_after.weight"]
        ln2_b = state[f"{src}.layernorm_after.bias"]
        normed2 = F.layer_norm(x, [D], ln2_w, ln2_b)

        fc1_w = state[f"{src}.intermediate.dense.weight"]
        fc1_b = state[f"{src}.intermediate.dense.bias"]
        fc2_w = state[f"{src}.output.dense.weight"]
        fc2_b = state[f"{src}.output.dense.bias"]
        mlp = F.gelu(F.linear(normed2, fc1_w, fc1_b))
        x = x + F.linear(mlp, fc2_w, fc2_b)
        _save_dump(blocks_dir / f"{i:02d}_after_mlp.safetensors", x.numpy())

    fn_w = state["encoder.layernorm.weight"]
    fn_b = state["encoder.layernorm.bias"]
    x = F.layer_norm(x, [D], fn_w, fn_b)
    _save_dump(dump_dir / "encoder/after_final_norm.safetensors", x.numpy())

    cls = x[:, 0, :]
    _save_dump(dump_dir / "encoder/cls.safetensors", cls.numpy())
    return cls


def _mlp_bn_forward(
    state: dict[str, Any],
    prefix: str,
    x: Any,
    out_path: Path,
) -> Any:
    """MLP-with-BatchNorm1d forward (projector / pred_proj) in eval mode.

    Input x: (N, D) torch tensor  →  output (N, output_dim) torch tensor.
    """
    import torch.nn.functional as F

    fc1_w = state[f"{prefix}.net.0.weight"]
    fc1_b = state[f"{prefix}.net.0.bias"]
    bn_w = state[f"{prefix}.net.1.weight"]
    bn_b = state[f"{prefix}.net.1.bias"]
    bn_mean = state[f"{prefix}.net.1.running_mean"]
    bn_var = state[f"{prefix}.net.1.running_var"]
    fc2_w = state[f"{prefix}.net.3.weight"]
    fc2_b = state[f"{prefix}.net.3.bias"]

    out = F.gelu(F.batch_norm(F.linear(x, fc1_w, fc1_b), bn_mean, bn_var, bn_w, bn_b, training=False))
    out = F.linear(out, fc2_w, fc2_b)
    _save_dump(out_path, out.numpy())
    return out


def _action_encoder_forward(
    state: dict[str, Any],
    actions: Any,
    out_path: Path,
) -> Any:
    """Action encoder forward (smoother + 2-layer MLP with SiLU).

    actions: (B, T, A) torch tensor  →  output (B, T, emb_dim) torch tensor.
    """
    import torch.nn.functional as F

    smoother_w = state["action_encoder.patch_embed.weight"]   # (smoothed, input, 1)
    smoother_b = state["action_encoder.patch_embed.bias"]
    fc1_w = state["action_encoder.embed.0.weight"]
    fc1_b = state["action_encoder.embed.0.bias"]
    fc2_w = state["action_encoder.embed.2.weight"]
    fc2_b = state["action_encoder.embed.2.bias"]

    x = F.conv1d(actions.permute(0, 2, 1), smoother_w.squeeze(-1), smoother_b).permute(0, 2, 1)
    out = F.linear(F.silu(F.linear(x, fc1_w, fc1_b)), fc2_w, fc2_b)
    _save_dump(out_path, out.numpy())
    return out


def _predictor_forward(
    state: dict[str, Any],
    context: Any,
    conditioning: Any,
    cfg: dict[str, Any],
    dump_dir: Path,
) -> Any:
    """Autoregressive predictor forward with AdaLN-zero and causal attention.

    context:     (B, T, D) torch tensor (projected encoder embeddings)
    conditioning:(B, T, D) torch tensor (action encoder embeddings)
    Returns:     (B, T, D) torch tensor
    """
    import torch
    import torch.nn.functional as F

    num_layers: int = cfg["depth"]
    num_heads: int = cfg["heads"]
    head_dim: int = cfg["dim_head"]
    inner_dim: int = cfg["attention_inner_dim"]

    pos_embed = state["predictor.pos_embedding"]
    B, T, D = context.shape

    tokens = context + pos_embed[:, :T, :]
    _save_dump(dump_dir / "predictor/after_pos_add.safetensors", tokens.numpy())

    blocks_dir = dump_dir / "predictor/blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    for i in range(num_layers):
        src = f"predictor.transformer.layers.{i}"

        adaln_w = state[f"{src}.adaLN_modulation.1.weight"]
        adaln_b = state[f"{src}.adaLN_modulation.1.bias"]
        mods = F.linear(F.silu(conditioning), adaln_w, adaln_b)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

        normed = F.layer_norm(tokens, [D])
        attn_input = normed * (1.0 + scale_msa) + shift_msa

        attn_norm_w = state[f"{src}.attn.norm.weight"]
        attn_norm_b = state[f"{src}.attn.norm.bias"]
        x = F.layer_norm(attn_input, [D], attn_norm_w, attn_norm_b)

        qkv_w = state[f"{src}.attn.to_qkv.weight"]
        qkv = F.linear(x, qkv_w)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)

        attn_w = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_w = attn_w.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_w = F.softmax(attn_w, dim=-1)
        attn_out = (attn_w @ v).transpose(1, 2).reshape(B, T, inner_dim)

        proj_w = state[f"{src}.attn.to_out.0.weight"]
        proj_b = state[f"{src}.attn.to_out.0.bias"]
        attn_out = F.linear(attn_out, proj_w, proj_b)
        tokens = tokens + gate_msa * attn_out
        _save_dump(blocks_dir / f"{i:02d}_after_attn.safetensors", tokens.numpy())

        normed2 = F.layer_norm(tokens, [D])
        mlp_input = normed2 * (1.0 + scale_mlp) + shift_mlp

        mlp_norm_w = state[f"{src}.mlp.net.0.weight"]
        mlp_norm_b = state[f"{src}.mlp.net.0.bias"]
        x_mlp = F.layer_norm(mlp_input, [D], mlp_norm_w, mlp_norm_b)

        fc1_w = state[f"{src}.mlp.net.1.weight"]
        fc1_b = state[f"{src}.mlp.net.1.bias"]
        fc2_w = state[f"{src}.mlp.net.4.weight"]
        fc2_b = state[f"{src}.mlp.net.4.bias"]
        x_mlp = F.linear(F.gelu(F.linear(x_mlp, fc1_w, fc1_b)), fc2_w, fc2_b)
        tokens = tokens + gate_mlp * x_mlp
        _save_dump(blocks_dir / f"{i:02d}_after_mlp.safetensors", tokens.numpy())

    final_norm_w = state["predictor.transformer.norm.weight"]
    final_norm_b = state["predictor.transformer.norm.bias"]
    output = F.layer_norm(tokens, [D], final_norm_w, final_norm_b)
    _save_dump(dump_dir / "predictor/output.safetensors", output.numpy())
    return output


def _sigreg_forward(
    embeddings: Any,
    *,
    seed: int,
    dump_dir: Path,
) -> None:
    """Compute SIGReg with a numpy-sampled projection and write dumps.

    embeddings: (B, T, D) numpy array (projector outputs)
    Writes:
      sigreg/projection_seed_{seed}.safetensors  (K, D)
      sigreg/empirical_c_s.safetensors           (2, J, K) [0]=cos, [1]=sin
      sigreg/value.safetensors                   (1,) scalar
    """
    import numpy as np

    K = 1024   # DEFAULT_SIGREG_NUM_PROJ
    J = 17     # DEFAULT_SIGREG_KNOTS
    T_MAX = 3.0

    B_T, D = int(embeddings.shape[0] * embeddings.shape[1]), int(embeddings.shape[2])
    flat = embeddings.reshape(B_T, D).astype(np.float32)

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((K, D)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    P = (raw / norms).astype(np.float32)
    _save_dump(dump_dir / f"sigreg/projection_seed_{seed}.safetensors", P)

    projected = flat @ P.T  # (N, K)

    step = T_MAX / (J - 1)
    t_grid = np.linspace(0.0, T_MAX, J, dtype=np.float32)
    phi = np.exp(-0.5 * t_grid ** 2).astype(np.float32)
    trap = np.full(J, step, dtype=np.float32)
    trap[0] = step / 2.0
    trap[-1] = step / 2.0

    arg = t_grid.reshape(J, 1, 1) * projected.T.reshape(1, K, B_T)  # (J, K, N)
    cos_stats = np.cos(arg).mean(axis=-1).astype(np.float32)          # (J, K)
    sin_stats = np.sin(arg).mean(axis=-1).astype(np.float32)          # (J, K)
    empirical_cs = np.stack([cos_stats, sin_stats], axis=0)            # (2, J, K)
    _save_dump(dump_dir / "sigreg/empirical_c_s.safetensors", empirical_cs)

    residual = (cos_stats - phi.reshape(-1, 1)) ** 2 + sin_stats ** 2
    weights = (phi * trap).reshape(-1, 1)
    loss_val = float((residual * weights).sum(axis=0).mean())
    _save_dump(dump_dir / "sigreg/value.safetensors", np.array([loss_val], dtype=np.float32))


def require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}; {hint}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
