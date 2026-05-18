#!/usr/bin/env python3
"""Verify LeWM ONNX export artifacts with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised by users without the extra.
    ort = None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="ONNX export directory. May be the root export or a single variant directory.",
    )
    parser.add_argument(
        "--variant",
        choices=("all", "onnxruntime", "tract-compat"),
        default="all",
        help="Variant to verify. Default: all variants discoverable under --dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if ort is None:
        raise SystemExit("onnxruntime is required; run with `uv run --with onnxruntime`")

    root = args.dir
    metadata = load_metadata(root)
    config = metadata["config"]
    variants = discover_variants(root, metadata, args.variant)
    if not variants:
        raise SystemExit(f"no ONNX variants found under {root}")

    for name, directory in variants:
        verify_variant(name, directory, config)
    return 0


def load_metadata(root: Path) -> dict[str, Any]:
    path = root / "onnx_export.json"
    if not path.exists():
        raise SystemExit(f"missing ONNX export metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def discover_variants(
    root: Path,
    metadata: dict[str, Any],
    requested: str,
) -> list[tuple[str, Path]]:
    variants = metadata.get("variants") or {}
    found: list[tuple[str, Path]] = []

    if variants:
        for name in sorted(variants):
            if requested != "all" and name != requested:
                continue
            directory = root / name
            if (directory / "encoder.onnx").exists() and (directory / "predictor.onnx").exists():
                found.append((name, directory))
        return found

    name = requested if requested != "all" else "single"
    if (root / "encoder.onnx").exists() and (root / "predictor.onnx").exists():
        return [(name, root)]
    return []


def verify_variant(name: str, directory: Path, config: dict[str, Any]) -> None:
    image_size = int(config["image_size"])
    history_size = int(config["history_size"])
    latent_dim = int(config["latent_dim"])
    action_dim = int(config["action_dim"])
    batches = (1, 2) if name == "onnxruntime" else (1,)

    encoder = ort.InferenceSession(str(directory / "encoder.onnx"))
    predictor = ort.InferenceSession(str(directory / "predictor.onnx"))

    for batch in batches:
        pixels = np.zeros((batch, 3, image_size, image_size), dtype=np.float32)
        embedding = encoder.run(None, {"pixels": pixels})[0]
        require_shape(
            embedding,
            (batch, latent_dim),
            f"{name} encoder output batch={batch}",
        )

        history = np.zeros((batch, history_size, latent_dim), dtype=np.float32)
        actions = np.zeros((batch, history_size, action_dim), dtype=np.float32)
        predicted = predictor.run(None, {"history": history, "actions": actions})[0]
        require_shape(
            predicted,
            (batch, history_size, latent_dim),
            f"{name} predictor output batch={batch}",
        )

    print(
        "onnx verify: "
        f"variant={name} ok=true batches={','.join(str(batch) for batch in batches)} "
        f"encoder_shape={(batches[-1], latent_dim)} "
        f"predictor_shape={(batches[-1], history_size, latent_dim)} "
        f"action_dim={action_dim}"
    )


def require_shape(actual: np.ndarray, expected: tuple[int, ...], label: str) -> None:
    if tuple(actual.shape) != expected:
        raise SystemExit(f"{label}: shape {tuple(actual.shape)} != expected {expected}")


if __name__ == "__main__":
    raise SystemExit(main())
