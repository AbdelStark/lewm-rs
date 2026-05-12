#!/usr/bin/env python3
"""Build the deterministic RFC 0008 parity fixture."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_FIXTURE = Path("tests/fixtures/parity_fixture.npz")
DEFAULT_META = Path("tests/fixtures/parity_fixture.meta.json")
DEFAULT_SEED = 0
PIXELS_SHAPE = (4, 4, 3, 224, 224)
ACTIONS_SHAPE = (4, 4, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the fixed RFC 0008 parity fixture."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIXTURE,
        help=f"Fixture output path. Default: {DEFAULT_FIXTURE}",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=DEFAULT_META,
        help=f"Fixture metadata output path. Default: {DEFAULT_META}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Torch RNG seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing fixture and metadata file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != DEFAULT_SEED:
        raise SystemExit(
            "RFC0008-006 gates fixture regeneration; use seed 0 unless an RFC bump "
            "updates the fixture contract."
        )
    for path in [args.out, args.meta_out]:
        if path.exists() and not args.force:
            raise SystemExit(f"{path} already exists; pass --force to regenerate it")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)

    pixels, actions = build_fixture(seed=args.seed)
    git_short_sha = current_git_short_sha()
    write_fixture(args.out, pixels, actions, args.seed, git_short_sha)
    fixture_hash = blake3_hash(args.out.read_bytes())
    write_meta(args.meta_out, args.out, fixture_hash, args.seed, git_short_sha)

    print(f"wrote {args.out}")
    print(f"wrote {args.meta_out}")
    print(f"fixture_hash={fixture_hash}")


def build_fixture(seed: int) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pixels = torch.rand(PIXELS_SHAPE, generator=generator, dtype=torch.float32)
    pixels = (pixels - 0.5) / 0.5
    actions = torch.randn(ACTIONS_SHAPE, generator=generator, dtype=torch.float32) * 0.5
    return (
        pixels.numpy().astype(np.float32, copy=False),
        actions.numpy().astype(np.float32, copy=False),
    )


def write_fixture(
    out: Path,
    pixels: np.ndarray,
    actions: np.ndarray,
    seed: int,
    git_short_sha: str,
) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            pixels=pixels,
            actions=actions,
            seed=np.array(seed, dtype=np.int32),
            git_short_sha=np.array(git_short_sha, dtype="S40"),
        )
    tmp.replace(out)


def write_meta(
    meta_out: Path,
    fixture_path: Path,
    fixture_hash: str,
    seed: int,
    git_short_sha: str,
) -> None:
    meta: dict[str, Any] = {
        "schema_version": "1.0",
        "fixture_path": str(fixture_path),
        "fixture_seed": seed,
        "fixture_hash": fixture_hash,
        "fixture_hash_algorithm": "blake3",
        "git_short_sha": git_short_sha,
        "pixels": {
            "shape": list(PIXELS_SHAPE),
            "dtype": "float32",
            "normalization": "(x - 0.5) / 0.5",
        },
        "actions": {
            "shape": list(ACTIONS_SHAPE),
            "dtype": "float32",
            "scale": 0.5,
        },
        "generator": {
            "framework": "torch",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "cuda_version": torch.version.cuda,
        },
        "regeneration_policy": "RFC0008-006 requires a minor RFC bump before regeneration.",
    }
    tmp = meta_out.with_suffix(meta_out.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    tmp.replace(meta_out)


def current_git_short_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def blake3_hash(data: bytes) -> str:
    try:
        import blake3
    except ImportError as exc:
        raise SystemExit(
            "python package 'blake3' is required to record fixture_hash"
        ) from exc
    return blake3.blake3(data).hexdigest()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
