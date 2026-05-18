#!/usr/bin/env python3
"""Validate a PushT bounded-core checkpoint before SO-100 warm-start launch."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("configs/pusht.toml")
EXPECTED_KIND = "lewm-rs-pusht-bounded-module-lewm-record"
EXPECTED_SCHEMA_VERSION = "1.1.0"
IMAGE_CHANNELS = 3


class WarmstartSourceError(RuntimeError):
    """Raised when the warm-start source checkpoint is not launch-compatible."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Downloaded PushT .mpk path")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"PushT config used to compute the bounded-core parameter count ({DEFAULT_CONFIG})",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def load_json(path: Path) -> dict[str, Any]:
    if path.suffix != ".mpk":
        raise WarmstartSourceError(f"{path}: warm-start source must end in .mpk")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WarmstartSourceError(f"{path}: warm-start source does not exist") from exc
    except UnicodeDecodeError as exc:
        raise WarmstartSourceError(
            f"{path}: expected current bounded-core JSON .mpk; full Burn/Jepa NamedMpk "
            "records are not supported by the current bounded-core SO-100 warm-start path"
        ) from exc
    except json.JSONDecodeError as exc:
        raise WarmstartSourceError(
            f"{path}: invalid bounded-core JSON .mpk: {exc}; full Burn/Jepa NamedMpk records "
            "are not supported by the current bounded-core SO-100 warm-start path"
        ) from exc
    if not isinstance(payload, dict):
        raise WarmstartSourceError(f"{path}: record root must be a JSON object")
    return payload


def load_config(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WarmstartSourceError(f"{path}: config does not exist") from exc
    except tomllib.TOMLDecodeError as exc:
        raise WarmstartSourceError(f"{path}: invalid TOML: {exc}") from exc


def expected_bounded_param_count(config: dict[str, Any]) -> int:
    try:
        model = config["model"]
        hidden_dim = int(model["predictor"]["hidden_dim"])
        action_dim = int(model["action_encoder"]["input_dim"])
        action_emb_dim = int(model["action_encoder"]["emb_dim"])
        history_size = int(model["history_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WarmstartSourceError(
            "config must define model.history_size, model.predictor.hidden_dim, "
            "model.action_encoder.input_dim, and model.action_encoder.emb_dim"
        ) from exc

    if min(hidden_dim, action_dim, action_emb_dim, history_size) <= 0:
        raise WarmstartSourceError("config model dimensions must be positive")

    encoder_params = (
        hidden_dim
        + hidden_dim
        + hidden_dim
        + (hidden_dim * IMAGE_CHANNELS)
        + hidden_dim
    )
    action_params = action_emb_dim + (action_emb_dim * action_dim)
    predictor_params = hidden_dim + (hidden_dim * history_size) + (hidden_dim * action_emb_dim)
    projector_params = hidden_dim + hidden_dim
    pred_proj_params = hidden_dim + hidden_dim
    return encoder_params + action_params + predictor_params + projector_params + pred_proj_params


def require_finite_number_list(
    payload: dict[str, Any],
    key: str,
    expected_len: int,
    path: Path,
) -> None:
    values = payload.get(key)
    if not isinstance(values, list):
        raise WarmstartSourceError(f"{path}: {key} must be a list")
    if len(values) != expected_len:
        raise WarmstartSourceError(
            f"{path}: {key} length {len(values)} does not match expected bounded-core "
            f"parameter count {expected_len}"
        )
    for index, value in enumerate(values):
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise WarmstartSourceError(f"{path}: {key}[{index}] must be a finite number")


def validate_record(path: Path, payload: dict[str, Any], expected_params: int) -> None:
    schema_version = payload.get("schema_version")
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise WarmstartSourceError(
            f"{path}: schema_version must be {EXPECTED_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    kind = payload.get("kind")
    if kind != EXPECTED_KIND:
        raise WarmstartSourceError(f"{path}: kind must be {EXPECTED_KIND!r}, got {kind!r}")

    step = payload.get("step")
    if not isinstance(step, int) or step <= 0:
        raise WarmstartSourceError(f"{path}: step must be a positive integer")

    require_finite_number_list(payload, "params", expected_params, path)

    adamw_params = payload.get("adamw_params", [])
    if not isinstance(adamw_params, list):
        raise WarmstartSourceError(f"{path}: adamw_params must be a list when present")
    if adamw_params and len(adamw_params) != expected_params:
        raise WarmstartSourceError(
            f"{path}: adamw_params length {len(adamw_params)} does not match expected "
            f"bounded-core parameter count {expected_params}"
        )


def main() -> int:
    args = parse_args()
    source_path = resolve_path(args.path)
    config_path = resolve_path(args.config)
    try:
        payload = load_json(source_path)
        expected_params = expected_bounded_param_count(load_config(config_path))
        validate_record(source_path, payload, expected_params)
    except WarmstartSourceError as exc:
        print(f"check_warmstart_source.py: {exc}", file=sys.stderr)
        return 1

    print(
        "warm-start source ok: "
        f"path={source_path} step={payload['step']} params={expected_params}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
