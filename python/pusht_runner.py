#!/usr/bin/env python3
"""Line-delimited JSON-RPC sidecar for the PushT simulator."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, TextIO

DEFAULT_ENV_ID = "gym_pusht/PushT-v0"
DEFAULT_IMAGE_SIZE = 224
RGB_CHANNELS = 3


class RpcError(Exception):
    """Protocol-level error returned to the caller without killing the sidecar."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class MockPushTBackend:
    """Deterministic backend used by CI and subprocess round-trip tests."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self.episode = 0
        self.seed: int | None = None
        self.step_count = 0
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0]

    def reset(self, episode: int, seed: int | None) -> dict[str, Any]:
        self.episode = episode
        self.seed = seed
        self.step_count = 0
        self.state = [
            float(episode),
            float(seed if seed is not None else -1),
            0.0,
            0.0,
            0.0,
        ]
        return self._result(reward=None, done=False, success=False)

    def step(self, action: list[float]) -> dict[str, Any]:
        self.step_count += 1
        success = self.step_count >= 5
        self.state = [
            float(self.episode),
            float(self.seed if self.seed is not None else -1),
            float(self.step_count),
            action[0],
            action[1],
        ]
        return self._result(
            reward=min(float(self.step_count) / 5.0, 1.0),
            done=success,
            success=success,
        )

    def close(self) -> None:
        return None

    def _result(
        self,
        *,
        reward: float | None,
        done: bool,
        success: bool,
    ) -> dict[str, Any]:
        value = (self.episode + self.step_count) % 256
        frame = bytes([value]) * rgb_byte_len(self.image_size)
        result: dict[str, Any] = {
            "obs": encode_rgb_bytes(frame, self.image_size),
            "obs_shape": [self.image_size, self.image_size, RGB_CHANNELS],
            "obs_dtype": "uint8",
            "state": self.state,
            "episode": self.episode,
            "step": self.step_count,
            "done": done,
            "success": success,
        }
        if reward is not None:
            result["reward"] = reward
        return result


class GymPushTBackend:
    """`gym_pusht` backend used for real evaluation runs."""

    def __init__(self, env_id: str, image_size: int) -> None:
        try:
            import gym_pusht  # noqa: F401
            import gymnasium as gym
            import numpy as np
        except ImportError as exc:
            raise RpcError(
                "dependency_missing",
                "install the pinned simulator extra with `uv sync --extra sim`",
            ) from exc

        self.env_id = env_id
        self.image_size = image_size
        self.np = np
        self.env = gym.make(
            env_id,
            obs_type="state",
            render_mode="rgb_array",
            visualization_width=image_size,
            visualization_height=image_size,
        )
        self.episode = 0
        self.step_count = 0
        self.state: list[float] = []

    def reset(self, episode: int, seed: int | None) -> dict[str, Any]:
        self.episode = episode
        self.step_count = 0
        effective_seed = None if seed is None else seed + episode
        observation, info = self.env.reset(seed=effective_seed)
        self.state = observation_to_state(observation)
        return self._result(info=info, reward=None, done=False)

    def step(self, action: list[float]) -> dict[str, Any]:
        action_array = self.np.asarray(action, dtype=self.np.float32)
        observation, reward, terminated, truncated, info = self.env.step(action_array)
        self.step_count += 1
        self.state = observation_to_state(observation)
        return self._result(
            info=info,
            reward=float(reward),
            done=bool(terminated or truncated),
        )

    def close(self) -> None:
        self.env.close()

    def _result(
        self,
        *,
        info: dict[str, Any],
        reward: float | None,
        done: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "obs": encode_rgb_frame(self.env.render(), self.image_size),
            "obs_shape": [self.image_size, self.image_size, RGB_CHANNELS],
            "obs_dtype": "uint8",
            "state": self.state,
            "episode": self.episode,
            "step": self.step_count,
            "done": done,
            "success": bool(info.get("is_success", False)),
        }
        if reward is not None:
            result["reward"] = reward
        return result


@dataclass
class JsonRpcServer:
    """Long-lived line-delimited JSON-RPC loop."""

    backend: MockPushTBackend | GymPushTBackend
    stdin: TextIO
    stdout: TextIO
    stderr: TextIO

    def serve_forever(self) -> int:
        for raw_line in self.stdin:
            line = raw_line.strip()
            if not line:
                continue
            response, should_stop = self.handle_line(line)
            self.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            self.stdout.flush()
            if should_stop:
                break
        self.backend.close()
        return 0

    def handle_line(self, line: str) -> tuple[dict[str, Any], bool]:
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise RpcError("invalid_request", "request must be a JSON object")
            request_id = request.get("id")
            result, should_stop = self.dispatch(request)
            return {"id": request_id, "ok": True, "result": result}, should_stop
        except RpcError as exc:
            return {
                "id": request_id,
                "ok": False,
                "error": {"code": exc.code, "message": exc.message},
            }, False
        except json.JSONDecodeError as exc:
            return {
                "id": request_id,
                "ok": False,
                "error": {
                    "code": "invalid_json",
                    "message": f"invalid JSON at byte {exc.pos}",
                },
            }, False
        except Exception as exc:  # pragma: no cover - defensive process guard.
            self.stderr.write(f"internal pusht_runner error: {exc}\n")
            self.stderr.flush()
            return {
                "id": request_id,
                "ok": False,
                "error": {"code": "internal_error", "message": str(exc)},
            }, False

    def dispatch(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str):
            raise RpcError("invalid_request", "method must be a string")
        if not isinstance(params, dict):
            raise RpcError("invalid_request", "params must be a JSON object")

        if method == "reset":
            result = self.backend.reset(
                episode=required_int(params, "episode"),
                seed=optional_int(params, "seed"),
            )
            return result, False
        if method == "step":
            return self.backend.step(required_action(params, "action")), False
        if method == "close":
            return {"closed": True}, True
        raise RpcError("method_not_found", f"unsupported method: {method}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve PushT reset/step calls over line-delimited JSON-RPC."
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic mock backend instead of gym_pusht.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive")

    use_mock = args.mock or os.environ.get("LEWM_PUSHT_RUNNER_MOCK") == "1"
    backend: MockPushTBackend | GymPushTBackend
    if use_mock:
        backend = MockPushTBackend(args.image_size)
    else:
        backend = GymPushTBackend(args.env_id, args.image_size)

    return JsonRpcServer(
        backend=backend,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    ).serve_forever()


def required_int(params: dict[str, Any], key: str) -> int:
    if key not in params:
        raise RpcError("invalid_params", f"missing integer param: {key}")
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RpcError("invalid_params", f"{key} must be an integer")
    return value


def optional_int(params: dict[str, Any], key: str) -> int | None:
    if key not in params or params[key] is None:
        return None
    return required_int(params, key)


def required_action(params: dict[str, Any], key: str) -> list[float]:
    value = params.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise RpcError("invalid_params", f"{key} must be a two-element list")
    action = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise RpcError("invalid_params", f"{key} values must be numeric")
        action.append(float(item))
    return action


def observation_to_state(observation: Any) -> list[float]:
    if hasattr(observation, "tolist"):
        observation = observation.tolist()
    if not isinstance(observation, list):
        raise RpcError("invalid_observation", "PushT state observation must be list-like")
    return [float(item) for item in observation]


def encode_rgb_frame(frame: Any, image_size: int) -> str:
    shape = getattr(frame, "shape", None)
    expected_shape = (image_size, image_size, RGB_CHANNELS)
    if tuple(shape) != expected_shape:
        raise RpcError(
            "invalid_observation",
            f"rendered frame shape must be {expected_shape}, got {shape}",
        )

    dtype = str(getattr(frame, "dtype", ""))
    if dtype != "uint8":
        if not hasattr(frame, "astype"):
            raise RpcError("invalid_observation", "rendered frame must be uint8 RGB")
        frame = frame.astype("uint8", copy=False)

    return encode_rgb_bytes(frame.tobytes(), image_size)


def encode_rgb_bytes(data: bytes, image_size: int) -> str:
    expected_len = rgb_byte_len(image_size)
    if len(data) != expected_len:
        raise RpcError(
            "invalid_observation",
            f"RGB frame must be {expected_len} bytes, got {len(data)}",
        )
    return base64.b64encode(data).decode("ascii")


def rgb_byte_len(image_size: int) -> int:
    return image_size * image_size * RGB_CHANNELS


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
