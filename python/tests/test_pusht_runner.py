from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).resolve().parents[1] / "pusht_runner.py"
IMAGE_SIZE = 224
RGB_BYTES = IMAGE_SIZE * IMAGE_SIZE * 3


def start_runner() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-u", str(RUNNER), "--mock"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def rpc(proc: subprocess.Popen[str], message: dict[str, Any]) -> dict[str, Any]:
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "runner closed stdout before replying"
    return json.loads(line)


def close_runner(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        rpc(proc, {"id": "close", "method": "close"})
        proc.wait(timeout=5)
    assert proc.returncode == 0


def assert_rgb_payload(result: dict[str, Any]) -> None:
    assert result["obs_shape"] == [IMAGE_SIZE, IMAGE_SIZE, 3]
    assert result["obs_dtype"] == "uint8"
    assert len(base64.b64decode(result["obs"])) == RGB_BYTES


def test_mock_runner_reset_and_step_round_trip() -> None:
    proc = start_runner()
    try:
        response = rpc(
            proc,
            {
                "id": 1,
                "method": "reset",
                "params": {"episode": 17, "seed": 42},
            },
        )
        assert response["ok"] is True
        assert response["id"] == 1
        result = response["result"]
        assert result["state"] == [17.0, 42.0, 0.0, 0.0, 0.0]
        assert result["done"] is False
        assert result["success"] is False
        assert_rgb_payload(result)

        for step in range(1, 6):
            response = rpc(
                proc,
                {
                    "id": f"step-{step}",
                    "method": "step",
                    "params": {"action": [0.25, -0.5]},
                },
            )
            assert response["ok"] is True

        result = response["result"]
        assert result["done"] is True
        assert result["success"] is True
        assert result["reward"] == 1.0
        assert result["state"] == [17.0, 42.0, 5.0, 0.25, -0.5]
        assert_rgb_payload(result)
    finally:
        close_runner(proc)


def test_runner_reports_error_and_stays_alive() -> None:
    proc = start_runner()
    try:
        response = rpc(proc, {"id": "bad", "method": "missing", "params": {}})
        assert response["ok"] is False
        assert response["error"]["code"] == "method_not_found"

        response = rpc(
            proc,
            {
                "id": "reset-after-error",
                "method": "reset",
                "params": {"episode": 1, "seed": 2},
            },
        )
        assert response["ok"] is True
        assert response["result"]["state"][:2] == [1.0, 2.0]
    finally:
        close_runner(proc)
