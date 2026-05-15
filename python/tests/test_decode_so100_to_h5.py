from __future__ import annotations

import json
import sys
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decode_so100_to_h5 import DecodeConfig, decode_dataset, validate_hdf5

FPS_NATIVE = 30
FPS_TARGET = 10
SIZE = 16
EPISODE_LENGTH = 6
EPISODES = 2


def test_decode_so100_to_h5_schema_correct(tmp_path: Path) -> None:
    src = write_lerobot_fixture(tmp_path / "src")
    out = tmp_path / "so100.h5"

    content_hash = run_decode(src, out, workers=2)

    with h5py.File(out, "r") as handle:
        assert handle.attrs["schema_version"] == "1.0"
        assert handle.attrs["fps_native"] == FPS_NATIVE
        assert handle.attrs["fps_target"] == FPS_TARGET
        assert handle.attrs["size"] == SIZE
        assert handle.attrs["interp"] == "bilinear"
        assert handle.attrs["source_dataset"] == "lerobot/svla_so100_pickplace"
        assert handle.attrs["source_revision"] == "test-revision"
        assert handle.attrs["content_hash"] == content_hash
        assert json.loads(handle.attrs["camera_views"]) == ["top", "wrist"]
        assert (
            json.loads(handle.attrs["decode_tool_versions"])["pyav"] == av.__version__
        )

        assert handle["episode_index"].dtype == np.dtype("int32")
        assert handle["timestep"].dtype == np.dtype("int32")
        assert handle["action"].dtype == np.dtype("float32")
        assert handle["joint_pos"].dtype == np.dtype("float32")
        assert handle["observation/pixels_top"].dtype == np.dtype("uint8")
        assert handle["observation/pixels_wrist"].dtype == np.dtype("uint8")
        assert handle["observation/pixels_top"].chunks == (3, SIZE, SIZE, 3)
        assert handle["observation/pixels_wrist"].shape == (3, SIZE, SIZE, 3)
        np.testing.assert_array_equal(handle["episode_index"][...], [0, 0, 1])
        np.testing.assert_array_equal(handle["timestep"][...], [0, 1, 0])

    validate_hdf5(out)


def test_decode_so100_to_h5_deterministic(tmp_path: Path) -> None:
    src = write_lerobot_fixture(tmp_path / "src", include_nan=False)
    out_a = tmp_path / "a.h5"
    out_b = tmp_path / "b.h5"

    hash_a = run_decode(src, out_a, workers=1)
    hash_b = run_decode(src, out_b, workers=2)

    assert hash_a == hash_b
    with h5py.File(out_a, "r") as left, h5py.File(out_b, "r") as right:
        for name in [
            "episode_index",
            "timestep",
            "action",
            "joint_pos",
            "observation/pixels_top",
            "observation/pixels_wrist",
        ]:
            np.testing.assert_array_equal(left[name][...], right[name][...])
        assert left.attrs["content_hash"] == right.attrs["content_hash"]


def test_decode_so100_drops_nan_actions(tmp_path: Path) -> None:
    src = write_lerobot_fixture(tmp_path / "src", include_nan=True)
    out = tmp_path / "so100.h5"

    run_decode(src, out)

    with h5py.File(out, "r") as handle:
        assert handle["action"].shape == (3, 6)
        assert not np.isnan(handle["action"][...]).any()
        np.testing.assert_array_equal(handle["episode_index"][...], [0, 0, 1])
        np.testing.assert_array_equal(handle["timestep"][...], [0, 1, 0])


def run_decode(src: Path, out: Path, workers: int = 2) -> str:
    return decode_dataset(
        DecodeConfig(
            src=src,
            out=out,
            fps_target=FPS_TARGET,
            size=SIZE,
            interp="bilinear",
            camera_views=("top", "wrist"),
            workers=workers,
            validate=True,
            source_dataset="lerobot/svla_so100_pickplace",
            source_revision="test-revision",
        )
    )


def write_lerobot_fixture(root: Path, include_nan: bool = True) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    for view in ("top", "wrist"):
        (root / "videos" / f"observation.images.{view}" / "chunk-000").mkdir(
            parents=True
        )

    write_info(root / "meta" / "info.json")
    write_data(
        root / "data" / "chunk-000" / "file-000.parquet", include_nan=include_nan
    )
    write_episodes(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    write_video(
        root / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4", 20
    )
    write_video(
        root / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4", 80
    )
    return root


def write_info(path: Path) -> None:
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so100",
        "total_episodes": EPISODES,
        "total_frames": EPISODES * EPISODE_LENGTH,
        "chunks_size": 1000,
        "fps": FPS_NATIVE,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [6], "fps": float(FPS_NATIVE)},
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "fps": float(FPS_NATIVE),
            },
            "observation.images.top": {"dtype": "video", "shape": [48, 64, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [48, 64, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    path.write_text(json.dumps(info, indent=2) + "\n")


def write_data(path: Path, include_nan: bool) -> None:
    actions = []
    joint_pos = []
    timestamps = []
    frame_indices = []
    episode_indices = []
    indices = []
    task_indices = []

    for episode in range(EPISODES):
        for frame in range(EPISODE_LENGTH):
            action = np.full(6, episode * 10 + frame, dtype=np.float32)
            if include_nan and episode == 1 and frame == 3:
                action[2] = np.nan
            actions.append(action.tolist())
            joint_pos.append((action + 0.5).tolist())
            timestamps.append(frame / FPS_NATIVE)
            frame_indices.append(frame)
            episode_indices.append(episode)
            indices.append(episode * EPISODE_LENGTH + frame)
            task_indices.append(0)

    table = pa.table(
        {
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            "observation.state": pa.array(joint_pos, type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def write_episodes(path: Path) -> None:
    starts = [0.0, EPISODE_LENGTH / FPS_NATIVE]
    rows = {
        "episode_index": pa.array([0, 1], type=pa.int64()),
        "data/chunk_index": pa.array([0, 0], type=pa.int64()),
        "data/file_index": pa.array([0, 0], type=pa.int64()),
        "dataset_from_index": pa.array([0, EPISODE_LENGTH], type=pa.int64()),
        "dataset_to_index": pa.array(
            [EPISODE_LENGTH, EPISODE_LENGTH * 2], type=pa.int64()
        ),
        "length": pa.array([EPISODE_LENGTH, EPISODE_LENGTH], type=pa.int64()),
    }
    for view in ("top", "wrist"):
        prefix = f"videos/observation.images.{view}"
        rows[f"{prefix}/chunk_index"] = pa.array([0, 0], type=pa.int64())
        rows[f"{prefix}/file_index"] = pa.array([0, 0], type=pa.int64())
        rows[f"{prefix}/from_timestamp"] = pa.array(starts, type=pa.float64())
        rows[f"{prefix}/to_timestamp"] = pa.array(
            [start + EPISODE_LENGTH / FPS_NATIVE for start in starts], type=pa.float64()
        )
    pq.write_table(pa.table(rows), path)


def write_video(path: Path, base: int) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS_NATIVE)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for frame_index in range(EPISODES * EPISODE_LENGTH):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:, :, 0] = (base + frame_index * 3) % 255
            pixels[:, :, 1] = (base + frame_index * 5) % 255
            pixels[:, :, 2] = (base + frame_index * 7) % 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
