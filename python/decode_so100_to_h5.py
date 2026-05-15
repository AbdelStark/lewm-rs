#!/usr/bin/env python3
"""Decode the LeRobot SO-100 dataset into the RFC 0012 HDF5 mirror."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import av
import blake3
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SOURCE_DATASET = "lerobot/svla_so100_pickplace"
SCHEMA_VERSION = "1.0"
ACTION_DIM = 6
RGB_CHANNELS = 3
DEFAULT_CAMERA_VIEWS = ("top", "wrist")


@dataclass(frozen=True)
class DecodeConfig:
    src: Path
    out: Path
    fps_target: int
    size: int
    interp: str
    camera_views: tuple[str, ...]
    workers: int
    validate: bool
    source_dataset: str
    source_revision: str


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    length: int
    data_path: Path
    video_paths: dict[str, Path]
    video_starts: dict[str, float]


@dataclass(frozen=True)
class EpisodeArrays:
    episode_index: int
    episode: np.ndarray
    timestep: np.ndarray
    action: np.ndarray
    joint_pos: np.ndarray
    pixels: dict[str, np.ndarray]


@dataclass
class DataTable:
    action: np.ndarray
    joint_pos: np.ndarray
    timestamp: np.ndarray
    frame_index: np.ndarray
    episode_index: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode LeRobot SO-100 Parquet + MP4 data into RFC 0012 HDF5."
    )
    parser.add_argument(
        "--src", type=Path, required=True, help="Downloaded dataset root."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Destination HDF5 path."
    )
    parser.add_argument("--fps-target", type=int, default=10, help="Output FPS.")
    parser.add_argument(
        "--size", type=int, default=224, help="Output square image size."
    )
    parser.add_argument(
        "--interp",
        choices=("bilinear", "bicubic"),
        default="bilinear",
        help="Pillow resize interpolation.",
    )
    parser.add_argument(
        "--camera-views",
        nargs="+",
        choices=DEFAULT_CAMERA_VIEWS,
        default=list(DEFAULT_CAMERA_VIEWS),
        help="Camera views to write.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel decode workers."
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate output after write."
    )
    parser.add_argument(
        "--source-dataset",
        default=SOURCE_DATASET,
        help=f"Source dataset id. Default: {SOURCE_DATASET}",
    )
    parser.add_argument(
        "--source-revision",
        default="unknown",
        help="Pinned source dataset revision SHA recorded in HDF5 attrs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = DecodeConfig(
        src=args.src,
        out=args.out,
        fps_target=args.fps_target,
        size=args.size,
        interp=args.interp,
        camera_views=tuple(args.camera_views),
        workers=args.workers,
        validate=args.validate,
        source_dataset=args.source_dataset,
        source_revision=args.source_revision,
    )
    decode_dataset(config)
    return 0


def decode_dataset(config: DecodeConfig) -> str:
    info = read_json(config.src / "meta" / "info.json")
    fps_native = int(round(float(info.get("fps", 30))))
    validate_config(config, fps_native)

    episodes = discover_episodes(config.src, info, config.camera_views)
    tables = DataCache()

    if config.workers <= 1 or len(episodes) <= 1:
        decoded = [
            decode_episode(config, info, fps_native, episode, tables)
            for episode in episodes
        ]
    else:
        decoded = []
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = [
                pool.submit(decode_episode, config, info, fps_native, episode, tables)
                for episode in episodes
            ]
            for future in as_completed(futures):
                decoded.append(future.result())

    decoded.sort(key=lambda item: item.episode_index)
    arrays = concatenate(decoded, config.camera_views)
    content_hash = compute_content_hash(arrays, config.camera_views)
    write_hdf5_atomic(config, fps_native, arrays, content_hash)

    if config.validate:
        validate_hdf5(config.out, config.camera_views)

    return content_hash


def validate_config(config: DecodeConfig, fps_native: int) -> None:
    if not config.src.exists():
        raise SystemExit(f"--src does not exist: {config.src}")
    if config.fps_target <= 0:
        raise SystemExit("--fps-target must be positive")
    if fps_native % config.fps_target != 0:
        raise SystemExit(
            f"fps_target={config.fps_target} must divide native fps={fps_native}"
        )
    if config.size <= 0:
        raise SystemExit("--size must be positive")
    if config.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not config.camera_views:
        raise SystemExit("--camera-views must include at least one view")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing required metadata file: {path}") from exc


class DataCache:
    def __init__(self) -> None:
        self._tables: dict[Path, DataTable] = {}
        self._lock = Lock()

    def table(self, path: Path) -> DataTable:
        with self._lock:
            if path not in self._tables:
                self._tables[path] = read_data_table(path)
            return self._tables[path]


def read_data_table(path: Path) -> DataTable:
    if not path.exists():
        raise FileNotFoundError(f"missing data parquet: {path}")
    table = pq.read_table(path)
    data = table.to_pydict()
    return DataTable(
        action=np.asarray(data["action"], dtype=np.float32),
        joint_pos=np.asarray(data["observation.state"], dtype=np.float32),
        timestamp=np.asarray(data["timestamp"], dtype=np.float64),
        frame_index=np.asarray(data["frame_index"], dtype=np.int64),
        episode_index=np.asarray(data["episode_index"], dtype=np.int64),
    )


def discover_episodes(
    root: Path, info: dict[str, Any], camera_views: tuple[str, ...]
) -> list[EpisodeSpec]:
    rows = read_episode_rows(root)
    if rows:
        episodes = [
            episode_from_meta_row(root, info, row, camera_views) for row in rows
        ]
    else:
        episodes = infer_episodes_from_data(root, info, camera_views)

    episodes.sort(key=lambda episode: episode.episode_index)
    if not episodes:
        raise SystemExit("no SO-100 episodes found under --src")
    return episodes


def read_episode_rows(root: Path) -> list[dict[str, Any]]:
    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.exists():
        rows = []
        for line in jsonl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def episode_from_meta_row(
    root: Path,
    info: dict[str, Any],
    row: dict[str, Any],
    camera_views: tuple[str, ...],
) -> EpisodeSpec:
    episode_index = int(row["episode_index"])
    length = int(
        row.get("length")
        or row.get("dataset_to_index", 0) - row.get("dataset_from_index", 0)
    )
    data_path = root / render_data_path(root, info, row, episode_index)
    video_paths: dict[str, Path] = {}
    video_starts: dict[str, float] = {}

    for view in camera_views:
        video_key = video_feature_key(view)
        video_paths[view] = root / render_video_path(
            root, info, row, episode_index, video_key
        )
        video_starts[view] = float(
            row.get(f"videos/{video_key}/from_timestamp", 0.0) or 0.0
        )

    return EpisodeSpec(
        episode_index=episode_index,
        length=length,
        data_path=data_path,
        video_paths=video_paths,
        video_starts=video_starts,
    )


def infer_episodes_from_data(
    root: Path, info: dict[str, Any], camera_views: tuple[str, ...]
) -> list[EpisodeSpec]:
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    if not data_paths:
        raise SystemExit("no data parquet files found under --src/data")

    episodes: list[EpisodeSpec] = []
    for data_path in data_paths:
        table = read_data_table(data_path)
        for episode_index in sorted(set(table.episode_index.tolist())):
            mask = table.episode_index == episode_index
            row = {
                "episode_index": int(episode_index),
                "length": int(mask.sum()),
                "data/chunk_index": 0,
                "data/file_index": 0,
            }
            video_paths = {
                view: root
                / render_video_path(
                    root, info, row, int(episode_index), video_feature_key(view)
                )
                for view in camera_views
            }
            episodes.append(
                EpisodeSpec(
                    episode_index=int(episode_index),
                    length=int(mask.sum()),
                    data_path=data_path,
                    video_paths=video_paths,
                    video_starts={view: 0.0 for view in camera_views},
                )
            )
    return episodes


def render_data_path(
    root: Path, info: dict[str, Any], row: dict[str, Any], episode_index: int
) -> Path:
    template = info.get(
        "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    values = format_values(info, row, episode_index)
    rendered = safe_format(template, values)
    candidates = [
        rendered,
        f"data/chunk-{values['chunk_index']:03d}/file-{values['file_index']:03d}.parquet",
        f"data/chunk-{values['episode_chunk']:03d}/episode_{episode_index:06d}.parquet",
    ]
    return Path(first_existing_or_first(candidates, root))


def render_video_path(
    root: Path,
    info: dict[str, Any],
    row: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    values = format_values(info, row, episode_index)
    view_prefix = f"videos/{video_key}"
    values["video_key"] = video_key
    values["chunk_index"] = int(
        row.get(f"{view_prefix}/chunk_index", values["chunk_index"])
    )
    values["file_index"] = int(
        row.get(f"{view_prefix}/file_index", values["file_index"])
    )

    template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    rendered = safe_format(template, values)
    candidates = [
        rendered,
        f"videos/{video_key}/chunk-{values['chunk_index']:03d}/file-{values['file_index']:03d}.mp4",
        f"videos/chunk-{values['episode_chunk']:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        f"videos/{video_key}/episode_{episode_index:06d}.mp4",
    ]
    return Path(first_existing_or_first(candidates, root))


def first_existing_or_first(candidates: Iterable[str], base: Path) -> str:
    candidates = list(candidates)
    for candidate in candidates:
        if (base / candidate).exists():
            return candidate
    return candidates[0]


def format_values(
    info: dict[str, Any], row: dict[str, Any], episode_index: int
) -> dict[str, int]:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    return {
        "episode_index": episode_index,
        "episode_chunk": int(row.get("episode_chunk", episode_chunk)),
        "chunk_index": int(
            row.get("data/chunk_index", row.get("chunk_index", episode_chunk))
        ),
        "file_index": int(
            row.get("data/file_index", row.get("file_index", episode_index))
        ),
    }


def safe_format(template: str, values: dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError:
        return template


def video_feature_key(view: str) -> str:
    return f"observation.images.{view}"


def hdf5_pixels_name(view: str) -> str:
    return f"pixels_{view}"


def decode_episode(
    config: DecodeConfig,
    info: dict[str, Any],
    fps_native: int,
    episode: EpisodeSpec,
    tables: DataCache,
) -> EpisodeArrays:
    table = tables.table(episode.data_path)
    mask = table.episode_index == episode.episode_index
    if not mask.any():
        raise RuntimeError(
            f"episode {episode.episode_index} has no rows in {episode.data_path}"
        )

    action = table.action[mask]
    joint_pos = table.joint_pos[mask]
    timestamp = table.timestamp[mask]
    frame_index = table.frame_index[mask]

    stride = fps_native // config.fps_target
    keep = (frame_index % stride) == 0
    keep &= ~np.isnan(action).any(axis=1)

    if not keep.any():
        raise RuntimeError(
            f"episode {episode.episode_index} has no valid frames after filtering"
        )

    kept_action = action[keep].astype(np.float32, copy=False)
    kept_joint_pos = joint_pos[keep].astype(np.float32, copy=False)
    kept_episode = np.full(kept_action.shape[0], episode.episode_index, dtype=np.int32)
    kept_timestep = (frame_index[keep] // stride).astype(np.int32, copy=False)
    kept_frame_index = frame_index[keep]
    kept_timestamp = timestamp[keep]

    pixels = {
        view: decode_view_frames(
            episode.video_paths[view],
            video_start=episode.video_starts[view],
            local_frame_indices=kept_frame_index,
            local_timestamps=kept_timestamp,
            fps_native=fps_native,
            size=config.size,
            interp=config.interp,
        )
        for view in config.camera_views
    }

    return EpisodeArrays(
        episode_index=episode.episode_index,
        episode=kept_episode,
        timestep=kept_timestep,
        action=kept_action,
        joint_pos=kept_joint_pos,
        pixels=pixels,
    )


def decode_view_frames(
    video_path: Path,
    video_start: float,
    local_frame_indices: np.ndarray,
    local_timestamps: np.ndarray,
    fps_native: int,
    size: int,
    interp: str,
) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(f"missing video: {video_path}")

    target_by_index = {
        int(frame): pos for pos, frame in enumerate(local_frame_indices.tolist())
    }
    output = np.empty((len(target_by_index), size, size, RGB_CHANNELS), dtype=np.uint8)
    found: set[int] = set()
    min_target = min(target_by_index)
    max_target = max(target_by_index)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if video_start > 0 and stream.time_base is not None:
            seek_seconds = max(0.0, video_start - 1.0)
            with contextlib.suppress(av.FFmpegError):
                container.seek(
                    int(seek_seconds / float(stream.time_base)),
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )

        decoded_without_time = 0
        for frame in container.decode(stream):
            local_index = frame_local_index(frame, stream, video_start, fps_native)
            if local_index is None:
                local_index = decoded_without_time
                decoded_without_time += 1

            if local_index < min_target:
                continue
            if local_index > max_target and len(found) == len(target_by_index):
                break

            pos = target_by_index.get(local_index)
            if pos is None or local_index in found:
                continue

            output[pos] = resize_frame(frame, size, interp)
            found.add(local_index)
            if len(found) == len(target_by_index):
                break

    if len(found) != len(target_by_index):
        missing = sorted(set(target_by_index) - found)
        fallback = decode_by_timestamp(
            video_path, video_start, local_timestamps, fps_native, size, interp
        )
        if fallback is not None:
            return fallback
        raise RuntimeError(
            f"{video_path} missing {len(missing)} selected frames; first missing={missing[:5]}"
        )

    return output


def frame_local_index(
    frame: av.VideoFrame,
    stream: av.video.stream.VideoStream,
    video_start: float,
    fps_native: int,
) -> int | None:
    if frame.time is not None:
        seconds = float(frame.time)
    elif frame.pts is not None and stream.time_base is not None:
        seconds = float(frame.pts * stream.time_base)
    else:
        return None
    return int(round((seconds - video_start) * fps_native))


def decode_by_timestamp(
    video_path: Path,
    video_start: float,
    local_timestamps: np.ndarray,
    fps_native: int,
    size: int,
    interp: str,
) -> np.ndarray | None:
    target_times = [video_start + float(ts) for ts in local_timestamps.tolist()]
    output = np.empty((len(target_times), size, size, RGB_CHANNELS), dtype=np.uint8)
    tolerance = 0.5 / fps_native
    found = [False] * len(target_times)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is None:
                return None
            seconds = float(frame.time)
            for idx, target in enumerate(target_times):
                if not found[idx] and math.isclose(seconds, target, abs_tol=tolerance):
                    output[idx] = resize_frame(frame, size, interp)
                    found[idx] = True
            if all(found):
                return output
    return None


def resize_frame(frame: av.VideoFrame, size: int, interp: str) -> np.ndarray:
    array = frame.to_ndarray(format="rgb24")
    image = Image.fromarray(array)
    resample = (
        Image.Resampling.BILINEAR if interp == "bilinear" else Image.Resampling.BICUBIC
    )
    if image.size != (size, size):
        image = image.resize((size, size), resample=resample)
    return np.asarray(image, dtype=np.uint8)


def concatenate(
    decoded: list[EpisodeArrays], camera_views: tuple[str, ...]
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    if not decoded:
        raise RuntimeError("no decoded episodes")

    arrays: dict[str, np.ndarray | dict[str, np.ndarray]] = {
        "episode_index": np.concatenate([item.episode for item in decoded]).astype(
            np.int32
        ),
        "timestep": np.concatenate([item.timestep for item in decoded]).astype(
            np.int32
        ),
        "action": np.concatenate([item.action for item in decoded]).astype(np.float32),
        "joint_pos": np.concatenate([item.joint_pos for item in decoded]).astype(
            np.float32
        ),
        "observation": {},
    }
    obs = arrays["observation"]
    assert isinstance(obs, dict)
    for view in camera_views:
        obs[hdf5_pixels_name(view)] = np.concatenate(
            [item.pixels[view] for item in decoded]
        )
    return arrays


def compute_content_hash(
    arrays: dict[str, np.ndarray | dict[str, np.ndarray]], camera_views: tuple[str, ...]
) -> str:
    hasher = blake3.blake3()
    for name in ("episode_index", "timestep", "action", "joint_pos"):
        array = arrays[name]
        assert isinstance(array, np.ndarray)
        hasher.update(np.ascontiguousarray(array).tobytes())
    obs = arrays["observation"]
    assert isinstance(obs, dict)
    for view in camera_views:
        hasher.update(np.ascontiguousarray(obs[hdf5_pixels_name(view)]).tobytes())
    return hasher.hexdigest()


def write_hdf5_atomic(
    config: DecodeConfig,
    fps_native: int,
    arrays: dict[str, np.ndarray | dict[str, np.ndarray]],
    content_hash: str,
) -> None:
    config.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.out.with_suffix(config.out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with h5py.File(tmp, "w") as handle:
        handle.create_dataset("episode_index", data=arrays["episode_index"], dtype="i4")
        handle.create_dataset("timestep", data=arrays["timestep"], dtype="i4")
        handle.create_dataset("action", data=arrays["action"], dtype="f4")
        handle.create_dataset("joint_pos", data=arrays["joint_pos"], dtype="f4")

        obs_group = handle.create_group("observation")
        obs = arrays["observation"]
        assert isinstance(obs, dict)
        chunks = (64, config.size, config.size, RGB_CHANNELS)
        for view in config.camera_views:
            name = hdf5_pixels_name(view)
            data = obs[name]
            chunk_rows = min(chunks[0], max(1, int(data.shape[0])))
            obs_group.create_dataset(
                name,
                data=data,
                dtype="u1",
                chunks=(chunk_rows, config.size, config.size, RGB_CHANNELS),
            )

        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["fps_native"] = fps_native
        handle.attrs["fps_target"] = config.fps_target
        handle.attrs["size"] = config.size
        handle.attrs["interp"] = config.interp
        handle.attrs["source_dataset"] = config.source_dataset
        handle.attrs["source_revision"] = config.source_revision
        handle.attrs["camera_views"] = json.dumps(
            list(config.camera_views), sort_keys=True
        )
        handle.attrs["decode_tool_versions"] = json.dumps(
            tool_versions(), sort_keys=True
        )
        handle.attrs["content_hash"] = content_hash

    os.replace(tmp, config.out)


def tool_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pyav": av.__version__,
        "pillow": Image.__version__,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "pyarrow": pa.__version__,
        "blake3": getattr(blake3, "__version__", "unknown"),
    }


def validate_hdf5(
    path: Path, camera_views: tuple[str, ...] = DEFAULT_CAMERA_VIEWS
) -> None:
    with h5py.File(path, "r") as handle:
        required_attrs = [
            "schema_version",
            "fps_native",
            "fps_target",
            "size",
            "interp",
            "source_dataset",
            "source_revision",
            "decode_tool_versions",
            "content_hash",
        ]
        for attr in required_attrs:
            if attr not in handle.attrs:
                raise RuntimeError(f"missing root attr {attr}")

        n = require_1d(handle, "episode_index", np.integer)
        if require_1d(handle, "timestep", np.integer) != n:
            raise RuntimeError("/timestep length mismatch")
        if require_2d(handle, "action", np.floating, ACTION_DIM) != n:
            raise RuntimeError("/action length mismatch")
        if require_2d(handle, "joint_pos", np.floating, ACTION_DIM) != n:
            raise RuntimeError("/joint_pos length mismatch")
        if np.isnan(handle["action"][...]).any():
            raise RuntimeError("/action contains NaN after RFC0012-003 filtering")

        size = int(handle.attrs["size"])
        for view in camera_views:
            name = f"observation/{hdf5_pixels_name(view)}"
            if name not in handle:
                raise RuntimeError(f"missing /{name}")
            data = handle[name]
            if data.shape != (n, size, size, RGB_CHANNELS):
                raise RuntimeError(
                    f"/{name} has shape {data.shape}, expected {(n, size, size, RGB_CHANNELS)}"
                )
            if data.dtype != np.dtype("uint8"):
                raise RuntimeError(f"/{name} has dtype {data.dtype}, expected uint8")

        arrays: dict[str, np.ndarray | dict[str, np.ndarray]] = {
            "episode_index": handle["episode_index"][...],
            "timestep": handle["timestep"][...],
            "action": handle["action"][...],
            "joint_pos": handle["joint_pos"][...],
            "observation": {
                hdf5_pixels_name(view): handle[f"observation/{hdf5_pixels_name(view)}"][
                    ...
                ]
                for view in camera_views
            },
        }
        actual_hash = compute_content_hash(arrays, camera_views)
        if actual_hash != handle.attrs["content_hash"]:
            raise RuntimeError(
                f"content_hash mismatch: attr={handle.attrs['content_hash']} actual={actual_hash}"
            )


def require_1d(handle: h5py.File, name: str, dtype_kind: type[np.generic]) -> int:
    if name not in handle:
        raise RuntimeError(f"missing /{name}")
    data = handle[name]
    if len(data.shape) != 1:
        raise RuntimeError(f"/{name} has shape {data.shape}, expected 1-D")
    if not np.issubdtype(data.dtype, dtype_kind):
        raise RuntimeError(f"/{name} has dtype {data.dtype}")
    return int(data.shape[0])


def require_2d(
    handle: h5py.File, name: str, dtype_kind: type[np.generic], dim: int
) -> int:
    if name not in handle:
        raise RuntimeError(f"missing /{name}")
    data = handle[name]
    if len(data.shape) != 2 or data.shape[1] != dim:
        raise RuntimeError(f"/{name} has shape {data.shape}, expected (N,{dim})")
    if not np.issubdtype(data.dtype, dtype_kind):
        raise RuntimeError(f"/{name} has dtype {data.dtype}")
    return int(data.shape[0])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
