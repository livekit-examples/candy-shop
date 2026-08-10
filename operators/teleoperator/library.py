"""Offline corpus operations: read the episode index, relabel, delete episodes.

Assumes no recording session holds the dataset open (mid-session the parquet
footers aren't written), so callers bracket every mutation with
``Recorder.suspend()``/``resume()``. Both mutations rewrite the whole corpus
(v3.0 layout: many episodes share one parquet + one concatenated mp4), taking
seconds to minutes — run on a worker thread, never the event loop.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

EPISODES_GLOB = "meta/episodes/**/*.parquet"
VIDEO_PATH = "videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def video_key(camera: str) -> str:
    """lerobot namespaces camera features; the video directory uses that name."""
    return f"observation.images.{camera}"


def video_slice(row: Mapping[str, object], camera: str) -> Optional[dict]:
    """Where `camera`'s footage for one episode lives: a path relative to the
    dataset root plus the second range it occupies (episodes share one mp4)."""
    key = video_key(camera)
    try:
        chunk = int(row[f"videos/{key}/chunk_index"])        # type: ignore[arg-type]
        file_index = int(row[f"videos/{key}/file_index"])    # type: ignore[arg-type]
        start = float(row[f"videos/{key}/from_timestamp"])   # type: ignore[arg-type]
        end = float(row[f"videos/{key}/to_timestamp"])       # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        return None
    return {"path": VIDEO_PATH.format(key=key, chunk=chunk, file=file_index),
            "from": start, "to": end}


# --- reading -----------------------------------------------------------------

def read_episodes(root: Path, fps: int, cameras: Sequence[str] = ()) -> list[dict]:
    """The episode index as plain dicts, ordered by index; ``[]`` if absent.
    Reads the parquet directly (``LeRobotDataset`` would memory-map every frame)."""
    import pandas as pd

    paths = sorted((root).glob(EPISODES_GLOB))
    if not paths:
        return []

    wanted = ["episode_index", "tasks", "length"]
    for camera in cameras:
        key = video_key(camera)
        wanted += [f"videos/{key}/{f}" for f in
                   ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]

    out: list[dict] = []
    for path in paths:
        # Only the columns that exist: a corpus recorded with different cameras
        # (or none) must still list, just without playback.
        available = set(pd.read_parquet(path, columns=["episode_index"]).columns) | set()
        columns = [c for c in wanted if c in _columns_of(path)]
        frame = pd.read_parquet(path, columns=columns)
        for row in frame.to_dict("records"):
            tasks = list(row.get("tasks") or [])
            videos = {c: v for c in cameras if (v := video_slice(row, c)) is not None}
            length = int(row["length"])
            out.append({
                "index": int(row["episode_index"]),
                "length": length,
                "seconds": round(length / fps, 2) if fps else 0.0,
                # We stamp one task per episode, so tasks[0] IS the label; join
                # any others so an externally-authored dataset still reads true.
                "task": tasks[0] if len(tasks) == 1 else " | ".join(str(t) for t in tasks),
                "videos": videos,
            })
    out.sort(key=lambda e: e["index"])
    return out


def _columns_of(path: Path) -> set:
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


# --- mutating ----------------------------------------------------------------

def relabel_episodes(root: Path, repo_id: str, mapping: Mapping[int, str]) -> int:
    """Rewrite specific episodes' task labels in place; returns the count. Costs
    the whole corpus even for one episode (``modify_tasks`` rebuilds
    ``meta/tasks.parquet`` and every ``task_index`` column)."""
    if not mapping:
        return 0
    from lerobot.datasets.dataset_tools import modify_tasks

    dataset = _open(root, repo_id)
    total = dataset.meta.total_episodes
    unknown = [i for i in mapping if not 0 <= i < total]
    if unknown:
        raise ValueError(f"no such episode(s): {sorted(unknown)} (dataset has {total})")

    logger.info("relabelling %d episode(s) in %s", len(mapping), root)
    modify_tasks(dataset, episode_tasks={int(k): str(v) for k, v in mapping.items()})
    return len(mapping)


def delete_episodes(root: Path, repo_id: str, indices: Sequence[int]) -> int:
    """Delete episodes, leaving the result at the same path, reindexed; returns
    the remaining count. lerobot only writes a *new* dataset, so build it in a
    sibling ``.rewrite/`` and swap by rename, parking the original as ``.trash/``
    until the swap succeeds — a crash mid-rewrite leaves the original untouched."""
    from lerobot.datasets.dataset_tools import delete_episodes as _delete

    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return _count(root)

    dataset = _open(root, repo_id)
    total = dataset.meta.total_episodes
    unknown = [i for i in wanted if not 0 <= i < total]
    if unknown:
        raise ValueError(f"no such episode(s): {unknown} (dataset has {total})")
    if len(wanted) >= total:
        # There is no such thing as a zero-episode dataset on disk. Say so
        # plainly rather than surface lerobot's opaque error from three frames up.
        raise ValueError(
            f"refusing to delete all {total} episode(s) — delete the dataset "
            f"directory instead ({root})"
        )

    rewrite = root.parent / f"{root.name}.rewrite"
    trash = root.parent / f"{root.name}.trash"
    shutil.rmtree(rewrite, ignore_errors=True)
    shutil.rmtree(trash, ignore_errors=True)

    logger.info("deleting episode(s) %s from %s", wanted, root)
    _delete(dataset, wanted, output_dir=rewrite, repo_id=repo_id)

    root.rename(trash)
    try:
        rewrite.rename(root)
    except Exception:
        trash.rename(root)  # put the original back before re-raising
        raise
    shutil.rmtree(trash, ignore_errors=True)
    return total - len(wanted)


# --- internals ---------------------------------------------------------------

def _open(root: Path, repo_id: str):
    """Open an existing on-disk dataset for mutation, without Hub access."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return LeRobotDataset(repo_id=repo_id, root=root)


def _count(root: Path) -> int:
    import json

    info = root / "meta" / "info.json"
    if not info.exists():
        return 0
    return int(json.loads(info.read_text()).get("total_episodes", 0))
