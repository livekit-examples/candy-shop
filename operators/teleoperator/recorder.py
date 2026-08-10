"""HITL LeRobotDataset recorder driving a live Portal stream.

``record`` pairs an executed action (any sender — human teleop or remote policy)
with the observation the operator was responding to and writes one row.

Invariant: one writer thread owns every dataset mutation. ``record`` runs on
Portal's callback thread and must be O(1) there — holding it starves Portal's
video-receive worker of the GIL, libwebrtc's queue overflows, and observation
delivery turns bursty. So ``record`` only pairs (a ~0.001 ms ring scan) and
enqueues; the writer thread does the decode, add_frame, and save_episode, all
through one ordered queue so a save never overtakes its rows.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence

from livekit.portal import Action, Observation, frame_bytes_to_numpy_rgb

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)

from operators.teleoperator.library import read_episodes, video_slice
from operators.teleoperator.dataset_repair import repair_dataset

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(
        self,
        *,
        repo_id: str,
        fps: int,
        state_field_names: Sequence[str],
        action_field_names: Sequence[str],
        cameras: Sequence[str],
        task: str,
        max_obs_age_us: int,
        root: str | Path | None = None,
        robot_type: str = "so101_follower",
        history: int = 16,
        write_queue: int = 60,
    ) -> None:
        self._repo_id = repo_id
        self._fps = fps
        self._state_field_names = list(state_field_names)
        self._action_field_names = list(action_field_names)
        self._cameras = tuple(cameras)
        self._task = task
        self._max_obs_age_us = max_obs_age_us
        # expanduser: a DATASET_ROOT of `~/.cache/...` would otherwise create a
        # directory literally named `~`.
        self._root = (
            Path(root).expanduser() if root is not None else Path("data") / repo_id
        )
        self._robot_type = robot_type

        # Ring of recent (obs, local_recv_us) pairs. local_recv_us is this
        # machine's wall clock at obs arrival, so leader actions match against
        # when the operator *saw* an obs (teleop clock), not when the robot
        # *captured* it (robot clock).
        self._history: deque[tuple[Observation, int]] = deque(maxlen=history)

        self._dataset: Optional[LeRobotDataset] = None
        self._features: dict | None = None
        self._recording = False
        self._episode_count = 0

        # Bounded queue: unbounded would trade dropped rows for unbounded memory,
        # since each item pins a whole observation's frame bytes (~1.8 MB for two
        # 640x480 cameras).
        self._queue: queue.Queue = queue.Queue(maxsize=max(write_queue, 1))
        self._writer: threading.Thread | None = None
        self._pending_saves = 0
        self._queued_rows = 0
        # Drops counted by cause: the causes have unrelated fixes.
        self._dropped_stale = 0    # paired obs older than max_obs_age_us
        self._dropped_error = 0    # add_frame raised
        self._dropped_unpaired = 0  # no obs old enough to pair with at all
        self._dropped_backlog = 0  # the writer thread couldn't keep up
        self._skips_since_report = 0
        self._warned_write_error = False

        # Rolling receive times, for the observed obs rate.
        self._obs_recv: deque[int] = deque(maxlen=64)
        self._last_age_us = 0
        self._worst_age_us = 0

        self._rows = 0

        # Set while an offline mutation owns the dataset; checked on the record
        # hot path so an in-flight write can't land on a half-closed dataset.
        self._suspended = threading.Event()

        self._revision = 0  # bumped when the saved-episode set changes

        # Never re-read on demand: mid-session the metadata parquet is footerless
        # (see dataset_repair). Seeded at open, appended on save, re-seeded after
        # a mutation.
        self._episodes: list[dict] = []

        # Reused verbatim on resume, so a rebuilt dataset keeps identical video features.
        self._frame_shape: tuple[int, int] | None = None

    # --- lifecycle ----------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._dataset is not None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_saving(self) -> bool:
        """True while a finished episode is still being written or encoded."""
        return self._pending_saves > 0

    @property
    def is_suspended(self) -> bool:
        return self._suspended.is_set()

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def rows(self) -> int:
        """Rows written into the in-flight episode (0 when not recording)."""
        return self._rows

    @property
    def skipped_frames(self) -> int:
        """Total rows dropped this episode, all causes."""
        return (self._dropped_stale + self._dropped_error
                + self._dropped_unpaired + self._dropped_backlog)

    @property
    def drop_causes(self) -> dict[str, int]:
        """Why rows were dropped this episode: `stale` (paired obs too old),
        `unpaired` (no obs old enough / action clock behind), `error` (add_frame
        raised — schema/disk), `backlog` (writer fell behind, queue full)."""
        return {"stale": self._dropped_stale, "unpaired": self._dropped_unpaired,
                "error": self._dropped_error, "backlog": self._dropped_backlog}

    @property
    def queue_depth(self) -> int:
        """Rows waiting on the writer. Sustained non-zero = disk or encoder
        bottleneck, not the network."""
        return self._queue.qsize()

    @property
    def obs_fps(self) -> float:
        """Observations per second, measured over the rolling window."""
        if len(self._obs_recv) < 2:
            return 0.0
        span_us = self._obs_recv[-1] - self._obs_recv[0]
        return (len(self._obs_recv) - 1) / (span_us / 1e6) if span_us > 0 else 0.0

    @property
    def pairing_age_ms(self) -> tuple[float, float]:
        """(last, worst) action-to-observation gap this episode, milliseconds."""
        return self._last_age_us / 1000.0, self._worst_age_us / 1000.0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def episodes(self) -> list[dict]:
        """The saved-episode index (`{index, length, seconds, task}` each).
        Returns the live list — callers must not mutate it."""
        return self._episodes

    @property
    def root(self) -> Path:
        return self._root

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def task(self) -> str:
        return self._task

    def set_task(self, task: str) -> None:
        """Relabel subsequent rows. Between episodes only — each frame is stamped
        at ``add_frame`` time, so changing it mid-episode would split one
        trajectory across two labels (the caller gates this)."""
        self._task = task

    def ensure_dataset(self, frame=None) -> bool:
        """Build (or resume) the dataset from a frame's resolution. True if it
        built on this call, False if already built. ``frame`` may be omitted once
        a resolution has been seen (the ``resume`` path after a mutation)."""
        if self._dataset is not None:
            return False
        if frame is not None:
            self._frame_shape = tuple(frame.shape[:2])  # type: ignore[assignment]
        if self._frame_shape is None:
            return False
        height, width = self._frame_shape

        # Mirror the wire schema into lerobot's hw_features shape: state fields
        # become named floats; each camera goes in as (H, W, 3).
        obs_hw: dict[str, Any] = {name: float for name in self._state_field_names}
        for cam in self._cameras:
            obs_hw[cam] = (height, width, 3)
        act_hw: dict[str, Any] = {name: float for name in self._action_field_names}
        self._features = combine_feature_dicts(
            hw_to_dataset_features(obs_hw, OBS_STR, use_video=True),
            hw_to_dataset_features(act_hw, ACTION, use_video=True),
        )

        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # local recording, no Hub calls
        if (self._root / "meta" / "tasks.parquet").exists():
            # Self-heal a dataset left footerless by a hard kill last run. No-op
            # if already valid. See dataset_repair.
            try:
                if repair_dataset(self._root):
                    logger.warning("repaired footerless parquet(s) from a prior crash")
            except Exception:
                logger.exception("dataset repair failed; attempting resume anyway")
            logger.info("resuming dataset at %s", self._root)
            self._dataset = LeRobotDataset.resume(
                repo_id=self._repo_id, root=self._root, image_writer_threads=4,
                streaming_encoding=True, encoder_threads=2,
            )
        else:
            if self._root.exists():
                logger.warning("wiping incomplete dataset at %s", self._root)
                shutil.rmtree(self._root)  # stub from an aborted run; create() would error
            logger.info("creating dataset at %s", self._root)
            self._dataset = LeRobotDataset.create(
                repo_id=self._repo_id, fps=self._fps, features=self._features,
                root=self._root, robot_type=self._robot_type, use_videos=True,
                image_writer_threads=4, streaming_encoding=True, encoder_threads=2,
            )
        # size=1 flushes each episode's metadata row immediately (default 10).
        # Buffered rows live only in memory, so a hard kill would lose up to 9
        # saved episodes' metadata; with size=1, footer repair recovers them all.
        self._dataset.meta._metadata_buffer_size = 1
        self._episode_count = self._dataset.num_episodes
        # Safe to read disk here and only here: just opened, so every footer is
        # written (repaired above if a prior run died).
        self._episodes = read_episodes(self._root, self._fps, self._cameras)
        self._start_writer()
        return True

    def start_episode(self) -> bool:
        """Begin recording. Resets the obs ring so the first row pairs against an
        obs from within the episode.

        Returns False (no-op) if the previous episode is still saving: blocking on
        that encode would stall Portal's callback thread, and racing the save
        worker would corrupt the dataset. The caller retries."""
        if (
            self._dataset is None
            or self._recording
            or self._pending_saves > 0
            or self._suspended.is_set()
        ):
            return False
        self._history.clear()
        self._dropped_stale = self._dropped_error = self._dropped_unpaired = 0
        self._skips_since_report = 0
        self._last_age_us = self._worst_age_us = 0
        self._rows = 0
        self._queued_rows = 0
        self._warned_write_error = False
        self._recording = True
        return True

    def end_episode(self) -> None:
        """Stop recording and hand the episode to the writer.

        Queued, not inline: rows may still be in flight, and the save must land
        *after* them. `has_pending_frames()` can't be consulted here — it reflects
        the dataset buffer, which the writer hasn't filled yet."""
        if not self._recording:
            return
        self._recording = False
        if not self._queued_rows:
            return  # nothing captured; no episode to save
        # `length` is filled in by the writer, the only thing that knows how many
        # rows really landed.
        self._pending_saves += 1
        self._enqueue(("save", {"index": self._episode_count, "task": self._task}))
        self._episode_count += 1

    def discard_episode(self) -> None:
        self._recording = False
        self._rows = 0
        self._queued_rows = 0
        # Queued behind any in-flight rows, so it clears them too.
        self._enqueue(("discard", None))

    def flush(self) -> None:
        """Block until every accepted row and save is written. For shutdown and
        tests. Never call from Portal's callback thread — that is the stall this
        design exists to prevent."""
        while not self._queue.empty():
            if self._writer is None or not self._writer.is_alive():
                return  # no one left to drain it; don't hang
            time.sleep(0.01)
        self._queue.join()

    def finalize(self) -> None:
        self._stop_writer()
        if self._dataset is not None:
            self._dataset.finalize()

    # --- offline mutation window --------------------------------------------

    def suspend(self) -> None:
        """Close the dataset so an offline rewrite can own the files.

        Flags drop *before* the dataset is touched, then a brief sleep lets any
        in-flight ``add_frame`` finish before ``finalize`` closes the writers.
        Flags not a lock: ``record`` runs on the callback thread, and a lock held
        across a multi-minute rewrite would stall the tick loop."""
        self._suspended.set()
        self._recording = False
        time.sleep(0.05)
        # Retire the writer first: clearing the buffer under it would race an add_frame.
        self._stop_writer()
        if self._dataset is not None:
            if self._dataset.has_pending_frames():
                # A rewrite drops the in-flight episode anyway; discard it so the
                # rewrite starts from a fully-saved corpus.
                self._dataset.clear_episode_buffer(delete_images=True)
            self._dataset.finalize()
        self._dataset = None

    def resume(self) -> None:
        """Re-open after a mutation. Safe when nothing is on disk — the dataset
        is then rebuilt on the next frame."""
        self._suspended.clear()
        self._rows = 0
        self._history.clear()
        self.ensure_dataset()
        self._revision += 1

    # --- streams ------------------------------------------------------------

    def observe(self, obs: Observation) -> None:
        """Append an observation to the ring, stamped with its local receive time."""
        now = int(time.time() * 1_000_000)
        self._history.append((obs, now))
        self._obs_recv.append(now)

    def record(self, action: Action) -> None:
        """Pair an executed action with the obs the operator was responding
        to and write one row. No-op until recording."""
        if not self._recording or self._suspended.is_set():
            return

        # Reference timestamp + obs clock to match on:
        #   * Policy actions carry `in_reply_to_ts_us` (robot-clock capture time
        #     naming the exact obs) — match on obs capture ts, an exact pairing.
        #   * Leader actions are open-loop: match the action's teleop send time
        #     against each obs's local receive time (both teleop clock), avoiding
        #     robot/teleop skew + transport latency.
        match_on_recv = action.in_reply_to_ts_us is None
        target_ts = action.timestamp_us if match_on_recv else action.in_reply_to_ts_us

        aligned: Observation | None = None
        age = 0
        for obs, recv_us in reversed(self._history):
            obs_ts = recv_us if match_on_recv else obs.timestamp_us
            if obs_ts <= target_ts:
                aligned, age = obs, target_ts - obs_ts
                break
        if aligned is None:
            # Normal for the first ticks; sustained, every buffered obs is newer
            # than the action — the action's clock runs behind ours (cross-machine skew).
            self._dropped_unpaired += 1
            self._skips_since_report += 1
            return
        self._last_age_us = age
        self._worst_age_us = max(self._worst_age_us, age)
        if age > self._max_obs_age_us:
            self._dropped_stale += 1
            self._skips_since_report += 1
            return  # would mislabel the row

        # Hand off: everything past here costs milliseconds and would be paid on
        # the callback thread (see module docstring).
        values = {k: float(v) for k, v in action.values.items()}
        if self._enqueue(("row", (aligned, values, self._task))):
            self._queued_rows += 1
        else:
            self._dropped_backlog += 1
            self._skips_since_report += 1

    def drain_skips(self) -> int:
        """Drops since the last call, for the console report. Separate from
        ``skipped_frames`` (cumulative) so resetting one doesn't flicker it."""
        n, self._skips_since_report = self._skips_since_report, 0
        return n

    # --- internals ----------------------------------------------------------

    def _obs_to_record(self, obs: Observation) -> dict:
        out: dict = dict(obs.state)
        for cam in self._cameras:
            if (f := obs.frames.get(cam)) is not None:
                out[cam] = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)
        return out

    # --- the writer thread --------------------------------------------------

    def _enqueue(self, item: tuple) -> bool:
        """Post to the writer without blocking the caller. False if the queue is
        full — blocking would push backpressure onto the callback thread."""
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def _start_writer(self) -> None:
        if self._writer is not None and self._writer.is_alive():
            return
        self._writer = threading.Thread(
            target=self._write_worker, name="dataset-writer", daemon=True,
        )
        self._writer.start()

    def _stop_writer(self) -> None:
        """Drain the queue, then retire the thread. So every accepted row is on
        disk before finalize or an offline mutation."""
        if self._writer is None:
            return
        self._queue.put((None, None))  # sentinel, after everything queued
        self._writer.join(timeout=120)
        self._writer = None

    def _write_worker(self) -> None:
        """The only thing that mutates the dataset. Strictly in queue order, so a
        save can never overtake its own rows."""
        while True:
            kind, payload = self._queue.get()
            try:
                if kind is None:
                    return
                if kind == "row":
                    self._write_row(*payload)
                elif kind == "save":
                    self._save_now(payload)
                elif kind == "discard":
                    self._discard_now()
            except Exception:
                logger.exception("dataset writer failed on a %r item", kind)
            finally:
                self._queue.task_done()

    def _write_row(self, obs: Observation, values: dict, task: str) -> None:
        try:
            self._dataset.add_frame({
                **build_dataset_frame(self._features, self._obs_to_record(obs), OBS_STR),
                **build_dataset_frame(self._features, values, ACTION),
                "task": task,
            })
            self._rows += 1
        except Exception:
            # A bad row must not end the demonstration: drop, count, continue;
            # log the first per episode with a traceback.
            if not self._warned_write_error:
                logger.exception("add_frame failed; dropping row and continuing")
                self._warned_write_error = True
            self._dropped_error += 1
            self._skips_since_report += 1

    def _save_now(self, entry: dict) -> None:
        episode = entry["index"]
        started = time.perf_counter()
        try:
            if not self._dataset.has_pending_frames():
                logger.warning("episode %d had no rows to save", episode)
                return
            self._dataset.save_episode()
        except Exception:
            logger.exception("save_episode failed for episode %d", episode)
            return
        finally:
            self._pending_saves -= 1
        # `length` and video offsets are only knowable here — read back off the
        # metadata lerobot just wrote.
        entry = {**entry, "length": self._rows,
                 "seconds": round(self._rows / self._fps, 2) if self._fps else 0.0,
                 "videos": self._latest_video_slices()}
        self._episodes.append(entry)
        self._revision += 1
        logger.info("episode %d saved (%d rows) in %.2fs",
                    episode, self._rows, time.perf_counter() - started)

    def _latest_video_slices(self) -> dict:
        """Per-camera video offsets for the episode just saved. `{}` if absent —
        playback degrades, nothing else."""
        try:
            latest = self._dataset.meta.latest_episode or {}
        except Exception:
            return {}
        # Values arrive as single-element lists.
        row = {k: (v[0] if isinstance(v, (list, tuple)) and v else v)
               for k, v in latest.items()}
        return {c: s for c in self._cameras if (s := video_slice(row, c)) is not None}

    def _discard_now(self) -> None:
        if self._dataset is not None and self._dataset.has_pending_frames():
            self._dataset.clear_episode_buffer(delete_images=True)
