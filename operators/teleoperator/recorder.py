"""HITL LeRobotDataset recorder for the leslider rig.

Drives a `LeRobotDataset` from a live Portal stream. ``observe`` feeds a
ring of recent observations; ``record`` pairs an executed action with the
observation the operator was responding to and writes one row. The dataset
is built lazily by ``ensure_dataset`` once a frame reveals the camera
resolution (the wire schema doesn't pin width/height), and resumed from
disk if it already exists, so a corpus can grow across sessions. Episodes
flush on a background thread so the operator's tick rate isn't held up by
encoding.

The wire fields are the leslider's: six arm ``.pos`` plus one ``slider.vel``
(raw ticks/s, sign-magnitude). Nothing here is special-cased to positions —
``state_field_names``/``action_field_names`` come straight from the wire
contract, so ``slider.vel`` lands in every row's state and action exactly like
a ``.pos`` field, as a plain float.

Because recording is driven by the *action* stream — and Portal forwards
every operator's actions (``action_subscription``) tagged with their
``sender`` — this records human teleop and remote-policy actions alike.

``suspend``/``resume`` bracket an offline mutation of the corpus: the dataset is
not safely readable, let alone rewritable, while a session holds it open.
Suspend closes it properly; resume re-opens via the same path a fresh process
takes.

**One writer thread owns every dataset mutation.** ``record`` runs on Portal's
callback thread and must be O(1) there: measured, ``add_frame`` costs ~2 ms median
and up to 24 ms, against a 33 ms tick at 30 fps. Held on the callback thread that
starves Portal's video-receive worker of the GIL, libwebrtc's native queue
overflows ("dropped N queued frames"), observation *delivery* turns bursty, and
rows then fail to pair because the newest observation received before an action is
hundreds of milliseconds old — even though the average obs rate still reads 30/s.
So ``record`` only pairs (a 0.001 ms ring scan) and enqueues; the writer thread
does the decode, the frame build, the ``add_frame``, and the ``save_episode``.
Rows, saves and discards go through one ordered queue, so a save can never
overtake the rows it belongs to.
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
        # expanduser: neither python-dotenv nor pathlib expands `~`, so a
        # DATASET_ROOT of `~/.cache/...` would create a directory *named* `~`.
        self._root = (
            Path(root).expanduser() if root is not None else Path("data") / repo_id
        )
        self._robot_type = robot_type

        # Ring of recent (obs, local_recv_us) pairs. local_recv_us is this
        # machine's wall clock when the obs arrived, so leader actions match
        # against when the operator *saw* an obs (teleop clock) rather than
        # when the robot *captured* it (robot clock). ~500ms at 30Hz.
        self._history: deque[tuple[Observation, int]] = deque(maxlen=history)

        # Built lazily by ensure_dataset() once a frame reveals the resolution.
        self._dataset: Optional[LeRobotDataset] = None
        self._features: dict | None = None
        self._recording = False
        self._episode_count = 0

        # The single writer. Bounded queue: unbounded would trade dropped rows
        # for unbounded memory, since each item pins a whole observation's frame
        # bytes (~1.8 MB for two 640x480 cameras).
        self._queue: queue.Queue = queue.Queue(maxsize=max(write_queue, 1))
        self._writer: threading.Thread | None = None
        self._pending_saves = 0
        self._queued_rows = 0
        # Drops are counted by CAUSE. One number can't tell you whether the obs
        # stream is slow, the two machines' clocks disagree, or add_frame is
        # raising on every row — and those have nothing to do with each other.
        self._dropped_stale = 0    # paired obs older than max_obs_age_us
        self._dropped_error = 0    # add_frame raised
        self._dropped_unpaired = 0  # no obs old enough to pair with at all
        self._dropped_backlog = 0  # the writer thread couldn't keep up
        self._skips_since_report = 0
        self._warned_write_error = False

        # Rolling receive times, for the observed obs rate. A pairing age can
        # only ever be as good as the gap between observations, so this is the
        # first thing to look at when rows drop as stale.
        self._obs_recv: deque[int] = deque(maxlen=64)
        self._last_age_us = 0
        self._worst_age_us = 0

        # Counted here, not read off the dataset's episode buffer, so the frame
        # counter doesn't depend on lerobot internals.
        self._rows = 0

        # Set while an offline mutation owns the dataset; checked on the record
        # hot path so an in-flight write can't land on a half-closed dataset.
        self._suspended = threading.Event()

        # Bumped when the saved-episode set changes.
        self._revision = 0

        # Seeded from disk at open, appended on save, re-seeded after a mutation
        # — never re-read on demand, since mid-session the metadata parquet is
        # footerless (see dataset.py's docstring).
        self._episodes: list[dict] = []

        # Reused verbatim on resume, so a rebuilt dataset keeps identical video
        # features.
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
        """Why rows were dropped this episode. `stale` = the paired observation
        was too old; `unpaired` = no observation old enough to pair with (a slow
        obs stream, or the action clock running behind ours); `error` = add_frame
        raised, which is a schema/disk problem, not a timing one; `backlog` = the
        writer thread fell behind and the queue was full (disk or encoder too
        slow for the frame rate)."""
        return {"stale": self._dropped_stale, "unpaired": self._dropped_unpaired,
                "error": self._dropped_error, "backlog": self._dropped_backlog}

    @property
    def queue_depth(self) -> int:
        """Rows waiting on the writer. Sustained non-zero means the disk or the
        video encoder is the bottleneck, not the network."""
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
        """Relabel subsequent rows. Intended for use between episodes — every
        frame is stamped with the task current at ``add_frame`` time, so
        changing it mid-episode would split one trajectory across two labels
        (the caller gates this to when not recording)."""
        self._task = task

    def ensure_dataset(self, frame=None) -> bool:
        """Build (or resume) the dataset from a received frame's resolution.
        Returns True if it built on this call, False if already built.

        ``frame`` may be omitted once a resolution has been seen — that's the
        ``resume`` path after a mutation, which must not wait for a new frame."""
        if self._dataset is not None:
            return False
        if frame is not None:
            self._frame_shape = tuple(frame.shape[:2])  # type: ignore[assignment]
        if self._frame_shape is None:
            return False
        height, width = self._frame_shape

        # Mirror the wire schema into lerobot's hw_features shape: state
        # fields become named floats; each camera goes in as (H, W, 3).
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
            # Self-heal a dataset left footerless by a hard kill / segfault last
            # run (ParquetWriter only writes footers at finalize()). No-op if the
            # files are already valid. See dataset_repair.
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
                shutil.rmtree(self._root)  # stub from an aborted run; create would error
            logger.info("creating dataset at %s", self._root)
            self._dataset = LeRobotDataset.create(
                repo_id=self._repo_id, fps=self._fps, features=self._features,
                root=self._root, robot_type=self._robot_type, use_videos=True,
                image_writer_threads=4, streaming_encoding=True, encoder_threads=2,
            )
        # Flush every episode's metadata row to disk immediately instead of
        # buffering (default 10). Buffered rows live only in memory, so a hard
        # kill would lose up to 9 saved episodes' metadata even though their data
        # + video are on disk. With size=1, footer repair can recover them all.
        self._dataset.meta._metadata_buffer_size = 1
        self._episode_count = self._dataset.num_episodes
        # Valid to read disk here and only here: the dataset was just opened, so
        # every footer is written (repaired above if a prior run died).
        self._episodes = read_episodes(self._root, self._fps, self._cameras)
        self._start_writer()
        return True

    def start_episode(self) -> bool:
        """Begin recording. Resets the obs ring so the first row pairs
        against an obs from within the episode, not from before.

        Returns False (a no-op) if the previous episode is still saving:
        blocking on that encode here would freeze the caller's thread (the
        teleop tick loop shares it with Portal's frame-delivery callbacks, so
        a join stalls the whole video receive path), and racing the save
        worker on the shared dataset would corrupt it. The caller retries."""
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

        Queued, not done inline: rows may still be in flight, and the save has to
        land *after* them. `has_pending_frames()` can't be consulted here — it
        reflects the dataset buffer, which the writer hasn't filled yet."""
        if not self._recording:
            return
        self._recording = False
        if not self._queued_rows:
            return  # nothing was captured; there is no episode to save
        # Index and task are snapshotted here; `length` is filled in by the
        # writer, which is the only thing that knows how many rows really landed.
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
        """Block until every accepted row and save has been written.

        For shutdown and tests. Never call from Portal's callback thread — that
        is exactly the stall this design exists to prevent."""
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

        Order matters: flags drop *before* the dataset is touched, then a brief
        sleep lets any ``add_frame`` already past the check finish (sub-ms)
        before ``finalize`` closes the writers under it. Flags rather than a lock
        because ``record`` runs on the Portal callback thread, and a lock held
        across a multi-minute rewrite would stall the whole tick loop."""
        self._suspended.set()
        self._recording = False
        time.sleep(0.05)
        # Retire the writer first: it is still draining rows, and clearing the
        # buffer underneath it would race an add_frame.
        self._stop_writer()
        if self._dataset is not None:
            if self._dataset.has_pending_frames():
                # A rewrite would silently drop the in-flight episode anyway;
                # discard it so the rewrite starts from a fully-saved corpus.
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
        """Append an observation to the ring, stamped with its local
        receive time (this fires on receipt)."""
        now = int(time.time() * 1_000_000)
        self._history.append((obs, now))
        self._obs_recv.append(now)

    def record(self, action: Action) -> None:
        """Pair an executed action with the obs the operator was responding
        to and write one row. No-op until recording."""
        if not self._recording or self._suspended.is_set():
            return

        # Pick the reference timestamp and the obs clock to compare against:
        #   * Policy actions carry `in_reply_to_ts_us`, a robot-clock capture
        #     time naming the exact obs consumed — match on obs capture ts
        #     (same clock), an exact pairing.
        #   * Leader actions are open-loop: the human reacts to the most
        #     recently *received* obs — match the action's teleop send time
        #     against each obs's local receive time (both teleop clock),
        #     avoiding the robot/teleop skew + transport latency.
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
            # Normal for the first ticks of an episode; sustained, it means every
            # buffered obs is NEWER than the action — i.e. the action's clock runs
            # behind this machine's, which is what cross-machine skew looks like.
            self._dropped_unpaired += 1
            self._skips_since_report += 1
            return
        self._last_age_us = age
        self._worst_age_us = max(self._worst_age_us, age)
        if age > self._max_obs_age_us:
            self._dropped_stale += 1
            self._skips_since_report += 1
            return  # would mislabel the row

        # Hand off and return. Everything past this point costs milliseconds and
        # would be paid on Portal's callback thread (see the module docstring).
        values = {k: float(v) for k, v in action.values.items()}
        if self._enqueue(("row", (aligned, values, self._task))):
            self._queued_rows += 1
        else:
            self._dropped_backlog += 1
            self._skips_since_report += 1

    def drain_skips(self) -> int:
        """Drops since the last call, for the console report. Separate from
        ``skipped_frames`` (cumulative per episode) so resetting one doesn't make
        the counter flicker to zero."""
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
        """Post to the writer without ever blocking the caller. False if the
        queue is full — blocking here would push backpressure straight onto
        Portal's callback thread, which is the whole thing we're avoiding."""
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
        """Drain the queue, then retire the thread. Called before finalize and
        before an offline mutation, so every accepted row is on disk first."""
        if self._writer is None:
            return
        self._queue.put((None, None))  # sentinel, after everything queued
        self._writer.join(timeout=120)
        self._writer = None

    def _write_worker(self) -> None:
        """The only thing that mutates the dataset. Single-threaded and strictly
        in queue order, so a save can never overtake its own rows."""
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
            # A bad row must not end the demonstration. Drop it, count it, keep
            # going; log the first per episode with a traceback.
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
        # `length` is only knowable here: it's the rows that actually landed. The
        # video offsets likewise — lerobot fills them in as it writes the episode,
        # so they're read back off the metadata it just produced.
        entry = {**entry, "length": self._rows,
                 "seconds": round(self._rows / self._fps, 2) if self._fps else 0.0,
                 "videos": self._latest_video_slices()}
        self._episodes.append(entry)
        self._revision += 1  # the episode list grew
        logger.info("episode %d saved (%d rows) in %.2fs",
                    episode, self._rows, time.perf_counter() - started)

    def _latest_video_slices(self) -> dict:
        """Per-camera video offsets for the episode just saved, from lerobot's own
        episode metadata. `{}` if it isn't there — playback degrades, nothing else."""
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
