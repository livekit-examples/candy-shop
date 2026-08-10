"""Decode one episode's cameras from the corpus, for review.

Reads the shared mp4s directly rather than through ``LeRobotDataset``: mid-session
the dataset's metadata parquet is footerless and pyarrow won't open it, while the
mp4s decode fine even while being appended to. Offsets come from the recorder over RPC.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CACHE_FRAMES = 48
"""Decoded frames kept per camera. Full-resolution RGB (~0.9 MB each at 640x480)."""


class _Track:
    """One camera's decoder, positioned within an episode's slice of an mp4."""

    def __init__(self, path: Path, start_s: float, end_s: float, frames: int) -> None:
        self.path = path
        self.start_s = start_s
        self.end_s = end_s
        self.frames = frames
        self._container = None
        self._stream = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._next_index: Optional[int] = None  # frame the decoder will yield next
        self._iter = None

    # --- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        import av

        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        # Single-threaded on purpose: "auto" spawns worker threads per container that
        # fought the UI for cores while scrubbing without decoding any faster here.
        self._stream.thread_type = "NONE"

    def close(self) -> None:
        self._cache.clear()
        self._iter = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    # --- frames ------------------------------------------------------------

    def frame(self, index: int) -> Optional[np.ndarray]:
        """The episode's `index`-th frame as HxWx3 RGB, or None if undecodable."""
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        if self._container is None:
            return None
        try:
            return self._decode_to(index)
        except Exception:
            logger.exception("could not decode frame %d of %s", index, self.path.name)
            return None

    def _decode_to(self, index: int) -> Optional[np.ndarray]:
        # Decode forward when the target is just ahead (the common case while playing);
        # seeking would discard decoder state. Seek only when jumping.
        if self._next_index is None or index < self._next_index or index > self._next_index + 24:
            self._seek(index)

        while self._next_index is not None and self._next_index <= index:
            frame = next(self._iter, None)   # type: ignore[arg-type]
            if frame is None:
                return None
            current, self._next_index = self._next_index, self._next_index + 1
            if current >= index - 2:  # keep a little scrub-back, not everything
                self._store(current, frame.to_ndarray(format="rgb24"))
        return self._cache.get(index)

    def _seek(self, index: int) -> None:
        assert self._container is not None and self._stream is not None
        target_s = self.start_s + index / max(self.frames, 1) * max(
            self.end_s - self.start_s, 1e-9)
        offset = int(target_s / self._stream.time_base)
        self._container.seek(offset, stream=self._stream)
        self._iter = self._container.decode(video=0)
        # A seek lands on the nearest keyframe at or before the target, so the
        # decoder's position is not the frame we asked for. Walk forward using
        # presentation timestamps to find out where we actually are.
        for frame in self._iter:
            seconds = float(frame.pts * self._stream.time_base)
            landed = int(round((seconds - self.start_s) / max(
                (self.end_s - self.start_s) / max(self.frames, 1), 1e-9)))
            self._next_index = max(landed, 0) + 1
            self._store(max(landed, 0), frame.to_ndarray(format="rgb24"))
            return
        self._next_index = None

    def _store(self, index: int, array: np.ndarray) -> None:
        self._cache[index] = array
        self._cache.move_to_end(index)
        while len(self._cache) > CACHE_FRAMES:
            self._cache.popitem(last=False)


class EpisodePlayer:
    """The open episode: one decoder per camera, one shared frame cursor."""

    def __init__(self) -> None:
        self.episode: Optional[int] = None
        self.frames = 0
        self.position = 0
        self.playing = False
        self.error = ""
        self._tracks: dict[str, _Track] = {}
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return bool(self._tracks)

    def open(self, root: Path, episode: int, videos: dict, frames: int) -> bool:
        """Open `episode`'s cameras. False (with `error` set) if the corpus isn't readable from here."""
        self.close()
        self.error = ""
        if not videos:
            self.error = "this episode has no video tracks"
            return False
        if not root.exists():
            self.error = f"corpus not readable from here ({root})"
            return False

        opened: dict[str, _Track] = {}
        for camera, slice_ in sorted(videos.items()):
            path = root / str(slice_.get("path", ""))
            if not path.exists():
                self.error = f"missing {path.name}"
                continue
            track = _Track(path, float(slice_.get("from", 0.0)),
                           float(slice_.get("to", 0.0)), max(frames, 1))
            try:
                track.open()
            except Exception as exc:
                self.error = f"{path.name}: {exc}"
                continue
            opened[camera] = track

        if not opened:
            return False
        with self._lock:
            self._tracks = opened
            self.episode = episode
            self.frames = max(frames, 1)
            self.position = 0
            self.playing = False
        return True

    def close(self) -> None:
        with self._lock:
            for track in self._tracks.values():
                track.close()
            self._tracks = {}
            self.episode = None
            self.frames = 0
            self.position = 0
            self.playing = False

    # --- playback ----------------------------------------------------------

    def cameras(self) -> list[str]:
        return sorted(self._tracks)

    def seek(self, index: int) -> None:
        self.position = max(0, min(int(index), self.frames - 1))

    def advance(self, elapsed_s: float, fps: int) -> None:
        """Move the cursor by wall-clock time. Stops at the end rather than looping."""
        if not self.playing:
            return
        step = int(elapsed_s * max(fps, 1))
        if step <= 0:
            return
        if self.position + step >= self.frames - 1:
            self.position = self.frames - 1
            self.playing = False
        else:
            self.position += step

    def frame(self, camera: str) -> Optional[np.ndarray]:
        track = self._tracks.get(camera)
        return None if track is None else track.frame(self.position)
