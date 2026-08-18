"""Room viewer + RPC client, on a background asyncio thread.

The ImGui thread never awaits: it reads snapshots (``status``, ``episodes``,
``frame``) and posts commands (``call``), so a stalled RPC shows stale numbers
rather than freezing the window.

Joins as a plain LiveKit participant, not a Portal peer: joining as an operator
would land a viewer in the room's operator list, where a `c` ring could cycle arm
control onto a window that sends no actions.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from typing import Any, Optional, Sequence

import numpy as np
from livekit import rtc

from operators.teleoperator import protocol

logger = logging.getLogger(__name__)

# One minute floor, matching the rest of the operators: these cross a relay to a peer
# that may be busy driving the arm, and a premature abort is worse than waiting. Probing
# a peer that is not the teleoperator still fails fast — that returns an error rather
# than hanging, so the timeout is not what bounds it.
RPC_TIMEOUT_S = 60.0
# The poll is not a command: `recorder_status` and friends are O(1) handlers, so a slow
# one means the teleoperator's loop is busy, not that the work is long. Failing fast and
# retrying on the next tick beats parking the whole poll loop — which is serial — on one
# call for a minute and showing a minute of frozen numbers.
POLL_TIMEOUT_S = 10.0
# How long a request may take to reach the peer and have its *ack* come back, before
# `response_timeout` even starts counting. The SDK default is 7 s, which the tick loop
# can exceed while it drives the leader — and the failure reads as "Connection timeout",
# nothing to do with the timeouts above. Sized for a peer that is briefly busy, not for
# one that is gone: discovery already removes those.
ACK_TIMEOUT_S = 20.0


class RecorderClient:
    """Live view of one teleoperator, plus the commands that drive it."""

    def __init__(
        self,
        *,
        url: str,
        token: str,
        room: str,
        cameras: Sequence[str],
        target: Optional[str] = None,
        poll_hz: float = 4.0,
    ) -> None:
        self._url = url
        self._token = token
        self._room_name = room
        self._cameras = tuple(cameras)
        self._pinned_target = target
        self._poll_interval = 1.0 / max(poll_hz, 0.5)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._room: Optional[rtc.Room] = None
        self._stop = threading.Event()

        # State the ImGui thread reads: replaced by whole assignment, never mutated
        # in place, so a reader sees a self-consistent value without a lock.
        self._status: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._episode_video: dict[str, Any] = {}
        self._setup_options: dict[str, Any] = {}
        self._episodes: list[dict] = []
        self._target: Optional[str] = None
        self._connection = "connecting"
        self._notice = ""
        # A refusal is the teleoperator's own answer and stands until dismissed; a
        # transport complaint is only true until the next reply proves otherwise.
        self._notice_sticky = False

        # A lock here only because a slot is a (seq, array) pair and a torn read
        # would mismatch them; held just for the dict assignment.
        self._frames_lock = threading.Lock()
        self._frames: dict[str, tuple[int, np.ndarray]] = {}
        self._seq = 0
        self._video_tasks: dict[str, asyncio.Task] = {}

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._thread_main, name="lk-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)  # wake the loop
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception:
            logger.exception("client thread died")
            self._connection = "error"
        finally:
            with contextlib.suppress(Exception):
                self._loop.close()

    # --- snapshot reads (ImGui thread) ---------------------------------------

    @property
    def status(self) -> dict[str, Any]:
        return self._status

    @property
    def metrics(self) -> dict[str, Any]:
        """Portal's counters. `{}` until the first reply lands."""
        return self._metrics

    @property
    def setup_options(self) -> dict[str, Any]:
        """Ports and corpora the teleoperator can see. `{}` until requested."""
        return self._setup_options

    def request_setup_options(self) -> None:
        loop = self._loop
        if loop is None or self._target is None:
            return
        asyncio.run_coroutine_threadsafe(self._fetch_setup_options(), loop)

    async def _fetch_setup_options(self) -> None:
        reply = await self._call(protocol.METHOD_SETUP_OPTIONS)
        if reply is not None:
            reply.pop("ok", None)
            self._setup_options = reply

    @property
    def episode_video(self) -> dict[str, Any]:
        """Reply to the last `request_episode_video`; carries its own `episode` index so a stale reply is easy to ignore."""
        return self._episode_video

    def request_episode_video(self, index: int) -> None:
        """Ask where an episode's footage lives. Fire-and-forget; the answer shows up in `episode_video`."""
        loop = self._loop
        if loop is None or self._target is None:
            return
        asyncio.run_coroutine_threadsafe(self._fetch_episode_video(index), loop)

    async def _fetch_episode_video(self, index: int) -> None:
        reply = await self._call(protocol.METHOD_EPISODE_VIDEO, {"episode": index})
        if reply is not None:
            reply.pop("ok", None)
            self._episode_video = reply

    @property
    def episodes(self) -> list[dict]:
        return self._episodes

    @property
    def target(self) -> Optional[str]:
        """Identity of the teleoperator we're driving, or None if none was found."""
        return self._target

    @property
    def connection(self) -> str:
        """`connecting` | `connected` | `no teleoperator` | `error`."""
        return self._connection

    @property
    def notice(self) -> str:
        """The last command's refusal or transport error; "" when all is well."""
        return self._notice

    def clear_notice(self) -> None:
        self._notice = ""
        self._notice_sticky = False

    def _set_notice(self, message: str, *, sticky: bool) -> None:
        self._notice, self._notice_sticky = message, sticky

    def frame(self, camera: str) -> Optional[tuple[int, np.ndarray]]:
        """Latest `(sequence, HxWx3 uint8 RGB)` for `camera`, or None. The
        sequence lets the renderer skip re-uploading unchanged pixels."""
        with self._frames_lock:
            return self._frames.get(camera)

    # --- commands (ImGui thread) ---------------------------------------------

    def call(self, method: str, payload: Optional[dict] = None) -> None:
        """Fire an RPC without waiting; a refusal lands in `notice`, state changes show up in the next poll."""
        loop = self._loop
        if loop is None or self._target is None:
            self._set_notice("not connected to a teleoperator", sticky=False)
            return
        asyncio.run_coroutine_threadsafe(self._call(method, payload), loop)

    async def _call(self, method: str, payload: Optional[dict] = None,
                    *, timeout: float = RPC_TIMEOUT_S) -> Optional[dict]:
        target, room = self._target, self._room
        if target is None or room is None:
            return None
        try:
            raw = await room.local_participant.perform_rpc(
                destination_identity=target,
                method=method,
                payload=json.dumps(payload or {}),
                response_timeout=timeout,
                max_round_trip_latency=ACK_TIMEOUT_S,
            )
        except Exception as exc:
            self._set_notice(f"{method}: {exc}", sticky=False)
            return None
        try:
            reply = json.loads(raw)
        except json.JSONDecodeError:
            self._set_notice(f"{method}: malformed reply", sticky=False)
            return None
        if not reply.get("ok"):
            self._set_notice(reply.get("error") or f"{method} refused", sticky=True)
            return None
        # A reply is proof the link is up, so a transport complaint from a moment ago is
        # no longer true — otherwise one stall leaves the banner up for the whole session.
        if not self._notice_sticky:
            self._notice = ""
        return reply

    # --- the background loop -------------------------------------------------

    async def _run(self) -> None:
        room = rtc.Room()
        self._room = room
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("participant_connected", lambda p: self._rediscover())
        room.on("participant_disconnected", lambda p: self._rediscover())
        room.on("participant_attributes_changed", lambda *_: self._rediscover())

        try:
            await room.connect(self._url, self._token)
        except Exception as exc:
            self._connection = "error"
            self._notice = f"connect failed: {exc}"
            return

        self._rediscover()
        try:
            await self._poll_forever()
        finally:
            for task in list(self._video_tasks.values()):
                task.cancel()
            with contextlib.suppress(Exception):
                await room.disconnect()

    def _rediscover(self) -> None:
        """Find the teleoperator by attribute, so renaming one doesn't break the UI.
        A pinned `TELEOPERATOR_UI_TARGET` wins, for when two share a room."""
        room = self._room
        if room is None:
            return
        if self._pinned_target:
            self._target = self._pinned_target
            self._connection = "connected"
            return
        for participant in room.remote_participants.values():
            if participant.attributes.get(protocol.ATTR_ROLE) == protocol.ROLE_RECORDER:
                if self._target != participant.identity:
                    logger.info("found teleoperator '%s'", participant.identity)
                self._target = participant.identity
                self._connection = "connected"
                return
        self._target = None
        self._connection = "no teleoperator"

    async def _poll_forever(self) -> None:
        last_revision = -1
        tick = 0
        # Metrics are counters, not state — a slower cadence keeps the status poll cheap.
        metrics_every = max(int(1.0 / self._poll_interval), 1)
        while not self._stop.is_set():
            if self._target is not None:
                if tick % metrics_every == 0:
                    if (metrics := await self._call(
                            protocol.METHOD_METRICS, timeout=POLL_TIMEOUT_S)) is not None:
                        metrics.pop("ok", None)
                        self._metrics = metrics
                tick += 1
                status = await self._call(protocol.METHOD_STATUS, timeout=POLL_TIMEOUT_S)
                if status is not None:
                    status.pop("ok", None)
                    self._status = status
                    # Refetch only when revision changes, else a 4 Hz poll drags the
                    # whole corpus over the wire 4x a second.
                    revision = int(status.get("revision", 0))
                    if revision != last_revision:
                        if await self._refetch_episodes(int(status.get("episodes", 0))):
                            last_revision = revision
            await asyncio.sleep(self._poll_interval)

    async def _refetch_episodes(self, total: int) -> bool:
        """Page the whole index in. False if a page failed, so the caller retries
        rather than cache a partial list."""
        collected: list[dict] = []
        offset = 0
        while offset < total:
            page = await self._call(
                protocol.METHOD_EPISODES,
                {"offset": offset, "limit": protocol.EPISODE_PAGE_LIMIT},
                timeout=POLL_TIMEOUT_S,
            )
            if page is None:
                return False
            batch = page.get("episodes") or []
            if not batch:
                break  # the teleoperator's list shrank under us; take what we have
            collected.extend(batch)
            offset += len(batch)
        self._episodes = collected
        return True

    # --- video ---------------------------------------------------------------

    def _on_track_subscribed(self, track, publication, participant) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        name = publication.name
        if name not in self._cameras:
            return
        loop = self._loop
        if loop is None:
            return
        self._video_tasks[name] = loop.create_task(self._pump_video(name, track))

    def _on_track_unsubscribed(self, track, publication, participant) -> None:
        if task := self._video_tasks.pop(publication.name, None):
            task.cancel()

    async def _pump_video(self, camera: str, track) -> None:
        """Decode one camera into a latest-wins slot: the renderer only ever wants the newest frame."""
        stream = rtc.VideoStream(track)
        try:
            async for event in stream:
                converted = event.frame.convert(rtc.VideoBufferType.RGB24)
                array = np.frombuffer(converted.data, dtype=np.uint8).reshape(
                    converted.height, converted.width, 3
                )
                self._seq += 1
                with self._frames_lock:
                    self._frames[camera] = (self._seq, array)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("video stream for %s ended", camera)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
