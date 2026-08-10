"""The teleoperator: fly the leslider from an SO-101 leader arm and record it.

Joins the LiveKit room as a Portal operator, reads one action per tick from the
leader, and forwards it on the wire. The six leader joints mirror to the
follower's arm ``.pos``; the leader's arrow keys drive ``slider.vel`` (the
slider runs in velocity mode). While recording, every executed action the robot
forwards — this leader's *and* a remote policy's, tagged by ``Action.sender`` —
is paired with the observation it was responding to and written to a
LeRobotDataset (see ``recorder``). So a session captures human demonstrations,
policy rollouts, and HITL corrections in one corpus, each row carrying
``slider.vel`` alongside the six arm ``.pos``.

This process owns the dataset and never renders anything. The review UI
(``teleoperator-ui``) is a **separate process** driving it over ``protocol``'s RPCs,
so nothing it does — repainting, crashing — can stall a recording; the two share
no interpreter and no GIL. This process spawns it as a child (see ``UiProcess``)
so one command starts both; ``--no-ui`` records headless, which is also the
automatic choice when no display is detected.

Setup — which serial port, which corpus — happens in the window, not here: the
recorder joins the room with nothing open and waits (see ``Runtime``). A fully
specified environment opens immediately instead, for unattended runs.

Terminal hotkeys, for driving a session once it is open:
  c  cycle the active operator through [self, *remote operators]; with no peers
     this just claims/releases control for self. When a policy is in the room,
     `c` is instant human <-> policy handoff.
  r  toggle episode recording.
  [  discard the in-flight episode.
  t  set the task label for subsequent episodes (typed in the terminal;
     refused while recording, so a task spans whole episodes).
  x  quit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable, Optional

from livekit.portal import (
    Action, Observation, Operator, OperatorConfig, RpcInvocationData,
    frame_bytes_to_numpy_rgb,
)

# lerobot renamed `lerobot.types` -> `lerobot.lerobot_types` (the `RobotAction`
# alias moved with it). The pinned lerobot has only the new path; the leslider
# leader package still imports the old one. Alias it before importing the leader
# so its top-level `from lerobot.types import RobotAction` resolves.
import lerobot.lerobot_types as _lerobot_types
sys.modules.setdefault("lerobot.types", _lerobot_types)

from lerobot_teleoperator_so101_with_slider import (
    SO101WithSliderLeader, SO101WithSliderLeaderConfig,
)

from operators.teleoperator import library, protocol, session, shortcuts
from operators.teleoperator.common import (
    camera_names, env, load_env, mint_token, pace, portal_config_path,
)
from operators.teleoperator.recorder import Recorder

logger = logging.getLogger(__name__)

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent


# --- hotkeys ----------------------------------------------------------------

class Hotkeys:
    """Non-blocking single-letter key capture from the terminal's own stdin.

    Reads keys off ``sys.stdin`` in cbreak mode rather than via a global OS
    keyboard hook, so hotkeys only fire while *this terminal* has focus — the
    OS only routes stdin to the focused terminal, so keys typed into other
    windows never reach us. (cbreak keeps ISIG on, so Ctrl-C still raises
    KeyboardInterrupt.) No-op if stdin isn't a tty (e.g. piped/backgrounded)."""

    def __init__(self, keys: set[str]) -> None:
        self._keys = keys
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._paused = threading.Event()
        self._parked = threading.Event()  # reader confirmed it's off stdin
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else -1
        self._original: Optional[list] = None

    @property
    def enabled(self) -> bool:
        return self._fd >= 0

    def start(self) -> None:
        if not self.enabled:
            print("[teleoperator] stdin is not a tty — keyboard hotkeys disabled "
                  "(drive it from `teleoperator-ui` instead)")
            return
        self._original = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._original is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)

    def pop(self) -> list[str]:
        with self._lock:
            keys, self._pending = self._pending, []
        return keys

    def pause(self) -> None:
        """Stop capturing keys and hand stdin back to cooked mode — for when
        the terminal is reading a line of input (e.g. typing a task) so
        ``input()`` gets echo + line editing and watched letters in the text
        aren't swallowed as hotkeys."""
        if not self.enabled:
            return
        self._paused.set()
        self._parked.wait(timeout=1.0)  # let the reader finish any in-flight read
        if self._original is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)

    def resume(self) -> None:
        if not self.enabled:
            return
        tty.setcbreak(self._fd)
        with self._lock:
            self._pending = []  # drop anything captured during the prompt
        self._paused.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                self._parked.set()  # signal pause() that we've let go of stdin
                self._paused_wait()
                continue
            self._parked.clear()
            r, _, _ = select.select([self._fd], [], [], 0.1)
            if not r:
                continue
            ch = os.read(self._fd, 1).decode("utf-8", "ignore")
            with self._lock:
                if not self._paused.is_set() and ch in self._keys:
                    self._pending.append(ch)

    def _paused_wait(self) -> None:
        """Sleep while paused, waking to re-check stop/paused. Never touches
        stdin so ``input()`` on the main thread owns it exclusively."""
        while self._paused.is_set() and not self._stop.is_set():
            time.sleep(0.05)


# --- corpus jobs ------------------------------------------------------------

class JobRunner:
    """Runs one corpus mutation at a time on a dedicated thread.

    A rewrite (see ``library``) is far too slow for an RPC handler — the caller
    would time out — and on the asyncio loop would freeze the leader's tick and
    Portal's frame delivery for minutes. So handlers validate and ``submit``;
    this thread brackets the work in ``Recorder.suspend()``/``resume()`` and
    reports through ``busy``/``error``, which the UI reads from the status RPC."""

    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.busy = ""
        self.error = ""

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def submit(self, label: str, work: Callable[[], Any]) -> bool:
        """Schedule `work`. False if a job is already running, or if an episode
        is in flight — a rewrite has to own the files."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._recorder.is_recording or self._recorder.is_saving:
                return False
            self.busy, self.error = label, ""
            self._thread = threading.Thread(
                target=self._run, args=(label, work), name="corpus-job", daemon=True,
            )
            self._thread.start()
        return True

    def _run(self, label: str, work: Callable[[], Any]) -> None:
        started = time.perf_counter()
        try:
            self._recorder.suspend()
            work()
        except Exception as exc:
            logger.exception("%s failed", label)
            self.error = f"{label} failed: {exc}"
        else:
            logger.info("%s done in %.1fs", label, time.perf_counter() - started)
        finally:
            try:
                self._recorder.resume()
            except Exception as exc:
                logger.exception("could not reopen the dataset after %s", label)
                self.error = f"{label}: dataset did not reopen: {exc}"
            self.busy = ""


# --- the UI, as a child process ---------------------------------------------

class UiProcess:
    """Launch `teleoperator-ui` as a child of this process. On by default.

    The recorder stays the parent and the foreground process, which settles
    lifetimes the tidy way round: closing the window leaves recording running
    (this terminal and its hotkeys are still live), quitting the recorder takes
    the window down, and neither leaves an orphan on the leader's serial port."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._reported_exit = False

    def start(self) -> None:
        command = self._command()
        if command is None:
            print("[teleoperator] could not find `teleoperator-ui` — run it yourself in "
                  "another terminal")
            return
        try:
            # DEVNULL: the recorder reads stdin in cbreak mode for hotkeys, and
            # two processes on one tty would race for every keystroke.
            self._proc = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        except OSError as exc:
            print(f"[teleoperator] could not start the UI: {exc}")
            return
        print(f"[teleoperator] review UI started (pid {self._proc.pid})")

    def poll(self) -> None:
        """Notice the window being closed, once. No respawn: reopening a window
        you just closed is worse than saying how."""
        if self._proc is None or self._reported_exit:
            return
        if self._proc.poll() is None:
            return
        self._reported_exit = True
        code = self._proc.returncode
        detail = "" if code == 0 else f" (exit {code})"
        print(f"[teleoperator] review UI closed{detail} — still recording; "
              "`uv run teleoperator-ui` reattaches, or x quits")

    def stop(self) -> None:
        """SIGTERM then wait, never a bare kill — the UI holds a room connection
        it should leave cleanly."""
        if self._proc is None or self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    @staticmethod
    def _command() -> Optional[list[str]]:
        """The console script beside this interpreter, so the child lands in the
        same venv however the recorder was launched; else run the module."""
        script = pathlib.Path(sys.executable).parent / "teleoperator-ui"
        if script.exists():
            return [str(script)]
        return [sys.executable, "-m", "operators.teleoperator.ui.app"]

    @staticmethod
    def display_available() -> bool:
        """Whether opening a window stands a chance — the default for `--ui`.

        Only the unambiguous headless case is detected: X11/Wayland env vars
        unset on Linux, which is what an ssh session or a systemd unit on a robot
        host looks like. Without this the child would start, fail to init GLFW,
        and die with a traceback across the recorder's output every launch.
        macOS/Windows always claim a display; `--ui`/`--no-ui` override either way.
        """
        if sys.platform == "darwin" or sys.platform.startswith("win"):
            return True
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _metrics_snapshot(op: Operator) -> dict:
    """Portal's counters as plain JSON.

    These are the numbers that answer the questions a recording session actually
    raises — round-trip time, how often a stale frame was reused, which track is
    holding up the sync, per-track jitter, and buffer evictions — none of which the
    recorder can infer on its own."""
    m = op.metrics()
    return {
        "rtt_ms": {
            "last": _us_ms(m.rtt.rtt_us_last),
            "mean": _us_ms(m.rtt.rtt_us_mean),
            "p95": _us_ms(m.rtt.rtt_us_p95),
        },
        "pings": {"sent": m.rtt.pings_sent, "pongs": m.rtt.pongs_received},
        "sync": {
            "observations": m.sync.observations_emitted,
            "stale_reused": m.sync.stale_observations_emitted,
            "states_dropped": m.sync.states_dropped,
            "match_p50_ms": _us_ms(m.sync.match_delta_us_p50),
            "match_p95_ms": _us_ms(m.sync.match_delta_us_p95),
            "blocker": m.sync.last_blocker_track or "",
        },
        "video": {
            track: {
                "received": m.transport.frames_received.get(track, 0),
                "jitter_ms": _us_ms(m.transport.frame_jitter_us.get(track)),
                "evicted": m.buffers.evictions.get(track, 0),
                "fill": m.buffers.video_fill.get(track, 0),
                "mbytes": round(m.transport.bytes_received.get(track, 0) / 1e6, 1),
            }
            for track in sorted(set(m.transport.frames_received) | set(m.buffers.video_fill))
        },
        "wire": {
            "states": m.transport.states_received,
            "actions": m.transport.actions_received,
            "state_jitter_ms": _us_ms(m.transport.state_jitter_us),
            "action_jitter_ms": _us_ms(m.transport.action_jitter_us),
        },
    }


def _us_ms(value) -> Optional[float]:
    return None if value is None else round(value / 1000.0, 1)


def _report_drops(recorder: Recorder, max_obs_age_us: int, fps: int) -> None:
    """Say *why* rows are being dropped, and what to do about it.

    A bare count is useless: the four causes have unrelated fixes, and picking
    the wrong one costs a recording session."""
    dropped = recorder.drain_skips()
    causes = recorder.drop_causes
    last_ms, worst_ms = recorder.pairing_age_ms
    limit_ms = max_obs_age_us // 1000
    obs_fps = recorder.obs_fps
    print(f"[teleoperator] dropped {dropped} rows in last 5s "
          f"({' '.join(f'{k}={v}' for k, v in causes.items() if v)}) — "
          f"obs {obs_fps:.1f}/s of {fps}, queue {recorder.queue_depth}, "
          f"pairing age last={last_ms:.0f}ms worst={worst_ms:.0f}ms, limit={limit_ms}ms")

    worst = max(causes, key=lambda k: causes[k])
    if worst == "error":
        print("[teleoperator]   -> add_frame is failing: a schema or disk problem, not "
              "timing. The traceback above names it; MAX_OBS_AGE_MS won't help.")
    elif worst == "backlog":
        print("[teleoperator]   -> the dataset writer can't keep up — disk or video "
              "encoder bound. Lower the camera resolution or fps, or write to a "
              "faster disk.")
    elif worst == "unpaired":
        print("[teleoperator]   -> no observation old enough to pair with. Either the obs "
              "stream is far slower than the action stream, or the action timestamps "
              "come from a clock behind this machine's (time-sync the hosts).")
    elif worst == "stale":
        if not obs_fps:
            print("[teleoperator]   -> no observations are arriving at all.")
        elif obs_fps >= fps * 0.8:
            # The trap that cost me an hour: the average rate looks healthy, so
            # "video is slow" is wrong. Frames are arriving in bursts.
            print(f"[teleoperator]   -> observations average {obs_fps:.0f}/s but arrive in "
                  f"bursts: a gap of {worst_ms:.0f}ms is what got paired. Look for "
                  f"'video stream queue overflow' / 'buffer full' warnings above — "
                  f"that is the receive path stalling, not a slow camera.")
        else:
            print(f"[teleoperator]   -> observations arrive {1000 / obs_fps:.0f}ms apart but "
                  f"rows must pair within {limit_ms}ms. The robot's video is slow; "
                  f"raise MAX_OBS_AGE_MS only if you accept the looser alignment.")


# --- deferred setup ---------------------------------------------------------

class Runtime:
    """The leader arm and the dataset writer — the two things setup provides.

    They start empty so the recorder can join the room with nothing open and be
    configured afterwards, from the window or the terminal. Everything on the hot
    path therefore has to tolerate `None`, which is the price of not making you
    answer questions in a terminal before the UI even exists."""

    def __init__(self) -> None:
        self.leader: Optional[SO101WithSliderLeader] = None
        self.recorder: Optional[Recorder] = None
        self.jobs: Optional[JobRunner] = None
        self.opening = ""      # what it's opening, "" when idle
        self.error = ""        # why the last attempt failed
        self.started_at = 0.0

    @property
    def configured(self) -> bool:
        return self.leader is not None and self.recorder is not None

    def open(self, *, port: str, leader_id: str, recorder: Recorder) -> None:
        """Open the leader bus, then adopt the dataset. Blocking — call on a worker
        thread, never the event loop: a serial open takes a moment, and if the
        motors disagree with the stored calibration lerobot stops here to run its
        calibration routine, which reads from the teleoperator's terminal.

        The leslider leader mirrors the six arm joints and drives ``slider.vel``
        from its arrow keys (velocity mode); ``cruise``/``max`` velocity are the
        held-arrow speed and the Up-arrow trim ceiling, in raw ticks/s."""
        leader = SO101WithSliderLeader(SO101WithSliderLeaderConfig(
            id=leader_id,
            port=port,
            cruise_velocity=int(env("TELEOP_CRUISE_VELOCITY", "1500")),
            max_velocity=int(env("TELEOP_MAX_VELOCITY", "3000")),
        ))
        leader.connect()
        self.leader = leader
        self.recorder = recorder
        self.jobs = JobRunner(recorder)

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.finalize()
        if self.leader is not None:
            try:
                self.leader.disconnect()
            except Exception:
                logger.exception("leader disconnect failed")


# --- main -------------------------------------------------------------------

async def cycle_operator(op: Operator, me: str | None) -> None:
    ring = [me, *op.operators()]
    try:
        idx = ring.index(op.active_operator())
    except ValueError:
        idx = -1  # active operator left the ring, or was never set — snap to self
    nxt = ring[(idx + 1) % len(ring)]
    await op.set_active_operator(nxt)
    print(f"[teleoperator] active operator → {nxt}")


async def main(*, with_ui: Optional[bool] = None) -> None:
    load_env(PACKAGE_DIR)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "candy-shop")
    identity = env("TELEOPERATOR_IDENTITY", "teleoperator")
    config_path = portal_config_path(PACKAGE_DIR)
    cfg = OperatorConfig.from_yaml_file(config_path, room)
    fps = int(env("PORTAL_FPS", "30"))
    max_obs_age_us = int(env("MAX_OBS_AGE_MS", "100")) * 1000
    cameras = camera_names(cfg)

    env_port = os.environ.get("SO101_LEADER_PORT")
    env_repo_id = os.environ.get("DATASET_REPO_ID")
    env_root = os.environ.get("DATASET_ROOT")
    env_task = os.environ.get("DATASET_TASK")
    leader_id = env("SO101_LEADER_ID", "so101_leader")

    op = Operator(cfg)
    rt = Runtime()

    def build_recorder(repo_id: str, root, task: str) -> Recorder:
        return Recorder(
            fps=fps,
            state_field_names=[f.name for f in cfg.state_schema],
            action_field_names=[f.name for f in cfg.action_schema],
            cameras=cameras,
            max_obs_age_us=max_obs_age_us,
            repo_id=repo_id,
            task=task,
            root=root,
        )

    def open_runtime(port: str, repo_id: str, root, task: str) -> None:
        """Open on a worker thread and report through `rt`. Never on the event
        loop: the serial open blocks, and calibration blocks on stdin."""
        rt.opening = f"opening {port}"
        rt.error = ""
        rt.started_at = time.monotonic()

        def work() -> None:
            try:
                rt.open(port=port, leader_id=leader_id,
                        recorder=build_recorder(repo_id, root, task))
            except Exception as exc:
                logger.exception("could not open the leader arm on %s", port)
                rt.error = f"{port}: {exc}"
            else:
                print(f"[teleoperator] leader on {port}; dataset {repo_id} -> {root}")
            finally:
                rt.opening = ""

        threading.Thread(target=work, name="open-runtime", daemon=True).start()

    # Latest synced obs, read on the tick to lazy-build the recorder. Both
    # callbacks stay O(1) so Portal's video-receive tokio worker can reacquire
    # the GIL promptly — the UI is a separate process for exactly this reason.
    latest_obs: Optional[Observation] = None

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs
        latest_obs = obs
        if rt.recorder is not None:
            rt.recorder.observe(obs)

    def on_action(action: Action) -> None:
        # Drives recording: one executed action (any sender) = one row.
        if rt.recorder is not None:
            rt.recorder.record(action)

    op.on_observation(on_observation)
    op.on_action(on_action)
    op.on_active_operator_changed(lambda i: print(f"[teleoperator] active operator now: {i}"))
    op.on_operator_joined(lambda i: print(f"[teleoperator] operator joined: {i}"))
    op.on_operator_left(lambda i: print(f"[teleoperator] operator left: {i}"))

    # --- RPC surface (see protocol) -----------------------------------------
    # Handlers run on the asyncio loop, so each is O(1) or hands off.

    def reply(**fields: Any) -> str:
        return json.dumps({"ok": True, **fields})

    def refuse(error: str) -> str:
        return json.dumps({"ok": False, "error": error})

    def payload(data: RpcInvocationData) -> dict:
        try:
            parsed = json.loads(data.payload or "{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def status() -> dict:
        """Always answers, configured or not — the setup screen needs a reply too."""
        base = {
            "identity": op.local_identity(),
            "configured": rt.configured,
            "opening": rt.opening,
            "open_error": rt.error,
            "cameras": list(cameras),
            "fps": fps,
            "robot": op.robot_identity(),
            "active_operator": op.active_operator(),
        }
        recorder, jobs = rt.recorder, rt.jobs
        if recorder is None or jobs is None:
            return {**base, "ready": False, "recording": False, "saving": False,
                    "busy": "", "error": "", "task": "", "episodes": 0, "rows": 0,
                    "dropped": 0, "drop_causes": {}, "obs_fps": 0.0,
                    "queue_depth": 0, "pairing_age_ms": [0.0, 0.0], "revision": 0,
                    "repo_id": "", "root": ""}
        return {
            **base,
            "ready": recorder.is_ready,
            "recording": recorder.is_recording,
            "saving": recorder.is_saving,
            "busy": jobs.busy,
            "error": jobs.error,
            "task": recorder.task,
            "episodes": len(recorder.episodes),
            "rows": recorder.rows,
            "dropped": recorder.skipped_frames,
            "drop_causes": recorder.drop_causes,
            "obs_fps": round(recorder.obs_fps, 1),
            "queue_depth": recorder.queue_depth,
            "pairing_age_ms": [round(v, 1) for v in recorder.pairing_age_ms],
            "revision": recorder.revision,
            "repo_id": recorder.repo_id,
            "root": str(recorder.root),
        }

    def need_setup() -> str:
        return refuse("not set up yet — choose a port and a dataset first")

    async def rpc_status(data: RpcInvocationData) -> str:
        return reply(**status())

    async def rpc_metrics(data: RpcInvocationData) -> str:
        return reply(**_metrics_snapshot(op))

    async def rpc_setup_options(data: RpcInvocationData) -> str:
        # Scans serial ports and walks directories, so off the event loop.
        options = await asyncio.to_thread(
            session.setup_options, env_port, env_repo_id, env_task)
        return reply(**options)

    async def rpc_open(data: RpcInvocationData) -> str:
        if rt.configured:
            return refuse("already set up; restart the recorder to change it")
        if rt.opening:
            return refuse(rt.opening)
        body = payload(data)
        port = str(body.get("port", "")).strip()
        repo_id = str(body.get("repo_id", "")).strip()
        root = str(body.get("root", "")).strip()
        task = str(body.get("task", "")).strip() or session.DEFAULT_TASK
        if not port:
            return refuse("a serial port is required")
        if not session.valid_repo_id(repo_id):
            return refuse("repo id must look like org/name")
        if not root:
            return refuse("a dataset location is required")
        open_runtime(port, repo_id, pathlib.Path(root).expanduser(), task)
        return reply(port=port, repo_id=repo_id, root=root)

    async def rpc_episodes(data: RpcInvocationData) -> str:
        if rt.recorder is None:
            return reply(total=0, offset=0, revision=0, episodes=[])
        body = payload(data)
        offset = max(int(body.get("offset", 0) or 0), 0)
        limit = min(int(body.get("limit", protocol.EPISODE_PAGE_LIMIT) or 0),
                    protocol.EPISODE_PAGE_LIMIT)
        episodes = rt.recorder.episodes
        # Strip the video offsets: they're per-camera and would roughly double the
        # page. METHOD_EPISODE_VIDEO serves them for the one episode being viewed.
        page = [{k: v for k, v in e.items() if k != "videos"}
                for e in episodes[offset:offset + limit]]
        return reply(total=len(episodes), offset=offset, revision=rt.recorder.revision,
                     episodes=page)

    async def rpc_episode_video(data: RpcInvocationData) -> str:
        if rt.recorder is None:
            return need_setup()
        try:
            index = int(payload(data).get("episode"))
        except (TypeError, ValueError):
            return refuse("episode must be an integer")
        for episode in rt.recorder.episodes:
            if episode["index"] == index:
                return reply(episode=index, length=episode.get("length", 0),
                             videos=episode.get("videos") or {})
        return refuse(f"no such episode: {index}")

    async def rpc_start(data: RpcInvocationData) -> str:
        recorder, jobs = rt.recorder, rt.jobs
        if recorder is None or jobs is None:
            return need_setup()
        if jobs.running:
            return refuse(f"busy: {jobs.busy}")
        if not recorder.is_ready:
            return refuse("no observation yet — is the robot publishing?")
        if recorder.is_recording:
            return refuse("already recording")
        if not recorder.start_episode():
            return refuse("previous episode still saving — try again in a moment")
        print(f"[teleoperator] episode {recorder.episode_count} recording")
        return reply(episode=recorder.episode_count)

    async def rpc_stop(data: RpcInvocationData) -> str:
        recorder = rt.recorder
        if recorder is None:
            return need_setup()
        if not recorder.is_recording:
            return refuse("not recording")
        episode = recorder.episode_count
        recorder.end_episode()
        print(f"[teleoperator] episode {episode} saving in background")
        return reply(episode=episode)

    async def rpc_discard(data: RpcInvocationData) -> str:
        recorder = rt.recorder
        if recorder is None:
            return need_setup()
        if not recorder.is_recording:
            return refuse("not recording")
        recorder.discard_episode()
        print("[teleoperator] episode discarded")
        return reply()

    async def rpc_set_task(data: RpcInvocationData) -> str:
        recorder = rt.recorder
        if recorder is None:
            return need_setup()
        task = str(payload(data).get("task", "")).strip()
        if not task:
            return refuse("task must be a non-empty string")
        if recorder.is_recording:
            # Switching mid-episode would split one trajectory across two labels.
            return refuse("stop recording before changing the task")
        recorder.set_task(task)
        print(f"[teleoperator] task -> {task!r}")
        return reply(task=task)

    async def rpc_claim(data: RpcInvocationData) -> str:
        me = op.local_identity()
        await op.set_active_operator(me)
        print(f"[teleoperator] control claimed by '{data.caller_identity}' -> leader driving")
        return reply(active=me)

    async def rpc_release(data: RpcInvocationData) -> str:
        await op.set_active_operator(None)
        print(f"[teleoperator] control released by '{data.caller_identity}'")
        return reply(active=None)

    async def rpc_relabel(data: RpcInvocationData) -> str:
        recorder, jobs = rt.recorder, rt.jobs
        if recorder is None or jobs is None:
            return need_setup()
        raw = payload(data).get("episodes") or {}
        if not isinstance(raw, dict) or not raw:
            return refuse('expected {"episodes": {"<index>": "new task"}}')
        try:
            mapping = {int(k): str(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            return refuse("episode keys must be integers")
        if any(not v.strip() for v in mapping.values()):
            return refuse("a task label cannot be empty")
        label = f"relabelling {len(mapping)} episode(s)"
        if not jobs.submit(label, lambda: library.relabel_episodes(
            recorder.root, recorder.repo_id, mapping,
        )):
            return refuse(jobs.busy or "stop recording first")
        return reply(job="relabel")

    async def rpc_delete(data: RpcInvocationData) -> str:
        recorder, jobs = rt.recorder, rt.jobs
        if recorder is None or jobs is None:
            return need_setup()
        raw = payload(data).get("episodes") or []
        if not isinstance(raw, list) or not raw:
            return refuse('expected {"episodes": [<index>, ...]}')
        try:
            indices = sorted({int(i) for i in raw})
        except (TypeError, ValueError):
            return refuse("episode indices must be integers")
        label = f"deleting {len(indices)} episode(s)"
        if not jobs.submit(label, lambda: library.delete_episodes(
            recorder.root, recorder.repo_id, indices,
        )):
            return refuse(jobs.busy or "stop recording first")
        return reply(job="delete")

    for method, handler in (
        (protocol.METHOD_STATUS, rpc_status),
        (protocol.METHOD_EPISODES, rpc_episodes),
        (protocol.METHOD_METRICS, rpc_metrics),
        (protocol.METHOD_EPISODE_VIDEO, rpc_episode_video),
        (protocol.METHOD_SETUP_OPTIONS, rpc_setup_options),
        (protocol.METHOD_OPEN, rpc_open),
        (protocol.METHOD_START, rpc_start),
        (protocol.METHOD_STOP, rpc_stop),
        (protocol.METHOD_DISCARD, rpc_discard),
        (protocol.METHOD_SET_TASK, rpc_set_task),
        (protocol.METHOD_CLAIM, rpc_claim),
        (protocol.METHOD_RELEASE, rpc_release),
        (protocol.METHOD_RELABEL, rpc_relabel),
        (protocol.METHOD_DELETE, rpc_delete),
    ):
        op.register_rpc_method(method, handler)

    # Before any hotkey capture: if the motors' stored calibration disagrees with
    # the file for this leader `id`, lerobot's connect() runs its calibration
    # routine, which drives the operator through blocking `input()` prompts. Those
    # need the terminal in normal (cooked) mode, so `Hotkeys.start()` — which puts
    # stdin in cbreak — must not have run yet. It hasn't; keep it that way.
    # So `teleoperator-ui` can find this peer. Not under `vla_demo.*` — see protocol.
    attrs = {protocol.ATTR_ROLE: protocol.ROLE_RECORDER}

    print(f"[teleoperator] wire contract: {config_path}")
    print(f"[teleoperator] connecting to {url} as '{identity}' in room '{room}' ...")
    await op.connect(url, mint_token(
        identity, room, name="Teleoperator (leader arm)", attributes=attrs,
    ))
    me = op.local_identity()

    print(f"[teleoperator] connected as '{me}' @ {fps} fps")
    # Spawn only once we're in the room, so the UI's first poll finds us instead
    # of flashing "no recorder".
    ui = UiProcess()
    if with_ui is None:
        with_ui = UiProcess.display_available()
        if not with_ui:
            print("[teleoperator] no display detected (DISPLAY/WAYLAND_DISPLAY unset) — "
                  "headless; pass --ui to force the window")
    if with_ui:
        ui.start()

    # Open now if the environment already says everything, otherwise wait to be
    # told. Setup is the window's job; there is no terminal questionnaire.
    preset = session.from_env(env_port, env_repo_id, env_root, env_task)
    if preset is not None:
        print(f"[teleoperator] {preset.describe()}")
        open_runtime(preset.port, preset.repo_id, preset.root, preset.task)
    elif with_ui:
        print("[teleoperator] waiting for setup — choose a port and a dataset in the window")
    else:
        print("[teleoperator] waiting for setup — run `teleoperator-ui` (here or on another "
              "machine) to choose a port and a dataset, or set SO101_LEADER_PORT "
              "plus DATASET_REPO_ID/DATASET_ROOT to skip this")

    # The same rebindable bindings the window uses, so a foot pedal works in
    # either place. `t`/`x` are terminal-only (there is no window equivalent of
    # typing a task or quitting the process), so they stay fixed.
    keys = shortcuts.load()
    key_record = shortcuts.terminal_chars(keys, "record")
    key_discard = shortcuts.terminal_chars(keys, "discard")
    key_claim = shortcuts.terminal_chars(keys, "claim")
    # Bindings are matched before t/x below, so a binding on either would shadow
    # them silently. Say so rather than leave you wondering why `t` stopped working.
    if clash := ({"t", "x"} & (key_record | key_discard | key_claim)):
        print(f"[teleoperator] note: {sorted(clash)} is bound to a recording action, "
              f"so it no longer sets the task / quits in this terminal")
    hotkeys = Hotkeys({"t", "x"} | key_record | key_discard | key_claim)
    print(f"[teleoperator] hotkeys: {shortcuts.describe(keys, 'claim')}=cycle operator  "
          f"{shortcuts.describe(keys, 'record')}=record  "
          f"{shortcuts.describe(keys, 'discard')}=discard  t=set task  x=quit")
    hotkeys.start()
    quit_requested = False

    # Treat kill / terminal-close as a clean quit so the `finally` runs
    # recorder.finalize() and the parquet footers land. Without this, SIGTERM/
    # SIGHUP leave footerless files — repairable, but best avoided. SIGINT
    # already surfaces as KeyboardInterrupt.
    def _request_quit(signame: str) -> None:
        nonlocal quit_requested
        if not quit_requested:
            print(f"\n[teleoperator] received {signame}; finalizing dataset ...")
        quit_requested = True

    loop = asyncio.get_running_loop()
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            loop.add_signal_handler(_sig, _request_quit, _sig.name)
        except (NotImplementedError, RuntimeError):
            pass  # platform without add_signal_handler

    try:
        async for tick in pace(fps):
            recorder, jobs = rt.recorder, rt.jobs
            for key in hotkeys.pop():
                if key in key_claim:
                    await cycle_operator(op, me)
                elif key == "x":
                    quit_requested = True
                elif recorder is None or jobs is None:
                    print("[teleoperator] not set up yet — choose a port and a dataset")
                elif key in key_record:
                    if jobs.running:
                        print(f"[teleoperator] {jobs.busy} — wait for it to finish")
                    elif not recorder.is_ready:
                        print("[teleoperator] waiting for first observation before recording")
                    elif recorder.is_recording:
                        recorder.end_episode()
                        print(f"[teleoperator] episode {recorder.episode_count - 1} saving in background")
                    elif recorder.start_episode():
                        print(f"[teleoperator] episode {recorder.episode_count} recording")
                    else:
                        print("[teleoperator] previous episode still saving — try again in a moment")
                elif key in key_discard and recorder.is_recording:
                    recorder.discard_episode()
                    print("[teleoperator] episode discarded")
                elif key == "t":
                    if recorder.is_recording:
                        print("[teleoperator] stop recording (r) before changing the task")
                    else:
                        # Pause capture so a typed c/r/[/x/t isn't eaten as a
                        # command, and read off-thread so the tick keeps pacing.
                        hotkeys.pause()
                        try:
                            new_task = (await loop.run_in_executor(
                                None,
                                lambda: input(f"[teleoperator] new task (current: {recorder.task!r}): "),
                            )).strip()
                        finally:
                            hotkeys.resume()
                        if new_task:
                            recorder.set_task(new_task)
                            print(f"[teleoperator] task → {new_task!r}")
                        else:
                            print(f"[teleoperator] task unchanged ({recorder.task!r})")
            if quit_requested:
                break

            # Always send once the arm is open; the robot gates on the active
            # operator, so keeping the leader in sync makes takeover instant.
            # Including during a job — the arm stays live while the dataset is
            # rewritten; only recording pauses.
            if rt.leader is not None:
                op.send_action(rt.leader.get_action(),
                               timestamp_us=int(time.time() * 1_000_000))

            # Lazy-build the dataset once a frame reveals the resolution.
            if (recorder is not None and not recorder.is_ready
                    and not recorder.is_suspended and latest_obs is not None
                    and all(latest_obs.frames.get(c) is not None for c in cameras)):
                f0 = latest_obs.frames[cameras[0]]
                if recorder.ensure_dataset(frame_bytes_to_numpy_rgb(f0.data, f0.width, f0.height)):
                    print(f"[teleoperator] dataset ready (episodes so far={recorder.episode_count})")

            # Every 5s, surface dropped rows (stalled obs stream), and notice the
            # UI window being closed.
            if tick % (fps * 5) == 0:
                if recorder is not None and recorder.skipped_frames > 0:
                    _report_drops(recorder, max_obs_age_us, fps)
                ui.poll()

    except KeyboardInterrupt:
        print("\n[teleoperator] stopping ...")
    finally:
        hotkeys.stop()
        ui.stop()
        rt.close()
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    """Console-script entry point (`uv run teleoperator`)."""
    parser = argparse.ArgumentParser(
        prog="teleoperator",
        description="Fly the leslider from an SO-101 leader arm (slider on the "
                    "arrow keys) and record it.",
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="open the review window as a child process (default: yes, unless no "
             "display is detected). --no-ui records headless.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main(with_ui=args.ui))


if __name__ == "__main__":
    cli()
