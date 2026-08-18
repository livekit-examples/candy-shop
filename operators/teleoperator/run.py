"""Teleoperator: fly the leslider from an SO-101 leader arm and record it.

Also the room's control desk: it discovers the other operators (move-to, policy,
reward), drives them over their own RPCs, preempts them to take the arm, and hands it
back — see ``peers``. ``mimic`` drives the *leader* from the follower's pose while one of
them has the arm, so claiming mid-policy is a handover rather than a jump.

The review UI (``teleoperator-ui``) is a separate process driving this one over
``protocol``'s RPCs, so its repaints/crashes can't stall a recording. Run:
``uv run teleoperator`` (spawns the UI child; ``--no-ui`` records headless).

Terminal hotkeys: c=claim/release the arm, m=mimic, f=free the leader (torque off),
r=record, [=discard, t=set task, x=quit.
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

# The leslider leader imports the old `lerobot.types` path; the pinned lerobot
# has only `lerobot.lerobot_types`. Alias before importing the leader.
import lerobot.lerobot_types as _lerobot_types
sys.modules.setdefault("lerobot.types", _lerobot_types)

from lerobot_teleoperator_so101_with_slider import (
    SO101WithSliderLeader, SO101WithSliderLeaderConfig,
)

from operators.teleoperator import library, protocol, session, shortcuts
from operators.teleoperator.common import (
    camera_names, env, load_env, mint_token, pace, portal_config_path,
)
from operators.teleoperator.mimic import MimicController
from operators.teleoperator.peers import PeerControl
from operators.teleoperator.recorder import Recorder

logger = logging.getLogger(__name__)

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent

# A tick over this has stopped answering RPCs for that long: the leader's bus reads run
# on the event loop, so an over-budget body is the loop being unavailable, not merely a
# late frame. Well under the 20 s the window allows for an ack, so the warning arrives
# before the timeout does.
SLOW_TICK_S = 0.5


class Hotkeys:
    """Non-blocking single-letter key capture from the terminal's own stdin.

    cbreak mode (not a global OS hook), so hotkeys only fire while this terminal
    has focus. cbreak keeps ISIG on, so Ctrl-C still raises KeyboardInterrupt.
    No-op if stdin isn't a tty."""

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
        # A pause already in force (calibration reading stdin) owns the tty; the
        # reader parks itself and `resume()` takes cbreak once that's done.
        if not self._paused.is_set():
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
        """Hand stdin back to cooked mode so ``input()`` gets echo + line editing
        and no byte of it is swallowed as a hotkey. Valid before ``start()``."""
        if not self.enabled:
            return
        self._paused.set()
        if self._thread is not None:
            self._parked.wait(timeout=1.0)  # let the reader finish any in-flight read
        if self._original is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)

    def resume(self) -> None:
        if not self.enabled:
            return
        if self._original is not None:  # else `start()` hasn't taken the tty yet
            tty.setcbreak(self._fd)
        with self._lock:
            self._pending = []
        self._parked.clear()  # before _paused, so the next pause() can't see it stale
        self._paused.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                self._parked.set()  # signal pause() we've let go of stdin
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
        """Sleep while paused. Never touches stdin, so ``input()`` on the main
        thread owns it exclusively."""
        while self._paused.is_set() and not self._stop.is_set():
            time.sleep(0.05)


class JobRunner:
    """Runs one corpus mutation at a time on a dedicated thread.

    A rewrite (see ``library``) is too slow for an RPC handler or the event loop,
    which it would freeze for minutes. Handlers validate and ``submit``; this
    thread brackets the work in ``Recorder.suspend()``/``resume()`` and reports
    through ``busy``/``error``."""

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


class UiProcess:
    """Launch `teleoperator-ui` as a child of this process. On by default.

    The recorder stays the parent/foreground process, so closing the window
    leaves recording running and quitting the recorder takes the window down —
    neither leaves an orphan on the leader's serial port."""

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
        """Notice the window being closed, once. No respawn."""
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
        same venv; else run the module."""
        script = pathlib.Path(sys.executable).parent / "teleoperator-ui"
        if script.exists():
            return [str(script)]
        return [sys.executable, "-m", "operators.teleoperator.ui.app"]

    @staticmethod
    def display_available() -> bool:
        """Whether opening a window stands a chance — the default for `--ui`.

        Only the unambiguous headless case is detected (X11/Wayland env vars
        unset on Linux); without this the child dies in GLFW init every launch.
        macOS/Windows always claim a display; `--ui`/`--no-ui` override either way."""
        if sys.platform == "darwin" or sys.platform.startswith("win"):
            return True
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _metrics_snapshot(op: Operator) -> dict:
    """Portal's counters as plain JSON."""
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


def _report_health(recorder: Recorder, fps: int) -> None:
    """Surface the two ways a recording goes wrong: rows dropped (three causes,
    unrelated fixes) and rows kept but paired against a lagging observation."""
    causes = recorder.drop_causes
    last_ms, worst_ms = recorder.pairing_age_ms
    obs_fps = recorder.obs_fps

    if recorder.skipped_frames:
        print(f"[teleoperator] dropped {recorder.drain_skips()} rows in last 5s "
              f"({' '.join(f'{k}={v}' for k, v in causes.items() if v)}) — "
              f"obs {obs_fps:.1f}/s of {fps}, queue {recorder.queue_depth}, "
              f"pairing age last={last_ms:.0f}ms worst={worst_ms:.0f}ms")

        worst = max(causes, key=lambda k: causes[k])
        if worst == "error":
            print("[teleoperator]   -> add_frame is failing: a schema or disk problem, not "
                  "timing. The traceback above names it.")
        elif worst == "backlog":
            print("[teleoperator]   -> the dataset writer can't keep up — disk or video "
                  "encoder bound. Lower the camera resolution or fps, or write to a "
                  "faster disk.")
        elif worst == "unpaired":
            print("[teleoperator]   -> no observation old enough to pair with. Either the obs "
                  "stream is far slower than the action stream, or the action timestamps "
                  "come from a clock behind this machine's (time-sync the hosts).")

    # A stalled video no longer costs rows, so it would otherwise be silent here.
    # `last`, not `worst`: worst never resets within an episode, so it would keep
    # reporting a stall that has already passed.
    elif obs_fps and last_ms > 2000 / fps:
        print(f"[teleoperator] pairing lagging {last_ms:.0f}ms behind the action "
              f"(worst {worst_ms:.0f}ms this episode, obs {obs_fps:.1f}/s of {fps}). "
              f"Rows are still recorded; their observations are that far out of date.")

    if recorder.resized_frames:
        print(f"[teleoperator] {recorder.resized_frames} frames rescaled to the dataset's "
              f"resolution this episode — the encoder is shedding resolution under "
              f"congestion. Rows are kept, but their detail is upscaled.")


# Seconds between reconnect attempts; the last value repeats forever. The early ones
# are for a port that blipped, the cap is for a human walking back to the bench — and it
# never gives up, because the alternative is a recorder sitting idle next to a leader
# that has been plugged back in.
RECONNECT_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0)


class Runtime:
    """The leader arm and the dataset writer — the two things setup provides.

    They start empty so the recorder can join the room with nothing open and be
    configured afterwards. Everything on the hot path must tolerate `None`.

    `leader` is a *handle*, not the session: a serial port that drops is replaced in
    place by `connect_leader`, and `configured` deliberately doesn't look at it, so a
    reconnect leaves the corpus open and the window on the session screen rather than
    throwing the operator back to setup mid-recording."""

    def __init__(self) -> None:
        self.leader: Optional[SO101WithSliderLeader] = None
        self.recorder: Optional[Recorder] = None
        self.jobs: Optional[JobRunner] = None
        self.opening = ""      # what it's opening, "" when idle
        self.error = ""        # why the last attempt failed
        self.started_at = 0.0
        self.port = ""         # the port this session opened; what a reconnect re-opens
        self.leader_id = ""
        self.leader_error = ""   # why the link dropped, "" while it's healthy
        self.reconnecting = ""   # what the retry is doing, "" when idle
        self.attempts = 0        # reconnect attempts since the link dropped

    @property
    def configured(self) -> bool:
        """A session exists: a port was chosen and the dataset is open. Not the same
        as the leader answering right now — see the class docstring."""
        return bool(self.port) and self.recorder is not None

    def open(self, *, port: str, leader_id: str, recorder: Recorder) -> None:
        """Open the leader bus, then adopt the dataset. Blocking — call on a worker
        thread, never the event loop: the serial open blocks, and calibration
        (if the motors disagree with stored calibration) reads from the terminal."""
        self.port, self.leader_id = port, leader_id
        try:
            self.connect_leader()
        except Exception:
            # A port only counts as this session's once it has answered: `port` is what
            # a reconnect re-opens and what the window reads as "there is a link here",
            # and a first open that failed has neither — setup is its retry.
            self.port = ""
            raise
        self.recorder = recorder
        self.jobs = JobRunner(recorder)

    def connect_leader(self) -> None:
        """(Re)open the leader's serial bus, replacing whatever handle was there.

        Blocking, worker-thread only, same as `open`. The old handle goes first —
        best-effort, since the usual reason to be here is that it stopped answering —
        and the new one is published only once `connect()` returned, so the tick loop
        never sees a half-open bus.

        ``cruise``/``max`` velocity are the held-arrow speed and Up-arrow trim
        ceiling, in raw ticks/s."""
        self.drop_leader()
        leader = SO101WithSliderLeader(SO101WithSliderLeaderConfig(
            id=self.leader_id,
            port=self.port,
            cruise_velocity=int(env("TELEOP_CRUISE_VELOCITY", "1500")),
            max_velocity=int(env("TELEOP_MAX_VELOCITY", "3000")),
        ))
        leader.connect()
        self.leader = leader

    def detach_leader(self) -> Optional[SO101WithSliderLeader]:
        """Take the handle off the hot path and hand it back, without touching the bus.

        Split from closing it because closing is not cheap: lerobot drops torque on the
        way out, and on a bus that has stopped answering that is five retries against a
        serial timeout. On the event loop that would stall the tick and Portal's
        callbacks; the caller closes it on a worker instead.
        """
        old, self.leader = self.leader, None
        return old

    @staticmethod
    def close_leader(old: Optional[SO101WithSliderLeader]) -> None:
        """Best-effort disconnect. Never raises: a bus that already died refuses its own
        disconnect, and there is nothing left to do about it. Blocking."""
        if old is None:
            return
        try:
            old.disconnect()
        except Exception as exc:
            logger.warning("leader disconnect failed (bus already gone?): %s", exc)

    def drop_leader(self) -> None:
        """Detach and close, for callers already off the event loop."""
        self.close_leader(self.detach_leader())

    def close(self) -> None:
        try:
            if self.recorder is not None:
                self.recorder.finalize()
        finally:
            # Runs even if finalize raised: leaving the motors driven is its own
            # failure, and the caller still has an operator to disconnect.
            self.drop_leader()


# --- main -------------------------------------------------------------------

async def main(*, with_ui: Optional[bool] = None) -> None:
    load_env(PACKAGE_DIR)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "candy-shop")
    identity = env("TELEOPERATOR_IDENTITY", "teleoperator")
    config_path = portal_config_path(PACKAGE_DIR)
    cfg = OperatorConfig.from_yaml_file(config_path, room)
    fps = int(env("PORTAL_FPS", "30"))
    cameras = camera_names(cfg)

    env_port = os.environ.get("SO101_LEADER_PORT")
    env_repo_id = os.environ.get("DATASET_REPO_ID")
    env_root = os.environ.get("DATASET_ROOT")
    env_task = os.environ.get("DATASET_TASK")
    leader_id = env("SO101_LEADER_ID", "so101_leader")

    op = Operator(cfg)
    rt = Runtime()
    control = PeerControl(op)
    mimic = MimicController(
        fps=fps,
        align_s=float(env("TELEOP_MIMIC_ALIGN_S", "1.5")),
        smooth_s=float(env("TELEOP_MIMIC_SMOOTH_S", "0.08")),
        torque_limit=int(env("TELEOP_MIMIC_TORQUE_LIMIT", "500")),
    )

    # The same rebindable bindings the window uses, so a foot pedal works in
    # either place. `t`/`x` are terminal-only and stay fixed. Built here, before
    # anything can reach `open_runtime`, because the open worker pauses these to
    # hand stdin to calibration; `start()` comes later.
    keys = shortcuts.load()
    key_record = shortcuts.terminal_chars(keys, "record")
    key_discard = shortcuts.terminal_chars(keys, "discard")
    key_claim = shortcuts.terminal_chars(keys, "claim")
    key_release = shortcuts.terminal_chars(keys, "release")
    key_resume = shortcuts.terminal_chars(keys, "resume")
    key_mimic = shortcuts.terminal_chars(keys, "mimic")
    key_relax = shortcuts.terminal_chars(keys, "relax")
    key_reconnect = shortcuts.terminal_chars(keys, "reconnect")
    key_stop_all = shortcuts.terminal_chars(keys, "stop_all")
    bound = (key_record | key_discard | key_claim | key_release | key_resume
             | key_mimic | key_relax | key_reconnect | key_stop_all)
    # Bindings match before t/x below, so a binding on either would shadow it
    # silently — warn rather than leave you wondering why `t` stopped working.
    if clash := ({"t", "x"} & bound):
        print(f"[teleoperator] note: {sorted(clash)} is bound to a session action, "
              f"so it no longer sets the task / quits in this terminal")
    hotkeys = Hotkeys({"t", "x"} | bound)

    def build_recorder(repo_id: str, root, task: str) -> Recorder:
        return Recorder(
            fps=fps,
            state_field_names=[f.name for f in cfg.state_schema],
            action_field_names=[f.name for f in cfg.action_schema],
            cameras=cameras,
            repo_id=repo_id,
            task=task,
            root=root,
        )

    def open_runtime(port: str, repo_id: str, root, task: str) -> None:
        """Open on a worker thread and report through `rt`. Never the event loop:
        the serial open blocks, and calibration blocks on stdin."""
        rt.opening = f"opening {port}"
        rt.error = ""
        rt.started_at = time.monotonic()

        def work() -> None:
            # lerobot calibrates on this thread by reading stdin, so the hotkey
            # reader has to let go of the tty first or it eats the ENTER.
            hotkeys.pause()
            try:
                rt.open(port=port, leader_id=leader_id,
                        recorder=build_recorder(repo_id, root, task))
            except Exception as exc:
                logger.exception("could not open the leader arm on %s", port)
                rt.error = f"{port}: {exc}"
            else:
                print(f"[teleoperator] leader on {port}; dataset {repo_id} -> {root}")
            finally:
                hotkeys.resume()
                rt.opening = ""

        threading.Thread(target=work, name="open-runtime", daemon=True).start()

    # --- the leader link ------------------------------------------------------
    # One serial port carries the action read *and* mimic's torque writes, so either
    # failing means the same thing: the leader is gone. It used to mean the recorder
    # was gone too — the read raised straight out of the tick loop, taking the
    # in-flight episode with it — so both now funnel into `lost_leader`.

    link_lock = threading.Lock()
    link_wake = threading.Event()      # jumps the backoff when a human asks
    shutting_down = threading.Event()  # stops the retry racing `rt.close()`

    def lost_leader(detail: str) -> None:
        """Park the leader link and start bringing it back. Idempotent: a second report
        while a retry is running just hurries it along.

        Called from the tick loop, so nothing here touches the bus — the handle is
        detached and the worker closes it.
        """
        with link_lock:
            if rt.reconnecting:
                link_wake.set()
                return
            rt.reconnecting = f"reconnecting to {rt.port}"
            rt.leader_error = detail
            rt.attempts = 0
        # Mimic is told to forget rather than to switch off: switching off writes to a
        # port that is no longer there.
        mimic.forget()
        stale = rt.detach_leader()
        print(f"[teleoperator] leader link down ({detail}); reconnecting to {rt.port}")
        threading.Thread(target=reconnect_work, args=(stale,),
                         name="leader-reconnect", daemon=True).start()

    def reconnect_work(stale: Optional[SO101WithSliderLeader]) -> None:
        """Close the dead handle, then retry the bus until it answers. Worker thread:
        both ends of this block — a disconnect on a dead bus waits out its timeouts, and
        lerobot may stop to calibrate, which reads stdin."""
        rt.close_leader(stale)
        attempt = 0
        while not shutting_down.is_set():
            attempt += 1
            rt.attempts = attempt
            rt.reconnecting = f"reconnecting to {rt.port} (attempt {attempt})"
            hotkeys.pause()  # calibration would otherwise lose its ENTER to the reader
            try:
                rt.connect_leader()
            except Exception as exc:
                rt.leader_error = f"{rt.port}: {exc}"
                logger.warning("leader reconnect attempt %d failed: %s", attempt, exc)
            else:
                rt.leader_error, rt.attempts, rt.reconnecting = "", 0, ""
                if shutting_down.is_set():
                    # Won the race against `rt.close()`; nothing will ever read this
                    # handle, so close it here rather than leaking the port.
                    rt.drop_leader()
                    return
                print(f"[teleoperator] leader back on {rt.port} — mimic is off; switch "
                      f"it on again once the arm is in your hand")
                return
            finally:
                hotkeys.resume()
            delay = RECONNECT_BACKOFF_S[min(attempt, len(RECONNECT_BACKOFF_S)) - 1]
            # Once a minute at the cap: this can run for as long as the cable is out,
            # and a line per second would bury everything else in the terminal.
            if attempt == 1 or attempt % 4 == 0:
                print(f"[teleoperator] leader still not answering ({rt.leader_error}); "
                      f"retrying every {delay:.0f}s — replug it, or hit Reconnect in "
                      f"the window to try now")
            link_wake.wait(delay)
            link_wake.clear()
        rt.reconnecting = ""

    def relax_leader(reason: str) -> None:
        """Torque off, mimic off. The escape hatch for an arm left stiff."""
        mimic.release(rt.leader)
        print(f"[teleoperator] leader torque off ({reason})")

    # Both callbacks must stay O(1) so Portal's video-receive worker reacquires
    # the GIL promptly.
    latest_obs: Optional[Observation] = None

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs
        latest_obs = obs
        if rt.recorder is not None:
            rt.recorder.observe(obs)

    def on_action(action: Action) -> None:
        # One executed action (any sender) = one recorded row.
        if rt.recorder is not None:
            rt.recorder.record(action)

    op.on_observation(on_observation)
    op.on_action(on_action)
    op.on_active_operator_changed(lambda i: print(f"[teleoperator] active operator now: {i}"))
    op.on_operator_joined(lambda i: print(f"[teleoperator] operator joined: {i}"))
    op.on_operator_left(lambda i: print(f"[teleoperator] operator left: {i}"))

    # --- taking and giving back the arm --------------------------------------
    # Shared by the RPC surface and the terminal hotkeys, so the two can't drift on what
    # a claim means.

    def fire(coro, label: str) -> None:
        """Run a control action alongside the tick instead of inside it.

        Every one of these waits on other peers' RPCs — a claim stops three operators
        before it returns — and awaiting that in the tick would stop the leader's action
        stream for as long as the slowest of them takes to answer.
        """
        asyncio.create_task(coro, name=label)

    async def take_arm(reason: str) -> list[str]:
        """Claim the arm: free the leader, preempt every peer, hold the pointer.

        Taking the arm is a human stepping into whatever was running, so the leader goes
        slack first and stays slack — a torqued leader is one you have to fight to fly.
        The flip side is that claiming means a hand on the leader: nothing holds it up
        once torque is off, and the robot follows it down. See `mimic`.
        """
        mimic.yield_to_human(rt.leader)
        suspended = await control.claim(op.local_identity())
        print(f"[teleoperator] arm claimed ({reason})"
              + (f"; suspended {', '.join(suspended)} — resume restarts it"
                 if suspended else ""))
        return suspended

    async def release_arm(reason: str) -> None:
        await control.release()
        print(f"[teleoperator] arm released ({reason})")

    async def resume_arm(reason: str) -> list[str]:
        resumed = await control.resume()
        print(f"[teleoperator] resumed {', '.join(resumed) or 'nothing'} ({reason})")
        return resumed

    async def stop_everything(reason: str) -> tuple[list[str], Optional[str]]:
        """The panic path: preempt every peer, then fold the arm.

        In that order — the robot's reset self-claims, so an operator whose loop is still
        alive would take the pointer straight back and carry on.
        """
        stopped = await control.stop_all(remember=False)
        folded = await control.fold_arm()
        print(f"[teleoperator] stop all ({reason}): {', '.join(stopped) or 'nothing'}"
              f"{f'; fold failed: {folded}' if folded else '; arm folded'}")
        return stopped, folded

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

    def leader_snapshot() -> dict:
        """The link, not the handle: `connected` false with `state` reconnecting is a
        session that is still set up and still holds its corpus."""
        return {
            "connected": rt.leader is not None,
            "port": rt.port,
            "state": ("open" if rt.leader is not None
                      else "reconnecting" if rt.reconnecting else "down"),
            "detail": rt.leader_error or rt.reconnecting,
            "attempts": rt.attempts,
        }

    def status() -> dict:
        """Always answers, configured or not — the setup screen needs a reply."""
        base = {
            "identity": op.local_identity(),
            "configured": rt.configured,
            "opening": rt.opening,
            "open_error": rt.error,
            "cameras": list(cameras),
            "fps": fps,
            "robot": op.robot_identity(),
            "active_operator": op.active_operator(),
            "claiming": control.claiming,
            "peers": control.snapshot(),
            "suspended": control.suspended,
            "mimic": mimic.snapshot(),
            "leader": leader_snapshot(),
        }
        recorder, jobs = rt.recorder, rt.jobs
        if recorder is None or jobs is None:
            return {**base, "ready": False, "recording": False, "saving": False,
                    "busy": "", "error": "", "task": "", "episodes": 0, "rows": 0,
                    "dropped": 0, "drop_causes": {}, "obs_fps": 0.0, "resized": 0,
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
            "resized": recorder.resized_frames,
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
        # Strip video offsets (per-camera, would ~double the page);
        # METHOD_EPISODE_VIDEO serves them per viewed episode.
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
        suspended = await take_arm(f"claimed by '{data.caller_identity}'")
        return reply(active=op.local_identity(), suspended=suspended)

    async def rpc_release(data: RpcInvocationData) -> str:
        await release_arm(f"by '{data.caller_identity}'")
        return reply(active=None)

    async def rpc_resume(data: RpcInvocationData) -> str:
        return reply(resumed=await resume_arm(f"by '{data.caller_identity}'"))

    async def rpc_peer_run(data: RpcInvocationData) -> str:
        body = payload(data)
        identity = str(body.get("identity", "")).strip()
        if refusal := control.run(identity, str(body.get("payload", "")).strip()):
            return refuse(refusal)
        return reply(identity=identity)

    async def rpc_peer_stop(data: RpcInvocationData) -> str:
        identity = str(payload(data).get("identity", "")).strip()
        if identity:
            if refusal := await control.stop(identity):
                return refuse(refusal)
            return reply(identity=identity)
        stopped, folded = await stop_everything(f"by '{data.caller_identity}'")
        if folded:
            return refuse(folded)
        return reply(stopped=stopped)

    async def rpc_mimic(data: RpcInvocationData) -> str:
        if rt.leader is None:
            return refuse(rt.reconnecting or "the leader arm is not open yet")
        mimic.set_enabled(bool(payload(data).get("enabled")), rt.leader)
        print(f"[teleoperator] mimic {'on' if mimic.enabled else 'off'} "
              f"(requested by '{data.caller_identity}')")
        return reply(mimic=mimic.snapshot())

    async def rpc_relax(data: RpcInvocationData) -> str:
        if rt.leader is None:
            return refuse("the leader arm is not connected")
        relax_leader(f"asked by '{data.caller_identity}'")
        return reply(mimic=mimic.snapshot())

    async def rpc_reconnect(data: RpcInvocationData) -> str:
        if not rt.port:
            return refuse("nothing to reconnect to — choose a port first")
        # Also the way to cycle a leader that is up but misbehaving, so this doesn't
        # check whether the link is down first.
        lost_leader(f"reconnect asked by '{data.caller_identity}'")
        return reply(leader=leader_snapshot())

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
        (protocol.METHOD_RESUME, rpc_resume),
        (protocol.METHOD_PEER_RUN, rpc_peer_run),
        (protocol.METHOD_PEER_STOP, rpc_peer_stop),
        (protocol.METHOD_MIMIC, rpc_mimic),
        (protocol.METHOD_RELAX, rpc_relax),
        (protocol.METHOD_RECONNECT, rpc_reconnect),
        (protocol.METHOD_RELABEL, rpc_relabel),
        (protocol.METHOD_DELETE, rpc_delete),
    ):
        op.register_rpc_method(method, handler)

    # ATTR_ROLE lets `teleoperator-ui` find this peer; not under `vla_demo.*` (see protocol).
    attrs = {protocol.ATTR_ROLE: protocol.ROLE_RECORDER}

    print(f"[teleoperator] wire contract: {config_path}")
    print(f"[teleoperator] connecting to {url} as '{identity}' in room '{room}' ...")
    await op.connect(url, mint_token(
        identity, room, name="Teleoperator (leader arm)", attributes=attrs,
    ))
    me = op.local_identity()

    print(f"[teleoperator] connected as '{me}' @ {fps} fps")
    # Spawn only once we're in the room, so the UI's first poll finds us.
    ui = UiProcess()
    if with_ui is None:
        with_ui = UiProcess.display_available()
        if not with_ui:
            print("[teleoperator] no display detected (DISPLAY/WAYLAND_DISPLAY unset) — "
                  "headless; pass --ui to force the window")
    if with_ui:
        ui.start()

    # Open now if the environment fully specifies a session, else wait for setup.
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

    print(f"[teleoperator] hotkeys: {shortcuts.describe(keys, 'claim')}=claim/release arm  "
          f"{shortcuts.describe(keys, 'mimic')}=mimic  "
          f"{shortcuts.describe(keys, 'relax')}=free leader  "
          f"{shortcuts.describe(keys, 'record')}=record  "
          f"{shortcuts.describe(keys, 'discard')}=discard  t=set task  x=quit")
    hotkeys.start()
    quit_requested = False

    # Treat kill / terminal-close as a clean quit so `finally` runs finalize()
    # and the parquet footers land. Otherwise SIGTERM/SIGHUP leave footerless
    # files (repairable, but best avoided). SIGINT already raises KeyboardInterrupt.
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

    worst_tick = 0.0
    try:
        async for tick in pace(fps):
            tick_started = time.perf_counter()
            recorder, jobs = rt.recorder, rt.jobs
            for key in hotkeys.pop():
                if key in key_claim:
                    # One key, both directions: holding the arm and pressing it again
                    # gives it back, which is what `c` did before as a ring of one.
                    fire(release_arm("hotkey") if control.claiming
                         else take_arm("hotkey"), "claim-key")
                elif key in key_release:
                    fire(release_arm("hotkey"), "release-key")
                elif key in key_resume:
                    fire(resume_arm("hotkey"), "resume-key")
                elif key in key_mimic:
                    if rt.leader is None:
                        print("[teleoperator] mimic needs the leader arm open")
                    else:
                        mimic.set_enabled(not mimic.enabled, rt.leader)
                        print(f"[teleoperator] mimic {'on' if mimic.enabled else 'off'}")
                elif key in key_relax:
                    if rt.leader is None:
                        print("[teleoperator] no leader arm to release")
                    else:
                        relax_leader("hotkey")
                elif key in key_reconnect:
                    if not rt.port:
                        print("[teleoperator] nothing to reconnect to — choose a port first")
                    else:
                        lost_leader("reconnect asked from the terminal")
                elif key in key_stop_all:
                    fire(stop_everything("hotkey"), "stop-all-key")
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
                        # Pause capture so typed chars aren't eaten as commands,
                        # and read off-thread so the tick keeps pacing.
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

            # Hold the pointer against every operator that clears it on the way out.
            control.reassert(me)

            # Always send once the arm is open (even during a job): the robot
            # gates on the active operator, so keeping the leader in sync makes
            # takeover instant. Only recording pauses during a rewrite.
            leader = rt.leader
            if leader is not None:
                try:
                    action = leader.get_action()
                    op.send_action(action, timestamp_us=int(time.time() * 1_000_000))
                except Exception as exc:
                    # Unplugged, or lerobot's own ESC handler closed the bus under us.
                    # Uncaught this ended the process, discarding the in-flight episode.
                    lost_leader(f"reading the leader failed: {exc}")
                else:
                    # Mimic only makes sense while somebody else drives: the leader can't
                    # be driven and be doing the driving at the same time.
                    #
                    # Off the raw state stream, not `latest_obs`: mimic wants joint positions
                    # as fresh as the wire has them, and an observation is a state that also
                    # found frames to pair with — it can be a tick or two behind, or absent
                    # entirely when the video runs ahead of the sync window.
                    robot_state = op.get_state()
                    mimic.update(leader, robot_state.values if robot_state else None,
                                 allowed=not control.claiming)
                    if mimic.failure:
                        # Mimic only ever fails on the bus, and a bus that refuses a
                        # torque write is the link the next read would fail on anyway.
                        lost_leader(mimic.failure)

            # Lazy-build the dataset once a frame reveals the resolution.
            if (recorder is not None and not recorder.is_ready
                    and not recorder.is_suspended and latest_obs is not None
                    and all(latest_obs.frames.get(c) is not None for c in cameras)):
                f0 = latest_obs.frames[cameras[0]]
                if recorder.ensure_dataset(frame_bytes_to_numpy_rgb(f0.data, f0.width, f0.height)):
                    print(f"[teleoperator] dataset ready (episodes so far={recorder.episode_count})")

            worst_tick = max(worst_tick, time.perf_counter() - tick_started)

            # Every 5s: surface dropped rows, and notice the UI window closing.
            if tick % (fps * 5) == 0:
                if worst_tick > SLOW_TICK_S:
                    print(f"[teleoperator] slow tick: {worst_tick * 1000:.0f}ms worst in "
                          f"the last 5s (budget {1000 / fps:.0f}ms) — the leader bus is "
                          f"blocking the event loop, so RPCs from the window and claims "
                          f"can time out. Check the leader's USB link.")
                worst_tick = 0.0
                if recorder is not None and recorder.is_recording:
                    _report_health(recorder, fps)
                ui.poll()

    except KeyboardInterrupt:
        print("\n[teleoperator] stopping ...")
    finally:
        shutting_down.set()
        link_wake.set()  # so a retry sleeping on the backoff notices and stops
        hotkeys.stop()
        ui.stop()
        # Before the bus closes: leaving the leader torqued would hold it stiff for
        # whoever picks it up next.
        mimic.free(rt.leader)
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
    # force: importing lerobot installs a root handler, which would make this a no-op
    # and leave the root logger at WARNING — every INFO line below silently dropped.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    asyncio.run(main(with_ui=args.ui))


if __name__ == "__main__":
    cli()
