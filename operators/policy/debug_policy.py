"""Standalone debug driver for the policy: load a checkpoint once, then start it,
stop it, and retype the instruction from the terminal. Its own operator
("policy-debug"); does NOT call the ``run_policy`` RPC.

Commands (typed at the ``policy>`` prompt):

    start [prompt]   run the policy, optionally retyping the instruction first
    stop             preempt the run and release active control
    prompt <text>    swap the instruction; a run in flight picks it up next tick
    status           connection / robot / run state
    <enter>          toggle: start if idle, stop if running
    quit             stop and exit (Ctrl-D and Ctrl-C work too)

While a run is in flight it prints one telemetry line every ~0.5 s: the tick
count, inference latency, the commanded arm pose, and how far the observed arm
is from that command -- so "the robot isn't moving" splits into *not inferring*,
*inferring but commanding the pose it is already in*, or *commanding a pose the
robot never reaches* (nobody applying the action, e.g. active control lost).

Only run ONE thing that takes active-operator control at a time -- do not run
`uv run policy` against the same room.

The native Portal/WebRTC logs are written to stderr by the Rust layer and will
chop up the prompt; quiet them with ``RUST_LOG`` or park stderr in a file.

Usage::

    uv run policy-debug --checkpoint outputs/pi0-candy/checkpoints/005000/pretrained_model
    RUST_LOG=error uv run policy-debug ...            # quiet native logs
    uv run policy-debug ... 2>/tmp/policy-debug.log   # or park them entirely
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import sys
import threading
import time

try:  # POSIX only; without them the prompt falls back to line-buffered input.
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = tty = None  # type: ignore[assignment]

from livekit.portal import Operator, OperatorConfig

from shared.common import env_str, load_env, mint_token, required_env
from shared.rest_pose import ARM_POS_KEYS

from operators.policy.run import CONFIG_PATH, PolicyRunner, add_policy_args, build_runner

IDENTITY = "policy-debug"
PROMPT = "policy> "
TELEMETRY_PERIOD_S = 0.5

HELP = """commands:
  start [prompt]   run the policy, optionally retyping the instruction first
  stop             preempt the run, release active control
  prompt <text>    swap the instruction (a run in flight picks it up next tick)
  status           connection / robot / run state
  <enter>          toggle start/stop
  quit             stop and exit"""

logger = logging.getLogger(__name__)


_EOF = object()


class _LineEditor:
    """Keeps what you are typing visible while the policy prints telemetry.

    A run emits a line every ~0.5 s, and each one has to wipe the prompt line to
    write above it. Under the tty's default canonical mode the typed characters
    live in the kernel's line buffer, unreadable from here, so a redraw can only
    put back the bare prompt -- the input is still buffered but no longer on
    screen, which makes the prompt unusable mid-run. So drop canonical mode, hold
    the edit buffer here, and re-echo it after every async write.
    """

    def __init__(self) -> None:
        # One lock for the buffer and for stdout: the pump thread echoes
        # keystrokes while the event loop writes telemetry.
        self._lock = threading.RLock()
        self._buf = ""
        self._saved: list | None = None
        self._esc = False

    def enable(self) -> bool:
        """Take the tty out of canonical mode. False if stdin is not a terminal."""
        if termios is None or not sys.stdin.isatty():
            return False
        self._saved = termios.tcgetattr(sys.stdin.fileno())
        # cbreak, not raw: ISIG stays on so Ctrl-C still raises KeyboardInterrupt,
        # and ONLCR stays on so plain print() still ends lines correctly.
        tty.setcbreak(sys.stdin.fileno())
        return True

    def restore(self) -> None:
        if self._saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
            self._saved = None

    @property
    def active(self) -> bool:
        return self._saved is not None

    def redraw(self) -> None:
        with self._lock:
            sys.stdout.write(f"\r\x1b[2K{PROMPT}{self._buf}")
            sys.stdout.flush()

    def emit(self, text: str) -> None:
        """Write async output above the prompt, then restore the edit line."""
        with self._lock:
            sys.stdout.write(f"\r\x1b[2K{text}\n{PROMPT}{self._buf}")
            sys.stdout.flush()

    def feed(self, ch: str) -> str | object | None:
        """Handle one keystroke: the finished line, ``_EOF``, or None if still editing."""
        with self._lock:
            if self._esc:  # tail of an escape sequence (arrow keys, Home, ...)
                if ch.isalpha() or ch == "~":
                    self._esc = False
                return None
            if ch == "\x1b":
                self._esc = True
                return None
            if ch in ("\r", "\n"):
                line, self._buf = self._buf, ""
                sys.stdout.write("\n")
                sys.stdout.flush()
                return line
            if ch == "\x04":  # Ctrl-D: EOF only on an empty line, as a shell does
                return _EOF if not self._buf else None
            if ch in ("\x7f", "\b"):
                self._buf = self._buf[:-1]
            elif ch == "\x15":  # Ctrl-U
                self._buf = ""
            elif ch >= " ":
                self._buf += ch
            else:
                return None
            self.redraw()
            return None


_editor = _LineEditor()


def _echo_prompt() -> None:
    _editor.redraw()


def _emit(text: str) -> None:
    """Print output that arrives while the user may be mid-line: wipe the line,
    write the text, redraw prompt + edit buffer. For async output only -- command
    replies print normally, since the read loop redraws the prompt after each."""
    _editor.emit(text)


class _PromptHandler(logging.StreamHandler):
    """Log handler that keeps the prompt line intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _emit(self.format(record))
        except Exception:
            self.handleError(record)


def _stdin_queue(loop: asyncio.AbstractEventLoop) -> asyncio.Queue[str | None]:
    """Pump stdin into a queue from a daemon thread; ``None`` marks EOF.

    A thread, not a reader transport: stdin may be a terminal on any platform,
    and the policy holds the event loop only between ticks.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def pump_keys() -> None:
        while True:
            ch = sys.stdin.read(1)
            item = _editor.feed(ch) if ch else _EOF
            if item is _EOF:
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return
            if item is not None:
                loop.call_soon_threadsafe(queue.put_nowait, item)

    def pump_lines() -> None:
        for line in sys.stdin:
            loop.call_soon_threadsafe(queue.put_nowait, line)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=pump_keys if _editor.active else pump_lines, daemon=True).start()
    return queue


class DebugConsole:
    """Terminal front end for one :class:`PolicyRunner`."""

    def __init__(self, op: Operator, task: str) -> None:
        self._op = op
        self._task = task
        self._run: asyncio.Task | None = None
        self._last_telemetry = 0.0
        self._runner: PolicyRunner | None = None  # set by attach(), after the checkpoint loads

    def attach(self, runner: PolicyRunner) -> None:
        self._runner = runner

    @property
    def running(self) -> bool:
        return self._run is not None and not self._run.done()

    def on_tick(self, info: dict) -> None:
        """Per-tick line: is it inferring, and is the command going anywhere?

        Replan ticks (the model actually running, ~every 50 ticks on a 50-step
        chunk) always print; the cheap mid-chunk pops in between are throttled.
        """
        now = time.monotonic()
        if not info["replan"] and now - self._last_telemetry < TELEMETRY_PERIOD_S:
            return
        self._last_telemetry = now
        cmd, state = info["cmd"], info["state"]
        pose = " ".join(f"{cmd[key]:6.1f}" for key in ARM_POS_KEYS)
        gap = max((abs(cmd[key] - state[key]) for key in ARM_POS_KEYS if key in state), default=float("nan"))
        _emit(f"tick {info['tick']:4d} {'replan' if info['replan'] else 'chunk '} | "
              f"infer {info['infer_ms']:5.0f}ms | cmd [{pose} ] | "
              f"max|cmd-obs| {gap:5.1f} | active={self._op.active_operator()}")

    async def _pick(self) -> None:
        try:
            result = await self._runner.pick(self._task)
            _emit(f"done: {json.dumps(result)}")
        except Exception:
            logger.exception("[debug] pick failed")

    def start(self) -> None:
        if self.running:
            print("already running -- `stop` first")
            return
        if not self._runner.ready:
            print("no robot state/frames yet -- is `uv run robot` up in this room?")
            return
        print(f"start: {self._task!r}")
        self._last_telemetry = 0.0
        self._run = asyncio.create_task(self._pick())

    def stop(self) -> None:
        if not self.running:
            print("not running")
            return
        self._runner.request_stop()

    def set_task(self, text: str) -> None:
        self._task = text
        self._runner.set_task(text)
        print(f"prompt: {text!r}{' (live: refolds, then plans on it)' if self.running else ''}")

    def status(self) -> None:
        print(f"{'running' if self.running else 'idle'} | prompt={self._task!r} | "
              f"robot={'ready' if self._runner.ready else 'waiting for state/frames'} | "
              f"me={self._op.local_identity()} | active={self._op.active_operator()}")

    async def handle(self, line: str) -> bool:
        """Run one command line. Returns False to quit."""
        cmd, _, rest = line.strip().partition(" ")
        cmd, rest = cmd.lower(), rest.strip()

        if not cmd:
            if self.running:
                self.stop()
            else:
                self.start()
        elif cmd in ("start", "s", "run", "go"):
            if rest:
                self.set_task(rest)
            self.start()
        elif cmd in ("stop", "x"):
            self.stop()
        elif cmd in ("prompt", "task", "p"):
            if rest:
                self.set_task(rest)
            else:
                print(f"prompt: {self._task!r}")
        elif cmd in ("status", "?"):
            self.status()
        elif cmd in ("quit", "exit", "q"):
            return False
        else:
            print(f"unknown command {cmd!r}\n{HELP}")
        return True

    async def shutdown(self) -> None:
        """Stop a run in flight so it releases active control before we disconnect."""
        if self.running:
            print("stopping ...")
            self._runner.request_stop()
            await asyncio.wait({self._run}, timeout=5.0)


async def main() -> None:
    # Before the parser: add_policy_args reads the POLICY_* defaults (and the
    # required --checkpoint) from os.environ as it declares the arguments.
    load_env(pathlib.Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser(description="Interactive terminal driver for the pi0 policy.")
    add_policy_args(parser)
    args = parser.parse_args()

    # force: importing lerobot installs a root handler, which would make this a
    # no-op — leaving _PromptHandler uninstalled, so logs chop up the prompt line.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[_PromptHandler()],
        force=True,
    )

    url = required_env("LIVEKIT_URL")
    room = env_str("LIVEKIT_ROOM", "candy-shop")
    token = mint_token(IDENTITY, room)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)
    console = DebugConsole(op, args.task)
    console.attach(build_runner(op, args, on_tick=console.on_tick))

    op.on_active_operator_changed(
        lambda identity: _emit(f"[robot] active operator now: {identity}")
    )

    logger.info("[debug] connecting to %s as '%s' in room '%s' ...", url, IDENTITY, room)
    await op.connect(url, token)
    logger.info("[debug] connected as '%s'", op.local_identity())

    print(f"\n{HELP}\n")
    console.status()

    _editor.enable()
    queue = _stdin_queue(asyncio.get_running_loop())
    try:
        while True:
            _echo_prompt()
            line = await queue.get()
            if line is None:  # Ctrl-D
                print()
                break
            if not await console.handle(line):
                break
    except (KeyboardInterrupt, asyncio.CancelledError):
        print()
    finally:
        _editor.restore()  # before the shutdown logs, so the shell is sane again
        await console.shutdown()
        logger.info("[debug] disconnecting ...")
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
