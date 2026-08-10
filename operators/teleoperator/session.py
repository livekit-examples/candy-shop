"""Setup-screen inputs: the serial ports and corpora on this machine.

The dataset choice is made per session in the window (env vars only *preselect*),
because a pinned `DATASET_ROOT` silently appends two rigs to one corpus or trains
on a stale copy. `from_env` is the exception for unattended runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_REPO_ID = "binhpham/candy-shop"
DEFAULT_TASK = "pick up the candy"
PROJECT_DIR = Path(__file__).resolve().parents[2]  # repo root (operators/teleoperator/..)


def lerobot_home() -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME)


# --- what's already on disk --------------------------------------------------

@dataclass(frozen=True)
class Corpus:
    repo_id: str
    root: Path
    episodes: int
    fps: int

    @property
    def label(self) -> str:
        return f"{self.repo_id}  ({self.episodes} episode{'s' if self.episodes != 1 else ''})"


def search_roots(extra: Iterable[Path] = ()) -> list[Path]:
    """lerobot's home, this project's `data/`, then sibling suites' `data/` (a
    related suite may hold a corpus worth continuing here)."""
    roots = [lerobot_home(), PROJECT_DIR / "data", Path("data").resolve()]
    roots.extend(PROJECT_DIR.parent / s / "data" for s in ("vla-demo", "candy-shop-demo"))
    roots.extend(Path(p).expanduser() for p in extra)
    seen: set[Path] = set()
    return [r for r in roots if not (r in seen or seen.add(r))]


def discover(roots: Sequence[Path]) -> list[Corpus]:
    """Every LeRobotDataset under `roots`, most episodes first (the biggest is the
    best default). Repo ids are one or two path segments, so scan exactly that
    depth rather than walking whole trees (lerobot's home also holds `hub/`)."""
    found: dict[Path, Corpus] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for depth in ("*/meta/info.json", "*/*/meta/info.json"):
            for info in root.glob(depth):
                corpus = _read_corpus(info, root)
                if corpus is not None and corpus.root not in found:
                    found[corpus.root] = corpus
    return sorted(found.values(), key=lambda c: (-c.episodes, str(c.root)))


def _read_corpus(info: Path, search_root: Path) -> Optional[Corpus]:
    dataset_root = info.parent.parent
    if any(part in {"hub", "outputs"} for part in dataset_root.parts):
        return None  # HF download cache / training output, not a recording target
    try:
        meta = json.loads(info.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        rel = dataset_root.relative_to(search_root)
    except ValueError:
        rel = Path(dataset_root.name)
    return Corpus(
        repo_id=str(meta.get("repo_id") or rel).replace("\\", "/"),
        root=dataset_root,
        episodes=int(meta.get("total_episodes", 0) or 0),
        fps=int(meta.get("fps", 30) or 30),
    )


def last_task(corpus: Corpus) -> Optional[str]:
    """The newest episode's task — a better default than anything in a config
    file when continuing a corpus."""
    from operators.teleoperator.library import read_episodes

    try:
        episodes = read_episodes(corpus.root, corpus.fps)
    except Exception:
        return None
    return str(episodes[-1]["task"]) if episodes else None


# --- serial ports ------------------------------------------------------------

USB_SERIAL_HINTS = ("usbmodem", "usbserial", "ttyacm", "ttyusb")


def looks_like_leader(device: str) -> bool:
    """Whether a port plausibly *is* an SO-101, not merely a port. Gates the
    default (not the list) — a Mac lists debug consoles and Bluetooth channels as
    serial devices, and pre-selecting one is worse than pre-selecting nothing."""
    name = device.lower()
    return (not any(k in name for k in ("bluetooth", "debug-console"))
            and any(k in name for k in USB_SERIAL_HINTS))


def leader_ports() -> list[tuple[str, str]]:
    """`(device, description)` for every serial port, likely leaders first."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []

    def rank(device: str) -> int:
        if looks_like_leader(device):
            return 0
        return 2 if "bluetooth" in device.lower() else 1

    ports = [(p.device, (p.description or "").strip()) for p in list_ports.comports()]
    return sorted(ports, key=lambda p: (rank(p[0]), p[0]))


def default_leader_port(env_port: Optional[str] = None) -> str:
    """A port safe to preselect: the configured one, else the first that actually
    looks like an arm, else nothing — better to be asked than to drive a mic."""
    if env_port:
        return env_port
    for device, _ in leader_ports():
        if looks_like_leader(device):
            return device
    return ""


# --- for the GUI ------------------------------------------------------------

def setup_options(env_port: Optional[str], env_repo_id: Optional[str],
                  env_task: Optional[str],
                  roots: Optional[Sequence[Path]] = None) -> dict:
    """Everything the setup screen needs, as plain JSON. Assembled on the
    recorder because it owns the serial bus and disk — the window may be remote."""
    return {
        "ports": [{"device": device, "description": description,
                    "likely": looks_like_leader(device)}
                  for device, description in leader_ports()],
        "datasets": [{"repo_id": c.repo_id, "root": str(c.root), "episodes": c.episodes}
                     for c in discover(roots if roots is not None else search_roots())],
        "defaults": {
            "port": default_leader_port(env_port),
            "repo_id": env_repo_id or DEFAULT_REPO_ID,
            "task": env_task or DEFAULT_TASK,
            "lerobot_home": str(lerobot_home()),
            "local_root": str(Path("data").resolve()),
        },
    }


def valid_repo_id(repo_id: str) -> bool:
    """`org/name` — the LeRobotDataset convention `--dataset.repo_id` resolves against."""
    return repo_id.count("/") == 1 and all(part.strip() for part in repo_id.split("/"))


# --- the resolved answer -----------------------------------------------------

@dataclass
class Session:
    port: str
    repo_id: str
    root: Path
    task: str
    resumed: bool

    def describe(self) -> str:
        return (f"{'appending to' if self.resumed else 'creating'} "
                f"{self.repo_id} at {self.root}")


def from_env(
    env_port: Optional[str],
    env_repo_id: Optional[str],
    env_root: Optional[str],
    env_task: Optional[str],
) -> Optional[Session]:
    """A fully-specified session from the environment, or None to ask the window.

    "Fully specified" means a port plus somewhere to write — the unattended path.
    None means the recorder joins the room and waits for setup instead."""
    if not env_port or not (env_root or env_repo_id):
        return None
    repo_id = env_repo_id or DEFAULT_REPO_ID
    root = Path(env_root).expanduser() if env_root else lerobot_home() / repo_id
    return Session(
        port=env_port, repo_id=repo_id, root=root,
        task=env_task or DEFAULT_TASK,
        resumed=(root / "meta" / "info.json").exists(),
    )
