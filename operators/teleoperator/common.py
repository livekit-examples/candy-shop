"""Shared scaffolding for the recorder peer and the review UI: env loading,
token minting, an fps pacer.

A trimmed sibling of `vla_demo.common`, which also carries GPU-residency,
autocast, and settle-gate helpers — none of which mean anything here.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import pathlib
import sys
import time
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.room import RoomConfiguration


def load_env(start: Optional[pathlib.Path] = None) -> None:
    """Load `.env` walking up from `start` to the filesystem root (plus cwd).

    Nearest wins (`override=False`), so the repo-root `.env` is found from
    anywhere while a project-local one still overrides it. `.env.local` always
    overrides."""
    start = (start or pathlib.Path.cwd()).resolve()
    seen: set[pathlib.Path] = set()
    for d in (start, *start.parents, pathlib.Path.cwd().resolve()):
        if d in seen:
            continue
        seen.add(d)
        if (f := d / ".env").exists():
            load_dotenv(f, override=False)
        if (f := d / ".env.local").exists():
            load_dotenv(f, override=True)


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.environ.get(name) or default
    if required and not val:
        raise RuntimeError(f"{name} must be set (see .env.example)")
    return val  # type: ignore[return-value]


def mint_token(
    identity: str,
    room: str,
    *,
    name: str | None = None,
    attributes: dict[str, str] | None = None,
) -> str:
    """Mint a 6h join token. `attributes` are published as participant
    attributes every peer can read live."""
    grants = api.VideoGrants(
        room_join=True, room=room, can_publish=True, can_publish_data=True,
        can_subscribe=True, can_update_own_metadata=True,
    )
    room_cfg = RoomConfiguration(name=room, min_playout_delay=0, max_playout_delay=1)
    builder = api.AccessToken(
        env("LIVEKIT_API_KEY", required=True), env("LIVEKIT_API_SECRET", required=True)
    ).with_identity(identity).with_grants(grants).with_room_config(room_cfg)
    if name:
        builder = builder.with_name(name)
    if attributes:
        builder = builder.with_attributes(attributes)
    return builder.with_ttl(datetime.timedelta(hours=6)).to_jwt()


def user_config_dir() -> pathlib.Path:
    """Per-user, per-machine settings dir — key bindings, window geometry.

    Not the project directory: which key your foot pedal sends is a property of
    your desk, not of the repo, and it must not land in a commit."""
    if sys.platform == "darwin":
        base = pathlib.Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        base = pathlib.Path(os.environ.get("APPDATA") or pathlib.Path.home())
    else:
        base = pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or
                            pathlib.Path.home() / ".config")
    return base / "candy-shop-teleoperator"


def portal_config_path(package_dir: pathlib.Path) -> pathlib.Path:
    """Resolve the wire contract: `PORTAL_CONFIG` if set, else the repo-root
    `portal.yaml` shared by every candy-shop process. The env override is how you
    record a rig whose contract lives in another suite (see portal.yaml's header).

    Resolved from this module's location (repo root is ``parents[2]`` of
    ``operators/teleoperator/common.py``), so it's correct regardless of what the
    caller passes for ``package_dir``."""
    if override := os.environ.get("PORTAL_CONFIG"):
        return pathlib.Path(override).expanduser().resolve()
    return pathlib.Path(__file__).resolve().parents[2] / "portal.yaml"


def camera_names(cfg) -> tuple[str, ...]:
    """Every video track the contract declares, in declaration order — derived
    rather than hardcoded, so `PORTAL_CONFIG` picks up another rig's cameras."""
    return (*cfg.video_tracks, *(s.name for s in cfg.frame_video_tracks))


def contract_camera_names(path: pathlib.Path) -> tuple[str, ...]:
    """The same list, read straight from the contract's YAML.

    For the review UI, which needs the track names but must **not** import
    ``livekit.portal``: Portal's FFI and the ``livekit`` rtc SDK each statically
    link libwebrtc, so loading both into one process makes dyld register every
    ``RTC*`` ObjC class twice — dozens of "implemented in both ... may cause
    spurious casting failures and mysterious crashes" warnings on macOS. The UI
    genuinely needs the rtc SDK (it joins as a plain participant), so this is the
    side that gives up Portal. Byte-stream and WebRTC codecs share the ``videos``
    key, so one read covers both, in declaration order."""
    import yaml

    contract = yaml.safe_load(path.read_text()) or {}
    return tuple(
        str(entry["name"])
        for entry in (contract.get("videos") or [])
        if isinstance(entry, dict) and entry.get("name")
    )


async def pace(fps: int) -> AsyncIterator[int]:
    """Async tick generator at `fps`; yields a monotonically increasing index."""
    interval, next_tick, i = 1.0 / fps, time.perf_counter(), 0
    while True:
        yield i
        i += 1
        next_tick += interval
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()
