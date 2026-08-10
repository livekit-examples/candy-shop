"""Shared scaffolding for the robot runtime and operators: env loading, LiveKit token minting, async fps pacer."""
from __future__ import annotations

import asyncio
import datetime
import os
import pathlib
import time
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.room import RoomConfiguration


def load_env(search_from: Optional[pathlib.Path] = None) -> None:
    """Load `.env` walking up from `search_from` to the filesystem root (plus cwd).

    Nearest `.env` wins (loaded first, `override=False`); `.env.local` always
    overrides. Pass the calling script's directory (`Path(__file__).parent`).
    """
    start = (search_from or pathlib.Path.cwd()).resolve()
    seen: set[pathlib.Path] = set()
    for directory in (start, *start.parents, pathlib.Path.cwd().resolve()):
        if directory in seen:
            continue
        seen.add(directory)
        if (env_file := directory / ".env").exists():
            load_dotenv(env_file, override=False)
        if (env_local := directory / ".env.local").exists():
            load_dotenv(env_local, override=True)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set (see .env.example)")
    return value


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_camera_id(name: str, default: str) -> int | str:
    """Camera identifier: int index on macOS, `/dev/videoN` string on Linux."""
    raw = os.environ.get(name, default)
    return int(raw) if raw.isdigit() else raw


def mint_token(identity: str, room: str, ttl_hours: int = 6) -> str:
    """Mint a LiveKit JWT scoped to `room` for a Portal peer (low-latency playout)."""
    key = required_env("LIVEKIT_API_KEY")
    secret = required_env("LIVEKIT_API_SECRET")
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_update_own_metadata=True,
    )
    room_cfg = RoomConfiguration(name=room, min_playout_delay=0, max_playout_delay=1)
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_room_config(room_cfg)
        .with_ttl(datetime.timedelta(hours=ttl_hours))
        .to_jwt()
    )


async def pace(fps: int) -> AsyncIterator[int]:
    """Async tick generator at `fps`; yields the index, then awaits. Use as `async for tick in pace(30): ...`.

    Async (not `time.sleep`) so Portal callbacks scheduled via
    `call_soon_threadsafe` run between ticks.
    """
    interval = 1.0 / fps
    next_tick = time.perf_counter()
    i = 0
    while True:
        yield i
        i += 1
        next_tick += interval
        now = time.perf_counter()
        sleep_for = next_tick - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_tick = now
