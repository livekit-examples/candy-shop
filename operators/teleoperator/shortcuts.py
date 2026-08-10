"""Rebindable key bindings, shared by the window and the terminal.

Recording is a foot-pedal job — your hands are on the leader arm — and a pedal
sends whatever key its vendor chose (`'`, an F-key, a media key), which you
cannot rename. So bindings are data, not literals.

Three layers, per action, most specific first: `TELEOPERATOR_KEYS_<ACTION>` in the
env, then the JSON file the window's binding editor writes, then the defaults
below. Per-action, so setting one env var doesn't silently clear the others.

Keys are stored by **ImGui enum member name** (`apostrophe`, `f13`, `space`) —
stable across versions and platforms, unlike `imgui.get_key_name()`, which the
upstream docs mark debug-only and explicitly not for persistence.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, Mapping

from operators.teleoperator.common import user_config_dir

logger = logging.getLogger(__name__)

ACTIONS = ("record", "discard", "claim", "release")

DEFAULTS: dict[str, tuple[str, ...]] = {
    # `apostrophe` is here by default because it is what a common USB foot pedal
    # sends out of the box.
    "record": ("r", "space", "apostrophe"),
    "discard": ("left_bracket", "backspace"),
    "claim": ("c",),
    "release": (),
}

# ImGui key name -> the character that key produces on a terminal, so the
# recorder's stdin hotkeys honour the same bindings. Keys with no single-byte
# representation (F-keys, arrows — they arrive as escape sequences) are simply
# absent: they work in the window and are ignored in the terminal.
TERMINAL_CHARS: dict[str, str] = {
    **{c: c for c in "abcdefghijklmnopqrstuvwxyz"},
    **{f"_{d}": d for d in "0123456789"},
    "space": " ", "apostrophe": "'", "comma": ",", "period": ".",
    "semicolon": ";", "slash": "/", "backslash": "\\", "minus": "-",
    "equal": "=", "grave_accent": "`", "left_bracket": "[", "right_bracket": "]",
    "tab": "\t", "enter": "\r", "backspace": "\x7f",
}


def config_path() -> Path:
    return user_config_dir() / "shortcuts.json"


def load() -> dict[str, list[str]]:
    """The effective bindings. Never raises — a corrupt file logs and is ignored,
    because losing your bindings must not stop you recording."""
    bindings = {action: list(keys) for action, keys in DEFAULTS.items()}

    path = config_path()
    if path.exists():
        try:
            stored = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("ignoring unreadable %s; using default key bindings", path)
            stored = {}
        for action in ACTIONS:
            if isinstance(stored.get(action), list):
                bindings[action] = [str(k) for k in stored[action]]

    for action in ACTIONS:
        raw = os.environ.get(f"TELEOPERATOR_KEYS_{action.upper()}")
        if raw is not None:
            bindings[action] = _parse(raw)
    return bindings


def _parse(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.replace(" ", ",").split(",") if part.strip()]


def save(bindings: Mapping[str, Iterable[str]]) -> None:
    """Persist to the user config dir. Best-effort: a read-only home shouldn't
    take the window down."""
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {action: list(bindings.get(action, ())) for action in ACTIONS}, indent=2,
        ) + "\n")
    except OSError:
        logger.warning("could not save key bindings to %s", path)


def terminal_chars(bindings: Mapping[str, Iterable[str]], action: str) -> set[str]:
    """The characters that trigger `action` when typed at a terminal."""
    return {TERMINAL_CHARS[key] for key in bindings.get(action, ()) if key in TERMINAL_CHARS}


def describe(bindings: Mapping[str, Iterable[str]], action: str) -> str:
    """Human label for a binding list, e.g. `R / space / '`."""
    pretty = {"apostrophe": "'", "left_bracket": "[", "right_bracket": "]",
              "backspace": "bksp", "grave_accent": "`", "space": "space"}
    keys = list(bindings.get(action, ()))
    if not keys:
        return "unbound"
    return " / ".join(pretty.get(k, k.upper() if len(k) == 1 else k) for k in keys)
