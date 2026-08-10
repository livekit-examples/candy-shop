"""Interactive 2-click safe-zone calibration: click the pos 0 line then the pos 100
line; bounds are saved to ``safe_zone.yaml`` as normalized image coordinates.

Usage::

    uv run move-to-calibrate   # robot must be in the room
"""
from __future__ import annotations

import asyncio
import pathlib
import time
from typing import Optional

import cv2
import numpy as np

from livekit.portal import (
    Observation,
    Operator,
    OperatorConfig,
    frame_bytes_to_numpy_rgb,
)

from utilities.common import env_str, load_env, mint_token, required_env

from operators.move_to import config

IDENTITY = "move-to-calibration"
CAMERA = "overhead_camera"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"
SAFE_ZONE_PATH = pathlib.Path(__file__).resolve().parent / "safe_zone.yaml"
SNAPSHOT_PATH = pathlib.Path(__file__).resolve().parent / "safe_zone_frame.png"

WINDOW = "move_to safe zone"
_LABELS = ("pos 0 line", "pos 100 line")
_COLORS = ((0, 128, 255), (0, 220, 0))  # BGR


async def _capture_frame(timeout_s: float = 20.0) -> tuple[np.ndarray, int, int]:
    load_env(pathlib.Path(__file__).resolve().parent)
    url = required_env("LIVEKIT_URL")
    room = env_str("LIVEKIT_ROOM", "candy-shop")
    token = mint_token(IDENTITY, room)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)

    holder: dict[str, object] = {"frame": None, "w": 0, "h": 0}

    def on_observation(obs: Observation) -> None:
        if holder["frame"] is not None:
            return
        f = obs.frames.get(CAMERA)
        if f is not None:
            holder["frame"] = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)
            holder["w"], holder["h"] = f.width, f.height

    op.on_observation(on_observation)

    print(f"[calib] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    print(f"[calib] connected; waiting for first '{CAMERA}' frame ...")

    t0 = time.monotonic()
    try:
        while holder["frame"] is None:
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(
                    f"no '{CAMERA}' frame within {timeout_s:.0f}s; is the robot running?"
                )
            await asyncio.sleep(0.05)
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()

    frame = holder["frame"]
    assert isinstance(frame, np.ndarray)
    print(f"[calib] got frame {holder['w']}x{holder['h']}; disconnected.")
    return frame, int(holder["w"]), int(holder["h"])


def _line(img: np.ndarray, coord_px: int, color) -> None:
    h, w = img.shape[:2]
    if config.AXIS == "vertical":
        cv2.line(img, (0, coord_px), (w, coord_px), color, 2, cv2.LINE_AA)
    else:
        cv2.line(img, (coord_px, 0), (coord_px, h), color, 2, cv2.LINE_AA)


def _draw(canvas_bgr: np.ndarray, coords: list[int]) -> np.ndarray:
    img = canvas_bgr.copy()
    for i, c in enumerate(coords):
        _line(img, c, _COLORS[i])
        cv2.putText(img, _LABELS[i], (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, _COLORS[i], 2, cv2.LINE_AA)

    msg = f"Click the {_LABELS[len(coords)]}" if len(coords) < 2 else "Enter=save  r=redo  q=cancel"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, f"[{config.AXIS}]  {msg}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _pick_lines(rgb: np.ndarray) -> Optional[list[int]]:
    """Return the two along-axis pixel coordinates, or None if cancelled."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    coords: list[int] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(coords) < 2:
            coords.append(y if config.AXIS == "vertical" else x)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"[calib] axis={config.AXIS}: click the pos 0 line, then the pos 100 line.")

    while True:
        cv2.imshow(WINDOW, _draw(bgr, coords))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None
        if key == ord("r"):
            coords.clear()
        elif key in (13, 10) and len(coords) == 2:
            break

    cv2.destroyAllWindows()
    return coords


def _write_yaml(coords: list[int], w: int, h: int) -> None:
    axis_len = h if config.AXIS == "vertical" else w
    pos0, pos100 = (c / axis_len for c in coords)
    text = f"""# move_to safe zone (2-click calibration by calibrate.py).
# The two lines the slider travels between, as normalized image coordinate
# along axis '{config.AXIS}'. The operator caps requests to [0, 100].
axis: {config.AXIS}
frame:
  width: {w}
  height: {h}
pos0: {pos0:.6f}     # px {coords[0]}
pos100: {pos100:.6f}   # px {coords[1]}
"""
    SAFE_ZONE_PATH.write_text(text)
    print(f"[calib] wrote {SAFE_ZONE_PATH}")


def main() -> None:
    frame, w, h = asyncio.run(_capture_frame())

    cv2.imwrite(str(SNAPSHOT_PATH), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"[calib] saved snapshot {SNAPSHOT_PATH}")

    coords = _pick_lines(frame)
    if coords is None:
        print("[calib] cancelled; nothing written.")
        return

    _write_yaml(coords, w, h)
    print("[calib] done.")


if __name__ == "__main__":
    main()
