"""Perception + geometry for the positioner: ArUco detection and the safe zone.

``SafeZone`` ``pos`` is 0..100 (0 = first line, 100 = second) along ``config.AXIS``,
stored normalized and capped to ``[0, 100]``.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAFE_ZONE = pathlib.Path(__file__).resolve().parent / "safe_zone.yaml"


@dataclass
class MarkerDetection:
    cx: float            # pixels
    cy: float            # pixels
    marker_id: int
    corners: np.ndarray  # (4, 2) pixel corners


class ArucoDetector:
    def __init__(self, dictionary: str, marker_id: int | None):
        aruco = cv2.aruco
        dict_id = getattr(aruco, dictionary, None)
        if dict_id is None:
            raise ValueError(f"unknown ArUco dictionary {dictionary!r}")
        self._detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(dict_id), aruco.DetectorParameters()
        )
        self.marker_id = marker_id

    def detect(self, frame_rgb: np.ndarray) -> Optional[MarkerDetection]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None

        ids = ids.flatten()
        idx = 0
        if self.marker_id is not None:
            matches = np.where(ids == self.marker_id)[0]
            if len(matches) == 0:
                return None
            idx = int(matches[0])

        quad = corners[idx].reshape(4, 2)
        center = quad.mean(axis=0)
        return MarkerDetection(
            cx=float(center[0]), cy=float(center[1]),
            marker_id=int(ids[idx]), corners=quad,
        )


@dataclass
class SafeZone:
    axis: str        # "vertical" | "horizontal"
    pos0: float      # normalized coord along axis at slider pos 0
    pos100: float    # normalized coord along axis at slider pos 100

    @staticmethod
    def clamp_pos(pos: float) -> float:
        return max(0.0, min(100.0, pos))

    def target_coord(self, pos: float) -> float:
        pos = self.clamp_pos(pos)
        return self.pos0 + (pos / 100.0) * (self.pos100 - self.pos0)

    def pos_of(self, coord: float) -> float:
        span = self.pos100 - self.pos0
        if abs(span) < 1e-9:
            return 0.0
        return self.clamp_pos(100.0 * (coord - self.pos0) / span)

    def coord_of_point(self, x: float, y: float, h: int, w: int) -> float:
        """Normalized coordinate along the axis for an image point."""
        return (y / h) if self.axis == "vertical" else (x / w)

    def marker_coord(self, marker: MarkerDetection, h: int, w: int) -> float:
        return self.coord_of_point(marker.cx, marker.cy, h, w)

    def axis_length(self, h: int, w: int) -> int:
        return h if self.axis == "vertical" else w

    @classmethod
    def from_yaml(cls, path: pathlib.Path) -> "SafeZone":
        import yaml

        if not path.exists():
            raise FileNotFoundError(str(path))
        data = yaml.safe_load(path.read_text())
        return cls(
            axis=str(data.get("axis", "vertical")),
            pos0=float(data["pos0"]),
            pos100=float(data["pos100"]),
        )


def load_safe_zone(path: Optional[pathlib.Path] = None) -> SafeZone:
    path = path or DEFAULT_SAFE_ZONE
    try:
        sz = SafeZone.from_yaml(path)
    except FileNotFoundError:
        raise SystemExit(
            f"no safe zone at {path}; run calibrate.py to click the two lines first."
        )
    from operators.move_to import config

    if sz.axis != config.AXIS:
        logger.warning(
            "[move-to] safe_zone axis %r != config.AXIS %r; re-run calibrate.py",
            sz.axis, config.AXIS,
        )
    logger.info("[move-to] safe zone %s: axis=%s pos0=%.3f pos100=%.3f",
                path, sz.axis, sz.pos0, sz.pos100)
    return sz
