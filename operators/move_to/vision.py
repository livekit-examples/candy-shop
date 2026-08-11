"""Perception + geometry for the positioner: ArUco detection and the safe zone.

``SafeZone`` ``pos`` is 0..100 (0 = first line, 100 = second) along ``config.AXIS``,
stored normalized and capped to ``[0, 100]``.
"""
from __future__ import annotations

import logging
import math
import pathlib
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from operators.move_to import config

logger = logging.getLogger(__name__)

DEFAULT_SAFE_ZONE = pathlib.Path(__file__).resolve().parent / "safe_zone.yaml"


@dataclass
class MarkerDetection:
    cx: float            # pixels
    cy: float            # pixels
    marker_id: int
    corners: np.ndarray  # (4, 2) pixel corners


def _gamma_lut(gamma: float) -> np.ndarray:
    """256-entry map for `gray = 255 * (gray/255) ** gamma`."""
    return np.clip(
        ((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255
    ).astype(np.uint8)


class ArucoDetector:
    """ArUco detection tuned for uneven lighting (see `config`).

    Three departures from stock OpenCV: a gamma LUT lifts the shadows, CLAHE
    equalizes local contrast, and the adaptive-threshold window sweep is wider and
    finer than the default, which assumes an evenly lit marker.
    """

    def __init__(self, dictionary: str, marker_id: int | None):
        aruco = cv2.aruco
        dict_id = getattr(aruco, dictionary, None)
        if dict_id is None:
            raise ValueError(f"unknown ArUco dictionary {dictionary!r}")

        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = config.ADAPTIVE_THRESH_WIN_MIN
        params.adaptiveThreshWinSizeMax = config.ADAPTIVE_THRESH_WIN_MAX
        params.adaptiveThreshWinSizeStep = config.ADAPTIVE_THRESH_WIN_STEP
        params.minMarkerPerimeterRate = config.MIN_MARKER_PERIMETER_RATE
        if config.CORNER_REFINE_SUBPIX:
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        self._detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(dict_id), params)
        self.marker_id = marker_id
        self._gamma = _gamma_lut(config.GAMMA) if config.GAMMA != 1.0 else None
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT,
                tileGridSize=(config.CLAHE_TILE_GRID, config.CLAHE_TILE_GRID),
            )
            if config.CLAHE_ENABLED
            else None
        )

    def enhance(self, frame_rgb: np.ndarray) -> np.ndarray:
        """The grayscale the detector actually sees. Public so the debug tool can
        show it — tuning gamma/CLAHE blind is guesswork."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self._gamma is not None:
            gray = cv2.LUT(gray, self._gamma)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        return gray

    def detect(self, frame_rgb: np.ndarray) -> Optional[MarkerDetection]:
        gray = self.enhance(frame_rgb)
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
class Estimate:
    """One tick of the fused marker position, in normalized axis coordinates."""

    coord: Optional[float]  # None until the first detection
    measured: bool          # a detection was accepted this tick
    usable: bool            # fresh enough to servo on (measured, or a valid coast)
    age_s: float            # since the last accepted detection; inf if never
    calibrated: bool        # the slider gain is fitted, so coasting is allowed


class AxisEstimator:
    """Fuses ArUco measurements with slider dead reckoning along one image axis.

    The marker is rigid on the carriage, so its normalized motion along the axis
    is ``d(coord)/dt = gain * slider_vel``. ``gain`` (axis fraction per tick/s) is
    fitted online by least squares through the origin, comparing the velocity
    integral between two accepted detections against the distance they moved. The
    fit runs measurement-to-measurement, never on the filter's own prediction, so
    it cannot drift into agreeing with itself; exponential forgetting lets it
    re-fit if the camera or rail is moved.

    With a fitted gain the estimate coasts through dropouts (glare, motion blur, a
    dark stretch of rail). Until then ``usable`` is False whenever the marker is
    missing, which keeps the servo stopped rather than driving on a guess.
    """

    def __init__(self) -> None:
        self._meas_var = config.EST_MEAS_STD ** 2
        self._drift_var_per_s = config.EST_DRIFT_STD_PER_S ** 2
        self._gain = 0.0
        self._sum_vv = 0.0
        self._sum_vd = 0.0
        self._gain_samples = 0
        self.reset_position()

    def reset_position(self) -> None:
        """Drop the position estimate, keeping the fitted gain. Call before a move:
        between RPCs the carriage may have been moved by something else."""
        self._coord: Optional[float] = None
        self._var = 0.0
        self._age_s = float("inf")
        self._rejects = 0
        self._last_meas: Optional[float] = None
        self._vel_integral = 0.0

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def calibrated(self) -> bool:
        return self._gain_samples >= config.GAIN_MIN_SAMPLES and self._gain != 0.0

    def update(
        self, measured_coord: Optional[float], slider_vel: float, dt: float
    ) -> Estimate:
        dt = max(0.0, dt)
        if self._coord is not None:
            self._coord += self._gain * slider_vel * dt
            self._var += self._drift_var_per_s * dt
            self._age_s += dt
        self._vel_integral += slider_vel * dt

        accepted = measured_coord is not None and self._accept(measured_coord)
        coasting = self.calibrated and self._age_s <= config.MAX_COAST_S
        return Estimate(
            coord=self._coord,
            measured=accepted,
            usable=self._coord is not None and (accepted or coasting),
            age_s=self._age_s,
            calibrated=self.calibrated,
        )

    def _accept(self, z: float) -> bool:
        """Gate a detection, then fold it in. Returns whether it was accepted."""
        hard_reset = False
        if self._coord is not None:
            gate = config.EST_GATE_SIGMA * math.sqrt(self._var + self._meas_var)
            if abs(z - self._coord) > gate:
                self._rejects += 1
                if self._rejects < config.EST_REACQUIRE_TICKS:
                    return False
                # Persistently disagreeing: the estimate, not the camera, is wrong.
                logger.warning(
                    "[move-to] estimator re-acquiring after %d gated detections",
                    self._rejects,
                )
                hard_reset = True

        if hard_reset or self._coord is None:
            self._coord, self._var = z, self._meas_var
        else:
            self._fit_gain(z)
            k = self._var / (self._var + self._meas_var)
            self._coord += k * (z - self._coord)
            self._var *= 1.0 - k

        self._age_s = 0.0
        self._rejects = 0
        self._last_meas = z
        self._vel_integral = 0.0
        return True

    def _fit_gain(self, z: float) -> None:
        v = self._vel_integral
        if self._last_meas is None or abs(v) < config.GAIN_MIN_TICKS:
            return
        travel = z - self._last_meas
        # Exactly zero means the same frame was detected twice (portal reuses
        # stale frames); two genuine detections are never bit-identical.
        if travel == 0.0:
            return
        self._sum_vv = self._sum_vv * config.GAIN_FORGET + v * v
        self._sum_vd = self._sum_vd * config.GAIN_FORGET + v * travel
        if self._sum_vv > 0.0:
            self._gain = self._sum_vd / self._sum_vv
            self._gain_samples += 1


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
    if sz.axis != config.AXIS:
        logger.warning(
            "[move-to] safe_zone axis %r != config.AXIS %r; re-run calibrate.py",
            sz.axis, config.AXIS,
        )
    logger.info("[move-to] safe zone %s: axis=%s pos0=%.3f pos100=%.3f",
                path, sz.axis, sz.pos0, sz.pos100)
    return sz
