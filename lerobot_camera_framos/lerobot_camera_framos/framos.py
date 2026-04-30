"""FRAMOS D400e camera wrapper using the FRAMOS-built pyrealsense2.

The FRAMOS D415e is a GigE-Vision RealSense module: depth and RGB share one
IP and arrive on different GVSP channels. Using the FRAMOS librealsense2 SDK
gives us simultaneous color + aligned depth in a single pipeline, which the
plain Aravis path cannot provide.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera

from .config_framos import FramosCameraConfig

logger = logging.getLogger(__name__)


_FORMAT_LOOKUP: dict[str, int] = {
    "bgr8": rs.format.bgr8,
    "rgb8": rs.format.rgb8,
    "yuyv": rs.format.yuyv,
    "z16": rs.format.z16,
    "y8": rs.format.y8,
    "y16": rs.format.y16,
}

_STREAM_LOOKUP: dict[str, int] = {
    "color": rs.stream.color,
    "depth": rs.stream.depth,
    "infrared": rs.stream.infrared,
}


class FramosCamera(Camera):
    def __init__(self, config: FramosCameraConfig):
        super().__init__(config)
        self._config = config
        self._name = config.name
        self._ip = config.ip
        self._serial = config.serial_number

        self._pipeline: rs.pipeline | None = None
        self._profile: rs.pipeline_profile | None = None
        self._aligner: rs.align | None = None
        self._last_color: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None

        self._output_width = int(config.width) if config.width is not None else int(config.color_width)
        self._output_height = int(config.height) if config.height is not None else int(config.color_height)

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None and self._profile is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        ctx = rs.context()
        out: list[dict[str, Any]] = []
        for dev in ctx.devices:
            entry = {
                "name": dev.get_info(rs.camera_info.name),
                "serial_number": dev.get_info(rs.camera_info.serial_number),
            }
            try:
                entry["ip"] = dev.get_info(rs.camera_info.ip_address)
            except Exception:
                pass
            try:
                entry["firmware_version"] = dev.get_info(rs.camera_info.firmware_version)
            except Exception:
                pass
            out.append(entry)
        return out

    def connect(self, warmup: bool = True) -> None:
        self.disconnect()

        cfg = rs.config()
        # Prefer pinning the device by serial number when supplied; otherwise
        # let the pipeline grab whichever D400e the context returns.
        if self._serial:
            cfg.enable_device(self._serial)

        fps = int(self._config.fps) if self._config.fps is not None else 30

        if self._config.enable_depth:
            cfg.enable_stream(
                rs.stream.depth,
                int(self._config.depth_width),
                int(self._config.depth_height),
                _FORMAT_LOOKUP.get(self._config.depth_format.lower(), rs.format.z16),
                fps,
            )
        if self._config.enable_color:
            cfg.enable_stream(
                rs.stream.color,
                int(self._config.color_width),
                int(self._config.color_height),
                _FORMAT_LOOKUP.get(self._config.color_format.lower(), rs.format.bgr8),
                fps,
            )

        pipeline = rs.pipeline()
        profile = pipeline.start(cfg)
        self._pipeline = pipeline
        self._profile = profile

        if self._config.enable_color and self._config.enable_depth:
            align_target = _STREAM_LOOKUP.get(self._config.align_to.lower(), rs.stream.color)
            self._aligner = rs.align(align_target)
        else:
            self._aligner = None

        device = profile.get_device()
        self._apply_options(device)

        if warmup:
            # FRAMOS D415e color sensor needs a few frames before AE settles.
            for _ in range(10):
                try:
                    self._pipeline.wait_for_frames(timeout_ms=500)
                except Exception:
                    pass
            self._last_color = self.read()

        logger.info(
            "Connected FRAMOS camera %s (sn=%s ip=%s)",
            self._name,
            self._serial,
            self._ip,
        )

    def read(self) -> NDArray[Any]:
        return self._fetch_color(timeout_ms=1000.0, allow_stale=False)

    def async_read(self, timeout_ms: float = 500) -> NDArray[Any]:
        return self._fetch_color(timeout_ms=timeout_ms, allow_stale=True)

    def read_depth(self, timeout_ms: float = 1000.0) -> NDArray[Any]:
        if self._pipeline is None:
            return self._blank_depth()
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=int(timeout_ms))
        except Exception as exc:
            if self._last_depth is not None:
                return self._last_depth.copy()
            raise TimeoutError(f"FRAMOS {self._name}: depth read timed out") from exc

        if self._aligner is not None:
            frames = self._aligner.process(frames)

        depth = frames.get_depth_frame()
        if not depth:
            if self._last_depth is not None:
                return self._last_depth.copy()
            return self._blank_depth()
        arr = np.asanyarray(depth.get_data())
        self._last_depth = arr
        return arr.copy()

    def disconnect(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                logger.debug(
                    "Failed to stop FRAMOS pipeline for %s", self._name, exc_info=True
                )
        self._pipeline = None
        self._profile = None
        self._aligner = None

    def _fetch_color(self, timeout_ms: float, allow_stale: bool) -> np.ndarray:
        if self._pipeline is None:
            return self._blank_color()

        deadline = time.monotonic() + max(timeout_ms / 1000.0, 0.0)
        while True:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=int(timeout_ms))
            except Exception:
                if allow_stale and self._last_color is not None:
                    return self._last_color.copy()
                if time.monotonic() >= deadline:
                    if self._last_color is not None:
                        return self._last_color.copy()
                    raise TimeoutError(
                        f"FRAMOS {self._name}: color read timed out"
                    )
                continue

            if self._aligner is not None:
                frames = self._aligner.process(frames)

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color and not depth and not self._config.enable_color:
                # Color stream is disabled - return the last known color or blank.
                if self._last_color is not None:
                    return self._last_color.copy()
                return self._blank_color()

            if depth and self._config.enable_depth:
                self._last_depth = np.asanyarray(depth.get_data())

            if not color:
                if allow_stale and self._last_color is not None:
                    return self._last_color.copy()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"FRAMOS {self._name}: color frame missing in frameset"
                    )
                continue

            arr = np.asanyarray(color.get_data())
            if arr.ndim == 3 and (
                arr.shape[0] != self._output_height or arr.shape[1] != self._output_width
            ):
                arr = cv2.resize(
                    arr,
                    (self._output_width, self._output_height),
                    interpolation=cv2.INTER_AREA,
                )
            arr = np.ascontiguousarray(arr)
            self._last_color = arr
            return arr.copy()

    def _apply_options(self, device: rs.device) -> None:
        if not self._config.options:
            return
        for sensor in device.sensors:
            for raw_key, value in self._config.options.items():
                opt_name = raw_key.lower()
                opt = getattr(rs.option, opt_name, None)
                if opt is None or not sensor.supports(opt):
                    continue
                try:
                    sensor.set_option(opt, float(value))
                    logger.info(
                        "FRAMOS %s: set %s=%s on %s",
                        self._name,
                        opt_name,
                        value,
                        sensor.get_info(rs.camera_info.name),
                    )
                except Exception:
                    logger.debug(
                        "FRAMOS %s: failed setting %s=%s",
                        self._name,
                        opt_name,
                        value,
                        exc_info=True,
                    )

    def _blank_color(self) -> np.ndarray:
        if self._last_color is not None:
            return self._last_color.copy()
        return np.zeros((self._output_height, self._output_width, 3), dtype=np.uint8)

    def _blank_depth(self) -> np.ndarray:
        if self._last_depth is not None:
            return self._last_depth.copy()
        return np.zeros(
            (int(self._config.depth_height), int(self._config.depth_width)),
            dtype=np.uint16,
        )

    def blank_frame(self) -> np.ndarray:
        return self._blank_color()
