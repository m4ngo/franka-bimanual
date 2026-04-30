"""Minimal Aravis camera wrapper for Ethernet GigE cameras."""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import gi
import numpy as np
from numpy.typing import NDArray

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis  # type: ignore[attr-defined]

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig  # noqa: F401 - re-exported for callers

from .config_arv import ArvCameraConfig

logger = logging.getLogger(__name__)

DOWNSCALE_FACTOR = 8

_BAYER_TO_RGB: dict[str, int] = {
    # "BayerBG8": cv2.COLOR_BAYER_BG2RGB,
    # "BayerGB8": cv2.COLOR_BAYER_GB2RGB,
    # "BayerGR8": cv2.COLOR_BAYER_GR2RGB,
    "BayerRG8": cv2.COLOR_BayerBG2BGR,
    # "BayerBG10": cv2.COLOR_BAYER_BG2RGB,
    # "BayerGB10": cv2.COLOR_BAYER_GB2RGB,
    # "BayerGR10": cv2.COLOR_BAYER_GR2RGB,
    # "BayerRG10": cv2.COLOR_BAYER_RG2RGB,
    # "BayerBG12": cv2.COLOR_BAYER_BG2RGB,
    # "BayerGB12": cv2.COLOR_BAYER_GB2RGB,
    # "BayerGR12": cv2.COLOR_BAYER_GR2RGB,
    # "BayerRG12": cv2.COLOR_BAYER_RG2RGB,
}


class ArvCamera(Camera):
    def __init__(self, config: ArvCameraConfig):
        super().__init__(config)
        self._config = config
        self._name = config.name
        self._ip = config.ip
        self._pixel_format = config.pixel_format
        self._width = int(config.width) if config.width is not None else 0
        self._height = int(config.height) if config.height is not None else 0

        self._camera: Aravis.Camera | None = None
        self._stream: Aravis.Stream | None = None
        self._payload: int = 0
        self._last_frame: np.ndarray | None = None

        self._sensor_width: int = self._width * DOWNSCALE_FACTOR
        self._sensor_height: int = self._height * DOWNSCALE_FACTOR

    @property
    def is_connected(self) -> bool:
        return self._camera is not None and self._stream is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("Don't use find cameras here")

    def connect(self, warmup: bool = True) -> None:
        self.disconnect()
        device = Aravis.open_device(self._ip)
        camera = Aravis.Camera.new_with_device(device)
        self._configure_camera(camera)

        stream = camera.create_stream(None, None)
        payload = int(camera.get_payload())
        for _ in range(20):
            stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        camera.start_acquisition()

        self._camera = camera
        self._stream = stream
        self._payload = payload
        self._pixel_format = self._safe_get_string(camera, "PixelFormat", "Mono8")
        self._last_frame = self.read()
        logger.info(
            "Connected camera %s @ %s (%s)",
            self._name,
            self._ip,
            self._pixel_format,
        )

    def read(self) -> NDArray[Any]:
        return self._fetch_frame(timeout_s=1.0, allow_stale=False)

    def async_read(self, timeout_ms: float = 500) -> NDArray[Any]:
        return self._fetch_frame(timeout_s=timeout_ms / 1000.0, allow_stale=True)

    def disconnect(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop_acquisition()
            except Exception:
                logger.debug("Failed stopping camera %s %s", self._name, self._ip, exc_info=True)
        self._camera = None
        self._stream = None
        self._payload = 0

    def blank_frame(self) -> np.ndarray:
        if self._last_frame is not None:
            return self._last_frame.copy()
        return np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def _fetch_frame(self, timeout_s: float, allow_stale: bool) -> np.ndarray:
        if self._stream is None:
            return self.blank_frame()

        deadline = time.monotonic() + max(timeout_s, 0.0)
        while True:
            buffer = self._stream.try_pop_buffer()
            if buffer is not None:
                try:
                    if buffer.get_status() == Aravis.BufferStatus.SUCCESS:
                        frame = self._buffer_to_rgb(buffer)
                        self._last_frame = frame
                        return frame.copy()
                finally:
                    self._stream.push_buffer(buffer)

            if allow_stale and self._last_frame is not None:
                return self._last_frame.copy()

            if time.monotonic() >= deadline:
                if self._last_frame is not None:
                    return self._last_frame.copy()
                raise TimeoutError(
                    f"Timed out reading camera {self._name} ({self._config.ip})."
                )

            time.sleep(0.005)

    def _buffer_to_rgb(self, buffer: Aravis.Buffer) -> np.ndarray:
        if self._camera is None:
            return self.blank_frame()
        # Use the camera's actual sensor dimensions when decoding the
        # incoming buffer so we keep the full frame (no camera-side crop).
        width = int(self._safe_get_int(self._camera, "Width", self._sensor_width))
        height = int(self._safe_get_int(self._camera, "Height", self._sensor_height))
        data = buffer.get_data()
        frame = self._decode_frame(data, width, height, self._pixel_format)

        # If the requested output size differs from the sensor size, downsample
        # in software (preserve the full frame then resize). This avoids camera
        # ROI/cropping behavior on the device.
        if (self._height and self._width) and (
            frame.shape[0] != self._height or frame.shape[1] != self._width
        ):
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame)

    def _configure_camera(self, camera: Aravis.Camera) -> None:
        try:
            camera.gv_set_packet_size(1400)
        except Exception:
            logger.debug("Could not set packet size on %s %s", self._name, self._ip, exc_info=True)

        camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
        self._safe_set_int(camera, "Width", self._sensor_width)
        self._safe_set_int(camera, "Height", self._sensor_height)
        if self._config.fps is not None:
            self._safe_set_float(camera, "AcquisitionFrameRate", float(self._config.fps))

    def _decode_frame(self, data: bytes, width: int, height: int, pixel_format: str) -> np.ndarray:
        if pixel_format == "Mono16":
            frame = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
            frame = (frame >> 8).astype(np.uint8)
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if pixel_format in _BAYER_TO_RGB:
            mono = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            return cv2.cvtColor(mono, _BAYER_TO_RGB[pixel_format])

        if pixel_format == "RGB8":
            frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            return frame

        if pixel_format == "Mono8":
            mono = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)

        # Fallback: best effort as mono8 to keep observation keys stable.
        try:
            mono = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
        except Exception as exc:
            raise ValueError(f"Unsupported pixel format {pixel_format} on {self._name}") from exc

    @staticmethod
    def _safe_set_int(camera: Aravis.Camera, key: str, value: int) -> None:
        try:
            camera.set_integer(key, int(value))
        except Exception:
            logger.debug("Could not set %s=%s", key, value, exc_info=True)

    @staticmethod
    def _safe_set_float(camera: Aravis.Camera, key: str, value: float) -> None:
        try:
            camera.set_float(key, float(value))
        except Exception:
            logger.debug("Could not set %s=%s", key, value, exc_info=True)

    @staticmethod
    def _safe_get_int(camera: Aravis.Camera, key: str, default: int) -> int:
        try:
            return int(camera.get_integer(key))
        except Exception:
            return default

    @staticmethod
    def _safe_get_string(camera: Aravis.Camera, key: str, default: str) -> str:
        try:
            return str(camera.get_string(key))
        except Exception:
            return default
