"""Minimal GenTL camera wrapper for GigE workspace/gripper cameras."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from harvesters.core import Harvester, ImageAcquirer

logger = logging.getLogger(__name__)

_BAYER_TO_RGB: dict[str, int] = {
    "BayerBG8": cv2.COLOR_BAYER_BG2RGB,
    "BayerGB8": cv2.COLOR_BAYER_GB2RGB,
    "BayerGR8": cv2.COLOR_BAYER_GR2RGB,
    "BayerRG8": cv2.COLOR_BAYER_RG2RGB,
    "BayerBG10": cv2.COLOR_BAYER_BG2RGB,
    "BayerGB10": cv2.COLOR_BAYER_GB2RGB,
    "BayerGR10": cv2.COLOR_BAYER_GR2RGB,
    "BayerRG10": cv2.COLOR_BAYER_RG2RGB,
    "BayerBG12": cv2.COLOR_BAYER_BG2RGB,
    "BayerGB12": cv2.COLOR_BAYER_GB2RGB,
    "BayerGR12": cv2.COLOR_BAYER_GR2RGB,
    "BayerRG12": cv2.COLOR_BAYER_RG2RGB,
}


@dataclass(frozen=True)
class GigECameraConfig:
    name: str
    ip: str
    serial_number: str
    width: int
    height: int
    fps: int | None
    cti_path: str


class GigECamera:
    """Best-effort camera reader that never interferes with arm control."""

    def __init__(self, config: GigECameraConfig):
        self.config = config
        self._harvester: Harvester | None = None
        self._acquirer: ImageAcquirer | None = None
        self._last_frame: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self._acquirer is not None

    def connect(self) -> None:
        self.disconnect()
        harvester = Harvester()
        harvester.add_cti_file(self.config.cti_path)
        harvester.update()

        acquirer = harvester.create_image_acquirer(serial_number=self.config.serial_number)
        remote_device = acquirer.remote_device
        node_map = remote_device.node_map

        self._configure_node(node_map, "Width", self.config.width)
        self._configure_node(node_map, "Height", self.config.height)
        if self.config.fps is not None:
            self._configure_node(node_map, "AcquisitionFrameRateEnable", True)
            self._configure_node(node_map, "AcquisitionFrameRate", float(self.config.fps))

        acquirer.start(run_as_thread=True)

        self._harvester = harvester
        self._acquirer = acquirer
        self._last_frame = self.read()
        logger.info(
            "Connected camera %s (%s @ %s)",
            self.config.name,
            self.config.serial_number,
            self.config.ip,
        )

    def read(self) -> np.ndarray:
        return self._fetch_frame(timeout_s=1.0, allow_stale=False)

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        return self._fetch_frame(timeout_s=max_age_ms / 1000.0, allow_stale=True)

    def disconnect(self) -> None:
        if self._acquirer is not None:
            try:
                self._acquirer.stop()
            except Exception:
                logger.debug("Failed stopping camera %s cleanly", self.config.name, exc_info=True)
            try:
                self._acquirer.destroy()
            except Exception:
                logger.debug("Failed destroying camera %s cleanly", self.config.name, exc_info=True)
        if self._harvester is not None:
            try:
                self._harvester.reset()
            except Exception:
                logger.debug("Failed resetting harvester for %s", self.config.name, exc_info=True)

        self._acquirer = None
        self._harvester = None

    def blank_frame(self) -> np.ndarray:
        if self._last_frame is not None:
            return self._last_frame.copy()
        return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

    def _fetch_frame(self, timeout_s: float, allow_stale: bool) -> np.ndarray:
        if self._acquirer is None:
            return self.blank_frame()

        deadline = time.monotonic() + max(timeout_s, 0.0)
        while True:
            buffer = self._acquirer.try_fetch(timeout=0.0)
            if buffer is not None:
                with buffer:
                    frame = self._buffer_to_rgb(buffer)
                self._last_frame = frame
                return frame.copy()

            if allow_stale and self._last_frame is not None:
                return self._last_frame.copy()

            if time.monotonic() >= deadline:
                if self._last_frame is not None:
                    return self._last_frame.copy()
                raise TimeoutError(
                    f"Timed out reading camera {self.config.name} ({self.config.serial_number})."
                )

            time.sleep(0.005)

    def _buffer_to_rgb(self, buffer: Any) -> np.ndarray:
        component = buffer.payload.components[0]
        height = int(component.height)
        width = int(component.width)
        channels = int(getattr(component, "num_components_per_pixel", 1))
        data_format = str(getattr(component, "data_format", ""))

        frame = np.array(component.data, copy=True)
        if channels > 1:
            frame = frame.reshape(height, width, channels)
        else:
            frame = frame.reshape(height, width)

        frame = self._convert_to_uint8(frame)
        frame = self._to_rgb(frame, data_format)
        if frame.shape[0] != self.config.height or frame.shape[1] != self.config.width:
            frame = cv2.resize(
                frame,
                (self.config.width, self.config.height),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(frame)

    def _to_rgb(self, frame: np.ndarray, data_format: str) -> np.ndarray:
        if frame.ndim == 2:
            conversion = _BAYER_TO_RGB.get(data_format)
            if conversion is not None:
                return cv2.cvtColor(frame, conversion)
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if frame.shape[2] >= 3:
            return frame[:, :, :3]

        raise ValueError(
            f"Unsupported frame shape for camera {self.config.name}: {frame.shape}"
        )

    @staticmethod
    def _convert_to_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        if np.issubdtype(frame.dtype, np.integer):
            max_value = max(int(np.iinfo(frame.dtype).max), 1)
            return ((frame.astype(np.float32) / max_value) * 255.0).clip(0, 255).astype(np.uint8)
        return np.clip(frame, 0, 255).astype(np.uint8)

    @staticmethod
    def _configure_node(node_map: Any, node_name: str, value: Any) -> None:
        try:
            node = getattr(node_map, node_name)
        except AttributeError:
            return

        if not getattr(node, "is_writable", True):
            return

        try:
            node.value = value
        except Exception:
            logger.debug("Skipping camera node %s=%s", node_name, value, exc_info=True)
