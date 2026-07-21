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
import open3d as o3d

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

# D415 / D415e RGB + depth video modes only expose these frame rates (not e.g. 20).
_SUPPORTED_STREAM_FPS: tuple[int, ...] = (6, 15, 30, 60, 90)


def _snap_stream_fps(requested: int | None) -> int:
    if requested is None:
        return 30
    req = int(max(1, min(requested, 90)))
    if req in _SUPPORTED_STREAM_FPS:
        return req
    return min(_SUPPORTED_STREAM_FPS, key=lambda f: abs(f - req))


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
        # Full-resolution color frame (before policy-output resize), kept in sync with
        # _last_depth so get_full_point_cloud() can look up per-pixel RGB values.
        self._last_color_full: np.ndarray | None = None
        self._vertices: np.ndarray | None = None
        self._depth_scale: float = 0.001

        self._intrinsics = np.asarray(config.intrinsic_matrix, dtype=np.float64)
        self._r_world_from_cam = np.asarray(config.r_cam_in_world, dtype=np.float64)
        self._t_world_from_cam = np.asarray(config.t_cam_in_world, dtype=np.float64)
        self._rng = np.random.default_rng()

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

    def connect(self, warmup: bool = False) -> None:
        self.disconnect()

        cfg = rs.config()
        # Prefer pinning the device by serial number when supplied; otherwise
        # let the pipeline grab whichever D400e the context returns.
        if self._serial:
            cfg.enable_device(self._serial)

        policy_fps = self._config.fps
        if self._config.streaming_fps is not None:
            stream_fps = _snap_stream_fps(self._config.streaming_fps)
        else:
            stream_fps = _snap_stream_fps(
                int(policy_fps) if policy_fps is not None else None
            )
        if policy_fps is not None and stream_fps != int(policy_fps):
            logger.warning(
                "FRAMOS %s: pipeline FPS %d is not supported by the device; using %d",
                self._name,
                int(policy_fps),
                stream_fps,
            )

        if self._config.enable_depth:
            cfg.enable_stream(
                rs.stream.depth,
                int(self._config.depth_width),
                int(self._config.depth_height),
                _FORMAT_LOOKUP.get(self._config.depth_format.lower(), rs.format.z16),
                stream_fps,
            )
        if self._config.enable_color:
            cfg.enable_stream(
                rs.stream.color,
                int(self._config.color_width),
                int(self._config.color_height),
                _FORMAT_LOOKUP.get(self._config.color_format.lower(), rs.format.bgr8),
                stream_fps,
            )

        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(cfg)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise RuntimeError(
                f"FRAMOS {self._name}: pipeline.start failed: {exc}. "
                "Check stream resolution/format/FPS against rs-enumerate-devices / "
                "sensor.get_stream_profiles(); LeRobot `fps` is often 20 but RealSense "
                "requires 6/15/30/60/90 — set `streaming_fps` on FramosCameraConfig if needed."
            ) from exc

        self._pipeline = pipeline
        self._profile = profile

        if self._config.enable_color and self._config.enable_depth:
            align_target = _STREAM_LOOKUP.get(self._config.align_to.lower(), rs.stream.color)
            self._aligner = rs.align(align_target)
        else:
            self._aligner = None

        device = profile.get_device()
        try:
            self._depth_scale = float(device.first_depth_sensor().get_depth_scale())
        except Exception:
            self._depth_scale = 0.001
        self._apply_options(device)

        # if warmup:
        #     # FRAMOS D415e color sensor needs a few frames before AE settles.
        #     for _ in range(10):
        #         try:
        #             self._pipeline.wait_for_frames(timeout_ms=500)
        #         except Exception:
        #             pass
        #     self._last_color = self.read()

        logger.info(
            "Connected FRAMOS camera %s (sn=%s ip=%s)",
            self._name,
            self._serial,
            self._ip,
        )

    def read(self) -> NDArray[Any]:
        if self._config.enable_color:
            return self._fetch_color(timeout_ms=1000.0, allow_stale=False)
        return self.read_depth()

    def async_read(self, timeout_ms: float = 500) -> NDArray[Any]:
        if self._config.enable_color:
            return self._fetch_color(timeout_ms=timeout_ms, allow_stale=True)
        return self.read_depth(timeout_ms=timeout_ms)

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
        # print("hi")
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

    def get_cropped_point_cloud(
        self,
        center: np.ndarray | None = None,
        radius_m: float | None = None,
        num_points: int = 2048,
        stride: int = 2,
    ) -> np.ndarray:
        """Project depth to world space, optionally crop to a world-axis-aligned
        BOX of half-extent `radius_m` around `center` (matching the sim collect
        crop; see STUDENT_INPUT_PARITY.md F8), then randomly subsample to
        `num_points`. Cropping/downsampling happens here, before the array
        crosses the thread-pool boundary.

        Hot path in the control loop (~2-4 ms at 720p, stride 2):
        - The pixel grid is strided before deprojection; a 720p frame still leaves
          ~10x more crop candidates than num_points, so the sample distribution is
          unaffected while the crop test shrinks by stride^2.
        - Crop math runs in float32; depth is uint16 * scale so no isfinite pass
          is needed (the product is always finite).
        - Two-stage crop: a camera-frame prefilter with the circumscribed ball
          (radius sqrt(3)*radius_m — rotation-invariant, so valid before the R/t
          transform), then the exact world-frame L-inf box test on the survivors.
          Only ball survivors get the matmul.
        """
        depth_image = self._last_depth
        if depth_image is None:
            return np.zeros((num_points, 3), dtype=np.float32)

        stride = max(1, int(stride))
        depth_m = np.asarray(depth_image)[::stride, ::stride].astype(np.float32) * self._depth_scale
        yy, xx = np.nonzero(depth_m > 0.0)
        if yy.size == 0:
            return np.zeros((num_points, 3), dtype=np.float32)
        z = depth_m[yy, xx]
        fx = float(self._intrinsics[0, 0])
        fy = float(self._intrinsics[1, 1])
        cx = float(self._intrinsics[0, 2])
        cy = float(self._intrinsics[1, 2])

        # Deproject to camera frame (strided indices map back to full-res pixels).
        x_cam = (xx.astype(np.float32) * stride - cx) * z / fx
        y_cam = (yy.astype(np.float32) * stride - cy) * z / fy

        if center is not None and radius_m is not None:
            center_w = np.asarray(center, dtype=np.float64)
            # center_cam = R^T @ (center_world - t)
            center_cam = (self._r_world_from_cam.T @ (center_w - self._t_world_from_cam)).astype(np.float32)
            # Circumscribed-ball prefilter (camera frame), then exact box test (world).
            d2 = (x_cam - center_cam[0]) ** 2 + (y_cam - center_cam[1]) ** 2 + (z - center_cam[2]) ** 2
            keep = np.flatnonzero(d2 <= 3.0 * np.float32(radius_m) ** 2)
            if keep.size == 0:
                return np.zeros((num_points, 3), dtype=np.float32)
            cam_points = np.stack((x_cam[keep], y_cam[keep], z[keep]), axis=1).astype(np.float64, copy=False)
            world_points = (self._r_world_from_cam @ cam_points.T).T + self._t_world_from_cam
            world_points = world_points[np.max(np.abs(world_points - center_w), axis=1) <= radius_m]
            if world_points.shape[0] == 0:
                return np.zeros((num_points, 3), dtype=np.float32)
            sel = self._rng.choice(world_points.shape[0], size=num_points,
                                   replace=world_points.shape[0] < num_points)
            return world_points[sel].astype(np.float32)

        sel = self._rng.choice(z.size, size=num_points, replace=z.size < num_points)
        cam_points = np.stack((x_cam[sel], y_cam[sel], z[sel]), axis=1).astype(np.float64, copy=False)
        world_points = (self._r_world_from_cam @ cam_points.T).T + self._t_world_from_cam
        return world_points.astype(np.float32)

    def get_full_point_cloud(self) -> np.ndarray:
        """Return ALL valid depth pixels projected to world space.

        Uses the depth frame cached by the most recent async_read / _fetch_color call.
        No subsampling is applied — the returned cloud may contain tens of thousands
        of points depending on camera resolution and scene coverage.

        Returns:
            float32 (N, 3) array in world-frame metres; shape (0, 3) if no depth
            data is available.
        """
        depth_image = self._last_depth
        if depth_image is None:
            return np.zeros((0, 3), dtype=np.float32)

        depth_m = np.asarray(depth_image, dtype=np.float32) * self._depth_scale
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        yy, xx = np.nonzero(valid)
        z = depth_m[yy, xx]
        fx = float(self._intrinsics[0, 0])
        fy = float(self._intrinsics[1, 1])
        cx = float(self._intrinsics[0, 2])
        cy = float(self._intrinsics[1, 2])

        x = (xx.astype(np.float32) - cx) * z / fx
        y = (yy.astype(np.float32) - cy) * z / fy
        cam_points = np.stack((x, y, z), axis=1).astype(np.float64, copy=False)
        world_points = (self._r_world_from_cam @ cam_points.T).T + self._t_world_from_cam
        xyz = world_points.astype(np.float32)

        # Attach per-point RGB when the full-res color frame is available and
        # its pixel grid matches the depth image (guaranteed when the aligner is
        # set to align depth→color, which is the default).
        # color_img = self._last_color_full
        # if (
        #     color_img is not None
        #     and color_img.ndim == 3
        #     and color_img.shape[:2] == depth_image.shape[:2]
        # ):
        #     rgb = color_img[yy, xx].astype(np.float32) / 255.0  # (N, 3) in [0, 1]
        #     return np.concatenate([xyz, rgb], axis=1)            # (N, 6)
        return xyz

    def get_depth(self) -> list[tuple[float, float, float]]:
        depth_image = self._last_depth
        if depth_image is None:
            return []

        depth_m = np.asarray(depth_image, dtype=np.float32) * self._depth_scale
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        if not np.any(valid):
            return []

        yy, xx = np.nonzero(valid)
        if yy.size >= 2048:
            idx = np.linspace(0, yy.size - 1, 2048, dtype=np.int64)
            yy = yy[idx]
            xx = xx[idx]
        else:
            reps = (2048 + yy.size - 1) // yy.size
            yy = np.tile(yy, reps)[:2048]
            xx = np.tile(xx, reps)[:2048]

        z = depth_m[yy, xx]
        fx = float(self._intrinsics[0, 0])
        fy = float(self._intrinsics[1, 1])
        cx = float(self._intrinsics[0, 2])
        cy = float(self._intrinsics[1, 2])

        x = (xx.astype(np.float32) - cx) * z / fx
        y = (yy.astype(np.float32) - cy) * z / fy
        cam_points = np.stack((x, y, z), axis=1).astype(np.float64, copy=False)
        world_points = (self._r_world_from_cam @ cam_points.T).T + self._t_world_from_cam
        return [(float(p[0]), float(p[1]), float(p[2])) for p in world_points]

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
            # Cache full-resolution frame (format-converted, not yet resized) so
            # get_full_point_cloud() can look up per-pixel RGB at depth resolution.
            full_arr = np.ascontiguousarray(arr)
            if full_arr.ndim == 3 and full_arr.shape[2] == 3 and self._config.color_format.lower() == "bgr8":
                full_arr = cv2.cvtColor(full_arr, cv2.COLOR_BGR2RGB)
            self._last_color_full = full_arr

            if arr.ndim == 3 and (
                arr.shape[0] != self._output_height or arr.shape[1] != self._output_width
            ):
                arr = cv2.resize(
                    arr,
                    (self._output_width, self._output_height),
                    interpolation=cv2.INTER_AREA,
                )
            arr = np.ascontiguousarray(arr)
            # Match lerobot_camera_arv (Bayer→RGB): tensors are RGB channel order for LeRobot/stack.
            if arr.ndim == 3 and arr.shape[2] == 3 and self._config.color_format.lower() == "bgr8":
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
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
