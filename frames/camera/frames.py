"""Continuously capture color frames from a specific FRAMOS D400e camera at 10 Hz.

Picks the camera matching TARGET_SERIAL (or the first device found if
TARGET_SERIAL is None) and saves an RGB PNG into `frames/` at a fixed
rate of 10 frames/second.

Press Ctrl+C to stop.
"""
from __future__ import annotations

import os
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs

OUTDIR = "frames"

# Set this to the serial number of the camera you want to use, e.g. "123456789012".
# Leave as None to just grab the first device the context finds.
TARGET_SERIAL: str | None = "6CD146030D71"

COLOR_W, COLOR_H = 1280, 720
FPS = 30  # native stream FPS requested from the camera

WARMUP_FRAMES = 15
FRAME_TIMEOUT_MS = 5000

CAPTURE_HZ = 1.0
CAPTURE_INTERVAL_S = 1.0 / CAPTURE_HZ


def pick_device(ctx: rs.context) -> rs.device | None:
    devices = list(ctx.devices)
    if not devices:
        print("No FRAMOS / RealSense devices found.")
        return None

    print(f"Found {len(devices)} device(s):")
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        sn = dev.get_info(rs.camera_info.serial_number)
        try:
            ip = dev.get_info(rs.camera_info.ip_address)
        except Exception:
            ip = "n/a"
        try:
            fw = dev.get_info(rs.camera_info.firmware_version)
        except Exception:
            fw = "n/a"
        print(f"  - {name} sn={sn} ip={ip} fw={fw}")

    if TARGET_SERIAL is None:
        chosen = devices[0]
        print(f"No TARGET_SERIAL set; using first device sn={chosen.get_info(rs.camera_info.serial_number)}")
        return chosen

    for dev in devices:
        if dev.get_info(rs.camera_info.serial_number) == TARGET_SERIAL:
            return dev

    print(f"ERROR: no device with serial number {TARGET_SERIAL} found.")
    return None


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)

    if not hasattr(rs, "d400e"):
        print(
            "ERROR: this pyrealsense2 build does not expose rs.d400e; "
            "rebuild from /usr/src/librealsense2 with -DBUILD_PYTHON_BINDINGS=true."
        )
        return 1

    ctx = rs.context()
    dev = pick_device(ctx)
    if dev is None:
        return 1

    sn = dev.get_info(rs.camera_info.serial_number)
    print(f"Using device sn={sn}")

    cfg = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, FPS)

    pipeline = rs.pipeline()
    try:
        pipeline.start(cfg)
    except Exception as exc:
        print(f"[{sn}] pipeline.start failed: {exc}")
        return 1

    try:
        # Warm up so auto-exposure etc. can settle before we start timing captures.
        for _ in range(WARMUP_FRAMES):
            try:
                pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except Exception:
                pass

        print(f"Capturing at {CAPTURE_HZ} Hz. Press Ctrl+C to stop.")
        frame_idx = 0
        next_capture_time = time.monotonic()

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except Exception as exc:
                print(f"[{sn}] wait_for_frames failed: {exc}")
                continue

            color = frames.get_color_frame()
            if not color:
                print(f"[{sn}] missing color frame")
                continue

            now = time.monotonic()
            if now < next_capture_time:
                # Not time to save a frame yet; keep pulling frames to stay
                # current, but don't write to disk.
                continue

            color_np = np.asanyarray(color.get_data())

            ts = time.strftime("%Y%m%d_%H%M%S")
            color_path = os.path.join(OUTDIR, f"FRAMOS_{sn}_{ts}_{frame_idx:06d}_color.png")

            # Frames are RGB in memory (`rgb8`); OpenCV PNG expects BGR.
            cv2.imwrite(color_path, cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))

            print(f"[{sn}] #{frame_idx} saved {color_path} ({color_np.shape})")

            frame_idx += 1
            # Schedule next capture based on the fixed cadence, not "now",
            # to avoid drift from processing/save time.
            next_capture_time += CAPTURE_INTERVAL_S
            if next_capture_time < now:
                # We fell behind (e.g. disk I/O stall); resync instead of
                # trying to burst-catch-up.
                next_capture_time = now + CAPTURE_INTERVAL_S

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())