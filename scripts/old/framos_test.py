"""Sanity-check FRAMOS D400e cameras via the FRAMOS-built pyrealsense2.

Saves an RGB PNG and a depth-colormap PNG (aligned to color) for each
detected D400e device into the `frames/` directory next to the existing
`camera_test.py` outputs.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUTDIR = "frames"
COLOR_W, COLOR_H = 1280, 720
DEPTH_W, DEPTH_H = 1280, 720
FPS = 30
WARMUP_FRAMES = 15
FRAME_TIMEOUT_MS = 5000


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)

    if not hasattr(rs, "d400e"):
        print(
            "ERROR: this pyrealsense2 build does not expose rs.d400e; "
            "rebuild from /usr/src/librealsense2 with -DBUILD_PYTHON_BINDINGS=true."
        )
        return 1

    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        print("No FRAMOS / RealSense devices found.")
        return 1

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

    align = rs.align(rs.stream.color)

    for dev in devices:
        sn = dev.get_info(rs.camera_info.serial_number)
        cfg = rs.config()
        cfg.enable_device(sn)
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)

        pipeline = rs.pipeline()
        try:
            pipeline.start(cfg)
        except Exception as exc:
            print(f"[{sn}] pipeline.start failed: {exc}")
            continue

        try:
            for _ in range(WARMUP_FRAMES):
                try:
                    pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
                except Exception:
                    pass

            frames = pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                print(f"[{sn}] missing color or depth frame")
                continue

            color_np = np.asanyarray(color.get_data())
            depth_np = np.asanyarray(depth.get_data())

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_np, alpha=0.03), cv2.COLORMAP_JET
            )

            color_path = os.path.join(OUTDIR, f"FRAMOS_{sn}_color.png")
            depth_path = os.path.join(OUTDIR, f"FRAMOS_{sn}_depth.png")
            cv2.imwrite(color_path, color_np)
            cv2.imwrite(depth_path, depth_vis)
            valid = (depth_np > 0).sum()
            print(
                f"[{sn}] saved {color_path} ({color_np.shape}) and "
                f"{depth_path} (valid_depth_px={valid})"
            )
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
