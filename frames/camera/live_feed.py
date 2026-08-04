"""Display a live color feed from a specific FRAMOS D400e camera.

Picks the camera matching TARGET_SERIAL (or the first device found if
TARGET_SERIAL is None), then shows the RGB stream continuously. Uses an
OpenCV window when available and falls back to matplotlib if OpenCV highgui
is missing.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
import pyrealsense2 as rs

# Set this to the serial number of the camera you want to use, e.g. "123456789012".
# Leave as None to just grab the first device the context finds.
TARGET_SERIAL: str | None = "6CD146030D71"

COLOR_W, COLOR_H = 1280, 720
FPS = 30  # native stream FPS requested from the camera

WARMUP_FRAMES = 15
FRAME_TIMEOUT_MS = 5000
WINDOW_NAME = "FRAMOS live feed"


def _make_display_backend():
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        return "opencv", None
    except cv2.error as exc:
        print(f"OpenCV GUI unavailable ({exc}); falling back to matplotlib.")
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(num=WINDOW_NAME)
        ax.set_title(WINDOW_NAME)
        ax.axis("off")
        image = ax.imshow(np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8))
        fig.show()
        return "matplotlib", (plt, fig, ax, image)


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

    display_backend, display_state = _make_display_backend()

    try:
        # Warm up so auto-exposure etc. can settle before we display frames.
        for _ in range(WARMUP_FRAMES):
            try:
                pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except Exception:
                pass

        print("Showing live feed. Press 'q' in the window or Ctrl+C to stop.")
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

            color_np = np.asanyarray(color.get_data())
            if display_backend == "opencv":
                # Frames arrive as RGB; OpenCV expects BGR for display.
                bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
                cv2.imshow(WINDOW_NAME, bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                plt, fig, ax, image = display_state
                image.set_data(color_np)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

                if not plt.fignum_exists(fig.number):
                    break

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        if display_backend == "opencv":
            cv2.destroyAllWindows()
        else:
            plt, fig, ax, image = display_state
            plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())