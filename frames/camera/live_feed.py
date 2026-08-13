"""Display a live color feed from a specific FRAMOS D400e camera.

Picks the camera matching TARGET_SERIAL (or the first device found if
TARGET_SERIAL is None), then shows the RGB stream continuously. Uses an
OpenCV window when available and falls back to matplotlib if OpenCV highgui
is missing.
"""
from __future__ import annotations

import sys

import cv2
import franka_config as fc
import numpy as np
import pyrealsense2 as rs

# Camera to show, by id in config/cameras.yaml. Set TARGET_SERIAL to None to
# just grab the first device the context finds.
CAM_KEY = "cam_2"
TARGET_SERIAL: str | None = fc.camera(CAM_KEY).serial_number

_FRAMOS = fc.framos_defaults()
COLOR_W, COLOR_H = _FRAMOS["color_width"], _FRAMOS["color_height"]
# Display square size (crop left/right to make frames square)
DISPLAY_SIZE = min(COLOR_W, COLOR_H)
FPS = fc.camera_stream_fps()  # native stream FPS requested from the camera

# Overlay reference image (path relative to workspace). Set to None to disable.
OVERLAY_PATH: str | None = "frames/sim-ref.png"
# Initial global overlay alpha in [0, 1]
OVERLAY_ALPHA = 0.5

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
        # Use a square display buffer so matplotlib window matches the
        # center-cropped frames (left/right margins removed).
        image = ax.imshow(np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.uint8))
        fig.show()
        return "matplotlib", (plt, fig, ax, image)


def _load_overlay(path: str) -> np.ndarray | None:
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None:
        return None
    # cv2 reads BGR(A) — convert to RGB(A)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[..., :4] if img.shape[2] == 4 else img[..., :3]
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # 4 channels: B G R A -> R G B A
            bgr = img[..., :3]
            a = img[..., 3:4]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = np.concatenate([rgb, a], axis=2)
    return img


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

    overlay_img = None
    overlay_enabled = False
    overlay_alpha = float(OVERLAY_ALPHA)
    if OVERLAY_PATH is not None:
        overlay_img = _load_overlay(OVERLAY_PATH)
        if overlay_img is not None:
            overlay_enabled = True

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
            # Center-crop horizontally to make the frame square (crop left/right)
            h, w = color_np.shape[:2]
            if w != h:
                crop_w = min(w, h)
                crop_x = (w - crop_w) // 2
                color_np = color_np[:, crop_x:crop_x + crop_w]
            # If overlay present and enabled, resize and alpha-blend.
            if overlay_img is not None and overlay_enabled:
                # Resize overlay to match frame
                oh = overlay_img.shape[0]
                ow = overlay_img.shape[1]
                th, tw = color_np.shape[:2]
                if (oh, ow) != (th, tw):
                    # Use INTER_AREA for downsampling, INTER_LINEAR for up
                    interp = cv2.INTER_AREA if (ow > tw or oh > th) else cv2.INTER_LINEAR
                    overlay_resized = cv2.resize(overlay_img, (tw, th), interpolation=interp)
                else:
                    overlay_resized = overlay_img

                if overlay_resized.ndim == 3 and overlay_resized.shape[2] == 4:
                    rgb = overlay_resized[..., :3].astype(np.float32)
                    a = overlay_resized[..., 3].astype(np.float32) / 255.0
                    a = a * overlay_alpha
                    a = a[..., None]
                else:
                    rgb = overlay_resized.astype(np.float32)
                    a = overlay_alpha
                    a = np.full((th, tw, 1), float(a), dtype=np.float32)

                # color_np is uint8 RGB; blend in float then convert back
                fg = rgb
                bg = color_np.astype(np.float32)
                blended = (a * fg) + ((1.0 - a) * bg)
                color_np = np.clip(blended, 0, 255).astype(np.uint8)
            if display_backend == "opencv":
                # Frames arrive as RGB; OpenCV expects BGR for display.
                bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
                cv2.imshow(WINDOW_NAME, bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                # Toggle overlay with 'o', increase/decrease alpha with '+'/'-'
                if key == ord('o'):
                    overlay_enabled = not overlay_enabled
                if key == ord('+') or key == ord('='):
                    overlay_alpha = min(1.0, overlay_alpha + 0.05)
                if key == ord('-'):
                    overlay_alpha = max(0.0, overlay_alpha - 0.05)
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