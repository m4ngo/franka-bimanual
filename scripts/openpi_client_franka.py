#!/usr/bin/env python3
"""OpenPI inference client for a single right-arm Franka.

Hardware defaults (right arm from bimanual setup):
  Robot server  192.168.3.10   port 18812
  Robot arm     192.168.201.10
  Gripper       192.168.2.20

Cameras:
  Base  (FRAMOS D71)  192.168.0.116  sn=6CD146030D71
  Wrist (ARV BFS)     192.168.1.138

Usage:
  python3 scripts/openpi_client_franka.py \\
      --server-ip <openpi-host> \\
      --server-port 8000 \\
      --prompt "pick up the red block"
"""

from __future__ import annotations

import argparse
import logging
import ssl as _ssl
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import websockets.sync.client

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_ROOT = Path("/home/franka/lerobot_ur5e_gello")
for _p in (WORKSPACE_ROOT, OPENPI_CLIENT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lerobot_camera_arv import ArvCamera, ArvCameraConfig
from lerobot_camera_framos import FramosCamera, FramosCameraConfig
from lerobot_robot_bimanual_franka.lerobot_robot_bimanual_franka.franka_process import (
    MultiRobotWrapper,
)
from lerobot_robot_bimanual_franka.lerobot_robot_bimanual_franka.wsg import WSG
from openpi_client import msgpack_numpy as _msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy

logger = logging.getLogger(__name__)


# ── WebSocket client (plain ws:// or wss://) ─────────────────────────────────

class _WebsocketClient(WebsocketClientPolicy):
    """Drop-in replacement for WebsocketClientPolicy with optional SSL.

    The upstream class always forces TLS (wss://), which breaks against the
    local openpi server that listens on plain ws://.  This subclass passes
    ssl=None for plain connections and only activates TLS when *use_ssl=True*.
    """

    def __init__(
        self,
        host: str,
        port: int | None,
        api_key: str | None,
        use_ssl: bool = False,
    ) -> None:
        if host.startswith("ws://") or host.startswith("wss://"):
            self._uri = host
        elif use_ssl:
            self._uri = f"wss://{host}"
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._packer = _msgpack_numpy.Packer()
        self._api_key = api_key
        self._use_ssl = use_ssl
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        logger.info("Waiting for OpenPI server at %s ...", self._uri)
        ssl_context: _ssl.SSLContext | None = None
        if self._use_ssl:
            ssl_context = _ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = _ssl.CERT_NONE

        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    ssl=ssl_context,
                )
                metadata = _msgpack_numpy.unpackb(conn.recv())
                logger.info("Connected to OpenPI server.")
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Server not ready yet, retrying in 5 s ...")
                time.sleep(5)
            except _ssl.SSLError as exc:
                logger.error("SSL error (try without --ssl?): %s", exc)
                time.sleep(5)


# ── Robot constants ──────────────────────────────────────────────────────────

NUM_JOINTS = 7
_CAMERA_READ_TIMEOUT_MS = 5.0
_PROCESS_STARTUP_S = 1.0
_CONNECT_RETRIES = 3
_CONNECT_TIMEOUT_S = 10.0
_RETRY_SLEEP_S = 1.0

# FR3 joint velocity limits (rad/s) — joints 1-4 @ 2.175, joints 5-7 @ 2.61.
_MAX_JOINT_VEL = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]) * 0.9

# FR3 joint acceleration limits (rad/s²).
# Limiting Δv/step prevents joint_motion_generator_velocity_discontinuity reflexes.
_MAX_JOINT_ACCEL = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]) * 0.75

# ── Hardware defaults ────────────────────────────────────────────────────────

_ARM_KEY = "r"

_DEFAULT_R_SERVER_IP  = "192.168.3.10"
_DEFAULT_R_ROBOT_IP   = "192.168.201.10"
_DEFAULT_R_GRIPPER_IP = "192.168.2.20"
_DEFAULT_R_PORT       = 18812

_DEFAULT_BASE_CAM_IP = "192.168.0.116"
_DEFAULT_BASE_CAM_SN = "6CD146030D71"
_DEFAULT_WRIST_CAM_IP = "192.168.1.138"

_INFERENCE_ERROR_SLEEP_S = 0.1


# ── Single-arm robot wrapper ─────────────────────────────────────────────────

class SingleFrankaRobot:
    """Right-arm Franka with one FRAMOS base camera and one ARV wrist camera.

    Observation keys:
        joint_1 … joint_7  – joint positions in radians
        gripper             – normalized finger gap in [0, 1]
        base_image          – HxWx3 uint8 RGB from the FRAMOS workspace camera
        wrist_image         – HxWx3 uint8 RGB from the ARV wrist camera

    ``send_action`` accepts an 8-element vector [joint_vel_1..7, gripper], where
    joint velocities are in rad/s (DROID joint_velocity action space) and
    gripper is an absolute position target in [0, 1].
    """

    def __init__(
        self,
        server_ip: str,
        robot_ip: str,
        gripper_ip: str,
        port: int,
        base_cam_cfg: FramosCameraConfig,
        wrist_cam_cfg: ArvCameraConfig,
        fps: float = 15.0,
    ) -> None:
        self._server_ip = server_ip
        self._robot_ip = robot_ip
        self._port = port
        self._step_period = 1.0 / fps

        self._robot_manager = MultiRobotWrapper()
        self._gripper = WSG(name=_ARM_KEY, TCP_IP=gripper_ip, do_print=False)
        self._base_camera = FramosCamera(base_cam_cfg)
        self._wrist_camera = ArvCamera(wrist_cam_cfg)
        self._camera_pool = ThreadPoolExecutor(max_workers=2)
        self._prev_vel = np.zeros(NUM_JOINTS)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._robot_manager.add_robot(
            _ARM_KEY,
            self._server_ip,
            self._robot_ip,
            self._port,
            use_ee_delta=False,
        )
        time.sleep(_PROCESS_STARTUP_S)
        self._probe_arm()
        self._prev_vel[:] = 0.0
        self._gripper.home()
        self._base_camera.connect()
        self._wrist_camera.connect()
        logger.info("SingleFrankaRobot connected.")

    def disconnect(self) -> None:
        self._camera_pool.shutdown(wait=False)
        for cam in (self._base_camera, self._wrist_camera):
            try:
                cam.disconnect()
            except Exception as exc:
                logger.debug("Camera disconnect error: %s", exc)
        self._robot_manager.shutdown()
        try:
            self._gripper.close()
        except Exception as exc:
            logger.debug("Gripper close error: %s", exc)
        logger.info("SingleFrankaRobot disconnected.")

    def _probe_arm(self) -> None:
        last_exc: Exception | None = None
        for _ in range(_CONNECT_RETRIES):
            try:
                self._robot_manager.current_kinematic_state(
                    _ARM_KEY, timeout_s=_CONNECT_TIMEOUT_S
                )
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(_RETRY_SLEEP_S)
        raise RuntimeError(
            f"Failed to communicate with robot at {self._robot_ip}: {last_exc}"
        )

    # ── Observation ───────────────────────────────────────────────────────────

    def get_observation(self) -> dict:
        base_fut = self._camera_pool.submit(
            self._base_camera.async_read, _CAMERA_READ_TIMEOUT_MS
        )
        wrist_fut = self._camera_pool.submit(
            self._wrist_camera.async_read, _CAMERA_READ_TIMEOUT_MS
        )

        q, *_ = self._robot_manager.current_kinematic_state(_ARM_KEY)
        obs: dict = {f"joint_{i + 1}": float(q[i]) for i in range(NUM_JOINTS)}

        raw_pos = self._gripper.position
        obs["gripper"] = (0.0 if raw_pos is None else raw_pos) / WSG.GRIPPER_TRUE_MAX_MM

        try:
            obs["base_image"] = base_fut.result()
        except Exception as exc:
            logger.warning("Base camera read failed: %s", exc)
            obs["base_image"] = self._base_camera.blank_frame()

        try:
            obs["wrist_image"] = wrist_fut.result()
        except Exception as exc:
            logger.warning("Wrist camera read failed: %s", exc)
            obs["wrist_image"] = self._wrist_camera.blank_frame()

        return obs

    # ── Action ────────────────────────────────────────────────────────────────

    def send_action(self, action: np.ndarray) -> None:
        """Send one action step to the arm and gripper.

        Args:
            action: 1-D array of length 8.
                action[:7]  – joint velocities in rad/s (DROID joint_velocity space).
                action[7]   – gripper absolute position in [0, 1].
        """
        gripper_norm = float(np.clip(action[NUM_JOINTS], 0.0, 1.0))
        self._gripper.move(gripper_norm * WSG.GRIPPER_TRUE_MAX_MM, blocking=False)

        velocity = np.asarray(action[:NUM_JOINTS], dtype=np.float64)

        # Hard velocity ceiling.
        velocity = np.clip(velocity, -_MAX_JOINT_VEL, _MAX_JOINT_VEL)

        # Acceleration limit: Δv per step ≤ max_accel × dt.
        # Prevents joint_motion_generator_velocity_discontinuity reflexes.
        max_delta = _MAX_JOINT_ACCEL * self._step_period
        velocity = np.clip(velocity, self._prev_vel - max_delta, self._prev_vel + max_delta)
        self._prev_vel = velocity.copy()

        self._robot_manager.move_joint_velocity_batch(
            {_ARM_KEY: velocity.tolist()}, asynchronous=True
        )


# ── Inference loop ────────────────────────────────────────────────────────────

def inference_loop(
    client: _WebsocketClient,
    robot: SingleFrankaRobot,
    stop_event: dict,
    prompt: str,
    fps: float,
    debug: bool = False,
) -> None:
    """Run the action-chunking inference loop until ``stop_event["stop"]`` is set.

    Observation keys sent to the DROID OpenPI server
    (see openpi/src/openpi/policies/droid_policy.py):
        observation/exterior_image_1_left – FRAMOS base camera (224x224x3 uint8)
        observation/wrist_image_left      – ARV wrist camera   (224x224x3 uint8)
        observation/joint_position        – 7-element float64 array (radians)
        observation/gripper_position      – 1-element float64 array in [0, 1]
        prompt                            – task description string

    The server returns ``actions[:, :8]``: 7 joint targets (radians) + 1 gripper.
    """
    action_queue: deque = deque()
    step_period = 1.0 / fps
    obs: dict | None = None
    chunk_count = 0

    while not stop_event["stop"]:
        loop_start = time.perf_counter()

        if not action_queue:
            obs = robot.get_observation()
            joint_pos = np.array(
                [obs[f"joint_{i}"] for i in range(1, NUM_JOINTS + 1)],
                dtype=np.float64,
            )
            gripper_pos = np.array([obs["gripper"]], dtype=np.float64)

            if debug:
                np.set_printoptions(precision=3, suppress=True)
                logger.info(
                    "[chunk %d] OBS joints (rad): %s  gripper: %.3f",
                    chunk_count,
                    joint_pos,
                    gripper_pos[0],
                )

            obs_dict = {
                "observation/exterior_image_1_left": obs["base_image"],
                "observation/wrist_image_left": obs["wrist_image"],
                "observation/joint_position": joint_pos,
                # Must be 1-D — DroidInputs concatenates it with joint_position.
                "observation/gripper_position": gripper_pos,
                "prompt": prompt,
            }

            try:
                result = client.infer(obs_dict)
                chunk = result["actions"]
            except Exception as exc:
                logger.error("Inference error: %s", exc)
                time.sleep(_INFERENCE_ERROR_SLEEP_S)
                continue

            if not isinstance(chunk, np.ndarray):
                chunk = np.array(chunk)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)

            if debug:
                err = chunk[0, :NUM_JOINTS] - joint_pos
                logger.info(
                    "[chunk %d] ACTION[0] joints (rad): %s  gripper: %.3f",
                    chunk_count,
                    chunk[0, :NUM_JOINTS],
                    chunk[0, NUM_JOINTS],
                )
                logger.info(
                    "[chunk %d] ERROR (target-current):  %s  max_abs=%.3f rad",
                    chunk_count,
                    err,
                    np.abs(err).max(),
                )

            chunk_count += 1
            action_queue.extend(chunk)

        if action_queue:
            action = np.asarray(action_queue.popleft(), dtype=np.float64)
            if debug:
                vel = action[:NUM_JOINTS]
                vel_clipped = np.clip(vel, -_MAX_JOINT_VEL, _MAX_JOINT_VEL)
                logger.debug("  vel_policy=%s", vel.round(3))
                if not np.array_equal(vel, vel_clipped):
                    logger.debug("  vel_clipped=%s", vel_clipped.round(3))
            robot.send_action(action)

        elapsed = time.perf_counter() - loop_start
        remaining = step_period - elapsed
        if remaining > 0:
            time.sleep(remaining)


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OpenPI inference on the right-arm Franka (single arm).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # OpenPI server
    p.add_argument("--server-ip",   default="0.0.0.0",
                   help="OpenPI server hostname or IP")
    p.add_argument("--server-port", type=int, default=8000,
                   help="OpenPI server port")
    p.add_argument("--api-key",     default=None,
                   help="OpenPI API key (omit if not required)")
    p.add_argument("--ssl",         action="store_true", default=False,
                   help="Use TLS (wss://) — omit for local servers running plain ws://")
    p.add_argument("--prompt",      required=True,
                   help="Language task prompt sent to the policy")
    p.add_argument("--fps",         type=float, default=15.0,
                   help="Control-loop frequency (Hz)")
    p.add_argument("--debug",       action="store_true", default=False,
                   help="Log joint obs, policy target, and PD error each chunk")

    # Right-arm hardware
    p.add_argument("--r-server-ip",  default=_DEFAULT_R_SERVER_IP,
                   help="Franka server IP (net_franky proxy)")
    p.add_argument("--r-robot-ip",   default=_DEFAULT_R_ROBOT_IP,
                   help="Franka robot IP")
    p.add_argument("--r-gripper-ip", default=_DEFAULT_R_GRIPPER_IP,
                   help="WSG gripper IP")
    p.add_argument("--r-port",       type=int, default=_DEFAULT_R_PORT,
                   help="net_franky proxy port")

    # Camera hardware
    p.add_argument("--base-cam-ip",  default=_DEFAULT_BASE_CAM_IP,
                   help="FRAMOS base camera IP")
    p.add_argument("--base-cam-sn",  default=_DEFAULT_BASE_CAM_SN,
                   help="FRAMOS base camera serial number")
    p.add_argument("--wrist-cam-ip", default=_DEFAULT_WRIST_CAM_IP,
                   help="ARV wrist camera IP")
    p.add_argument("--cam-width",    type=int, default=224,
                   help="Output image width (pixels)")
    p.add_argument("--cam-height",   type=int, default=224,
                   help="Output image height (pixels)")
    p.add_argument("--cam-fps",      type=int, default=15,
                   help="Camera streaming frame rate")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    base_cam_cfg = FramosCameraConfig(
        name="workspace_framos_d71",
        ip=args.base_cam_ip,
        serial_number=args.base_cam_sn,
        fps=args.cam_fps,
        width=args.cam_width,
        height=args.cam_height,
    )
    wrist_cam_cfg = ArvCameraConfig(
        name="gripper_bfs_23595723",
        ip=args.wrist_cam_ip,
        fps=args.cam_fps,
        width=args.cam_width,
        height=args.cam_height,
    )

    robot = SingleFrankaRobot(
        server_ip=args.r_server_ip,
        robot_ip=args.r_robot_ip,
        gripper_ip=args.r_gripper_ip,
        port=args.r_port,
        base_cam_cfg=base_cam_cfg,
        wrist_cam_cfg=wrist_cam_cfg,
        fps=args.fps,
    )

    stop_event: dict = {"stop": False}
    client: _WebsocketClient | None = None

    try:
        logger.info("Connecting robot (right arm)...")
        robot.connect()

        logger.info(
            "Connecting to OpenPI server at %s:%d ...",
            args.server_ip,
            args.server_port,
        )
        client = _WebsocketClient(
            host=args.server_ip,
            port=args.server_port,
            api_key=args.api_key,
            use_ssl=args.ssl,
        )

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Starting inference loop at %.1f Hz — prompt: %r", args.fps, args.prompt)
        inference_loop(client, robot, stop_event, args.prompt, args.fps, debug=args.debug)

    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl-C).")
    except Exception:
        logger.exception("Fatal error in inference client.")
        raise
    finally:
        stop_event["stop"] = True
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        robot.disconnect()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
