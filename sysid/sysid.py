"""System-ID trajectory replay: drive the robot along a reference trajectory and record response.

Usage
-----
python sysid/sysid.py <traj.hdf5> [--output sysid/data_replayed.hdf5] [--fps 20] [--kp 0] [--kd 0]

The input HDF5 must have the structure produced by the sim:
    f[group][episode][field] → (T, D) dataset

Fields consumed from the input:
    action  (T, 7) – [dx_norm, dy_norm, dz_norm, rx_norm, ry_norm, rz_norm, gripper]
                     Position deltas in units of _POS_SCALE (0.05 m); rotation as
                     axis-angle in units of _ROT_SCALE (0.5 rad).
    eef_pos (T, 3) – reference EE position (used for error visualisation only)
    qpos    (T, 7) – joint angles used to initialise the home pose

Fields recorded in the output HDF5 (same structure):
    action      (T, 7) – [dpos(3), drot_quat(4)] — position delta in metres and
                         rotation delta as unnormalized quaternion [drot/2, 1] (xyzw)
    eef_ang_vel (T, 3) – actual EE angular velocity from robot state
    eef_lin_vel (T, 3) – actual EE linear velocity from robot state
    eef_pos     (T, 3) – actual EE position from robot state
    eef_quat    (T, 4) – actual EE quaternion from robot state
    qpos        (T, 7) – actual joint angles
    qvel        (T, 7) – actual joint velocities
    t_sim       (T, 1) – wall-clock time since episode start
    tau_cmd     (T, 7) – zeros (not accessible via current RPC interface)

A comparison HTML visualization is also written alongside the HDF5.
"""

import argparse
import logging
import os
import select
import sys
import termios
import tty
import time
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "residual_wrapper"))

from env_wrapper import start_controller  # noqa: E402
from lerobot_robot_bimanual_franka import SingleArmFranka
from _viz import save_comparison_html  # noqa: E402

logger = logging.getLogger(__name__)

# Action kp/kd in [-1, 1]; send_action maps them via kp_gain = 10**kp.
# Default 0.0 → kp_gain = 1.0 (minimum, safest for an open-loop sysid replay).
_DEFAULT_KP = 0.0
_DEFAULT_KD = 0.0


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def parse_num_traj(filename: str) -> int:
    with h5py.File(filename, "r") as f:
        group_key = list(f.keys())[0]
        group = f[group_key]
        return len(group.keys())

def parse_traj(filename: str, index: int) -> tuple[str, dict[str, np.ndarray]]:
    """Load the first episode from a sysid HDF5 file.

    Expected structure: f[group_key][episode_key][field] → (T, D) dataset.
    All datasets are copied into numpy arrays so the file can be closed.
    """
    traj: dict[str, np.ndarray] = {}
    key = ""
    with h5py.File(filename, "r") as f:
        group_key = list(f.keys())[0]
        group = f[group_key]
        episode_key = list(group.keys())[index]
        key = episode_key
        episode = group[episode_key]
        for field in episode:
            traj[field] = episode[field][:]
    return (key, traj)


# ---------------------------------------------------------------------------
# Keyboard helpers (identical pattern to run_residual.py)
# ---------------------------------------------------------------------------

def _stdin_key_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


def _read_key() -> str:
    time.sleep(0.03)
    data = os.read(sys.stdin.fileno(), 16)
    if b"\x03" in data:
        return "ctrl_c"
    if data.startswith(b"\x1b[C") or data.startswith(b"\x1bOC"):
        return "right"
    return ""


# ---------------------------------------------------------------------------
# Episode replay loop
# ---------------------------------------------------------------------------

def _run_episode(
    controller: SingleArmFranka,
    traj: dict[str, np.ndarray],
    fps: float = 20.0,
    kp: float = _DEFAULT_KP,
    kd: float = _DEFAULT_KD,
    gripper_norm: float = 1.0,
) -> dict[str, np.ndarray]:
    """Replay *traj* on the robot and record the kinematic response.

    At each step the robot kinematic state is read via a direct call to
    ``controller.robot_manager.current_kinematic_state_batch`` (which returns
    q, dq, jacobian, ee_pos, ee_quat, ee_vel_6d).  That snapshot is stored in
    ``controller._cached_kin_state`` so the subsequent ``send_action`` call
    consumes it without an extra RPyC round-trip.

    Press right-arrow to end early, Ctrl-C to abort.

    Returns a dict of stacked numpy arrays (one row per step).
    """
    action_all = traj["action"]  # (T, 7): [dx_norm, dy_norm, dz_norm, rx_norm, ry_norm, rz_norm, gripper]
    n_steps = len(action_all)

    _POS_SCALE = 0.05  # metres per normalised unit (must match env_wrapper._POS_SCALE)
    _ROT_SCALE = 0.5   # radians per normalised unit (must match env_wrapper._ROT_SCALE)

    buf: dict[str, list] = {k: [] for k in (
        "action", "eef_ang_vel",
        "eef_lin_vel", "eef_pos", "eef_quat", "qpos", "qvel", "t_sim", "tau_cmd",
    )}

    dt = 1.0 / fps
    t_start = time.perf_counter()
    old_term = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    try:
        for step in range(n_steps):
            t_step = time.perf_counter()

            if _stdin_key_pressed():
                key = _read_key()
                if key == "ctrl_c":
                    raise KeyboardInterrupt
                if key == "right":
                    print("\r\nearly stop requested", flush=True)
                    break

            # --- read kinematic state ----------------------------------------
            # Store in _cached_kin_state so send_action re-uses it, avoiding
            # a redundant RPyC round-trip.
            kin = controller.robot_manager.current_kinematic_state_batch(["r"])
            controller._cached_kin_state = kin
            q, dq, _jac, ee_pos, ee_quat, ee_vel = kin["r"]
            # ee_vel layout: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]

            # --- build and send action ----------------------------------------
            dpos: np.ndarray = action_all[step][0:3].astype(np.float64) * _POS_SCALE
            drot: np.ndarray = action_all[step][3:6].astype(np.float64) * _ROT_SCALE

            # Convert axis-angle drot to unnormalized quaternion [drot/2, 1] (xyzw).
            # _ee_delta recovers the axis-angle via 2·arctan2(|v|, w) which gives
            # back drot exactly for small angles and is consistent for larger angles.
            drot_quat = np.array([drot[0] / 2, drot[1] / 2, drot[2] / 2, 1.0], dtype=np.float64)

            action = {
                "r_x":       float(dpos[0]),
                "r_y":       float(dpos[1]),
                "r_z":       float(dpos[2]),
                "r_qx":      float(drot_quat[0]),
                "r_qy":      float(drot_quat[1]),
                "r_qz":      float(drot_quat[2]),
                "r_qw":      float(drot_quat[3]),
                "r_gripper": float(gripper_norm),
                "kp":        kp,
                "kd":        kd,
            }
            controller.send_action(action)

            # --- record -------------------------------------------------------
            t_now = time.perf_counter() - t_start
            buf["action"].append(np.concatenate([dpos, drot_quat]).astype(np.float32))
            buf["eef_ang_vel"].append(np.asarray(ee_vel[3:], dtype=np.float32))
            buf["eef_lin_vel"].append(np.asarray(ee_vel[:3], dtype=np.float32))
            buf["eef_pos"].append(np.asarray(ee_pos, dtype=np.float32))
            buf["eef_quat"].append(np.asarray(ee_quat, dtype=np.float32))
            buf["qpos"].append(np.asarray(q, dtype=np.float32))
            buf["qvel"].append(np.asarray(dq, dtype=np.float32))
            buf["t_sim"].append(np.array([t_now], dtype=np.float32))
            buf["tau_cmd"].append(np.zeros(7, dtype=np.float32))

            elapsed = time.perf_counter() - t_step
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

    return {k: np.stack(v) for k, v in buf.items() if v}


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def save_sysid_hdf5(recorded: dict[str, np.ndarray], path: str) -> None:
    """Write the recorded episode to an HDF5 file with the sim-compatible layout."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        ep = f.create_group("data").create_group("episode_0")
        for field, arr in recorded.items():
            ep.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
    logger.info("saved %d steps to %s", next(iter(recorded.values())).shape[0], path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a reference trajectory on the Franka and record the kinematic response."
    )
    parser.add_argument("traj_file", help="Input HDF5 trajectory file to replay")
    parser.add_argument("--fps", type=float, default=20.0, help="Control rate in Hz")
    parser.add_argument(
        "--kp", type=float, default=_DEFAULT_KP,
        help="EE PD kp in [-1, 1]; maps to gain 10**kp (default 0 → gain 1.0)",
    )
    parser.add_argument(
        "--kd", type=float, default=_DEFAULT_KD,
        help="EE PD kd in [-1, 1] (default 0)",
    )
    parser.add_argument("--gripper-norm", type=float, default=1.0,
                        help="Gripper openness [0, 1] held constant during replay")
    parser.add_argument("--home-max-time-s", type=float, default=5.0,
                        help="Maximum seconds allowed for the homing move")
    parser.add_argument("--home-tol-rad", type=float, default=0.005,
                        help="Joint-angle convergence tolerance (rad) for homing")
    parser.add_argument("--home-tol-m", type=float, default=0.005,
                        help="EE position convergence tolerance (m) for homing")
    parser.add_argument("--viz-out", default=None,
                        help="Path for the comparison HTML; defaults to <output>.html")
    parser.add_argument("--viz-stride", type=int, default=1,
                        help="Animate every Nth step in the visualization (use 2-4 for long episodes)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.DEBUG)

    # Connect robot
    logger.info("connecting to robot...")
    controller = start_controller()
    logger.info("robot connected")

    try:
        for i in range(0, parse_num_traj(args.traj_file)):
            # Load reference trajectory
            logger.info("loading trajectory from %s", args.traj_file)
            name, traj = parse_traj(args.traj_file, i)
            output = "record_" + name
            n_steps = len(traj["eef_pos"])
            logger.info(f"recording trajectory: {name} with {n_steps} steps, keys: {sorted(traj.keys())}")

            # Home to first trajectory position (uses qpos from trajectory if available)
            if "qpos" in traj:
                home_q = traj["qpos"][0].astype(np.float64)
                logger.info("homing to first trajectory qpos: %s", np.round(home_q, 3))
            else:
                logger.warning("trajectory has no 'qpos'; skipping pose-specific homing")
                home_q = None

            if home_q is not None:
                converged = controller.home(
                    home_q_left=None,
                    home_q_right=home_q,
                    gripper_norm=args.gripper_norm,
                    max_time_s=args.home_max_time_s,
                    tol_rad=args.home_tol_rad,
                    tol_pos_m=args.home_tol_m,
                )
                if not converged:
                    logger.warning("homing did not converge; proceeding anyway")

            # Replay and record
            logger.info(
                "replaying %d steps at %.1f Hz (kp=%.2f → gain=%.2f) — press right-arrow to stop early",
                n_steps, args.fps, args.kp, 10.0 ** args.kp,
            )
            recorded = _run_episode(
                controller=controller,
                traj=traj,
                fps=args.fps,
                kp=args.kp,
                kd=args.kd,
                gripper_norm=args.gripper_norm,
            )

            if not recorded:
                logger.warning("no steps recorded; exiting")
                return

            # Save HDF5
            save_sysid_hdf5(recorded, f"sysid/outputs/{i}_" + output + ".hdf5")

            # Visualization
            viz_out = args.viz_out or str(Path(f"sysid/outputs/{i}_" + output).with_suffix(".html"))
            save_comparison_html(traj, recorded, viz_out, fps=args.fps, frame_stride=args.viz_stride)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
