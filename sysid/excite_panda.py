#!/usr/bin/env python3
"""Record panda_control's v3/v4 excitation trajectories in BOTH EE action spaces.

    python sysid/excite_panda.py --yes
    python sysid/excite_panda.py --dry-run --tag smoke
    python sysid/excite_panda.py --selftest            # port fidelity, no hardware
    python sysid/excite_panda.py --verify ~/sysid/outputs/<run>/

Drives each trajectory ONCE and writes two HDF5 files that describe the same run
in two action spaces:

    <run>/ee_delta/excitation.hdf5   action = [dpos_m(3), dquat_xyzw(4)]
    <run>/ee_pose/excitation.hdf5    action = [goal_pos(3), goal_quat(4)]

They sit in SEPARATE directories on purpose. fit_sim_controller's `load_real_dir`
globs `*.hdf5` non-recursively and applies whichever `action_format` the CALLER
passes -- it does not read the `action_format` attr stamped here. So one directory
holding both would double-count every episode and, worse, silently read the
absolute file's ~0.4 m goal positions as deltas: normalized actions came out at
9.9 where the policy bound is 1.0. A directory each makes that unrepresentable.
Point `load_real_dir` at `<run>/ee_delta` or `<run>/ee_pose`, never at `<run>`
(which holds the per-episode flush files, in the usual `*_record_*.hdf5` form).

Why one run is enough
---------------------
Both control modes ship the same absolute goal over RPyC (franka_process
send_osc_goal: pos(3) quat(4) kp(6) kd(6) nullspace_q(7)), so equal goals mean
equal torques -- there is no separate EE_POS torque path. With
`(p, q)` the anchor send_action reads and tf/rf the tuning fudges:

    EE_DELTA   goal = (p + tf*clip(a_p),  Rotation.from_rotvec(rf*clip(v)) * q)
    EE_POS     goal = (P*, Q*)                              (no envelope, no latch)

so commanding `a_p = (P* - p)/tf` and `a_q = quat(log(Q* q^-1) / rf)` makes the
delta path pursue exactly `(P*, Q*)`. Both files therefore carry the identical
`eef_goal_pos`/`eef_goal_quat`, and either replays to the same goals. `--verify`
checks that offline through the robot's own OSCGoalBuilder.

That equivalence holds only while the delta stays inside the +/-0.05 m / +/-0.5 rad
envelope, which is per axis. Our OSC runs default_kp 125 against panda_control's
500, so at their amplitudes the tracking error is several times the envelope; the
probe pass below scales the generators' amplitudes until it fits.

The anchor
----------
`_kinematic_state` reuses get_observation's snapshot only if `_cached_kin_ts` is
within observation.kin_cache_max_age_s, and that timestamp is written ONLY by
get_observation. sysid.py and tune.py set `_cached_kin_state` without it, so the
cache never hits and send_action re-reads -- meaning the pose they log is not the
anchor the controller used. This script sets both and asserts `_kin_cache_stale`
never increments, because the dual labelling is only exact if the anchor is known.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import franka_config as fc  # noqa: E402
import panda_traj  # noqa: E402
from sysid import (  # noqa: E402
    _aa_to_quat,
    _control_stack_hashes,
    _METADATA_CONSTANT_NAMES,
    _METADATA_MODULE_LABELS,
    _quat_mul,
    _rotvec_between,
    _sha256,
    _write_run_json,
    save_sysid_hdf5,
)

logger = logging.getLogger("excite_panda")

_PROFILE = "single_arm_franka"
_ARM_KEY = fc.profile(_PROFILE).depth_center_arm
_SPEC_DIR = Path(__file__).resolve().parent / "specs"

#: Fields buffered per step. Mirrors sysid.py's set plus eef_goal_lin_vel (the
#: generators' analytic dx_des -- the OSC goal interface has no feedforward
#: velocity channel, so it is a reference diagnostic, not a command).
_FIELDS = (
    "action", "eef_goal_pos", "eef_goal_quat", "eef_goal_lin_vel",
    "eef_ang_vel", "eef_lin_vel", "eef_pos", "eef_quat",
    "fault_count", "qpos", "qvel", "t_sim",
    "tau_cmd", "tau_measured", "tau_ext",
)

_ACTION_COLUMNS = {
    "delta": ["dpos_x", "dpos_y", "dpos_z", "dquat_x", "dquat_y", "dquat_z", "dquat_w"],
    "ee_pose": ["goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"],
}
_ACTION_FORMAT = {"delta": "metric_quat", "ee_pose": "absolute_pose_quat"}
_ACTION_SPACE = {"delta": "EE_DELTA", "ee_pose": "EE_POS"}


# ---------------------------------------------------------------------------
# Lazy robot stack (importable without hardware; not without the packages)
# ---------------------------------------------------------------------------

_STACK: SimpleNamespace | None = None


def _stack() -> SimpleNamespace:
    """The robot-side modules. No hardware needed to import them, so --dry-run
    and --verify use the real goal builder rather than a copy of it."""
    global _STACK
    if _STACK is None:
        from lerobot_robot_bimanual_franka import (  # noqa: PLC0415
            ControlMode, SingleArmFranka, SingleArmFrankaConfig,
        )
        from lerobot_robot_bimanual_franka import (  # noqa: PLC0415
            bimanual_franka as bf, franka_process as fp, homing as hm,
            osc_torque_controller as osc, safety as sf,
        )
        from lerobot_robot_bimanual_franka.ee_goals import (  # noqa: PLC0415
            OSCGoalBuilder, delta_rotvec,
        )
        _STACK = SimpleNamespace(
            ControlMode=ControlMode, SingleArmFranka=SingleArmFranka,
            SingleArmFrankaConfig=SingleArmFrankaConfig,
            OSCGoalBuilder=OSCGoalBuilder, delta_rotvec=delta_rotvec,
            ActionSafetyScreen=sf.ActionSafetyScreen,
            bf=bf, osc=osc, hm=hm, safety=sf, fp=fp,
        )
    return _STACK


def _rig_config(control_mode):
    """Connection fields and every tuning knob come from config/ via
    default_factory; cameras off because sysid consumes kinematics only."""
    s = _stack()
    return s.SingleArmFrankaConfig(control_mode=control_mode, cameras={},
                                   depth=False, depth_cam={})


def _shadow(cfg):
    """A second OSCGoalBuilder + ActionSafetyScreen, fed the same anchor as the
    robot's own, so the goal we log is the goal the controller pursues rather
    than the reference we asked for. Same construction as replay_dataset.py."""
    s = _stack()
    goals = s.OSCGoalBuilder(
        translation_fudge=cfg.ee_translation_fudge,
        rotation_fudge=cfg.ee_rotation_fudge,
        use_noise=cfg.use_noise,
        noise_pos_scale=cfg.noise_pos_scale,
        noise_rot_scale=cfg.noise_rot_scale,
    )
    safety = s.ActionSafetyScreen(
        {_ARM_KEY: cfg.base_in_world(_ARM_KEY)},
        {_ARM_KEY: fc.ee_sphere(cfg.arm_name(_ARM_KEY))},
    )
    return goals, safety


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def _load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    for key in ("kind", "duration_s", "params"):
        if key not in spec:
            raise KeyError(f"{path} missing key: {key}")
    if spec["kind"] not in panda_traj.AMP_KEYS:
        raise ValueError(f"{path}: unknown kind {spec['kind']!r}")
    # Several specs share a kind (step5b/step5c/v3 are all v3_step5d), so the
    # episode name is its own field -- it is what val_regex matches on.
    spec.setdefault("name", spec["kind"])
    return spec


def _scaled_params(spec: dict, s_pos: float, s_rot: float) -> dict:
    """Apply the amplitude scales to the generator's own amp_* arguments.

    Scaling the amplitudes rather than the resulting offsets preserves the
    designed inter-axis ratios (v4's z = 1.5x xy, v3's per-band assignment),
    which are the point of the spectral design.
    """
    pos_keys, rot_keys = panda_traj.AMP_KEYS[spec["kind"]]
    params = dict(spec["params"])
    for k in pos_keys:
        params[k] = float(params[k]) * s_pos
    for k in rot_keys:
        params[k] = float(params[k]) * s_rot
    return params


def _reference(spec: dict, fps: float, s_pos: float, s_rot: float,
               x_anchor, q_anchor, ramp_out_s: float) -> dict:
    # +1 so t_s[-1] is exactly duration_s, as upstream's `duration*hz + 1` grid is.
    # The builders read total_T off t_s[-1], so dropping the endpoint would compress
    # the chirp sweep and the ramp into one sample less than the intended window.
    n = int(round(float(spec["duration_s"]) * fps)) + 1
    t_s = np.arange(n, dtype=np.float64) / float(fps)
    return panda_traj.build(spec["kind"], t_s, x_anchor, q_anchor,
                            _scaled_params(spec, s_pos, s_rot), ramp_out_s)


# ---------------------------------------------------------------------------
# Keyboard early stop (same pattern as sysid.py)
# ---------------------------------------------------------------------------

def _key_pressed() -> bool:
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
# Homing
# ---------------------------------------------------------------------------

#: Default verified-homing tolerance. THE TOLERANCE HAS A FLOOR SET BY FRICTION,
#: NOT BY PATIENCE: home() holds at torque.joint_impedance.kp, so friction leaves
#: a steady-state error of coulomb_nm/kp -- 0.0057 rad on joint 2 with the shipped
#: values, and more in practice because breakaway is pose- and direction-dependent.
#: Asking for less than that never converges; it just burns `attempts * max_time_s`
#: and skips the episode. Same value and same reason as delta_sweep.py's reset().
#:
#: Loose homing costs this script much less than it costs delta_sweep/tune: those
#: compare against a sim reference at a specific q, whereas the excitation is
#: anchored on the MEASURED pose at the first tick, so a small offset shifts where
#: the trajectory happens rather than corrupting it. What matters here is that the
#: start pose is recorded (it is: qpos, x_anchor, q_anchor) and repeatable between
#: the probe pass and the record pass.
HOME_TOL_RAD = 0.02


def _friction_floor_rad() -> float:
    """Worst-joint steady-state hold error, coulomb_nm / joint_impedance.kp."""
    return float(np.max(np.asarray(fc.control("torque.friction.coulomb_nm"))
                        / np.asarray(fc.control("torque.joint_impedance.kp"))))


def _home_verified(robot, q_target, tol: float = HOME_TOL_RAD,
                   attempts: int = 2) -> tuple[bool, float]:
    """Drive to q_target and confirm it, returning (converged, achieved error).

    home() returning True is not enough: a wrong start pose measures a different
    inertia, and that spread has swamped the signal before. See HOME_TOL_RAD for
    why the check cannot be tightened past the friction floor.
    """
    q_target = np.asarray(q_target, dtype=np.float64)
    err = float("inf")
    for _ in range(attempts):
        robot.home(home_q_left=None, home_q_right=q_target, max_time_s=20.0,
                   tol_rad=tol, fps=30)
        q = np.asarray(
            robot.robot_manager.current_kinematic_state_batch([_ARM_KEY])[_ARM_KEY][0],
            dtype=np.float64)
        err = float(np.max(np.abs(q - q_target)))
        if err < tol:
            return True, err
    return False, err


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def run_episode(robot, cfg, spec: dict, *, fps: float, s_pos: float, s_rot: float,
                drive_mode: str, kp: float, kd: float, gripper_norm: float,
                ramp_out_s: float, abort_m: float, abort_rad: float,
                flush_path: Path | None = None, flush_attrs: dict | None = None,
                flush_every: int = 100, label: str = "") -> tuple[dict, dict]:
    """Drive one trajectory, dual-logging the delta and the pursued goal.

    Returns (recorded, stats). `recorded["action"]` is the delta as sent; the
    absolute-action file is built from eef_goal_pos/eef_goal_quat, which is the
    goal the shadow builder says the controller actually pursued.
    """
    s = _stack()
    tf = float(cfg.ee_translation_fudge)
    rf = float(cfg.ee_rotation_fudge)
    pos_max = float(s.osc.DELTA_POS_MAX)
    rot_max = float(s.osc.DELTA_ROT_MAX)

    goals, safety = _shadow(cfg)
    buf: dict[str, list] = {k: [] for k in _FIELDS}

    # Anchor on the settled post-home pose, the same read the first step uses.
    kin = robot.robot_manager.current_kinematic_state_batch([_ARM_KEY])[_ARM_KEY]
    x_anchor = np.asarray(kin[3], dtype=np.float64).copy()
    q_anchor = np.asarray(kin[4], dtype=np.float64).copy()
    goals.reset(_ARM_KEY, q_anchor)          # mirrors home()'s own reset_goal()

    ref = _reference(spec, fps, s_pos, s_rot, x_anchor, q_anchor, ramp_out_s)
    goal_pos_ref, goal_quat_ref = ref["goal_pos"], ref["goal_quat"]
    n_steps = len(goal_pos_ref)

    dt = 1.0 / float(fps)
    t_start = time.perf_counter()
    stop_reason = None
    peak_dpos = np.zeros(3)
    peak_drot = np.zeros(3)
    clip_steps = zero_rot_steps = 0
    stale0 = int(getattr(robot, "_kin_cache_stale", 0))
    fault0 = int(robot.robot_manager.recovery_counts().get(_ARM_KEY, 0))

    interactive = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if interactive else None
    if interactive:
        tty.setcbreak(sys.stdin)
    try:
        for step in range(n_steps):
            t_step = time.perf_counter()

            if interactive and _key_pressed():
                key = _read_key()
                if key == "ctrl_c":
                    raise KeyboardInterrupt
                if key == "right":
                    logger.info("early stop requested")
                    stop_reason = "early_stop"
                    break

            # The anchor: set the timestamp too, or send_action re-reads and the
            # pose we log stops being the one the goal was built on.
            kin_all = robot.robot_manager.current_kinematic_state_batch([_ARM_KEY])
            robot._cached_kin_state = kin_all
            robot._cached_kin_ts = time.perf_counter()
            q, dq, _jac, ee_pos, ee_quat, ee_vel = kin_all[_ARM_KEY]
            ee_pos64 = np.asarray(ee_pos, dtype=np.float64)
            ee_quat64 = np.asarray(ee_quat, dtype=np.float64)

            # Invert the reference onto the measured pose.
            pos_err = goal_pos_ref[step] - ee_pos64
            rot_err = _rotvec_between(goal_quat_ref[step], ee_quat64)
            if float(np.linalg.norm(pos_err)) > abort_m:
                logger.error("tracking error %.3f m exceeds --track-abort-m %.3f at step %d",
                             float(np.linalg.norm(pos_err)), abort_m, step)
                stop_reason = "track_abort_pos"
                break
            if float(np.linalg.norm(rot_err)) > abort_rad:
                logger.error("orientation error %.3f rad exceeds --track-abort-rad %.3f at step %d",
                             float(np.linalg.norm(rot_err)), abort_rad, step)
                stop_reason = "track_abort_rot"
                break

            dpos = pos_err / tf
            drot = rot_err / rf
            drot_quat = _aa_to_quat(drot)

            peak_dpos = np.maximum(peak_dpos, np.abs(dpos))
            peak_drot = np.maximum(peak_drot, np.abs(drot))
            if np.any(np.abs(dpos) > pos_max) or np.any(np.abs(drot) > rot_max):
                clip_steps += 1
            if not np.any(drot):
                # goal_ori stays latched: the goal is not the reference. Modelled
                # by the shadow builder, so the log stays correct either way.
                zero_rot_steps += 1

            # The goal the controller will pursue, post-clip / fudge / screen.
            if drive_mode == "delta":
                goal = goals.from_delta(_ARM_KEY, dpos, drot, ee_pos64, ee_quat64)
            else:
                goal = goals.absolute(goal_pos_ref[step], goal_quat_ref[step])
            goal_pos, goal_quat = safety.shape_goal({_ARM_KEY: goal})[_ARM_KEY]

            if drive_mode == "delta":
                cmd_pos, cmd_quat = dpos, drot_quat
            else:
                cmd_pos, cmd_quat = goal_pos_ref[step], goal_quat_ref[step]
            robot.send_action({
                f"{_ARM_KEY}_x": float(cmd_pos[0]),
                f"{_ARM_KEY}_y": float(cmd_pos[1]),
                f"{_ARM_KEY}_z": float(cmd_pos[2]),
                f"{_ARM_KEY}_qx": float(cmd_quat[0]),
                f"{_ARM_KEY}_qy": float(cmd_quat[1]),
                f"{_ARM_KEY}_qz": float(cmd_quat[2]),
                f"{_ARM_KEY}_qw": float(cmd_quat[3]),
                f"{_ARM_KEY}_gripper": float(gripper_norm),
                "kp": kp,
                "kd": kd,
            })

            t_now = time.perf_counter() - t_start
            buf["action"].append(np.concatenate([dpos, drot_quat]).astype(np.float32))
            buf["eef_goal_pos"].append(np.asarray(goal_pos, dtype=np.float32))
            gq = np.asarray(goal_quat, dtype=np.float64)
            gq = gq / max(float(np.linalg.norm(gq)), 1e-12)
            buf["eef_goal_quat"].append(gq.astype(np.float32))
            buf["eef_goal_lin_vel"].append(ref["goal_lin_vel"][step].astype(np.float32))
            buf["fault_count"].append(
                np.int32(robot.robot_manager.recovery_counts().get(_ARM_KEY, 0)))
            buf["eef_ang_vel"].append(np.asarray(ee_vel[3:], dtype=np.float32))
            buf["eef_lin_vel"].append(np.asarray(ee_vel[:3], dtype=np.float32))
            buf["eef_pos"].append(np.asarray(ee_pos, dtype=np.float32))
            buf["eef_quat"].append(np.asarray(ee_quat, dtype=np.float32))
            buf["qpos"].append(np.asarray(q, dtype=np.float32))
            buf["qvel"].append(np.asarray(dq, dtype=np.float32))
            buf["t_sim"].append(np.array([t_now], dtype=np.float32))
            tau_cmd, tau_meas, tau_ext = robot.robot_manager.torque_snapshot(_ARM_KEY)
            buf["tau_cmd"].append(np.asarray(tau_cmd, dtype=np.float32))
            buf["tau_measured"].append(np.asarray(tau_meas, dtype=np.float32))
            buf["tau_ext"].append(np.asarray(tau_ext, dtype=np.float32))

            if flush_path is not None and (step + 1) % flush_every == 0:
                save_sysid_hdf5({k: np.stack(v) for k, v in buf.items() if v},
                                str(flush_path), attrs=flush_attrs, quiet=True)

            elapsed = time.perf_counter() - t_step
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

    recorded = {k: np.stack(v) for k, v in buf.items() if v}
    stale = int(getattr(robot, "_kin_cache_stale", 0)) - stale0
    faults = int(robot.robot_manager.recovery_counts().get(_ARM_KEY, 0)) - fault0
    stats = {
        "label": label or spec["kind"],
        "steps": len(recorded.get("t_sim", [])),
        "expected_steps": n_steps,
        "stop_reason": stop_reason,
        "peak_delta_pos": peak_dpos.tolist(),
        "peak_delta_rotvec": peak_drot.tolist(),
        "clip_steps": clip_steps,
        "zero_rot_steps": zero_rot_steps,
        "kin_cache_misses": stale,
        "faults": faults,
        "x_anchor": x_anchor.tolist(),
        "q_anchor": q_anchor.tolist(),
        "peak_rates": ref["peak_rates"],
        "peak_ori_offset_rad": ref["peak_ori_offset_rad"],
        "amp_scale_pos": s_pos,
        "amp_scale_rot": s_rot,
        "params": _scaled_params(spec, s_pos, s_rot),
        "exact_dual_label": bool(stale == 0 and clip_steps == 0),
    }
    if recorded:
        stats["end_offset_m"] = float(np.linalg.norm(
            recorded["eef_pos"][-1].astype(np.float64) - x_anchor))
    return recorded, stats


# ---------------------------------------------------------------------------
# Amplitude probe
# ---------------------------------------------------------------------------

def derive_scales(stats: dict, probe_scale: float, pos_margin: float,
                  rot_margin: float) -> tuple[float, float]:
    """Extrapolate the probe's peak delta to the amplitude that just fits.

    Linear in amplitude, which is conservative: Coulomb friction is a deadband,
    so error/amplitude is HIGHER at small amplitude (SYSID.md section 2) and the
    derived scale under-shoots rather than over-shoots.
    """
    peak_pos = float(np.max(stats["peak_delta_pos"]))
    peak_rot = float(np.max(stats["peak_delta_rotvec"]))
    s_pos = 1.0 if peak_pos < 1e-9 else probe_scale * pos_margin / peak_pos
    s_rot = 1.0 if peak_rot < 1e-9 else probe_scale * rot_margin / peak_rot
    return float(np.clip(s_pos, 0.0, 1.0)), float(np.clip(s_rot, 0.0, 1.0))


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def _pose_action(recorded: dict) -> np.ndarray:
    """The EE_POS action that commands the same goal: the goal itself."""
    return np.concatenate([recorded["eef_goal_pos"], recorded["eef_goal_quat"]],
                          axis=1).astype(np.float32)


def save_dual_hdf5(episodes: list[tuple[str, dict, dict]], path: Path,
                   space: str) -> None:
    """Write one multi-episode file in the layout fit_sim_controller reads
    (`data/<episode>/<field>`; it hardcodes the group name `data`).

    `space` selects which array lands in `action`; every other field is the same
    recorded data in both files. save_sim_format_hdf5 cannot be reused -- it
    requires `action_norm` and is replay-mode only.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        grp = f.create_group("data")
        for name, recorded, attrs in episodes:
            ep = grp.create_group(name)
            for field, arr in recorded.items():
                if field == "action":
                    continue
                ep.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
            action = recorded["action"] if space == "delta" else _pose_action(recorded)
            ep.create_dataset("action", data=action, compression="gzip", compression_opts=4)
            ep.attrs["num_samples"] = int(action.shape[0])
            ep.attrs["action_format"] = _ACTION_FORMAT[space]
            ep.attrs["action_space"] = _ACTION_SPACE[space]
            ep.attrs["action_columns"] = _ACTION_COLUMNS[space]
            for key, val in (attrs or {}).items():
                if val is not None:
                    ep.attrs[key] = val
    os.replace(tmp, path)
    logger.info("saved %d episode(s) to %s", len(episodes), path)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

#: Round-trip tolerance. The recorded arrays are float32 (the sysid convention),
#: so a float64 recomputation cannot agree closer than a float32 half-ulp -- about
#: 6e-8 at unit magnitude. Anything tighter fails on storage, not on the math.
_VERIFY_TOL = 1e-6


def verify_run(run_dir: Path, tol: float = _VERIFY_TOL) -> bool:
    """Replay each file's `action` through the robot's own goal builder and
    check it reproduces the logged goal, and that both files agree.

    This is the claim "same torques, different action space" made checkable.
    """
    s = _stack()
    delta_path = run_dir / "ee_delta" / "excitation.hdf5"
    pose_path = run_dir / "ee_pose" / "excitation.hdf5"
    for p in (delta_path, pose_path):
        if not p.is_file():
            logger.error("missing %s", p)
            return False

    cfg = _rig_config(s.ControlMode.EE_DELTA)
    ok = True
    with h5py.File(delta_path, "r") as fd, h5py.File(pose_path, "r") as fp_:
        names = sorted(fd["data"].keys())
        if names != sorted(fp_["data"].keys()):
            logger.error("episode sets differ between the two files")
            return False
        for name in names:
            d, p = fd["data"][name], fp_["data"][name]
            gp = np.asarray(d["eef_goal_pos"], dtype=np.float64)
            gq = np.asarray(d["eef_goal_quat"], dtype=np.float64)
            ee_pos = np.asarray(d["eef_pos"], dtype=np.float64)
            ee_quat = np.asarray(d["eef_quat"], dtype=np.float64)

            # The two files must describe the same goals.
            for field in ("eef_goal_pos", "eef_goal_quat"):
                if not np.array_equal(np.asarray(d[field]), np.asarray(p[field])):
                    logger.error("%s: %s differs between files", name, field)
                    ok = False

            # Each file's action must reproduce those goals.
            for space, grp in (("delta", d), ("ee_pose", p)):
                goals, safety = _shadow(cfg)
                goals.reset(_ARM_KEY, ee_quat[0])
                a = np.asarray(grp["action"], dtype=np.float64)
                err_p = err_q = 0.0
                for t in range(len(a)):
                    if space == "delta":
                        g = goals.from_delta(_ARM_KEY, a[t, 0:3],
                                             s.delta_rotvec(a[t, 3:7]),
                                             ee_pos[t], ee_quat[t])
                    else:
                        g = goals.absolute(a[t, 0:3], a[t, 3:7])
                    rp, rq = safety.shape_goal({_ARM_KEY: g})[_ARM_KEY]
                    err_p = max(err_p, float(np.max(np.abs(rp - gp[t]))))
                    # Quaternion double cover: compare on the closer hemisphere.
                    err_q = max(err_q, float(min(np.max(np.abs(rq - gq[t])),
                                                 np.max(np.abs(rq + gq[t])))))
                status = "OK" if (err_p <= tol and err_q <= tol) else "FAIL"
                if status == "FAIL":
                    ok = False
                logger.info("%-28s %-8s pos %.2e  quat %.2e  %s",
                            name, space, err_p, err_q, status)
    return ok


# ---------------------------------------------------------------------------
# Self-test (port fidelity, no hardware, no robot packages)
# ---------------------------------------------------------------------------

def selftest() -> bool:
    """Check the ported generators against the assertions upstream makes, on
    upstream's own 1000 Hz grid."""
    x_anchor = np.array([0.4, 0.0, 0.4])
    q_anchor = np.array([0.0, 1.0, 0.0, 0.0])
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        ok = ok and bool(cond)

    for name, path in (("v4", "panda_v4.json"), ("v3", "panda_v3.json")):
        spec = _load_spec(_SPEC_DIR / path)
        t_s = np.arange(int(spec["duration_s"] * 1000) + 1) / 1000.0
        r = panda_traj.build(spec["kind"], t_s, x_anchor, q_anchor, spec["params"])
        gp, gq = r["goal_pos"], r["goal_quat"]
        print(f"{name} ({spec['kind']}): {r['peak_rates']}")
        check(float(np.linalg.norm(gp[0] - x_anchor)) < 1e-6,
              f"{name}: |x_des(0) - x_anchor| < 1e-6")
        check(abs(float(np.dot(gq[0], q_anchor))) > 1.0 - 1e-6,
              f"{name}: |q_des(0) . q_anchor| ~ 1")
        check(np.all(np.abs(np.linalg.norm(gq, axis=1) - 1.0) < 1e-9),
              f"{name}: goal quaternions are unit norm")

    # The ramp-out asymmetry, asserted rather than assumed.
    v4 = _load_spec(_SPEC_DIR / "panda_v4.json")
    t4 = np.arange(int(v4["duration_s"] * 1000) + 1) / 1000.0
    r4 = panda_traj.build(v4["kind"], t4, x_anchor, q_anchor, v4["params"])
    check(float(np.linalg.norm(r4["goal_pos"][-1] - x_anchor)) < 1e-9,
          "v4 ramps out: returns to the anchor at t=T")

    v3 = _load_spec(_SPEC_DIR / "panda_v3.json")
    t3 = np.arange(int(v3["duration_s"] * 1000) + 1) / 1000.0
    r3 = panda_traj.build(v3["kind"], t3, x_anchor, q_anchor, v3["params"])
    end3 = float(np.linalg.norm(r3["goal_pos"][-1] - x_anchor))
    check(end3 > 1e-3, f"v3 does NOT ramp out: ends {end3 * 1000:.1f} mm off the anchor")

    r3t = panda_traj.build(v3["kind"], t3, x_anchor, q_anchor, v3["params"], ramp_out_s=2.0)
    check(float(np.linalg.norm(r3t["goal_pos"][-1] - x_anchor)) < 1e-9,
          "v3 with --ramp-out-s returns to the anchor")

    # v3's stated amplitude is the low band's; the sum peaks at 1.2x.
    peak_x = float(np.max(np.abs(r3["goal_pos"][:, 0] - x_anchor[0])))
    ratio = peak_x / v3["params"]["amp_x"]
    check(1.1 < ratio <= 1.2 + 1e-6,
          f"v3 x peak is {ratio:.3f}x the stated amplitude (1 + high_band_ratio)")

    # Amplitude scaling is linear in the amp_* arguments.
    half = panda_traj.build(v3["kind"], t3, x_anchor, q_anchor,
                            _scaled_params(v3, 0.5, 0.5))
    check(np.allclose(half["goal_pos"] - x_anchor,
                      0.5 * (r3["goal_pos"] - x_anchor), atol=1e-12),
          "position offsets scale linearly with the amplitude scale")
    return ok


# ---------------------------------------------------------------------------
# Dry-run mock
# ---------------------------------------------------------------------------

class _Mock:
    """Kinematics-only stand-in: first-order tracking toward each tick's goal,
    with both control modes and the cache fields the real class carries."""

    def __init__(self, trans_fudge: float, rot_fudge: float, rate: float = 0.35):
        self._tf, self._rf, self._rate = trans_fudge, rot_fudge, rate
        self._q = np.zeros(7)
        self._pos = np.array([0.4, 0.0, 0.4])
        self._quat = np.array([0.0, 1.0, 0.0, 0.0])
        self._cached_kin_state = None
        self._cached_kin_ts = 0.0
        self._kin_cache_stale = 0
        self.control_mode = None
        outer = self

        class _RM:
            def current_kinematic_state_batch(self, arms):
                snap = (outer._q.copy(), np.zeros(7), np.zeros((6, 7)),
                        outer._pos.copy(), outer._quat.copy(), np.zeros(6))
                return {a: snap for a in arms}

            def recovery_counts(self):
                return {_ARM_KEY: 0}

            def torque_snapshot(self, name):
                return (np.zeros(7), np.zeros(7), np.zeros(7))

        self.robot_manager = _RM()

    def home(self, home_q_left=None, home_q_right=None, **kwargs):
        if home_q_right is not None:
            self._q = np.asarray(home_q_right, dtype=np.float64).copy()
        return True

    def send_action(self, action: dict):
        p = np.array([action[f"{_ARM_KEY}_{a}"] for a in ("x", "y", "z")])
        qd = np.array([action[f"{_ARM_KEY}_q{a}"] for a in ("x", "y", "z", "w")])
        qd = qd / max(float(np.linalg.norm(qd)), 1e-12)
        pos_max, rot_max = 0.05, 0.5
        if self.control_mode == "ee_pose":
            goal_pos, goal_quat = p, qd
        else:
            dpos = np.clip(p, -pos_max, pos_max) * self._tf
            drot = np.clip(_rotvec_between(qd, np.array([0.0, 0.0, 0.0, 1.0])),
                           -rot_max, rot_max) * self._rf
            goal_pos = self._pos + dpos
            goal_quat = _quat_mul(_aa_to_quat(drot), self._quat)
        self._pos = self._pos + self._rate * (goal_pos - self._pos)
        self._quat = _quat_mul(_aa_to_quat(self._rate * _rotvec_between(goal_quat, self._quat)),
                               self._quat)
        self._quat /= max(float(np.linalg.norm(self._quat)), 1e-12)
        return action

    def disconnect(self):
        pass


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def _run_metadata(args, specs, stack) -> dict:
    constants = {
        _METADATA_MODULE_LABELS[key]: {
            name: getattr(getattr(stack, key, None), name, None) for name in names
        }
        for key, names in _METADATA_CONSTANT_NAMES.items()
    }
    constants["control.yaml tuning"] = dict(fc.control("tuning"))
    return {
        "status": "running",
        "mode": "excite_panda",
        "quat_encoding": "exact",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "args": vars(args),
        "specs": {s["name"]: {"path": str(p), "sha256": _sha256(str(p)), "spec": s}
                  for p, s in specs},
        "episodes_completed": [],
        "constants": constants,
        "config": fc.all_sections(),
        "config_dir": str(fc.config_dir()),
        "control_stack_sha256": _control_stack_hashes(stack),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record panda_control's v3/v4 excitation trajectories in both EE action spaces.")
    p.add_argument("--specs", nargs="*", default=["panda_v3.json", "panda_v4.json"],
                   help="Spec files (bare names resolve under sysid/specs/)")
    p.add_argument("--fps", type=float, default=float(fc.control_fps()))
    p.add_argument("--drive-mode", choices=("delta", "ee_pose"), default="delta",
                   help="Which action space actually drives the arm. delta is the "
                        "path that can clip, so driving it makes the pose file a "
                        "faithful derivation rather than a possibly-unreachable one.")
    p.add_argument("--kp", type=float, default=0.0, help="OSC kp action entry")
    p.add_argument("--kd", type=float, default=0.0, help="OSC damping-ratio action entry")
    p.add_argument("--gripper-norm", type=float,
                   default=float(fc.control("homing.gripper_norm")))
    p.add_argument("--home-tol-rad", type=float, default=HOME_TOL_RAD,
                   help="Verified-homing tolerance. Floored by friction at "
                        "coulomb_nm/joint_impedance.kp (~0.006 rad as shipped); "
                        "below that it cannot converge, it can only time out.")
    p.add_argument("--ramp-out-s", type=float, default=0.0,
                   help="Half-cosine tail returning the reference to the anchor. "
                        "0 = upstream behaviour; v3 then ends mid-swing.")
    # Amplitude scaling
    p.add_argument("--probe-scale", type=float, default=0.25,
                   help="Amplitude scale for the probe pass")
    p.add_argument("--pos-margin", type=float, default=0.040,
                   help="Target peak |delta_pos| per axis, against the 0.05 m clip")
    p.add_argument("--rot-margin", type=float, default=0.40,
                   help="Target peak |delta_rotvec| per component, against the 0.5 rad clip")
    p.add_argument("--scale-pos", type=float, default=None, help="Pin the translation scale, skip the probe")
    p.add_argument("--scale-rot", type=float, default=None, help="Pin the rotation scale, skip the probe")
    # Aborts
    # Both aborts are a property of the TRAJECTORY, so they live in the spec and
    # these only override. panda_control runs step5d at 0.05 m / 0.30 rad and the
    # chirp at 0.15 m / 0.80 rad -- the chirp reference alone reaches ~0.61 rad of
    # rotation offset, so 0.30 aborts it mid-run however well the arm is tracking.
    p.add_argument("--track-abort-m", type=float, default=None,
                   help="Override the spec's position abort")
    p.add_argument("--track-abort-rad", type=float, default=None,
                   help="Override the spec's orientation abort")
    # Output / modes
    p.add_argument("--out-root", type=str, default="~/sysid/outputs")
    p.add_argument("--tag", type=str, default="panda_excite")
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--dry-run", action="store_true", help="kinematic mock, no hardware")
    p.add_argument("--selftest", action="store_true", help="port fidelity only, then exit")
    p.add_argument("--verify", type=str, default=None, help="verify an existing run directory, then exit")
    p.add_argument("--yes", action="store_true", help="skip the workspace prompt")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    if args.selftest:
        return 0 if selftest() else 1
    if args.verify:
        return 0 if verify_run(Path(args.verify).expanduser()) else 1

    spec_paths = [(_SPEC_DIR / s if "/" not in s else Path(s).expanduser())
                  for s in args.specs]
    specs = [(p, _load_spec(p)) for p in spec_paths]

    s = _stack()
    mode = (s.ControlMode.EE_DELTA if args.drive_mode == "delta" else s.ControlMode.EE_POS)
    cfg = _rig_config(mode)
    logger.info("profile %s: key %r -> physical %s, %s at %g Hz",
                _PROFILE, _ARM_KEY, cfg.arm_name(_ARM_KEY), mode.value, args.fps)
    logger.info("controller: default_kp=%s cross_coupling_compensation=%s "
                "uncouple_pos_ori=%s",
                fc.control("torque.osc.default_kp"),
                fc.control("torque.osc.cross_coupling_compensation"),
                fc.control("torque.osc.uncouple_pos_ori"))
    logger.info("fudges: translation=%g rotation=%g; delta envelope +/-%g m, +/-%g rad",
                cfg.ee_translation_fudge, cfg.ee_rotation_fudge,
                s.osc.DELTA_POS_MAX, s.osc.DELTA_ROT_MAX)

    # Pre-flight the references at full amplitude, as upstream does.
    for path, spec in specs:
        r = panda_traj.build(spec["kind"], np.arange(int(spec["duration_s"] * 1000) + 1) / 1000.0,
                             np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), spec["params"])
        pk = r["peak_rates"]
        logger.info("%s at full amplitude: peak |dx| %.3f m/s (convention %.2f), "
                    "peak ang %.3f rad/s (convention %.2f), max|rot_offset| %.3f rad",
                    spec["kind"], pk["cart_speed_m_s"], panda_traj.CART_DX_PEAK_LIMIT_MPS,
                    pk["ang_rate_rad_s"], panda_traj.ORI_DOT_PEAK_LIMIT_RPS,
                    r["peak_ori_offset_rad"])

    n_pass = len(specs) * (1 if (args.scale_pos is not None and args.scale_rot is not None) else 2)
    total_s = sum(float(sp["duration_s"]) for _, sp in specs)
    est_min = (n_pass / max(len(specs), 1) * total_s + 25.0 * n_pass) / 60.0
    print(f"{len(specs)} trajectory(ies), {n_pass} pass(es), ~{est_min:.0f} min "
          "including homing. CLEAR THE WORKSPACE.")
    if not args.yes and input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        return 1

    run_dir = Path(args.out_root).expanduser() / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    meta = _run_metadata(args, specs, s)
    _write_run_json(run_dir, meta)
    logger.info("run directory: %s", run_dir)

    if args.dry_run:
        logger.info("dry run: kinematic mock")
        robot = _Mock(cfg.ee_translation_fudge, cfg.ee_rotation_fudge)
        robot.control_mode = args.drive_mode
    else:
        robot = s.SingleArmFranka(cfg)
        robot.connect()

    base_attrs = {
        "mode": "excite_panda",
        "drive_mode": args.drive_mode,
        "quat_encoding": "exact",
        "kp": args.kp,
        "kd": args.kd,
        "fps": args.fps,
        "gripper_norm": args.gripper_norm,
        "ee_translation_fudge_factor": float(cfg.ee_translation_fudge),
        "ee_rotation_fudge_factor": float(cfg.ee_rotation_fudge),
        "osc_base_kp": getattr(s.osc, "DEFAULT_KP", None),
        "kp_gain": getattr(s.osc, "KP_EXP_SCALE", 1.0) ** args.kp,
        "dry_run": bool(args.dry_run),
        "source": "panda_control gen_excitation_traj.py / gen_chirp_traj.py",
    }

    episodes: list[tuple[str, dict, dict]] = []
    used_names: set[str] = set()
    status = "completed"
    try:
        for i, (path, spec) in enumerate(specs):
            kind = spec["kind"]
            name = spec["name"]
            init_q = np.asarray(spec["init_qpos"], dtype=np.float64)
            common = dict(fps=args.fps, drive_mode=args.drive_mode, kp=args.kp,
                          kd=args.kd, gripper_norm=args.gripper_norm,
                          ramp_out_s=args.ramp_out_s,
                          abort_m=(args.track_abort_m if args.track_abort_m is not None
                                   else float(spec.get("track_abort_m", 0.05))),
                          abort_rad=(args.track_abort_rad if args.track_abort_rad is not None
                                     else float(spec.get("track_abort_rad",
                                                         panda_traj.ORI_TRACK_ABORT_RAD))))

            s_pos, s_rot = args.scale_pos, args.scale_rot
            if s_pos is None or s_rot is None:
                logger.info("[%s] probe pass at scale %g", name, args.probe_scale)
                homed, herr = _home_verified(robot, init_q, args.home_tol_rad)
                if not homed:
                    logger.error("[%s] homing stalled at %.4f rad (tolerance %.4f, "
                                 "friction floor ~%.4f); skipping", name, herr,
                                 args.home_tol_rad, _friction_floor_rad())
                    continue
                _, pstats = run_episode(robot, cfg, spec, s_pos=args.probe_scale,
                                        s_rot=args.probe_scale, label=f"{name}_probe",
                                        **common)
                d_pos, d_rot = derive_scales(pstats, args.probe_scale,
                                             args.pos_margin, args.rot_margin)
                logger.info("[%s] probe peak delta: pos %s m, rotvec %s rad -> "
                            "scales pos %.3f rot %.3f", name,
                            np.round(pstats["peak_delta_pos"], 4),
                            np.round(pstats["peak_delta_rotvec"], 4), d_pos, d_rot)
                s_pos = d_pos if args.scale_pos is None else args.scale_pos
                s_rot = d_rot if args.scale_rot is None else args.scale_rot

            logger.info("[%s] recording at scales pos %.3f rot %.3f", name, s_pos, s_rot)
            homed, home_err = _home_verified(robot, init_q, args.home_tol_rad)
            if not homed:
                logger.error("[%s] homing stalled at %.4f rad (tolerance %.4f, "
                             "friction floor ~%.4f); skipping", name, home_err,
                             args.home_tol_rad, _friction_floor_rad())
                continue
            logger.info("[%s] homed to %.4f rad of the target", name, home_err)
            flush = run_dir / f"{i}_record_{name}.hdf5"
            attrs = {**base_attrs, "traj_kind": kind, "init_qpos": init_q,
                     "reference_episode": name, "spec_file": str(path),
                     "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}
            recorded, stats = run_episode(robot, cfg, spec, s_pos=s_pos, s_rot=s_rot,
                                          flush_path=flush, flush_attrs=attrs,
                                          flush_every=args.flush_every, label=name,
                                          **common)
            if not recorded:
                logger.error("[%s] no steps recorded (%s)", name, stats["stop_reason"])
                continue

            ep_attrs = {**attrs,
                        "amp_scale_pos": s_pos, "amp_scale_rot": s_rot,
                        "clip_steps": stats["clip_steps"],
                        "zero_rot_steps": stats["zero_rot_steps"],
                        "kin_cache_misses": stats["kin_cache_misses"],
                        "exact_dual_label": stats["exact_dual_label"],
                        "steps": stats["steps"],
                        "expected_steps": stats["expected_steps"],
                        "stop_reason": stats["stop_reason"],
                        "peak_delta_pos": stats["peak_delta_pos"],
                        "peak_delta_rotvec": stats["peak_delta_rotvec"],
                        "peak_ori_offset_rad": stats["peak_ori_offset_rad"],
                        "home_err_rad": home_err,
                        "x_anchor": stats["x_anchor"], "q_anchor": stats["q_anchor"],
                        "resolved_params": json.dumps(stats["params"]),
                        "peak_rates": json.dumps(stats["peak_rates"])}
            save_sysid_hdf5(recorded, str(flush), attrs=ep_attrs, quiet=True)
            assert name not in used_names, f"duplicate episode name {name!r}"
            used_names.add(name)
            episodes.append((name, recorded, ep_attrs))
            meta["episodes_completed"].append(stats)
            _write_run_json(run_dir, meta)

            if stats["kin_cache_misses"]:
                logger.warning("[%s] %d anchor cache miss(es): those steps' goals were "
                               "built on a pose send_action did not use; NOT exactly "
                               "dual-labelled", kind, stats["kin_cache_misses"])
            if stats["clip_steps"]:
                logger.warning("[%s] %d/%d steps hit the delta envelope; the goal is "
                               "slew-limited there and the reference was not reached",
                               kind, stats["clip_steps"], stats["steps"])
            if stats["stop_reason"]:
                logger.warning("[%s] episode ENDED EARLY (%s) at step %d of %d -- this is "
                               "a %.0f%% fragment of the trajectory, not the trajectory",
                               kind, stats["stop_reason"], stats["steps"],
                               stats["expected_steps"],
                               100.0 * stats["steps"] / max(stats["expected_steps"], 1))
            if stats["faults"]:
                logger.warning("[%s] %d recoverable fault(s) during the run; this is "
                               "not a measurement", kind, stats["faults"])
    except KeyboardInterrupt:
        status = "aborted"
        logger.warning("interrupted")
    finally:
        robot.disconnect()

    if episodes:
        save_dual_hdf5(episodes, run_dir / "ee_delta" / "excitation.hdf5", "delta")
        save_dual_hdf5(episodes, run_dir / "ee_pose" / "excitation.hdf5", "ee_pose")
    meta["status"] = status
    _write_run_json(run_dir, meta)

    if episodes and not args.dry_run:
        logger.info("verifying the two action spaces describe the same goals...")
        if not verify_run(run_dir):
            logger.error("verification FAILED")
            return 1
    logger.info("run directory: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
