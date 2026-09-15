#!/usr/bin/env python3
"""Roll a trained B-Spline Policy out on the bimanual_franka hardware (single arm).

    # in the bspline conda env (robodiff), first:
    python baselines/bspline_bridge/policy_server.py --ckpt-path <CKPT> --port 5555

    # then, in the workspace venv:
    python -m baselines.bspline_bridge.rollout --rig single_arm_right --num-episodes 10

or both at once: ./scripts/bspline_rollout.sh --start-server --ckpt <CKPT>

That server subclasses upstream's `policy_server_bspline.py` only to add the
meta handshake -- upstream replies {} to an unknown key, so it cannot say what
the checkpoint expects, and the client needs the image shapes and the spline
degree. The prediction itself is upstream's.

It returns spline PARAMETERS rather than actions, which is the right split: this
process owns the clock, so no per-tick round trip lands between the state read
and the goal write. (Upstream's `action` response format is also broken in this
checkout -- it reaches for a `simple_mobile/yam_teleop` path the repo no longer
has; see baselines/BSPLINE_POLICY.md.)

Control mode is EE_POS: `bspline_bridge/dataset.py` trains on the absolute
commanded pose. The sampled action is 10-dim -- xyz + rot6d + gripper -- because
B-Spline's dataset loader turns our 7-dim [pos, rotvec, gripper] rows into
rotation_6d. That width is what the client checks; the server's `action_format`
string is upstream's own label for its arm and is recorded, not interpreted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import franka_config as fc  # noqa: E402

from baselines import rollout_common as rc  # noqa: E402
from baselines import run_record as rr  # noqa: E402
from baselines.bspline_bridge.spline_plan import SplinePlanner  # noqa: E402
from baselines.zmq_client import PolicyClient  # noqa: E402
from lerobot_robot_bimanual_franka import ControlMode  # noqa: E402

logger = logging.getLogger("baselines.bspline")

POLICY = "bspline"
# [pos(3), rot6d(6), gripper(1)]. The pose columns are what the stitch matches
# on; the gripper is index 9.
_ACT_DIM = 10
_GRIPPER_INDEX = 9
_POSE_DIM = 9


def decode(row: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Sampled spline row -> (pos, quat_xyzw, gripper).

    Decoded here rather than through upstream's `decode_action_vector`, whose
    formats all return YAM/X5 action dicts.
    """
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size != _ACT_DIM:
        raise ValueError(
            f"expected a {_ACT_DIM}-dim pos+rot6d+gripper action, got {row.size}. "
            "The checkpoint's action space does not match what "
            "baselines/bspline_bridge/dataset.py writes."
        )
    return row[:3], rc.rot6d_to_quat_xyzw(row[3:9]), float(row[_GRIPPER_INDEX])


def make_episode_fn(client: PolicyClient, meta: dict, args, planner_kwargs: dict,
                    shapes: dict, run_dir):
    exec_fps = float(args.exec_fps or fc.policy("baselines.exec.fast_fps"))
    o_fps = float(args.obs_fps or rc.obs_fps())
    per_obs = max(1, int(round(exec_fps / o_fps)))
    logger.info("EE_POS at %.0f Hz, observations at %.0f Hz (%d goals per obs), "
                "speed_up=%.2f origin_time_scale=%.1f",
                exec_fps, o_fps, per_obs,
                planner_kwargs["speed_up_times"], planner_kwargs["origin_time_scale"])

    def episode_fn(controller, dispatcher, dataset, ep, stopper) -> None:
        client.reset()
        planner = SplinePlanner(client, **planner_kwargs)
        dispatcher.start()
        writers: dict = {}
        video_dir = run_dir.video_dir if args.save_videos else None
        # Counts WRITTEN frames, not dispatched goals: the label is
        # `index / fps`, and one frame per observation indexed by the dispatch
        # step would burn in a clock running per_obs x fast.
        video_frames = 0
        goal: tuple[np.ndarray, np.ndarray, float] | None = None

        try:
            while True:
                verdict = stopper.check()
                if verdict is not None:
                    ep.success = verdict == "success"
                    ep.wall_time_s = stopper.elapsed()
                    break

                obs = controller.get_observation()
                m = rc.measure(obs)
                if goal is not None:
                    pos_err, rot_err = rc.check_lead(goal[0], goal[1], m)
                    ep.max_lead_m = max(ep.max_lead_m, pos_err)
                    ep.max_lead_rad = max(ep.max_lead_rad, rot_err)

                sample = planner.step(rc.bspline_obs(m, shapes))
                if sample is None and goal is None:
                    # No plan yet. Hold the homed pose rather than dispatching a
                    # zero goal, which would command the base frame origin.
                    goal = (m.pos.copy(), m.quat_xyzw.copy(), m.gripper)

                # One observation, `per_obs` goals: this is where the speed-up is
                # realised, `t` advancing on the wall clock between camera reads.
                for i in range(per_obs):
                    if i:
                        sample = planner.poll_action()
                    if sample is not None:
                        goal = decode(sample)
                    action = rc.ee_pos_action(*goal)
                    dispatcher.send(action, exec_fps)
                    if dataset is not None:
                        rc.add_frame(dataset, obs, action, args.task, controller.cameras)
                    if video_dir is not None and i == 0:
                        for cam, img in m.images.items():
                            rc.write_video_frame(writers, video_dir,
                                                 f"{POLICY}_ep{ep.episode:03d}",
                                                 o_fps, cam, img, video_frames)
                        video_frames += 1
                    if stopper.check() is not None:
                        break
        finally:
            ep.inferences = planner.plans
            if planner.align_errors:
                logger.info("stitches: %d, time-align error max %.4f",
                            len(planner.align_errors), max(planner.align_errors))
            planner.wait_for_pending_inference()
            planner.close()
            for w in writers.values():
                w.release()

    return episode_fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(p, policy_name=POLICY)
    p.add_argument("--speed-up-times", type=float, default=None,
                   help="wall-clock playback multiplier; default "
                        "baselines.bspline.speed_up_times. Start at 1.0")
    p.add_argument("--origin-time-scale", type=float, default=None,
                   help="knot-index units per second, i.e. the RECORDING fps; "
                        "default control_fps(). Do not leave this at upstream's 10")
    p.add_argument("--predict-before-end", type=float, default=None)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--gripper-slowdown", action="store_true",
                   help="drop toward 1x for a few steps when the gripper moves")
    p.add_argument("--restart-on-time-align-error", action="store_true")
    p.add_argument("--consider-gripper-during-align", action="store_true")
    p.add_argument("--disable-time-align", action="store_true",
                   help="start every new plan at min_t instead of stitching")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    port = args.port or int(fc.policy("baselines.zmq.bspline_port"))
    client = PolicyClient(port, int(fc.policy("baselines.zmq.recv_timeout_ms")), args.host)
    meta = client.meta()
    if meta.get("backend") != POLICY:
        p.error(f"server on port {port} is a {meta.get('backend')!r} server, not bspline")
    logger.info("checkpoint: %s", meta)

    ots = float(args.origin_time_scale or rc.origin_time_scale())
    speed = float(args.speed_up_times
                  if args.speed_up_times is not None
                  else fc.policy("baselines.bspline.speed_up_times"))
    planner_kwargs = dict(
        degree=int(meta.get("degree") or args.degree),
        n_obs_steps=int(meta.get("n_obs_steps")
                        or fc.policy("baselines.bspline.n_obs_steps")),
        obs_stride=int(fc.policy("baselines.bspline.obs_stride")),
        origin_time_scale=ots,
        speed_up_times=speed,
        predict_before_end=float(
            args.predict_before_end
            if args.predict_before_end is not None
            else fc.policy("baselines.bspline.predict_before_end")),
        time_align_error_threshold=float(
            fc.policy("baselines.bspline.time_align_error_threshold")),
        time_align_larger_t=fc.policy("baselines.bspline.time_align_larger_t"),
        disable_time_align=args.disable_time_align,
        restart_on_time_align_error=args.restart_on_time_align_error,
        consider_gripper_during_align=args.consider_gripper_during_align,
        gripper_slowdown_enabled=(args.gripper_slowdown
                                  or bool(fc.policy("baselines.bspline.gripper_slowdown_enabled"))),
        gripper_slowdown_threshold=float(
            fc.policy("baselines.bspline.gripper_slowdown_threshold")),
        gripper_slowdown_steps=int(fc.policy("baselines.bspline.gripper_slowdown_steps")),
        gripper_index=_GRIPPER_INDEX,
        compare_dim=_POSE_DIM,
    )

    try:
        train_dataset = rr.resolve_train_dataset(
            args.train_dataset, meta.get("train_dataset"), meta.get("training_hdf5"))
    except ValueError as exc:
        p.error(str(exc))
    run_dir, record = rc.open_run(args, POLICY, train_dataset)

    record.set("policy",
               checkpoint=rr.file_provenance(args.ckpt),
               server={"host": args.host, "port": port, **{k: v for k, v in meta.items()}},
               control_mode=ControlMode.EE_POS.value,
               control_mode_source="fixed",
               control_mode_reason="bspline_bridge/dataset.py trains on absolute poses")
    record.set("parameters", **rc.bspline_parameters(args, meta, planner_kwargs))

    controller = rc.build_robot(args.rig, ControlMode.EE_POS)
    controller.connect()
    status, reason = "completed", None
    try:
        record.set("environment", **rc.environment(args, controller))
        shapes = rc.check_camera_coverage(meta, controller, args.allow_missing_cameras)
        rc.run_episodes(args, controller, run_dir, record,
                        make_episode_fn(client, meta, args, planner_kwargs, shapes, run_dir))
    except KeyboardInterrupt:
        status, reason = "interrupted", "KeyboardInterrupt at the robot"
        print("\r\ninterrupted\r", flush=True)
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.finish(status, reason)
        print(f"\r\nrun written to {run_dir.path}\r", flush=True)
        controller.disconnect()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
