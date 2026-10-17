#!/usr/bin/env python3
"""Roll a trained B-Spline Policy out on the bimanual_franka hardware (single arm).

    # in the B-Spline venv, first:
    python baselines/bspline_bridge/policy_server.py --ckpt-path <CKPT> --port 5555

    # then, in the workspace venv:
    python -m baselines.bspline_bridge.rollout --rig single_arm_right --speed 2 --num-episodes 10

or both at once: ./scripts/bspline_rollout.sh --start-server --ckpt <CKPT>

The control loop is upstream's `rollout_episode`
(real_env/yam_teleop/rollout_local_policy.py) and the plan management is
`PolicyLocalBSpline` (spline_plan.py), with the model behind the policy server:
every 1/control_freq s a new spline is requested if the current one ends within
`predict_before_end` s, a new spline is time-aligned onto the old one, and the
plan is sampled at wall-clock `t = elapsed * speed_up_times * origin_time_scale`.
No goal is sent before the first plan exists, and nothing bounds how far a goal
may lead the arm: upstream has no such bound, and a sped-up plan leads on purpose.

The policy's observations are the newest snapshot and the one nearest a
demonstration step before it (ObservationPump.window), which is what upstream's
`obs_stride` selects from its per-tick history.

Upstream speeds its servo up with the plan (`set_ik_dt_scale(speed_up_times)`),
so the arm trails a sped-up plan by the distance it trailed it at 1x. Here the
OSC stays the stock controller the demonstrations were recorded under, at every
speed, and each goal is led along the plan by (kd/kp)(1 - 1/speed) s instead
(rollout_common.bspline_lead), which holds that 1x lag.

Not upstream's: when the plan's gripper command closes (crosses
baselines.bspline.grasp_threshold downward) the arm holds the pose it is at for
baselines.bspline.grasp_hold_s with the gripper commanded closed, whether or not
the fingers have finished, while the plan's clock stands still, so the plan
resumes where it stopped: pause, grasp, unpause. The gripper stays
commanded closed until the plan opens it again: resumed mid-transition, the
plan's own value would reopen the fingers around what they just grasped.

Control mode is EE_POS: `bspline_bridge/dataset.py` trains on the absolute
commanded pose. The sampled action is 10-dim -- xyz + rot6d + gripper -- because
B-Spline's dataset loader turns our 7-dim [pos, rotvec, gripper] rows into
rotation_6d.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import franka_config as fc  # noqa: E402

from baselines import rollout_common as rc  # noqa: E402
from baselines import run_record as rr  # noqa: E402
from baselines.policy_math import lead_goal  # noqa: E402
from baselines.rollout_viz import render_after_run  # noqa: E402
from baselines.bspline_bridge.spline_plan import (  # noqa: E402
    GRIPPER_INDEX, POSE_DIM, HoldClock, SplinePlanner, decode_action as decode,
)
from baselines.zmq_client import PolicyClient  # noqa: E402
from lerobot_robot_bimanual_franka import ControlMode  # noqa: E402

logger = logging.getLogger("baselines.bspline")

POLICY = "bspline"
# The normalised gripper command for shut; the demonstrations command exactly this.
_CLOSED = 0.0


def make_episode_fn(client: PolicyClient, planner_kwargs: dict, control_freq: float,
                    shapes: dict, gains: dict, lead: np.ndarray):
    period = 1.0 / float(control_freq)
    spacing = 1.0 / rc.data_fps()
    leading = bool(np.any(lead > 0.0))
    hold_s = float(fc.policy("baselines.bspline.grasp_hold_s"))
    threshold = float(fc.policy("baselines.bspline.grasp_threshold"))
    rearm = float(fc.policy("baselines.bspline.grasp_rearm"))

    def episode_fn(controller, dispatcher, pump, ep, stopper) -> None:
        client.reset()
        clock = HoldClock()
        planner = SplinePlanner(client, clock=clock.now, **planner_kwargs)
        n_obs = planner.n_obs_steps
        plans_seen = 0
        goal: tuple[np.ndarray, np.ndarray] | None = None
        snap_seq = 0
        window: list[dict] = []
        cache: dict[int, dict] = {}
        exhausted = 0
        last_grip: float | None = None
        closed = False            # a grasp held the gripper shut; the plan has not reopened it
        armed = False             # the command has been fully open since the last grasp
        hold_goal: dict | None = None
        hold_start: float | None = None   # when the current grasp hold began
        holds = 0
        held_s: list[float] = []
        try:
            start_time = time.perf_counter()
            for step_idx in itertools.count():
                verdict = stopper.check()
                if verdict is not None:
                    ep.success = verdict == "success"
                    ep.wall_time_s = stopper.elapsed()
                    break
                # Upstream's absolute schedule: late ticks run back to back.
                rc.sleep_until(start_time + step_idx * period)

                snap = pump.latest()
                if snap.seq != snap_seq:
                    snap_seq = snap.seq
                    # One demonstration step apart, as upstream's obs_stride spaces them.
                    snaps = pump.window(n_obs, spacing)
                    cache = {sn.seq: cache.get(sn.seq) or rc.bspline_obs(sn.m, shapes)
                             for sn in snaps}
                    # Upstream waits for a full history before its first request.
                    if len(cache) == n_obs:
                        window = [cache[sn.seq] for sn in snaps]
                    if goal is not None:
                        pos_err, rot_err = rc.lead(goal[0], goal[1], snap.m)
                        ep.max_lead_m = max(ep.max_lead_m, pos_err)
                        ep.max_lead_rad = max(ep.max_lead_rad, rot_err)

                now = time.perf_counter()
                if hold_start is not None:
                    if now - hold_start < hold_s:
                        dispatcher.send(hold_goal, anchor=snap.kin)
                        continue
                    clock.release(now)
                    held_s.append(now - hold_start)
                    hold_start = None
                sample = planner.step(window, obs_time=clock.effective(snap.t))
                if planner.plans != plans_seen:
                    plans_seen = planner.plans
                    dispatcher.chunk([np.concatenate(decode(r)[:2])
                                      for r in planner.plan_samples()])
                if sample is None:
                    exhausted += int(plans_seen > 0)
                    continue
                pos, quat, grip = decode(sample)
                if hold_s > 0.0:
                    # Only a close from fully open is a grasp: a release whose command
                    # wobbles back under the threshold (a replan, spline overshoot)
                    # must never pause the arm.
                    grasping = (armed and last_grip is not None
                                and last_grip > threshold >= grip)
                    last_grip = grip
                    if closed and grip > threshold:
                        closed = False            # the plan opens the gripper itself
                    if grip >= rearm:
                        armed = True
                    if grasping:
                        armed = False
                        # Hold where the arm is, not where the plan's goal is: it leads the arm.
                        closed = True
                        holds += 1
                        clock.hold(now)
                        hold_start = now
                        goal = rc.fresh_pose(controller)
                        hold_goal = rc.ee_pos_action(*goal, _CLOSED, gains)
                        dispatcher.send(hold_goal, anchor=snap.kin)
                        continue
                    if closed:
                        grip = _CLOSED
                if leading:
                    pos, quat = lead_goal(pos, quat, *decode(planner.peek(period))[:2],
                                          period, lead)
                dispatcher.send(rc.ee_pos_action(pos, quat, grip, gains), anchor=snap.kin)
                goal = (pos, quat)
        finally:
            planner.wait_for_pending_inference()
            planner.close()
            ep.inferences = planner.plans
            errors = [e for e in planner.align_errors if np.isfinite(e)]
            ep.notes["time_align_error_max"] = round(max(errors), 6) if errors else None
            ep.notes["time_align_over_threshold"] = sum(
                e > planner.time_align_error_threshold for e in planner.align_errors)
            # Ticks with no plan to sample after the first one landed: the arm held its goal.
            ep.notes["plan_exhausted_ticks"] = exhausted
            ep.notes["spline_request_failures"] = planner.failures
            ep.notes["stitches_deferred"] = planner.deferred
            ep.notes["grasp_holds"] = holds
            ep.notes["grasp_hold_s"] = [round(s, 3) for s in held_s]

    return episode_fn


def planner_settings(meta: dict, args, control_freq: float) -> dict:
    """PolicyLocalBSpline's constructor arguments, as rollout_local_policy.make_policy builds them."""
    return dict(
        degree=int(meta.get("degree") or args.degree),
        n_obs_steps=int(meta.get("n_obs_steps") or fc.policy("baselines.bspline.n_obs_steps")),
        origin_time_scale=float(args.origin_time_scale or rc.origin_time_scale()),
        speed_up_times=float(args.speed),
        predict_before_end=float(
            args.predict_before_end if args.predict_before_end is not None
            else fc.policy("baselines.bspline.predict_before_end")),
        time_align_error_threshold=float(
            fc.policy("baselines.bspline.time_align_error_threshold")),
        time_align_larger_t=fc.policy("baselines.bspline.time_align_larger_t"),
        disable_time_align=args.disable_time_align,
        restart_on_time_align_error=args.restart_on_time_align_error,
        consider_gripper_during_align=args.consider_gripper_during_align,
        gripper_slowdown_enabled=(args.gripper_slowdown
                                  or bool(fc.policy("baselines.bspline.gripper_slowdown_enabled"))),
        gripper_slowdown_threshold=float(fc.policy("baselines.bspline.gripper_slowdown_threshold")),
        gripper_slowdown_steps=int(fc.policy("baselines.bspline.gripper_slowdown_steps")),
        gripper_index=GRIPPER_INDEX,
        compare_dim=POSE_DIM,
        defer_stitch=bool(fc.policy("baselines.bspline.defer_stitch")),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(p, policy_name=POLICY)
    p.add_argument("--speed", "--speed-up-times", dest="speed", type=float, default=1.0,
                   help="upstream's speed_up_times: the plan plays this many times faster "
                        "than the demonstrations (1, 2, 3 ...)")
    p.add_argument("--control-freq", type=float, default=None,
                   help="goal rate; default baselines.bspline.control_freq")
    p.add_argument("--origin-time-scale", type=float, default=None,
                   help="knot-index units per second, i.e. the RECORDING fps; "
                        "default baselines.bspline.origin_time_scale")
    p.add_argument("--predict-before-end", type=float, default=None)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--gripper-slowdown", action="store_true",
                   help="drop toward 1x for a few steps when the gripper moves")
    p.add_argument("--restart-on-time-align-error", action="store_true")
    p.add_argument("--consider-gripper-during-align", action="store_true")
    p.add_argument("--disable-time-align", action="store_true",
                   help="start every new plan at t=0 instead of stitching")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.speed <= 0:
        p.error(f"--speed must be positive, got {args.speed}")
    rc.configure_process()
    port = args.port or int(fc.policy("baselines.zmq.bspline_port"))
    client = PolicyClient(port, int(fc.policy("baselines.zmq.recv_timeout_ms")), args.host)
    meta = client.meta()
    if meta.get("backend") != POLICY:
        p.error(f"server on port {port} is a {meta.get('backend')!r} server, not bspline")
    logger.info("checkpoint: %s", meta)

    control_freq = float(args.control_freq or fc.policy("baselines.bspline.control_freq"))
    planner_kwargs = planner_settings(meta, args, control_freq)

    try:
        train_dataset = rr.resolve_train_dataset(
            args.train_dataset, meta.get("train_dataset"), meta.get("training_hdf5"))
        home_match = rc.resolve_home(args, train_dataset)
    except ValueError as exc:
        p.error(str(exc))
    run_dir, record = rc.open_run(args, POLICY, train_dataset)

    record.set("policy",
               checkpoint=rr.file_provenance(args.ckpt),
               server={"host": args.host, "port": port, **{k: v for k, v in meta.items()}},
               control_mode=ControlMode.EE_POS.value,
               control_mode_source="fixed",
               control_mode_reason="bspline_bridge/dataset.py trains on absolute poses")

    controller = rc.build_robot(args.rig, ControlMode.EE_POS)
    gains = rc.stock_gains()
    lead = rc.bspline_lead(args.speed, controller)
    record.set("parameters", **rc.bspline_parameters(args, meta, planner_kwargs, control_freq,
                                                     lead))
    logger.info("speed %.2fx: goals at %.0f Hz, origin_time_scale %.1f, "
                "predict_before_end %.2f s, stock OSC, goal lead %.4f s",
                planner_kwargs["speed_up_times"], control_freq,
                planner_kwargs["origin_time_scale"], planner_kwargs["predict_before_end"],
                float(lead[0]))
    controller.connect()
    status, reason = "completed", None
    try:
        record.set("environment", **rc.environment(args, controller), home_match=home_match)
        shapes = rc.check_camera_coverage(meta, controller, args.allow_missing_cameras)
        rc.run_episodes(args, controller, run_dir, record,
                        make_episode_fn(client, planner_kwargs, control_freq, shapes,
                                        gains, lead),
                        nominal_hz=control_freq)
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
        render_after_run(run_dir.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
