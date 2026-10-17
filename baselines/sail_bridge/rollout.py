#!/usr/bin/env python3
"""Roll a trained SAIL policy out on the bimanual_franka hardware (single arm).

    # in the SAIL venv, first:
    python baselines/sail_bridge/policy_server.py --ckpt-path <CKPT.pth> --port 5556

    # then, in the workspace venv:
    python -m baselines.sail_bridge.rollout --rig single_arm_right --speed 2 --num-episodes 10

The repo ships only SAIL's simulator executor
(SAIL/run_trained_agent_receding_horizon.py). It assumes the simulator stops while
the policy thinks, and that each row is a goal robosuite's stiff OSC holds for one
control period. Run on the arm, that is a staircase of goals with a zero velocity
target: the arm brakes toward every row and lurches to the next, trails the plan
by kd/kp, and the trailing then switches error-adaptive guidance off, so
successive plans disagree. That is the shaking and the bouncing.

This is the hardware executor the paper describes instead (arXiv 2506.11948,
Sec. 4.4 and App. D.1; see sail_bridge/schedule.py):

  * receding horizon: execute `execute_n_actions` rows of a plan from the row it
    took over at, then observe and infer. The plan keeps running meanwhile, and
    the next one takes over when it lands, at the row its own schedule says is
    due -- what upstream's fixed `inf_delay` rows stand in for in simulation. The
    request goes out by row H - inf_delay at the latest, and earlier when the
    measured inference time would otherwise run the plan dry;
  * precision speed modulation: a row is due 1/slow_hz after the previous one
    when any label in the window around it is set, else 1/fast_hz
    (`get_slowdown_mode_from_model`, as upstream);
  * error-adaptive guidance: the next inference is conditioned on the current
    plan's rows over its first `future_action_condition.horizon` steps, only
    when the arm, at the observation, is within `pos_teb` / `ori_teb` of where
    the plan wanted it (`check_if_tracking_error_low`, as upstream);
  * tracking: a spline is fitted over the plan and streamed at
    baselines.sail.goal_hz, each goal led along its velocity by the OSC's own
    kd/kp, which turns its kp*e_p - kd*v into the paper's kp*e_p + kd*e_v. kp is
    baselines.sail.osc_kp_scale times the stock kp, the paper's hardware kp 300
    (App. J, Table J.5), at damping ratio baselines.sail.osc_damping_ratio.

Speed: `--speed s` makes fast rows due s times faster than the demonstrations
were recorded; slow rows stay at baselines.exec.slow_fps (1x). The two frames the
policy sees are one fast step apart, as upstream's frame stack is one env step,
as far as the cameras' frame rate allows.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import franka_config as fc  # noqa: E402

from baselines import rollout_common as rc  # noqa: E402
from baselines import run_record as rr  # noqa: E402
from baselines.policy_math import apply_lead  # noqa: E402
from baselines.rollout_viz import render_after_run  # noqa: E402
from baselines.sail_bridge.schedule import ScheduledPlan  # noqa: E402
from baselines.zmq_client import PolicyClient  # noqa: E402
from lerobot_robot_bimanual_franka import ControlMode  # noqa: E402

logger = logging.getLogger("baselines.sail")

POLICY = "sail"
# Upstream evaluates absolute poses with control_delta False; `actions` is our converter's delta.
_ABS_MARKER = "absolute_actions"


def resolve_control_mode(meta: dict) -> ControlMode:
    keys = list(meta.get("action_keys") or [])
    if len(keys) != 1 or _ABS_MARKER not in keys[0]:
        raise ValueError(
            f"checkpoint declares action_keys={keys}; SAIL's evaluation executes one "
            f"absolute pose key (one containing {_ABS_MARKER!r}) under control_delta "
            "False. Retrain on absolute_actions_with_precision."
        )
    return ControlMode.EE_POS


def settings(meta: dict, args) -> dict:
    """Upstream's kwargs literal, from config/policy.yaml and the checkpoint."""
    precision = bool(meta["precision_column"]) and not args.no_precision
    eag = bool(meta["guided"] and meta["fac_enabled"]) and not args.no_eag
    speed = float(args.speed)
    if speed <= 0:
        raise ValueError(f"--speed must be positive, got {speed}")
    s = {
        "speed": speed,
        "fast_hz": speed * rc.data_fps(),
        "slow_hz": float(fc.policy("baselines.exec.slow_fps")),
        "goal_hz": float(fc.policy("baselines.sail.goal_hz")),
        "knot_rows": int(fc.policy("baselines.sail.spline_knot_rows")),
        "precision": precision,
        "eag": eag,
        "t_f": int(meta["fac_horizon"]) if eag else 0,
        "horizon": int(meta["action_horizon"]),
        "frame_stack": int(meta.get("frame_stack") or meta["n_obs_steps"]),
        "frame_spacing_s": 1.0 / (speed * rc.data_fps()),
        "inf_delay": int(fc.policy("baselines.sail.inf_delay")),
        "execute_n": int(fc.policy("baselines.sail.execute_n_actions")),
        "window": int(fc.policy("baselines.sail.slowdown_window_size")),
        "pos_teb": float(fc.policy("baselines.sail.pos_teb")),
        "ori_teb": float(fc.policy("baselines.sail.ori_teb")),
        "gains": rc.sail_gains(),
        # The OSC's kd/kp, set from the arm once it exists (rc.osc_lag); zero = no velocity term.
        "lead": np.zeros(6),
    }
    # A request goes out by row H - inf_delay, and the guide it carries is that
    # row and the t_f after it, so both have to fit in the plan.
    if s["horizon"] <= s["inf_delay"] or s["t_f"] > s["inf_delay"]:
        raise ValueError(
            f"checkpoint's action_horizon {s['horizon']} and fac_horizon {s['t_f']} do not "
            f"fit inf_delay {s['inf_delay']}: the guide must lie inside the last "
            "inf_delay rows of the plan it is read from"
        )
    return s


def _timed_request(client: PolicyClient, payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    rep = client.request(payload)
    return rep, time.perf_counter() - t0


def _pct(values, q) -> float | None:
    return round(float(np.percentile(values, q)), 2) if len(values) else None


def make_episode_fn(client: PolicyClient, s: dict, shapes: dict):
    period = 1.0 / s["goal_hz"]

    def episode_fn(controller, dispatcher, pump, ep, stopper) -> None:
        client.reset()
        pool = ThreadPoolExecutor(max_workers=1)
        cache: dict[int, dict] = {}
        plan: ScheduledPlan | None = None
        pending = None                 # (future, t_obs, t_sent)
        trigger = 0.0                  # row of `plan` at which the next request goes out
        due = 0                        # rows of `plan` already due
        executed: deque = deque(maxlen=16)   # (t_due, row) of rows already due
        turnaround: float | None = None      # request -> reply, s (EMA)
        last_seq = 0
        took, ages, entries, jump_cm, jump_deg, track_cm = [], [], [], [], [], []
        exhausted = 0

        def request():
            """Observe, decide guidance, and start an inference; returns the pending triple."""
            nonlocal cache
            snaps = pump.window(s["frame_stack"], s["frame_spacing_s"])
            cache = {sn.seq: cache.get(sn.seq) or rc.sail_obs(sn.m, shapes) for sn in snaps}
            newest = snaps[-1]
            guide = None
            if s["eag"] and plan is not None:
                ref = plan.rows_at(plan.index_at(newest.t), s["t_f"])
                if rc.tracking_error_low(newest.m.pos, newest.m.quat_xyzw, ref[0, :3],
                                         ref[0, 3:6], s["pos_teb"], s["ori_teb"]):
                    guide = ref
                    ep.guided_inferences += 1
            fut = pool.submit(_timed_request, client,
                              {"obs": [cache[sn.seq] for sn in snaps], "guide_actions": guide})
            return fut, newest.t, time.perf_counter()

        def install(rep: dict, t_obs: float, now: float) -> None:
            """The new plan takes over at the row due now; the old one's later rows are dropped."""
            nonlocal plan, trigger, due
            if "error" in rep:
                raise RuntimeError(f"policy server: {rep['error']}")
            history = [row for t, row in executed if t < t_obs]
            new = ScheduledPlan(rep["chunk"], t_obs, s["fast_hz"], s["slow_hz"],
                                precision=s["precision"], window=s["window"],
                                knot_rows=s["knot_rows"], history=history)
            entry = new.entry(now)
            if plan is not None:
                (p_old, r_old), (p_new, r_new) = plan.pose(now), new.pose(now)
                jump_cm.append(100.0 * float(np.linalg.norm(p_new - p_old)))
                jump_deg.append(float(np.degrees((r_new * r_old.inv()).magnitude())))
            plan, due = new, entry
            trigger = float(min(entry + s["execute_n"], s["horizon"] - s["inf_delay"]))
            entries.append(entry)
            ep.inferences += 1
            rows = new.rows[entry:]
            if len(rows):
                dispatcher.chunk(np.hstack([rows[:, :3],
                                            Rotation.from_rotvec(rows[:, 3:6]).as_quat()]))

        t0 = time.perf_counter()
        try:
            pending = request()
            for tick in itertools.count():
                verdict = stopper.check()
                if verdict is not None:
                    ep.success = verdict == "success"
                    ep.wall_time_s = stopper.elapsed()
                    break
                rc.sleep_until(t0 + tick * period)
                now = time.perf_counter()

                if pending is not None and pending[0].done():
                    (rep, inference_s), t_obs, t_sent = pending[0].result(), pending[1], pending[2]
                    pending = None
                    took.append(inference_s)
                    ages.append(t_sent - t_obs)
                    turnaround = (now - t_sent if turnaround is None
                                  else 0.8 * turnaround + 0.2 * (now - t_sent))
                    install(rep, t_obs, now)
                if plan is None:
                    continue    # nothing is sent before the first plan exists

                while due < len(plan) and plan.times[due] <= now:
                    executed.append((plan.times[due], plan.rows[due]))
                    ep.slow_steps += int(plan.slow[due])
                    due += 1
                running_dry = turnaround is not None and plan.end - now <= turnaround + period
                if pending is None and (plan.index_at(now) >= trigger or running_dry):
                    pending = request()
                exhausted += int(now >= plan.end)

                pos, rot = plan.pose(now)
                vel, ang = plan.velocity(now)
                goal_pos, goal_quat = apply_lead(pos, rot.as_quat(), vel, ang, s["lead"])
                snap = pump.latest()
                dispatcher.send(rc.ee_pos_action(goal_pos, goal_quat, plan.gripper(now), s["gains"]),
                                anchor=snap.kin)
                pos_err, rot_err = rc.lead(goal_pos, goal_quat, snap.m)
                ep.max_lead_m = max(ep.max_lead_m, pos_err)
                ep.max_lead_rad = max(ep.max_lead_rad, rot_err)
                if snap.seq != last_seq and snap.t >= plan.start:
                    last_seq = snap.seq
                    want, _ = plan.pose(snap.t)
                    track_cm.append(100.0 * float(np.linalg.norm(want - snap.m.pos)))
        finally:
            pool.shutdown(wait=True)
            if took:
                ep.notes["inference_ms_mean"] = round(1e3 * float(np.mean(took)), 1)
                ep.notes["inference_ms_max"] = round(1e3 * float(np.max(took)), 1)
                ep.notes["observation_age_ms_mean"] = round(1e3 * float(np.mean(ages)), 1)
            if entries:
                ep.notes["entry_row_mean"] = round(float(np.mean(entries)), 2)
            # Where the new plan disagreed with the old at the moment it took over.
            ep.notes["switch_jump_cm_p50"] = _pct(jump_cm, 50)
            ep.notes["switch_jump_cm_max"] = _pct(jump_cm, 100)
            ep.notes["switch_jump_deg_max"] = _pct(jump_deg, 100)
            # The plan's pose at each observation against the arm's.
            ep.notes["tracking_error_cm_p50"] = _pct(track_cm, 50)
            ep.notes["tracking_error_cm_p95"] = _pct(track_cm, 95)
            # Time the arm held the last row because the next plan had not landed.
            ep.notes["plan_exhausted_s"] = round(exhausted * period, 3)

    return episode_fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(p, policy_name=POLICY)
    p.add_argument("--speed", type=float, default=1.0,
                   help="fast rows run this many times faster than the demonstrations "
                        "(1, 2, 3 ...); precision rows run at baselines.exec.slow_fps")
    p.add_argument("--no-precision", action="store_true",
                   help="ignore the precision label and run every step at the fast rate")
    p.add_argument("--no-eag", action="store_true", help="disable error-adaptive guidance")
    p.add_argument("--no-velocity-lead", action="store_true",
                   help="track the plan's pose only, without its velocity (diagnostic)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    rc.configure_process()
    port = args.port or int(fc.policy("baselines.zmq.sail_port"))
    client = PolicyClient(port, int(fc.policy("baselines.zmq.recv_timeout_ms")), args.host)
    meta = client.meta()
    if meta.get("backend") != POLICY:
        p.error(f"server on port {port} is a {meta.get('backend')!r} server, not sail")
    logger.info("checkpoint: %s", meta)

    try:
        control_mode = resolve_control_mode(meta)
        s = settings(meta, args)
        train_dataset = rr.resolve_train_dataset(
            args.train_dataset, meta.get("train_dataset"), meta.get("training_hdf5"))
        home_match = rc.resolve_home(args, train_dataset)
    except ValueError as exc:
        p.error(str(exc))
    run_dir, record = rc.open_run(args, POLICY, train_dataset)

    record.set("policy",
               checkpoint=rr.file_provenance(args.ckpt),
               server={"host": args.host, "port": port, **{k: v for k, v in meta.items()}},
               control_mode=control_mode.value,
               control_mode_source="checkpoint action_keys")

    controller = rc.build_robot(args.rig, control_mode)
    if not args.no_velocity_lead:
        s["lead"] = rc.osc_lag(s["gains"], controller)
    record.set("parameters", **rc.sail_parameters(args, meta, s))
    logger.info("speed %.2fx: rows due at %.0f Hz (slow %.0f Hz), streamed at %.0f Hz; "
                "precision=%s eag=%s execute_n=%d horizon=%d; osc kp %.0f, velocity lead %.3f s",
                s["speed"], s["fast_hz"], s["slow_hz"], s["goal_hz"], s["precision"], s["eag"],
                s["execute_n"], s["horizon"], rc.sail_kp(), float(s["lead"][0]))
    controller.connect()
    status, reason = "completed", None
    try:
        record.set("environment", **rc.environment(args, controller), home_match=home_match)
        shapes = rc.check_camera_coverage(meta, controller, args.allow_missing_cameras)
        rc.run_episodes(args, controller, run_dir, record,
                        make_episode_fn(client, s, shapes), nominal_hz=s["goal_hz"])
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
