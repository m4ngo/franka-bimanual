#!/usr/bin/env python3
"""Roll a trained SAIL policy out on the bimanual_franka hardware (single arm).

    # in the SAIL venv, first:
    python baselines/sail_bridge/policy_server.py --ckpt-path <CKPT.pth> --port 5556

    # then, in the workspace venv:
    python -m baselines.sail_bridge.rollout --rig single_arm_right --speed 2 --num-episodes 10

The episode loop is upstream's `rollout_diffusion_policy`
(SAIL/run_trained_agent_receding_horizon.py) line for line, with `env.step(a,
control_freq=f)` realised as "send the goal, hold it 1/f s":

  * receding horizon: infer from the observation after the last executed step,
    execute `inf_delay` rows of the PREVIOUS plan, then `execute_n_actions` rows
    of the new one starting at row `inf_delay`. Upstream's simulator stops while
    the policy thinks; here inference runs while those `inf_delay` rows execute,
    and if it is still running after them the arm holds the last goal until it
    returns, so the plan indexing is upstream's exactly. The frame stack is the
    newest observation and the one a demonstration step before it
    (ObservationPump.window): a camera cannot deliver a frame per control step;
  * precision speed modulation: a step runs at the slow rate when any precision
    label in the window around it is set (`get_slowdown_mode_from_model`);
  * error-adaptive guidance: condition the next prediction on the next
    `future_action_condition.horizon` rows of the current plan, but only when the
    arm is within `pos_teb` / `ori_teb` of the first of them.

Speed: `--speed s` sets upstream's `fast_control_freq` to s times the rate the
demonstrations were recorded at; `slow_control_freq` stays at
baselines.exec.slow_fps (upstream's 20 Hz, i.e. 1x). The OSC runs at
baselines.sail.osc_kp_scale times the stock kp.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import franka_config as fc  # noqa: E402

from baselines import rollout_common as rc  # noqa: E402
from baselines import run_record as rr  # noqa: E402
from baselines.rollout_viz import render_after_run  # noqa: E402
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
    s = {
        "speed": float(args.speed),
        "fast_hz": float(args.speed) * rc.data_fps(),
        "slow_hz": float(fc.policy("baselines.exec.slow_fps")),
        "precision": precision,
        "eag": eag,
        "t_f": int(meta["fac_horizon"]) if eag else 0,
        "horizon": int(meta["action_horizon"]),
        "frame_stack": int(meta.get("frame_stack") or meta["n_obs_steps"]),
        "inf_delay": int(fc.policy("baselines.sail.inf_delay")),
        "execute_n": int(fc.policy("baselines.sail.execute_n_actions")),
        "window": int(fc.policy("baselines.sail.slowdown_window_size")),
        "pos_teb": float(fc.policy("baselines.sail.pos_teb")),
        "ori_teb": float(fc.policy("baselines.sail.ori_teb")),
        "gains": rc.sail_gains(),
    }
    # Steady state reads rows [inf_delay + execute_n, + max(inf_delay, t_f)) of a plan.
    need = s["inf_delay"] + s["execute_n"] + max(s["inf_delay"], s["t_f"])
    if s["horizon"] < need:
        raise ValueError(
            f"checkpoint's action_horizon is {s['horizon']}, but inf_delay "
            f"{s['inf_delay']} + execute_n_actions {s['execute_n']} reads {need} rows"
        )
    return s


def _timed_request(client: PolicyClient, payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    rep = client.request(payload)
    return rep, time.perf_counter() - t0


def make_episode_fn(client: PolicyClient, s: dict, shapes: dict):
    def episode_fn(controller, dispatcher, pump, ep, stopper) -> None:
        client.reset()
        executed: list[np.ndarray] = []
        prev_act: np.ndarray | None = None
        prev_action_index = 0
        guide_actions_unnorm: np.ndarray | None = None
        goal: tuple[np.ndarray, np.ndarray] | None = None
        spacing = 1.0 / rc.data_fps()
        cache: dict[int, dict] = {}
        took_s: list[float] = []
        waits = 0
        wait_s = 0.0

        def done() -> bool:
            verdict = stopper.check()
            if verdict is not None and not ep.wall_time_s:
                ep.success = verdict == "success"
                ep.wall_time_s = stopper.elapsed()
            return verdict is not None

        def end_of_step() -> None:
            """The current goal's hold is over; the lead monitor reads the newest snapshot."""
            dispatcher.wait_hold()
            if goal is not None:
                pos_err, rot_err = rc.check_lead(goal[0], goal[1], pump.latest().m)
                ep.max_lead_m = max(ep.max_lead_m, pos_err)
                ep.max_lead_rad = max(ep.max_lead_rad, rot_err)

        def frames() -> list[dict]:
            """FrameStackWrapper's stack: the newest observation and the one a
            demonstration step before it, as the policy was trained on."""
            nonlocal cache
            snaps = pump.window(s["frame_stack"], spacing)
            cache = {sn.seq: cache.get(sn.seq) or rc.sail_obs(sn.m, shapes) for sn in snaps}
            return [cache[sn.seq] for sn in snaps]

        def env_step(act: np.ndarray, index: int) -> None:
            nonlocal goal
            end_of_step()
            a = act[index]
            executed.append(a)
            if s["precision"]:
                slow = rc.slowdown_mode(executed, a, act[index:], s["window"])
                a = a[:-1]
            else:
                slow = False
            pos, rotvec, grip = a[:3], a[3:6], float(a[6])
            quat = Rotation.from_rotvec(rotvec).as_quat()
            dispatcher.step(rc.ee_pos_action(pos, quat, grip, s["gains"]),
                            s["slow_hz"] if slow else s["fast_hz"])
            goal = (pos, quat)
            ep.slow_steps += int(slow)

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            while not done():
                end_of_step()
                guide = None
                if guide_actions_unnorm is not None:
                    desired = guide_actions_unnorm[0]
                    pos, quat = rc.fresh_pose(controller)
                    if rc.tracking_error_low(pos, quat, desired[:3], desired[3:6],
                                             s["pos_teb"], s["ori_teb"]):
                        guide = guide_actions_unnorm
                        ep.guided_inferences += 1
                future = pool.submit(_timed_request, client,
                                     {"obs": frames(), "guide_actions": guide})

                current_action_index = 0
                if prev_act is not None:
                    # Executed while the new plan is inferred: upstream's inference delay.
                    for _ in range(s["inf_delay"]):
                        if done():
                            break
                        env_step(prev_act, prev_action_index)
                        prev_action_index += 1
                        current_action_index += 1

                dispatcher.wait_hold()
                t_wait = time.perf_counter()
                late = prev_act is not None and not future.done()
                rep, took = future.result()
                if late:
                    waits += 1
                    wait_s += time.perf_counter() - t_wait
                if "error" in rep:
                    raise RuntimeError(f"policy server: {rep['error']}")
                act = np.asarray(rep["chunk"], dtype=np.float64)
                ep.inferences += 1
                took_s.append(took)
                # Logged from the first row this plan executes; rows before it are the observed past.
                rows = act[current_action_index:]
                dispatcher.chunk(np.hstack([rows[:, :3],
                                            Rotation.from_rotvec(rows[:, 3:6]).as_quat()]))
                if done():
                    break

                for _ in range(s["execute_n"]):
                    if done():
                        break
                    env_step(act, current_action_index)
                    current_action_index += 1

                prev_action_index = current_action_index
                prev_act = act
                if s["eag"]:
                    guide_actions_unnorm = act[prev_action_index:prev_action_index + s["t_f"]]
        finally:
            pool.shutdown(wait=True)
            if took_s:
                ep.notes["inference_ms_mean"] = round(1e3 * float(np.mean(took_s)), 1)
                ep.notes["inference_ms_max"] = round(1e3 * float(np.max(took_s)), 1)
            # Inferences still running after the inf_delay rows: the arm held its goal meanwhile.
            ep.notes["inference_waits"] = waits
            ep.notes["inference_wait_s"] = round(wait_s, 3)

    return episode_fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(p, policy_name=POLICY)
    p.add_argument("--speed", type=float, default=1.0,
                   help="fast_control_freq as a multiple of the demonstrations' rate "
                        "(1, 2, 3 ...); precision steps run at baselines.exec.slow_fps")
    p.add_argument("--no-precision", action="store_true",
                   help="ignore the precision label and run every step at the fast rate")
    p.add_argument("--no-eag", action="store_true", help="disable error-adaptive guidance")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
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
    except ValueError as exc:
        p.error(str(exc))
    logger.info("speed %.2fx: fast/slow %.0f/%.0f Hz, precision=%s eag=%s, "
                "inf_delay=%d execute_n=%d horizon=%d, osc kp %.0f",
                s["speed"], s["fast_hz"], s["slow_hz"], s["precision"], s["eag"],
                s["inf_delay"], s["execute_n"], s["horizon"], rc.sail_kp())
    run_dir, record = rc.open_run(args, POLICY, train_dataset)

    record.set("policy",
               checkpoint=rr.file_provenance(args.ckpt),
               server={"host": args.host, "port": port, **{k: v for k, v in meta.items()}},
               control_mode=control_mode.value,
               control_mode_source="checkpoint action_keys")
    record.set("parameters", **rc.sail_parameters(args, meta, s))

    controller = rc.build_robot(args.rig, control_mode)
    controller.connect()
    status, reason = "completed", None
    try:
        record.set("environment", **rc.environment(args, controller))
        shapes = rc.check_camera_coverage(meta, controller, args.allow_missing_cameras)
        rc.run_episodes(args, controller, run_dir, record,
                        make_episode_fn(client, s, shapes), nominal_hz=s["fast_hz"])
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
