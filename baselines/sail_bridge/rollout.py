#!/usr/bin/env python3
"""Roll a trained SAIL policy out on the bimanual_franka hardware (single arm).

    # in the SAIL conda env, first:
    python baselines/sail_bridge/policy_server.py --ckpt-path <CKPT.pth> --port 5556

    # then, in the workspace venv:
    python -m baselines.sail_bridge.rollout --rig single_arm_right --num-episodes 10

SAIL's own executor is not released and is robosuite-bound, so this is the loop
written from its description (baselines/SAIL.md, pass 5). It implements all three
of its eval-time mechanisms:

  * receding horizon -- execute `inf_delay` steps of the PREVIOUS prediction
    before switching to `execute_n_actions` of the new one, so inference latency
    is modelled rather than hidden;
  * precision speed modulation -- the last action column is a precision label;
    any label set inside a window around the current step drops that step from
    fast_fps to slow_fps;
  * error-adaptive guidance -- condition the next prediction on the tail of the
    current plan, but only while the arm is actually tracking it.

The control mode is resolved from the checkpoint, not chosen here: SAIL's
training template trains on absolute poses while our converter also writes a
delta `actions` key, and only the checkpoint knows which it learned.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import franka_config as fc  # noqa: E402

from baselines import rollout_common as rc  # noqa: E402
from baselines.zmq_client import PolicyClient  # noqa: E402
from lerobot_robot_bimanual_franka import ControlMode  # noqa: E402

logger = logging.getLogger("baselines.sail")

POLICY = "sail"
# Action-key -> control mode. `actions` is the delta our converter derives with
# OSCGoalBuilder.delta_from_absolute; the two absolute keys are the poses SAIL's
# own pipeline trains on.
_DELTA_KEYS = ("actions",)
_ABS_PREFIX = "absolute_actions"


def resolve_control_mode(meta: dict, override: str) -> ControlMode:
    if override == "ee_delta":
        return ControlMode.EE_DELTA
    if override == "ee_pos":
        return ControlMode.EE_POS
    keys = list(meta.get("action_keys") or [])
    if len(keys) != 1:
        raise ValueError(
            f"checkpoint declares action_keys={keys}; exactly one is supported. "
            "Pass --control-mode to choose explicitly."
        )
    key = keys[0]
    if key in _DELTA_KEYS:
        return ControlMode.EE_DELTA
    if key.startswith(_ABS_PREFIX):
        return ControlMode.EE_POS
    raise ValueError(
        f"cannot tell the action space from action_keys={keys!r}. Expected "
        f"{_DELTA_KEYS[0]!r} (delta) or {_ABS_PREFIX}* (absolute); pass "
        "--control-mode to override."
    )


def _decode(row: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One action row -> (xyz, rotvec, gripper), the layout every converter here
    writes (baselines/common.py:pos_rotvec_gripper). The precision label, if any,
    has already been stripped by the caller."""
    return row[:3].astype(np.float64), row[3:6].astype(np.float64), float(row[6])


def make_episode_fn(client: PolicyClient, meta: dict, args,
                    control_mode: ControlMode, shapes: dict):
    precision = meta["precision_column"] and not args.no_precision
    eag = meta["guided"] and meta["fac_enabled"] and not args.no_eag
    t_f = int(meta["fac_horizon"])
    horizon = int(meta["action_horizon"])

    inf_delay = int(fc.policy("baselines.sail.inf_delay"))
    execute_n = int(fc.policy("baselines.sail.execute_n_actions"))
    window = int(fc.policy("baselines.sail.slowdown_window_size"))
    pos_teb = float(fc.policy("baselines.sail.pos_teb"))
    ori_teb = float(fc.policy("baselines.sail.ori_teb"))

    fast_fps = float(args.exec_fps or fc.policy("baselines.exec.fast_fps"))
    slow_fps = float(args.slow_fps or fc.policy("baselines.exec.slow_fps"))
    o_fps = float(args.obs_fps or rc.obs_fps())

    if horizon < inf_delay + execute_n:
        raise ValueError(
            f"checkpoint's action_horizon is {horizon} but the receding horizon "
            f"consumes inf_delay + execute_n_actions = {inf_delay + execute_n} "
            "actions per inference. Lower them in config/policy.yaml "
            "(baselines.sail) or retrain with a longer horizon."
        )
    logger.info(
        "mode=%s precision=%s eag=%s fast/slow=%.0f/%.0f Hz obs=%.0f Hz "
        "inf_delay=%d execute_n=%d horizon=%d",
        control_mode.value, precision, eag, fast_fps, slow_fps, o_fps,
        inf_delay, execute_n, horizon,
    )

    def episode_fn(controller, dispatcher, dataset, ep, stopper) -> None:
        client.reset()
        dispatcher.start()

        prev_chunk: np.ndarray | None = None
        prev_index = 0
        ref: np.ndarray | None = None       # unnormalised reference for EAG
        executed: list[np.ndarray] = []
        goal: tuple[np.ndarray, np.ndarray] | None = None
        writers: dict = {}
        video_dir = Path(args.save_videos).expanduser() if args.save_videos else None
        # One frame per OBSERVATION, indexed by written frames. Writing per
        # dispatched step and labelling at fast_fps mislabels every slow step,
        # and the two baselines' videos would not line up.
        video_frames = 0
        # The pose the live plan was formed at, and the deltas commanded from it
        # since. Only read on the delta path's EAG check, which cannot run before
        # the first plan has produced a reference.
        anchor: tuple[np.ndarray, np.ndarray] = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
        committed: list[np.ndarray] = []

        try:
            while True:
                verdict = stopper.check()
                if verdict is not None:
                    ep.success = verdict == "success"
                    ep.wall_time_s = stopper.elapsed()
                    break

                obs = controller.get_observation()
                m = rc.measure(obs)

                # EE_POS only: EE_DELTA re-anchors every step, so its lead is
                # structurally one clipped delta and there is nothing to monitor.
                if goal is not None and control_mode is ControlMode.EE_POS:
                    pos_err, rot_err = rc.check_lead(goal[0], goal[1], m)
                    ep.max_lead_m = max(ep.max_lead_m, pos_err)
                    ep.max_lead_rad = max(ep.max_lead_rad, rot_err)

                guide = None
                if eag and ref is not None and len(ref):
                    if control_mode is ControlMode.EE_POS:
                        want_pos, want_rot = ref[0][:3], ref[0][3:6]
                    else:
                        # The reference rows are deltas, each relative to its own
                        # step's measured pose, so row 0 on its own carries no
                        # tracking information. Propagate the anchor through the
                        # deltas already commanded from this plan to recover the pose
                        # perfect tracking would have reached -- see
                        # rollout_common.propagate_pose.
                        p, q = rc.propagate_pose(anchor[0], anchor[1], committed)
                        want_pos = p + ref[0][:3]
                        want_rot = (
                            Rotation.from_rotvec(ref[0][3:6])
                            * Rotation.from_quat(q)
                        ).as_rotvec()
                    if rc.tracking_error_low(m.pos, m.quat_xyzw, want_pos, want_rot,
                                             pos_teb, ori_teb):
                        guide = ref
                        ep.guided_inferences += 1

                t_infer = time.perf_counter()
                rep = client.request({"obs": rc.sail_obs(m, shapes), "guide_actions": guide})
                if "error" in rep:
                    raise RuntimeError(f"policy server: {rep['error']}")
                chunk = np.asarray(rep["chunk"], dtype=np.float64)
                ep.inferences += 1
                anchor = (m.pos.copy(), m.quat_xyzw.copy())
                committed: list[np.ndarray] = []

                # Upstream's index bookkeeping: current_action_index advances through
                # the inf_delay phase too, so the new chunk is entered at inf_delay
                # rather than 0 -- that is what makes the horizon recede.
                current_index = 0
                plan: list[tuple[np.ndarray, int]] = []
                if prev_chunk is not None:
                    for i in range(inf_delay):
                        if prev_index + i >= len(prev_chunk):
                            break
                        plan.append((prev_chunk, prev_index + i))
                        current_index += 1
                entry = current_index
                for j in range(execute_n):
                    if entry + j >= len(chunk):
                        break
                    plan.append((chunk, entry + j))
                    current_index += 1

                for src, idx in plan:
                    row = src[idx]
                    # Append BEFORE the window check, as upstream does: its
                    # traj["executed_actions"].append(a) precedes the
                    # get_slowdown_mode_from_model call, so the current row sits
                    # in the left window too. Appending after shifts the whole
                    # left window one step further back and can flip the verdict.
                    executed.append(row)
                    slow = (precision
                            and rc.slowdown_mode(executed, row, src[idx:], window))
                    a = row[:-1] if precision else row
                    dpos_or_pos, rot, grip = _decode(a)
                    if control_mode is ControlMode.EE_POS:
                        quat = Rotation.from_rotvec(rot).as_quat()
                        action = rc.ee_pos_action(dpos_or_pos, quat, grip)
                        goal = (dpos_or_pos, quat)
                    else:
                        action = rc.ee_delta_action(dpos_or_pos, rot, grip)
                    committed.append(np.concatenate([dpos_or_pos, rot]))
                    if slow:
                        ep.slow_steps += 1
                    dispatcher.send(action, slow_fps if slow else fast_fps)

                    if dataset is not None:
                        rc.add_frame(dataset, obs, action, args.task, controller.cameras)

                if video_dir is not None:
                    for cam, img in m.images.items():
                        rc.write_video_frame(writers, video_dir,
                                             f"{POLICY}_ep{ep.episode:03d}",
                                             o_fps, cam, img, video_frames)
                    video_frames += 1

                prev_index = current_index
                prev_chunk = chunk
                # The reference is what this plan is ABOUT to execute next, sliced at
                # the post-execution index exactly as upstream does.
                ref = chunk[current_index:current_index + t_f] if eag else None

                # Hold the obs cadence. Normally a no-op: execute_n steps at
                # fast_fps already exceed one obs period, and the per-step sleeps
                # happen in the dispatcher. This only pads a degenerately short plan
                # so a stub or truncated chunk cannot spin the camera reads flat out.
                spare = 1.0 / o_fps - (time.perf_counter() - t_infer)
                if spare > 0:
                    time.sleep(spare)
        finally:
            for w in writers.values():
                w.release()

    return episode_fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(p, policy_name=POLICY)
    p.add_argument("--control-mode", choices=("auto", "ee_delta", "ee_pos"), default="auto",
                   help="auto resolves it from the checkpoint's train.action_keys")
    p.add_argument("--slow-fps", type=float, default=None,
                   help="goal-push rate on precision steps; default baselines.exec.slow_fps")
    p.add_argument("--no-precision", action="store_true",
                   help="ignore the precision label and run at a single rate")
    p.add_argument("--no-eag", action="store_true", help="disable error-adaptive guidance")
    args = p.parse_args()

    if args.repo_id and not args.output_dir:
        p.error("--output-dir is required when --repo-id is set")

    logging.basicConfig(level=logging.INFO, force=True)
    port = args.port or int(fc.policy("baselines.zmq.sail_port"))
    client = PolicyClient(port, int(fc.policy("baselines.zmq.recv_timeout_ms")), args.host)
    meta = client.meta()
    if meta.get("backend") != POLICY:
        p.error(f"server on port {port} is a {meta.get('backend')!r} server, not sail")
    logger.info("checkpoint: %s", meta)

    control_mode = resolve_control_mode(meta, args.control_mode)
    metrics = rc.RunMetrics(policy=POLICY, header={
        "ckpt": args.ckpt,
        "ckpt_sha256": rc.sha256(args.ckpt),
        "rig": args.rig,
        "control_mode": control_mode.value,
        "control_mode_source": args.control_mode,
        "exec_fps": args.exec_fps or fc.policy("baselines.exec.fast_fps"),
        "slow_fps": args.slow_fps or fc.policy("baselines.exec.slow_fps"),
        "obs_fps": args.obs_fps or rc.obs_fps(),
        "checkpoint_meta": {k: v for k, v in meta.items() if k != "obs_key_shapes"},
        "argv": sys.argv,
    })

    controller = rc.build_robot(args.rig, control_mode)
    controller.connect()
    try:
        # Before homing, not inside the loop: a camera the rig lacks would
        # otherwise surface as a KeyError in the policy server mid-episode.
        shapes = rc.check_camera_coverage(meta, controller, args.allow_missing_cameras)
        rc.run_episodes(args, controller, metrics,
                        make_episode_fn(client, meta, args, control_mode, shapes))
    except KeyboardInterrupt:
        print("\r\ninterrupted\r", flush=True)
    finally:
        metrics.write(rc.metrics_path(args, POLICY))
        controller.disconnect()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
