#!/usr/bin/env python3
"""ZMQ policy server for a trained SAIL checkpoint. Runs in the `SAIL` conda env.

    conda activate SAIL
    python baselines/sail_bridge/policy_server.py --ckpt-path <CKPT.pth> --port 5556

Why a server at all: the SAIL env is python 3.9 / torch 2.1 / patched robosuite
and cannot coexist with the workspace venv that owns lerobot and the RPyC link
to the arm (baselines/README.md). B-Spline ships an equivalent server upstream;
SAIL does not, so this is it. The protocol is the same one, so
baselines/zmq_client.py serves both.

This file deliberately does NOT roll out. SAIL's own executor is robosuite-bound
(`env.getSimTimeInfo()`, `env.unwrapped.env.robots[0].controller`), so the loop
lives on the workstation in sail_bridge/rollout.py and this process only answers
"given this observation, what is the action sequence".

Requests:
    {"meta": True}                          -> capability dict
    {"reset": True}                         -> {}
    {"obs": {...},                          -> {"chunk": (N, act_dim) float32}
     "guide_actions": ndarray | None}
"""

from __future__ import annotations

import argparse
import json
import traceback

import numpy as np
import torch
import zmq

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory

# The pose half of every action this stack writes: xyz + rotvec + gripper.
# baselines/common.py:pos_rotvec_gripper is the single definition of that layout.
_POSE_ACTION_DIM = 7


def _normalize_guide_actions(actions, policy):
    """Inlined from run_trained_agent_receding_horizon.py:process_actions_for_guiding.

    Inlined rather than imported: that module pulls in robosuite and its patched
    env at import time, and this process has no simulator.

    The client sends the reference UNNORMALISED, because the normalisation stats
    live in this process and nowhere else.
    """
    stats = policy.action_normalization_stats
    device = policy.policy.device
    actions = TensorUtils.to_device(TensorUtils.to_tensor(actions), device)
    if stats is None:
        return actions
    assert len(stats.keys()) == 1, "Only one action key is supported for now"
    key = list(stats.keys())[0]
    stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(stats), device))
    return ObsUtils.normalize_dict({key: actions}, normalization_stats=stats)[key]


def _load_guide_config(path: str | None, policy):
    """Upstream's loader, verbatim in structure.

    Only CFG is live upstream -- the rollout loop `assert False`s on inpainting
    and the consistency loss -- so those are refused here with that reason rather
    than silently running unguided.
    """
    if path is None:
        return None
    guide = config_factory("guided_diffusion_policy", dic=None).guide
    with open(path, "rb") as f:
        with guide.values_unlocked():
            guide.update(json.load(f)["guiding"])
    if not guide.enabled:
        return None
    if guide.consistency_loss.enabled or guide.inpainting.enabled:
        raise ValueError(
            f"{path} selects consistency-loss or inpainting guidance. Upstream's "
            "own rollout loop asserts False on both; only CFG is implemented. Use "
            "one of SAIL/guide_template/base_cfg_weight_*.json."
        )
    if not guide.cfg.enabled:
        raise ValueError(f"{path} enables guiding but selects no guiding mode")
    fac = policy.policy.algo_config.future_action_condition
    # Upstream asserts this too: CFG conditions on a future-action reference the
    # model was never trained to accept otherwise.
    assert fac.enabled, (
        "CFG requires a checkpoint trained with algo.future_action_condition. "
        "Train with diffusion_policy_SAIL.json, or drop --guide-config."
    )
    return guide


class SAILPolicyServer:
    def __init__(self, ckpt_path: str, port: int, device: str | None = None,
                 guide_config: str | None = None, precision: str = "auto") -> None:
        torch_device = (TorchUtils.get_torch_device(try_to_use_cuda=True)
                        if device is None else torch.device(device))
        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path, device=torch_device, verbose=False
        )
        self.guide_config = _load_guide_config(guide_config, self.policy)
        self.meta = self._build_meta(precision)

        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"SAIL policy server on port {port}")
        print(json.dumps(self.meta, indent=2, default=str))

    def _build_meta(self, precision: str) -> dict:
        algo = self.policy.policy
        cfg = algo.global_config
        action_keys = list(cfg.train.action_keys)
        act_dim = int(self.ckpt_dict["shape_metadata"]["ac_dim"])
        fac = algo.algo_config.future_action_condition

        # Which action space this checkpoint speaks, and therefore whether the
        # client drives EE_DELTA or EE_POS. The checkpoint is the only authority:
        # SAIL's training template says `absolute_actions_with_precision` while
        # its shipped guide_template says `absolute_actions`.
        if precision == "auto":
            has_precision = (any("with_precision" in k for k in action_keys)
                             or act_dim == _POSE_ACTION_DIM + 1)
        else:
            has_precision = precision == "yes"
        if has_precision and act_dim != _POSE_ACTION_DIM + 1:
            raise ValueError(
                f"precision column claimed but ac_dim is {act_dim}, not "
                f"{_POSE_ACTION_DIM + 1}. Stripping the last column would amputate "
                f"the gripper channel; pass --precision-actions no."
            )
        return {
            "backend": "sail",
            "action_keys": action_keys,
            "act_dim": act_dim,
            "action_horizon": int(algo.algo_config.horizon.action_horizon),
            "prediction_horizon": int(algo.algo_config.horizon.prediction_horizon),
            "obs_key_shapes": {k: list(v) for k, v in algo.obs_key_shapes.items()},
            "n_obs_steps": int(algo.algo_config.horizon.observation_horizon),
            "precision_column": bool(has_precision),
            "fac_enabled": bool(fac.enabled),
            "fac_horizon": int(fac.horizon) if fac.enabled else 0,
            "guided": self.guide_config is not None,
        }

    def _infer(self, req: dict) -> dict:
        obs = req["obs"]
        if isinstance(obs, (list, tuple)):
            # RolloutPolicy keeps its own frame-stack queue, so only the newest
            # observation is fed; a list is accepted for protocol symmetry with
            # upstream's B-Spline server.
            obs = obs[-1]
        kwargs: dict = {}
        if self.guide_config is not None:
            kwargs["guide_config"] = self.guide_config
            ref = req.get("guide_actions")
            kwargs["guide_actions"] = (
                None if ref is None else _normalize_guide_actions(ref, self.policy)
            )
        chunk = self.policy(ob=obs, **kwargs)
        return {"chunk": np.asarray(chunk, dtype=np.float32)}

    def run(self) -> None:
        while True:
            req = self.socket.recv_pyobj()
            rep: dict = {}
            try:
                if "meta" in req:
                    rep = dict(self.meta)
                elif "reset" in req:
                    self.policy.start_episode()
                elif "obs" in req:
                    rep = self._infer(req)
            except Exception as exc:
                traceback.print_exc()
                # Reply rather than dying: a REP socket that never answers leaves
                # the client blocked until its timeout, with no reason to report.
                rep = {"error": f"{type(exc).__name__}: {exc}"}
            self.socket.send_pyobj(rep)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True, help="trained SAIL checkpoint (.pth)")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--device", default=None, help="default: cuda if available")
    p.add_argument("--guide-config", default=None,
                   help="SAIL/guide_template/base_cfg_weight_*.json; enables EAG")
    p.add_argument("--precision-actions", choices=("auto", "yes", "no"), default="auto",
                   help="whether the last action column is a precision label")
    args = p.parse_args()
    SAILPolicyServer(args.ckpt_path, args.port, args.device,
                     args.guide_config, args.precision_actions).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
