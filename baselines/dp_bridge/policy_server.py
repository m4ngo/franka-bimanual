#!/usr/bin/env python3
"""ZMQ policy server for our diffusion base trained on a LIBERO sim dataset.

    .venv/bin/python baselines/dp_bridge/policy_server.py \
        --ckpt-path <run>/checkpoints/last/pretrained_model --port 5557

Why a server: the checkpoint is lerobot 0.5.1's, and multi-fast/.venv -- the
only interpreter with robosuite and libero -- pins lerobot 0.1.0 for openpi. So
the policy runs here, in the workspace venv, and the LIBERO side
(baselines/libero_bridge/rollout.py, multi-fast's lerobot_dp base) asks it
for chunks over the protocol the SAIL and B-Spline servers speak.

Inference is baselines/lerobot_inference.ChunkPolicy, the same code
residual_wrapper's BasePolicy plans with on the arm. The server is stateless:
every request carries its own window of consecutive control-step frames, oldest
first, and the policy's queue is refilled from it. A client that keeps several
envs (multi-fast's vectorised RL) therefore needs no per-env session here, and
asking twice about the same state answers twice from that state.

A frame is LIBERO's raw keys: robot0_eef_pos, robot0_eef_quat,
robot0_gripper_qpos, and each camera as the HWC uint8 frame the converter
stored (sim_env.policy_frame).

Requests:
    {"meta": True}                     -> capability dict
    {"reset": True}                    -> {}
    {"obs": [frame, ...]}              -> {"chunk": (T, act_dim) float32}
    {"batch": [[frame, ...], ...]}     -> {"chunks": (B, T, act_dim) float32}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import zmq

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from baselines.lerobot_inference import ChunkPolicy  # noqa: E402
from baselines.libero_bridge.sim_env import CAMERAS, DP_STATE_KEYS  # noqa: E402

STATE_DIM = sum(n for _, n in DP_STATE_KEYS)


def training_dataset(ckpt: Path) -> tuple[str | None, dict]:
    """(repo id, the dataset's source.json) from the run's train_config.json."""
    cfg_path = ckpt / "train_config.json"
    if not cfg_path.is_file():
        return None, {}
    data = json.loads(cfg_path.read_text()).get("dataset", {})
    repo_id = data.get("repo_id")
    root = data.get("root")
    if root is None and repo_id:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        root = HF_LEROBOT_HOME / repo_id
    source = Path(root).expanduser() / "source.json" if root else None
    return repo_id, (json.loads(source.read_text()) if source and source.is_file() else {})


class DPPolicyServer:
    def __init__(self, ckpt_path: str, port: int, device: str = "cuda",
                 amp: str | None = None) -> None:
        self.policy = ChunkPolicy(ckpt_path, device=device, amp=amp)
        cfg = self.policy.config
        state = cfg.input_features.get("observation.state")
        if state is None or int(state.shape[0]) != STATE_DIM:
            raise ValueError(f"{ckpt_path} reads observation.state of shape "
                             f"{None if state is None else state.shape}; this server builds "
                             f"{STATE_DIM} from {[k for k, _ in DP_STATE_KEYS]}")
        self.cameras = [k.removeprefix("observation.images.") for k in cfg.image_features]
        known = {dst for _, dst in CAMERAS}
        if not set(self.cameras) <= known:
            raise ValueError(f"{ckpt_path} reads cameras {self.cameras}; LIBERO renders {sorted(known)}")
        self.meta = self._build_meta(Path(ckpt_path))
        self._warm_up()

        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"dp policy server on port {port}")
        print(json.dumps(self.meta, indent=2, default=str), flush=True)

    def _build_meta(self, ckpt: Path) -> dict:
        cfg = self.policy.config
        repo_id, source = training_dataset(ckpt)
        shapes = {k: [n] for k, n in DP_STATE_KEYS}
        for cam in self.cameras:
            h, w, c = (int(x) for x in self._image_shape(cam))
            shapes[cam] = [c, h, w]
        return {
            "backend": "dp",
            "checkpoint": str(ckpt),
            "act_dim": int(cfg.output_features["action"].shape[0]),
            "horizon": int(cfg.horizon),
            "n_action_steps": int(cfg.n_action_steps),
            "n_obs_steps": self.policy.n_obs_steps,
            "frame_stack": self.policy.n_obs_steps,
            "obs_key_shapes": shapes,
            "train_dataset": source.get("source_repo_id") or repo_id,
            "comparison_env": source.get("comparison_env"),
            "render_resolution": source.get("render_resolution"),
            "source_dataset": source.get("source_dataset"),
        }

    def _image_shape(self, cam: str) -> tuple[int, int, int]:
        """(H, W, C) as stored, from the checkpoint's (C, H, W) input feature."""
        c, h, w = self.policy.config.input_features[f"observation.images.{cam}"].shape
        return h, w, c

    def _format(self, frame: dict) -> dict:
        missing = [k for k, _ in DP_STATE_KEYS if k not in frame]
        missing += [c for c in self.cameras if c not in frame]
        if missing:
            raise KeyError(f"frame lacks {missing}")
        out = {"observation.state": np.concatenate(
            [np.asarray(frame[k], dtype=np.float32).reshape(n) for k, n in DP_STATE_KEYS])}
        for cam in self.cameras:
            img = np.asarray(frame[cam])
            if img.dtype != np.uint8 or img.shape != self._image_shape(cam):
                raise ValueError(f"{cam}: expected uint8 {self._image_shape(cam)} (HWC, as the "
                                 f"converter stored it), got {img.dtype} {img.shape}")
            out[f"observation.images.{cam}"] = img
        return out

    def _chunk(self, frames: list[dict]) -> np.ndarray:
        return self.policy.infer_frames([self._format(f) for f in frames]).astype(np.float32)

    def _warm_up(self) -> None:
        """One inference on zeros before the socket opens, so the first real request
        does not pay CUDA's kernel loading inside the client's recv timeout."""
        frame = {k: np.zeros(n, np.float32) for k, n in DP_STATE_KEYS}
        for cam in self.cameras:
            frame[cam] = np.zeros(self._image_shape(cam), np.uint8)
        t0 = time.perf_counter()
        self._chunk([frame] * self.policy.n_obs_steps)
        print(f"warm-up inference took {time.perf_counter() - t0:.1f}s", flush=True)

    def run(self) -> None:
        while True:
            req = self.socket.recv_pyobj()
            rep: dict = {}
            try:
                if "meta" in req:
                    rep = dict(self.meta)
                elif "reset" in req:
                    self.policy.reset()
                elif "obs" in req:
                    obs = req["obs"]
                    rep = {"chunk": self._chunk(list(obs) if isinstance(obs, (list, tuple)) else [obs])}
                elif "batch" in req:
                    rep = {"chunks": np.stack([self._chunk(list(w)) for w in req["batch"]])}
            except Exception as exc:
                traceback.print_exc()
                # Reply rather than die: an unanswered REP socket blocks the client.
                rep = {"error": f"{type(exc).__name__}: {exc}"}
            self.socket.send_pyobj(rep)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True,
                   help="a lerobot pretrained_model/ directory (EMA weights)")
    p.add_argument("--port", type=int, default=5557)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", default=None, choices=("fp16", "bf16", "none"))
    args = p.parse_args()
    DPPolicyServer(args.ckpt_path, args.port, args.device, args.amp).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
