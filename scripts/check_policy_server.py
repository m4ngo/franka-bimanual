#!/usr/bin/env python3
"""Ask a running baseline policy server for one plan and check it is sane.

    python scripts/check_policy_server.py sail    --port 5556 [--eag]
    python scripts/check_policy_server.py bspline --port 5555

The offline harness (check_baseline_rollout_offline.py) drives the rollout loops
against FAKE servers, so it can never tell whether a real checkpoint, loaded by
the real server in its own venv, answers with the shape and units the loop
expects. This can: it performs the handshake the rollout performs, sends
synthetic observations in the exact key layout rollout_common builds, and
checks the reply -- without an arm. Run it before the first rollout of any new
checkpoint; it takes seconds.

What it checks: the handshake names the right backend and the training dataset;
SAIL answers with an (action_horizon, act_dim) chunk on every request (frame
stacking and return_action_sequence are the server's job, and either missing
shows up here as a shape error); with --eag, a guided request works too;
B-Spline answers with spline parameters that scipy can rebuild and that decode
to a unit quaternion. Exits PASS/FAIL.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import franka_config as fc  # noqa: E402

from baselines import rollout_common as rc  # noqa: E402
from baselines.zmq_client import PolicyClient  # noqa: E402


class Results:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.n = 0

    def check(self, ok: bool, what: str) -> bool:
        self.n += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            self.fails.append(what)
        return ok


def synthetic_measured(shapes: dict, rng: np.random.Generator) -> rc.Measured:
    """A plausible observation: home-ish pose, noise images of the right size."""
    images = {}
    for key, shape in shapes.items():
        if key.endswith("_image"):
            h, w = int(shape[-2]), int(shape[-1])
            images[key[: -len("_image")]] = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return rc.Measured(
        q=np.zeros(rc.NUM_JOINTS), pos=np.array([0.4, 0.0, 0.3]),
        quat_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), gripper=0.5, images=images,
    )


def check_sail(client: PolicyClient, meta: dict, args, res: Results) -> None:
    rng = np.random.default_rng(0)
    shapes = dict(meta["obs_key_shapes"])
    horizon, act_dim = int(meta["action_horizon"]), int(meta["act_dim"])
    stack = int(meta.get("frame_stack", 1))
    res.check("frame_stack" in meta and "obs_keys" in meta,
              "meta carries frame_stack and obs_keys (server predates the fix otherwise)")
    client.reset()
    chunks = []
    for i in range(max(3, stack + 1)):
        obs = rc.sail_obs(synthetic_measured(shapes, rng), shapes)
        t0 = time.perf_counter()
        rep = client.request({"obs": obs, "guide_actions": None})
        dt = time.perf_counter() - t0
        if not res.check("error" not in rep, f"request {i}: no error ({rep.get('error', '')})"):
            return
        chunk = np.asarray(rep.get("chunk"))
        chunks.append(chunk)
        res.check(chunk.shape == (horizon, act_dim),
                  f"request {i}: chunk {chunk.shape} == (action_horizon {horizon}, act_dim {act_dim}) "
                  f"in {dt * 1000:.0f} ms")
        res.check(bool(np.all(np.isfinite(chunk))), f"request {i}: chunk is finite")
    if meta.get("precision_column"):
        labels = chunks[-1][:, -1]
        res.check(bool(np.all((labels > -0.5) & (labels < 1.5))),
                  f"precision column in [0, 1] (min {labels.min():.2f} max {labels.max():.2f})")
    pos = chunks[-1][:, :3]
    print(f"  info first row pos {np.round(chunks[-1][0, :3], 3)} gripper {chunks[-1][0, 6]:.2f}; "
          f"pos span {np.round(pos.max(0) - pos.min(0), 3)} m over the chunk")
    if args.eag:
        if not res.check(bool(meta.get("guided")), "server started with --guide-config (needed for --eag)"):
            return
        t_f = int(meta.get("fac_horizon") or 0)
        inf_delay = int(fc.policy("baselines.sail.inf_delay"))
        ref = chunks[-1][inf_delay:inf_delay + t_f]
        obs = rc.sail_obs(synthetic_measured(shapes, rng), shapes)
        rep = client.request({"obs": obs, "guide_actions": ref})
        if res.check("error" not in rep, f"guided request: no error ({rep.get('error', '')})"):
            chunk = np.asarray(rep["chunk"])
            res.check(chunk.shape == (horizon, act_dim), f"guided request: chunk {chunk.shape}")
        # A clamped +-1 output round-trips to just past +-1; the server must not assert on it.
        sat = np.array(ref, dtype=np.float64)
        sat[:, -2:] = 1.001
        rep = client.request({"obs": obs, "guide_actions": sat})
        res.check("error" not in rep,
                  f"guided request with a saturated reference: no error ({rep.get('error', '')})")


def check_bspline(client: PolicyClient, meta: dict, args, res: Results) -> None:
    from scipy.interpolate import BSpline

    from baselines.bspline_bridge.rollout import decode
    from baselines.bspline_bridge.spline_plan import safer_knots

    rng = np.random.default_rng(0)
    shapes = dict(meta["obs_key_shapes"])
    act_dim, degree, n_obs = int(meta["act_dim"]), int(meta["degree"]), int(meta["n_obs_steps"])
    res.check(act_dim == 10, f"act_dim {act_dim} == 10 (pos + rot6d + gripper)")
    client.reset()
    for i in range(2):
        seq = [rc.bspline_obs(synthetic_measured(shapes, rng), shapes) for _ in range(n_obs)]
        t0 = time.perf_counter()
        rep = client.request({"obs": seq})
        dt = time.perf_counter() - t0
        if not res.check("error" not in rep, f"request {i}: no error ({rep.get('error', '')})"):
            return
        raw = rep.get("bspline")
        if not res.check(raw is not None, f"request {i}: reply carries 'bspline' in {dt * 1000:.0f} ms"):
            return
        b = np.asarray(raw, dtype=np.float64)
        res.check(b.ndim == 2 and b.shape[1] == act_dim + 1,
                  f"request {i}: parameters {b.shape} == (horizon, act_dim + 1)")
        horizon = meta.get("horizon")
        if horizon:
            res.check(b.shape[0] == int(horizon), f"request {i}: horizon {b.shape[0]} == {horizon}")
        knots = safer_knots(b[:, 0])
        try:
            spline = BSpline(t=knots, c=b[:, 1:][: -(degree + 1)], k=degree)
            t_min, t_max = float(spline.t[degree]), float(spline.t[-degree - 1])
            samples = spline(np.linspace(t_min, t_max, 8))
            res.check(bool(np.all(np.isfinite(samples))) and samples.shape == (8, act_dim),
                      f"request {i}: scipy rebuilds the spline, t in [{t_min:.2f}, {t_max:.2f}]")
            pos_, quat, grip = decode(samples[0])
            res.check(abs(np.linalg.norm(quat) - 1.0) < 1e-6, f"request {i}: decoded quaternion is unit")
            print(f"  info t=min pos {np.round(pos_, 3)} gripper {grip:.2f}; "
                  f"pos span {np.round(samples[:, :3].max(0) - samples[:, :3].min(0), 3)} m over the plan")
        except Exception as exc:
            res.check(False, f"request {i}: spline reconstruction raised {exc!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("backend", choices=("sail", "bspline"))
    p.add_argument("--port", type=int, default=None, help="default: policy.yaml's port for the backend")
    p.add_argument("--host", default="localhost")
    p.add_argument("--eag", action="store_true", help="sail: also send a guided request")
    args = p.parse_args()

    port = args.port or int(fc.policy(f"baselines.zmq.{args.backend}_port"))
    # Generous on purpose: this is a preflight, and a slow first inference is a
    # finding to report (see the latency it prints), not a reason to time out.
    client = PolicyClient(port, 120_000, args.host)
    res = Results()
    print(f"[{args.backend}] handshake on {args.host}:{port}")
    try:
        meta = client.meta()
    except Exception as exc:
        print(f"  FAIL no meta handshake: {exc}")
        return 1
    for k, v in meta.items():
        print(f"    {k:18s} {v}")
    res.check(meta.get("backend") == args.backend, f"backend is {args.backend!r}")
    res.check(bool(meta.get("train_dataset")),
              f"train_dataset known ({meta.get('train_dataset')!r}); the rollout will otherwise "
              "need --train-dataset")
    print(f"[{args.backend}] inference")
    try:
        (check_sail if args.backend == "sail" else check_bspline)(client, meta, args, res)
    finally:
        client.close()
    print(f"\n{'PASS' if not res.fails else 'FAIL'}: {res.n - len(res.fails)}/{res.n} checks")
    for f in res.fails:
        print(f"  - {f}")
    return 0 if not res.fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
