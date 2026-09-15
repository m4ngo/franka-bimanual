#!/usr/bin/env python3
"""Meta-handshake wrapper around upstream's B-Spline policy server.

    conda activate robodiff
    python baselines/bspline_bridge/policy_server.py --ckpt-path <CKPT> --port 5555

Upstream's `policy_server_bspline.py` answers `{"reset": ...}` and `{"obs": ...}`
and replies `{}` to anything else, so it cannot tell the client what the
checkpoint expects. This subclasses it -- it is a submodule file and we do not
edit it -- and adds one request:

    {"meta": True} -> {"backend": "bspline", act_dim, degree, n_obs_steps,
                       horizon, action_format, obs_key_shapes, precision_column,
                       train_dataset, training_hdf5}

Two things need it. `obs_key_shapes` lets the client resize camera frames on its
own side rather than shipping full-resolution frames over the wire every tick,
and `degree` has to match the checkpoint or scipy reconstructs the wrong spline
from the returned parameters.

Everything else -- the spline prediction, the obs conversion, the resize
fallback -- is upstream's own code, reached by importing its script rather than
copying it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baselines.run_record import read_source_repo_id  # noqa: E402  stdlib-only

_HERE = Path(__file__).resolve()
_UPSTREAM = (_HERE.parent.parent / "bspline_policy" / "bspline_policy"
             / "bspline_policy" / "scripts" / "policy_server_bspline.py")


def _load_upstream():
    """Import upstream's server script by path.

    Not a package import: it lives under `scripts/` and sets up its own
    sys.path for `bspline_policy` and `diffusion_policy` at import time, which
    is exactly the setup we want to inherit.
    """
    if not _UPSTREAM.is_file():
        raise FileNotFoundError(
            f"{_UPSTREAM} is missing. The bspline_policy submodule is not "
            "checked out: git submodule update --init baselines/bspline_policy"
        )
    spec = importlib.util.spec_from_file_location("policy_server_bspline", _UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_cfg(ckpt_path: str):
    """The hydra config a checkpoint was trained under, or None.

    Upstream's BSplinePolicy reads `payload["cfg"]` and keeps none of it, so the
    checkpoint is opened a second time here -- CPU-mapped, at startup only, and
    cheap next to loading the model onto the GPU. Two things need it: the
    training dataset for provenance, and `n_obs_steps`, which the client's
    observation window has to match.
    """
    import dill
    import torch

    try:
        with open(ckpt_path, "rb") as f:
            return torch.load(f, pickle_module=dill, map_location="cpu")["cfg"]
    except Exception as exc:
        print(f"could not read the checkpoint's cfg: {exc!r}")
        return None


def _cfg_lookup(cfg, *getters):
    for getter in getters:
        try:
            value = getter(cfg)
        except Exception:
            continue
        if value is not None and value != "":
            return value
    return None


def _training_hdf5(cfg, ckpt_path: str) -> str | None:
    """The HDF5 behind the checkpoint's hydra config, if it can be found.

    `task.dataset_path` is relative to wherever training ran (upstream's configs
    say `../diffusion_policy/data/x.hdf5`), so it is tried against each ancestor
    of the checkpoint as well as the cwd. A path that resolves nowhere is not an
    error: the stamp is simply unavailable and the rollout asks for
    --train-dataset.
    """
    raw = _cfg_lookup(cfg, lambda c: c.task.dataset_path,
                      lambda c: c.task.dataset.dataset_path,
                      lambda c: c.dataset_path)
    if not raw:
        return None
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    bases = [Path.cwd(), *Path(ckpt_path).expanduser().resolve().parents]
    for base in bases:
        if (base / candidate).is_file():
            return str((base / candidate).resolve())
    return str(candidate)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--n-obs-steps", type=int, default=None,
                   help="observation window; default: what the checkpoint was trained with")
    p.add_argument("--degree", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    up = _load_upstream()
    cfg = _checkpoint_cfg(args.ckpt_path)
    # The model was trained on a fixed observation window; feeding it another
    # length is a silent shape change, not a knob.
    trained_n_obs = _cfg_lookup(cfg, lambda c: int(c.n_obs_steps))
    n_obs_steps = args.n_obs_steps if args.n_obs_steps is not None else trained_n_obs
    if n_obs_steps is None:
        p.error("the checkpoint does not say its n_obs_steps; pass --n-obs-steps")
    if trained_n_obs is not None and n_obs_steps != trained_n_obs:
        print(f"WARNING: --n-obs-steps {n_obs_steps} but the checkpoint trained with "
              f"{trained_n_obs}; the policy will see a window it was not trained on")

    policy = up.BSplinePolicy(args.ckpt_path, degree=args.degree, device=args.device)
    wrapper = up.PolicyWrapper(policy, n_obs_steps=n_obs_steps)

    action_meta = policy.get_action_metadata()
    # cfg.task.dataset_path is the HDF5 the replay buffer was built from; it
    # carries the training dataset's id when it came through our converter.
    training_hdf5 = _training_hdf5(cfg, args.ckpt_path)
    train_dataset = read_source_repo_id(training_hdf5)
    meta = {
        "backend": "bspline",
        "act_dim": int(action_meta["action_dim"]),
        "degree": int(policy.degree),
        "n_obs_steps": int(n_obs_steps),
        "n_action_steps": int(policy.n_action_steps),
        "horizon": _cfg_lookup(cfg, lambda c: int(c.horizon)),
        "action_format": action_meta.get("action_format"),
        "relative_knots": bool(action_meta.get("relative_knots")),
        # A B-Spline action carries no precision label; the field exists so one
        # client serves both backends.
        "precision_column": False,
        "train_dataset": train_dataset,
        "training_hdf5": str(training_hdf5) if training_hdf5 else None,
        "obs_key_shapes": {k: list(v["shape"])
                           for k, v in policy.obs_shape_meta.items()},
    }

    class Server(up.PolicyServer):
        def run(self) -> None:
            print(f"B-Spline policy server (meta-wrapped) on port {args.port}")
            print(meta)
            while True:
                req = self.socket.recv_pyobj()
                rep: dict = {}
                try:
                    if "meta" in req:
                        rep = dict(meta)
                    elif "reset" in req:
                        self.policy.reset()
                        print("Policy has been reset")
                    elif "obs" in req:
                        rep["bspline"] = self.step(req["obs"])
                        rep["bspline_meta"] = self.policy.policy.get_action_metadata()
                except Exception as exc:
                    traceback.print_exc()
                    # Reply rather than dying: a REP socket that never answers
                    # leaves the client blocked with no reason to report.
                    rep = {"error": f"{type(exc).__name__}: {exc}"}
                self.socket.send_pyobj(rep)

    Server(wrapper, port=args.port).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
