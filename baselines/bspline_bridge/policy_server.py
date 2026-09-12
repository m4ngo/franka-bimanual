#!/usr/bin/env python3
"""Meta-handshake wrapper around upstream's B-Spline policy server.

    conda activate robodiff
    python baselines/bspline_bridge/policy_server.py --ckpt-path <CKPT> --port 5555

Upstream's `policy_server_bspline.py` answers `{"reset": ...}` and `{"obs": ...}`
and replies `{}` to anything else, so it cannot tell the client what the
checkpoint expects. This subclasses it -- it is a submodule file and we do not
edit it -- and adds one request:

    {"meta": True} -> {"backend": "bspline", act_dim, degree, n_obs_steps,
                       action_format, obs_key_shapes, precision_column}

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--n-obs-steps", type=int, default=2)
    p.add_argument("--degree", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    up = _load_upstream()
    policy = up.BSplinePolicy(args.ckpt_path, degree=args.degree, device=args.device)
    wrapper = up.PolicyWrapper(policy, n_obs_steps=args.n_obs_steps)

    action_meta = policy.get_action_metadata()
    meta = {
        "backend": "bspline",
        "act_dim": int(action_meta["action_dim"]),
        "degree": int(policy.degree),
        "n_obs_steps": int(args.n_obs_steps),
        "n_action_steps": int(policy.n_action_steps),
        "action_format": action_meta.get("action_format"),
        "relative_knots": bool(action_meta.get("relative_knots")),
        # A B-Spline action carries no precision label; the field exists so one
        # client serves both backends.
        "precision_column": False,
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
