"""Overlay policy-input point clouds from one or more run_residual npz dumps.

Each npz is produced by run_residual.py alongside the episode HTML
(episode_policy_pcd.npz) and holds the exact per-inference clouds the residual
network received. Overlaying dumps from different rollouts (e.g. real vs sim)
makes distribution gaps visible directly in the network's input space.

Usage:
    python plot_policy_pcd.py real.npz sim.npz --labels real sim -o compare.html
    python plot_policy_pcd.py episode_policy_pcd.npz          # single rollout
"""

import argparse
import os

import numpy as np

from viz import save_policy_pcd_html


def _load_events(path: str) -> tuple[list[dict], bool, float]:
    data = np.load(path)
    events = [
        {"step": int(step), "pcd": pcd}
        for step, pcd in zip(data["steps"], data["pcds"])
    ]
    return events, bool(data["center_on_eef"]), float(data["fps"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", nargs="+", help="policy_pcd npz dump(s) from run_residual.py")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="One label per npz (default: file basenames)")
    parser.add_argument("-o", "--output", default="policy_pcd_compare.html",
                        help="Output HTML path")
    parser.add_argument("--stride", type=int, default=1,
                        help="Animate every Nth inference event")
    args = parser.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(f))[0] for f in args.npz]
    if len(labels) != len(args.npz):
        parser.error(f"got {len(labels)} labels for {len(args.npz)} npz files")

    series: list[tuple[str, list[dict]]] = []
    centered_flags: list[bool] = []
    fps = 20.0
    for label, path in zip(labels, args.npz):
        events, centered, fps = _load_events(path)
        series.append((label, events))
        centered_flags.append(centered)
        print(f"{label}: {len(events)} inference events from {path}"
              f" ({'EE-centered' if centered else 'world frame'})")

    if len(set(centered_flags)) > 1:
        print("WARNING: mixing EE-centered and world-frame dumps — "
              "clouds are in different frames and will not align")

    frame = "EE-centered" if centered_flags[0] else "world frame"
    save_policy_pcd_html(
        series, args.output,
        title=f"network input ({frame}) — {' vs '.join(labels)}",
        frame_stride=args.stride, fps=fps,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
