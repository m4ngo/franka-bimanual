"""End-effector force bookkeeping shared by the sim and real rollouts.

Both record the force and torque at the end effector once per control step and
summarise |F| the way multi-fast's eval_fast.py does. The sim reads robosuite's
wrist sensor (libero_bridge/sim_env.py); the arm reads libfranka's estimated
external wrench, O_F_ext_hat_K (BimanualFranka.last_ee_wrench).

Numpy only, so it imports in multi-fast/.venv and the workspace venv alike.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("baselines.force_log")


def force_stats(force) -> dict | None:
    """eval_fast.py's per-episode |F| statistics; None for an episode with no steps.

    eval_fast masks every step after success. A rollout here ends at success
    (or at the operator's verdict), so the series needs no mask.
    """
    norms = np.linalg.norm(np.asarray(force, dtype=np.float64).reshape(-1, 3), axis=-1)
    if not len(norms):
        return None
    return {"mean": round(float(norms.mean()), 4),
            "median": round(float(np.median(norms)), 4),
            "p95": round(float(np.quantile(norms, 0.95)), 4),
            "max": round(float(norms.max()), 4)}


def note(stats: dict | None) -> str:
    """`, |F| mean/p95/max 3.1/12/40 N` for an end-of-episode line, or nothing."""
    if not stats:
        return ""
    return f", |F| mean/p95/max {stats['mean']:.1f}/{stats['p95']:.0f}/{stats['max']:.0f} N"


def force_attrs(arrays: dict) -> dict:
    """Flat episode-HDF5 attrs for an episode's `ee_force` dataset (ee_force_mean_n,
    ee_force_median_n, ee_force_p95_n, ee_force_max_n); empty when none was recorded."""
    stats = force_stats(arrays["ee_force"]) if "ee_force" in arrays else None
    return {f"ee_force_{k}_n": v for k, v in (stats or {}).items()}


class ForceLog:
    """Every episode's per-step force and torque for one run, as force_profiles.npz.

    Keyed by episode index: ee_force_003 and ee_torque_003 are episode 3's
    (steps, 3) vectors. A sim run also stores `control_freq`; a real run stores
    time_003 instead, each sample's seconds since the episode started, because
    its rate is the wall clock's.
    """

    def __init__(self, path: Path, control_freq: float | None = None) -> None:
        self.path = Path(path)
        self.arrays: dict[str, np.ndarray] = {}
        if control_freq is not None:
            self.arrays["control_freq"] = np.float64(control_freq)
        self._warned = False

    def add(self, episode: int, force, torque, t=None) -> dict | None:
        """Store one episode and return its `ee_force_n`; None if it has no samples.

        The file is rewritten whole each time, so an interrupted run keeps every
        episode that finished.
        """
        force = np.asarray(force, dtype=np.float64).reshape(-1, 3)
        if not len(force):
            return None
        self.arrays[f"ee_force_{episode:03d}"] = force.astype(np.float32)
        self.arrays[f"ee_torque_{episode:03d}"] = (
            np.asarray(torque, dtype=np.float32).reshape(-1, 3))
        if t is not None:
            self.arrays[f"time_{episode:03d}"] = np.asarray(t, dtype=np.float64).reshape(-1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **self.arrays)
        tmp.replace(self.path)
        return force_stats(force)

    def add_trace(self, episode: int, trace: "WrenchTrace") -> dict | None:
        if trace.missing and not trace.force and not self._warned:
            self._warned = True
            logger.warning(
                "EE force not recorded: arm %r's NUC server does not publish the "
                "wrench yet. Re-run scripts/deploy_nuc_server.sh for that NUC.",
                trace.arm)
        return self.add(episode, trace.force, trace.torque, trace.t)


class WrenchTrace:
    """One real episode's wrench samples, one per goal sent to the arm.

    Sampled after send_action, whose state read is the one that goal was
    anchored on -- the arm's counterpart of eval_fast reading the sensor after
    each env step.
    """

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.t0 = time.perf_counter()
        self.force: list[np.ndarray] = []
        self.torque: list[np.ndarray] = []
        self.t: list[float] = []
        self.missing = 0

    def sample(self, controller) -> None:
        w = controller.last_ee_wrench.get(self.arm)
        if w is None:
            self.missing += 1
            return
        w = np.asarray(w, dtype=np.float64)
        self.force.append(w[:3].copy())
        self.torque.append(w[3:6].copy())
        self.t.append(time.perf_counter() - self.t0)
