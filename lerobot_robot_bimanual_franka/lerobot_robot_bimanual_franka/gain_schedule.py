"""Time-based kp/kd gain-action schedules for gain-excitation recordings.

A policy emits a normalised gain action every step and both stacks remap it the
same way (`resolve_gains` here, `LIBEROObservationWrapper.step` in sim). These
schedules put that channel under test: the excitation runs while the gains move,
and a sim replay is fed the same per-step actions.

Quadrature, one frequency: a_kp rides sin and a_kd rides cos, so the (kp, kd)
point traces a circle and every quadrant of the gain plane is visited each
cycle. A sinusoid rather than steps because a gain step is a torque step, and
the FR3's 800 Nm/s torque rate limit (which sim lacks) would then dominate the
comparison instead of the gain response.

Numpy only; `remap_constants` is the one function that reads config/, so the
schedule itself stays importable anywhere.
"""

from __future__ import annotations

import numpy as np

DEFAULT_AMP = 0.3
DEFAULT_FREQ_HZ = 0.25
DEFAULT_RAMP_S = 1.0
DEFAULT_PHASE_KD_RAD = float(np.pi / 2)


def quadrature_schedule(t_s: np.ndarray, amp_kp: float, amp_kd: float,
                        freq_hz: float = DEFAULT_FREQ_HZ, ramp_s: float = DEFAULT_RAMP_S,
                        phase_kd_rad: float = DEFAULT_PHASE_KD_RAD,
                        kp0: float = 0.0, kd0: float = 0.0) -> np.ndarray:
    """(T, 2) normalised [a_kp, a_kd] on the grid `t_s`, clipped to [-1, 1].

    a_kp = kp0 + amp_kp * r(t) * sin(2 pi f t)
    a_kd = kd0 + amp_kd * r(t) * sin(2 pi f t + phase_kd)

    r(t) is a half-cosine ramp-in over `ramp_s`, so the modulation starts at the
    centre (kp0, kd0) rather than jumping on the first tick. Zero amplitudes give
    a constant (kp0, kd0) schedule, i.e. today's behaviour.
    """
    t = np.asarray(t_s, dtype=np.float64)
    if ramp_s > 0.0:
        r = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / ramp_s, 0.0, 1.0)))
    else:
        r = np.ones_like(t)
    w = 2.0 * np.pi * float(freq_hz) * t
    a_kp = float(kp0) + float(amp_kp) * r * np.sin(w)
    a_kd = float(kd0) + float(amp_kd) * r * np.sin(w + float(phase_kd_rad))
    return np.clip(np.stack([a_kp, a_kd], axis=1), -1.0, 1.0)


def describe(amp_kp: float, amp_kd: float, freq_hz: float, ramp_s: float,
             phase_kd_rad: float, kp0: float = 0.0, kd0: float = 0.0) -> dict:
    """The parameters as a JSON-able dict, for run metadata and file attrs."""
    return {"kind": "quadrature", "amp_kp": float(amp_kp), "amp_kd": float(amp_kd),
            "freq_hz": float(freq_hz), "ramp_s": float(ramp_s),
            "phase_kd_rad": float(phase_kd_rad), "kp0": float(kp0), "kd0": float(kd0)}


def varies(schedule: dict | None) -> bool:
    return bool(schedule) and (schedule["amp_kp"] != 0.0 or schedule["amp_kd"] != 0.0)


def remap_constants() -> dict:
    """What the rig's action -> gain map is, from config/, for the sim to check
    against its own before trusting a recording's gain actions."""
    import franka_config as fc  # noqa: PLC0415

    return {
        "osc_base_kp": float(fc.control("torque.osc.default_kp")),
        "osc_default_damping_ratio": float(fc.control("torque.osc.default_damping_ratio")),
        "gain_exp_base": float(fc.control("torque.osc.gain_exp_base")),
        "kp_limits": [float(v) for v in fc.control("torque.osc.kp_limits")],
        "damping_ratio_limits": [float(v) for v in fc.control("torque.osc.damping_ratio_limits")],
        "tuning_gain_scales": {
            k: [float(v) for v in np.broadcast_to(np.asarray(fc.control(f"tuning.{k}"), dtype=np.float64), (3,))]
            for k in ("kp_pos_scale", "kp_ori_scale", "kd_pos_scale", "kd_ori_scale")
        },
    }
