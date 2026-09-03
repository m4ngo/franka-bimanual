"""panda_control's v3 / v4 excitation trajectory generators, ported.

Source: tsrobcvai/panda_control `scripts/gen_excitation_traj.py` (v3, "step5d")
and `scripts/gen_chirp_traj.py` (v4 chirp). The builders and their helpers are
lifted verbatim so a later upstream change reads as a diff; only the argparse,
sidecar loading, CSV writing and `main()` are dropped. Nothing here imports the
robot stack -- it is analytic numpy over a time grid.

Both builders take `(t_s, x_anchor, q_anchor_xyzw)` and return an ABSOLUTE goal
pose per sample. That is the interface `excite_panda.py` needs and the reason it
does not reuse `sysid.py`'s `_reference_offsets`: v3 composes its orientation as
`q_yaw_base (x) q_anchor (x) q_roll_local` -- yaw about world z, roll about EE
local z -- which no single base-frame rotvec offset can express.

Upstream evaluates these on a 1000 Hz grid and writes a CSV; they are analytic in
`t_s`, so evaluating them on the 20 Hz control grid is exact, not resampled.

Two asymmetries between the two that callers must not assume away:
  * v4 ramps out (rho(T) = 0, returns to the anchor); v3 does NOT -- its envelope
    clips at 1 and the episode ends mid-swing.
  * v3's stated amplitude is the LOW band's; `sin(...) + ratio*sin(...)` peaks at
    (1 + high_band_ratio) = 1.2x that.
"""

from __future__ import annotations

import numpy as np

# --- v4 chirp ---------------------------------------------------------------

CHIRP_F0_DEFAULT = 0.1
CHIRP_F1_DEFAULT = 0.7
# Per-axis phase offsets (6 axes, 6 evenly spaced offsets k * pi/3).
PHASE_OFFSETS = np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0,
                          np.pi, 4.0 * np.pi / 3.0, 5.0 * np.pi / 3.0])

# --- v3 two-band ------------------------------------------------------------

POS_FREQS = {
    "x": (0.15, 0.70, 0.0),                # (low, high, phase)
    "y": (0.20, 0.90, np.pi / 3.0),
    "z": (0.30, 1.10, np.pi / 4.0),
}

# Rotation bands sit between position bands to give CMA-ES a clean
# spectral fingerprint for each DOF (no aliasing with translation harmonics).
ORI_FREQS = {
    "yaw":  (0.18, 0.55, 0.0),
    "roll": (0.22, 0.65, np.pi / 5.0),
}

# --- safety conventions (from panda_control's step5b/step5d pre-flights) -----

CART_DX_PEAK_LIMIT_MPS = 0.30
ORI_DOT_PEAK_LIMIT_RPS = 0.50
ORI_TRACK_ABORT_RAD = 0.30


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (...,4) xyzw quaternion arrays."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack((x, y, z, w), axis=-1)


def _axis_angle_to_quat_xyzw(rot_vec: np.ndarray) -> np.ndarray:
    """(T,3) axis-angle -> (T,4) unit quaternion in xyzw.  Identity at theta=0."""
    theta = np.linalg.norm(rot_vec, axis=1)
    eps = 1e-9
    safe_theta = np.where(theta < eps, 1.0, theta)
    axis = rot_vec / safe_theta[:, None]
    half = 0.5 * theta
    sin_half = np.sin(half)
    cos_half = np.cos(half)
    # Force identity for near-zero rotations.
    sin_half = np.where(theta < eps, 0.0, sin_half)
    cos_half = np.where(theta < eps, 1.0, cos_half)
    q = np.zeros((rot_vec.shape[0], 4), dtype=np.float64)
    q[:, 0] = axis[:, 0] * sin_half
    q[:, 1] = axis[:, 1] * sin_half
    q[:, 2] = axis[:, 2] * sin_half
    q[:, 3] = cos_half
    return q


def _quat_from_axis_angle_z(angle: np.ndarray) -> np.ndarray:
    """Return (T,4) xyzw quaternion array for rotation about local-z by ``angle`` [rad]."""
    half = 0.5 * angle
    out = np.zeros((angle.shape[0], 4), dtype=np.float64)
    out[:, 2] = np.sin(half)
    out[:, 3] = np.cos(half)
    return out


# ---------------------------------------------------------------------------
# v4 chirp (gen_chirp_traj.py)
# ---------------------------------------------------------------------------

def _linear_ramp(t_s: np.ndarray, total_T: float, ramp_up_s: float,
                 ramp_down_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric linear ramp envelope rho(t) in [0,1] with derivative rho_dot.

    rho(0) = 0, rho(ramp_up_s) = 1, rho(T - ramp_down_s) = 1, rho(T) = 0.
    rho is piecewise linear: rising slope = 1/ramp_up_s, flat at 1, falling
    slope = -1/ramp_down_s.  rho_dot is the corresponding slope (0 in flat
    region) -- not C1 at the corners, but UR5e's chirp uses the same shape
    and the controller smooths the resulting target with its inertia + Kd.
    """
    rho = np.ones_like(t_s)
    rho_dot = np.zeros_like(t_s)
    if ramp_up_s > 0.0:
        in_up = t_s < ramp_up_s
        rho[in_up] = t_s[in_up] / ramp_up_s
        rho_dot[in_up] = 1.0 / ramp_up_s
    if ramp_down_s > 0.0 and total_T > ramp_down_s:
        t_down_start = total_T - ramp_down_s
        in_down = t_s >= t_down_start
        rho[in_down] = np.clip((total_T - t_s[in_down]) / ramp_down_s, 0.0, 1.0)
        rho_dot[in_down] = -1.0 / ramp_down_s
    rho = np.clip(rho, 0.0, 1.0)
    return rho, rho_dot


def _chirp_phase(t_s: np.ndarray, f0: float, f1: float,
                 total_T: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear chirp instantaneous phase and angular frequency.

    phi(t)   = 2*pi*(f0*t + 0.5*(f1-f0)/T * t^2)
    phi'(t)  = 2*pi*(f0 + (f1-f0)/T * t)
    """
    if total_T <= 0.0:
        return np.zeros_like(t_s), np.zeros_like(t_s)
    sweep_rate = (f1 - f0) / total_T  # Hz / s
    phi = 2.0 * np.pi * (f0 * t_s + 0.5 * sweep_rate * t_s ** 2)
    phi_dot = 2.0 * np.pi * (f0 + sweep_rate * t_s)
    return phi, phi_dot


def build_chirp_trajectory(
    t_s: np.ndarray,
    x_anchor: np.ndarray,
    q_anchor_xyzw: np.ndarray,
    *,
    f0: float = CHIRP_F0_DEFAULT,
    f1: float = CHIRP_F1_DEFAULT,
    amp_x: float = 0.10,
    amp_y: float = 0.10,
    amp_z: float = 0.15,
    amp_rx: float = 0.50,
    amp_ry: float = 0.25,
    amp_rz: float = 0.50,
    ramp_up_s: float = 2.0,
    ramp_down_s: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 6-DOF chirp reference trajectory around the given anchor pose.

    Returns:
        x_des:           (T, 3) Cartesian target positions.
        dx_des:          (T, 3) Cartesian target velocities (analytic).
        quat_des_xyzw:   (T, 4) target orientation (xyzw), anchor pre-rotated
                         by the world-frame axis-angle offset.
        rot_offsets:     (T, 3) axis-angle offset vector applied to the anchor
                         (returned for diagnostics + sidecar metadata).
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    x_anchor = np.asarray(x_anchor, dtype=np.float64).reshape(3)
    q_anchor_xyzw = np.asarray(q_anchor_xyzw, dtype=np.float64).reshape(4)
    total_T = float(t_s[-1]) if t_s.size > 0 else 0.0

    phi, phi_dot = _chirp_phase(t_s, f0, f1, total_T)
    rho, rho_dot = _linear_ramp(t_s, total_T, ramp_up_s, ramp_down_s)

    amps = np.array([amp_x, amp_y, amp_z, amp_rx, amp_ry, amp_rz], dtype=np.float64)
    pos_offsets = np.zeros((t_s.shape[0], 3), dtype=np.float64)
    dpos_offsets = np.zeros((t_s.shape[0], 3), dtype=np.float64)
    rot_offsets = np.zeros((t_s.shape[0], 3), dtype=np.float64)

    for i in range(6):
        s = np.sin(phi + PHASE_OFFSETS[i])
        c = np.cos(phi + PHASE_OFFSETS[i])
        val = amps[i] * rho * s
        dval = amps[i] * (rho_dot * s + rho * phi_dot * c)
        if i < 3:
            pos_offsets[:, i] = val
            dpos_offsets[:, i] = dval
        else:
            rot_offsets[:, i - 3] = val
            # Rotation velocity is not returned (matches UR5e -- target_quat
            # alone is enough for the OSC, and analytic angular velocity is
            # noisy near identity).

    x_des = x_anchor[None, :] + pos_offsets
    dx_des = dpos_offsets

    # Compose target quaternion: q_des = q_offset(world) (x) q_anchor
    q_offset = _axis_angle_to_quat_xyzw(rot_offsets)
    q_anchor_tiled = np.broadcast_to(q_anchor_xyzw[None, :], (t_s.shape[0], 4)).copy()
    quat_des = _quat_mul_xyzw(q_offset, q_anchor_tiled)
    quat_des = quat_des / np.clip(np.linalg.norm(quat_des, axis=1, keepdims=True), 1e-12, None)

    return x_des, dx_des, quat_des, rot_offsets


# ---------------------------------------------------------------------------
# v3 two-band / step5d (gen_excitation_traj.py)
# ---------------------------------------------------------------------------

def _half_cosine_envelope(t_s: np.ndarray, ramp_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (rho, rho_dot) where rho is a smooth 0->1 envelope.

    rho(t)  = 0.5 * (1 - cos(pi * t / T))   for 0 <= t <= T,  else 1
    rho'(t) = 0.5 * (pi / T) * sin(pi * t / T) for 0 <= t <= T, else 0

    Boundary conditions: rho(0) = rho'(0) = 0 and rho(T) = 1, rho'(T) = 0,
    so the resulting trajectory starts exactly at the anchor with zero
    velocity / zero acceleration jump, and ramp-out is smooth too.

    NOTE (ours, not upstream's): despite the docstring's "ramp-out is smooth
    too", this is a ramp-IN only -- s is clipped at 1, so rho stays at 1 for
    the rest of the episode and v3 ends mid-swing. See `ramp_out_envelope`.
    """
    if ramp_s <= 0.0:
        return np.ones_like(t_s), np.zeros_like(t_s)
    s = np.clip(t_s / ramp_s, 0.0, 1.0)
    rho = 0.5 * (1.0 - np.cos(np.pi * s))
    in_ramp = (t_s >= 0.0) & (t_s < ramp_s)
    rho_dot = np.zeros_like(t_s)
    rho_dot[in_ramp] = 0.5 * (np.pi / ramp_s) * np.sin(np.pi * s[in_ramp])
    return rho, rho_dot


def _two_band(t_s: np.ndarray, freq_lo: float, freq_hi: float, phase_lo: float,
              ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (value, derivative) of ``sin(2*pi*f_lo*t + phi) + ratio*sin(2*pi*f_hi*t)``."""
    two_pi = 2.0 * np.pi
    val = np.sin(two_pi * freq_lo * t_s + phase_lo) + ratio * np.sin(two_pi * freq_hi * t_s)
    dval = (
        two_pi * freq_lo * np.cos(two_pi * freq_lo * t_s + phase_lo)
        + ratio * two_pi * freq_hi * np.cos(two_pi * freq_hi * t_s)
    )
    return val, dval


def _build_pos_traj(
    t_s: np.ndarray,
    x_anchor: np.ndarray,
    amp_x: float,
    amp_y: float,
    amp_z: float,
    rho: np.ndarray,
    rho_dot: np.ndarray,
    high_band_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    bx, dbx = _two_band(t_s, POS_FREQS["x"][0], POS_FREQS["x"][1], POS_FREQS["x"][2], high_band_ratio)
    by, dby = _two_band(t_s, POS_FREQS["y"][0], POS_FREQS["y"][1], POS_FREQS["y"][2], high_band_ratio)
    bz, dbz = _two_band(t_s, POS_FREQS["z"][0], POS_FREQS["z"][1], POS_FREQS["z"][2], high_band_ratio)
    bx, by, bz = amp_x * bx, amp_y * by, amp_z * bz
    dbx, dby, dbz = amp_x * dbx, amp_y * dby, amp_z * dbz

    x = x_anchor[0] + rho * bx
    y = x_anchor[1] + rho * by
    z = x_anchor[2] + rho * bz
    dx = rho_dot * bx + rho * dbx
    dy = rho_dot * by + rho * dby
    dz = rho_dot * bz + rho * dbz
    return np.column_stack((x, y, z)), np.column_stack((dx, dy, dz))


def _build_quat_traj(
    t_s: np.ndarray,
    q_anchor_xyzw: np.ndarray,
    amp_yaw: float,
    amp_roll: float,
    rho: np.ndarray,
    high_band_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose q_des(t) = q_yaw_base(t) (x) q_anchor (x) q_roll_local(t).

    * yaw : rotation about world (base) z-axis, multi-band, drives j1.
    * roll: rotation about EE local z-axis, multi-band, drives j5/j7.
    Both default to amp=0, in which case q_des(t) = q_anchor (step5c parity).
    """
    yaw_freqs = ORI_FREQS["yaw"]
    roll_freqs = ORI_FREQS["roll"]

    if amp_yaw > 0.0:
        yaw_unit, _ = _two_band(t_s, yaw_freqs[0], yaw_freqs[1], yaw_freqs[2], high_band_ratio)
        yaw = rho * amp_yaw * yaw_unit
    else:
        yaw = np.zeros_like(t_s)
    if amp_roll > 0.0:
        roll_unit, _ = _two_band(t_s, roll_freqs[0], roll_freqs[1], roll_freqs[2], high_band_ratio)
        roll = rho * amp_roll * roll_unit
    else:
        roll = np.zeros_like(t_s)

    q_yaw_base = _quat_from_axis_angle_z(yaw)
    q_roll_local = _quat_from_axis_angle_z(roll)
    q_anchor_tiled = np.broadcast_to(q_anchor_xyzw[None, :], (t_s.shape[0], 4))
    q_des = _quat_mul_xyzw(q_yaw_base, _quat_mul_xyzw(q_anchor_tiled, q_roll_local))
    q_des = q_des / np.clip(np.linalg.norm(q_des, axis=1, keepdims=True), 1e-12, None)
    return q_des, yaw, roll


def build_step5d_trajectory(
    t_s: np.ndarray,
    x_anchor: np.ndarray,
    q_anchor_xyzw: np.ndarray,
    amp_x: float = 0.10,
    amp_y: float = 0.10,
    amp_z: float = 0.08,
    amp_yaw: float = 0.25,
    amp_roll: float = 0.20,
    high_band_ratio: float = 0.20,
    amp_ramp_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the step5d multi-band excitation trajectory on an arbitrary time grid.

    Returns
    -------
    x_des : (N, 3) Cartesian target positions, in the same frame as ``x_anchor``.
    dx_des : (N, 3) target Cartesian velocities (analytic derivative).
    quat_des : (N, 4) target orientation quaternions, **xyzw** order, unit-norm.
    yaw : (N,) yaw angle (about world-z) applied to the anchor.
    roll : (N,) roll angle (about EE local-z) applied to the anchor.

    Setting both ``amp_yaw`` and ``amp_roll`` to 0 reproduces step5c (orientation
    held at ``q_anchor_xyzw``). Default amplitudes match the
    ``step5d_20260525_143929`` collection (0.10/0.10/0.08 m, yaw 0.25, roll 0.20,
    high_band_ratio 0.20).
    """
    rho, rho_dot = _half_cosine_envelope(t_s, amp_ramp_s)
    x_des, dx_des = _build_pos_traj(t_s, x_anchor, amp_x, amp_y, amp_z, rho, rho_dot, high_band_ratio)
    quat_des, yaw, roll = _build_quat_traj(t_s, q_anchor_xyzw, amp_yaw, amp_roll, rho, high_band_ratio)
    return x_des, dx_des, quat_des, yaw, roll


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _peak_rates(t_s: np.ndarray, dx_des: np.ndarray, rot_offsets: np.ndarray) -> dict:
    """Peak Cartesian speed + rotation rates, computed for safety pre-flight."""
    if t_s.size > 1:
        dt = float(t_s[1] - t_s[0])
    else:
        dt = 0.001
    peak_dx = float(np.max(np.abs(dx_des[:, 0])))
    peak_dy = float(np.max(np.abs(dx_des[:, 1])))
    peak_dz = float(np.max(np.abs(dx_des[:, 2])))
    peak_cart_speed = float(np.max(np.linalg.norm(dx_des, axis=1)))
    # Rotation rate: finite-difference of each axis-angle component.  Cross-axis
    # angular velocity coupling is small at these amplitudes (~0.4 rad), so
    # per-axis dRx/dt etc. is a fine proxy.
    drot = np.gradient(rot_offsets, dt, axis=0)
    peak_drx = float(np.max(np.abs(drot[:, 0])))
    peak_dry = float(np.max(np.abs(drot[:, 1])))
    peak_drz = float(np.max(np.abs(drot[:, 2])))
    peak_ang_rate = float(np.max(np.linalg.norm(drot, axis=1)))
    return {
        "cart_speed_m_s": peak_cart_speed,
        "dx_m_s": [peak_dx, peak_dy, peak_dz],
        "ang_rate_rad_s": peak_ang_rate,
        "drot_rad_s": [peak_drx, peak_dry, peak_drz],
    }


# ---------------------------------------------------------------------------
# Ours: the uniform entry point excite_panda.py drives
# ---------------------------------------------------------------------------

#: Amplitude keys each kind scales, split by channel. The probe derives one
#: scalar per channel and multiplies these, which preserves the designed
#: inter-axis ratios (v4's z = 1.5x xy, v3's per-band assignment).
AMP_KEYS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "v4_chirp": (("amp_x", "amp_y", "amp_z"), ("amp_rx", "amp_ry", "amp_rz")),
    "v3_step5d": (("amp_x", "amp_y", "amp_z"), ("amp_yaw", "amp_roll")),
}


def ramp_out_envelope(t_s: np.ndarray, ramp_out_s: float) -> np.ndarray:
    """Half-cosine 1->0 over the last `ramp_out_s`, else all ones.

    NOT upstream. v3's envelope is ramp-in only, so its episode ends mid-swing
    at up to ~1.2x amplitude with nonzero velocity; this is the opt-in tail.
    Applied to the offset from the anchor, so it cannot move the anchor itself.
    """
    if ramp_out_s <= 0.0 or t_s.size == 0:
        return np.ones_like(t_s)
    total_T = float(t_s[-1])
    start = total_T - ramp_out_s
    s = np.clip((t_s - start) / ramp_out_s, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * s))


def build(kind: str, t_s: np.ndarray, x_anchor: np.ndarray, q_anchor_xyzw: np.ndarray,
          params: dict, ramp_out_s: float = 0.0) -> dict:
    """Dispatch to a builder and return a uniform reference bundle.

    `params` carries whatever the kind's builder accepts (already amplitude-
    scaled). Returns goal_pos/goal_quat/goal_lin_vel plus the peak-rate dict
    upstream pre-flights against.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    x_anchor = np.asarray(x_anchor, dtype=np.float64).reshape(3)
    q_anchor_xyzw = np.asarray(q_anchor_xyzw, dtype=np.float64).reshape(4)
    q_anchor_xyzw = q_anchor_xyzw / max(float(np.linalg.norm(q_anchor_xyzw)), 1e-12)

    if kind == "v4_chirp":
        x_des, dx_des, quat_des, rot_offsets = build_chirp_trajectory(
            t_s, x_anchor, q_anchor_xyzw, **params)
    elif kind == "v3_step5d":
        x_des, dx_des, quat_des, yaw, roll = build_step5d_trajectory(
            t_s, x_anchor, q_anchor_xyzw, **params)
        # v3 returns yaw/roll angles rather than a single rotvec; the equivalent
        # net offset from the anchor is what the rate diagnostic wants.
        rot_offsets = _rotvec_from_anchor(quat_des, q_anchor_xyzw)
    else:
        raise ValueError(f"unknown trajectory kind {kind!r}")

    if ramp_out_s > 0.0:
        tail = ramp_out_envelope(t_s, ramp_out_s)[:, None]
        x_des = x_anchor[None, :] + tail * (x_des - x_anchor[None, :])
        dx_des = dx_des * tail
        quat_des = _quat_mul_xyzw(
            _axis_angle_to_quat_xyzw(rot_offsets * tail),
            np.broadcast_to(q_anchor_xyzw[None, :], quat_des.shape).copy(),
        )
        quat_des /= np.clip(np.linalg.norm(quat_des, axis=1, keepdims=True), 1e-12, None)
        rot_offsets = rot_offsets * tail

    return {
        "goal_pos": x_des,
        "goal_quat": quat_des,
        "goal_lin_vel": dx_des,
        "rot_offsets": rot_offsets,
        "peak_rates": _peak_rates(t_s, dx_des, rot_offsets),
        "peak_ori_offset_rad": float(np.max(np.linalg.norm(rot_offsets, axis=1))),
    }


def _rotvec_from_anchor(quat_des: np.ndarray, q_anchor_xyzw: np.ndarray) -> np.ndarray:
    """Net axis-angle taking the anchor to each goal orientation, shortest path."""
    conj = q_anchor_xyzw * np.array([-1.0, -1.0, -1.0, 1.0])
    q_err = _quat_mul_xyzw(quat_des, np.broadcast_to(conj[None, :], quat_des.shape))
    q_err = q_err * np.where(q_err[:, 3:4] < 0.0, -1.0, 1.0)
    v = q_err[:, :3]
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(v_norm, np.clip(q_err[:, 3:4], -1.0, 1.0))
    return np.divide(v, v_norm, out=np.zeros_like(v), where=v_norm > 1e-12) * angle
