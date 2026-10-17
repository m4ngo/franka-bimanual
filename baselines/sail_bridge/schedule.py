"""SAIL's real-robot executor: an action chunk as a trajectory in wall-clock time.

The repo's executor, run_trained_agent_receding_horizon.py:rollout_diffusion_policy,
drives a simulator. The simulator stops while the policy thinks, so latency is
modelled as a fixed `inf_delay` rows of the previous plan, and each row is a goal
robosuite's OSC holds for one control period. The paper's hardware executor
(arXiv 2506.11948, Sec. 4.4 "Action scheduling" and App. D.1; the code is not
released) is the real-time form of the same loop, and this is that form:

  * row k of a plan is scheduled at t_o + sum of its step intervals, t_o being
    when the observation it was planned from was taken. A step lasts
    1/fast_hz, or 1/slow_hz where the precision label slows it -- the repo's
    get_slowdown_mode_from_model, window and all;
  * the previous plan keeps executing while the next one is inferred, and its
    rows scheduled after the new plan's arrival are discarded;
  * a spline is fitted over the chunk, and the arm tracks its pose AND its
    velocity: tau = J^T M (Kp e_p + Kv e_v).

The fit is least squares, not interpolation. The rows are reached poses, and the
demonstrations' reached poses carry the recording's frame-to-frame jitter (speed
swinging 2-15 cm/s between 20 Hz frames); a spline forced through every row hands
that jitter to the velocity target. The paper fits and does not say how
smoothly, so `knot_rows` is ours: one interior knot every that many rows.

Pure numpy/scipy, so the offline harness drives it without hardware.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_lsq_spline
from scipy.spatial.transform import Rotation

from baselines.policy_math import slowdown_mode

_DEGREE = 3
# Step of the central difference the angular velocity is taken over, s.
_ANG_DT = 1e-3


def _lsq(times: np.ndarray, values: np.ndarray, knot_rows: int):
    """Cubic least-squares spline over `values` (n, d) at `times`, a knot every `knot_rows` rows."""
    inner = times[knot_rows:-1:knot_rows] if len(times) > knot_rows + 1 else times[:0]
    # A knot too near the end leaves the last span with too few rows to fit.
    inner = inner[inner < times[-1] - 0.5 * (times[-1] - times[-2])]
    knots = np.r_[[times[0]] * (_DEGREE + 1), inner, [times[-1]] * (_DEGREE + 1)]
    return make_lsq_spline(times, values, knots, k=_DEGREE)


class ScheduledPlan:
    """One inference's rows, each with the wall-clock time it is due.

    `rows` are unnormalised [pos(3), rotvec(3), gripper(1), label(1) if
    `precision`], in the plan's own rotation-vector branch. `history` is the rows
    executed before this plan, for the left half of the first rows' precision
    window.
    """

    def __init__(self, rows, t_obs: float, fast_hz: float, slow_hz: float, *,
                 precision: bool, window: int, knot_rows: int, history=()) -> None:
        self.rows = np.asarray(rows, dtype=np.float64)
        n = len(self.rows)
        if n < _DEGREE + 1:
            raise ValueError(f"a plan needs at least {_DEGREE + 1} rows to fit, got {n}")
        slow = np.zeros(n, dtype=bool)
        if precision:
            # Upstream appends a row to the executed list and then asks about it.
            prev = [np.asarray(r, dtype=np.float64) for r in history]
            for k in range(n):
                prev.append(self.rows[k])
                slow[k] = slowdown_mode(prev, self.rows[k], self.rows[k:], window)
        self.slow = slow
        self.precision = bool(precision)
        # The step INTO row k runs at row k's rate, as upstream holds row k for 1/f(k).
        dt = np.where(slow[1:], 1.0 / float(slow_hz), 1.0 / float(fast_hz))
        self.times = float(t_obs) + np.concatenate([[0.0], np.cumsum(dt)])
        self._pos = _lsq(self.times, self.rows[:, :3], knot_rows)
        self._vel = self._pos.derivative()
        # Orientation as small rotation vectors about the middle row's, so the fit
        # never meets the branch cut at pi.
        rot = Rotation.from_rotvec(self.rows[:, 3:6])
        self._rot_ref = rot[n // 2]
        self._rot_fit = _lsq(self.times, (self._rot_ref.inv() * rot).as_rotvec(), knot_rows)

    def _rot(self, t: float) -> Rotation:
        return self._rot_ref * Rotation.from_rotvec(self._rot_fit(t))

    @property
    def start(self) -> float:
        return float(self.times[0])

    @property
    def end(self) -> float:
        return float(self.times[-1])

    def __len__(self) -> int:
        return len(self.rows)

    def index_at(self, t: float) -> float:
        """Fractional row index due at `t`, clamped to the plan."""
        return float(np.interp(t, self.times, np.arange(len(self.times), dtype=np.float64)))

    def entry(self, t: float) -> int:
        """The first row not yet due at `t`: where a plan arriving at `t` takes over."""
        return int(np.searchsorted(self.times, t, side="left"))

    def pose(self, t: float) -> tuple[np.ndarray, Rotation]:
        """Position and orientation the plan wants at `t`; its ends outside it."""
        t = float(np.clip(t, self.start, self.end))
        return np.asarray(self._pos(t), dtype=np.float64), self._rot(t)

    def velocity(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(linear m/s, angular rad/s), both in the base frame; zero outside the plan."""
        if not self.start <= t < self.end:
            return np.zeros(3), np.zeros(3)
        lo, hi = max(t - _ANG_DT, self.start), min(t + _ANG_DT, self.end)
        ang = (self._rot(hi) * self._rot(lo).inv()).as_rotvec() / (hi - lo)
        return np.asarray(self._vel(t), dtype=np.float64), ang

    def gripper(self, t: float) -> float:
        """The gripper of the row being moved toward, as upstream commands row k during its step."""
        k = min(int(np.searchsorted(self.times, t, side="left")), len(self.rows) - 1)
        return float(self.rows[k, 6])

    def rows_at(self, index: float, n: int) -> np.ndarray:
        """`n` rows from fractional `index` on, each linear between its neighbours.

        The guidance reference: the plan's rows covering the next plan's first
        `n` steps, at whatever fraction of a row the observation fell on. Linear
        in the plan's own rotation vectors, which are continuous along it; the
        precision label is taken from the nearest row, never blended.
        """
        last = len(self.rows) - 1
        idx = np.clip(float(index) + np.arange(n, dtype=np.float64), 0.0, float(last))
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, last)
        frac = (idx - lo)[:, None]
        out = (1.0 - frac) * self.rows[lo] + frac * self.rows[hi]
        if self.precision:
            out[:, -1] = self.rows[np.rint(idx).astype(int), -1]
        return out
