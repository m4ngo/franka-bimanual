"""Client-side B-spline plan: the half of PolicyLocalBSpline that owns the clock.

`policy_local_bspline.py` imports torch, hydra and dill at module level, so it
cannot be imported in the workspace venv. This is a port of its plan management
with the model replaced by a request to the policy server, which returns raw
spline parameters. Names and structure follow the original so the two can be
diffed; numpy and scipy only.

Keeping the clock on this side is the point of the split. The server returns a
spline, not an action, so no per-tick round trip lands between the state read and
the goal write -- and evaluating the plan at wall-clock `t` is where the whole
speed-up lives:

    t = (now - plan_start) * speed_up_times * origin_time_scale

`origin_time_scale` converts the spline's index-space knots into seconds, so it
is the rate the demonstrations were RECORDED at -- see rollout_common and the
note in config/policy.yaml.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

logger = logging.getLogger("baselines.bspline.plan")

# The sampled action's layout: [pos(3), rot6d(6), gripper(1)]. B-Spline's own
# dataset loader turns the converters' 7-dim [pos, rotvec, gripper] rows into
# rotation_6d, so this width is the checkpoint's, not a choice made here. The
# pose columns are what a stitch matches on; the gripper is a step.
ACT_DIM = 10
POSE_DIM = 9
GRIPPER_INDEX = 9


def decode_action(row) -> tuple[np.ndarray, np.ndarray, float]:
    """Sampled spline row -> (pos, quat_xyzw, gripper).

    Decoded here rather than through upstream's `decode_action_vector`, whose
    formats all return YAM/X5 action dicts.
    """
    from baselines.policy_math import rot6d_to_quat_xyzw

    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size != ACT_DIM:
        raise ValueError(
            f"expected a {ACT_DIM}-dim pos+rot6d+gripper action, got {row.size}. "
            "The checkpoint's action space does not match what the converter writes."
        )
    return row[:3], rot6d_to_quat_xyzw(row[3:9]), float(row[GRIPPER_INDEX])


def safer_knots(knots) -> np.ndarray:
    """Ported verbatim. A predicted knot column is not guaranteed monotonic and
    scipy's BSpline requires it."""
    knots = np.asarray(knots, dtype=np.float64).copy()
    for idx in range(1, len(knots)):
        if knots[idx] < knots[idx - 1]:
            knots[idx] = knots[idx - 1] + 1e-6
    return knots


class SplinePlanner:
    def __init__(
        self,
        client,
        *,
        degree: int = 3,
        n_obs_steps: int = 2,
        obs_stride: int = 1,
        origin_time_scale: float = 20.0,
        speed_up_times: float = 1.0,
        predict_before_end: float = 0.06,
        time_align_error_threshold: float = 0.1,
        time_align_larger_t: float | None = 0.2,
        disable_time_align: bool = False,
        restart_on_time_align_error: bool = False,
        consider_gripper_during_align: bool = False,
        gripper_slowdown_enabled: bool = False,
        gripper_slowdown_threshold: float = 0.08,
        gripper_slowdown_steps: int = 7,
        gripper_index: int = 9,
        compare_dim: int = 9,
        clock=None,
        synchronous: bool = False,
    ) -> None:
        self.client = client
        self.degree = int(degree)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_stride = max(1, int(obs_stride))
        self.obs_history: deque = deque(maxlen=self.n_obs_steps * self.obs_stride)
        self.origin_time_scale = float(origin_time_scale)
        self.speed_up_times = float(speed_up_times)
        self.predict_before_end = float(predict_before_end)
        self.time_align_error_threshold = float(time_align_error_threshold)
        self.time_align_larger_t = time_align_larger_t
        self.disable_time_align = bool(disable_time_align)
        self.restart_on_time_align_error = bool(restart_on_time_align_error)
        # Pose columns only by default: the gripper is a step, and letting it
        # dominate the match makes the stitch land on the wrong side of a grasp.
        self.compare_dim = int(compare_dim) + (1 if consider_gripper_during_align else 0)
        self.gripper_slowdown_enabled = bool(gripper_slowdown_enabled)
        self.gripper_slowdown_threshold = float(gripper_slowdown_threshold)
        self.gripper_slowdown_steps = int(gripper_slowdown_steps)
        self.gripper_index = int(gripper_index)
        # The plan is sampled at `clock()`, so a simulator passes its own step
        # counter and `t` advances with SIM time instead of the wall clock.
        # Wall-clock diagnostics below stay on perf_counter.
        self._clock = clock if clock is not None else time.perf_counter
        # Synchronous: the request is served inside _request_if_needed rather
        # than by the worker. In sim the caller owns the clock, so an inference
        # that overlaps stepping would make the plan's phase depend on GPU speed.
        self._synchronous = bool(synchronous)

        self.lock = threading.Lock()
        self.req_queue: queue.Queue = queue.Queue(maxsize=1)
        self.predictor: BSpline | None = None
        self.min_t = 0.0
        self.max_t = 0.0
        self.last_obs_time_to_predict: float | None = None
        self.last_t_normalized: float | None = None
        self._sampled = None    # (predictor, t, max_t) of the last poll_action sample
        self.getting_spline = False
        self.plans = 0
        self.align_errors: list[float] = []
        self._epoch = 0
        self._accumulated_t = 0.0
        self._last_step_time: float | None = None
        self._last_gripper: float | None = None
        self._slowdown_remaining = 0
        self._stop = False
        self._thread = None
        if not self._synchronous:
            self._thread = threading.Thread(target=self._predict_process, daemon=True)
            self._thread.start()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        with self.lock:
            self._epoch += 1
            self.predictor = None
            self.last_obs_time_to_predict = None
            self.last_t_normalized = None
            self.getting_spline = False
            self._accumulated_t = 0.0
            self._last_step_time = None
            self._last_gripper = None
            self._slowdown_remaining = 0
        self.obs_history.clear()
        while not self.req_queue.empty():
            self.req_queue.get_nowait()

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self.req_queue.put(None)

    def waiting_for_first_plan(self) -> bool:
        with self.lock:
            return self.predictor is None

    def wait_for_pending_inference(self, timeout: float = 3.0) -> bool:
        """Block until no request is in flight.

        Call before teardown: tearing down while the worker is inside a request
        is what makes upstream's process abort on exit, and here it would leave
        the ZMQ socket closing under an active send.
        """
        if self._synchronous:
            return True
        deadline = time.perf_counter() + float(timeout)
        while time.perf_counter() < deadline:
            with self.lock:
                if not self.getting_spline:
                    return True
            time.sleep(0.01)
        return False

    # -- stepping ----------------------------------------------------------

    def step(self, obs: dict) -> np.ndarray | None:
        """Feed one observation, then sample. None until a plan exists."""
        self.obs_history.append(obs)
        if len(self.obs_history) >= self.n_obs_steps * self.obs_stride:
            sequence = [self.obs_history[i]
                        for i in range(self.obs_stride - 1, len(self.obs_history),
                                       self.obs_stride)]
            self._request_if_needed(sequence)
        return self.poll_action()

    def poll_action(self) -> np.ndarray | None:
        """Sample the current plan without feeding an observation."""
        with self.lock:
            if self.predictor is None or self.last_obs_time_to_predict is None:
                return None
            now = self._clock()
            if self.gripper_slowdown_enabled:
                t = self._step_gripper_slowdown_time(now)
            else:
                t = ((now - self.last_obs_time_to_predict)
                     * self.speed_up_times * self.origin_time_scale)
            if t < self.min_t:
                t = self.min_t
            if t > self.max_t:
                # The plan is spent and the replacement has not landed. Upstream
                # returns None here too; the caller holds the last goal rather
                # than extrapolating a spline past its own support.
                return None
            self.last_t_normalized = t
            self._sampled = (self.predictor, t, self.max_t)
            return np.asarray(self.predictor(np.array([t])), dtype=np.float64).squeeze()

    def peek(self, dt: float) -> np.ndarray:
        """The plan poll_action last sampled, `dt` clock seconds after that sample."""
        with self.lock:
            predictor, t, max_t = self._sampled
            t = min(t + dt * self.speed_up_times * self.origin_time_scale, max_t)
            return np.asarray(predictor(np.array([t])), dtype=np.float64).squeeze()

    def plan_samples(self, step: float = 1.0) -> np.ndarray:
        """The installed plan sampled every `step` knot units over its whole support."""
        with self.lock:
            return np.asarray(self.predictor(np.arange(self.min_t, self.max_t + 1e-9, step)))

    # -- planning ----------------------------------------------------------

    def _request_if_needed(self, sequence) -> None:
        with self.lock:
            if self.getting_spline:
                return
            if self.predictor is None or self.last_obs_time_to_predict is None:
                needs = True
            else:
                elapsed = self._clock() - self.last_obs_time_to_predict
                remaining = (self.max_t / self.origin_time_scale
                             - elapsed * self.speed_up_times)
                # remaining is in origin-trajectory seconds, consumed at
                # speed_up_times x wall clock. Scaling the threshold keeps the
                # wall-clock lead given to the server constant; without it the
                # plan runs dry mid-inference at high speed-up and the motion
                # stalls segment by segment.
                needs = remaining < self.predict_before_end * self.speed_up_times
            if not needs:
                return
            self.getting_spline = True
            epoch = self._epoch
        req = {
            "obs": [dict(o) for o in sequence],
            "obs_time": self._clock(),
            "epoch": epoch,
        }
        if self._synchronous:
            self._serve(req)
            return
        try:
            self.req_queue.put_nowait(req)
        except queue.Full:
            with self.lock:
                self.getting_spline = False

    def _predict_process(self) -> None:
        while not self._stop:
            req = self.req_queue.get()
            if req is None:
                return
            self._serve(req)

    def _serve(self, req: dict) -> None:
        try:
            t0 = time.perf_counter()
            rep = self.client.request({"obs": req["obs"]})
            if "error" in rep:
                raise RuntimeError(f"policy server: {rep['error']}")
            bspline = rep.get("bspline")
            if bspline is None:
                raise RuntimeError(
                    "server returned no 'bspline'. Start it with "
                    "--response-format bspline."
                )
            logger.info("new spline in %.3fs", time.perf_counter() - t0)
            self._install(np.asarray(bspline, dtype=np.float64), req)
        except Exception:
            logger.exception("spline request failed")
            with self.lock:
                self.getting_spline = False

    def _install(self, bspline: np.ndarray, req: dict) -> None:
        with self.lock:
            if req["epoch"] != self._epoch:
                self.getting_spline = False
                return
            # last_t_normalized is only set once poll_action has sampled. A
            # second plan can land before that (a slow loop, a large
            # predict_before_end), and there is nothing to align to yet.
            if (self.predictor is None or self.disable_time_align
                    or self.last_t_normalized is None):
                self._flush(bspline)
                self.last_obs_time_to_predict = self._clock()
                t_new, error = self.min_t, 0.0
            else:
                old = np.asarray(
                    self.predictor(np.array([self.last_t_normalized])), dtype=np.float64
                ).squeeze()
                self._flush(bspline)
                t_new, error = self._align(old, req["obs_time"])
                if self.restart_on_time_align_error and error > self.time_align_error_threshold:
                    logger.warning("time-align error %.6f too large; restarting at min_t", error)
                    t_new = self.min_t
                self.last_obs_time_to_predict = (
                    self._clock()
                    - t_new / self.speed_up_times / self.origin_time_scale
                )
                if error > self.time_align_error_threshold:
                    logger.warning("time-align error %.6f over threshold %.6f",
                                   error, self.time_align_error_threshold)
                self.align_errors.append(float(error))
            if self.gripper_slowdown_enabled:
                self._accumulated_t = float(t_new)
                self._last_step_time = self._clock()
            self.plans += 1
            self.getting_spline = False

    def _flush(self, bspline_raw: np.ndarray) -> None:
        """Ported. Column 0 is the knot vector, the rest are control points; the
        last degree+1 control rows are padding scipy supplies itself."""
        knots = safer_knots(bspline_raw[..., 0])
        control = np.asarray(bspline_raw[..., 1:], dtype=np.float64)
        self.predictor = BSpline(t=knots, c=control[: -(self.degree + 1)], k=self.degree)
        if self.predictor.c.ndim != 2:
            raise ValueError(f"expected 2D control points, got {self.predictor.c.shape}")
        self.min_t, self.max_t = (
            float(self.predictor.t[self.degree]),
            float(self.predictor.t[-self.degree - 1]),
        )

    def _align(self, old_action: np.ndarray, obs_time: float) -> tuple[float, float]:
        """Ported `_align_new_plan`: start the new plan at the `t` whose action
        is closest to the last one emitted from the old plan, not at t=0, so the
        stitch does not jump the arm backwards."""
        new_max_t = float(np.clip(
            (self._clock() - obs_time) * self.speed_up_times * self.origin_time_scale,
            self.min_t, self.max_t,
        ))
        max_allowed = self.max_t - self.predict_before_end * self.origin_time_scale - 0.1
        if self.time_align_larger_t is not None:
            frac = float(self.time_align_larger_t)
            max_allowed = min(max_allowed, self.max_t * frac + self.min_t * (1.0 - frac))
        max_allowed = max(max_allowed, self.min_t + 1e-3)

        lam, best_t, best_error = 1.0, self.min_t, math.inf
        while best_error > self.time_align_error_threshold:
            this_max_t = min(new_max_t * lam, max_allowed)
            if this_max_t <= self.min_t:
                break
            best_t, best_error = self._closest_t(old_action, self.min_t, this_max_t)
            if lam * new_max_t > max_allowed or lam > 20:
                break
            lam *= 1.5
        return best_t, best_error

    def _closest_t(self, target: np.ndarray, min_t: float, max_t: float) -> tuple[float, float]:
        target = np.asarray(target).reshape(-1)
        d = self.compare_dim

        def dist(t):
            current = np.asarray(self.predictor(t)).squeeze()
            # Upstream's own objective: sqrt of each squared residual, summed --
            # an L1 norm, not an L2. Kept as-is so the stitch lands where theirs
            # does.
            return np.sqrt((current[:d] - target[:d]) ** 2).sum()

        res = minimize_scalar(dist, bounds=(min_t, max_t), method="bounded")
        err = np.abs(np.asarray(self.predictor(res.x)).squeeze()[:d] - target[:d])
        return float(res.x), float(err.max())

    def _step_gripper_slowdown_time(self, now: float) -> float:
        """Ported. Drops back toward 1x for a few steps when the commanded
        gripper moves: a sped-up grasp is where speed costs success."""
        if self._last_step_time is None:
            self._last_step_time = now
            self._accumulated_t = 0.0
            return self._accumulated_t
        dt = now - self._last_step_time
        tentative = float(np.clip(
            self._accumulated_t + dt * self.speed_up_times * self.origin_time_scale,
            self.min_t, self.max_t,
        ))
        grip = float(np.asarray(self.predictor(np.array([tentative]))).squeeze()[self.gripper_index])
        if self._last_gripper is not None:
            if abs(grip - self._last_gripper) > self.gripper_slowdown_threshold:
                self._slowdown_remaining = self.gripper_slowdown_steps
        self._last_gripper = grip
        if self._slowdown_remaining > 0:
            speed = 1.0 + (
                (self.gripper_slowdown_steps - self._slowdown_remaining)
                / max(self.gripper_slowdown_steps, 1)
            ) * (self.speed_up_times - 1.0)
            self._slowdown_remaining -= 1
        else:
            speed = self.speed_up_times
        self._accumulated_t += dt * speed * self.origin_time_scale
        self._last_step_time = now
        return self._accumulated_t
