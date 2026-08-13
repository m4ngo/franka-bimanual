"""Measure the arm's vibration directly, at the publish rate, under a ZERO command.

Why this exists
---------------
`delta_sweep`'s `reversals` cannot see the shake: it samples one point per policy
step (20 Hz) and the shake sits around 50 Hz, so it aliases away. Every run so far
reported 0-1 reversals while the arm was visibly buzzing.

The test condition is a held pose with a zero EE delta, because that is the WORST
case for the friction assist, not the mildest: the assist is a tanh of the commanded
torque, so at zero command it sits exactly on the zero crossing, where its
incremental gain is highest (~1 + kc*(coul-0.1)/(eps_frac*coul), i.e. ~4x at the
shipped constants). See _friction_feedforward in pylibfranka_control.

friction_kc is swept LIVE through set_tuning, so one run gives the whole curve
without a redeploy and without re-homing between points.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import franka_config as fc


def sample(robot, key, seconds, publish_hz):
    """Poll above the publish rate and keep only fresh samples."""
    mgr = robot.robot_manager
    t_end = time.perf_counter() + seconds
    period = 1.0 / (2.0 * publish_hz)
    q_s, dq_s, tau_s, t_s, pos_s = [], [], [], [], []
    last_q = None
    while time.perf_counter() < t_end:
        snap = mgr.current_kinematic_state(key)
        q = np.asarray(snap[0], dtype=np.float64)
        if last_q is None or not np.array_equal(q, last_q):
            last_q = q
            q_s.append(q)
            dq_s.append(np.asarray(snap[1], dtype=np.float64))
            pos_s.append(np.asarray(snap[3], dtype=np.float64))
            tau_s.append(np.asarray(mgr.torque_snapshot(key)[0], dtype=np.float64))
            t_s.append(time.perf_counter())
        time.sleep(period)
    return (np.asarray(t_s), np.asarray(q_s), np.asarray(dq_s),
            np.asarray(tau_s), np.asarray(pos_s))


def spectrum(t, x, fs, f_min=25.0):
    """Dominant frequency and its amplitude, per column, above ``f_min``.

    25 Hz, not 5: the goal is a staircase updated at control_fps (20 Hz), so a
    reversing command puts real energy at 20 Hz and its harmonics. Anything the
    ARM adds on its own has to be looked for above that -- at 5 Hz the reversal
    fundamental itself dominates and the metric just measures the test signal.
    """
    n = len(t)
    if n < 32:
        return np.nan, np.zeros(x.shape[1])
    win = np.hanning(n)[:, None]
    X = np.abs(np.fft.rfft((x - x.mean(0)) * win, axis=0)) * 2.0 / (n / 2)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = f >= f_min
    if not band.any():
        return np.nan, np.zeros(x.shape[1])
    power = X[band].sum(axis=1)
    return float(f[band][np.argmax(power)]), X[band].max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--axis", type=int, default=None,
                    help="0-2 translation, 3-5 rotation. Omit to hold a zero command. "
                         "A held pose does NOT excite the loop -- measured flat at every "
                         "kc -- so use this to reverse through the assist's zero crossing")
    ap.add_argument("--amp", type=float, default=0.3, help="normalized action amplitude")
    ap.add_argument("--half-period-steps", type=int, default=6,
                    help="policy steps per direction before reversing. The reversal is "
                         "the event under test: it is where the commanded torque crosses "
                         "zero and the tanh band's incremental gain is highest")
    ap.add_argument("--arm", default="r")
    ap.add_argument("--drift-abort-m", type=float, default=0.03,
                    help="stop the sweep if the EE walks this far from where it started")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    from lerobot_robot_bimanual_franka import (
        ControlMode, SingleArmFranka, SingleArmFrankaConfig)

    publish_hz = 1000.0 / float(fc.control("torque.loop.publish_decimation"))
    print(f"publish rate {publish_hz:.0f} Hz, holding a ZERO delta, "
          f"sweeping friction_kc {args.kcs}")
    if not args.yes:
        raise SystemExit("pass --yes; the arm is energised under OSC (it does not move "
                         "on command, but the assist can make it walk)")

    robot = SingleArmFranka(SingleArmFrankaConfig(
        control_mode=ControlMode.EE_DELTA, cameras={}, depth=False, depth_cam={}))
    robot.connect()
    key = args.arm
    try:
        zero = {f"{key}_{a}": 0.0 for a in ("x", "y", "z", "qx", "qy", "qz")}
        zero[f"{key}_qw"] = 1.0
        zero[f"{key}_gripper"] = 0.0
        zero["kp"] = zero["kd"] = 0.0

        p0 = np.asarray(robot.robot_manager.current_kinematic_state(key)[3],
                        dtype=np.float64)
        AX = ("x", "y", "z", "qx", "qy", "qz")
        pos_max = float(fc.control("torque.delta.pos_max_m"))
        rot_max = float(fc.control("torque.delta.rot_max_rad"))

        def action(sign):
            a = dict(zero)
            if args.axis is None:
                return a
            if args.axis < 3:
                a[f"{key}_{AX[args.axis]}"] = sign * args.amp * pos_max
            else:
                rv = np.zeros(3); rv[args.axis - 3] = sign * args.amp * rot_max
                from scipy.spatial.transform import Rotation
                q = Rotation.from_rotvec(rv).as_quat()
                for i, c in enumerate(("qx", "qy", "qz", "qw")):
                    a[f"{key}_{c}"] = float(q[i])
            return a
        rows = []
        for kc in args.kcs:
            robot.set_friction_kc(kc)
            # Hold the goal alive across the settle AND the sample: the loop parks the
            # arm after torque.loop.stale_goal_timeout_s without a fresh goal, and a
            # parked arm is a joint-impedance hold, which is not what we are measuring.
            t_settle = time.perf_counter() + 1.0
            while time.perf_counter() < t_settle:
                robot.send_action(dict(zero))
                time.sleep(0.05)

            t, q, dq, tau, pos = [], [], [], [], []
            t_end = time.perf_counter() + args.seconds
            step = 0
            while time.perf_counter() < t_end:
                sign = 1.0 if (step // args.half_period_steps) % 2 == 0 else -1.0
                robot.send_action(action(sign))
                step += 1
                s = sample(robot, key, 0.05, publish_hz)
                t.append(s[0]); q.append(s[1]); dq.append(s[2])
                tau.append(s[3]); pos.append(s[4])
            t = np.concatenate(t); dq = np.concatenate(dq)
            tau = np.concatenate(tau); pos = np.concatenate(pos)

            fs = len(t) / max(t[-1] - t[0], 1e-9)
            # High-pass by differencing out the commanded ramp: the tracked motion is
            # not vibration, and at amp 0.3 it dwarfs everything else in the spectrum.
            if args.axis is not None and len(dq) > 8:
                k = max(int(fs / 20.0), 3)          # one policy period
                trend = np.apply_along_axis(
                    lambda c: np.convolve(c, np.ones(k) / k, mode="same"), 0, dq)
                dq = dq - trend
            f_dq, amp_dq = spectrum(t, dq, fs)
            f_tau, amp_tau = spectrum(t, tau, fs)
            drift = float(np.linalg.norm(pos[-1] - p0))
            rms = float(np.sqrt(np.mean(dq ** 2)))
            rows.append((kc, fs, rms, f_dq, amp_dq, f_tau, amp_tau, drift))
            print(f"  kc={kc:4.2f}  fs={fs:6.1f}Hz  |dq|rms={rms:7.5f} rad/s  "
                  f"peak {f_dq:5.1f}Hz  worst-joint dq amp={amp_dq.max():7.5f}  "
                  f"tau amp={amp_tau.max():6.3f}Nm  drift={drift*1e3:6.2f}mm")
            if drift > args.drift_abort_m:
                print(f"  ABORT: EE walked {drift*1e3:.1f} mm > "
                      f"{args.drift_abort_m*1e3:.0f} mm limit")
                break
            if args.axis is not None and kc != args.kcs[-1]:
                # Back to the anchor before the next point: a reversal test under a
                # deadband does not return to centre, and each kc must start from the
                # same pose or it measures a different Jacobian.
                robot.home(home_q_left=None,
                           home_q_right=np.asarray(fc.home_q("home_pose", key=key)),
                           max_time_s=25.0, tol_rad=0.02, fps=30)
                p0 = np.asarray(robot.robot_manager.current_kinematic_state(key)[3],
                                dtype=np.float64)

        print(f"\n{'kc':>6}{'|dq| rms':>11}{'peak Hz':>9}{'dq amp':>10}"
              f"{'tau amp Nm':>12}{'drift mm':>10}")
        for kc, fs, rms, f_dq, amp_dq, f_tau, amp_tau, drift in rows:
            print(f"{kc:>6.2f}{rms:>11.5f}{f_dq:>9.1f}{amp_dq.max():>10.5f}"
                  f"{amp_tau.max():>12.3f}{drift*1e3:>10.2f}")
        print("\nworst joint per kc (1-indexed), by dq amplitude at the peak:")
        for kc, fs, rms, f_dq, amp_dq, f_tau, amp_tau, drift in rows:
            print(f"  kc={kc:4.2f}  joint {int(np.argmax(amp_dq))+1}  "
                  f"{np.round(amp_dq, 5)}")
    finally:
        robot.set_friction_kc(float(fc.control("tuning.friction_kc")))
        robot.disconnect()


if __name__ == "__main__":
    main()
