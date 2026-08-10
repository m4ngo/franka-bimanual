#!/usr/bin/env python3
"""RPyC torque-control server for one NUC. Runs on mario/luigi, not the workstation.

Replaces the ``rpyc_classic`` + net_franky/franky server. franky exposes no
torque interface, so the OSC law could only ever be approximated in the velocity
domain; pylibfranka's ``start_torque_control()`` gives the real thing.

Why the control law lives here and not in BimanualFranka
--------------------------------------------------------
pylibfranka's realtime interface is ``control = robot.start_torque_control()``
then a tight ``readOnce()`` / ``writeOnce(Torques(tau))`` loop. ``control`` is a
live handle into libfranka's realtime thread -- not picklable, not shareable
across a process boundary -- so the loop must run inside this process. And the
loop has to *recompute* tau each tick rather than replay a latched one: this is
what robosuite does (``run_controller()`` every 2 ms sim step with the goal held
between 20 Hz policy steps), and a constant torque held across a 30-50 ms policy
period has no live ``-kd * ee_vel`` term, so the arm accelerates unopposed.

The workstation therefore pushes *goals* at policy rate (``set_osc_goal`` /
``set_joint_goal``, both non-blocking) and reads state whenever it likes
(``get_state``, non-blocking). This thread owns everything in between. Measured
budget for the full robosuite-parity law on this hardware: ~107 us mean,
~164 us max, against a 1000 us tick.

Wire-format and lifecycle constraints
-------------------------------------
- **Tuples, not lists.** brine encodes immutable values; lists cross as netrefs
  costing a round-trip per element and spamming AttributeError when numpy probes
  ``__array__``. Every payload in and out of here is a tuple of native floats.
- **libfranka matrices are column-major.** ``O_T_EE`` and ``Model.mass`` return
  vectorized column-major arrays, so every reshape here is ``order="F"``.
- **``Model.zero_jacobian`` is unusable on this build.** It returns all zeros
  (verified on hardware: mass/coriolis/gravity are all sane from the same
  RobotState, only the Jacobian is zeroed), and ``franka::Frame`` is not exported
  by pylibfranka so the frame-taking overload cannot be reached from Python.
  The Jacobian therefore comes from ``franka_jacobian.zero_jacobian`` -- the
  analytic modified-DH one, anchored on the measured ``O_T_EE`` translation so it
  stays consistent with the pose error the OSC law forms. franky's
  ``zero_jacobian`` had the identical all-zero behaviour, so this is a property
  of the robot's model library here, not of either binding.
- **Torque rate limiting is mandatory.** libfranka rejects jumps beyond
  kMaxTorqueRate (1000 Nm/s); nothing enforces it for us the way
  ``franka::limitRate`` does in the C++ examples, so it is applied here against
  ``state.tau_J_d``.
- **SlaveService base is load-bearing.** FrankaGripper connects to this same
  host:port with ``rpyc.classic.connect``, which hardcodes the client to expect
  a SlaveService peer (``conn.execute``, ``conn.namespace``, ``getmodule``).
  Dropping the base to "simplify" this breaks the gripper with
  ``AttributeError: ... has no attribute 'getmodule'``.

Launch via ``run_server.sh`` (taskset + chrt -f 80); see
``scripts/deploy_nuc_server.sh`` on the workstation for the deploy path.
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import os
import sys
import threading
import time

import numpy as np
import rpyc
from rpyc.core.service import SlaveService
from pathlib import Path

from rpyc.utils.server import ThreadedServer

try:  # deployed flat next to this file on the NUC; importable as a package on the workstation
    from . import pylibfranka_shm as shm
    from .torque_config import torque
except ImportError:
    import pylibfranka_shm as shm  # type: ignore[no-redef]
    from torque_config import torque  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

DEFAULT_PORT = 18812
NUM_JOINTS = 7

# The control loop is in another process now, so handler latency here cannot
# starve it. Kept small anyway so RPC threads hand off promptly to each other.
_GIL_SWITCH_INTERVAL_S = float(torque("loop.gil_switch_interval_s"))

# Substrings, matched case-insensitively. Note "constraint" without the closing
# 's': libfranka 0.18 emits "communication_constraints_violation" where the
# franky-era code matched "communication_constrains_violation", so nothing ever
# matched and automatic_error_recovery() was never called -- the arm sat faulted
# while the loop retried start_torque_control() forever.
_RECOVERABLE_ERRORS = (
    "communication_constraint",
    "communication_constrains",
    "reflex",
    "udp receive: timeout",
    "control_command_success_rate",
    # Seen when arming too soon after a previous session's robot.stop(): the
    # robot is still releasing the old control mode and rejects the new one.
    "command not possible in the current mode",
)

# Arming retries. Every script's shutdown() calls robot.stop(), so a script run
# back-to-back with another routinely tries to arm while libfranka is still
# tearing the old session down.
_ARM_ATTEMPTS = int(torque("loop.arm_attempts"))
_ARM_BACKOFF_S = float(torque("loop.arm_backoff_s"))

MODE_FLOAT = "float"
MODE_HOLD = "hold"
MODE_JOINT = "joint"
MODE_JOINT_VEL = "joint_vel"
MODE_OSC = "osc"

# Cores left for the RT loop; 0-1 carry the ACPI/PM kworkers and everything we
# deliberately pin away from it (the RPyC server, the gripper server).
_RT_CPU_CANDIDATES = tuple(int(c) for c in torque("loop.rt_cpu_candidates"))


def _quietest_cpu() -> str:
    """The candidate core taking the fewest device interrupts.

    The arm's UDP and the workstation's RPyC share one NIC here, and its MSI-X
    queues are spread across most of 2-7. A NET_RX softirq on the loop's core
    delays the loop regardless of SCHED_FIFO, which is what turned 30 Hz of
    get_state into a cascade of communication_constraints_violation aborts.
    Steering the IRQs away would need root; picking a core they don't land on
    needs nothing. Recomputed per start() so it tracks whatever irqbalance did,
    and so it adapts to the two NUCs' different layouts.
    """
    counts = dict.fromkeys(_RT_CPU_CANDIDATES, 0)
    try:
        with open("/proc/interrupts") as fh:
            cpus = [int(c[3:]) for c in fh.readline().split() if c.startswith("CPU")]
            for line in fh:
                parts = line.split()
                if len(parts) < 1 + len(cpus) or not parts[0].rstrip(":").isdigit():
                    continue
                for cpu, value in zip(cpus, parts[1:1 + len(cpus)]):
                    if cpu in counts:
                        counts[cpu] += int(value)
    except (OSError, ValueError):
        return "2-7"
    return str(min(counts, key=lambda c: counts[c]))


class _ArmSession:
    """Proxy for one arm. Owns NO libfranka handle and runs NO control law.

    The loop lives in pylibfranka_control.py as a child process; this side only
    reads/writes the shared-memory channel. That separation is the point: with
    the loop in-process, every RPyC handler held the GIL for ~2 ms of protocol
    work against a 1 ms tick, so the loop overran and libfranka aborted
    mid-trajectory. Here a handler cannot starve the loop no matter how slow it
    is, because they do not share an interpreter.
    """

    def __init__(self, robot_ip: str, use_realtime: bool = True):
        self.robot_ip = robot_ip
        self.ch = shm.ShmChannel(create=True)
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # A client that never calls set_tuning gets zero friction assist, i.e.
        # exact osc.py behaviour.
        self.ch.goal[shm.G_MODE] = shm.MODE_HOLD
        logger.info("shm channel %s created for %s", self.ch.name, robot_ip)

    def start(self) -> None:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self.ch.set_running(True)
            script = str(Path(__file__).resolve().parent / "pylibfranka_control.py")
            cpu = _quietest_cpu()
            logger.info("pinning control loop for %s to cpu %s", self.robot_ip, cpu)
            self.proc = subprocess.Popen(
                ["taskset", "-c", cpu, "chrt", "-f", "80", sys.executable, script,
                 "--robot-ip", self.robot_ip, "--shm", self.ch.name],
                stdout=sys.stdout, stderr=sys.stderr)
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if self.ch.state[shm.S_ALIVE] == 1.0:
                    logger.info("control process %d live for %s", self.proc.pid, self.robot_ip)
                    return
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"control process for {self.robot_ip} exited with "
                        f"{self.proc.returncode} before arming")
                time.sleep(0.05)
            raise RuntimeError(f"control process for {self.robot_ip} did not arm in 15 s")

    def stop(self) -> None:
        with self._lock:
            self.ch.set_running(False)
            if self.proc is not None:
                try:
                    self.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.error("control process did not exit; terminating")
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                self.proc = None

    @property
    def recovery_count(self) -> int:
        return int(self.ch.state[shm.S_RECOVERY])

    # ---- goal setters: a seqlocked write, nothing more ----

    def set_osc_goal(self, goal_pos, goal_quat_xyzw, kp, kd, nullspace_q) -> None:
        self.ch.write_goal(
            G_MODE=shm.MODE_OSC,
            G_POS=np.asarray(goal_pos, dtype=np.float64),
            G_QUAT=np.asarray(goal_quat_xyzw, dtype=np.float64),
            G_KP=np.asarray(kp, dtype=np.float64),
            G_KD=np.asarray(kd, dtype=np.float64),
            G_NULLSPACE=(np.zeros(NUM_JOINTS) if nullspace_q is None
                         else np.asarray(nullspace_q, dtype=np.float64)),
        )

    def set_joint_goal(self, goal_q, kp, damping_ratio) -> None:
        self.ch.write_goal(
            G_MODE=shm.MODE_JOINT,
            G_JOINT_Q=np.asarray(goal_q, dtype=np.float64),
            G_JOINT_KP=1.0 if kp is None else float(kp),
            G_JOINT_RATIO=1.0 if damping_ratio is None else float(damping_ratio),
        )

    def set_joint_velocity_goal(self, goal_dq, kd_scale) -> None:
        self.ch.write_goal(G_MODE=shm.MODE_JOINT_VEL,
                           G_JOINT_Q=np.asarray(goal_dq, dtype=np.float64),
                           G_JOINT_RATIO=float(kd_scale))

    def set_mode(self, mode: str) -> None:
        self.ch.write_goal(G_MODE=shm.MODE_NAMES.get(mode, shm.MODE_HOLD))

    def set_tuning(self, friction_kc=None) -> None:
        """The only live-tunable server-side knob. 0 is exact osc.py.

        friction_kc accepts a scalar (all joints, both directions), a 7-vector
        (per joint, symmetric), or a 2x7 / flat-14 (per joint per rotation
        direction, [positive-torque row, negative-torque row]).
        """
        if friction_kc is not None:
            kc = np.asarray(friction_kc, dtype=np.float64)
            if kc.size == 2 * NUM_JOINTS:
                kc = kc.reshape(2, NUM_JOINTS)
            else:
                # Scalar or 7-vector: same gain in both directions.
                kc = np.broadcast_to(kc, (NUM_JOINTS,))
                kc = np.stack([kc, kc])
            self.ch.write_goal(G_FRICTION_KC_POS=kc[0], G_FRICTION_KC_NEG=kc[1])
        g = self.ch.goal
        logger.info("%s: tuning -> friction_kc +%s / -%s", self.robot_ip,
                    np.array2string(g[shm.G_FRICTION_KC_POS], precision=2),
                    np.array2string(g[shm.G_FRICTION_KC_NEG], precision=2))

    # ---- state ----

    def get_state(self) -> tuple:
        st = self.ch.read_state()
        if st is None or st[shm.S_STATE_SEQ] == 0:
            return None, None, 0
        bundle = tuple(float(x) for x in np.concatenate([
            st[shm.S_Q], st[shm.S_DQ], st[shm.S_POS], st[shm.S_QUAT], st[shm.S_TWIST],
            [st[shm.S_RECOVERY]], st[shm.S_TAU_CMD], st[shm.S_TAU_MEAS], st[shm.S_TAU_EXT],
        ]))
        err = None if st[shm.S_ALIVE] == 1.0 else "control process not alive"
        return bundle, err, int(st[shm.S_STATE_SEQ])

    def wait_next(self, since_seq: int, timeout: float | None) -> tuple:
        deadline = time.monotonic() + (timeout or 0.0)
        while True:
            bundle, err, seq = self.get_state()
            if seq > since_seq:
                return bundle, err, seq, True
            if time.monotonic() >= deadline:
                return bundle, err, seq, False
            time.sleep(0.002)

    def close(self) -> None:
        self.stop()
        self.ch.close(unlink=True)


class FrankaTorqueService(SlaveService):
    """Torque-control RPCs plus, via the SlaveService base, FrankaGripper's
    classic ``conn.execute``/``conn.namespace`` interface on the same port.

    Sessions are keyed by robot_ip and shared process-wide, so a reconnecting
    client re-attaches to the running RT loop instead of re-arming the robot.
    """

    _sessions: dict[str, _ArmSession] = {}
    _sessions_lock = threading.Lock()

    def exposed_init_robot(self, robot_ip: str, use_realtime: bool = True) -> bool:
        with self._sessions_lock:
            if robot_ip not in self._sessions:
                self._sessions[robot_ip] = _ArmSession(robot_ip, use_realtime)
        return True

    def exposed_start_control(self, robot_ip: str) -> bool:
        self._sessions[robot_ip].start()
        return True

    def exposed_read_state(self, robot_ip: str) -> tuple:
        """Latest published state. Unlike the old version this does not do its own
        read_once -- only the control process may touch the robot handle."""
        bundle, _, _ = self._sessions[robot_ip].get_state()
        return bundle

    def exposed_set_osc_goal(self, robot_ip, goal_pos, goal_quat_xyzw, kp, kd, nullspace_q=None) -> bool:
        self._sessions[robot_ip].set_osc_goal(goal_pos, goal_quat_xyzw, kp, kd, nullspace_q)
        return True

    def exposed_set_osc_goal_flat(self, robot_ip: str, packed) -> bool:
        """One flat 26-float tuple: pos(3) quat(4) kp(6) kd(6) nullspace_q(7).

        Same as exposed_set_osc_goal but a single brine object instead of five,
        which is measurably less GIL time in the handler.
        """
        p = np.asarray(packed, dtype=np.float64)
        self._sessions[robot_ip].set_osc_goal(p[0:3], p[3:7], p[7:13], p[13:19], p[19:26])
        return True

    def exposed_set_joint_goal(self, robot_ip, goal_q, kp=None, damping_ratio=None) -> bool:
        self._sessions[robot_ip].set_joint_goal(goal_q, kp, damping_ratio)
        return True

    def exposed_set_joint_velocity_goal(self, robot_ip, goal_dq, kd_scale=1.0) -> bool:
        self._sessions[robot_ip].set_joint_velocity_goal(goal_dq, kd_scale)
        return True

    def exposed_set_mode(self, robot_ip: str, mode: str) -> bool:
        self._sessions[robot_ip].set_mode(mode)
        return True

    def exposed_set_tuning(self, robot_ip: str, friction_kc: float | None = None) -> bool:
        self._sessions[robot_ip].set_tuning(friction_kc)
        return True

    def exposed_get_state(self, robot_ip: str) -> tuple:
        return self._sessions[robot_ip].get_state()

    def exposed_wait_next(self, robot_ip: str, since_seq: int, timeout: float | None = None) -> tuple:
        return self._sessions[robot_ip].wait_next(since_seq, timeout)

    def exposed_recovery_count(self, robot_ip: str) -> int:
        return self._sessions[robot_ip].recovery_count

    def exposed_stop(self, robot_ip: str) -> bool:
        self._sessions[robot_ip].stop()
        return True

    def exposed_set_load(self, robot_ip: str, load_mass: float, f_x_cload, load_inertia) -> bool:
        # The Robot handle lives in the control process now; set_load would have to
        # be staged through shm and applied there. Nothing in this stack calls it,
        # so it fails loudly rather than silently doing nothing.
        raise NotImplementedError(
            "set_load is not routed to the control process; add a shm field if needed")

    # SlaveService's allow_all_attrs _rpyc_getattr looks the name up directly and
    # so skips rpyc's exposed_ prefix stripping; without these aliases only the
    # exposed_-prefixed spelling resolves over the wire.
    init_robot = exposed_init_robot
    start_control = exposed_start_control
    read_state = exposed_read_state
    set_osc_goal = exposed_set_osc_goal
    set_osc_goal_flat = exposed_set_osc_goal_flat
    set_joint_goal = exposed_set_joint_goal
    set_joint_velocity_goal = exposed_set_joint_velocity_goal
    set_mode = exposed_set_mode
    set_tuning = exposed_set_tuning
    get_state = exposed_get_state
    wait_next = exposed_wait_next
    recovery_count = exposed_recovery_count
    stop = exposed_stop
    set_load = exposed_set_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.setswitchinterval(_GIL_SWITCH_INTERVAL_S)
    # The RT loop allocates only numpy arrays and tuples -- no reference cycles --
    # so refcounting alone reclaims it. Leaving the cycle collector on just buys
    # an unpredictable multi-hundred-microsecond pause inside a 1 ms tick.
    gc.collect()
    gc.freeze()
    gc.disable()

    server = ThreadedServer(
        FrankaTorqueService,
        port=args.port,
        hostname=args.host,
        protocol_config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": 10},
    )
    logger.info("pylibfranka torque server listening on %s:%d", args.host, args.port)
    server.start()


if __name__ == "__main__":
    main()
