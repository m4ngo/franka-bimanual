"""Shared-memory channel between the RPyC server and the 1 kHz control process.

Why this exists
---------------
Running an RPyC server in the same Python process as the control loop starves
it: each handler holds the GIL for ~2 ms of protocol work while the loop's
entire tick budget is 1 ms. Measured on this rig, that put "compute" at
537-1170 us against a ~140 us pure-compute cost, dropped libfranka's command
success rate to 0.83, and produced `communication_constraints_violation` aborts
mid-trajectory -- each of which latches a stiff joint hold and shows up as a
visible jerk.

So the loop gets its own process with no RPyC in it, and the two sides talk
through this block. No locks cross the boundary: a seqlock lets the reader
detect a torn read and retry, which is correct for one writer per block and
costs the writer two integer stores.

Layout is fixed and duplicated nowhere else -- both processes import these
offsets, so adding a field cannot desynchronise them.
"""

from __future__ import annotations

import numpy as np
from multiprocessing import shared_memory

NUM_JOINTS = 7

MODE_FLOAT, MODE_HOLD, MODE_JOINT, MODE_JOINT_VEL, MODE_OSC = 0.0, 1.0, 2.0, 3.0, 4.0
# Open-loop feedforward torque on ONE joint, impedance hold on the rest. For
# friction identification only: it is the one mode whose torque is not a function
# of the measured state, so it does not self-limit -- see G_TAU_WINDOW.
MODE_TORQUE = 5.0
MODE_NAMES = {"float": MODE_FLOAT, "hold": MODE_HOLD, "joint": MODE_JOINT,
              "joint_vel": MODE_JOINT_VEL, "osc": MODE_OSC, "torque": MODE_TORQUE}

# ---- goal block (written by the server, read by the control process) ----
G_SEQ = 0            # seqlock counter; odd = write in progress
G_MODE = 1
G_CMD_SEQ = 2        # bumped on every new goal, so control can detect staleness
G_POS = slice(3, 6)
G_QUAT = slice(6, 10)
G_KP = slice(10, 16)
G_KD = slice(16, 22)
G_NULLSPACE = slice(22, 29)
G_JOINT_Q = slice(29, 36)
G_JOINT_KP = 36
G_JOINT_RATIO = 37
# Friction assist gain, per joint AND per rotation direction: breakaway on this
# arm differs by which way the joint turns. _POS applies when the commanded
# torque is positive on that joint, _NEG when negative.
G_FRICTION_KC_POS = slice(38, 45)
G_FRICTION_KC_NEG = slice(45, 52)
G_RUNNING = 52       # server -> control: 0 asks the loop to stop
# MODE_TORQUE only. G_TAU_FF is the feedforward torque; G_TAU_JOINT selects which
# joint receives it (the rest are impedance-held at G_JOINT_Q); G_TAU_WINDOW is the
# rad of travel past the goal q at which the loop LATCHES BACK TO MODE_HOLD.
#
# The window lives here, not in the caller, because open-loop torque does not
# self-limit and the workstation cannot react fast enough: joint 7 has M ~ 0.01, so
# 2 Nm is ~200 rad/s^2 -- 0.25 rad inside one 50 ms RPyC round trip. It is a mode
# TRANSITION, not another bound on tau: _enforce_limits remains the only thing that
# rescales what reaches the joints.
G_TAU_FF = slice(53, 60)
G_TAU_JOINT = 60     # index of the joint under test; <0 disables the mode
G_TAU_WINDOW = 61    # rad
# Velocity bound on the same transition, and NOT redundant with the travel window:
# on the wrist the two are orders of magnitude apart. 0.7 Nm net on joint 7 is
# ~100 rad/s^2, so it passes its 2.61 rad/s datasheet limit in ~25 ms having moved
# only ~0.03 rad -- the robot's own joint_velocity_violation fires, and faults the
# arm, long before a 0.25 rad travel window notices. A position bound cannot stand in
# for a velocity bound on a low-inertia joint.
G_TAU_DQ_MAX = 62    # rad/s
# 56 -> 64: the block grew, so the server and the control child must be redeployed
# TOGETHER. A half-updated pair reads these fields off the end of the old block.
GOAL_SIZE = 64

# ---- state block (written by the control process, read by the server) ----
S_SEQ = 0
S_Q = slice(1, 8)
S_DQ = slice(8, 15)
S_POS = slice(15, 18)
S_QUAT = slice(18, 22)
S_TWIST = slice(22, 28)
S_RECOVERY = 28
S_TAU_CMD = slice(29, 36)
S_TAU_MEAS = slice(36, 43)
S_TAU_EXT = slice(43, 50)
S_SUCCESS_RATE = 50
S_CLAMP_TRIPS = 51   # ticks the law asked past the torque clamp
S_ALIVE = 52         # 1 while the control loop is armed and ticking
S_STATE_SEQ = 53     # bumped per published state; 0 = nothing published yet
# MODE_TORQUE tripped its travel window and latched to hold. The caller cannot infer
# this from q alone -- a joint that stopped because the window caught it looks the
# same as one that never broke away -- and the difference decides whether a row is a
# measurement.
S_TORQUE_TRIP = 54
# Law ticks on which SIM's +/-12 Nm wrist ctrlrange clip engaged (emulate_sim_plant
# only). NOT limits.joint_torque_nm, which is the real hardware clamp and a different
# question entirely: sim's rotation authority saturates where the FR3's does not, and
# whether we reproduce that is what decides the rotation overshoot. Counted here
# because it is invisible anywhere else -- the clip is inside run_controller, upstream
# of _enforce_limits, and the torque that leaves is post-transform.
S_SIM_CLIP = 55
# Law ticks on which lambda_full's conditioning dropped a direction (osc.lambda_rcond).
# Nonzero means the arm is working near a singularity, where the 6x6 operational-space
# inertia is the term that blows up. Fits inside the existing block, so unlike the
# G_TAU_* additions this one does not force a paired server/child redeploy.
S_LAMBDA_TRUNC = 56
STATE_SIZE = 64

_TOTAL = GOAL_SIZE + STATE_SIZE
_MAX_RETRY = 64


class ShmChannel:
    """One goal block + one state block in a single shared buffer."""

    def __init__(self, name: str | None = None, create: bool = False):
        nbytes = _TOTAL * 8
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            # Python's resource_tracker unlinks any segment a process ATTACHES to
            # when that process exits -- not just ones it created. So a control
            # process restarting would destroy the server's segment on its way
            # out, and the next one would die with FileNotFoundError. Only the
            # creator should own the lifetime.
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(self.shm._name, "shared_memory")
            except Exception:
                pass
        self.name = self.shm.name
        buf = np.ndarray((_TOTAL,), dtype=np.float64, buffer=self.shm.buf)
        if create:
            buf[:] = 0.0
        self.goal = buf[:GOAL_SIZE]
        self.state = buf[GOAL_SIZE:]

    # -- seqlock primitives -------------------------------------------------

    @staticmethod
    def _write(block: np.ndarray, seq_index: int, fn) -> None:
        # try/finally, because a raise inside fn would otherwise leave the counter
        # odd forever and every subsequent _read returns None for the life of the
        # process. Publishing a half-written block is the lesser failure: the reader
        # holds its last good copy for one tick, where a latched-odd counter routes
        # every tick through that fallback.
        block[seq_index] += 1          # odd: write in progress
        try:
            fn(block)
        finally:
            block[seq_index] += 1      # even: consistent again

    @staticmethod
    def _read(block: np.ndarray, seq_index: int) -> np.ndarray | None:
        """Snapshot, or None if the writer kept beating us."""
        for _ in range(_MAX_RETRY):
            s0 = block[seq_index]
            if s0 % 2:                 # writer mid-update
                continue
            snap = block.copy()
            if block[seq_index] == s0:
                return snap
        return None

    # -- goal ---------------------------------------------------------------

    def write_goal(self, *, new_command: bool = True, **fields) -> None:
        """Seqlocked write. ``new_command=False`` for session settings.

        G_CMD_SEQ is what the loop latches a goal on, and it also re-arms the
        stale-goal watchdog. A write that carries no new pose -- set_mode,
        set_tuning -- must not bump it: doing so re-latches the whole OSC goal
        from whatever G_POS/G_QUAT still hold and tells the watchdog a goal just
        arrived, so a controller that had gone stale silently resumes driving
        toward an old pose. The MODE field is read every tick regardless of the
        sequence, so a mode change still takes effect without one.
        """
        def _apply(b):
            for key, value in fields.items():
                target = globals()[key]
                b[target] = value
            if new_command:
                b[G_CMD_SEQ] += 1
        self._write(self.goal, G_SEQ, _apply)

    def read_goal(self) -> np.ndarray | None:
        return self._read(self.goal, G_SEQ)

    def set_running(self, running: bool) -> None:
        self.goal[G_RUNNING] = 1.0 if running else 0.0

    # -- state --------------------------------------------------------------

    def write_state(self, values: dict) -> None:
        def _apply(b):
            for target, value in values.items():
                b[target] = value
            b[S_STATE_SEQ] += 1
        self._write(self.state, S_SEQ, _apply)

    def read_state(self) -> np.ndarray | None:
        return self._read(self.state, S_SEQ)

    def close(self, unlink: bool = False) -> None:
        try:
            self.shm.close()
        except Exception:
            pass
        if unlink:
            try:
                self.shm.unlink()
            except Exception:
                pass
