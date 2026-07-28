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
MODE_NAMES = {"float": MODE_FLOAT, "hold": MODE_HOLD, "joint": MODE_JOINT,
              "joint_vel": MODE_JOINT_VEL, "osc": MODE_OSC}

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
G_FRICTION_KC = 38
G_JOINT_DAMPING_KV = 39
G_UNCOUPLE = 40
G_RUNNING = 41       # server -> control: 0 asks the loop to stop
GOAL_SIZE = 48

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
S_GUARD_TRIPS = 51
S_ALIVE = 52         # 1 while the control loop is armed and ticking
S_STATE_SEQ = 53     # bumped per published state; 0 = nothing published yet
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
        block[seq_index] += 1          # odd: write in progress
        fn(block)
        block[seq_index] += 1          # even: consistent again

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

    def write_goal(self, **fields) -> None:
        def _apply(b):
            for key, value in fields.items():
                target = globals()[key]
                b[target] = value
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
