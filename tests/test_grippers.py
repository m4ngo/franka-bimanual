"""Gripper driver tests.

FrankaGripper is what the rig runs (the scripts pass r_gripper_ip == r_robot_ip,
so BimanualFranka._make_gripper picks it over WSG). Its commands block for the
whole motion on a single worker, so the interesting behaviour is what happens to
a command issued while the previous one is still moving.

WSG still backs the bimanual scripts (gripper on 192.168.2.20/21) and is covered
against a loopback fake.

Run standalone (no pytest needed):

    python tests/test_grippers.py
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot_robot_bimanual_franka.franka_gripper import FrankaGripper
from lerobot_robot_bimanual_franka.wsg import WSG

MAX = FrankaGripper.GRIPPER_TRUE_MAX_MM


# --------------------------------------------------------------------------
# FrankaGripper -- the driver the rig actually uses
# --------------------------------------------------------------------------


class _FakeFranka:
    """FrankaGripper with the RPyC connection replaced by recorded, slow calls."""

    def __init__(self, call_duration_s=0.15):
        self.calls: list[str] = []
        self._duration = call_duration_s
        self._lock = threading.Lock()
        self.release = threading.Event()
        self.release.set()

        g = object.__new__(FrankaGripper)
        g.name = "test"
        g.do_print = False
        g._position_mm = MAX
        g._executor = ThreadPoolExecutor(max_workers=1)
        g._controller = object()
        g._rpc_grasp = self._make("grasp")
        g._rpc_open = self._make("open")
        g._rpc_home = self._make("home")
        g._is_open = True
        g._position_ts = None
        g._last_send = 0.0
        g._lock = threading.Lock()
        g._pending = None
        g._running = False
        g._inflight = None   # only read by the pre-fix driver
        self.g = g

    def _make(self, label):
        def rpc(*_args):
            with self._lock:
                self.calls.append(label)
            self.release.wait(timeout=5.0)
            time.sleep(self._duration)
        return rpc

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.calls)

    def idle(self, timeout=5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.g._lock:
                if not self.g._running and self.g._pending is None:
                    return True
            time.sleep(0.005)
        return False

    def close(self):
        self.release.set()
        self.g._executor.shutdown(wait=False)


def _wait(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_open_issued_during_a_grasp_still_reaches_the_gripper():
    """The reported failure: after closing, the gripper refuses to open.

    grasp() blocks for the whole motion. _submit used to DROP any command
    arriving while the worker was busy -- but move() had already committed
    `_is_open = True`, so the elif that issues open() never fired again and the
    gripper stayed shut until reconnect.
    """
    f = _FakeFranka()
    try:
        f.release.clear()                      # hold the grasp mid-motion
        f.g.move(0.0)                          # close
        assert _wait(lambda: f.snapshot() == ["grasp"]), f"grasp never ran: {f.snapshot()}"

        f.g._last_send = 0.0                   # bypass the 0.5 s debounce, not under test
        f.g.move(MAX)                          # open, while the grasp is still running
        assert f.g._is_open is True, "move() did not record the open intent"

        f.release.set()
        assert _wait(lambda: "open" in f.snapshot()), (
            f"open was dropped; gripper stays shut. calls={f.snapshot()}")
        assert f.idle()
        assert f.snapshot()[-1] == "open", f"last command was not the open: {f.snapshot()}"
    finally:
        f.close()
    return f"calls={f.snapshot()}"


def test_state_never_desyncs_from_the_last_executed_command():
    """_is_open is the driver's belief about the hardware; the last command the
    hardware actually ran must agree with it, whatever the timing."""
    f = _FakeFranka(call_duration_s=0.02)
    try:
        f.release.clear()
        for i in range(8):
            f.g._last_send = 0.0
            f.g.move(0.0 if i % 2 == 0 else MAX)
        f.release.set()
        assert f.idle()
        last = f.snapshot()[-1]
        want = "open" if f.g._is_open else "grasp"
        assert last == want, f"_is_open={f.g._is_open} but last command was {last}"
    finally:
        f.close()


def test_commands_coalesce_instead_of_queueing():
    """Latest-wins, not replay: a burst issued during one blocking motion must
    collapse to a single follow-up, or stale commands fire seconds late."""
    f = _FakeFranka()
    try:
        f.release.clear()
        f.g.move(0.0)
        assert _wait(lambda: len(f.snapshot()) == 1)
        for i in range(10):                    # burst while the worker is busy
            f.g._last_send = 0.0
            f.g.move(MAX if i % 2 == 0 else 0.0)
        f.release.set()
        assert f.idle()
        assert len(f.snapshot()) == 2, f"expected the burst to coalesce to one: {f.snapshot()}"
    finally:
        f.close()
    return f"10 commands during one motion -> {len(f.snapshot())} calls"


def test_position_is_stamped_when_the_motion_starts():
    """Position is dead-reckoned from _position_ts, so stamping it at queue time
    reports a motion that has not begun -- a queued command can sit behind a
    blocking grasp for most of a second."""
    f = _FakeFranka()
    try:
        f.release.clear()
        f.g.move(0.0)
        assert _wait(lambda: f.snapshot() == ["grasp"])
        f.g._last_send = 0.0
        f.g.move(MAX)                          # queued behind the grasp
        queued_ts = f.g._position_ts
        time.sleep(0.15)
        f.release.set()
        assert _wait(lambda: "open" in f.snapshot())
        assert f.g._position_ts is not None and f.g._position_ts > queued_ts, (
            "position clock started before the motion did")
    finally:
        f.close()


# --------------------------------------------------------------------------
# WSG -- still used by the bimanual scripts
# --------------------------------------------------------------------------


class FakeWSG:
    """Minimal GCL responder: records every command line it is sent."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.lines: list[str] = []
        self._lock = threading.Lock()
        self._conn: socket.socket | None = None
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        self._conn, _ = self.sock.accept()
        self._conn.settimeout(0.2)
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode().strip()
                if not line:
                    continue
                with self._lock:
                    self.lines.append(line)
                if line.startswith("FSACK"):
                    self.send("ACK FSACK")
                elif line.startswith("POS?"):
                    self.send("POS=42.0")

    def send(self, text: str):
        if self._conn is not None:
            try:
                self._conn.sendall((text + "\n").encode())
            except OSError:
                pass

    def commands(self, prefix: str) -> list[str]:
        with self._lock:
            return [ln for ln in self.lines if ln.startswith(prefix)]

    def clear(self):
        with self._lock:
            self.lines.clear()

    def close(self):
        self._stop.set()
        for s in (self._conn, self.sock):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass


def _connect_wsg():
    fake = FakeWSG()
    g = WSG(name="test", TCP_IP="127.0.0.1", TCP_PORT=fake.port, do_print=False)
    assert _wait(lambda: fake.commands("FSACK")), "driver never sent FSACK"
    fake.clear()
    return fake, g


def test_wsg_move_is_rate_capped():
    """The rate cap used to be unreachable: `dirty or elapsed >= cap and target
    is not None` binds `and` tighter than `or`, so a changed target bypassed the
    cap and every policy step preempted the previous motion plan."""
    fake, g = _connect_wsg()
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            g.move(20.0)
            time.sleep(0.01)
            g.move(90.0)
            time.sleep(0.01)
        time.sleep(0.15)
        moves = fake.commands("MOVE")
        ceiling = int(0.65 / WSG._MIN_MOVE_INTERVAL_S) + 2
        assert len(moves) <= ceiling, f"{len(moves)} MOVEs in 0.5 s, cap allows ~{ceiling}"
        assert len(moves) >= 2, f"rate cap starved the gripper entirely: {moves}"
    finally:
        g.close(); fake.close()
    return f"{len(moves)} MOVEs over 0.5 s at a {WSG._MIN_MOVE_INTERVAL_S}s cap"


def test_wsg_unchanged_target_is_not_resent():
    """A settled target must go quiet. Re-sending into a gripper already closed
    on a workpiece is what kept the axis faulting."""
    fake, g = _connect_wsg()
    try:
        g.move(60.0)
        assert _wait(lambda: fake.commands("MOVE"))
        time.sleep(0.05)
        fake.clear()
        time.sleep(0.5)
        assert fake.commands("MOVE") == [], f"re-sent a settled target: {fake.commands('MOVE')}"
        assert fake.commands("POS?"), "stopped polling position"
    finally:
        g.close(); fake.close()


def test_wsg_faulted_axis_is_cleared_before_the_next_move():
    """A MOVE that ends with the axis blocked -- every grasp -- leaves the WSG
    rejecting the next MOVE. move() is fire-and-forget, so the error was dropped
    and nothing cleared it."""
    fake, g = _connect_wsg()
    try:
        g.move(10.0)
        assert _wait(lambda: fake.commands("MOVE"))
        fake.send("ERR MOVE 12")
        assert _wait(lambda: g._move_error)
        fake.clear()

        g.move(100.0)
        assert _wait(lambda: fake.commands("MOVE")), "no MOVE issued after the fault"
        assert fake.commands("STOP"), "faulted axis was never cleared; gripper stays shut"
        assert fake.lines.index("STOP()") < next(
            i for i, ln in enumerate(fake.lines) if ln.startswith("MOVE")
        ), "STOP must precede the recovering MOVE"
        assert not g._move_error, "error flag not consumed"
    finally:
        g.close(); fake.close()


def test_wsg_no_stop_without_a_fault():
    """The clear is recovery, not routine traffic."""
    fake, g = _connect_wsg()
    try:
        g.move(30.0)
        assert _wait(lambda: fake.commands("MOVE"))
        time.sleep(0.15)
        g.move(80.0)
        assert _wait(lambda: len(fake.commands("MOVE")) >= 2)
        assert fake.commands("STOP") == [], "emitted STOP with no fault outstanding"
    finally:
        g.close(); fake.close()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            note = t()
            print(f"  PASS  {t.__name__}" + (f"   [{note}]" if note else ""))
        except Exception as e:  # noqa: BLE001
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f"  FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
