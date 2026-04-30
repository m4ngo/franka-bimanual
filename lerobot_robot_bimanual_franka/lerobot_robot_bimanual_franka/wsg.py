"""Schunk WSG (GCL) gripper driver — responsive position streaming.

The design is built around the single goal of making the gripper react
the instant a new target is commanded.

Architecture (one socket, two threads):

* Reader thread  blocks on ``recv``, parses every line, updates the
  cached position / gripper state, and fires waiters for blocking
  commands (HOME, GRIP, …) when their expected response token shows
  up.

* Sender thread  sleeps on a ``Condition`` and only wakes when there
  is something to send.  It services one-shot commands FIFO,
  dispatches ``MOVE`` only when the requested target *actually
  changes* (so the WSG is never carpet-bombed with overlapping
  motion plans), and polls ``POS?`` at a low rate to keep the cache
  fresh.

* ``move()`` hot path  clamps the value, stores it as the latest
  target, and notifies the sender.  The sender wakes within
  microseconds and the ``MOVE`` hits the wire immediately.  Repeat
  calls with the same (or near-same) target are a no-op, which is
  what keeps the gripper from feeling laggy: each commanded motion
  runs to completion instead of being interrupted by another MOVE
  every loop tick.

Public API:
    move(position, blocking=False)   stream a target position (mm)
    position                         last cached gap (mm), never blocks
    gripper_state                    last cached GRIPSTATE int
    grip / release / home            blocking one-shot commands
    ack_fast_stop / set_verbose / bye / close
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Waiter:
    """Pending blocking command waiting on a response token."""

    expected: bytes
    event: threading.Event = field(default_factory=threading.Event)


class WSG:
    # Motion / range tuning.
    MOVE_SPEED_MM_S = 420.0
    GRIPPER_MIN_MM = 10.0
    GRIPPER_MAX_MM = 100.0
    DEFAULT_TIMEOUT_S = 10.0

    # Rate caps.  ``_MIN_MOVE_INTERVAL_S`` keeps overlapping motion plans
    # from piling up at the gripper, ``_TARGET_CHANGE_THRESH_MM`` absorbs
    # noisy teleop input without an actual dead-zone.
    _MIN_MOVE_INTERVAL_S = 0.050       # ~20 Hz max MOVE rate
    _TARGET_CHANGE_THRESH_MM = 0.5
    _POS_POLL_INTERVAL_S = 0.050       # ~20 Hz POS? poll
    _SOCK_RECV_TIMEOUT_S = 0.5
    _RECV_BUF_SIZE = 4096
    _RELEASE_PULL_BACK_MM = 10.0
    _CLOSE_JOIN_TIMEOUT_S = 1.0

    def __init__(
        self,
        name: str = "",
        TCP_IP: str = "192.168.1.20",
        TCP_PORT: int = 1000,
        do_print: bool = False,
    ):
        self.name = name
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT
        self.do_print = do_print

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((TCP_IP, TCP_PORT))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(self._SOCK_RECV_TIMEOUT_S)
        self._send_lock = threading.Lock()  # serialises socket writes

        # Reader-owned cached state.
        self._state_lock = threading.Lock()
        self._position_mm: float | None = None
        self._gripper_state: int | None = None

        # All sender state — the streaming target, queued one-shots,
        # and pending waiters — live behind ``_cond``'s lock.
        self._cond = threading.Condition()
        self._target_mm: float | None = None
        self._last_sent_target_mm: float | None = None
        self._last_move_send_t: float = 0.0
        self._cmd_queue: deque[tuple[bytes, _Waiter | None]] = deque()
        self._waiters: deque[_Waiter] = deque()

        self._closed = threading.Event()

        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"wsg-rd-{name}"
        )
        self._sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True, name=f"wsg-tx-{name}"
        )
        self._reader_thread.start()
        self._sender_thread.start()

        # Clear any latched FAST STOP from a previous session.
        self.ack_fast_stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def position(self) -> float | None:
        """Last known finger gap in mm; never blocks."""
        with self._state_lock:
            return self._position_mm

    @property
    def gripper_state(self) -> int | None:
        """Last known GRIPSTATE integer; never blocks."""
        with self._state_lock:
            return self._gripper_state

    def move(self, position_mm: float, blocking: bool = False) -> bool:
        """Stream a new target position (mm).

        Non-blocking by default: the latest target is published to the
        sender thread which wakes immediately and emits ``MOVE`` on the
        wire.  Repeat calls with the same target are coalesced.

        * position 10  – fully closed (lower clamp)
        * position 100 – fully open  (upper clamp)
        """
        target = self._clip_target(position_mm)
        with self._cond:
            self._target_mm = target
            self._cond.notify()

        if blocking:
            return self._await_command(self._move_cmd(target), b"FIN MOVE")
        return True

    def home(self) -> bool:
        """Reference the gripper fingers (blocking)."""
        return self._await_command(b"HOME()\n", b"FIN HOME")

    def home_async(self) -> threading.Thread:
        t = threading.Thread(target=self.home, daemon=True)
        t.start()
        return t

    def grip(self, force_n: float, blocking: bool = True):
        """Grasp with ``force_n`` Newtons.  Blocks by default."""
        cmd = f"GRIP({force_n})\n".encode()
        if blocking:
            return self._await_command(cmd, b"FIN GRIP")
        return self._fire_and_forget(cmd, b"FIN GRIP")

    def release(self, blocking: bool = True):
        """Release a previously gripped part.  Blocks by default."""
        cmd = f"RELEASE({self._RELEASE_PULL_BACK_MM})\n".encode()
        if blocking:
            return self._await_command(cmd, b"FIN RELEASE")
        return self._fire_and_forget(cmd, b"FIN RELEASE")

    def ack_fast_stop(self) -> bool:
        return self._await_command(b"FSACK()\n", b"ACK FSACK")

    def set_verbose(self, verbose: bool = True) -> bool:
        msg = f"VERBOSE={1 if verbose else 0}\n".encode()
        return self._await_command(msg, msg.rstrip(b"\n"))

    def bye(self) -> None:
        """Announce disconnect so the gripper does not raise FAST STOP."""
        try:
            self._send_raw(b"BYE()\n")
        except Exception:
            pass

    def close(self) -> None:
        if self._closed.is_set():
            return
        # Send BYE before tearing the threads down so the gripper does
        # not latch a FAST STOP on abrupt disconnect.
        try:
            self.bye()
        except Exception:
            pass
        self._closed.set()
        with self._cond:
            self._cond.notify_all()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        self._reader_thread.join(timeout=self._CLOSE_JOIN_TIMEOUT_S)
        self._sender_thread.join(timeout=self._CLOSE_JOIN_TIMEOUT_S)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        buf = b""
        while not self._closed.is_set():
            try:
                chunk = self._sock.recv(self._RECV_BUF_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._handle_line(line.strip())

    def _handle_line(self, line: bytes) -> None:
        if not line:
            return
        text = line.decode("utf-8", errors="replace")

        if text.startswith("POS="):
            try:
                with self._state_lock:
                    self._position_mm = float(text.split("=", 1)[1])
            except ValueError:
                pass
        elif text.startswith("GRIPSTATE="):
            try:
                with self._state_lock:
                    self._gripper_state = int(text.split("=", 1)[1])
            except ValueError:
                pass
        elif text.startswith("ERR") and self.do_print:
            print(f"{self.name}: [WSG] {text}")

        # Match the oldest waiter whose expected token appears in this
        # line.  One match per line; subsequent waiters keep waiting.
        with self._cond:
            for waiter in self._waiters:
                if waiter.expected in line:
                    waiter.event.set()
                    self._waiters.remove(waiter)
                    return

    # ------------------------------------------------------------------
    # Sender thread
    # ------------------------------------------------------------------

    def _sender_loop(self) -> None:
        last_pos_poll_t = 0.0
        while not self._closed.is_set():
            cmd_to_send: bytes | None = None
            with self._cond:
                if self._closed.is_set():
                    break

                if self._cmd_queue:
                    cmd_bytes, waiter = self._cmd_queue.popleft()
                    if waiter is not None:
                        # Register before send so the response cannot
                        # arrive before the waiter is published.
                        self._waiters.append(waiter)
                    cmd_to_send = cmd_bytes
                else:
                    now = time.monotonic()
                    target_dirty = self._target_dirty_locked()

                    if (
                        target_dirty
                        and (now - self._last_move_send_t) >= self._MIN_MOVE_INTERVAL_S
                    ):
                        target = self._target_mm
                        self._last_sent_target_mm = target
                        self._last_move_send_t = now
                        cmd_to_send = self._move_cmd(target)
                    elif (now - last_pos_poll_t) >= self._POS_POLL_INTERVAL_S:
                        cmd_to_send = b"POS?\n"
                        last_pos_poll_t = now
                    else:
                        next_move = (
                            self._last_move_send_t + self._MIN_MOVE_INTERVAL_S
                            if target_dirty
                            else float("inf")
                        )
                        next_poll = last_pos_poll_t + self._POS_POLL_INTERVAL_S
                        timeout_s = max(0.0, min(next_move, next_poll) - now)
                        self._cond.wait(timeout=timeout_s)
                        continue

            if cmd_to_send is not None:
                self._send_raw(cmd_to_send)

    def _target_dirty_locked(self) -> bool:
        """True if the streaming target meaningfully differs from the
        last value we sent on the wire.  Caller must hold ``_cond``."""
        if self._target_mm is None:
            return False
        if self._last_sent_target_mm is None:
            return True
        return abs(self._target_mm - self._last_sent_target_mm) >= self._TARGET_CHANGE_THRESH_MM

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _clip_target(cls, position_mm: float) -> float:
        return float(min(max(position_mm, cls.GRIPPER_MIN_MM), cls.GRIPPER_MAX_MM))

    @classmethod
    def _move_cmd(cls, target_mm: float) -> bytes:
        print("moeving to", target_mm)
        return f"MOVE({target_mm:.2f},{cls.MOVE_SPEED_MM_S:.1f})\n".encode()

    def _send_raw(self, data: bytes) -> bool:
        try:
            with self._send_lock:
                self._sock.sendall(data)
            return True
        except OSError as e:
            if self.do_print:
                print(f"{self.name}: [WSG] send error: {e}")
            return False

    def _await_command(
        self,
        cmd: bytes,
        expected: bytes,
        timeout_s: float | None = None,
    ) -> bool:
        timeout_s = self.DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        waiter = _Waiter(expected=expected)
        with self._cond:
            self._cmd_queue.append((cmd, waiter))
            self._cond.notify()

        if not waiter.event.wait(timeout=timeout_s):
            with self._cond:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
            if self.do_print:
                print(f"{self.name}: [WSG] timeout waiting for {expected!r}")
            return False
        return True

    def _fire_and_forget(self, cmd: bytes, expected: bytes) -> threading.Thread:
        t = threading.Thread(
            target=lambda: self._await_command(cmd, expected),
            daemon=True,
        )
        t.start()
        return t
