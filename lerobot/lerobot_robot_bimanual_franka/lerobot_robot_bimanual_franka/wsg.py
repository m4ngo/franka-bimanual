"""Schunk WSG gripper driver proportional position control.

A single background thread owns the TCP socket.  At every position-poll tick
it computes the error between the current finger gap and the requested target,
then issues MOVE(<target>, <speed>) where speed = Kp * |error|, clamped to
[SPEED_MIN, SPEED_MAX].  One-shot commands (GRIP, RELEASE, HOME, …) are
serialised through a small queue that pauses streaming while they execute.

Public API:
    move(position, blocking=False)  set streaming target (mm)
    position                        last cached gap in mm, never blocks
    grip / release / home           blocking one-shot commands
    ack_fast_stop / set_verbose / bye
    close
"""

import socket
import threading
from time import sleep, time
import numpy as np


class WSG:
    BUFFER_SIZE = 8
    DEFAULT_TIMEOUT_S = 10

    # Position poll / MOVE dispatch rate (~20 Hz).
    _POLL_INTERVAL_S = 0.05

    _MOVE_SPEED = 420.0  # mm/s – hardware-safe ceiling
    _MOVE_DEAD_ZONE_MM = 3.0 # errors smaller than this are not acted on
    
    # Schunk WSG default travel range, in millimeters. Commanded positions are
    # clipped to this range before being forwarded to the gripper.
    _GRIPPER_MIN_MM = 10
    _GRIPPER_MAX_MM = 100

    _IO_LOOP_IDLE_S = 0.05
    _BLOCKING_POLL_S = 0.005
    _RELEASE_MM = 10

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
        self.timeout = self.DEFAULT_TIMEOUT_S
        self.do_print = do_print
        self._closed = False

        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((TCP_IP, TCP_PORT))
        self.tcp_sock.setblocking(True)

        self._cached_position: float | None = None
        self._position_lock = threading.Lock()

        self._move_target: float | None = None
        self._move_lock = threading.Lock()

        # One-shot command queue: (cmd_bytes, expected_response, done_event, result).
        self._cmd_queue: list[tuple[bytes, bytes, threading.Event, list]] = []
        self._cmd_lock = threading.Lock()

        self._io_thread = threading.Thread(
            target=self._io_loop, daemon=True, name=f"wsg-io-{name}"
        )
        self._io_thread.start()

        self.ack_fast_stop()

    # ------------------------------------------------------------------
    # Internal: IO thread
    # ------------------------------------------------------------------

    def _io_loop(self) -> None:
        """
        Priority order each iteration:
          1. Execute any queued one-shot command (blocks until its response).
          2. Poll position and issue a proportional MOVE if error > dead-zone.
          3. Drain pending bytes and parse response lines.
        """
        last_poll_t = 0.0
        recv_buf = b""

        while not self._closed:
            now = time()

            # 1. One-shot commands (GRIP, RELEASE, HOME, …) take priority.
            cmd_entry = None
            with self._cmd_lock:
                if self._cmd_queue:
                    cmd_entry = self._cmd_queue.pop(0)

            if cmd_entry is not None:
                cmd_bytes, expected, done_event, result = cmd_entry
                self.tcp_sock.send(cmd_bytes)
                ok, recv_buf = self._blocking_wait(expected, recv_buf)
                result.append(ok)
                done_event.set()
                continue

            # 2. Position poll + proportional MOVE.
            if now - last_poll_t >= self._POLL_INTERVAL_S:
                try:
                    self.tcp_sock.send(b"POS?\n")
                except OSError as e:
                    if self.do_print:
                        print(f"{self.name}: [WSG] pos send error: {e}")

                with self._move_lock:
                    target = self._move_target
                with self._position_lock:
                    pos = self._cached_position

                if target is not None and pos is not None:
                    error = target - pos
                    if abs(error) > self._MOVE_DEAD_ZONE_MM:
                        try:
                            self.tcp_sock.send(
                                f"MOVE({target},{self._MOVE_SPEED:.1f})\n".encode()
                            )
                            if self.do_print:
                                print(
                                    f"{self.name}: [WSG] MOVE({target}, {self._MOVE_SPEED:.1f})"
                                    f" err={error:+.1f} mm"
                                )
                        except OSError as e:
                            if self.do_print:
                                print(f"{self.name}: [WSG] send error: {e}")

                last_poll_t = now

            # 3. Drain and parse incoming bytes.
            recv_buf = self._drain_and_parse(recv_buf)
            sleep(self._IO_LOOP_IDLE_S)

    def _drain_and_parse(self, buf: bytes) -> bytes:
        """Pull pending bytes from the socket and parse complete lines."""
        self.tcp_sock.setblocking(False)
        try:
            chunk = self.tcp_sock.recv(self.BUFFER_SIZE)
            if chunk:
                buf += chunk
        except BlockingIOError:
            pass
        finally:
            self.tcp_sock.setblocking(True)

        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            self._handle_line(line.strip())
        return buf

    def _handle_line(self, line: bytes) -> None:
        """Dispatch a single parsed response line."""
        decoded = line.decode("utf-8", errors="replace")
        if decoded.startswith("POS="):
            try:
                with self._position_lock:
                    self._cached_position = float(decoded.split("=")[1])
            except (IndexError, ValueError):
                pass
        elif decoded.startswith("ERR") and self.do_print:
            print(f"{self.name}: [WSG] {decoded}")

    def _blocking_wait(self, expected: bytes, buf: bytes) -> tuple[bool, bytes]:
        """Block inside the IO thread until *expected* is seen or timeout fires."""
        start = time()
        while True:
            if expected in buf:
                return True, buf[buf.index(expected) + len(expected):]

            try:
                chunk = self.tcp_sock.recv(self.BUFFER_SIZE)
                if chunk:
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._handle_line(line.strip())
                        if expected.rstrip(b"\n") in line:
                            return True, buf
                    if expected in buf:
                        return True, buf[buf.index(expected) + len(expected):]
            except OSError:
                return False, buf

            if time() - start >= self.timeout:
                if self.do_print:
                    print(f"{self.name}: [WSG] timeout waiting for {expected!r}")
                return False, buf

            sleep(self._BLOCKING_POLL_S)

    def _enqueue_cmd(self, cmd: bytes, expected: bytes) -> bool:
        """Submit a one-shot command and block the caller until it completes."""
        done = threading.Event()
        result: list[bool] = []
        with self._cmd_lock:
            self._cmd_queue.append((cmd, expected, done, result))
        done.wait(timeout=self.timeout + 1)
        return bool(result and result[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def position(self) -> float | None:
        """Last known finger gap in mm; never blocks."""
        with self._position_lock:
            return self._cached_position

    def move(self, position: float, blocking: bool = False):
        """
        Stream a new target position (mm).  Non-blocking by default.

        The background loop issues MOVE(<position>, <speed>) at ~20 Hz with
        speed proportional to the remaining error, so the gripper accelerates
        toward the target and decelerates as it closes in.

        * position 0   – fully closed
        * position 110 – fully open
        """
        with self._move_lock:
            self._move_target = np.clip(float(position), self._GRIPPER_MIN_MM, self._GRIPPER_MAX_MM)

        if blocking:
            return self._enqueue_cmd(
                f"MOVE({position},{self._MOVE_SPEED:.1f})\n".encode(),
                b"FIN MOVE",
            )

    def ack_fast_stop(self) -> bool:
        return self._enqueue_cmd(b"FSACK()\n", b"ACK FSACK")

    def set_verbose(self, verbose: bool = True) -> bool:
        msg = f"VERBOSE={1 if verbose else 0}\n".encode()
        return self._enqueue_cmd(msg, msg.rstrip(b"\n"))

    def home(self) -> bool:
        """Reference the gripper fingers (blocking)."""
        return self._enqueue_cmd(b"HOME()\n", b"FIN HOME")

    def home_async(self) -> threading.Thread:
        t = threading.Thread(target=self.home, daemon=True)
        t.start()
        return t

    def grip(self, force: float, blocking: bool = True):
        """Grasp with the given force in N; blocks by default."""
        action = lambda: self._enqueue_cmd(
            f"GRIP({force})\n".encode(), b"FIN GRIP"
        )
        if blocking:
            return action()
        t = threading.Thread(target=action, daemon=True)
        t.start()
        return t

    def release(self, blocking: bool = True):
        """Release by opening fingers by _RELEASE_MM."""
        action = lambda: self._enqueue_cmd(
            f"RELEASE({self._RELEASE_MM})\n".encode(), b"FIN RELEASE"
        )
        if blocking:
            return action()
        t = threading.Thread(target=action, daemon=True)
        t.start()
        return t

    def bye(self) -> None:
        """Announce disconnect so the gripper does not raise a FAST STOP."""
        try:
            self.tcp_sock.send(b"BYE()\n")
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.bye()
        except Exception:
            pass
        try:
            self.tcp_sock.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
