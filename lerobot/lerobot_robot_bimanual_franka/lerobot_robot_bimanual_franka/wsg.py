import socket
from time import time, sleep
import atexit
import threading
from concurrent.futures import Future, ThreadPoolExecutor


class WSG:
    """
    Schunk WSG gripper driver.

    All socket I/O is owned by a single background thread (_io_loop) to
    eliminate races between move commands and position reads.

    Public API
    ----------
    move(position)        — non-blocking, safe to call at any Hz
    position              — returns last cached value, never blocks
    grip / release / home — blocking one-shot commands (queued onto _io_loop)
    """

    def __init__(self, name="", TCP_IP="192.168.1.20", TCP_PORT=1000, do_print=False):
        self.name = name
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT
        self.BUFFER_SIZE = 4096
        self.timeout = 10
        self.do_print = do_print
        self._closed = False

        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((TCP_IP, TCP_PORT))
        self.tcp_sock.setblocking(True)

        # --- Cached position (written only by _io_loop) ---
        self._cached_position: float | None = None
        self._position_lock = threading.Lock()

        # --- Streaming move state ---
        self._move_target: int | None = None       # latest requested target
        self._move_sent: int | None = None         # last target actually sent
        self._move_lock = threading.Lock()
        self._move_event = threading.Event()

        # --- One-shot command queue (home / grip / release / ack) ---
        # Each item: (bytes_to_send, expected_response_bytes, done_event, result_list)
        self._cmd_queue: list[tuple[bytes, bytes, threading.Event, list]] = []
        self._cmd_lock = threading.Lock()
        self._cmd_event = threading.Event()

        # Single thread that owns the socket.
        self._io_thread = threading.Thread(
            target=self._io_loop, daemon=True, name=f"wsg-io-{name}"
        )
        self._io_thread.start()

        # Acknowledge fast stop from failure if any
        self.ack_fast_stop()

    # ------------------------------------------------------------------
    # Internal: socket owner thread
    # ------------------------------------------------------------------

    def _io_loop(self):
        """
        Single thread that owns all socket reads and writes.

        Priority order each iteration:
          1. Drain any one-shot command from the queue (blocking until ACK).
          2. If a new move target differs from what was sent, send MOVE.
          3. Poll position at ~10 Hz so `self.position` stays fresh.
          4. Non-blocking drain to parse any arrived data.
        """
        MOVE_INTERVAL_S = 0.05   # 20 Hz max move command rate
        POS_INTERVAL_S  = 0.05   # 10 Hz position refresh
        last_move_t = 0.0
        last_pos_t  = 0.0

        recv_buf = b""

        while not self._closed:
            now = time()

            # 1. One-shot commands take priority (home, grip, release, ack)
            cmd_entry = None
            with self._cmd_lock:
                if self._cmd_queue:
                    cmd_entry = self._cmd_queue.pop(0)

            if cmd_entry is not None:
                cmd_bytes, expected, done_event, result_holder = cmd_entry
                self.tcp_sock.send(cmd_bytes)
                ok, recv_buf = self._blocking_wait(expected, recv_buf)
                result_holder.append(ok)
                done_event.set()
                continue  # restart loop — don't interleave with moves

            # 2. Send move command if target changed or re-assert interval elapsed
            with self._move_lock:
                target = self._move_target
                sent   = self._move_sent

            if target is not None and (target != sent or now - last_move_t >= MOVE_INTERVAL_S * 5):
                if now - last_move_t >= MOVE_INTERVAL_S:
                    cmd = f"MOVE({target})\n".encode()
                    try:
                        self.tcp_sock.send(cmd)
                        with self._move_lock:
                            self._move_sent = target
                        last_move_t = now
                        if self.do_print:
                            print(f"{self.name}: [WSG] → MOVE({target})")
                    except OSError as e:
                        if self.do_print:
                            print(f"{self.name}: [WSG] send error: {e}")

            # 3. Periodic position poll
            if now - last_pos_t >= POS_INTERVAL_S:
                try:
                    self.tcp_sock.send(b"POS?\n")
                    last_pos_t = now
                except OSError as e:
                    if self.do_print:
                        print(f"{self.name}: [WSG] pos send error: {e}")

            # 4. Non-blocking drain — parse whatever arrived
            recv_buf = self._drain_and_parse(recv_buf)

            sleep(0.01)

    def _drain_and_parse(self, buf: bytes) -> bytes:
        """Non-blocking read; parse complete lines from buffer."""
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

    def _handle_line(self, line: bytes):
        """Dispatch a single parsed response line."""
        decoded = line.decode("utf-8", errors="replace")
        if decoded.startswith("POS="):
            try:
                val = float(decoded.split("=")[1])
                with self._position_lock:
                    self._cached_position = val
            except (IndexError, ValueError):
                pass
        elif decoded.startswith("ERR") and self.do_print:
            print(f"{self.name}: [WSG] Error: {decoded}")

    def _blocking_wait(self, expected: bytes, buf: bytes) -> tuple[bool, bytes]:
        """
        Block (within the io thread) until `expected` is seen or timeout.
        Parses and caches POS lines seen while waiting.
        Returns (success, remaining_buf).
        """
        since = time()
        while True:
            if expected in buf:
                idx = buf.index(expected) + len(expected)
                return True, buf[idx:]

            try:
                chunk = self.tcp_sock.recv(self.BUFFER_SIZE)
                if chunk:
                    buf += chunk
                    # Parse lines, keep looking for expected token
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._handle_line(line.strip())
                        if expected.rstrip(b"\n") in line:
                            return True, buf
                    if expected in buf:
                        return True, buf[buf.index(expected) + len(expected):]
            except OSError:
                return False, buf

            if time() - since >= self.timeout:
                if self.do_print:
                    print(f"{self.name}: [WSG] Timeout waiting for {expected!r}")
                return False, buf

            sleep(0.005)

    def _enqueue_cmd(self, cmd: bytes, expected: bytes) -> bool:
        """Queue a one-shot command and block the *caller* until it completes."""
        done = threading.Event()
        result: list[bool] = []
        with self._cmd_lock:
            self._cmd_queue.append((cmd, expected, done, result))
        self._cmd_event.set()
        done.wait(timeout=self.timeout + 1)
        return bool(result and result[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def position(self) -> float | None:
        """
        Last known finger gap in mm. Updated at ~10 Hz by the io loop.
        Never blocks — safe to call every observation cycle.
        """
        with self._position_lock:
            return self._cached_position

    def move(self, position: int, blocking: bool = False):
        """
        Move fingers toward *position* (mm). Non-blocking by default.

        Safe to call at any frequency — only the latest target is ever sent.
        The io loop sends at up to 20 Hz; intermediate values are dropped.

        * position 0   – fully closed
        * position 110 – fully open
        """
        with self._move_lock:
            self._move_target = int(position)
        self._move_event.set()

        if blocking:
            return self._enqueue_cmd(
                f"MOVE({int(position)})\n".encode(), b"FIN MOVE\n"
            )

    def ack_fast_stop(self):
        return self._enqueue_cmd(b"FSACK()\n", b"ACK FSACK\n")

    def set_verbose(self, verbose=True):
        """Set verbose True for detailed error messages."""
        msg = f"VERBOSE={1 if verbose else 0}\n".encode()
        return self._enqueue_cmd(msg, msg)

    def home(self):
        """Fully open the gripper (blocking)."""
        return self._enqueue_cmd(b"HOME()\n", b"FIN HOME\n")

    def home_async(self) -> threading.Thread:
        t = threading.Thread(target=self.home, daemon=True)
        t.start()
        return t

    def grip(self, force, blocking: bool = True):
        future_fn = lambda: self._enqueue_cmd(
            f"GRIP({force})\n".encode(), b"FIN GRIP\n"
        )
        if blocking:
            return future_fn()
        t = threading.Thread(target=future_fn, daemon=True)
        t.start()
        return t

    def release(self, blocking: bool = True):
        """Release: open fingers by 10 mm."""
        future_fn = lambda: self._enqueue_cmd(b"RELEASE(10)\n", b"FIN RELEASE\n")
        if blocking:
            return future_fn()
        t = threading.Thread(target=future_fn, daemon=True)
        t.start()
        return t

    def bye(self):
        try:
            self.tcp_sock.send(b"BYE()\n")
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._move_event.set()
        self._cmd_event.set()
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