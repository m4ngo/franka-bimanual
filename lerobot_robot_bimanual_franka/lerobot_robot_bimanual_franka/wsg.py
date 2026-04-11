import socket
from time import time, sleep
import atexit
from concurrent.futures import Future, ThreadPoolExecutor

class WSG:
    def __init__(self, TCP_IP = "192.168.1.20", TCP_PORT = 1000):
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT 
        self.BUFFER_SIZE = 1024
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((TCP_IP, TCP_PORT))
        self.timeout = 10
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wsg")
        atexit.register(self.__del__)
        # Acknowledge fast stop from failure if any
        self.ack_fast_stop()

    def _submit(self, fn, *args) -> Future:
        if self._closed:
            raise RuntimeError("WSG connection is closed")
        return self._executor.submit(fn, *args)

    def _send_and_wait(self, message: bytes, expected: bytes) -> bool:
        self.tcp_sock.send(message)
        return self.wait_for_msg(expected)

    def _read_position(self) -> float | None:
        self.tcp_sock.send(b"POS?\n")

        since = time()
        while True:
            data = self.tcp_sock.recv(self.BUFFER_SIZE)
            decoded = data.decode("utf-8").strip()
            if decoded.startswith("POS="):
                try:
                    return float(decoded.split("=")[1])
                except (IndexError, ValueError) as e:
                    print(f"[WSG] Failed to parse position response: {decoded} ({e})")
                    return None
            elif decoded.startswith("ERR"):
                print(f"[WSG] Error: {decoded}")
                return None
            if time() - since >= self.timeout:
                print(f"[WSG] Timeout ({self.timeout} s) occurred.")
                return None
            sleep(0.1)

    def wait_for_msg(self, msg) -> bool:
        since = time()
        ret: bool = False
        while True:
            data = self.tcp_sock.recv(self.BUFFER_SIZE)
            if data == msg:
                ret = True
                break
            elif data.decode("utf-8").startswith("ERR"):
                print(f"[WSG] Error: {data}")
                break
            if time() - since >= self.timeout:
                print(f"[WSG] Timeout ({self.timeout} s) occurred.")
                break
            sleep(0.1)
        return ret

    def ack_fast_stop(self):
        MESSAGE = str.encode("FSACK()\n")
        return self._submit(self._send_and_wait, MESSAGE, b"ACK FSACK\n").result()

    def set_verbose(self, verbose=True):
        """
        Set verbose True for detailed error messages
        """
        MESSAGE = str.encode(f"VERBOSE={1 if verbose else 0}\n")
        return self._submit(self._send_and_wait, MESSAGE, MESSAGE).result()

    def home(self):
        """
        Fully open the gripper
        """
        MESSAGE = str.encode("HOME()\n")
        return self._submit(self._send_and_wait, MESSAGE, b"FIN HOME\n").result()

    def home_async(self) -> Future:
        MESSAGE = str.encode("HOME()\n")
        return self._submit(self._send_and_wait, MESSAGE, b"FIN HOME\n")

    @property
    def position(self) -> float | None:
        """
        Get the current position of the gripper fingers in mm.
        Returns position as a float, or None on error/timeout.
        """
        return self._submit(self._read_position).result()

    def position_async(self) -> Future:
        return self._submit(self._read_position)

    def move(self, position: int, blocking: bool = True):
        """
        Move fingers to specific position
        * position 0 :- fully close
        * position 110 :- fully open
        """
        MESSAGE = str.encode(f"MOVE({position})\n")
        future = self._submit(self._send_and_wait, MESSAGE, b"FIN MOVE\n")
        return future.result() if blocking else future

    def grip(self, force, blocking: bool = True):
        MESSAGE = str.encode(f"GRIP({force})\n")
        future = self._submit(self._send_and_wait, MESSAGE, b"FIN GRIP\n")
        return future.result() if blocking else future

    def release(self, blocking: bool = True):
        """
        Release: Release object by opening fingers by 10 mm.
        """
        MESSAGE = str.encode("RELEASE(10)\n")
        future = self._submit(self._send_and_wait, MESSAGE, b"FIN RELEASE\n")
        return future.result() if blocking else future

    def bye(self):
        MESSAGE = str.encode("BYE()\n")
        self.tcp_sock.send(MESSAGE)
        return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.bye()
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.tcp_sock.close()
        except Exception:
            pass

    def __del__(self):
        self.close()