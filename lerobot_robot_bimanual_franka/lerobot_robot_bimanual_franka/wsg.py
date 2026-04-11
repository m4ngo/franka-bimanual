import socket
from time import time, sleep
from enum import Enum
import atexit

class WSG:
    def __init__(self, TCP_IP = "192.168.1.20", TCP_PORT = 1000):
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT 
        self.BUFFER_SIZE = 1024
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((TCP_IP, TCP_PORT))
        self.timeout = 10
        atexit.register(self.__del__)
        # Acknowledge fast stop from failure if any
        self.ack_fast_stop()

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
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"ACK FSACK\n")

    def set_verbose(self, verbose=True):
        """
        Set verbose True for detailed error messages
        """
        MESSAGE = str.encode(f"VERBOSE={1 if verbose else 0}\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(MESSAGE)

    def home(self):
        """
        Fully open the gripper
        """
        MESSAGE = str.encode("HOME()\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"FIN HOME\n")

    @property
    def position(self) -> float | None:
        """
        Get the current position of the gripper fingers in mm.
        Returns position as a float, or None on error/timeout.
        """
        MESSAGE = str.encode("POS?\n")
        self.tcp_sock.send(MESSAGE)

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

    def move(self, position):
        """
        Move fingers to specific position
        * position 0 :- fully close
        * position 110 :- fully open
        """
        MESSAGE = str.encode(f"MOVE({position})\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"FIN MOVE\n")

    def grip(self, force):
        MESSAGE = str.encode(f"GRIP({force})\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"FIN GRIP\n")

    def release(self):
        """
        Release: Release object by opening fingers by 10 mm.
        """
        MESSAGE = str.encode("RELEASE(10)\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"FIN RELEASE\n")

    def bye(self):
        MESSAGE = str.encode("BYE()\n")
        self.tcp_sock.send(MESSAGE)
        return

    def __del__(self):
        self.bye()