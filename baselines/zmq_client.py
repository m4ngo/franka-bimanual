"""ZMQ REQ client for the two baseline policy servers.

Both baselines run in their own conda env (baselines/README.md: they conflict
with each other and with ours), so the policy is a separate process and this is
the only thing that talks to it. One client serves both because the two servers
answer the same four request shapes:

    {"meta": True}   -> capability dict, see PolicyClient.meta
    {"reset": True}  -> {}
    {"obs": ...}     -> SAIL {"chunk": (N, act_dim)}
                        B-Spline {"bspline": ndarray, "bspline_meta": {...}}

The B-Spline half of that protocol is upstream's, unchanged: its
`policy_server_bspline.py` is a submodule file and we do not edit it. SAIL's
server (sail_bridge/policy_server.py) was written to match.

Ported from bspline_policy/real_env/yam_teleop/policies.py's RemotePolicy, which
is behind flask/teleop imports we cannot take here.
"""

from __future__ import annotations

import logging
import threading

import zmq

logger = logging.getLogger("baselines.zmq")

# recv_pyobj is pickle. Only ever point this at a server you started.
_RESET_TIMEOUT_MS = 1000


class PolicyTimeout(RuntimeError):
    """The server did not answer inside recv_timeout_ms.

    Fatal to the EPISODE -- an inference that never arrived means there is no
    plan to execute -- but not to the client: REQ_RELAXED below keeps the socket
    usable, so the next episode's reset() still works.
    """


class PolicyClient:
    def __init__(self, port: int, recv_timeout_ms: int, host: str = "localhost") -> None:
        # A REQ socket is not thread-safe and REQ/REP is strictly alternating.
        # The B-Spline planner requests from a worker thread while the main
        # thread may reset() or close(), so every use is serialised here rather
        # than relying on the caller to keep them apart.
        self._lock = threading.Lock()
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        # Strict REQ/REP alternation makes a timed-out request poison the socket:
        # every later send raises EFSM because the un-received reply is still
        # owed. RELAXED lets a new request supersede it, CORRELATE tags each one
        # so the superseded reply is dropped rather than answering the next
        # question with the previous answer.
        self._socket.setsockopt(zmq.REQ_RELAXED, 1)
        self._socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self._addr = f"tcp://{host}:{int(port)}"
        self._socket.connect(self._addr)
        logger.info("policy client connected to %s", self._addr)

    def request(self, payload: dict) -> dict:
        with self._lock:
            return self._request_locked(payload)

    def _request_locked(self, payload: dict) -> dict:
        self._socket.send_pyobj(payload)
        try:
            return self._socket.recv_pyobj()
        except zmq.error.Again as exc:
            raise PolicyTimeout(
                f"no reply from the policy server at {self._addr} within "
                f"{self._socket.getsockopt(zmq.RCVTIMEO)} ms. Is it running, and "
                f"is it the right backend for this rollout?"
            ) from exc

    def meta(self) -> dict:
        """What the loaded checkpoint declares about itself.

        This is the handshake the SAIL entrypoint resolves its control mode from,
        and the one both entrypoints resize camera frames against:

            backend          "sail" | "bspline"
            act_dim          int
            obs_key_shapes   {key: shape} -- rgb keys are (C, H, W)
            n_obs_steps      int
            precision_column bool          -- last action column is a label
            action_keys      [str]         -- sail; picks EE_DELTA vs EE_POS
            action_horizon   int           -- sail
            fac_enabled      bool          -- sail; gates EAG
            fac_horizon      int           -- sail
            action_format    str           -- bspline, e.g. "single_yam_rot6d"
            degree           int           -- bspline
        """
        rep = self.request({"meta": True})
        if not isinstance(rep, dict) or "backend" not in rep:
            raise RuntimeError(
                f"{self._addr} did not answer the meta handshake (got {rep!r}). "
                "Upstream's policy_server_bspline.py ignores unknown keys and "
                "replies {} -- start it through scripts/bspline_rollout.sh, which "
                "wraps it with the meta responder."
            )
        return rep

    def reset(self) -> None:
        """Start a new episode on the server.

        Deliberately short-timeout and tolerant, as upstream's RemotePolicy is:
        the ack carries no information, and REQ_RELAXED means a missed one does
        not poison the next request.
        """
        with self._lock:
            default = self._socket.getsockopt(zmq.RCVTIMEO)
            self._socket.setsockopt(zmq.RCVTIMEO, _RESET_TIMEOUT_MS)
            self._socket.send_pyobj({"reset": True})
            try:
                self._socket.recv_pyobj()
            except zmq.error.Again:
                logger.warning("policy server did not ack reset within %d ms",
                               _RESET_TIMEOUT_MS)
            finally:
                self._socket.setsockopt(zmq.RCVTIMEO, default)

    def close(self) -> None:
        # Waits for any in-flight request: closing under one aborts the send.
        with self._lock:
            self._socket.close()
            self._ctx.term()
