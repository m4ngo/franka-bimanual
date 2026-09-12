#!/usr/bin/env python3
"""Block until a baseline policy server answers the meta handshake.

    python scripts/_wait_policy_server.py <backend> <port> [timeout_s]

Used by sail_rollout.sh and bspline_rollout.sh instead of a fixed sleep: loading
a diffusion checkpoint onto the GPU takes tens of seconds and varies, and a
sleep that is too short fails as a confusing timeout inside the rollout instead
of here.
"""

from __future__ import annotations

import sys
import time

import zmq


def main() -> int:
    backend, port = sys.argv[1], int(sys.argv[2])
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.setsockopt(zmq.LINGER, 0)
    # Without RELAXED a timed-out probe poisons the socket for every later one.
    sock.setsockopt(zmq.REQ_RELAXED, 1)
    sock.setsockopt(zmq.REQ_CORRELATE, 1)
    sock.connect(f"tcp://localhost:{port}")

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        sock.send_pyobj({"meta": True})
        try:
            rep = sock.recv_pyobj()
        except zmq.error.Again:
            continue
        got = rep.get("backend") if isinstance(rep, dict) else None
        if got == backend:
            print(f"{backend} policy server ready on port {port}")
            return 0
        if got is not None:
            print(f"port {port} is a {got!r} server, not {backend!r}", file=sys.stderr)
            return 1
        # Upstream's bspline server replies {} to anything it does not know, so
        # an empty dict means it is up but not wrapped with a meta responder.
        print(f"port {port} answered without a backend: {rep!r}", file=sys.stderr)
        return 1
    print(f"no {backend} policy server on port {port} after {timeout:.0f}s",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
