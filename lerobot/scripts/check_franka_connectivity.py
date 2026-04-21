"""Quick connectivity check for Franka control boxes before teleop.

Tests network reachability and RPC connection to each Franka server.
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


def check_tcp_connectivity(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """Check if a TCP port is reachable."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            return True
    except (socket.timeout, socket.gaierror, ConnectionRefusedError, OSError) as e:
        print(f"  ✗ TCP connection failed: {e}")
        return False


def check_rpyc_connection(server_ip: str, port: int, timeout_s: float = 3.0) -> bool:
    """Check if rpyc server is reachable."""
    try:
        import rpyc

        conn = rpyc.classic.connect(server_ip, port)
        conn.close()
        print(f"  ✓ RPC connection successful")
        return True
    except Exception as e:
        print(f"  ✗ RPC connection failed: {e}")
        return False


def check_robot_instance(robot_ip: str, server_ip: str, port: int, timeout_s: float = 5.0) -> bool:
    """Try to instantiate a remote Robot object (the actual blocker)."""
    try:
        from net_franky import setup_net_franky
        
        setup_net_franky(server_ip, port)
        
        from net_franky.franky import Robot

        robot = Robot(robot_ip)
        print(f"  ✓ Robot instance created successfully")
        return True
    except TimeoutError as e:
        print(f"  ✗ Robot instance creation timed out (likely unreachable robot IP {robot_ip}): {e}")
        return False
    except Exception as e:
        print(f"  ✗ Robot instance creation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check Franka connectivity")
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to check",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout per check in seconds")
    args = parser.parse_args()

    checks = {}

    if args.arm in ("left", "both"):
        l_server_ip = os.getenv("L_SERVER_IP", "192.168.3.11")
        l_robot_ip = os.getenv("L_ROBOT_IP", "192.168.200.2")
        l_port = int(os.getenv("L_PORT", 18813))

        print(f"\nLeft Arm (server={l_server_ip}:{l_port}, robot={l_robot_ip})")
        checks["left"] = all(
            [
                check_tcp_connectivity(l_server_ip, l_port, timeout_s=args.timeout),
                check_rpyc_connection(l_server_ip, l_port, timeout_s=args.timeout),
                check_robot_instance(l_robot_ip, l_server_ip, l_port, timeout_s=args.timeout),
            ]
        )

    if args.arm in ("right", "both"):
        r_server_ip = os.getenv("R_SERVER_IP", "192.168.3.10")
        r_robot_ip = os.getenv("R_ROBOT_IP", "192.168.201.10")
        r_port = int(os.getenv("R_PORT", 18812))

        print(f"\nRight Arm (server={r_server_ip}:{r_port}, robot={r_robot_ip})")
        checks["right"] = all(
            [
                check_tcp_connectivity(r_server_ip, r_port, timeout_s=args.timeout),
                check_rpyc_connection(r_server_ip, r_port, timeout_s=args.timeout),
                check_robot_instance(r_robot_ip, r_server_ip, r_port, timeout_s=args.timeout),
            ]
        )

    print("\n" + "=" * 60)
    if all(checks.values()):
        print("✓ All checks passed. Teleop should work.")
        return 0
    else:
        failed = [name for name, ok in checks.items() if not ok]
        print(f"✗ Checks failed for: {', '.join(failed)}")
        print("\nTroubleshooting:")
        print("  1. Verify servers are powered and control boxes are listening")
        print("  2. Verify network routes and firewalls allow traffic on above ports")
        print("  3. Check robot IP is reachable from the control box network")
        print("  4. Override env vars if IPs differ: L_SERVER_IP, L_ROBOT_IP, L_PORT, etc.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
