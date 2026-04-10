import asyncio
import pyspacemouse
import numpy as np
from wsg import WSG
from franky import Robot, CartesianVelocityMotion, Twist

TRANS_SCALE = 0.1
ROT_SCALE = 0.5
DEAD_ZONE = 0.0

async def gripper_worker(q: asyncio.Queue):
    """Runs gripper commands one at a time in order."""
    loop = asyncio.get_event_loop()
    while True:
        fn, args = await q.get()
        await loop.run_in_executor(None, fn, *args)
        q.task_done()

async def create_controller(robot_ip, gripper_ip, mouse_index):
    loop = asyncio.get_event_loop()

    # Setup (blocking) — run in executor so we don't block the event loop
    gripper = await loop.run_in_executor(None, lambda: WSG(TCP_IP=gripper_ip))
    robot = await loop.run_in_executor(None, lambda: Robot(robot_ip))
    robot.relative_dynamics_factor = 0.05
    await loop.run_in_executor(None, robot.recover_from_errors)

    gq = asyncio.Queue()
    asyncio.create_task(gripper_worker(gq))

    # Queue home on startup
    await gq.put((gripper.home, ()))

    is_grip = False

    with pyspacemouse.open(device_index=mouse_index) as device:
        print(f"[{robot_ip}] Ready.")
        while True:
            # Read spacemouse in executor (blocking)
            state = await loop.run_in_executor(None, device.read)
            if not state:
                await asyncio.sleep(0)
                continue

            tx, ty, tz = state.y, -state.x, state.z
            rx, ry, rz = state.roll, state.pitch, state.yaw

            if state.buttons[0] and not is_grip:
                await gq.put((gripper.grip, (40,)))
                is_grip = True
            elif state.buttons[1] and is_grip:
                await gq.put((gripper.release, ()))
                await gq.put((gripper.home, ()))
                is_grip = False

            if np.linalg.norm([tx, ty, tz, rx, ry, rz]) < DEAD_ZONE:
                await asyncio.sleep(0)
                continue

            twist = Twist(
                [tx * TRANS_SCALE, ty * TRANS_SCALE, tz * TRANS_SCALE],
                [rx * ROT_SCALE,   ry * ROT_SCALE,  -rz * ROT_SCALE],
            )
            await loop.run_in_executor(
                None, lambda t=twist: robot.move(CartesianVelocityMotion(target=t), asynchronous=True)
            )
            await asyncio.sleep(0.01)

async def main():
    configs = [
        dict(robot_ip="192.168.200.2",  gripper_ip="192.168.2.21", mouse_index=3),
        dict(robot_ip="192.168.201.10", gripper_ip="192.168.2.20", mouse_index=7),
    ]

    try:
        await asyncio.gather(*[create_controller(**c) for c in configs])
    except KeyboardInterrupt:
        print("Stopping all robots.")

if __name__ == "__main__":
    asyncio.run(main())