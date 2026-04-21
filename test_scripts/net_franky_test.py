from typing import Any

import numpy as np

from net_franky import setup_net_franky


SERVER_IP = "192.168.3.11"
ARM_IP = "192.168.200.2"
PORT = 18813


def main() -> None:
    setup_net_franky(SERVER_IP, PORT)

    from net_franky.franky import Affine, CartesianMotion, ReferenceType, Robot

    robot = Robot(ARM_IP)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.05

    translation: Any = np.array([[0.0], [0.0], [0.02]], dtype=np.float64)
    motion = CartesianMotion(target=Affine(translation=translation), reference_type=ReferenceType.Relative)
    robot.move(motion, asynchronous=False)
    translation: Any = np.array([[0.0], [0.0], [-0.02]], dtype=np.float64)
    motion = CartesianMotion(target=Affine(translation=translation), reference_type=ReferenceType.Relative)
    robot.move(motion, asynchronous=False)
    translation: Any = np.array([[0.02], [0.0], [0.0]], dtype=np.float64)
    motion = CartesianMotion(target=Affine(translation=translation), reference_type=ReferenceType.Relative)
    robot.move(motion, asynchronous=False)
    translation: Any = np.array([[0.0], [0.02], [0.0]], dtype=np.float64)
    motion = CartesianMotion(target=Affine(translation=translation), reference_type=ReferenceType.Relative)
    robot.move(motion, asynchronous=False)
    translation: Any = np.array([[-0.02], [0.0], [0.0]], dtype=np.float64)
    motion = CartesianMotion(target=Affine(translation=translation), reference_type=ReferenceType.Relative)
    robot.move(motion, asynchronous=False)
    translation: Any = np.array([[0.0], [-0.02], [0.0]], dtype=np.float64)
    motion = CartesianMotion(target=Affine(translation=translation), reference_type=ReferenceType.Relative)
    robot.move(motion, asynchronous=False)


if __name__ == "__main__":
    main()