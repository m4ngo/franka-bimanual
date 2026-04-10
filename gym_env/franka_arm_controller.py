from wsg import WSG
import numpy as np
from franky import *

relative_dynamics: float = 0.05
max_grip_open: int = 110


class FrankaArmController():
    
    def __init__(self,
                 robot_ip: str,
                 gripper_ip: str):
        # initialize arm
        self.robot: Robot = Robot(robot_ip)
        self.relative_dynamics_factor = relative_dynamics

        # initialize gripper
        self.gripper: WSG = WSG(TCP_IP=gripper_ip)

    def move_arm(self, motion: np.ndarray[8]):
        twist = Twist(
            [motion[0], motion[1], motion[2]],
            [motion[3], motion[4], motion[5]],
        )
        m = CartesianVelocityMotion(target=twist)
        self.robot.move(m, ReferenceType.Relative)
        self.gripper.move(motion[6])
