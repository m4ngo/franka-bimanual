from wsg import WSG
import numpy as np
from time import sleep
# from franky import *

# def test_franka_arms():
#     left = Robot("192.168.200.2")
#     right = Robot("192.168.201.10")

#     # lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits
#     left.relative_dynamics_factor = 0.05
#     right.relative_dynamics_factor = 0.05

#     # Move the robot 20cm along the relative X-axis of its end-effector
#     # print(left.current_pose)
#     motion = CartesianMotion(Affine([0.0, 0.0, 0.01]), ReferenceType.Relative)
#     left.move(motion, asynchronous=False)
    
#     motion = CartesianMotion(Affine([0.01, 0.0, 0.0]), ReferenceType.Relative)
#     # right.move(motion, asynchronous=False)

def test_gripper(wsg: WSG):
    wsg.move(40, False)
    # print(wsg.position)

def test_grippers():
    # test_gripper(WSG(TCP_IP="192.168.2.21"))  # left
    # test_gripper(WSG(TCP_IP="192.168.2.20"))  # right
    WSG(TCP_IP="192.168.2.21").move(50, False)  # left
    WSG(TCP_IP="192.168.2.20").move(50, True)  # right

# test_franka_arms()
test_grippers()