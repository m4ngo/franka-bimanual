from net_franky import setup_net_franky
# Connect to remote server
setup_net_franky("192.168.3.11", 18813)

from net_franky.franky import Robot, CartesianMotion, Affine, ReferenceType

robot = Robot("192.168.200.2")  # Replace this with your robot's IP

# Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
robot.relative_dynamics_factor = 0.05

# Move the robot 20cm along the relative X-axis of its end-effector
motion = CartesianMotion(Affine([-0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion)