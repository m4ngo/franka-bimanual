# LeRobot SpaceMouse Teleoperator

A [LeRobot](https://github.com/huggingface/lerobot) plugin for the 3Dconnexion
SpaceMouse, intended for end-effector (EE) delta control of robots that accept
6-DoF Cartesian twist commands.

Built as a thin wrapper around
[`pyspacemouse`](https://github.com/JakubAndrysek/PySpaceMouse).

## Action

`SpaceMouse.get_action()` returns a 7-element dict:

- `x`, `y`, `z`: linear velocity components (m/s)
- `roll`, `pitch`, `yaw`: angular velocity components (rad/s)
- `gripper`: target gripper position (mm)

Each device has two buttons. The left button closes the gripper (drives the
target to `gripper_min_mm`); the right button opens it (drives the target to
`gripper_max_mm`). The gripper target is latched, so the gripper holds its
last commanded state when no button is pressed.

## Hardware

- Tested with the 3Dconnexion SpaceMouse Compact (USB VID `0x256F`,
  PID `0xC635`). Any device supported by `pyspacemouse` should work.
- Read access to the corresponding `/dev/hidrawN` node (typically the
  `plugdev` group on Ubuntu).

## Installation

```bash
uv pip install -e ./lerobot_teleoperator_spacemouse
```

## Configuration

See `config_spacemouse.py` for available options: `hidraw_path`,
`translation_scale`, `rotation_scale`, and gripper min/max/initial limits.
