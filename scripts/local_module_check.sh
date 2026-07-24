# source /home/franka/.venv/bin/activate

# FRAMOS pyrealsense2 — NOT on PyPI; must be installed from the local FRAMOS SDK build.
uv pip install --no-build-isolation ~/librealsense2/wrappers/python/

# Extra deps not pulled in by lerobot[all]
uv pip install net_franky "PyGObject<3.52" pyspacemouse dynamixel-sdk

# Install (editable, with compat mode so editable installs play nice with namespace dirs)
uv pip install --no-deps -e ~/franka_ws/lerobot_camera_arv              -C editable_mode=compat
uv pip install --no-deps -e ~/franka_ws/lerobot_camera_framos           -C editable_mode=compat
uv pip install --no-deps -e ~/franka_ws/lerobot_robot_bimanual_franka   -C editable_mode=compat
uv pip install --no-deps -e ~/franka_ws/lerobot_teleoperator_gello      -C editable_mode=compat
uv pip install --no-deps -e ~/franka_ws/lerobot_teleoperator_spacemouse -C editable_mode=compat

# Uninstall
# uv pip uninstall lerobot_camera_arv lerobot_camera_framos \
#                  lerobot_robot_bimanual_franka \
#                  lerobot_teleoperator_gello lerobot_teleoperator_spacemouse

# Verify everything resolves to the uv venv
which python                  # should be /home/franka/.venv/bin/python
which lerobot-teleoperate     # should be /home/franka/.venv/bin/lerobot-teleoperate
