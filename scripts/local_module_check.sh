conda activate lerobot

# Install (editable, with compat mode so editable installs play nice with namespace dirs)
pip install --no-deps -e ~/franka_ws/lerobot_camera_arv             -C editable_mode=compat
pip install --no-deps -e ~/franka_ws/lerobot_camera_framos          -C editable_mode=compat
pip install --no-deps -e ~/franka_ws/lerobot_robot_bimanual_franka  -C editable_mode=compat
pip install --no-deps -e ~/franka_ws/lerobot_teleoperator_gello     -C editable_mode=compat
pip install --no-deps -e ~/franka_ws/lerobot_teleoperator_spacemouse -C editable_mode=compat

# Uninstall
pip uninstall -y lerobot_camera_arv lerobot_camera_framos \
                 lerobot_robot_bimanual_franka \
                 lerobot_teleoperator_gello lerobot_teleoperator_spacemouse

# Verify everything resolves to the conda env
which python                  # should be /home/franka/miniforge3/envs/lerobot/bin/python
which lerobot-teleoperate     # should be /home/franka/miniforge3/envs/lerobot/bin/lerobot-teleoperate