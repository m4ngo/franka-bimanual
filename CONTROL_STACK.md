# Bimanual Franka Control Stack: Delta-Action Computation Graph

This document traces the full computation path from a delta-action input to
the real-time wire calls in franky/libfranka, with exact file and line
citations.  The sysid replay path (`sysid/sysid.py`) is the primary entry
point used throughout, but all three action modes (delta, absolute EE, joint)
are covered.

---

## 0. Hardware Topology (reference)

```
franka@deepblue (workstation)
  ↕ RPyC  (TCP, port 18812 / 18813)
mario@192.168.3.10  (NUC — right arm)        luigi@192.168.3.11  (NUC — left arm)
  └─ robot 192.168.201.10  (FR3 right)         └─ robot 192.168.200.2  (FR3 left)
  └─ gripper 192.168.2.20  (WSG/FrankaGripper) └─ gripper 192.168.2.21
```

The workstation never talks to the robot directly; all real-time motion runs
on the NUC, which has a 1 kHz FCI/UDP connection to the robot controller.

---

## 1. Entry Point — `sysid/sysid.py` / `residual_wrapper/env_wrapper.py`

### 1.1 Controller construction

`env_wrapper.start_controller()` — [`residual_wrapper/env_wrapper.py:99-110`]

```python
config = SingleArmFrankaConfig(
    r_server_ip="192.168.3.10",   # NUC IP (RPyC server)
    r_robot_ip="192.168.201.10",  # FR3 IP (libfranka FCI target)
    r_gripper_ip="192.168.201.10",
    r_port=18812,
    use_ee_pos=False,
    use_delta=True,               # ← delta mode selected
)
robot = SingleArmFranka(config)
robot.connect()
```

`SingleArmFranka` is a thin subclass of `BimanualFranka`
([`lerobot_robot_bimanual_franka/single_arm_franka.py:5-7`]).

`BimanualFranka.connect()` calls
`MultiRobotWrapper.add_robot(arm, server_ip, robot_ip, port, use_ee_delta=True)`
which constructs a `RobotDriver` for each arm
([`bimanual_franka.py:148-155`]).

### 1.2 Per-step action assembly

`sysid._run_episode()` — [`sysid/sysid.py:144-193`]

At each control step `step`:

```python
# 1. Read kinematic state and cache it so send_action can reuse it.
kin = controller.robot_manager.current_kinematic_state_batch(["r"])
controller._cached_kin_state = kin          # sysid.py:158-159
q, dq, _jac, ee_pos, ee_quat, ee_vel = kin["r"]

# 2. Assemble the action dict (8 elements: x,y,z,qx,qy,qz,qw + gripper).
chunk_step = np.concatenate([goal_pos, goal_quat, [gripper_norm]])   # sysid.py:167
action = build_action(chunk_step, kp=kp, kd=kd)                     # sysid.py:168

# 3. Compute and cache residual delta offsets from the base-policy action.
dpos = action_all[step][0:3] * 0.05    # 5 cm/unit                  # sysid.py:169
drot = action_all[step][3:6] * 0.5     # 0.5 rad/unit               # sysid.py:170
controller.cache_delta(dpos, drot)      # stores in .delta_pos/.delta_rot

# 4. Execute.
controller.send_action(action, ignore_action=True)  # sysid.py:172
```

`build_action()` ([`env_wrapper.py:87-92`]) creates:
```python
{
  "r_x": goal_pos[0],  "r_y": goal_pos[1],  "r_z": goal_pos[2],
  "r_qx": goal_quat[0], "r_qy": goal_quat[1], "r_qz": goal_quat[2], "r_qw": goal_quat[3],
  "r_gripper": gripper_norm,
  "kp": kp,   # default 0.0  → kp_gain = 10^0 = 1.0
  "kd": kd,   # default 0.0  → kd_gain = 1.0
}
```

---

## 2. `BimanualFranka.send_action()` — top-level dispatch

[`lerobot_robot_bimanual_franka/bimanual_franka.py:259-288`]

```python
def send_action(self, action, ignore_action=False):
    # Re-use cached state if available; otherwise fetch.
    kin = self._cached_kin_state or self.robot_manager.current_kinematic_state_batch(...)
    self._cached_kin_state = None

    # Scale gains: kp ∈ [-1,1] → kp_gain = 10^kp;  kd ∈ [-1,1] → 1^(kd*2*sqrt(kp))
    kp_gain = _KP_GAIN_BASE ** np.clip(action["kp"], -1, 1)   # _KP_GAIN_BASE = 10.0
    kd_gain = _KD_GAIN_BASE ** (kd_clipped * 2 * np.sqrt(kp_clipped))  # _KD_GAIN_BASE = 1.0

    # ── Gripper ─────────────────────────────────────────────────────────────
    self.grippers[arm].move(action[f"{arm}_gripper"] * GRIPPER_TRUE_MAX_MM, blocking=False)

    # ── Arm (branch on mode) ─────────────────────────────────────────────────
    if self.use_delta:                       # ← sysid path
        cmds = safety.shape_ee(
            {arm: self._ee_delta(kp_gain, kd_gain, action, arm, kin[arm],
                                  self.delta_pos, self.delta_rot)
             for arm in self.active_arms}, kin)
        robot_manager.move_ee_delta_batch({a: c.tolist() for a, c in cmds.items()})

    elif self.use_ee_pos:                    # absolute EE setpoints
        cmds = safety.shape_ee(
            {arm: self._ee_pd(kp_gain, kd_gain, action, arm, kin[arm],
                               self.delta_pos, self.delta_rot, ignore_action)
             for arm in self.active_arms}, kin)
        robot_manager.move_ee_delta_batch(...)

    else:                                    # joint velocity
        cmds = safety.shape_joint(
            {arm: self._joint_pd(kp_gain, kd_gain, action, arm, kin[arm])
             for arm in self.active_arms}, kin)
        robot_manager.move_joint_velocity_batch(...)
```

---

## 3. Velocity Command Computation

### 3A. Delta mode — `_ee_delta()`

[`bimanual_franka.py:469-498`]

Produces a 6-DOF Cartesian twist **directly** from the action without
computing a goal-pose error.

```python
# Step 1: Extract position delta from action (metres).
action_dpos = [action["r_x"], action["r_y"], action["r_z"]]       # (3,)

# Step 2: Extract delta quaternion (xyzw) and convert to axis-angle.
dq = [action["r_qx"], action["r_qy"], action["r_qz"], action["r_qw"]]  # (4,)
v = dq[:3]
v_norm = ||v||
if v_norm < 1e-9:
    action_drot = 2 * v                                    # small-angle approximation
else:
    action_drot = (v / v_norm) * 2 * atan2(v_norm, dq[3]) # axis-angle from quat

# Step 3: Add cached delta offsets (from sysid residual scaling).
total_dpos = action_dpos + self.delta_pos                  # (3,)
total_drot = action_drot + self.delta_rot                  # (3,)

# Step 4: PD damping against current twist.
#   snap[5] = O_dP_EE_c from robot state (6D: [vx,vy,vz,wx,wy,wz])
twist_current = snap[5]
vel_cmd = (EE_PD_KP * kp_gain) * concat(total_dpos, total_drot)
        - (EE_PD_KD * kd_gain) * twist_current
# EE_PD_KP = 2.0,  EE_PD_KD = 0.1  (bimanual_franka.py:30-31)
# With default kp=kd=0: kp_gain=1.0, kd_gain=1.0
```

**Output**: `vel_cmd` ∈ ℝ⁶  (linear m/s || angular rad/s, base frame).

### 3B. Absolute EE mode — `_ee_pd()` → `_ee_velocity_toward_pose()`

[`bimanual_franka.py:460-466` and `438-449`]

Used when `use_ee_pos=True`.  Computes a goal-pose error first:

```python
# Position error (3,)
pos_error = target[:3] - ee_pos_current

# Rotation error (3,) — Hamilton product of target_quat * conj(current_quat)
# then convert to axis-angle; hemisphere corrected by negating if q_err[3] < 0.
# See bimanual_franka.py:417-434 for the full quaternion algebra.
rot_error = axis_angle_from_quat_error(target[3:], current_quat)

# Optional additive delta offsets (same delta_pos/delta_rot as above).
pos_error += dpos
rot_error += drot

# If ignore_action: zero out pos_error and rot_error before damping.

vel_cmd = (EE_PD_KP * kp_gain) * concat(pos_error, rot_error)
        - (EE_PD_KD * kd_gain) * twist_current
```

### 3C. Joint mode — `_joint_pd()`

[`bimanual_franka.py:452-458`]

Used when `use_ee_pos=False` and `use_delta=False`.

```python
target = [action["r_joint_1"], ..., action["r_joint_7"]]   # (7,) desired joint angles
q, dq = snap[0], snap[1]                                    # current q, dq from state

vel_cmd = (JOINT_PD_KP * kp_gain) * (target - q)
        - (JOINT_PD_KD * kd_gain) * dq
# JOINT_PD_KP = 2.0,  JOINT_PD_KD = 0.1
```

**Output**: `vel_cmd` ∈ ℝ⁷  (joint velocities, rad/s).

---

## 4. Safety Shaping — `ActionSafetyScreen`

[`lerobot_robot_bimanual_franka/safety.py`]

All three modes pass through a safety screen before dispatch.

### 4A. Worktable brake — `_apply_worktable_brake()`

[`safety.py:67-116`]

```
constants:
  WORKTABLE_HEIGHT        = 0.12 m  (base-frame Z of table surface)
  WORKTABLE_DISTANCE_MIN  = 0.03 m  (minimum clearance)
  WORKTABLE_MAX_DECEL     = 0.5  m/s²
  WORKTABLE_VELOCITY_EPS  = 1e-4 m/s

for each arm:
  ee_z     = kin_state[arm][3][2]          # snap.ee_pos.z (base frame)
  safe_dist = ee_z - WORKTABLE_HEIGHT - WORKTABLE_DISTANCE_MIN

  v_envelope = sqrt(2 * MAX_DECEL * safe_dist)   # allowed downward speed
  v_actual_z = J[2,:] @ dq                        # current EE Z from Jacobian
  v_commanded_z = twist[2]        (EE mode)
               OR J[2,:] @ action (joint mode)

  if v_commanded_z >= -VELOCITY_EPS:
      pass  # moving up or stopped, no braking needed

  v_target_z = max(v_commanded_z, -v_envelope)
  if -v_actual_z > v_envelope:
      v_target_z = max(v_target_z, 0.0)   # already exceeding → zero out

  EE mode:   twist[2] = v_target_z          (only Z modified)
  Joint mode: action  *= v_target_z / v_commanded_z  (uniform scale)
```

### 4B. Velocity norm clamps

[`safety.py:22-38`]

```
EE mode (shape_ee):
  ||linear||  capped to EE_LINEAR_VELOCITY_MAX  = 0.30 m/s
  ||angular|| capped to EE_ANGULAR_VELOCITY_MAX = 1.20 rad/s

Joint mode (shape_joint):
  ||joint_vel|| (L2) capped to JOINT_VELOCITY_MAX = 2.0 rad/s
```

---

## 5. RPyC Dispatch — `MultiRobotWrapper` → `RobotDriver`

### 5.1 `MultiRobotWrapper.move_ee_delta_batch()`

[`franka_process.py:214-215`]

```python
def move_ee_delta_batch(self, twists: dict[str, list]) -> None:
    self._gather(lambda n: self.drivers[n].send_velocity(twists[n]), list(twists))
```

`_gather()` submits one task per arm to a `ThreadPoolExecutor(max_workers=4)`,
so left and right arm calls are truly concurrent.

### 5.2 `RobotDriver.send_velocity()`

[`franka_process.py:157-169`]

```python
def send_velocity(self, vel: list[float]) -> None:
    rpc = self._rpc_send_ee if self.use_ee_delta else self._rpc_send_jv
    rpc(self.robot, tuple(vel))          # tuple! brine encodes by value
```

> **Wire-format constraint** (documented in module docstring and CLAUDE.md):
> `tuple()` is mandatory.  A `list` would cross the RPyC wire as a netref,
> requiring one round-trip per element access and triggering `AttributeError`
> spam when numpy probes `__array__` on the remote object.

The RPyC `classic.connect(server_ip, port)` connection was established at
`RobotDriver.__init__()` ([`franka_process.py:121-130`]).  The request
timeout is set to `RPYC_TIMEOUT_S = 10 s`.

Error recovery: if the RPyC call raises any string matching `_RECOVERABLE_ERRORS`
(UDP timeout, Reflex mode, motion-type conflict), `robot.recover_from_errors()`
is called on the remote and the error is logged as a warning rather than
re-raised ([`franka_process.py:149-155`]).

---

## 6. NUC-Side Helpers (executed via `_SERVER_HELPERS`)

At connect time, `_SERVER_HELPERS` is `exec()`'d on the NUC over RPyC
([`franka_process.py:51-107`]).  All motion construction therefore happens
server-side; the per-step round-trip is a single RPC call.

### 6.1 Robot initialisation — `init_robot()`

[`franka_process.py:67-76`]

```python
def init_robot(ip, ee):
    r = _cbm.CBRobot(ip)          # franky.Robot subclass, connects to FCI
    r.recover_from_errors()
    if ee:
        r.relative_dynamics_factor = RelativeDynamicsFactor(1.0, 0.3, 1.0)
        # (velocity_factor=1.0, acceleration_factor=0.3, jerk_factor=1.0)
    else:
        r.relative_dynamics_factor = RelativeDynamicsFactor(1.0, 0.25, 1.0)
    r.set_collision_behavior(100.0, 200.0)  # Nm, N thresholds
    r.set_joint_impedance([350, 350, 300, 500, 350, 150, 150])  # Nm/rad
    return r
```

`CBRobot(ip)` is defined in
`net_franky/cb_robot.py` and subclasses `franky.Robot`.  Its constructor
calls `franky.Robot.__init__(fci_hostname=ip)` which opens a TCP connection
to the robot controller (libfranka FCI).

### 6.2 `send_ee()` — Cartesian velocity dispatch

[`franka_process.py:97-99`]

```python
def send_ee(robot, twist):
    t = np.asarray(twist, dtype=np.float64)                   # (6,) [vx,vy,vz,wx,wy,wz]
    robot.move(
        fr.CartesianVelocityMotion(
            fr.Twist(t[:3], t[3:]),                            # linear, angular
            _DUR,                                              # Duration(100) ms
            _EE_DYN,                                           # RelativeDynamicsFactor(1,0.3,1)
        ),
        asynchronous=True,                                     # non-blocking
    )
```

`Duration(100)` means the motion target is active for 100 ms.  Because the
workstation issues a new command every ~50 ms (20 Hz), successive `move()`
calls preempt each other; franky re-plans the Ruckig trajectory on-the-fly.

### 6.3 `send_jv()` — joint velocity dispatch

[`franka_process.py:94-95`]

```python
def send_jv(robot, vel):
    robot.move(
        fr.JointVelocityMotion(np.asarray(vel, dtype=np.float64), _DUR),
        asynchronous=True,
    )
```

### 6.4 State reading — `get_state()`

[`franka_process.py:78-88`]

```python
def get_state(robot):
    with _cbm.state_mutex:
        s = _cbm.state              # latest HWState from CBRobot callback
    s = s.robot_state if s is not None else robot.state

    return (
        tuple(float(x) for x in s.q),           # joint positions (7,)
        tuple(float(x) for x in s.dq),          # joint velocities (7,)
        tuple(float(x) for x in s.O_T_EE.translation),   # EE pos (3,)
        tuple(float(x) for x in s.O_T_EE.quaternion),    # EE quat xyzw (4,)
        tuple(float(x) for x in s.O_dP_EE_c.linear)      # EE vel linear (3,)
        + tuple(float(x) for x in s.O_dP_EE_c.angular),  # EE vel angular (3,)
    )
```

> **Mutex pattern**: `CBRobot.state_mutex` guards the global `_cbm.state`
> object set by the 1 kHz franky callback.  If a previous session crashed
> while holding the mutex, the `_SERVER_HELPERS` preamble replaces the
> broken lock with a fresh one ([`franka_process.py:56-60`]).  The helper
> then reads `cb_robot.state` directly under a `with` block — **not** via
> `get_last_callback_data()`, which leaks the mutex on `AttributeError`.

### 6.5 Jacobian caching — `get_jacobian()`

[`franka_process.py:90-92`]

```python
def get_jacobian(robot):
    j = np.asarray(robot.model.zero_jacobian(fr.Frame.EndEffector, robot.state))
    return tuple(float(x) for x in j.flat)    # 6×7 = 42 floats
```

On the workstation side (`RobotDriver.get_kinematic_state()`
[`franka_process.py:136-142`]):

```python
if self._jac is None or np.max(np.abs(q - self._jac_q)) > _JACOBIAN_CACHE_Q_THRESHOLD:
    self._jac = np.array(self._rpc_jacobian(self.robot)).reshape(6, 7)
    self._jac_q = q.copy()
# _JACOBIAN_CACHE_Q_THRESHOLD = 0.50 rad (L∞)
```

The Jacobian is recomputed only when any joint angle drifts more than 0.5 rad
since the last query, saving an RPyC round-trip on most control steps.

---

## 7. `CBRobot.move()` — callback registration

[`~/.venv/lib/python3.12/site-packages/net_franky/cb_robot.py:40-43`]

```python
class CBRobot(Robot):    # Robot = franky.Robot
    def move(self, motion, asynchronous=False):
        motion.register_callback(hw_state_callback)   # registers 1 kHz callback
        super().move(motion, asynchronous=asynchronous)
```

`hw_state_callback` stores a `HWState(robot_state, time_step, rel_time,
abs_time, control_signal)` into the module-global `state` under `state_mutex`
on every control tick.  This is what `get_state()` reads.

---

## 8. franky C++ — Motion Objects and Ruckig OTG

`franky` is a C++ library (with pybind11 Python bindings) that wraps libfranka
and adds a high-level motion API.  Source: https://github.com/TimSchneider42/franky

### 8.1 `franky::CartesianVelocityMotion`

Inheritance chain:
```
CartesianVelocityMotion
  └─ CartesianVelocityWaypointMotion
       └─ VelocityWaypointMotion<franka::CartesianVelocities, RobotVelocity>
            └─ WaypointMotion<ControlSignalType, WaypointType, TargetType>
                 └─ Motion<franka::CartesianVelocities>
```

Constructor signature (franky docs):
```cpp
CartesianVelocityMotion(
    const RobotVelocity& target,          // Twist: linear + angular
    const franka::Duration& duration,     // 100 ms in this stack
    const RelativeDynamicsFactor& rdf,    // (vel=1.0, acc=0.3, jerk=1.0)
    const Affine& frame = Identity()
)
```

At each 1 kHz tick `WaypointMotion::nextCommandImpl()` calls `ruckig::Ruckig`
to step the online trajectory generator:

```
Input (from Ruckig):
  current_velocity   ← previous franka::CartesianVelocities (6D)
  target_velocity    ← Twist from the motion object
  max_velocity       ← robot.translation_velocity_limit * rdf.velocity
  max_acceleration   ← robot.translation_acceleration_limit * rdf.acceleration
  max_jerk           ← robot.translation_jerk_limit * rdf.jerk

Output:
  next_velocity      → franka::CartesianVelocities{vx,vy,vz,wx,wy,wz}
```

Ruckig generates **time-optimal, jerk-limited** velocity profiles; this is
what lets franky operate from a non-real-time Python process — the NUC thread
running the control loop regenerates a smooth trajectory even if the
workstation's RPyC commands arrive with jitter.

When the motion duration expires (100 ms) `motion_finished = true` is set and
libfranka terminates the control loop for that motion.  A subsequent
`robot.move(..., asynchronous=True)` preempts the running motion.

### 8.2 `franky::JointVelocityMotion`

Analogous structure:
```
JointVelocityMotion
  └─ JointVelocityWaypointMotion
       └─ VelocityWaypointMotion<franka::JointVelocities, Vector7d>
            └─ WaypointMotion<...>
                 └─ Motion<franka::JointVelocities>
```

Same Ruckig integration; the 7D joint velocity is smoothly ramped.

### 8.3 `franky::Robot::move()`

```cpp
void franky::Robot::move(
    const std::shared_ptr<Motion<franka::CartesianVelocities>>& motion,
    bool async = false)
```

Internally calls `libfranka::Robot::control(callback, ControllerMode::kJointImpedance)`.
When `async=true`, the control loop runs in a background thread; `pollMotion()`
/ `joinMotion()` can be used to query or block on completion.

---

## 9. libfranka — Real-Time 1 kHz UDP Control Loop

libfranka source: https://github.com/frankarobotics/libfranka

### 9.1 `franka::Robot::control()` — Cartesian velocity overload

```cpp
void franka::Robot::control(
    std::function<CartesianVelocities(const RobotState&, franka::Duration)>
        motion_generator_callback,
    ControllerMode controller_mode = ControllerMode::kJointImpedance,
    bool limit_rate = false,
    double cutoff_frequency = kDefaultCutoffFrequency   // 100 Hz Butterworth
)
```

Per-tick execution (1000 Hz, UDP over 192.168.201.x):

```
1. Receive RobotState packet from robot controller (UDP).
   RobotState contains:
     q[7]            joint positions (rad)
     dq[7]           joint velocities (rad/s)
     O_T_EE          EE homogeneous pose in base frame (4×4)
     O_dP_EE_c[6]    commanded EE twist [vx,vy,vz,wx,wy,wz]
     tau_J[7]        measured joint torques
     F_ext[6]        estimated external force/torque
     ... (many more fields)

2. Call motion_generator_callback(robot_state, duration) →
   franka::CartesianVelocities{vx, vy, vz, wx, wy, wz}
   (this is franky's Ruckig-stepped output)

3. Optional: apply rate limiter (disabled here: limit_rate=false).
   Optional: apply 100 Hz Butterworth low-pass filter (cutoff_frequency).

4. Send CartesianVelocities back to robot controller via UDP.

5. Set motion_finished if callback returns CartesianVelocities with
   motion_finished=true.
```

### 9.2 `franka::Robot::control()` — joint velocity overload

```cpp
void franka::Robot::control(
    std::function<JointVelocities(const RobotState&, franka::Duration)>
        motion_generator_callback,
    ControllerMode controller_mode = ControllerMode::kJointImpedance,
    bool limit_rate = false,
    double cutoff_frequency = kDefaultCutoffFrequency
)
```

Identical structure, returns `franka::JointVelocities{dq_0,...,dq_6}`.

### 9.3 Robot controller (onboard ARM, inside Franka Control Box)

The UDP packets reach the Franka robot controller, which is **not** libfranka
— it is the proprietary firmware running on a dedicated ARM processor.

With `ControllerMode::kJointImpedance`:

```
CartesianVelocities [vx,vy,vz,wx,wy,wz]
    ↓  Differential IK (J^† @ v_cartesian, where J is robot Jacobian)
JointVelocities_desired [dq_d_0,...,dq_d_6]
    ↓  Joint impedance law:
       τ_cmd = K_q * (q_d - q) + K_dq * (dq_d - dq) + τ_gravity
               ↑ set via set_joint_impedance([350,350,300,500,350,150,150] Nm/rad)
Joint torques τ_cmd [7]
    ↓  EtherCAT (1 kHz)
Joint motor drives (7× brushless PMSM)
    ↓  Field-oriented current control (~10 kHz, inside each drive)
Physical joint motion
```

For the joint velocity control mode the first step is skipped; the
`JointVelocities_desired` are used directly.

---

## 10. Gripper Paths

### 10A. WSG Schunk Gripper (`wsg.py`)

Used when `gripper_ip ≠ robot_ip`.

```
BimanualFranka.send_action()
  └─ WSG.move(target_mm, blocking=False)  [wsg.py:134-151]
       Sets self._target_mm and notifies _cond.
       ↓ (in sender thread, wakes within microseconds)
  WSG._sender_loop()  [wsg.py:272-312]
       Checks if |target - last_sent| > _TARGET_CHANGE_THRESH_MM (5 mm)
       If dirty: sends "MOVE(target_mm,420.0)\n" via TCP socket
       ↓  TCP  (port 1000 on gripper IP)
  WSG gripper GCL protocol
       Executes position-controlled jaw motion at MOVE_SPEED_MM_S=420 mm/s.
```

Rate-limiting: `_MIN_MOVE_INTERVAL_S = 0.1 s` (≤ 10 Hz MOVE commands) prevents
overlapping motion plans from building up on the gripper.

Position feedback: the sender polls `POS?` at 20 Hz; the reader thread parses
`POS=<mm>` responses and stores them in `_position_mm` under `_state_lock`.

### 10B. Franka Gripper (`franka_gripper.py`)

Used when `gripper_ip == robot_ip` (gripper is the integrated Franka Hand).

```
BimanualFranka.send_action()
  └─ FrankaGripper.move(target_mm)  [franka_gripper.py:86-98]
       Threshold logic: target < 40 mm → grasp(0.0, speed, force)
                        target > 40 mm → open(speed)
       ↓ (via ThreadPoolExecutor, non-blocking)
  FrankaGripper.grasp() / open()  [franka_gripper.py:115-119]
       _executor.submit(_rpc_grasp/open, self._controller, ...)
       ↓  RPyC to NUC (same connection as arm)
  grasp_gripper() / open_gripper() on NUC  [franka_gripper.py:46-55]
       controller.grasp_async(width_m, speed_m_s, force_n, epsilon_outer=1.0, ...)
       ↓  franky.Gripper → libfranka::Gripper
       libfranka::Gripper::grasp() / open() over Franka Gripper TCP
```

---

## 11. Full Annotated Computation Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WORKSTATION (franka@deepblue)                                               │
│                                                                             │
│  sysid.py:_run_episode()                                                    │
│    │                                                                        │
│    ├─ goal_pos (T,3) + goal_quat (T,4)    ← from trajectory HDF5           │
│    ├─ build_action() [env_wrapper.py:87]                                    │
│    │     → RobotAction dict {r_x,r_y,r_z,r_qx,...,r_qw,r_gripper,kp,kd}   │
│    ├─ dpos = action_all[step][0:3] * 0.05                                  │
│    ├─ drot = action_all[step][3:6] * 0.5                                   │
│    ├─ controller.cache_delta(dpos, drot)  [bimanual_franka.py:401]          │
│    │                                                                        │
│    └─ controller.send_action(action, ignore_action=True)                   │
│                                                                             │
│  BimanualFranka.send_action() [bimanual_franka.py:259]                     │
│    │                                                                        │
│    ├─ kin = _cached_kin_state  ← snapshot: (q,dq,J,ee_pos,ee_quat,twist)   │
│    ├─ kp_gain = 10^kp  (=1.0 for kp=0)                                     │
│    ├─ kd_gain = 1^(kd*2*√kp)  (=1.0 for kd=0)                             │
│    ├─ WSG/FrankaGripper.move(gripper_mm, blocking=False)                   │
│    │                                                                        │
│    └─ [use_delta=True path]                                                 │
│         _ee_delta(kp_gain, kd_gain, action, arm, snap,                     │
│                   delta_pos, delta_rot)  [bimanual_franka.py:469]           │
│           │                                                                 │
│           ├─ action_dpos = action[r_x, r_y, r_z]   (m)                    │
│           ├─ dq = action[r_qx, r_qy, r_qz, r_qw]  (delta quat xyzw)      │
│           ├─ action_drot = axis_angle(dq)           (rad)                  │
│           ├─ total_dpos = action_dpos + cache.delta_pos                    │
│           ├─ total_drot = action_drot + cache.delta_rot                    │
│           └─ vel_cmd = 2.0*kp_gain*[total_dpos,total_drot]                 │
│                      - 0.1*kd_gain*snap.twist          (ℝ⁶)                │
│                                                                             │
│  ActionSafetyScreen.shape_ee() [safety.py:49]                              │
│    │                                                                        │
│    ├─ _apply_worktable_brake():                                             │
│    │    safe_dist = ee_z - 0.12 - 0.03                                     │
│    │    v_envelope = √(2 * 0.5 * safe_dist)                                │
│    │    if vel_cmd[2] < -v_envelope: vel_cmd[2] = -v_envelope              │
│    │    if actual_z > v_envelope:   vel_cmd[2] = max(vel_cmd[2], 0)        │
│    └─ _clamp_ee_twist():                                                   │
│         ||linear||  ≤ 0.30 m/s                                             │
│         ||angular|| ≤ 1.20 rad/s                                           │
│                                                                             │
│  MultiRobotWrapper.move_ee_delta_batch() [franka_process.py:214]           │
│    └─ ThreadPoolExecutor: per-arm RobotDriver.send_velocity()              │
│                                                                             │
│  RobotDriver.send_velocity() [franka_process.py:157]                       │
│    └─ RPyC: _rpc_send_ee(robot, tuple(vel))     ← brine-encoded tuple      │
│                                                                             │
│  ─────────────── RPyC TCP (port 18812) ──────────────────────────────────  │
└─────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ NUC (mario@192.168.3.10)                                                    │
│                                                                             │
│  send_ee(robot, twist) [franka_process.py:97, exec'd from _SERVER_HELPERS] │
│    └─ robot.move(                                                           │
│           CartesianVelocityMotion(                                          │
│               Twist(t[:3], t[3:]),           # linear + angular             │
│               Duration(100),                 # 100 ms active window         │
│               RelativeDynamicsFactor(1,0.3,1) # vel/acc/jerk factors        │
│           ),                                                                │
│           asynchronous=True                  # non-blocking, background     │
│       )                                                                     │
│                                                                             │
│  CBRobot.move() [net_franky/cb_robot.py:40]                                │
│    ├─ motion.register_callback(hw_state_callback)   # 1 kHz state capture  │
│    └─ franky.Robot.move(motion, async=True)                                 │
│                                                                             │
│  franky::Robot::move() [franky C++]                                        │
│    └─ Spawns background thread, calls:                                      │
│       libfranka::Robot::control(                                            │
│           callback,                                                         │
│           ControllerMode::kJointImpedance,                                  │
│           limit_rate=false,                                                 │
│           cutoff_frequency=100Hz                                            │
│       )                                                                     │
│                                                                             │
│  Per-tick (1000 Hz):                                                        │
│    1. Ruckig OTG step: ramp current → target velocity                       │
│       respecting max_vel, max_acc, max_jerk from RelativeDynamicsFactor     │
│       → franka::CartesianVelocities {vx,vy,vz,wx,wy,wz}                   │
│                                                                             │
│  ─────────────── FCI UDP (192.168.201.10:1337) ─────────────────────────── │
└─────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRANKA CONTROL BOX (ARM processor, proprietary firmware)                    │
│                                                                             │
│  Receives CartesianVelocities {vx,vy,vz,wx,wy,wz}                          │
│                                                                             │
│  Differential IK (with ControllerMode::kJointImpedance):                   │
│    dq_d = J^† @ [vx,vy,vz,wx,wy,wz]      (Jacobian pseudoinverse)          │
│                                                                             │
│  Joint impedance controller:                                                │
│    τ_cmd = K_q*(q_d - q) + K_dq*(dq_d - dq) + τ_gravity                   │
│    K_q = [350,350,300,500,350,150,150] Nm/rad  (set via set_joint_impedance)│
│                                                                             │
│  ─────────────── EtherCAT (1 kHz) ─────────────────────────────────────── │
└─────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7× JOINT MOTOR DRIVES (brushless PMSM)                                      │
│   Field-oriented current control at ~10 kHz                                 │
│   Physical torque → joint acceleration → joint motion                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. State Readback Path

This is the reverse channel — how the robot's current state gets back to
Python before each control step.

```
Joint encoder / torque sensor  (1 kHz, inside robot)
    ↓  EtherCAT
Robot controller ARM
    ↓  FCI UDP (192.168.201.x → NUC)
libfranka RobotState struct (1 kHz) delivered to franky callback
    ↓
CBRobot.hw_state_callback() [net_franky/cb_robot.py:18-34]
    Stores HWState{robot_state, ...} in _cbm.state under state_mutex
    ↓
get_state(robot) [franka_process.py:78-88] — called via RPyC
    Reads _cbm.state under state_mutex, extracts:
        q (7)          from robot_state.q
        dq (7)         from robot_state.dq
        ee_pos (3)     from robot_state.O_T_EE.translation
        ee_quat (4)    from robot_state.O_T_EE.quaternion  (xyzw)
        ee_twist (6)   from robot_state.O_dP_EE_c.linear + .angular
    Returns as nested tuple of Python floats (brine-encodable)
    ↓  RPyC response
RobotDriver.get_kinematic_state() [franka_process.py:136-142]
    Unpacks tuples → numpy arrays
    Recomputes Jacobian if joint drift > 0.5 rad (L∞)
    Returns KinematicSnapshot = (q, dq, J, ee_pos, ee_rot, ee_twist)
    ↓
BimanualFranka._cached_kin_state  (consumed by send_action on next step)
```

---

## 13. Key Numerical Constants (all sources cited)

| Constant | Value | File : Line | Role |
|---|---|---|---|
| `EE_PD_KP` | 2.0 | `bimanual_franka.py:30` | EE proportional gain |
| `EE_PD_KD` | 0.1 | `bimanual_franka.py:31` | EE derivative (damping) |
| `JOINT_PD_KP` | 2.0 | `bimanual_franka.py:29` | Joint proportional gain |
| `JOINT_PD_KD` | 0.1 | `bimanual_franka.py:29` | Joint derivative |
| `_KP_GAIN_BASE` | 10.0 | `bimanual_franka.py:32` | Exponential gain base |
| `_KD_GAIN_BASE` | 1.0 | `bimanual_franka.py:33` | Damping gain base |
| `VELOCITY_COMMAND_DURATION_MS` | 100 | `franka_process.py:18` | franky motion duration |
| `_EE_DELTA_RELATIVE_DYNAMICS` | `(1.0, 0.3, 1.0)` | `franka_process.py:27` | Ruckig vel/acc/jerk factors (EE) |
| `_JOINT_RELATIVE_DYNAMICS` | `(1.0, 0.25, 1.0)` | `franka_process.py:26` | Ruckig factors (joint) |
| `_JOINT_STIFFNESS` | `[350,350,300,500,350,150,150]` | `franka_process.py:30` | Joint impedance (Nm/rad) |
| `_TORQUE_THRESHOLD` | 100.0 Nm | `franka_process.py:29` | Collision detection |
| `_FORCE_THRESHOLD` | 200.0 N | `franka_process.py:29` | Collision detection |
| `_JACOBIAN_CACHE_Q_THRESHOLD` | 0.50 rad | `franka_process.py:25` | Jacobian staleness threshold |
| `WORKTABLE_HEIGHT` | 0.12 m | `safety.py:11` | Z of table surface |
| `WORKTABLE_DISTANCE_MIN` | 0.03 m | `safety.py:13` | Minimum clearance |
| `WORKTABLE_MAX_DECEL` | 0.5 m/s² | `safety.py:14` | Braking deceleration |
| `EE_LINEAR_VELOCITY_MAX` | 0.30 m/s | `safety.py:18` | Safety clamp |
| `EE_ANGULAR_VELOCITY_MAX` | 1.20 rad/s | `safety.py:19` | Safety clamp |
| `JOINT_VELOCITY_MAX` | 2.0 rad/s | `safety.py:17` | Safety clamp (L2) |
| `_POS_SCALE` | 0.05 m/unit | `env_wrapper.py:13` | sysid delta scaling |
| `_ROT_SCALE` | 0.5 rad/unit | `env_wrapper.py:14` | sysid delta scaling |
| `WSG.MOVE_SPEED_MM_S` | 420 mm/s | `wsg.py:56` | Gripper motion speed |
| `WSG._TARGET_CHANGE_THRESH_MM` | 5.0 mm | `wsg.py:65` | Gripper coalescing |
| `FrankaGripper.GRIPPER_TRUE_MAX_MM` | 80 mm | `franka_gripper.py:22` | Franka Hand open width |
| `WSG.GRIPPER_TRUE_MAX_MM` | 110 mm | `wsg.py:55` | WSG open width |
| `RPYC_TIMEOUT_S` | 10 s | `franka_process.py:23` | RPyC request timeout |

---

## 14. Summary of Library Boundaries

| Layer | Code | Technology |
|---|---|---|
| Policy / sysid | `sysid/sysid.py`, `residual_wrapper/env_wrapper.py` | Python, NumPy |
| Robot abstraction | `lerobot_robot_bimanual_franka/bimanual_franka.py` | Python, LeRobot |
| Safety shaping | `lerobot_robot_bimanual_franka/safety.py` | Python, NumPy |
| RPyC client | `lerobot_robot_bimanual_franka/franka_process.py` | Python, RPyC |
| **─── TCP network boundary (192.168.3.x) ───** | | |
| RPyC server | `_SERVER_HELPERS` (exec'd on NUC) | Python, NumPy |
| State callback | `net_franky/cb_robot.py` | Python, franky pybind11 |
| High-level motion | `franky::CartesianVelocityMotion` / `JointVelocityMotion` | C++, Ruckig |
| Trajectory smoothing | Ruckig OTG | C++ |
| **─── FCI UDP boundary (192.168.201.x, 1 kHz) ───** | | |
| libfranka control loop | `franka::Robot::control()` | C++, UDP |
| **─── EtherCAT boundary (1 kHz) ───** | | |
| Franka firmware | Joint impedance controller, differential IK | Proprietary ARM |
| Motor drives | Field-oriented current control | Proprietary, ~10 kHz |
