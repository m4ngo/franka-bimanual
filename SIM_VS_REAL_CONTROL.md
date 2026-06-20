# Sim vs Real Control Stack: Differences and Alignment Analysis

This document compares the simulated controller
(`multi-fast/robosuite/robosuite/controllers/osc.py`) against the physical
control stack documented in `CONTROL_STACK.md`, identifying every
architectural difference and quantifying the gap for each.

---

## 1. One-Line Summary of Each Controller

| Dimension | Sim (OSC) | Real (franky/libfranka) |
|---|---|---|
| **Output** | Joint torques τ ∈ ℝ⁷ | Cartesian velocity v ∈ ℝ⁶ |
| **Feedback variable** | Persistent goal pose (incremented per step) | No persistent goal; delta → instantaneous velocity |
| **Inertia decoupling** | Yes — λ = (J M⁻¹ Jᵀ)⁻¹ | None at Python level |
| **Nullspace control** | Yes — drives toward initial joint config | None at Python level |
| **Gravity compensation** | Explicit, via MuJoCo `qfrc_bias` | Internal to Franka firmware |
| **Trajectory smoothing** | Linear pose interpolator (optional) | Ruckig OTG jerk-limited ramp |
| **Control frequency** | Policy: 20 Hz; physics substeps: 500 Hz | Control loop: ~20 Hz; servo: 1 kHz |

---

## 2. Control Law Derivation — Side by Side

### 2A. Input action format

Both controllers nominally expect a 6-DOF delta action per step.

**Sim** — after `scale_action()` ([`osc.py:237`], [`base_controller.py:104-123`]):
```
input ∈ [-1, 1]^6  →  scaled_delta ∈ ℝ^6
  [dx, dy, dz]  in metres  (output_max = ±0.05 m per axis)
  [ax, ay, az]  axis-angle (output_max = ±0.5 rad per axis)
```
After scaling, position and orientation deltas are accumulated into a
**persistent goal pose** before any PD error is computed.

**Real** — sysid path ([`sysid/sysid.py:163-171`], [`env_wrapper.py:87-92`]):
```
input = goal_pos (3,) + delta_quat (4,)  from base policy [physical units already]
dpos  = action_all[step][0:3] * 0.05     residual offset  (m)
drot  = action_all[step][3:6] * 0.5      residual offset  (rad)
```
No goal accumulation. Each step directly produces a velocity command.

---

### 2B. Goal / error computation

**Sim** ([`osc.py:259-312`], [`control_utils.py:114-236`]):

```
Step 1 — update goal (in set_goal()):
  goal_pos = current_ee_pos + scaled_delta[:3]      (persistent, accumulates)
  goal_ori = R_delta @ current_ee_ori               (persistent, accumulates)

Step 2 — compute errors (in run_controller()):
  pos_error = goal_pos - ee_pos                     (m)
  ori_error = 0.5 * Σ cross(rc_i, rd_i)            (rad, matrix formulation)
              where rc_i = columns of current rot mat
                    rd_i = columns of desired rot mat

Step 3 — desired wrench (task-space PD):
  F_des = kp[0:3] * pos_error + kd[0:3] * (-ee_vel_pos)
  T_des = kp[3:6] * ori_error + kd[3:6] * (-ee_vel_ori)
  kp = 150 (default, uniform), kd = 2*√kp*damping_ratio = 2*√150*1 ≈ 24.49
```

**Real** ([`bimanual_franka.py:469-498`]):

```
_ee_delta():
  action_dpos = action["r_x", "r_y", "r_z"]        (m, delta position)
  dq   = action["r_qx", "r_qy", "r_qz", "r_qw"]   (unit quaternion, delta rotation)
  action_drot = axis_angle(dq)                      convert delta quat → (3,) rad

  total_dpos = action_dpos + cached_delta_pos
  total_drot = action_drot + cached_delta_rot

  vel_cmd = (EE_PD_KP * kp_gain) * [total_dpos, total_drot]
           - (EE_PD_KD * kd_gain) * current_twist
  EE_PD_KP = 2.0,  EE_PD_KD = 0.1
  kp_gain  = 10^kp_param  (default kp_param=0 → kp_gain=1.0)
  kd_gain  = 1^(...)      (default kp_param=kd_param=0 → kd_gain=1.0)
```

---

### 2C. Task-space inertia decoupling

**Sim** ([`control_utils.py:43-82`]):

```
M_inv = inv(mass_matrix)                       7×7 joint-space inertia inverse
λ_full_inv = J_full M_inv J_full^T             6×6
λ_pos_inv  = J_pos  M_inv J_pos^T              3×3
λ_ori_inv  = J_ori  M_inv J_ori^T              3×3

λ_full = pinv(λ_full_inv)
λ_pos  = pinv(λ_pos_inv)
λ_ori  = pinv(λ_ori_inv)

# Decoupled (uncouple_pos_ori=True, default):
F_op = λ_pos * F_des                           force projected into op-space
T_op = λ_ori * T_des                           torque projected into op-space
wrench = [F_op, T_op]

# Joint torques from op-space wrench:
τ = J^T * wrench + τ_gravity
```

The inertia decoupling makes the effective mass seen in each Cartesian axis
approximately 1 kg (normalized), removing dynamic coupling between axes.

**Real**: No equivalent computation anywhere in the Python stack.  The robot
receives a raw velocity command.  Axis coupling and inertia effects are
handled by the Franka firmware's internal joint impedance controller —
which does *not* explicitly cancel inertia at the Cartesian level.

---

### 2D. Nullspace control

**Sim** ([`osc.py:349-352`], [`control_utils.py:8-40`]):

```
Jbar            = M_inv * J^T * λ_full            (dynamically consistent pseudoinverse)
N               = I - Jbar * J                    (nullspace projector)

joint_kp_null   = 10  (fixed)
joint_kv_null   = 2 * √joint_kp_null ≈ 6.32

pose_torques    = M * (kp_null * (q_init - q) - kv_null * dq)
τ_null          = N^T * pose_torques

τ_total        += τ_null
```

The nullspace term drives the arm back toward its initial joint configuration
without disturbing EE position or orientation, improving arm-away-from-limits
behavior over long trajectories.

**Real**: No nullspace control at the Python level.  `set_joint_impedance(
[350, 350, 300, 500, 350, 150, 150])` ([`franka_process.py:30`]) applies a
passive joint stiffness at rest (during a stop motion), but this has no
nullspace projection and does not operate during active Cartesian velocity
commands.

---

### 2E. Gravity compensation

**Sim** ([`base_controller.py:230-237`]):
```python
torque_compensation = sim.data.qfrc_bias[qvel_index]
τ_total += torque_compensation    # explicit Coriolis + gravity
```
MuJoCo's `qfrc_bias` includes gravity, Coriolis, and centrifugal forces
at each substep, so the commanded torques only need to cover the inertial
and error terms.

**Real**: Gravity and Coriolis compensation happen inside the Franka
controller (firmware), not at the Python level.  `set_joint_impedance()`
tunes the passive restoring stiffness; the firmware adds its own gravity
model.  The Python stack never observes nor adds a gravity term.

---

### 2F. Orientation error formulation

**Sim** ([`control_utils.py:85-111`]):
```python
# Rotation-matrix cross-product formula (Khatib 1987):
error = 0.5 * (cross(rc1, rd1) + cross(rc2, rd2) + cross(rc3, rd3))
# rc_i = columns of current R; rd_i = columns of desired R
```
This produces an axis-angle-like 3-vector in the base frame, with magnitude
approximately equal to the rotation angle for small errors.

**Real** ([`bimanual_franka.py:416-434`]):
```python
# Hamilton quaternion product: q_err = q_target ⊗ conj(q_current)
q_err = q_target ⊗ [-x_c, -y_c, -z_c, w_c]
if q_err[3] < 0: q_err = -q_err       # hemisphere selection
v = q_err[:3]; v_norm = ||v||
rot_error = (v/v_norm) * 2 * atan2(v_norm, q_err[3])   # axis-angle
```
Both produce a 3-vector in the same geometric direction for small errors.
The real implementation is numerically more stable at large rotation angles;
the sim implementation has a small-angle linearisation and may behave
differently for errors > ~30°.

---

### 2G. Feedback derivative term

**Sim** ([`osc.py:316-328`]):
```
vel_pos_error = -ee_pos_vel    # velocity read from MuJoCo site velocity
vel_ori_error = -ee_ori_vel    # angular velocity read from MuJoCo
F_des += kd[0:3] * vel_pos_error
T_des += kd[3:6] * vel_ori_error
```
`ee_pos_vel` and `ee_ori_vel` come from `sim.data.get_site_xvelp/xvelr()`
which are the true MuJoCo EE linear/angular velocities in the world frame.

**Real** ([`bimanual_franka.py:448`]):
```
twist_current = snap[5]   # from robot_state.O_dP_EE_c (6D: linear+angular)
vel_cmd -= EE_PD_KD * kd_gain * twist_current
```
`O_dP_EE_c` is the **commanded** (filtered/smoothed) EE twist from the
robot controller, not the raw encoder-derived velocity.  It may lag the
actual EE velocity by ~1–2 ms due to internal filtering.

---

### 2H. Control rates and substep structure

**Sim** ([`base.py:388-393`], [`macros.py:1`]):
```
policy_freq       = 20 Hz        (action input rate)
model_timestep    = 0.002 s      (MuJoCo physics step, 500 Hz)
n_substeps        = control_timestep / model_timestep = 25

Per policy step:
  1. set_goal(action)               once
  2. for i in range(25):            inner physics loop
       _pre_action(action, policy_step=(i==0))
       sim.step()
       # run_controller() is called each substep with the same goal
```
The OSC runs at full 500 Hz with the same goal pose, so torques are
continuously updated as the arm moves.  This inner loop provides implicit
smoothing — torques transition naturally as the arm approaches the goal.

**Real** ([`sysid/sysid.py:186-189`]):
```
policy_freq = 20 Hz    (default, controlled by dt = 1/fps)
per step:
  1. read state
  2. compute velocity command
  3. robot.move(CartesianVelocityMotion(..., Duration(100ms)), async=True)
  4. sleep until next step
```
The Ruckig OTG inside franky then runs at 1 kHz to smooth the 100 ms velocity
command, but it always ramps *toward* the same target for the full 100 ms.
There is no mid-step recomputation based on updated state.

---

### 2I. Goal pose interpolation

**Sim** — optional linear interpolator between policy steps
([`osc.py:268-276`, `osc.py:295-312`]):
```python
if self.interpolator_pos is not None:
    desired_pos = self.interpolator_pos.get_interpolated_goal()
    # linearly blends from current goal to new goal across substeps
```

**Real** — no pose-level interpolation.  Ruckig performs velocity-profile
smoothing (jerk-limited ramp), which is kinematically different from
interpolating a goal position.

---

## 3. Gain / Parameter Comparison

| Parameter | Sim (OSC default) | Real (sysid / delta mode) |
|---|---|---|
| Proportional gain | `kp = 150` (N/m or Nm/rad) | `EE_PD_KP = 2.0` (velocity/error) |
| Derivative gain | `kd = 2√150 ≈ 24.5` (critically damped) | `EE_PD_KD = 0.1` |
| Gain formula | `kd = 2 * √kp * damping_ratio` | Fixed constants |
| Gain scaling | Fixed per-session | `kp_gain = 10^kp_param` (log scale, action-modulated) |
| Position scale | ±0.05 m per normalized unit | Same (0.05 m/unit applied in sysid.py) |
| Rotation scale | ±0.5 rad per normalized unit | Same (0.5 rad/unit applied in sysid.py) |
| Nullspace kp | 10 (Nm/rad, in joint space) | — |
| Joint impedance | — (MuJoCo default spring-damper) | [350,350,300,500,350,150,150] Nm/rad |
| Velocity limits | None (torques clipped to actuator range) | 0.30 m/s linear, 1.20 rad/s angular |

The sim `kp = 150` is a **Cartesian-space force/torque gain**:
`F = 150 * pos_error_m` → `~7.5 N` for a 5 cm error.  On the real robot,
`EE_PD_KP = 2.0` is a **velocity gain**: `v = 2.0 * delta_m` → `0.10 m/s`
for a 5 cm delta.  These operate on fundamentally different physical
quantities (N vs m/s) and cannot be directly equated without knowing the
robot's effective Cartesian impedance.

---

## 4. Signal Flow Diagrams

### Sim (OSC) per policy step

```
Policy action (6D normalized delta)
    │
    ▼ scale_action() [osc.py:237]
    │   input ±1 → output ±(0.05 m, 0.5 rad)
    ▼
set_goal() [osc.py:202]
    │   goal_pos += delta[:3]
    │   goal_ori = R_delta @ current_ori
    ▼
[25× per policy step at 500 Hz]
run_controller() [osc.py:278]
    │
    ├─ pos_error = goal_pos - ee_pos                  (m)
    ├─ ori_error = 0.5*Σ cross(rc_i, rd_i)           (rad)
    │
    ├─ F_des = kp[0:3]*pos_err + kd[0:3]*(-vel_pos)  (N)
    ├─ T_des = kp[3:6]*ori_err + kd[3:6]*(-vel_ori)  (Nm)
    │
    ├─ λ_pos = pinv(J_pos M^-1 J_pos^T)              (kg)
    ├─ λ_ori = pinv(J_ori M^-1 J_ori^T)              (kg·m²)
    │
    ├─ wrench = [λ_pos*F_des, λ_ori*T_des]
    │
    ├─ τ = J^T * wrench + τ_gravity
    ├─ τ += N^T * M * (kp_null*(q_0-q) - kv_null*dq)
    ▼
sim.data.ctrl = τ (joint torques, 7D)
    │
    ▼ MuJoCo physics @ 500 Hz
Joint accelerations → velocities → positions
```

### Real (bimanual_franka, delta mode) per control step

```
Base policy output (6D: dpos_m, delta_quat) + residual (6D raw→ *0.05/*0.5)
    │
    ▼ build_action() + cache_delta() [env_wrapper.py:87, sysid.py:168]
    │
    ▼ send_action() [bimanual_franka.py:259]
    │
    ├─ kp_gain = 10^kp_param (=1.0 default)
    │
    ├─ WSG/FrankaGripper.move(target_mm)  ←─ gripper (independent path)
    │
    ├─ _ee_delta() [bimanual_franka.py:469]
    │    action_dpos = action[r_x, r_y, r_z]
    │    action_drot = axis_angle(delta_quat)
    │    total_dpos  = action_dpos + cached_dpos
    │    total_drot  = action_drot + cached_drot
    │    vel_cmd = 2.0*kp_gain*[total_dpos,total_drot]
    │            - 0.1*kd_gain*current_twist            (ℝ⁶, m/s + rad/s)
    │
    ├─ ActionSafetyScreen.shape_ee() [safety.py:49]
    │    worktable brake, ‖v_lin‖≤0.30, ‖v_ang‖≤1.20
    │
    ├─ RPyC → NUC: send_ee(robot, tuple(vel))
    │
    ├─ robot.move(CartesianVelocityMotion(Twist, 100ms, rdf(1,0.3,1)), async=True)
    │
    │  [1000 Hz on NUC — Ruckig OTG]
    │    Ramps velocity toward target in 100 ms
    │    Respects vel/acc/jerk limits * RelativeDynamicsFactor
    │    → franka::CartesianVelocities{vx,vy,vz,wx,wy,wz}
    │
    ├─ libfranka UDP → Robot controller
    │
    │  [1000 Hz — Franka firmware]
    │    Cartesian vel → joint vel (J† × v)
    │    Joint impedance: τ = K_q*(q_d-q) + K_dq*(dq_d-dq) + τ_grav
    │    K_q = [350,350,300,500,350,150,150] Nm/rad
    ▼
EtherCAT → Motor drives → Physical joint motion
```

---

## 5. Structural Differences Ranked by Impact

### 5.1 Control output type (torque vs velocity) — **HIGHEST IMPACT**

**Sim** outputs torques computed from the full dynamics model.  The mass matrix
and gravity compensation are derived from MuJoCo's internal model.

**Real** outputs Cartesian velocities; torque computation is black-box inside
the Franka firmware with fixed joint impedance parameters.

**Effect**: The sim controller can produce large transient torques to
accelerate the arm against gravity and inertia.  The real controller relies on
the firmware's impedance to translate velocities into torques, meaning
the real arm is always critically impedance-controlled — it resists being
pushed, but the commanded motion is velocity, not force.

---

### 5.2 Goal state persistence — **HIGH IMPACT**

**Sim** maintains `goal_pos` and `goal_ori` between policy steps.  If the arm
is disturbed or lags behind, the goal stays fixed and the error builds,
creating a restoring force/torque.

**Real** has no goal memory.  If no new command is sent, the arm stops
(`CartesianVelocityMotion` decays to zero after its 100 ms duration).
Persistent tracking of a target must be re-sent every step by the policy.

**Effect**: In sim, the arm naturally "catches up" after a perturbation.  On
the real robot, if the policy outputs a small delta (e.g., near a goal), the
arm slows and stops at whatever position it's at — not necessarily the goal.
This creates a tracking bias when the policy assumes the error is being
accumulated across steps (as the OSC does).

---

### 5.3 Inertia decoupling / operational space mapping — **HIGH IMPACT**

**Sim**: `F_cmd = λ * kp * error`.  The effective kp felt by the end-effector
is `kp` (N/m) regardless of which direction is commanded or the arm
configuration.

**Real**: `v_cmd = kp_gain * delta`.  Without λ, the arm's apparent Cartesian
compliance changes with configuration (because joint inertias project
differently into Cartesian space at each pose).  Commands in some directions
may be sluggish while others respond quickly.

---

### 5.4 Nullspace control — **MEDIUM IMPACT**

The OSC drives the arm toward `initial_joint` in the nullspace.  Over a long
trajectory the arm never drifts into joint limits or poor-conditioned
configurations.

The real stack has no nullspace term.  Over time the arm may drift to
configurations with poor Jacobian conditioning, making subsequent Cartesian
commands less effective.

---

### 5.5 Derivative gain coupling — **MEDIUM IMPACT**

**Sim**: `kd = 2 * √kp * damping_ratio`.  This is the critically damped
formulation — kd scales with √kp, ensuring the damping ratio stays constant
across different stiffnesses.

**Real**: `EE_PD_KD = 0.1` is fixed.  With `EE_PD_KP = 2.0`, the ratio is
`kd/kp = 0.05`, which is underdamped compared to the sim's critically damped
formulation.  The real arm will therefore exhibit more oscillation around the
target when using position-tracking behavior.

---

### 5.6 Velocity vs torque bandwidth — **MEDIUM IMPACT**

**Sim** recomputes torques at 500 Hz (every MuJoCo substep) with updated
state.  The arm is continuously corrected within each 50 ms policy interval.

**Real** computes the velocity command once per 50 ms cycle.  Ruckig smooths
this at 1 kHz, but the velocity target itself is constant within the interval.
If the arm deviates mid-step, there is no correction until the next policy
cycle.

---

### 5.7 Orientation error sign convention — **LOW IMPACT**

Both formulations produce the same direction for errors < ~30°.  The sim
matrix formula and the real quaternion formula diverge slightly for large
orientation errors (>60°), but typical manipulation tasks stay well below this.

---

## 6. Recommended Changes to Align Real Stack with Sim

In decreasing order of alignment benefit:

### R1 — Add goal pose accumulation (highest priority)

The biggest structural mismatch is that the real controller treats deltas as
instantaneous velocity, while the sim accumulates deltas into a persistent goal
and drives toward it.  To match sim behavior:

- Maintain a `goal_pos` (3,) and `goal_quat` (4,) on the real controller.
- Each step: `goal_pos += delta_pos; goal_quat = quat_multiply(delta_quat, goal_quat)`.
- Compute the PD velocity command from `goal_pos - ee_pos` and orientation
  error, **not** directly from the raw action delta.
- This is structurally what `_ee_pd()` does (with `use_ee_pos=True`), but
  with a goal that is reset from current pose at the start of each episode
  rather than being seeded from the trajectory.

### R2 — Increase derivative gain / use critically damped formula

Set `EE_PD_KD = 2 * sqrt(EE_PD_KP) * damping_ratio`.  At `EE_PD_KP = 2.0`
and `damping_ratio = 1`, this gives `kd ≈ 2.83` — roughly 28× larger than
the current 0.1.  This will reduce oscillation and match the sim's
critically damped response.

### R3 — Add per-axis gain tuning to match sim kp = 150 effective compliance

The sim's `kp = 150` in torque units is not directly comparable to the real
`EE_PD_KP = 2.0` in velocity units.  To equate them, estimate the robot's
effective Cartesian admittance (m/s per N) from a sysid sweep.  Gain tuning
based on sysid data is the recommended path.

### R4 — Normalize by approximate Jacobian condition (partial inertia matching)

A lighter-weight alternative to full operational space control: multiply the
6D velocity command by a diagonal approximation of the effective Cartesian
inertia (estimated from `J * J^T` near a nominal configuration).  This will
partially normalize the compliance across axes without requiring the full mass
matrix.

### R5 — Implement Jacobian-projected nullspace velocity

Compute `dq_null = (I - J^† J) * kp_null * (q_init - q)` and add this to the
arm velocity command as an additional joint-space offset before sending to
franky.  Note: this requires switching the real arm from Cartesian velocity
to joint velocity commands, or mixing Cartesian and joint-velocity control,
which franky does not currently support in a single motion object.  This is
a larger architectural change.

### R6 — Match orientation error formulation

Replace the real's `axis_angle(delta_quat)` with the rotation-matrix
cross-product formula `0.5 * Σ cross(rc_i, rd_i)` after implementing R1
(goal accumulation).  This is a small change but ensures parity with the
sim's numerical behavior.

---

## 7. Summary Table

| Difference | Sim behavior | Real behavior | Impact | Fix |
|---|---|---|---|---|
| Output type | Torques (7D) | Cartesian velocity (6D) | Architecture | — (accept) |
| Goal accumulation | Persistent goal pose | Instantaneous delta velocity | High | R1 |
| Inertia decoupling | λ = (J M⁻¹ Jᵀ)⁻¹ | None | High | R4 (approx) |
| Nullspace control | `(I - Jbar*J)*τ_null` | None | Medium | R5 |
| Gravity compensation | Explicit via qfrc_bias | Firmware internal | Architecture | — (accept) |
| Derivative gain formula | `kd = 2√kp * ζ` | Fixed `kd = 0.1` | Medium | R2 |
| Orientation error | Matrix cross-product | Quaternion Hamilton product | Low | R6 |
| Control bandwidth | 500 Hz recompute | 20 Hz + 1 kHz Ruckig | Medium | — (hardware limit) |
| Mid-step correction | Yes (25 substeps) | No (1 command per step) | Medium | — (hardware limit) |
| Velocity limits | None (torque-limited) | 0.30 m/s, 1.20 rad/s | Low-Med | Match via sysid |
