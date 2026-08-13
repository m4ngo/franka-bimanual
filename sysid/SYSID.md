# SysID: matching the real FR3 to the sim controller

Procedure for closing the sim2real gap on this rig. The sim side (fitting,
validation, adoption into training configs) lives in **multi-fast**; this file
is the real-side plan of record and the contract between the two.

## What we are matching

Policies are trained against robosuite's `OSC_POSE` in `impedance_mode=variable`.
`config/control.yaml`'s `torque:` block **is** that controller's config, ported 1:1
(`osc_torque_controller.py`, checked by `scripts/osc_check/check_osc_parity.py`).
So the law is not in question. What differs is the **plant**: the FR3 carries
Coulomb friction and an undeclared payload that mujoco does not, and mujoco carries
a fictitious armature inertia the FR3 does not.

That shapes the whole procedure. The gap is **a deadband, not a gain**, and the two
are not interchangeable: a deadband is an offset in the travel-vs-command line and
no scalar fudge can represent it. Every knob under `tuning:` is a no-op at its
sim-parity value; the job is to find which of them the evidence actually licenses.

## The stack, in one paragraph

EE pose deltas → `BimanualFranka.send_action` (EE_DELTA) → `clip_delta`, then the
`tuning:` fudges → OSC goal pushed over RPyC at 20 Hz → the NUC's
`pylibfranka_control.py` runs `OSCTorqueController.run_controller` at **500 Hz**
(robosuite's substep rate) → torque rate-limit + clamp → libfranka at 1 kHz.
`send_action` is robosuite's `set_goal`; the NUC runs `run_controller`. **The goal
is re-anchored on the measured pose every policy step** (`goal = ee_pos + delta`),
never accumulated — which is why a static tracking error becomes a *velocity* drift
rather than a bounded offset, and why residual-force measurements below read as
rates.

Older revisions of this file described a franky `JointVelocityMotion` stack with an
`OSCVelocityController` and a `_patch_jacobian` workaround. **That stack is gone**
— franky has no torque interface. Anything referring to velocity-kp, `OSC_BASE_KP`
= 5.0, or an inert `kd` channel is stale.

## The ladder

Four measurements, each answering one question, ordered so each removes a confound
from the next. Running them out of order is the main way this goes wrong: a gain
fitted over an uncompensated payload absorbs the payload, and a deadband fitted
over a drifting arm measures the drift.

Everything writes to `~/sysid/outputs/<stamp>_<tag>/` (outside the repo).
`--yes` skips the "workspace clear?" prompt. **Clear the workspace** — all of these
move the arm.

### 0. Make the plant honest — payload, then residual drift

An under-declared payload is a *pose-dependent* bias force. It masquerades as a
gain error, because it grows with reach, and it corrupts every measurement below.

```
python sysid/identify_payload.py --selftest        # conventions check, no hardware
python sysid/identify_payload.py --yes
```

Fits `m` and the flange-frame COM `c` out of libfranka's own `tau_ext_hat_filtered`,
plus a per-joint constant `b` absorbing friction and cable torque. Pose dependence is
what separates the two: the payload term moves with the elbow, `b` does not, and the
q1 sweep is the control (rotating about gravity leaves every gravity torque
invariant, so anything moving with q1 is cable, not payload). Declare the result to
the arm, then confirm it is nulled:

```
python sysid/delta_sweep.py --backend real --sag --yes
```

`SAG RATE` positive = falling = declared mass too low. Drive it to zero.

Then confirm nothing **configuration-dependent** is left — this is the check that
catches residuals the anchor cannot see, because `--sag` resets to `q0`, which is
also the OSC's nullspace reference, where the nullspace regulator is identically
zero:

```
python sysid/delta_sweep.py --backend real --sag --sag-offset-steps 10 --axes 0 --yes
```

`SPEED` should not grow with `dev_rad`. If it does, something in the torque path
depends on being off the reference and is leaking into task space — fix that before
fitting anything, or the fit will absorb it. (One such bug is fixed: the Coulomb
assist was keyed off the total commanded torque, which includes the nullspace
regulator, so being off the anchor bought 0.3–2.2 N of EE force under a strictly
zero delta. See `_friction_feedforward` and
`test_no_op_action_puts_no_force_on_the_ee_away_from_home`.)

### 0.5. The standing torque — `identify_bias.py`

`osc.py` is a pure PD law with no integral term, so it cannot reject a static
disturbance, and EE_DELTA re-anchors `goal_pos` on the measured pose every step —
which turns that standing error into a drift velocity rather than a bounded offset.
The franky velocity stack never showed this because Franka's firmware servo closed an
inner loop with integral action and absorbed the term before it reached the task-space
law. `torque.bias.joint_nm` replaces what that servo was doing.

```
python sysid/identify_bias.py --selftest
python sysid/identify_bias.py --yes [--grid wide]
```

**The sign has to be measured, not derived.** `tau_ext_hat_filtered` looks like it
should give it directly, but unmodelled joint friction lands in that estimate too, so
the residual at rest is not cleanly the disturbance. Deriving it got joint 4 backwards
once — `+0.43 Nm` against a `[-3.07, -0.07]` range drove the elbow toward straight and
the arm extended outward and up under teleop.

Settle it with three sag runs, redeploying between each (the field is NUC-side):

```
# zeros -> the baseline; then the printed vector; then the same vector negated
python sysid/delta_sweep.py --backend real --sag --yes
python sysid/delta_sweep.py --backend real --sag --sag-offset-steps 10 --axes 0 --yes
```

Lowest `SPEED` wins, and the offset run is the discriminating one: the bias is a
constant, so a correct one lowers drift everywhere while a sign error shows up worst
where the arm has the most authority to run away.

### 1. Per-joint plant — `joint_id.py`

One joint at a time under **open-loop torque**, every other joint impedance-held
(`MODE_TORQUE`). tau is the independent variable, so nothing here depends on a
servo gain and the identical protocol runs against mujoco and against the arm.

**Anchor the real run first**; sim is free to be moved to match, real is not:

```
python sysid/joint_id.py --backend real --yes            # prints the q it used
PYTHONPATH=franka_config multi-fast/.venv/bin/python \
    sysid/joint_id.py --backend sim --q <those 7 numbers>
python sysid/joint_id.py --compare <sim_dir> <real_dir>
```

Read the **`dead r/s`** column, not `tc r/s`. `dead = tau_c/M` is the commanded
acceleration lost to friction; it is the ratio that transfers, because a joint can
carry more friction than sim and still track it if it carries proportionally more
inertia. Expect `M real/sim < 1` (sim's armature inflates inertia); that is a known
property of the sim, not a fit error.

The sim backend is also the estimator's self-test: mujoco's M and frictionloss are
known exactly, so a sim run that fails to recover them means the fit is wrong and
the real numbers are not worth reading.

### 2. Task level — `delta_sweep.py`

`joint_id` measures the plant; this measures the thing the policy actually drives.
Per axis it holds a constant normalized EE_DELTA of magnitude `a` for N steps and
records travel, sweeping `a` over the range real task actions occupy.

```
python sysid/delta_sweep.py --backend real --yes
PYTHONPATH=franka_config multi-fast/.venv/bin/python sysid/delta_sweep.py --backend sim
python sysid/delta_sweep.py --compare <sim_dir> <real_dir>
```

Both backends run their own shipped config — that is the question being asked, so
neither side is normalised to the other. The **shape** of travel against amplitude
is the point, and the compare prints the verdict per axis:

| Reading | Meaning | Knob |
|---|---|---|
| ratio flat, ≈1.0 | matched | none |
| ratio flat, ≠1.0 | pure gain error | `kp_pos_scale` / `kp_ori_scale` |
| ratio falls as a→0 | deadband — friction | `friction_kc` (≤ 1.0) |

The second table fits `travel = k*(commanded - d)` and reports `d` and `k`
separately. **`d` is what a fudge cannot represent**: it is an offset, not a gain.
`k_real/k_sim` is the only part a scalar can fix. This is the measurement that tells
you whether `ee_translation_fudge` can ever be the right answer — usually it is not,
and a fudge tuned on one trajectory is only correct at whichever amplitude happened
to dominate it.

### 3. Trajectory level — `tune.py`

Scores the arm against a sim reference trajectory and sweeps knobs:

```
python sysid/tune.py sysid/data.hdf5 --yes                        # score as configured
python sysid/tune.py sysid/data.hdf5 --sweep friction_kc=0.5,0.7,1.0 --yes
python sysid/tune.py sysid/data.hdf5 --score-only ~/sysid/outputs/<run>/
```

Objective is the **per-step task response ratio**, real/sim — not accumulated
endpoint error, which measures horizon rather than plant (once the plants diverge
the same deltas resolve to different goals, and on a position circle it exceeds the
frozen-arm floor).

`TASK` and `NULLSPACE` are scored separately because the OSC controls 6 DOF at `kp`
and the 7th at `nullspace_kp`, 15× weaker — different plants. **A nullspace ratio
near 1 with a low task ratio rules out a global torque deficit**, and makes per-joint
ratios misleading: a nullspace-heavy joint reads high while every task direction
reads low. That combination points at task-space authority (`uncouple_pos_ori`,
`lambda`), not at friction.

### Also available

- `scripts/osc_check/check_osc_axes.py` — commanded-vs-measured per OSC axis, the
  ground truth for "are the axes right". Results are **only comparable at the same
  pose** (`lambda_pos` rotates with the arm); use `--poses sim` and `--repeat`.
- `scripts/measure_joint_friction.py --method torque --directional` — per-joint
  breakaway feeding `torque.friction.coulomb_nm`.
- `sysid/sysid.py` — the older bulk collection entry point (`--mode track` against
  `specs/*.json`, or open-loop replay of a multi-fast sweep file) for the multi-fast
  fitting pipeline. `--dry-run` exercises it with no hardware.

## Where the knobs live

`config/control.yaml`'s **`tuning:` block is the only place to tune.** Everything
under `torque:` is the control law, defined by the sim — changing it to fix the rig
makes the arm stop being the thing the policy was trained on.

- `ee_translation_fudge`, `ee_rotation_fudge` — delta scale. 1.0 = sim.
- `friction_kc` — Coulomb assist scale. **Invariant: `friction_kc * kc_joint ≤ 1.0`.**
  Above 1.0 the assist drives the joint instead of freeing it — negative damping,
  which has faulted joint 7. Past 1.0 is an authority deficit; use `kp_*_scale`.
- `friction_kc_joint_pos` / `_neg` — directional split, selected by the sign of the
  **commanded torque**, not of `dq` (signing by `dq` cancels the friction holding a
  still arm and the arm walks). Currently all ones: the directional hypothesis was
  tested and rejected — the joint-1 asymmetry is cable torque, not friction.
- `kp_pos_scale` / `kp_ori_scale` — stiffness only; `kd` is re-derived to hold the
  ratio, so this buys friction rejection, not settling speed.
- `kd_pos_scale` / `kd_ori_scale` — the damping ratio itself. Above 1.0 costs sim match.

## Gotchas

- **A gain change reaches the arm only after `scripts/deploy_nuc_server.sh <mario|luigi>`.**
  The workstation copy is not what runs the loop. `coulomb_nm` lives on the NUC;
  `friction_kc*` do not, so changing one without redeploying multiplies them out of step.
- **Both plants must sit at the same q** or the comparison means nothing. Anchor real,
  match sim with `--q`.
- **One condition per run directory.** Don't mix gain settings in one `real_dir`.
- `--arm` is a **key prefix in a `config/rig.yaml` profile, not a side**:
  `single_arm_franka` maps `r` to the physical *left* arm. Every script prints which
  it resolved to before connecting.
- A run with `fault_count`/recoveries incrementing is not a measurement; the scripts
  warn, and those rows should be dropped.
- Anything that rescales tau outside `_enforce_limits` is a regression, and
  `test_torque_limits_are_the_only_thing_that_rescales_tau` is structural for that
  reason — a reintroduced guard passes every numerical test, because it only fires
  outside their range.
