# SysID Pipeline (real-robot side)

Reference for working on the sim2real system-identification pipeline from this
repo / the robot workstation. The sim side (fitting, validation, diagnostics)
lives in **multi-fast** — `SYSID_UPDATE.md` there is the design document and
plan of record; this file covers what runs *here* and the contracts between
the two repos.

## Big picture

We fit the sim controller (robosuite OSC_POSE) so that sim-trained policies
transfer to the FR3. The pipeline: collect matched excitation data on both
plants → fit sim `(kp, damping_ratio)` → validate on a held-out metric →
adopt into training configs. Three comparison modes with distinct roles:

| Mode | Mechanic | Role |
|---|---|---|
| Absolute replay | Both plants track the same goal trajectory | Fit objective (well-posed ID input) |
| Teleport (k-step) | Sim re-synced to real's `(q, q̇)` every k steps, replay k deltas | Validation (k=1 primary; error-vs-k curve) |
| Open-loop replay | Same delta sequence from the same init | Monitor / stress test only |

This repo produces the real-side data for all three; `sysid/sysid.py` is the
collection entry point. Analysis happens in multi-fast
(`scripts/sysid/fit_sim_controller.py`), which consumes the run directories
produced here directly.

## Control stack in one paragraph

Policy-side EE pose deltas → `BimanualFranka.send_action` (EE_DELTA mode) →
`OSCVelocityController.compute_qdot`: goal re-anchored on the measured pose
each tick (`goal = measured ∘ delta`), pure-P velocity law
(`v = kp · pose_error`), DLS pseudo-inverse to joint velocities + nullspace
bias → franky `JointVelocityMotion` (100 ms window, Ruckig-ramped, accel
factor 0.25) → firmware joint impedance. Full trace: `CONTROL_STACK.md`.
Numbers that matter here: translation deltas are multiplied by
`_EE_TRANSLATION_FUDGE_FACTOR` (1.2) inside `send_action`; the ×0.9 rotation
fudge acts as an error scale inside `compute_qdot` (not a goal transform);
the `kp` action field maps to velocity-kp `10^kp × OSC_BASE_KP` (5.0); the
`kd` action field is inert (`_KD_GAIN_BASE = 1.0`). These constants are
snapshotted into every `run.json` AND pinned by multi-fast's
`scripts/sysid/test_controller_parity.py` — **retuning them requires updating
that pin and `cfg/sysid/fit_controller.yaml`** (`translation_fudge`), or
subsequent fits silently assume the wrong plant.

## sysid.py modes

### track (primary collection mode)

```
python sysid/sysid.py --mode track --track-spec sysid/specs/sweep_full.json \
    --hold-s 5 [--kp 0.0] [--dry-run]
```

Closed-loop reference tracking: a pose-space reference path (sine/circle
offsets, physical units, anchored at the measured pose after homing to the
spec's `init_qpos`) is pursued by commanding
`dpos = (ref[t] − measured) / fudge` each tick — the divide makes the
post-fudge goal land exactly on the reference. Self-correcting: tracking
error cannot compound. `--track-abort-m` (default 0.15 m) aborts an episode
on tracking runaway. `--hold-s N` prepends a constant-reference HOLD episode
(static-offset calibration tier).

Spec JSON: `init_qpos` (7,), optional `ramp_s` (amplitude ease-in/out,
episodes start and end at rest), and `tracks`: per-episode
`{kind: sine|circle, axes, amp, freq_hz, duration_s}`. Axes 0–2 = position
offsets in **meters**, 3–5 = rotation rotvec offsets in **radians** (base
frame); circles take an `[u, v]` pair. Peak commanded speed = 2π·f·amp —
keep well under the safety clamps (0.30 m/s, 1.2 rad/s). Checked-in specs:
`specs/smoke.json` (gentle 2-episode hardware smoke), `specs/sweep_full.json`
(full grid, ~12 min/run incl. homing; designed ≤ 63%/78% of the clamps).

### replay (legacy / open-loop mode)

```
python sysid/sysid.py <sim_sweep.hdf5> [--kp 0.0]
```

Open-loop replay of a multi-fast `collect_osc_sweeps.py` dataset
(normalized delta actions, scaled here by 0.05 m / 0.5 rad per unit).
Produces the open-loop monitor data. Comparison HTML + `errors.json` are
generated in this mode only.

### --dry-run

Either mode against a kinematic mock (`_MockController`) — no hardware, no
lerobot/franky imports (they're lazy). Exercises spec parsing, homing calls,
the real-time-paced loop, logging, flushes, and all file outputs. Run this
after any change to sysid.py, and once on the workstation venv before a
hardware session.

## Output contract (what multi-fast consumes)

Run directory: `<out_root>/<timestamp>_<tag>/` with `run.json` (mode, args,
constants snapshot, input sha256, episode status) and one HDF5 per episode
(`data/episode_0`), flushed atomically every `--flush-every` steps.

Datasets (per step): `action` (7: dpos_m(3) **pre-fudge** + delta_quat
xyzw(4)), `action_norm` (replay mode only), `eef_goal_pos`/`eef_goal_quat`
(the goal the controller pursued, post-fudge — the absolute-replay reference),
`eef_pos`/`eef_quat` (measured, **base frame**, quats xyzw), `qpos`/`qvel`,
`eef_lin_vel`/`eef_ang_vel` (firmware `O_dP_EE_d`; older data used
`O_dP_EE_c`), `fault_count` (cumulative reflex recoveries — analysis should
drop episodes where it increments), `t_sim` (wall-clock since episode start),
`tau_cmd` (zeros).

Conventions the fit relies on:
- **Pre-action logging**: row t is the state the tick-t action was issued
  from (state read → action sent → row appended). Sim recordings are
  post-action; multi-fast's loader shifts accordingly (`state_convention`).
- **`quat_encoding` attr**: `"exact"` on this branch. Legacy runs (absent
  attr) used the small-angle `[drot/2, 1]` encoding (~2% angle shortfall at
  0.5 rad); the loader recovers the executed rotation either way.
- Episode attrs carry the fudge factors, gains, fps, and mode, so a file
  stays interpretable away from its run.json.

Analysis: rsync the run dir to the dev box and run
`uv run python scripts/sysid/fit_sim_controller.py fit.real_dir=<run_dir>`
in multi-fast — defaults match this format (`metric_quat`, base frame,
recorded goals preferred). One fit per gain condition; don't mix `--kp`
settings in one `real_dir`.

## Session runbook

1. `--dry-run` with `specs/smoke.json` (venv/imports/file-output check).
2. Hardware smoke: `--mode track --track-spec sysid/specs/smoke.json
   --hold-s 5`. Sanity: HOLD episode static offset small and steady;
   sine episodes track with ~cm lag; `fault_count` stays 0.
3. Full collection, one run per gain condition (values mirror the sim-side
   grid in multi-fast `cfg/sysid/collect.yaml`):
   `--track-spec sysid/specs/sweep_full.json --hold-s 5 --kp {-0.5, 0.0, 0.5}`.
4. Optional validation-set collection: replay task-rollout action sequences
   (multi-fast `collect_task_rollouts.py` output) in replay mode.
5. Rsync run dirs to the dev box; fit; read the `validation:` block and
   per-frequency diagnostics in `fit_results.yaml` before adopting params.

## Gotchas / invariants

- **Orientation drifts by design** under zero rotation commands: this stack
  re-anchors its goal on the measured pose every tick, so orientation has no
  absolute anchor (sim's OSC latches its ori goal on bit-exact-zero rot
  deltas instead — a real semantic difference for translation-only motion).
  The fit's `real_ori_drift_max_deg` diagnostic quantifies the wander; if it
  comes out large, the fix under discussion is an opt-in orientation-hold
  mode, not a change to the delta semantics.
- franky's `zero_jacobian` is broken on this build; `_patch_jacobian`
  recomputes it analytically. Don't "simplify" that away.
- Keyboard early-stop (right-arrow / Ctrl-C) needs a tty; under
  nohup/pipes the loop runs headless (flushes still protect the data).
- The reference generator here is a vendored copy of multi-fast
  `utils/sysid/sweeps.py:reference_pose_offsets` — keep in sync on changes
  (exact parity is not load-bearing: the pursued goals are dual-logged, and
  the fit consumes the recording, not a regeneration).
- 20 Hz pacing is sleep-based; `t_sim` records actual wall-clock per tick,
  so rate jitter is measurable after the fact.
