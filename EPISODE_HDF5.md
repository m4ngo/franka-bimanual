# Episode HDF5 files

Every robot trajectory in this workspace is stored the same way: one HDF5 file
holding one or more episodes. Real recordings and sim replays use the same
layout, so one reader serves the plant fit, the sim replay, and the figures.

This replaces the older `results.json` / `sim_episode_NNN.json` run files.
LeRobot datasets are separate: they are the training format and stay as they
are. `sysid/lerobot_to_hdf5.py` converts one into this layout when the sim
needs it.

The code is `multi-fast/utils/sysid/episode_hdf5.py` (numpy + h5py only).
Use it from both venvs; franka_ws scripts add `multi-fast` to `sys.path`.

## Layout

```
file.hdf5
  attrs
    schema      "franka_episodes/1"
    created     when it was written (UTC)
    producer    which script wrote it
    meta        (optional) run metadata as a JSON string: git SHAs, controller
                config, tuning trims -- what the arm was running
  data/
    <episode>/            e.g. ep000, v4_chirp, traj_sine_ax3_a0.5_f0.25_rep0
      attrs               see below
      action              (T, 7)
      qpos, qvel          (T, 7)  joint angles and velocities
      eef_pos             (T, 3)  end-effector position
      eef_quat            (T, 4)  end-effector orientation, xyzw
      eef_goal_pos        (T, 3)  the goal the controller was sent
      eef_goal_quat       (T, 4)
      ...optional datasets, listed below
      curve/              (reach episodes only)
        waypoints         (N, 3)
        velocity_scales   (N,)
        attr goal         (3,)
```

The goal at row `t` is what step `t` sent. Which state sits on the same row is
said by the `obs_timing` attr:

- absent or `pre_action`: row `t` is the state **before** action `t`
  (excitation runs, LeRobot conversions, `--source dataset` trajectories). Row
  0 is the start pose. A sim replaying the file applies goal `t` and is
  compared against row `t+1`; this is the plant fit's convention.
- `post_period`: row `t` is the state **after** action `t` acted for one
  control period (reach and trajectory rollouts on the arm). The start pose is
  in `init_qpos` / `ee_pos0` / `ee_quat0` (and `qvel0` when measured), and sim
  row `t` compares with row `t`.

The readers handle both: the figure shifts `pre_action` rows so the two sides
line up, and the plant fit re-rows a `post_period` episode as `pre_action` on
load (`episode_hdf5.as_pre_action`: the start becomes row 0, the measured rows
move down one, the last measured row drops off). A merged fit file can
therefore mix the two.

All floats are stored as float32. Quaternions are always `xyzw`.

### Required attrs

| attr | meaning |
|---|---|
| `num_samples` | T, the number of rows |
| `fps` | control rate the rows were recorded at (20) |
| `frame` | `base` (the arm's base frame) or `base_sim` (the sim's base frame) |
| `quat_order` | `xyzw` |
| `ee_convention` | `O_T_EE` (the FR3 flange frame) or `robosuite_grip_site` (the sim's grip site) |
| `action_format` | `absolute_pose_quat`: `action = [goal_pos(3), goal_quat(4)]`. `metric_quat`: `action = [dpos(3), dquat(4)]`, an EE_DELTA in metres |
| `action_space` | `EE_POS` or `EE_DELTA`, the space `action` is written in |
| `init_qpos` | (7,) joint angles the episode starts from |

Files written before the frame attrs existed are all `base` / `xyzw` /
`O_T_EE`; readers fill those in and the validator reports them as legacy.

### Optional datasets

| dataset | shape | meaning |
|---|---|---|
| `policy_action` | (T, K) | the policy's own action before it became a goal; `policy_action_format` says what K is (`osc_delta_norm7`, `lerobot_row10`) |
| `gain_action` | (T, 2) | normalised kp/kd action `[a_kp, a_kd]` sent each step |
| `kp`, `kd` | (T, 6) | the physical gains the arm resolved from it |
| `tau_cmd`, `tau_measured`, `tau_ext` | (T, 7) | joint torques |
| `ee_force`, `ee_torque` | (T, 3) | external wrench at the EE, base frame (N, Nm): libfranka's `O_F_ext_hat_K` on the arm. The episode's \|F\| statistics are attrs `ee_force_{mean,median,p95,max}_n` |
| `eef_lin_vel`, `eef_ang_vel` | (T, 3) | measured EE velocity |
| `eef_goal_lin_vel` | (T, 3) | the reference's velocity (excitation runs) |
| `t_sim` | (T,) | wall-clock time of each row |
| `fault_count` | (T,) | recoverable-fault counter |
| `cursor` | (T,) | reach task: which waypoint the arm was tracking |
| `anchor_gap_m` | (T,) | how far the goal's anchor was from the pose the policy saw |

### Common optional attrs

`episode`, `arm`, `seed`, `source` (`arm`, `dataset`, `sim`), `obs_timing`,
`control_mode`, `ee_pos0`, `ee_quat0`, `qvel0`, `success`, `cursor`,
`curve_len`, `dry_run`, `qvel_source` (`measured` or `central_difference`).

Gain records add `gain_varies`, `gain_schedule`, and the rig's remap constants
(`osc_base_kp`, `osc_default_damping_ratio`, `gain_exp_base`, `kp_limits`,
`damping_ratio_limits`, `tuning_gain_scales`) so the sim can check its own
action-to-gain map before trusting the file.

Sim replays add `site_in_otee_rotvec` / `site_in_otee_pos` (where the sim's
grip site sits in the FR3 flange frame, measured at the start pose),
`goal_transport_max_m`, `start_qpos_max_err_rad`, `impedance_mode`,
`controller`, `plant`.

## Who writes them

| file | writer | contents |
|---|---|---|
| `<run>/episodes.hdf5` | `scripts/real_reach_rollout.py`, `scripts/real_trajectory_rollout.py` (one episode, or `--all` of a dataset), `scripts/check_real_reach_offline.py` (fake arm) | real episodes, base frame, with run metadata in `meta` |
| `<run>/sim_replay.hdf5` | `multi-fast/scripts/reach/replay_goals_in_sim.py` | the sim's replay of those episodes (the real goals re-issued open loop), one group per replayed episode, added as you go |
| `<run>/sim_replay.hdf5` | `multi-fast/scripts/reach/rerun_policy_in_sim.py` | the same policy re-run closed loop in sim on the real episode's start and curve (`mode: policy_rerun`), for `run_residual.py` reach runs |
| `<run>/ee_pose/excitation.hdf5`, `<run>/ee_delta/excitation.hdf5` | `sysid/excite_panda.py` | one excitation run in both action spaces |
| `<out>/ee_pose/<name>.hdf5` | `sysid/lerobot_to_hdf5.py` | a LeRobot recording converted for the fit |
| `<run>/episodes.hdf5`, `<run>/sim_replay.hdf5` | `scripts/results_json_to_hdf5.py` | a legacy `results.json` run translated |
| any | `sysid/merge_episodes.py` | several episode files joined into one, with renames and the `_train` / `_validate` split |

Writers go through `episode_hdf5.write_episodes` (whole file, written to a
temp file and renamed, so a crash never leaves a torn file) or
`episode_hdf5.upsert_episodes` (add or replace episodes in an existing file).

## Who reads them

- `multi-fast/scripts/sysid/fit_sim_controller.py` and `rollout_fit.py`: every
  `*.hdf5` in a directory, `absolute_pose_quat` only.
- `multi-fast/scripts/reach/replay_goals_in_sim.py`: one episode of `episodes.hdf5`.
- `scripts/plot_episodes.py`: `episodes.hdf5` and, when present,
  `sim_replay.hdf5`, paired by episode name.

Readers use `episode_hdf5.read_episodes(path)`, which returns
`(name, arrays, attrs, curve)` per episode, and `with_defaults(attrs)` for the
legacy frame defaults.

## Checking a file

```
multi-fast/.venv/bin/python multi-fast/utils/sysid/episode_hdf5.py <file.hdf5>
multi-fast/.venv/bin/python multi-fast/utils/sysid/episode_hdf5.py --strict <file.hdf5>
```

Prints one line per episode (steps, rate, frame, what extras it carries) and
every problem found: missing datasets or attrs, wrong shapes, values outside
the allowed set, non-unit quaternions, `action` not matching `eef_goal_*`,
non-finite values, `kp` without `kd`, gain actions outside [-1, 1], a curve
whose arrays disagree. `--strict` refuses the legacy defaults. The same checks
run in code as `episode_hdf5.validate(path)` and `validate_episode(...)`; the
fit and the sim replay run them on every file they load.

Two deeper checks exist for the files they apply to:

- `python sysid/excite_panda.py --verify <run>`: both excitation files describe
  the same goals, each file's `action` reproduces the logged goal through the
  robot's own goal builder, and `kp`/`kd` are what `resolve_gains` makes of
  `gain_action`.
- `python scripts/check_reach_compare_offline.py <run>`: the real-vs-sim
  comparison round-trips exactly on a synthesised sim episode.

## Translating old runs

```
python scripts/results_json_to_hdf5.py ~/franka_data/real_reach/<ts> [...]
python scripts/results_json_to_hdf5.py ~/franka_data/real_traj/*/* --remove-json
```

Reads `results.json`, `meta.json` and any `sim_episode_NNN.json` in each run
directory and writes `episodes.hdf5` and `sim_replay.hdf5` beside them. The
JSON files stay unless `--remove-json` is given, which only removes them after
the new files validate. Records from before the `replay` block (no dispatched
goals) cannot be replayed and are skipped with a message.

`results.json` stored the measured trace in world frame; this script is the
one place `robot_base_in_world` is inverted, to put it back in base frame.
Nothing that reads episode files does this.
