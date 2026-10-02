# Entry points: recording, replaying and comparing real and sim

Every script that moves the real arm, runs the sim, or compares the two, grouped
by what you want to do. Workstation venv (`~/franka_ws/.venv`) unless the row says
`multi-fast/.venv`. File formats are in `EPISODE_HDF5.md`.

## replay vs rerun, in one paragraph

`replay_goals_in_sim.py` **replays goals**: it takes the OSC goal the real arm was
actually sent at every step and sends the same goal to the sim arm, open loop. The
policy is not involved; the sim arm is told exactly where real's arm was told to
go, so any difference in where it ends up is the plant (and the controller
constants). Use it for sysid questions.

`rerun_policy_in_sim.py` **reruns the policy**: it puts the sim arm at the real
episode's start joints on the real episode's curve, loads the same checkpoint,
and lets the policy act on what the sim arm sees, closed loop. The goals it sends
are its own. Use it for policy questions: would this residual have done the same
thing in sim, and where does its behaviour diverge once the plant differs.

Both write `sim_replay.hdf5` beside the real `episodes.hdf5` and then render
the comparison (`compare_*.html`, `errors.json`) through `scripts/plot_episodes.py`
in the workstation venv; `--no-viz` skips that, and `plot_episodes.py <run>`
re-renders on its own.

## The three data formats

| format | what it is | who makes it | who reads it |
|---|---|---|---|
| LeRobot dataset (`~/franka_data/<repo>/`) | training data: images, state, action rows `[x y z qx qy qz qw gripper kp kd]` | `lerobot-record` wrappers, `run_residual.py` (best.pt), `rollout_policy.sh`, the baseline rollouts | `lerobot-train`, `scripts/train_pipeline.py` (conversion + all three trainings from one yaml), `replay.sh`, `replay_dataset.py`, `real_trajectory_rollout.py`, `lerobot_to_hdf5.py` |
| episode HDF5 (`episodes.hdf5`, `sim_replay.hdf5`, `excitation.hdf5`) | one trajectory per group: goals, joints, EE pose, gains, curve | the rollouts, excitation and conversions below | the sim replays, the plant fit, the figures |
| sim sweep HDF5 (`~/sysid/<condition>/data.hdf5`) | sim-generated reference sweeps (sine/circle deltas) | `multi-fast/scripts/sysid/collect_osc_sweeps.py` | `sysid/sysid.py --mode replay`, `sysid/tune.py` |

## Recording on the real arm

| script | drives the arm with | writes | notes |
|---|---|---|---|
| `scripts/teleop.sh`, `gello_ee_teleop.sh`, `spacemouse_teleop.sh`, `single_arm_teleop.sh <mode>` | a human leader (GELLO joint / GELLO EE / SpaceMouse) | nothing | teleop only |
| `scripts/record_data.sh`, `ee_record_data.sh` | GELLO teleop, bimanual | LeRobot dataset | joint-mode / EE-mode |
| `scripts/lerobot_record_homed_single_arm.py` | one leader on one arm (`--teleop-mode` from `teleop_single_arm.py`'s table), homing between episodes | LeRobot dataset | the single-arm recorder |
| `scripts/rollout_policy.sh` | a trained LeRobot policy | LeRobot dataset | bimanual |
| `scripts/sail_rollout.sh`, `bspline_rollout.sh` | a baseline policy in its own venv, via a policy server | LeRobot dataset under `~/franka_data/outputs/<train-dataset>/`, the end-effector force in `force_profiles.npz`, every plan in `chunks.npz`, and an `episode_<NNN>.html` per episode | `check_policy_server.py` is the preflight; `python -m baselines.rollout_viz <dir>` re-renders the pages for every real run under `<dir>` |
| `residual_wrapper/run_residual.py --base-policy <ckpt> --residual-policy best.pt` | LeRobot base policy (EE_POS targets; a delta-trained checkpoint is refused) + point-cloud residual, executed as the reach path executes its base: targets composed relative to the chunk anchor, each dispatched as the one-step delta from the measured pose | LeRobot dataset + viz HTML + `force_profiles.npz` | `run_residual_openpi.py` is the same loop with an OpenPI base |
| `residual_wrapper/run_residual.py --residual-policy <FAST .zip> --seed S` | analytic reach base + FAST residual (`--policy base\|residual\|both`) | `~/franka_data/reach_residual/<ts>/episodes.hdf5` + viz HTML | the run `rerun_policy_in_sim.py` twins |
| `scripts/real_reach_rollout.py` | analytic reach base policy only, seeded curve | `~/franka_data/real_reach/<ts>/episodes.hdf5` + HTML | the run `replay_goals_in_sim.py` replays |
| `sysid/excite_panda.py [--kp/--kd] [--gain-amp]` | panda_control excitation trajectories, fixed or oscillating gains | `~/sysid/outputs/<ts>_<tag>/{ee_pose,ee_delta}/excitation.hdf5` | the sysid recording; `--dry-run` off-hardware |
| `sysid/sysid.py --mode track\|replay` | sine/circle/hold specs, or a sim sweep file | `~/sysid/outputs/<ts>_<tag>/` | the older bulk collector |
| `sysid/delta_sweep.py --backend real`, `joint_id.py --backend real`, `identify_payload.py`, `identify_bias.py`, `measure_joint_friction.py`, `measure_vibration.py` | fixed probes | `~/sysid/outputs/` | the sysid ladder (SYSID.md); each has a `--backend sim` or sim twin except the last three |
| `scripts/osc_check/check_osc_axes.py` | one OSC axis at a time | stdout | commanded vs measured per axis |

## Replaying on the real arm

| script | replays | as | notes |
|---|---|---|---|
| `scripts/replay.sh <repo> <episode>` | one LeRobot episode | joint positions | `lerobot-replay` |
| `scripts/replay_dataset.py --mode delta\|ee_pose` | a whole LeRobot dataset | EE_DELTA as recorded, or the absolute goals those deltas produced (EE_POS) | re-measures old recordings under the current controller |
| `scripts/real_trajectory_rollout.py --source arm [--gain-amp]` | one LeRobot episode's absolute goals (`--episode N`), or every episode's (`--all`); EE_DELTA or EE_POS source | EE_POS | writes `episodes.hdf5`, one group per episode; `--source dataset` writes the recording itself instead (no arm). `sysid/merge_episodes.py` joins it with an excitation run for the fit |
| `sysid/tune.py <sim ref> [--sweep knob=...]` | a sim sweep trajectory's deltas | EE_DELTA | scores real against sim per step while sweeping `tuning:` knobs |
| `sysid/sysid.py --mode replay <sim.hdf5>` | a sim sweep file's deltas | EE_DELTA | older |

## Running a real trajectory in sim (`multi-fast/.venv`)

| script | input | what sim does | output |
|---|---|---|---|
| `scripts/reach/replay_goals_in_sim.py <run> --episode N\|<name> [--plant]` | `episodes.hdf5` (reach, trajectory, or an excitation file) | re-issues the real goals open loop from the real start | `sim_replay.hdf5` (`mode: replay_absolute_goal`) |
| `scripts/reach/rerun_policy_in_sim.py <run> [--episode] [--plant]` | a `run_residual.py` FAST run | same checkpoint, closed loop, on the real start and curve | `sim_replay.hdf5` (`mode: policy_rerun`) |
| `scripts/sysid/rollout_fit.py <fit dir> --real-dir <dir>` | any directory of episode files | the fitted plant tracks the real goals open loop, the fit's own env | overlays + `rollout_summary.yaml` with error by gain tercile |
| `scripts/sysid/fit_sim_controller.py` | a directory of episode files (`fit.real_dir`) | fits the plant (armature, friction) to them | `logs/sysid_fit/<ts>/` |
| `sysid/delta_sweep.py --backend sim`, `joint_id.py --backend sim --q ...` | the real run's anchor | the same probe in mujoco | `~/sysid/outputs/` for `--compare` |
| `scripts/sysid/collect_osc_sweeps.py` | a config | open-loop delta sweeps in sim, optional gain grid | sim sweep HDF5 |
| `eval_fast.py env_name=reach` | a FAST checkpoint | the sim's own evaluation on sim-sampled curves | wandb / videos |

## Rolling a baseline out in LIBERO (`multi-fast/.venv` + the policy's venv)

The sim half of the SAIL / B-Spline comparison, described in full in
[baselines/LIBERO_SIM.md](baselines/LIBERO_SIM.md). The env is **stock**
robosuite -- no plant, controller or gripper overrides -- because that is the
model `regenerate_libero_dataset.py` recorded the demonstrations under.

| script | does | output |
|---|---|---|
| `baselines/libero_bridge/dataset.py <demo.hdf5\|suite dir> --out-dir <dir>` | a regenerated LIBERO file to `task_<i>/{sail,bspline}.hdf5`, `<i>` its index in the suite | training data, one pair per task |
| `baselines/libero_bridge/train.py <prep dir> --tasks-file ... --steps N` | the unchanged trainers, one policy per task per backend, one after another; `--resume` continues an unfinished one from its newest checkpoint | `~/franka_data/policies/<suite>/task_<i>/<backend>/` + `.trained.json` |
| `baselines/tillicum/submit_libero_train.sh <suite> --tasks "3 4" --steps N --apply` | on Tillicum: the same trainings as a Slurm array, one GPU each, all at once, in the image `baselines/tillicum/Dockerfile` builds | the same layout under `/gpfs/scrubbed/$USER/franka_home/franka_data/`, synced back by rsync |
| `scripts/libero_rollout.sh --backend sail\|bspline --start-server --ckpt <C> --task <i>` | one task: policy server in its own venv, LIBERO env in multi-fast's, absolute pose targets inverted to OSC deltas. SAIL runs its own OSC gains (`baselines.sail.sim_osc`); `--osc-kp` and `--osc-damping-ratio` override them | a run dir under `~/franka_data/outputs/<suite>/task_<i>/`, with each episode's wrist force in `force_profiles.npz` |
| `scripts/libero_rollout.sh --backend pi05 --task <i>` | multi-fast's base policy (pi0.5) on the same env, init states and clock; loads in-process, no server | a run dir filed as `pi05` |
| `baselines/libero_bridge/evaluate.py <prep dir> --tasks-file ...` | that rollout for every task and all three backends; tags them all with one sweep id and prints the pooled table at the end. `--save-video` for an mp4 per episode (~145 KB), `--extra --speed-up-times N` to sweep B-Spline's speed | the same run dirs, each carrying `run.sweep` |
| `scripts/rollout_summary.py --sweep <id>\|latest` | one sweep's rollouts wherever they landed, plus a pooled row per backend | the comparison table |
| `scripts/rollout_report.py --sweep <id> [<id> ...]` | a 16:9 PNG summary sheet per sweep, method and suite (stat tiles, outcome per task, every episode, episode length, run parameters including the OSC gains and step limit, and for runs that recorded it the wrist force per task, peak force and every force profile overlaid) plus a comparison sheet per suite | `<root>/reports/*.png` |
| `scripts/collect_rollout_report.sh <out-dir> <id> [<id> ...]` | those sheets, each sweep's `rollout_summary.py` table, and every rollout video, in one handover folder, videos split by outcome | `<out-dir>/*.png`, `<sweep>.summary.txt`, `videos/<sweep>/<task>/{success,timeout}/`, `index.json` |
| `scripts/check_libero_sim_rollout.py --task <T> [--targets reached] [--osc-kp ...]` | replays a demo's own recorded targets through the executor; `--targets reached` replays SAIL's targets on SAIL's schedule instead, to measure a controller | pass/fail per demo, the delta-vs-recorded error, and the share of control steps at a torque limit |

Episode time is reported in **simulated** seconds, in `wall_time_s`, so
`rollout_summary.py` compares methods on a clock that does not depend on the
machine; `clock_time_s` carries the real elapsed time beside it.

`replay_goals_in_sim.py` and `rollout_fit.py` both track real goals open loop. The
difference is the sim they build: `rollout_fit.py` rebuilds the fit's env exactly
(the plant it found, its Stribeck friction, its law armature) and is the sysid
answer; `replay_goals_in_sim.py` uses the reach env with an exported `cfg/plant`
yaml plus the real torque rate limit and reports success and cursor like the
task does.

## Comparing real and sim

| script | compares | output |
|---|---|---|
| `scripts/plot_episodes.py <run>` | `episodes.hdf5` against `sim_replay.hdf5`, episode by episode | `compare_<ep>.html` (3D + panels, gain row), `errors.json` (position, rotation, cursor lag, error by gain tercile) |
| `multi-fast/scripts/sysid/rollout_fit.py --real-dir` | the fitted plant against the recording | per-trajectory HTML, `rollout_summary.yaml` |
| `sysid/delta_sweep.py --compare`, `joint_id.py --compare` | the sim and real probe runs | tables: travel vs amplitude, per-joint inertia and friction |
| `sysid/tune.py` | live, while sweeping knobs | per-step response ratios |
| `multi-fast/scripts/utils/compare_real_sim.py` | dataset distributions, not trajectories | workspace and action histograms |

## Converting and checking files

| script | does |
|---|---|
| `sysid/lerobot_to_hdf5.py <dataset>` | LeRobot EE_POS recording to an episode file for the fit; per-step gains carried when they moved |
| `scripts/prepare_baseline_datasets.py` | one EE_POS recording to the sysid, SAIL and B-Spline files |
| `scripts/results_json_to_hdf5.py <run dirs>` | legacy `results.json` runs to `episodes.hdf5` / `sim_replay.hdf5` |
| `scripts/filter_noop_actions.py --source-repo-id <dataset> --target-repo-id <id>` | a recording minus its no-op frames, as a new dataset; `ee_pose` drops frames whose target did not move, `delta` frames whose delta is under threshold; `--dry-run` first |
| `sysid/merge_episodes.py <out> <files...>` | join episode files into one fit dataset, with `--prefix`, `--rename`, `--validate` |
| `multi-fast/utils/sysid/episode_hdf5.py <file>` | describe and validate any episode file |
| `sysid/excite_panda.py --verify <run>` | an excitation run's two files agree and its gains resolve |

## Off-hardware checks

| script | exercises |
|---|---|
| `scripts/check_real_reach_offline.py -n N --out <dir>` | the reach rollout against a fake arm; writes an `episodes.hdf5` the sim replay accepts |
| `scripts/check_reach_compare_offline.py <run>` | the comparison and figure on a synthesised sim episode |
| `scripts/check_baseline_rollout_offline.py` | both baseline loops against fake servers |
| `scripts/check_libero_sim_rollout.py --task <T>` | the sim rollout's absolute-pose executor, by replaying recorded demos through it (needs mujoco, no policy) |
| `scripts/check_doc_refs.py` | every `file.py#L<n>` link in the docs still lands on the symbol it names |
| `sysid/excite_panda.py --selftest`, `--dry-run` | the excitation generators and the whole write path |
| `multi-fast/scripts/sysid/test_plant_fit.py <combined ee_pose dir>` | the fit, the loader, variable-impedance replay |
| `multi-fast/scripts/sysid/check_fastpath.py` | the fit's step fast path is bit-identical |
| `scripts/osc_check/check_osc_parity.py`, `check_osc_e2e.py` | the ported controller against robosuite's |
| `tests/test_osc_stack.py`, `test_spacemouse_action.py`, `test_grippers.py` | the control stack against robosuite |

## Three usual chains

Sysid (does the plant match, with or without gain actions):
```
python sysid/excite_panda.py --gain-amp 0.3 --tag panda_excite_gain --yes
cd multi-fast && .venv/bin/python scripts/sysid/rollout_fit.py logs/sysid_fit/<fit> \
    --real-dir ~/sysid/outputs/<run>/ee_pose --out ~/sysid/outputs/<run>/sim
```

Reach task, goal replay (is the arm's response the sim's):
```
python scripts/real_reach_rollout.py --episodes 5
cd multi-fast && .venv/bin/python scripts/reach/replay_goals_in_sim.py ~/franka_data/real_reach/<ts> --episode 0 --plant sysid_2026_09_12
# compare_ep000.html + errors.json are rendered at the end; plot_episodes.py <run> re-renders
```

Reach task, policy twin (would the residual have done the same in sim):
```
python residual_wrapper/run_residual.py --residual-policy ft_policy_400000_steps.zip --seed 42 --policy both --num-episodes 3
cd multi-fast && .venv/bin/python scripts/reach/rerun_policy_in_sim.py ~/franka_data/reach_residual/<ts>
# compare_*.html + errors.json are rendered at the end
```
