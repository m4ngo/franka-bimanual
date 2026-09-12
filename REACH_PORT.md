# Reach port — session handoff

Working notes for porting multi-fast's sim reach task to the real right FR3,
plus the uncommitted `multi-fast` merge it sits on top of. Written to hand the
work to a fresh session. Delete once the port lands.

Last session: 2026-09-05. Nothing has touched hardware yet.

---

## 1. Repo state — read before running any git command

**`multi-fast` has an open, uncommitted merge.** `MERGE_HEAD` is set. 83 files
are staged, all four conflicts resolved. `git merge --abort` returns to
`21945f2`.

| | |
|---|---|
| merging | `origin/master` `ec02a00` into `sysid-testing` `21945f2` |
| merge base | `f204e24` (2026-08-02) |
| divergence | 29 commits from master, 9 from sysid-testing |

**`franka_ws` commit `a5d0d90` is mislabelled.** Its message reads
"multi-fast: merge master into sysid-testing", but it records the multi-fast
pointer at `21945f2` — the *pre-merge* commit — because the submodule merge was
never committed. It also swept in `.gitmodules` and the `baselines/bspline_policy`
/ `baselines/sail` submodules that were staged before the session. Once the
multi-fast merge is committed, `franka_ws` needs a *second* pointer commit; the
message on `a5d0d90` will still be wrong about what it contained.

Commit order that works (the staged `reach.py` is the merge result with samplers
still inline, so the merge commits cleanly on its own before the split):

```
cd ~/franka_ws/multi-fast
git commit                      # the merge alone, 83 files
git add utils/envs/reach_sampling.py utils/envs/reach.py
git commit -m "reach: split pure-numpy sampling out for the real-robot port"
cd ~/franka_ws
git add config/reach.yaml franka_config/ lerobot_robot_bimanual_franka/ scripts/
git commit -m "real reach: workspace geometry, RealReach env, offline harness"
git add multi-fast && git commit -m "multi-fast: record the merged sysid-testing"
git submodule update --init --recursive openpi stable-baselines3
```

That last line matters: the merge moved `openpi` `27031d65` -> `c3341fcc` and
`stable-baselines3` `4c9cb8a5` -> `c8a1c35f`, but the working tree still has the
old checkouts. `robosuite` is unchanged.

Uncommitted for the reach port (`franka_ws`, all new except `_loader.py`):

```
config/reach.yaml
franka_config/franka_config/_loader.py          (+"reach" in _SECTIONS)
lerobot_robot_bimanual_franka/.../real_reach.py
lerobot_robot_bimanual_franka/.../real_reach_geometry.py
scripts/check_reach_workspace.py
scripts/check_real_reach_offline.py
scripts/real_reach_rollout.py
```

Unrelated pre-existing dirt, left alone: `config/rig.yaml`, `ee_goals.py`,
`teleop_single_arm.py`, `sysid/lerobot_to_hdf5.py`, `baselines/*`, and in
multi-fast `cfg/sysid/fit_controller.yaml` (population 1024 -> 512) and
`nohup.out`.

---

## 2. The merge

All four conflicts were sysid. The framing that resolved them: **master built
the runtime that applies an absolute plant fit; sysid-testing built the fitter
that produces one.** Same parameter names, same Stribeck blend, same
`dof_solref: [0.004, 1.0]`. They are two halves of one feature, so the
resolution is a union — dropping either half leaves the fitter emitting
parameters nothing can apply.

| file | resolution |
|---|---|
| `scripts/sysid/collect_task_rollouts.py` | took ours; master's two hunks are a subset |
| `cfg/plant/default.yaml` | union, 18 keys |
| `cfg/fast_default.yaml` | union of 3 hunks: kept `gain_exp_base` + `law_armature`, added master's 8 absolute-schema interpolations |
| `utils/envs/plant_overrides.py` | took master's restructure, ported `law_armature` back |

Two traps that git reports as clean:

- **"Take ours" on `fast_default.yaml` loads without error and is wrong.** It
  silently drops master's absolute schema, so `plant=sysid_2026_08_28` runs the
  stock plant while claiming to be a CMA-ES fit.
- **Plant presets replace, they do not inherit** (no `defaults:` header), so
  every preset must define every key `fast_default` interpolates. A missing one
  is a hard `InterpolationKeyError`. All four `sysid_*.yaml` were backfilled to
  18 keys. `sysid_2026_08_17` and `sysid_2026_08_20` were **already broken
  before the merge** — that is pre-existing, not merge damage.

`law_armature` port: master's `apply_controller_pins` only derived it from
`freeze_law_armature` and omitted it from `_ACTIVE_KEYS`, so a preset setting
only that key would early-return and the pin would silently never apply. Both
fixed. `utils/envs/libero.py` auto-merged correctly and keeps the pinned
`gain_exp_base`.

Verified: all 5 plant presets compose against both `libero/fast_libero_object`
and `reach/fast_reach`; 6/6 stub tests for the `law_armature` port.

**Not verified:** `scripts/sysid/test_plant_fit.py` and `test_plant_libero.py`
could not run — `robosuite` is not importable on this workstation at all
(`robosuite.__file__` is `None`; even `robosuite.utils` is missing). Sim runs on
the cluster. Run both there before trusting the merge.

---

## 3. Reach port — plan and progress

Phase 1 and 2 are code-complete and verified offline. Nothing has moved the arm.

| phase | state |
|---|---|
| 1. Sampling, geometry, offline validation | done |
| 2. `RealReach` + shim, base policy, `EE_DELTA`, orientation off | code done, **hardware run pending** |
| 3. Orientation (`delta_max_deg` 60, `goal_mode: target`) | not started |
| 4. Sim/real trace diff | harness done, needs a hardware run to diff |
| 5. Sim reference oracle | deferred; only if phase 4 supports it |

### What transfers

`ReachBaseWrapper` has zero sim references — it reads `robot0_eef_pos` and
`waypoints` from an obs dict. Reused verbatim. So are all the samplers, and
`ReachObservationWrapper` behind the shim below.

### What was dropped

`_rollout_base_reference()` snapshots sim state, rolls the chunked base forward,
and restores. It feeds `reject_infeasible` and the per-episode DTW reference.
You cannot rewind a real arm. Since RL does not run on hardware, the **reward is
dropped entirely** rather than reimplemented — `RealReach.step` returns 0.0.
Phase 5 would restore it by rolling the reference in mujoco under a fitted
plant.

### Files

- `multi-fast/utils/envs/reach_sampling.py` — the samplers, importable without
  robosuite. `reach.py` re-exports every name (1566 -> 1259 lines); an AST
  comparison confirmed no top-level name was lost. The two `transform_utils`
  functions are vendored verbatim, `float32` return included, and match
  robosuite's own doctest. `math.isclose` not `np.isclose` — they differ at
  near-zero.
- `config/reach.yaml` — `workspace`, `curve`, `orientation`, `task`,
  `base_policy`.
- `real_reach_geometry.py` — bounds, derived floor, `sample_episode`.
- `real_reach.py` — `RealReach` + `_RobotView`.
- `scripts/check_reach_workspace.py` — offline sampler validation.
- `scripts/check_real_reach_offline.py` — real base policy vs fake arm.
- `scripts/real_reach_rollout.py` — the hardware driver (`--dry-run` available).
- `scripts/real_reach_viz.py` — animated HTML per episode, written by the
  rollout and re-renderable offline from a run's `results.json`. Modelled on
  `sysid/_viz.py`: 3D scene plus time-series panels on one slider. With `--sim`
  it overlays a sim replay and writes `compare_NNN.html` + `errors.json`.
- `lerobot_robot_bimanual_franka/reach_record.py` — the episode record, shared by
  the rollout and the offline harness so both emit the same artifact.
- `scripts/check_reach_compare_offline.py` — the sim/real comparison against a
  synthesised sim record; no robot, no robosuite.
- `multi-fast/scripts/reach/replay_real_reach.py` — the sim side. Runs in
  `multi-fast/.venv`.

---

## 4. Design choices worth not re-deriving

**Sampling happens in BASE frame, not world.** The right arm is yawed about
136 degrees, so sim's world-frame box would not cover the same reachable
volume. Converting once makes the box arm-agnostic; `robot_base_in_world` maps
it back out and is never inverted.

**The floor is derived, not hardcoded.** `worktable.height_m` +
`distance_min_m` + (`|center_tool|` + `radius`). The sphere term is the
worst case over *any* orientation, which makes the floor orientation-independent
and deliberately conservative. The reason for the conservatism: if
`ActionSafetyScreen` ever fires it silently rescales the goal, and then the
executed trace is no longer the commanded curve. The sampler must stay inside
the region where the screen never fires, not merely inside what is legal.

**Curves are checked whole and redrawn, not clipped.** Bounding the endpoints
does not bound the curve. Clipping would kink the path the base policy tracks.

**The shim forwards, it never computes.** `_RobotView` exists so
`ReachObservationWrapper` runs unmodified rather than being forked into two
copies that drift. It is pure attribute forwarding by design — if it ever
computed a value it could report a number the arm is not running, and the
wrapper could not tell.

**`n_segments: 1`, while sim runs 2.** Deliberate: a single Bezier is one smooth
arc, while multi-segment chains two through a random node and produces a *kink*
— the hardest thing for the cursor and the worst first thing to put on hardware.
Measured acceptance in the real workspace: 77.2% at 1 segment, 61.0% at 2,
41.2% at 3, all fine against `max_sample_attempts: 50`. Wiring it is ~15 lines
(`total_dense = min(n_waypoints * n_segments, MAX_WAYPOINTS)`, plus
`node_velocities: [1.0, 0.0, 0.0]` and `segment_peak_velocity: 1.0` from
`fast_reach.yaml`). **Must close before phase 4** — the trace diff needs both
sides sampling identically.

---

## 5. Numbers

```
worktable (world z)            0.905
brake distance_min             0.010
ee_sphere center_tool/radius   (0,0,-0.1) / 0.1   -> 0.200 below EE worst case
safety floor (world z)         1.115              EE minimum, any orientation
sampler floor (+0.02 margin)   1.135
box ceiling                    1.400
usable z span                  0.265              (sim has 0.55, no table)

right base (world)             [0.669, 0.003, 0.912]  yaw ~-135.75 deg
                               verified: FALSE
bounds (base frame)            [0.25,-0.60,0.223] .. [0.80,0.60,0.488]
homed EE (base / world)        [0.451,0,0.254] / [0.346,-0.312,1.166]
                               31 mm above the sampler floor

max_step * osc_output_max      1.0 * 0.05 = 0.05  == torque.delta.pos_max_m
```

Validation results, both reproducible offline:

```
check_reach_workspace.py --arm right -n 20000
  worst floor clearance +0.0200 m   0 breaches
  worst reach distance   0.7500 / 0.7500   0 breaches
  starts outside box     13 (0.07%) at 1 cm jitter
  PASS   (left arm also PASS)

check_real_reach_offline.py -n 50
  success 50/50, median 29 steps, cursor reached end 50/50
  deltas over envelope 0
  min goal clearance +0.0239 m above the safety floor
  PASS
```

That last line is the phase-1 bound checked end-to-end: not the sampled curve
but every goal `send_action` would compose, driven by the real
`ReachBaseWrapper`. The screen never fires.

---

## 6. Traps found the hard way

- **Gripper.** The base policy emits `0` in the gripper slot meaning "no
  manipulation", but `send_action` reads `{arm}_gripper` as an ABSOLUTE
  normalised position. Forwarding it would drive the gripper shut on step one.
  Held at `task.gripper_norm: 1.0`.
- **Units at the boundary.** `ReachBaseWrapper` emits normalised OSC units;
  `send_action` EE_DELTA takes metres and a delta quaternion. `_action_to_delta`
  converts. Getting it wrong is silent — a normalised 1.0 passed through reads
  as 1 metre, clips to `torque.delta`, and looks like a tracking problem.
- **The delta envelope is exactly saturated.** 1.0 * 0.05 == 0.05 m with zero
  headroom. `RealReach._assert_delta_envelope` fails at construction rather than
  letting it show up in a trace. Do not raise `torque.delta` to fix it — that is
  the sim's own envelope.
- **An out-of-box start cannot be fixed by resampling**, because the curve
  begins at the start point. Now raises immediately instead of burning 50
  attempts.
- **Sampler defaults use the SIM's base_pos.** `sample_goal` and
  `sample_multi_segment_curve` default `base_pos` to `[-0.6, 0, 0.912]`; in base
  frame that must be passed as the origin or reachability is measured from a
  point 1.1 m away.
- `reach.py` line ~1365 still derives `stiffness_exp_scale = kp_limits[1]/kp`
  rather than using the pinned `gain_exp_base`; the pin only landed on the
  LIBERO path. Numerically a no-op at stock gains (1500/150 == 10.0 == the real
  stack's `gain_exp_base`), so it is latent, not live.

---

## 7. Pending, in order

1. **Commit** — see section 1. Do this first; it is the clean point to return to.
2. **Left-arm clearance.** The right arm's sampling box comes within **0.172 m**
   of the left arm's homed EE, and `safety.py` states bimanual arm-repel is not
   implemented. Either park the left arm clear or shrink the `y` bound in
   `config/reach.yaml`. Settle before any motion.
3. **Base pose.** `robot_base_in_world("right")` is `verified: false` — xy and
   yaw carried from a 2026-07 table-frame calibration. Precise risk: the
   rotation is pure yaw so it **cannot** affect a world z, meaning table
   clearance depends only on base translation z = 0.912, inherited from the
   verified left arm on same-table reasoning. `world.yaml` notes the old
   calibration would imply 1.028, which is *higher* and so errs safe. What is
   genuinely unverified is where in the room the arm reaches — i.e. item 2.
4. **Dry run** — `python scripts/real_reach_rollout.py --episodes 1 --dry-run`.
   Connects, homes, samples, prints goals in both frames, sends nothing.
5. **First hardware run** — `--episodes 5`, operator at the e-stop. Watch that
   the trace tracks the curve and that no `!! EE under floor` line prints.
   Traces land in `~/franka_data/real_reach/<timestamp>/`, as `results.json`
   plus one `episode_NNN.html` each — open those to see the measured trail
   against the commanded curve, the commanded-vs-reached gap, and the floor
   clearance over time. Everything in `results.json` is world frame now; the
   2026-09-12 run predates that and stored `goal` in base frame while its
   siblings were world, which `real_reach_viz.py` converts on the way in.
6. Multi-segment (section 4), then phase 3 orientation, then phase 4 trace diff.

For phase 4, turn `torque.osc.cross_coupling_compensation` **off** — it is a
deviation from sim, not toward it. It already is.

### Phase 4 — how the two sides are paired

**A seed cannot pin the same episode on both sides.** Sim samples from the global
legacy `np.random` while `RealReach` uses `default_rng`; sim's bounds lack the
real safety floor (z span 0.550 m against 0.265 m); `sample_goal` is a rejection
sampler whose accept test reads the measured start, so one extra rejection
desynchronises every later draw. Measured: same seed with the start 1 cm apart
moves waypoint 12 by 6 mm; same seed with only the floor applied moves the goal
from z 0.440 to 0.465.

So **the real run samples and sim replays it**. `results.json` carries a
base-frame `replay` block — the post-home `qpos0`, the curve, and per step the
OSC goal that was actually *dispatched* (post `clip_delta`, post fudge, post
`ActionSafetyScreen`) plus the raw normalised action.

```
python scripts/real_reach_rollout.py --episodes 5
cd multi-fast && ./scripts/reach/replay_real_reach.sh \
    ~/franka_data/real_reach/<ts>/results.json --episode 0 [--plant sysid_2026_08_28]
python scripts/real_reach_viz.py ~/franka_data/real_reach/<ts>   # picks up sim_episode_*.json
```

What is replayed is the **absolute OSC goal pose**, not the normalised delta.
Both sides run EE target-pose control, so re-issuing the goal makes the
controller input identical every step and cannot compound — the open-loop
warning in `scripts/utils/replay_real_demo_in_sim.py` is about replaying
*deltas*. `control_delta: False` puts the commanded pose into
`controller.goal_pos` verbatim (verified bit-exact, and the figure prints
`REPLAY NOT FAITHFUL` if it ever is not).

Three traps, all found by running it:

- **`initialization_noise`**. `Reach.__init__` defaults it to `"default"` =
  gaussian sigma 0.02 per joint, and `make_reach_env` does not expose it.
  Setting `init_qpos` and resetting lands 0.0448 rad (2.6 deg) away. The replay
  sets `robots[0].initialization_noise = {"magnitude": 0.0, ...}` before reset,
  which makes it exact.
- **`update_initial_joints`**. Forcing qpos is not enough; without it the OSC
  nullspace bias still points at the Panda's own `init_qpos` all episode.
- **EE convention — and it is NOT `sim_alignment.ee_convention`.** Robosuite's
  OSC drives the gripper's grip *site*; real drives `O_T_EE`. The site sits
  6.9 mm up the tool axis **and is rotated −90° about it**. The config constant
  (−45°) describes the hand *body*, which sim-trained policies observe but the
  controller does not drive. Commanding an `O_T_EE` orientation to the site
  controller asked for an 86° rotation, and the joint swing that corrected it
  dragged the EE the wrong way on step one — the first hardware diff showed
  exactly this. The replay now *measures* the offset at `qpos0` (real's
  recorded `ee_quat0` against sim's site at the same joints), maps every goal
  through it, and writes it to the sim record so the figure inverts the
  identical transform. Nothing is hardcoded.
- **Read the state at the END of the control period.** Reading right after
  `send_action` returns the pose before the arm has responded, so real's
  `trace[t]` was the response to goal t−1 while sim's is the response to goal
  t — real sat one step behind sim in every comparison, and in the first
  trajectory diff real showed 0.0 mm of motion in step 0 against sim's 18 mm,
  which read as a start offset. `RealReach.step` now paces itself
  (send → wait out the period → read), as `Reach.step` returns the post-action
  state in sim; the trajectory script does the same. Records carry
  `obs_timing: post_period`; the figure shifts older ones by the known step and
  says so in the subtitle. Frame 0 of every figure is now the shared start
  pose, so an identical start looks identical.
- **The injected curve must be in sim WORLD.** `Reach._post_action` scores the
  site against `self._waypoints`, and the env keeps its own curve where its
  sampler puts it — world. A base-frame curve is a metre from the arm; the
  cursor still creeps to 24 on the translation-invariant nearest-next rule and
  then never clears the last waypoint, which needs proximity. The replay now
  refuses to run if the site does not start on waypoint 0.

Comparison is in **base frame** (sim's base is a pure translation from its world,
real's carries a yaw), rendered in real's world. Alignment is index-for-index:
sim step *i* consumes real's dispatched goal *i*, so the pairing is definitional
and nothing is warped.

First hardware diff (`20260912_071938`, right arm, 83 steps, real success at
step 83 with cursor 25 at step 41 — after the one-step timing realignment
above; the pre-realignment numbers were flattering by a step):

| plant | pos mean | rot mean | cursor→25 |
|---|---|---|---|
| default | ~20 mm | ~4° | ahead of real |
| `sysid_2026_08_28` | **10.1 mm** | **1.80°** | step 42 (real: 41) |
| `sysid_2026_09_02` | ~13 mm | ~3.3° | between |

**Any LeRobot episode works as the real side too.** `scripts/real_trajectory_rollout.py`
converts the episode's EE_DELTA actions offline into the absolute OSC goals they
produced (through `replay_dataset.py::to_ee_pose_actions`, i.e. the robot's own
goal builder and screen) and writes the same `results.json`, curve-free. Two
sources of "real": `--source dataset` uses the recording itself (FK of the
recorded joints; no arm), `--source arm` re-runs the goals on the arm now. The
same sim script and figure consume both. Mind which controller "real" means:
the cached datasets predate the `cross_coupling_compensation` change, so on
`basket-7-24` ep 3 the *stock* plant matched the recording better (8.7 mm)
than the 08-28 fit (16.6 mm) — the fit was identified against today's
controller and the recording was not made with it. `--source arm` is the fair
comparison for a fitted plant.

**`plant=` is the axis worth sweeping.** `cfg/plant/` holds four CMA-ES fits
beside stock `default`, and the fits move only plant terms — `kp`, `damping` and
`uncouple_pos_ori` are held — so the law is identical and only the plant moves.
Measured on one episode: default vs `sysid_2026_08_28` diverges by up to 33.9 mm
on identical commands, i.e. more than the success threshold. The replay also
applies the real arm's `max_torque_rate_nm_s: 800` and datasheet clamp, which
every shipped plant leaves unlimited.

---

## 8. Reproduce the checks

```
python scripts/check_reach_workspace.py --arm right -n 20000
python scripts/check_real_reach_offline.py -n 50
python -m franka_config dump world
```

Neither needs hardware or robosuite. The sim-side plant tests
(`multi-fast/scripts/sysid/test_plant_fit.py`, `test_plant_libero.py`) need the
cluster.

A rendered version of this plan is at `~/franka_data/real-reach-port.html` --
standalone, opens without signing in, and agrees with this document. It was also
published as a claude.ai artifact, but artifacts belong to the account that made
them, so that URL does not survive the account switch. The local file is the copy
that carries over.
