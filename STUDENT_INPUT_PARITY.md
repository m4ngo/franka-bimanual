# Student-Policy Input Parity: Sim (multi-fast) vs Real (residual_wrapper)

Reference for verifying that the CrossAttentionPolicy residual (student) sees
identical inputs in sim training/eval and in real deployment. Motivated by
sim2real failures larger than dynamics error alone can explain; the goal is to
rule out input-contract bugs (frame leaks, channel misordering, unit errors)
systematically rather than one at a time.

Sim-side citations refer to the multi-fast snapshot vendored in this workspace
(`multi-fast/`), which is BEHIND the plan-of-record repo — re-verify line
numbers there before acting on them. Real-side citations are this repo
(branch `sysid-tracking`, 2026-07-19). The proprio orientation-frame fix
(`--raw-proprio`, 45.03° / 6.9 mm sim-convention correction) is already in.

---

## 1. The input contract

`CrossAttentionPolicy.forward(point_cloud, proprio, base_action)`
(`multi-fast/utils/distill/policy.py:342`). All float32. Each cross-attention
token = `[base_action_step(7), waypoint_xyz(3), proprio(17)]` — note proprio
(including absolute eef_pos) enters EVERY token raw.

### 1a. proprio (17,)

Sim concat order = `proprio_keys` order (`cfg/distill/train.yaml:69`,
`utils/distill/dataset.py:421`). `policy.py:165` asserts slots 0:3 are eef_pos.

| slot | sim (training) | real (`run_residual.py:235-244`) | parity |
|---|---|---|---|
| 0:3 | `robot0_eef_pos` — EE-site pos, sim WORLD frame, raw meters | fk pos + sim-convention correction, world frame via calibration (`proprio_frame="world"`) | frame ORIGIN unverified — see finding F5 |
| 3:7 | `robot0_eef_quat` — xyzw, world frame | fk quat + 45° hand-body correction, world-rotated | fixed by `--raw-proprio` commit; verify numerically (Tier 3) |
| 7:9 | `robot0_gripper_qpos` — TWO finger joint positions, METERS (≈ ±0.02–0.04) | `split_gripper` = `(g, −g)`, g = r_gripper × 0.04 m (Panda hand on rig; width/80 mm obs) | F3 fixed 2026-07-19 |
| 9:11 | `controller_state` = `[damping_norm, kp_norm]`, log-normalized, [0,0] at defaults (`utils/envs/robomimic.py:181-187`) | `[prev_kd, prev_kp]` — last-applied normalized gain actions (= sim convention) | F2 fixed 2026-07-19 |
| 11:17 | `robot0_eef_vel` — MEASURED world-frame twist | `O_dP_EE_d` — COMMANDED twist, ROBOT-BASE frame, never world-rotated | **F4: frame + semantics** |

No normalization or noise is applied to proprio in the student pipeline
(train-time z-rotation augment co-rotates pos/quat/vel about world z only).

### 1b. point_cloud (N, 3)

| aspect | sim | real |
|---|---|---|
| source | rendered MuJoCo depth, cams `[agentview ×3 perturbed views, robot0_eye_in_hand]`, one view sampled per training step | 2 FRAMOS D415e scene cams (no wrist depth) |
| frame | camera → world via stored extrinsics (`dataset.py:251`) | world via chessboard calibration |
| crop | axis-aligned BOX, half_extent 0.4 m around eef (`pointcloud.py:275`) | SPHERE, radius 0.4 m around eef (`bimanual_franka.py:302`) — F8 |
| count | 4096 stored → 2048 random subsample at train (eval ~4096) | 2048, fixed per-camera quota |
| centering | `center_on_eef: True` (train.yaml) — subtract world eef_pos | hardcoded True in `policy_wrapper.py:156`, ckpt `data_kwargs` IGNORED — F6 |
| rgb | dropped (`use_rgb: False`) → xyz only | xyz only |

### 1c. base_action (70,) = 10 steps × 7

Both sides: `[dx, dy, dz, drx, dry, drz, grip]` per step, normalized [-1,1]
(pos ±1 = ±0.05 m, rot ±1 = ±0.5 rad axis-angle), gains never included.
Real reconstructs normalization in `env_wrapper.process_chunk` (meters/0.05,
quat→rotvec/0.5, gripper (g−0.5)×2). Open items: gripper sign convention
match; rotation-delta FRAME (sim world-frame axis-angle vs real base-frame
composition — z-yaw absorbed by train augment, tilt is not).

### 1d. Output (5 steps × 9) and downstream handling

Per-step order (variable impedance): **`[damping, stiffness, dx, dy, dz,
drx, dry, drz, grip]` — DAMPING FIRST** (SB3 `fast/utils.py:57-78`,
`dataset.py:373-395`, `inference.py:246-252`). No tanh/clip inside the model.

| step | sim (`StudentPredictor.predict_diffused`) | real |
|---|---|---|
| clip | gains ±0.5, residual ±0.2 | same constants (`policy_wrapper.py:203-206`) |
| combine | `clip(base + residual, −1, 1)` | `cache_delta` add, NO final clip — F7 |
| gains | `[damping, kp]` → env exp-maps | `kp = res[0]; kd = res[1]` — **F1: reads damping as kp** |

---

## 2. Findings (ranked)

- **F1 — residual gain channels swapped** — FIXED 2026-07-19: robot stiffness
  was driven by the student's damping output. `run_residual.py` now reads
  `kp = res[1]` (stiffness), `kd = res[0]` (damping; inert on this stack but
  kept in the right channel for logging/parity). Ordering verified first-hand
  at three sim sites: SB3 `fast/utils.augment_controller_action` prepend,
  `inference.py` gains clip `[:2]`, env wrapper `action[0]=damping,
  action[1]=stiffness`.
- **F2 — proprio gain slots wrong convention + order** — FIXED 2026-07-19:
  slots 9:11 now carry `[prev_kd, prev_kp]` — the last-applied normalized gain
  actions, which equal sim's log-normalized `[damping_norm, kp_norm]` exactly
  (the log-normalization inverts the env's exponential gain mapping; 0,0 at
  defaults, matching the reset state). The dead `"gains"` obs key was removed.
  Caveat: on real, the damping channel reports the *commanded* value; the
  plant ignores it (kd inert) — same numbers as sim sees, different physics.
- **F3 — gripper qpos units** — FIXED 2026-07-19: the rig now carries the
  Franka Panda hand (swapped back from the WSG for sim consistency), so
  `split_gripper` converts the normalized [0,1] width obs (vs 80 mm) to exact
  Panda finger qpos: `g = r_gripper * 0.04` m, slots 7:9 = `(g, -g)` —
  physically identical to robosuite's `robot0_gripper_qpos`.
- **F4 — eef_vel**: rotate into world frame; prefer measured over commanded
  twist if available (`O_dP_EE_c` vs `O_dP_EE_d` — note sysid.py records the
  d-variant; check what training used before matching).
- **F5 — world-frame origin**: verify real calibrated world origin/axes match
  robosuite's world (table height!). Absolute eef_pos enters every token.
  Check training-data eef_pos range vs real (Tier 2 catches this).
- **F6 — `center_on_eef` hardcoded** (`policy_wrapper.py:156`): honor
  `ckpt["data_kwargs"]` for center_on_eef, num_points, use_rgb,
  crop_half_extent.
- **F7 — missing final clip** on base+residual sum (sim: ±1 normalized).
- **F8 — crop geometry**: box (sim) vs sphere (real); per-camera quota vs
  single-view sampling; no real analogue of wrist-view training clouds.

Status: F1-F3 fixed 2026-07-19; F4-F8 OPEN. Update this list as fixes land.

---

## 3. Verification plan

### Tier 0 — contract fixes + pin
Fix F1–F4, F6, F7. Add `test_student_input_parity.py` asserting channel
order/scales/conventions of the real wrapper against constants imported from
multi-fast (companion to `test_controller_parity.py`), so the contract cannot
drift silently again.

### Tier 1 — golden-bundle cross-inference (no robot; the decisive test)
Ground truth = distill HDF5s (`data/demo_i/{obs/*, base_action, actions}`)
plus ckpt `data_kwargs`.
1. Drive the same demo rows through BOTH preprocessing paths (multi-fast
   `DistillDataset`/`StudentPredictor` vs real `ResidualPolicy.infer` fed a
   reformatted `residual_obs`). Assert the three forward() tensors match to
   ~1e-6, then outputs.
2. Reverse direction: `--dump-obs` flag in `run_residual.py` saving every
   `residual_obs` bundle (+ ckpt SHA) to npz; replay through multi-fast eval
   preprocessing; diff.
Any wrapper bug becomes a deterministic per-channel diff.

### Tier 2 — per-channel distribution audit
Script computing per-channel stats/histograms of chunk (70,), proprio (17,),
and cloud summaries (per-axis extents, points-in-crop, table-plane distance)
over (a) training HDF5s, (b) dumped real bundles. Flag any channel whose real
values fall outside the train p1–p99 envelope. This is the systematic
detector: it catches F3/F5-class bugs immediately, including future ones.

### Tier 3 — matched-state + modality-swap probes
1. Matched state: set sim robot q = real homed q (sysid teleport mechanism),
   capture both bundles, diff channel-by-channel — catches state-dependent
   issues (FK conventions, calibration).
2. Modality bisection: sim eval with real-recorded clouds (and vice versa);
   `zero_residual` / `disable_pcd` ablations on both sides. If performance
   tracks one swapped modality, the remaining gap lives there.

Order of execution: Tier 0 (F1–F3 first — hours, likely largest win) →
Tier 1 harness (validates the fixes) → instrument + Tier 2 → Tier 3.

---

## 4. Pointers

- Real: `residual_wrapper/run_residual.py` (obs assembly ~195-320),
  `residual_wrapper/policy_wrapper.py` (`ResidualPolicy`),
  `residual_wrapper/env_wrapper.py` (scales, `current_ee_pose`,
  `process_chunk`, `split_gripper`),
  `lerobot_robot_bimanual_franka/.../bimanual_franka.py` (point cloud,
  world calibration `_r_robot_in_world`).
- Sim: `multi-fast/utils/distill/{policy,dataset,inference}.py`,
  `multi-fast/utils/envs/{robomimic,libero,pointcloud}.py`,
  `multi-fast/cfg/distill/{train,collect,eval}.yaml`.
- Checkpoint contract: `ckpt["model_init_kwargs"]` (architecture) and
  `ckpt["data_kwargs"]` (preprocessing) — eval must reconstruct preprocessing
  from `data_kwargs`, never hardcode.
- Related: `CONTROL_STACK.md` (action path), `SIM_VS_REAL_CONTROL.md`
  (controller gap), `sysid/SYSID.md` (dynamics fitting pipeline).
