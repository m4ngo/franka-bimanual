#!/usr/bin/env python3
"""Overlay a real EE_POS teleop dataset on the LIBERO demos of the task it copies.

Two static HTMLs, written under --out-dir:

    eef_aggregate.html    every EE path from both datasets in one 3D scene, with
                          start, grasp and release points marked
    delta_aggregate.html  per-step deltas: the commanded delta (goal - current
                          pose) and the realised per-step motion, as 3D vector
                          clouds and as magnitude / per-axis histograms

plus summary.json with the overlap and delta statistics printed to stdout.

Frames. Both sides are compared as the robosuite grip site in the ROBOT BASE
frame, the one frame the two recordings share:
  real  `action` is an absolute O_T_EE target in the arm's base frame and
        `observation.state` is joint angles. Joints go through the real stack's
        own FK (ee_kinematics), then both poses are moved to the grip site with
        fc.sim_ee_convention's tool-z offset.
  sim   obs/ee_pos is the grip site in LIBERO's world frame. The base pose in
        that frame is FITTED, not assumed: the same FK run on the demo's own
        joint_states, Kabsch-aligned to ee_pos. That fit doubles as the check
        that the FK and tool offset reproduce sim's EE to sub-millimetre; the
        script refuses to plot if they do not.
The eef figure also has a "raw coordinates" view: each dataset exactly as
recorded (real in base frame, sim in LIBERO world), which is what a naive
overlay would show.

Deltas. Sim's commanded delta is actions x torque.delta (osc_pose.json's
output_max: 0.05 m, 0.5 rad), which equals goal_pos - ee_pos. The real
equivalent is target - measured pose at the same frame: the OSC error the
controller sees at the start of the step in both cases. Rotation is compared as
an angle, which is the same in every frame and tool convention.

Usage:
    python scripts/compare_real_libero_demos.py
    python scripts/compare_real_libero_demos.py --repo-id HuskyMango/<ds> \\
        --sim-hdf5 ~/libero_data/libero_90/<task>_demo.hdf5 --out-dir <dir>
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import wasserstein_distance

import franka_config as fc
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos

DEFAULT_REPO = "HuskyMango/filter-9-19-libero-90-9"
# libero_90 task 9 (benchmark.get_benchmark_dict()["libero_90"]().get_task(9)).
DEFAULT_SIM = "~/libero_data/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5"

C_REAL, C_SIM = "#2a78d6", "#e0621f"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
FR3_VEL_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
COVERAGE_RADII_M = (0.01, 0.02, 0.05)
VOXEL_M = 0.02
# Kabsch fit tolerances: past these the FK does not reproduce sim's EE.
FIT_MAX_RESID_M = 2e-3
FIT_MAX_ROT_RAD = 5e-3


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _to_grip_site(pos, quat_xyzw):
    """O_T_EE position -> robosuite grip site, via the tool-z offset in world.yaml."""
    _, pos_tool = fc.sim_ee_convention()
    return pos + Rotation.from_quat(quat_xyzw).apply(pos_tool)


def load_real(repo_id):
    """Per-episode dicts of measured/target grip-site poses in base frame."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(repo_id, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no data/*.parquet in {repo_id}")
    df = pd.concat([pd.read_parquet(hf_hub_download(repo_id, f, repo_type="dataset"))
                    for f in files], ignore_index=True)

    episodes, n_bad = [], 0
    for ep in sorted(df["episode_index"].unique()):
        rows = df[df["episode_index"] == ep].sort_values("frame_index")
        q = np.stack(rows["observation.state"].to_numpy()).astype(np.float64)[:, :7]
        a = np.stack(rows["action"].to_numpy()).astype(np.float64)
        bad = (np.abs(np.diff(q, axis=0)) * fc.control_fps() > FR3_VEL_LIMITS).any(axis=1)
        if bad.any():
            print(f"[real] episode {ep}: frames {np.where(bad)[0].tolist()} exceed FR3 joint "
                  f"velocity limits")
        n_bad += int(bad.sum())
        pos, quat = eef_poses_from_qpos(q)
        tquat = a[:, 3:7] / np.linalg.norm(a[:, 3:7], axis=1, keepdims=True)
        episodes.append(dict(
            name=f"real ep {ep}",
            pos=_to_grip_site(pos, quat), quat=quat,
            goal=_to_grip_site(a[:, :3], tquat), goal_quat=tquat,
            # [0, 1] normalised opening: 1 open, 0 closed.
            closed=a[:, 7] < 0.5,
        ))
    if n_bad > 0.01 * len(df):
        raise SystemExit(f"{n_bad} real frames exceed FR3 joint velocity limits; the joints "
                         f"are not a physical trajectory and FK of them would be meaningless")
    return episodes


def fit_sim_base(demos):
    """Kabsch fit of base -> LIBERO world from FK(joint_states) to obs/ee_pos.

    Returns (R, t, rms_resid_m) with p_world = R @ p_base + t.
    """
    site_base = np.vstack([_to_grip_site(*eef_poses_from_qpos(d["q"])) for d in demos])
    world = np.vstack([d["pos_world"] for d in demos])
    a, b = site_base - site_base.mean(0), world - world.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = world.mean(0) - rot @ site_base.mean(0)
    resid = np.linalg.norm(world - (site_base @ rot.T + t), axis=1)
    if resid.max() > FIT_MAX_RESID_M:
        raise SystemExit(f"FK does not reproduce sim ee_pos: max residual "
                         f"{resid.max() * 1e3:.2f} mm (limit {FIT_MAX_RESID_M * 1e3:.1f} mm)")
    angle = Rotation.from_matrix(rot).magnitude()
    if angle > FIT_MAX_ROT_RAD:
        raise SystemExit(f"sim base is rotated {np.degrees(angle):.2f} deg in LIBERO world; "
                         f"check the fit before trusting a translation-only frame")
    return rot, t, float(np.sqrt((resid ** 2).mean()))


def load_sim(path):
    """Per-demo dicts in base frame, plus the fitted base pose."""
    path = os.path.expanduser(path)
    pos_max, rot_max = fc.control("torque.delta.pos_max_m"), fc.control("torque.delta.rot_max_rad")
    demos = []
    with h5py.File(path, "r") as f:
        for k in sorted(f["data"], key=lambda k: int(k.split("_")[1])):
            g = f[f"data/{k}"]
            act = g["actions"][:].astype(np.float64)
            demos.append(dict(
                name=f"sim {k}",
                q=g["obs/joint_states"][:].astype(np.float64),
                pos_world=g["obs/ee_pos"][:].astype(np.float64),
                d_pos=act[:, :3] * pos_max,
                d_rot=np.linalg.norm(act[:, 3:6] * rot_max, axis=1),
                quat=Rotation.from_rotvec(g["obs/ee_ori"][:]).as_quat(),
                # LIBERO gripper: -1 open, +1 close.
                closed=act[:, 6] > 0,
            ))
    rot, t, rms = fit_sim_base(demos)
    for d in demos:
        d["pos"] = (d["pos_world"] - t) @ rot  # R^T (p - t), row form
        d["d_pos"] = d["d_pos"] @ rot
    return demos, dict(rotation=rot, translation=t, rms_resid_m=rms, path=path)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def events(eps):
    """(starts, grasps, releases) grip-site points, one row per occurrence."""
    starts, grasps, releases = [], [], []
    for e in eps:
        starts.append(e["pos"][0])
        c = e["closed"].astype(int)
        for i in np.where(np.diff(c) == 1)[0] + 1:
            grasps.append(e["pos"][i])
        for i in np.where(np.diff(c) == -1)[0] + 1:
            releases.append(e["pos"][i])
    return tuple(np.asarray(x).reshape(-1, 3) for x in (starts, grasps, releases))


def real_deltas(eps):
    d_pos = np.vstack([e["goal"] - e["pos"] for e in eps])
    d_rot = np.concatenate([(Rotation.from_quat(e["quat"]).inv()
                             * Rotation.from_quat(e["goal_quat"])).magnitude() for e in eps])
    return d_pos, d_rot


def realised_steps(eps):
    """Per-step EE motion: (N,3) metres, (N,) radians."""
    d_pos = np.vstack([np.diff(e["pos"], axis=0) for e in eps])
    d_rot = np.concatenate([(Rotation.from_quat(e["quat"][:-1]).inv()
                             * Rotation.from_quat(e["quat"][1:])).magnitude() for e in eps])
    return d_pos, d_rot


def coverage(a, b, radii):
    """Fraction of a's points within r of some b point, per r."""
    dist, _ = cKDTree(b).query(a)
    return {f"{r * 100:.0f}cm": float((dist <= r).mean()) for r in radii}, dist


def voxel_iou(a, b, size):
    va = {tuple(v) for v in np.floor(a / size).astype(int)}
    vb = {tuple(v) for v in np.floor(b / size).astype(int)}
    return len(va & vb) / len(va | vb)


def _dist_stats(x):
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(np.max(x)))


def _cluster(pts):
    if not len(pts):
        return None
    return dict(n=int(len(pts)), mean=pts.mean(0).round(4).tolist(),
                std=pts.std(0).round(4).tolist())


def summarise(real, sim, sim_base):
    rp, sp = np.vstack([e["pos"] for e in real]), np.vstack([e["pos"] for e in sim])
    r_in_s, r_dist = coverage(rp, sp, COVERAGE_RADII_M)
    s_in_r, s_dist = coverage(sp, rp, COVERAGE_RADII_M)
    rev, sev = events(real), events(sim)
    rc, sc = real_deltas(real), (np.vstack([d["d_pos"] for d in sim]),
                                 np.concatenate([d["d_rot"] for d in sim]))
    rs, ss = realised_steps(real), realised_steps(sim)
    pos_max = fc.control("torque.delta.pos_max_m")

    def delta_block(r, s):
        rn, sn = np.linalg.norm(r[0], axis=1), np.linalg.norm(s[0], axis=1)
        return dict(
            pos_norm_m=dict(real=_dist_stats(rn), sim=_dist_stats(sn),
                            median_ratio_real_over_sim=float(np.median(rn) / np.median(sn)),
                            wasserstein_m=float(wasserstein_distance(rn, sn))),
            rot_rad=dict(real=_dist_stats(r[1]), sim=_dist_stats(s[1]),
                         median_ratio_real_over_sim=float(np.median(r[1]) / np.median(s[1])),
                         wasserstein_rad=float(wasserstein_distance(r[1], s[1]))),
            mean_abs_per_axis_m=dict(real=np.abs(r[0]).mean(0).round(5).tolist(),
                                     sim=np.abs(s[0]).mean(0).round(5).tolist()),
        )

    out = dict(
        sim_base_in_libero_world=dict(
            translation_m=sim_base["translation"].round(5).tolist(),
            rotation_deg=float(np.degrees(Rotation.from_matrix(sim_base["rotation"]).magnitude())),
            fk_rms_resid_mm=sim_base["rms_resid_m"] * 1e3),
        real_base_in_real_world=dict(
            translation_m=fc.robot_base_in_world(fc.profile("single_arm_right").arms["r"]).translation.tolist()),
        counts=dict(real_episodes=len(real), real_frames=len(rp),
                    sim_demos=len(sim), sim_frames=len(sp),
                    real_len_median=float(np.median([len(e["pos"]) for e in real])),
                    sim_len_median=float(np.median([len(e["pos"]) for e in sim]))),
        workspace=dict(
            real_min=rp.min(0).round(4).tolist(), real_max=rp.max(0).round(4).tolist(),
            sim_min=sp.min(0).round(4).tolist(), sim_max=sp.max(0).round(4).tolist(),
            real_points_near_sim=r_in_s, sim_points_near_real=s_in_r,
            real_to_sim_nn_median_m=float(np.median(r_dist)),
            sim_to_real_nn_median_m=float(np.median(s_dist)),
            voxel_iou=dict(size_m=VOXEL_M, iou=voxel_iou(rp, sp, VOXEL_M))),
        events={name: dict(real=_cluster(r), sim=_cluster(s),
                           centroid_offset_real_minus_sim_m=(
                               (r.mean(0) - s.mean(0)).round(4).tolist() if len(r) and len(s) else None))
                for name, r, s in zip(("start", "grasp", "release"), rev, sev)},
        commanded_delta=delta_block(rc, sc),
        realised_step=delta_block(rs, ss),
        real_commanded_over_clip=dict(
            any_axis_frac=float((np.abs(rc[0]) > pos_max).any(1).mean()),
            rot_frac=float((rc[1] > fc.control("torque.delta.rot_max_rad")).mean())),
    )
    return out, (rc, sc), (rs, ss), (rev, sev)


def print_summary(s):
    w, c = s["workspace"], s["counts"]
    print(f"\nreal: {c['real_episodes']} episodes / {c['real_frames']} frames "
          f"(median {c['real_len_median']:.0f}); sim: {c['sim_demos']} demos / "
          f"{c['sim_frames']} frames (median {c['sim_len_median']:.0f})")
    b = s["sim_base_in_libero_world"]
    print(f"sim base in LIBERO world: {b['translation_m']} (rot {b['rotation_deg']:.3f} deg, "
          f"FK resid {b['fk_rms_resid_mm']:.2f} mm rms); real base in real world: "
          f"{s['real_base_in_real_world']['translation_m']}")
    print("\nWORKSPACE OVERLAP (grip site, base frame)")
    print(f"  real points within r of sim: {w['real_points_near_sim']}")
    print(f"  sim points within r of real: {w['sim_points_near_real']}")
    print(f"  nearest-neighbour median: real->sim {w['real_to_sim_nn_median_m'] * 100:.1f} cm, "
          f"sim->real {w['sim_to_real_nn_median_m'] * 100:.1f} cm; "
          f"voxel IoU @{VOXEL_M * 100:.0f} cm {w['voxel_iou']['iou']:.2f}")
    for name, e in s["events"].items():
        if e["real"] and e["sim"]:
            print(f"  {name:8s} real {e['real']['mean']} sim {e['sim']['mean']}  "
                  f"offset {e['centroid_offset_real_minus_sim_m']} m")
    for key, label in (("commanded_delta", "COMMANDED DELTA (goal - current)"),
                       ("realised_step", "REALISED PER-STEP MOTION")):
        d = s[key]
        print(f"\n{label}")
        for q, unit, scale in (("pos_norm_m", "cm", 100), ("rot_rad", "rad", 1)):
            r, m = d[q]["real"], d[q]["sim"]
            print(f"  {q:10s} real median {r['median'] * scale:.3f} p95 {r['p95'] * scale:.3f} | "
                  f"sim median {m['median'] * scale:.3f} p95 {m['p95'] * scale:.3f} {unit}  "
                  f"(real/sim {d[q]['median_ratio_real_over_sim']:.2f})")
        print(f"  mean |d| per axis (m): real {d['mean_abs_per_axis_m']['real']} "
              f"sim {d['mean_abs_per_axis_m']['sim']}")
    o = s["real_commanded_over_clip"]
    print(f"\nreal commanded deltas beyond sim's clip: any axis {o['any_axis_frac'] * 100:.1f}%, "
          f"rotation {o['rot_frac'] * 100:.1f}%")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _nan_join(trajs):
    out, gap = [], np.full((1, 3), np.nan)
    for t in trajs:
        out.extend([t, gap])
    return np.vstack(out[:-1]) if out else np.zeros((0, 3))


def _fixed_cube(arrays, pad=0.03):
    pts = np.vstack([a for a in arrays if len(a)])
    pts = pts[~np.isnan(pts).any(axis=1)]
    lo, hi = pts.min(0), pts.max(0)
    center, half = (lo + hi) / 2, (hi - lo).max() / 2 + pad
    return [[float(c - half), float(c + half)] for c in center]


_AXIS = dict(backgroundcolor=SURFACE, gridcolor=GRID, zerolinecolor=GRID,
             tickfont=dict(color=MUTED, size=10))


def _ttl(t):
    return dict(text=t, font=dict(color=INK, size=12))


def _family_traces(eps, events_, color, label, offset=np.zeros(3)):
    """Paths + start/grasp/release markers for one dataset, shifted by `offset`."""
    pts = _nan_join([e["pos"] + offset for e in eps])
    traces = [go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
        line=dict(color=color, width=2), opacity=0.5,
        name=f"{label} paths ({len(eps)})", legendgroup=label,
        hovertemplate=f"<b>{label}</b><br>%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}} m<extra></extra>",
    )]
    for (ev, symbol, size), pts in zip((("start", "circle", 5), ("grasp", "diamond", 6),
                                        ("release", "square", 5)), events_):
        p = pts + offset
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(color=color, size=size, symbol=symbol,
                        line=dict(color=SURFACE, width=1)),
            name=f"{label} {ev} ({len(p)})", legendgroup=label,
            hovertemplate=f"<b>{label} {ev}</b><br>%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}} m<extra></extra>",
        ))
    return traces


def build_eef_figure(real, sim, evs, sim_base, summary, title):
    """Two views behind one toggle: aligned (base frame) and raw coordinates."""
    rev, sev = evs
    t = sim_base["translation"]
    base_marker = lambda p, name: go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
        marker=dict(size=6, color=INK, symbol="x"), name=name,
        hovertemplate=f"<b>{name}</b><extra></extra>")

    aligned = (_family_traces(sim, sev, C_SIM, "sim") + _family_traces(real, rev, C_REAL, "real")
               + [base_marker(np.zeros(3), "robot base (both)")])
    # Raw: real as recorded (base frame), sim as recorded (LIBERO world).
    raw = (_family_traces(sim, sev, C_SIM, "sim", offset=t)
           + _family_traces(real, rev, C_REAL, "real")
           + [base_marker(np.zeros(3), "real base (its origin)"),
              base_marker(t, "sim base in LIBERO world")])

    for tr in raw:
        tr.visible = False
    fig = go.Figure(aligned + raw)
    n_a, n_r = len(aligned), len(raw)

    def cube(traces):
        return _fixed_cube([np.column_stack([tr.x, tr.y, tr.z]).astype(float) for tr in traces])

    def scene_ranges(r, frame):
        return {f"scene.{ax}axis.range": r[i] for i, ax in enumerate("xyz")} | {
            f"scene.{ax}axis.title.text": f"{ax} (m, {frame})" for ax in "xyz"}

    ra, rr = cube(aligned), cube(raw)
    w = summary["workspace"]
    sub_aligned = (f"Both in robot base frame (grip site). Real points within 2 cm of sim: "
                   f"{w['real_points_near_sim']['2cm'] * 100:.0f}% · sim within 2 cm of real: "
                   f"{w['sim_points_near_real']['2cm'] * 100:.0f}% · voxel IoU "
                   f"@{VOXEL_M * 100:.0f} cm: {w['voxel_iou']['iou']:.2f}")
    sub_raw = (f"Each as recorded: real in its base frame, sim in LIBERO world "
               f"(base at {np.round(t, 3).tolist()} m). A naive overlay is off by exactly that.")
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="right", x=0, y=1.0, xanchor="left", yanchor="bottom",
            bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK, size=11),
            buttons=[
                dict(label="Aligned · robot base frame", method="update",
                     args=[{"visible": [True] * n_a + [False] * n_r},
                           {"title.text": f"{title}<br><sup>{sub_aligned}</sup>",
                            **scene_ranges(ra, "base frame")}]),
                dict(label="Raw recorded coordinates", method="update",
                     args=[{"visible": [False] * n_a + [True] * n_r},
                           {"title.text": f"{title}<br><sup>{sub_raw}</sup>",
                            **scene_ranges(rr, "as recorded")}]),
            ])],
        title=dict(text=f"{title}<br><sup>{sub_aligned}</sup>", font=dict(color=INK, size=15)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        legend=dict(font=dict(color=INK, size=11), bgcolor=SURFACE, bordercolor=GRID,
                    borderwidth=1, itemsizing="constant", groupclick="toggleitem"),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title=_ttl("x (m, base frame)"), range=ra[0], autorange=False, **_AXIS),
            yaxis=dict(title=_ttl("y (m, base frame)"), range=ra[1], autorange=False, **_AXIS),
            zaxis=dict(title=_ttl("z (m, base frame)"), range=ra[2], autorange=False, **_AXIS),
        ),
        height=850, margin=dict(l=0, r=0, t=110, b=0),
    )
    return fig


def _clip_cube(half):
    """Wireframe of the ±half per-axis clip box, as one NaN-joined line."""
    c = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(c[i] != c[j]) == 1]
    return _nan_join([c[[i, j]] for i, j in edges])


def build_delta_figure(commanded, realised, title):
    """Row 1: commanded / realised Δpos clouds in 3D. Rows 2-3: magnitude and per-axis histograms."""
    (rc, sc), (rs, ss) = commanded, realised
    pos_max, rot_max = fc.control("torque.delta.pos_max_m"), fc.control("torque.delta.rot_max_rad")
    fig = make_subplots(
        rows=3, cols=4,
        specs=[[{"type": "scene", "colspan": 2}, None, {"type": "scene", "colspan": 2}, None],
               [{}, {}, {}, {}],
               [{}, {}, {}, None]],
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.07, horizontal_spacing=0.06,
        subplot_titles=["Commanded Δpos (goal − current), cm", "Realised per-step EE motion, cm",
                        "‖commanded Δpos‖", "commanded |Δrot|", "‖realised step‖", "realised |step rot|",
                        "commanded Δx", "commanded Δy", "commanded Δz"])

    rng = np.random.default_rng(0)
    for scene_i, ((r, _), (s, _)) in enumerate(((rc, sc), (rs, ss)), start=1):
        for label, d, color, show in (("sim", s, C_SIM, scene_i == 1), ("real", r, C_REAL, scene_i == 1)):
            idx = rng.permutation(len(d))
            p = d[idx] * 100
            fig.add_trace(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=2, color=color, opacity=0.45),
                name=label, legendgroup=label, showlegend=show,
                hovertemplate=f"<b>{label}</b><br>%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}} cm<extra></extra>",
            ), row=1, col=1 if scene_i == 1 else 3)
    box = _clip_cube(pos_max * 100)
    fig.add_trace(go.Scatter3d(
        x=box[:, 0], y=box[:, 1], z=box[:, 2], mode="lines",
        line=dict(color=MUTED, width=2, dash="dash"),
        name=f"sim clip ±{pos_max * 100:.0f} cm/axis", hoverinfo="skip"), row=1, col=1)

    half = max(np.abs(np.vstack([rc[0], sc[0]])).max(), pos_max) * 100 * 1.05
    for name in ("scene", "scene2"):
        fig.layout[name].update(
            aspectmode="cube",
            xaxis=dict(title=_ttl("Δx"), range=[-half, half], autorange=False, **_AXIS),
            yaxis=dict(title=_ttl("Δy"), range=[-half, half], autorange=False, **_AXIS),
            zaxis=dict(title=_ttl("Δz"), range=[-half, half], autorange=False, **_AXIS))
    # Realised steps are an order smaller; give them their own scale.
    h2 = np.abs(np.vstack([rs[0], ss[0]])).max() * 100 * 1.05
    for ax in ("xaxis", "yaxis", "zaxis"):
        fig.layout.scene2[ax].range = [-h2, h2]

    panels = [
        (np.linalg.norm(rc[0], axis=1) * 100, np.linalg.norm(sc[0], axis=1) * 100, "cm",
         pos_max * 100 * np.sqrt(3), True, (2, 1)),
        (rc[1], sc[1], "rad", rot_max * np.sqrt(3), True, (2, 2)),
        (np.linalg.norm(rs[0], axis=1) * 100, np.linalg.norm(ss[0], axis=1) * 100, "cm", None, True, (2, 3)),
        (rs[1], ss[1], "rad", None, True, (2, 4)),
        (rc[0][:, 0] * 100, sc[0][:, 0] * 100, "cm", pos_max * 100, False, (3, 1)),
        (rc[0][:, 1] * 100, sc[0][:, 1] * 100, "cm", pos_max * 100, False, (3, 2)),
        (rc[0][:, 2] * 100, sc[0][:, 2] * 100, "cm", pos_max * 100, False, (3, 3)),
    ]
    for r, s, unit, bound, one_sided, (row, col) in panels:
        lim = float(np.percentile(np.abs(np.concatenate([r, s])), 99.5))
        start = 0.0 if one_sided else -lim
        bins = dict(start=start, end=lim, size=(lim - start) / 60)
        for label, data, color in (("sim", s, C_SIM), ("real", r, C_REAL)):
            fig.add_trace(go.Histogram(
                x=data, name=label, legendgroup=label, showlegend=False,
                marker=dict(color=color, line=dict(color=SURFACE, width=1)), opacity=0.55,
                xbins=bins, histnorm="probability density",
                hovertemplate=f"{label}<br>%{{x:.3f}} {unit}<br>density %{{y:.2f}}<extra></extra>",
            ), row=row, col=col)
            fig.add_vline(x=float(np.median(data)), line=dict(color=color, width=2),
                          exclude_empty_subplots=False,
                          row=row, col=col)
        if bound is not None and bound <= lim:
            for sign in ([1] if one_sided else [1, -1]):
                fig.add_vline(x=sign * bound, line=dict(color=MUTED, width=1, dash="dash"),
                              exclude_empty_subplots=False,
                              row=row, col=col)
        fig.update_xaxes(title=dict(text=unit, font=dict(color=MUTED, size=10)), gridcolor=GRID,
                         tickfont=dict(color=MUTED, size=9), row=row, col=col)
        fig.update_yaxes(gridcolor=GRID, tickfont=dict(color=MUTED, size=9), row=row, col=col)

    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=15)),
        barmode="overlay", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        legend=dict(font=dict(color=INK, size=11), bgcolor=SURFACE, bordercolor=GRID,
                    borderwidth=1, itemsizing="constant", orientation="h", x=0, y=1.04),
        height=1350, margin=dict(l=50, r=20, t=130, b=40),
    )
    for a in fig.layout.annotations:
        a.font.update(color=INK, size=12)
    return fig


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--sim-hdf5", default=DEFAULT_SIM)
    p.add_argument("--out-dir", default=None,
                   help="default: ~/franka_data/analysis/<dataset>_vs_<sim task>/")
    args = p.parse_args()

    sim_stem = Path(args.sim_hdf5).stem.removesuffix("_demo")
    out = Path(os.path.expanduser(args.out_dir or
               f"~/franka_data/analysis/{args.repo_id.split('/')[-1]}_vs_{sim_stem}"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"[real] {args.repo_id}")
    real = load_real(args.repo_id)
    print(f"[sim]  {args.sim_hdf5}")
    sim, sim_base = load_sim(args.sim_hdf5)

    summary, commanded, realised, evs = summarise(real, sim, sim_base)
    print_summary(summary)

    label = f"{args.repo_id} vs LIBERO {sim_stem}"
    build_eef_figure(real, sim, evs, sim_base, summary,
                     f"EE paths · {label}").write_html(out / "eef_aggregate.html", include_plotlyjs="cdn")
    c, r = summary["commanded_delta"], summary["realised_step"]
    sub = (f"commanded ‖Δpos‖ median real {c['pos_norm_m']['real']['median'] * 100:.2f} cm vs "
           f"sim {c['pos_norm_m']['sim']['median'] * 100:.2f} cm · realised step median real "
           f"{r['pos_norm_m']['real']['median'] * 100:.2f} cm vs sim "
           f"{r['pos_norm_m']['sim']['median'] * 100:.2f} cm · solid line = median, "
           f"dashed = sim's action clip")
    build_delta_figure(commanded, realised, f"Per-step deltas · {label}<br><sup>{sub}</sup>").write_html(
        out / "delta_aggregate.html", include_plotlyjs="cdn")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'eef_aggregate.html'}\nwrote {out / 'delta_aggregate.html'}"
          f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
