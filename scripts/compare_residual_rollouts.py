#!/usr/bin/env python3
"""How far do real-arm rollouts stray from the real demonstrations they learned from?

Reads every `run_residual.py` rollout filed under the training dataset
(`~/franka_data/outputs/<train-dataset>/*-multifast/`, see baselines/ROLLOUT.md)
and splits it by whether the residual was on, so the residual's own effect is
read against the base policy's, not against zero. Writes, under --out-dir:

    rollout_eef_aggregate.html        every EE path in one 3D scene, by method, with a
                                      second view colouring rollout points by their
                                      distance from the nearest demo point
    rollout_deviation_aggregate.html  deviation (per point, per episode, along task
                                      progress) and per-step delta distributions
    summary.json                      the numbers printed to stdout

Frames. Demos and rollouts are recorded on the same arm in the same base frame,
so no alignment is needed: joints go through the real stack's FK and onto the
robosuite grip site exactly as compare_real_libero_demos.py does it.

Deviation, two ways:
  per point    distance from each rollout EE sample to the nearest sample of ANY
               demo. Ignores timing: "is the arm somewhere a demo went?"
  per episode  DTW distance to the closest single demo, as the mean grip-site gap
               along the optimal time alignment. Penalises a path that visits the
               right places in the wrong order or detours between them.
Both are reported for the demos themselves, leave-one-out, as the floor: the
spread 30 human demonstrations already have among themselves.

Deltas. A rollout's commanded delta is the recorded EE_DELTA action (target minus
the pose read at send time); a demo's is its absolute target minus the measured
pose. Rotation is compared as an angle. The rollout delta is the TCP's and the
demo's the grip site's; the 6.9 mm tool offset moves a delta by under 0.5 mm at
these rotation sizes. Realised motion and its second difference (a jitter
measure) come from FK of the recorded joints on both sides.

Usage:
    python scripts/compare_residual_rollouts.py
    python scripts/compare_residual_rollouts.py --runs 20261003_155259-multifast 20261001_165525-multifast
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numba import njit
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import compare_real_libero_demos as D
import franka_config as fc
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos

DEMOS, BASE, RESIDUAL = "demos", "base only", "base + residual"
FAMILIES = (DEMOS, BASE, RESIDUAL)
# Categorical slots 1-3, the reference palette's all-pairs-validated set.
COLORS = {DEMOS: "#2a78d6", BASE: "#1baf7a", RESIDUAL: "#eb6834"}
SYMBOLS = {BASE: "circle", RESIDUAL: "diamond"}
# multi-fast utils/sysid/viz.py's orange ordinal ramp, light -> dark.
DEV_RAMP = ["#f39d6b", "#ee7f45", "#e0621f", "#b04a18", "#7d3410"]
PROGRESS_BINS = 20


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def discover_runs(outputs_root, run_ids=None):
    """[(run_dir, manifest)] for multifast runs with a recorded dataset."""
    runs = []
    for path in sorted(glob.glob(os.path.join(outputs_root, "*-multifast", "manifest.json"))):
        run_dir = os.path.dirname(path)
        if run_ids and os.path.basename(run_dir) not in run_ids:
            continue
        m = json.load(open(path))
        if not glob.glob(os.path.join(run_dir, "dataset", "data", "*", "*.parquet")):
            print(f"[runs] {os.path.basename(run_dir)}: no recorded dataset, skipped")
            continue
        runs.append((run_dir, m))
    if not runs:
        raise SystemExit(f"no multifast runs with a recorded dataset under {outputs_root}")
    return runs


def load_rollouts(run_dir, manifest):
    """Per-episode dicts in the same layout D.load_real produces, plus delta/verdict fields."""
    policy, params = manifest["policy"], manifest["parameters"]
    family = RESIDUAL if policy["residual_enabled"] else BASE
    run_id = os.path.basename(run_dir)
    verdicts = {}
    ep_log = os.path.join(run_dir, "episodes.jsonl")
    if os.path.exists(ep_log):
        for line in open(ep_log):
            e = json.loads(line)
            verdicts[e.get("dataset_episode_index", e["episode"])] = bool(e.get("success"))
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob(os.path.join(run_dir, "dataset", "data", "*", "*.parquet")))],
                   ignore_index=True)
    out = []
    for ep in sorted(df["episode_index"].unique()):
        rows = df[df["episode_index"] == ep].sort_values("frame_index")
        q = np.stack(rows["observation.state"].to_numpy()).astype(np.float64)[:, :7]
        a = np.stack(rows["action"].to_numpy()).astype(np.float64)
        if len(q) < 3:
            continue
        pos, quat = eef_poses_from_qpos(q)
        out.append(dict(
            name=f"{run_id} ep {ep}", run=run_id, family=family,
            success=verdicts.get(int(ep)),
            pos=D._to_grip_site(pos, quat), quat=quat,
            closed=a[:, 7] < 0.5,
            d_pos=a[:, :3],
            # Delta quaternion xyzw; abs(w) takes the shorter rotation.
            d_rot=2.0 * np.arctan2(np.linalg.norm(a[:, 3:6], axis=1), np.abs(a[:, 6])),
        ))
    return out, dict(run=run_id, family=family, episodes=len(out),
                     residual=policy.get("residual_policy", {}).get("path") if family == RESIDUAL else None,
                     base=policy["base_policy"]["path"],
                     infer_lead=params.get("infer_lead"),
                     residual_params=params.get("residual") if family == RESIDUAL else None)


# ---------------------------------------------------------------------------
# deviation
# ---------------------------------------------------------------------------
@njit(cache=True)
def _dtw_mean(a, b):
    """Mean point gap along the optimal DTW alignment of (n,3) a and (m,3) b."""
    n, m = a.shape[0], b.shape[0]
    cost = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.sqrt(((a[i - 1] - b[j - 1]) ** 2).sum())
            best, k = cost[i - 1, j - 1], steps[i - 1, j - 1]
            if cost[i - 1, j] < best:
                best, k = cost[i - 1, j], steps[i - 1, j]
            if cost[i, j - 1] < best:
                best, k = cost[i, j - 1], steps[i, j - 1]
            cost[i, j] = best + d
            steps[i, j] = k + 1.0
    return cost[n, m] / steps[n, m]


def nearest_demo_dtw(traj, demos, skip=None):
    """(mean gap m, index of the closest demo), skipping index `skip`."""
    best, idx = np.inf, -1
    for i, d in enumerate(demos):
        if i == skip:
            continue
        v = _dtw_mean(traj, d["pos"])
        if v < best:
            best, idx = v, i
    return float(best), idx


def attach_deviation(demos, rollouts):
    """Adds `nn` (per-point distance to the demo cloud) and `dtw` to every episode.

    Demos are scored leave-one-out, against the other demos only.
    """
    owner = np.concatenate([np.full(len(d["pos"]), i) for i, d in enumerate(demos)])
    cloud = np.vstack([d["pos"] for d in demos])
    tree = cKDTree(cloud)
    for e in rollouts:
        e["nn"], _ = tree.query(e["pos"])
        e["dtw"], e["dtw_demo"] = nearest_demo_dtw(e["pos"], demos)
    for i, d in enumerate(demos):
        others = cKDTree(cloud[owner != i])
        d["nn"], _ = others.query(d["pos"])
        d["dtw"], d["dtw_demo"] = nearest_demo_dtw(d["pos"], demos, skip=i)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def task_events(eps):
    """(starts, first grasps, final releases), one row per episode that has the event.

    A rollout can close and reopen many times; the first close is the pick and the
    last open the place. Counting every toggle drags the release centroid onto the
    bowl wherever the policy re-grasped.
    """
    starts, grasps, releases = [], [], []
    for e in eps:
        starts.append(e["pos"][0])
        c = np.diff(e["closed"].astype(int))
        close, open_ = np.where(c == 1)[0] + 1, np.where(c == -1)[0] + 1
        if len(close):
            grasps.append(e["pos"][close[0]])
            if len(open_) and open_[-1] > close[0]:
                releases.append(e["pos"][open_[-1]])
    return tuple(np.asarray(x).reshape(-1, 3) for x in (starts, grasps, releases))


def gripper_closes(e):
    return int((np.diff(e["closed"].astype(int)) == 1).sum())


def commanded(eps, family):
    if family == DEMOS:
        return D.real_deltas(eps)
    return np.vstack([e["d_pos"] for e in eps]), np.concatenate([e["d_rot"] for e in eps])


def accel(eps):
    """‖second difference‖ of the grip-site path, m per step²: high = jitter."""
    return np.concatenate([np.linalg.norm(np.diff(e["pos"], n=2, axis=0), axis=1) for e in eps])


def family_metrics(groups):
    """{family: {quantity: values}} over every quantity the figures and summary use."""
    out = {}
    for fam, eps in groups.items():
        if not eps:
            continue
        cp, cr = commanded(eps, fam)
        sp, sr = D.realised_steps(eps)
        out[fam] = dict(
            nn=np.concatenate([e["nn"] for e in eps]),
            dtw=np.array([e["dtw"] for e in eps]),
            cmd_pos=np.linalg.norm(cp, axis=1), cmd_rot=cr,
            step_pos=np.linalg.norm(sp, axis=1), step_rot=sr,
            accel=accel(eps),
            length=np.array([len(e["pos"]) for e in eps]),
        )
    return out


def _stats(x):
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                mean=float(np.mean(x)))


def summarise(groups, metrics, runs_info):
    ref = metrics[DEMOS]
    s = dict(runs=runs_info, families={})
    for fam, m in metrics.items():
        eps = groups[fam]
        evs = task_events(eps)
        succ = [e.get("success") for e in eps if e.get("success") is not None]
        s["families"][fam] = dict(
            episodes=len(eps), frames=int(sum(len(e["pos"]) for e in eps)),
            successes=int(sum(succ)) if succ else None,
            length_median=float(np.median(m["length"])),
            gripper_closes_per_episode=dict(
                mean=float(np.mean([gripper_closes(e) for e in eps])),
                max=int(max(gripper_closes(e) for e in eps))),
            nn_m=_stats(m["nn"]),
            nn_within={f"{r * 100:.0f}cm": float((m["nn"] <= r).mean()) for r in D.COVERAGE_RADII_M},
            dtw_m=_stats(m["dtw"]),
            dtw_median_over_demo_floor=float(np.median(m["dtw"]) / np.median(ref["dtw"])),
            cmd_pos_m=_stats(m["cmd_pos"]), cmd_rot_rad=_stats(m["cmd_rot"]),
            step_pos_m=_stats(m["step_pos"]), step_rot_rad=_stats(m["step_rot"]),
            accel_m=_stats(m["accel"]),
            events={name: D._cluster(pts) for name, pts in zip(("start", "grasp", "release"), evs)},
        )
        if fam != DEMOS:
            demo_ev = task_events(groups[DEMOS])
            s["families"][fam]["event_offset_from_demos_m"] = {
                name: ((pts.mean(0) - dp.mean(0)).round(4).tolist() if len(pts) and len(dp) else None)
                for name, pts, dp in zip(("start", "grasp", "release"), evs, demo_ev)}
    s["per_run"] = {}
    for fam in (BASE, RESIDUAL):
        for e in groups[fam]:
            r = s["per_run"].setdefault(e["run"], dict(family=fam, dtw_cm=[], success=[]))
            r["dtw_cm"].append(round(e["dtw"] * 100, 2))
            r["success"].append(e.get("success"))
            r.setdefault("gripper_closes", []).append(gripper_closes(e))
    return s


def print_summary(s):
    print("\nRUNS")
    for r in s["runs"]:
        print(f"  {r['run']:28s} {r['family']:16s} {r['episodes']:2d} eps  infer_lead={r['infer_lead']}")
    print("\nGRIPPER CLOSES PER EPISODE (demos close once)")
    for fam, f in s["families"].items():
        g = f["gripper_closes_per_episode"]
        print(f"  {fam:16s} mean {g['mean']:.1f}  max {g['max']}")
    print(f"\n{'':18s} {'eps':>4s} {'succ':>5s} {'len':>5s} | {'nn med':>7s} {'nn p95':>7s} "
          f"{'<2cm':>5s} | {'DTW med':>8s} {'xfloor':>6s} | {'cmd|dp|':>8s} {'cmd|dr|':>8s} "
          f"{'step':>6s} {'accel':>6s}")
    for fam, f in s["families"].items():
        succ = "-" if f["successes"] is None else str(f["successes"])
        print(f"  {fam:16s} {f['episodes']:4d} {succ:>5s} {f['length_median']:5.0f} | "
              f"{f['nn_m']['median'] * 100:6.2f}c {f['nn_m']['p95'] * 100:6.2f}c "
              f"{f['nn_within']['2cm'] * 100:4.0f}% | {f['dtw_m']['median'] * 100:7.2f}c "
              f"{f['dtw_median_over_demo_floor']:6.2f} | {f['cmd_pos_m']['median'] * 100:7.2f}c "
              f"{f['cmd_rot_rad']['median']:8.3f} {f['step_pos_m']['median'] * 100:5.2f}c "
              f"{f['accel_m']['median'] * 1000:5.2f}mm")
    print("  (nn = distance to nearest demo point; DTW = mean gap to the closest demo along time "
          "alignment;\n   xfloor = DTW median / the demos' own leave-one-out median; "
          "accel = ‖2nd difference‖ per step)")
    for fam in (BASE, RESIDUAL):
        if fam in s["families"]:
            off = s["families"][fam]["event_offset_from_demos_m"]
            print(f"  {fam} event centroid - demo centroid (m): {off}")
    print("\nPER RUN (DTW cm per episode, success, gripper closes)")
    for run, r in s["per_run"].items():
        print(f"  {run:28s} {r['family']:16s} {r['dtw_cm']}  {r['success']}  closes {r['gripper_closes']}")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _hover(e):
    verdict = "" if e.get("success") is None else (" · success" if e["success"] else " · FAILED")
    return f"{e['name']}{verdict}"


def _method_traces(groups):
    traces = []
    for fam in FAMILIES:
        eps = groups[fam]
        if not eps:
            continue
        traces += D._family_traces(eps, task_events(eps), COLORS[fam], fam)
        failed = np.array([e["pos"][-1] for e in eps if e.get("success") is False]).reshape(-1, 3)
        if len(failed):
            traces.append(go.Scatter3d(
                x=failed[:, 0], y=failed[:, 1], z=failed[:, 2], mode="markers",
                marker=dict(color=COLORS[fam], size=6, symbol="x"),
                name=f"{fam} failed end ({len(failed)})", legendgroup=fam,
                text=[_hover(e) for e in eps if e.get("success") is False],
                hovertemplate="<b>%{text}</b> end<extra></extra>"))
    return traces


def _deviation_traces(groups, cmax):
    pts = D._nan_join([e["pos"] for e in groups[DEMOS]])
    traces = [go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
        line=dict(color=D.MUTED, width=2), opacity=0.35,
        name=f"demos ({len(groups[DEMOS])})", hoverinfo="skip")]
    first = True
    for fam in (BASE, RESIDUAL):
        eps = groups[fam]
        if not eps:
            continue
        p = np.vstack([e["pos"] for e in eps])
        nn = np.concatenate([e["nn"] for e in eps])
        label = np.concatenate([[_hover(e)] * len(e["pos"]) for e in eps])
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=3 if fam == BASE else 3.5, symbol=SYMBOLS[fam], color=nn * 100,
                        colorscale=[[i / (len(DEV_RAMP) - 1), c] for i, c in enumerate(DEV_RAMP)],
                        cmin=0, cmax=cmax * 100, showscale=first,
                        colorbar=dict(title=dict(text="cm to nearest<br>demo point",
                                                 font=dict(color=D.INK, size=11)),
                                      tickfont=dict(color=D.MUTED, size=10), len=0.5, thickness=12)),
            name=f"{fam} ({SYMBOLS[fam]})", text=label,
            hovertemplate="<b>%{text}</b><br>%{marker.color:.1f} cm from nearest demo point"
                          "<extra></extra>"))
        first = False
    return traces


def build_eef_figure(groups, summary, title):
    method = _method_traces(groups)
    rollout_nn = np.concatenate([e["nn"] for f in (BASE, RESIDUAL) for e in groups[f]])
    deviation = _deviation_traces(groups, float(np.percentile(rollout_nn, 99)))
    for tr in deviation:
        tr.visible = False
    fig = go.Figure(method + deviation)
    n_m, n_d = len(method), len(deviation)
    ranges = D._fixed_cube([e["pos"] for f in FAMILIES for e in groups[f]] + [np.zeros((1, 3))])

    fams = summary["families"]
    sub = " · ".join(f"{fam}: median {fams[fam]['nn_m']['median'] * 100:.1f} cm to nearest demo point, "
                     f"DTW {fams[fam]['dtw_m']['median'] * 100:.1f} cm"
                     for fam in (BASE, RESIDUAL) if fam in fams)
    sub += f" · demos among themselves: DTW {fams[DEMOS]['dtw_m']['median'] * 100:.1f} cm"
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=6, color=D.INK, symbol="x"),
        name="robot base", hovertemplate="<b>robot base</b><extra></extra>"))
    vis = lambda a, b: [a] * n_m + [b] * n_d + [True]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="right", x=0, y=1.0, xanchor="left", yanchor="bottom",
            bgcolor=D.SURFACE, bordercolor=D.GRID, font=dict(color=D.INK, size=11),
            buttons=[dict(label="By method", method="update", args=[{"visible": vis(True, False)}]),
                     dict(label="Deviation from demos", method="update",
                          args=[{"visible": vis(False, True)}])])],
        title=dict(text=f"{title}<br><sup>{sub}</sup>", font=dict(color=D.INK, size=15)),
        paper_bgcolor=D.SURFACE, plot_bgcolor=D.SURFACE,
        legend=dict(font=dict(color=D.INK, size=11), bgcolor=D.SURFACE, bordercolor=D.GRID,
                    borderwidth=1, itemsizing="constant", groupclick="toggleitem"),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title=D._ttl("x (m, base frame)"), range=ranges[0], autorange=False, **D._AXIS),
            yaxis=dict(title=D._ttl("y (m, base frame)"), range=ranges[1], autorange=False, **D._AXIS),
            zaxis=dict(title=D._ttl("z (m, base frame)"), range=ranges[2], autorange=False, **D._AXIS),
        ),
        height=850, margin=dict(l=0, r=0, t=110, b=0),
    )
    return fig


def _ecdf(fig, values, fam, scale, unit, row, col, show):
    x = np.sort(values) * scale
    y = np.arange(1, len(x) + 1) / len(x)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=COLORS[fam], width=2),
        name=fam, legendgroup=fam, showlegend=show,
        hovertemplate=f"{fam}<br>%{{x:.3f}} {unit}<br>%{{y:.0%}} of samples below<extra></extra>",
    ), row=row, col=col)
    med = float(np.median(values)) * scale
    fig.add_trace(go.Scatter(
        x=[med], y=[0.5], mode="markers",
        marker=dict(color=COLORS[fam], size=8, line=dict(color=D.SURFACE, width=2)),
        legendgroup=fam, showlegend=False,
        hovertemplate=f"{fam} median %{{x:.3f}} {unit}<extra></extra>",
    ), row=row, col=col)


def build_deviation_figure(groups, metrics, summary, title):
    fig = make_subplots(
        rows=4, cols=3,
        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}],
               [{"type": "table", "colspan": 3}, None, None]],
        row_heights=[0.23, 0.21, 0.21, 0.35], vertical_spacing=0.07, horizontal_spacing=0.07,
        subplot_titles=[
            "Distance to nearest demo point (ECDF)", "Per-episode DTW gap to closest demo",
            "Distance to nearest demo point vs task progress",
            "Commanded ‖Δpos‖ (ECDF)", "Commanded |Δrot| (ECDF)", "Realised step ‖Δpos‖ (ECDF)",
            "Realised step |Δrot| (ECDF)", "Path ‖2nd difference‖ — jitter (ECDF)",
            "Episode length (ECDF)", ""])

    fams = [f for f in FAMILIES if f in metrics]

    def clip_x(key, scale, row, col):
        """Range to the 99.5th percentile: one outlier would otherwise flatten every curve."""
        hi = max(np.percentile(metrics[f][key], 99.5) for f in fams) * scale * 1.05
        fig.update_xaxes(range=[0, hi], row=row, col=col)

    for f in fams:
        _ecdf(fig, metrics[f]["nn"], f, 100, "cm", 1, 1, True)
    clip_x("nn", 100, 1, 1)

    # Per-episode DTW: one dot per episode, filled = success, open = failed / no verdict.
    rng = np.random.default_rng(0)
    for i, f in enumerate(fams):
        eps = groups[f]
        y = np.array([e["dtw"] for e in eps]) * 100
        x = i + rng.uniform(-0.18, 0.18, len(eps))
        ok = np.array([e.get("success") is not False for e in eps])
        for mask, symbol, suffix in ((ok, "circle", ""), (~ok, "circle-open", " failed")):
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=x[mask], y=y[mask], mode="markers",
                    marker=dict(color=COLORS[f], size=9, symbol=symbol,
                                line=dict(color=D.SURFACE if symbol == "circle" else COLORS[f], width=1.5)),
                    legendgroup=f, showlegend=False,
                    text=[_hover(e) for e, m in zip(eps, mask) if m],
                    hovertemplate="<b>%{text}</b><br>DTW %{y:.2f} cm<extra></extra>",
                ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[i - 0.3, i + 0.3], y=[np.median(y)] * 2, mode="lines",
            line=dict(color=D.INK, width=2), showlegend=False,
            hovertemplate=f"{f} median %{{y:.2f}} cm<extra></extra>"), row=1, col=2)
    fig.update_xaxes(tickvals=list(range(len(fams))),
                     ticktext=[f"{f}<br>({len(groups[f])})" for f in fams], row=1, col=2)
    fig.update_yaxes(title=dict(text="cm", font=dict(color=D.MUTED, size=10)), rangemode="tozero",
                     row=1, col=2)

    # Deviation along normalised progress: median line, interquartile band.
    edges = np.linspace(0, 1, PROGRESS_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for f in fams:
        prog = np.concatenate([np.linspace(0, 1, len(e["pos"])) for e in groups[f]])
        nn = np.concatenate([e["nn"] for e in groups[f]]) * 100
        b = np.clip(np.digitize(prog, edges) - 1, 0, PROGRESS_BINS - 1)
        q = np.array([np.percentile(nn[b == k], [25, 50, 75]) for k in range(PROGRESS_BINS)])
        fig.add_trace(go.Scatter(
            x=np.concatenate([centers, centers[::-1]]), y=np.concatenate([q[:, 2], q[::-1, 0]]),
            fill="toself", fillcolor=COLORS[f], opacity=0.15, line=dict(width=0),
            legendgroup=f, showlegend=False, hoverinfo="skip"), row=1, col=3)
        fig.add_trace(go.Scatter(
            x=centers, y=q[:, 1], mode="lines", line=dict(color=COLORS[f], width=2),
            legendgroup=f, showlegend=False,
            hovertemplate=f"{f}<br>progress %{{x:.0%}}<br>median %{{y:.2f}} cm<extra></extra>",
        ), row=1, col=3)
    fig.update_xaxes(tickformat=".0%", title=dict(text="episode progress", font=dict(color=D.MUTED, size=10)),
                     row=1, col=3)
    fig.update_yaxes(title=dict(text="cm (median, IQR band)", font=dict(color=D.MUTED, size=10)),
                     rangemode="tozero", row=1, col=3)

    panels = [("cmd_pos", 100, "cm", 2, 1), ("cmd_rot", 1, "rad", 2, 2), ("step_pos", 100, "cm", 2, 3),
              ("step_rot", 1, "rad", 3, 1), ("accel", 1000, "mm", 3, 2), ("length", 1, "steps", 3, 3)]
    for key, scale, unit, row, col in panels:
        for f in fams:
            _ecdf(fig, metrics[f][key], f, scale, unit, row, col, False)
        if key != "length":
            clip_x(key, scale, row, col)
        fig.update_xaxes(title=dict(text=unit, font=dict(color=D.MUTED, size=10)), row=row, col=col)
    for row, col in [(1, 1)] + [(r, c) for _, _, _, r, c in panels]:
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02], row=row, col=col)
    fig.update_xaxes(title=dict(text="cm", font=dict(color=D.MUTED, size=10)), row=1, col=1)

    # The table carries every number the colours do (aqua is below 3:1 on the surface).
    s = summary["families"]
    cols = [f for f in fams]
    rows = [
        ("episodes (successes)", lambda f: f"{s[f]['episodes']}"
         + ("" if s[f]["successes"] is None else f" ({s[f]['successes']})")),
        ("median length (steps)", lambda f: f"{s[f]['length_median']:.0f}"),
        ("nearest demo point: median / p95", lambda f: f"{s[f]['nn_m']['median'] * 100:.2f} / "
                                                       f"{s[f]['nn_m']['p95'] * 100:.2f} cm"),
        ("points within 2 cm / 5 cm of a demo", lambda f: f"{s[f]['nn_within']['2cm'] * 100:.0f}% / "
                                                          f"{s[f]['nn_within']['5cm'] * 100:.0f}%"),
        ("DTW to closest demo: median (× demo floor)",
         lambda f: f"{s[f]['dtw_m']['median'] * 100:.2f} cm ({s[f]['dtw_median_over_demo_floor']:.2f}×)"),
        ("commanded ‖Δpos‖ median / p95", lambda f: f"{s[f]['cmd_pos_m']['median'] * 100:.2f} / "
                                                    f"{s[f]['cmd_pos_m']['p95'] * 100:.2f} cm"),
        ("commanded |Δrot| median / p95", lambda f: f"{s[f]['cmd_rot_rad']['median']:.3f} / "
                                                    f"{s[f]['cmd_rot_rad']['p95']:.3f} rad"),
        ("realised step median / p95", lambda f: f"{s[f]['step_pos_m']['median'] * 100:.2f} / "
                                                 f"{s[f]['step_pos_m']['p95'] * 100:.2f} cm"),
        ("jitter ‖2nd diff‖ median / p95", lambda f: f"{s[f]['accel_m']['median'] * 1000:.2f} / "
                                                     f"{s[f]['accel_m']['p95'] * 1000:.2f} mm"),
        ("gripper closes per episode: mean / max",
         lambda f: f"{s[f]['gripper_closes_per_episode']['mean']:.1f} / {s[f]['gripper_closes_per_episode']['max']}"),
        ("first grasp centroid − demos", lambda f: "—" if f == DEMOS else
         str(s[f]["event_offset_from_demos_m"]["grasp"])),
        ("final release centroid − demos", lambda f: "—" if f == DEMOS else
         str(s[f]["event_offset_from_demos_m"]["release"])),
    ]
    fig.add_trace(go.Table(
        header=dict(values=["<b>metric</b>"] + [f"<b>{f}</b>" for f in cols],
                    fill_color=D.SURFACE, line_color=D.GRID, align="left",
                    font=dict(color=[D.INK] + [COLORS[f] for f in cols], size=12)),
        cells=dict(values=[[r[0] for r in rows]] + [[r[1](f) for r in rows] for f in cols],
                   fill_color=D.SURFACE, line_color=D.GRID, align="left",
                   font=dict(color=D.INK, size=11), height=24),
        columnwidth=[2.2] + [1.6] * len(cols),
    ), row=4, col=1)

    fig.update_xaxes(gridcolor=D.GRID, tickfont=dict(color=D.MUTED, size=9), zeroline=False)
    fig.update_yaxes(gridcolor=D.GRID, tickfont=dict(color=D.MUTED, size=9), zeroline=False)
    fig.update_layout(
        title=dict(text=title, font=dict(color=D.INK, size=15)),
        paper_bgcolor=D.SURFACE, plot_bgcolor=D.SURFACE,
        legend=dict(font=dict(color=D.INK, size=11), bgcolor=D.SURFACE, bordercolor=D.GRID,
                    borderwidth=1, orientation="h", x=1, xanchor="right", y=1.045, yanchor="bottom"),
        height=1780, margin=dict(l=55, r=20, t=120, b=30),
    )
    for a in fig.layout.annotations:
        a.font.update(color=D.INK, size=12)
    return fig


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=D.DEFAULT_REPO, help="the demos the policies trained on")
    p.add_argument("--outputs-root", default=None,
                   help="default: ~/franka_data/outputs/<repo-id>")
    p.add_argument("--runs", nargs="*", default=None,
                   help="run directory names to include (default: every multifast run)")
    p.add_argument("--successful-only", action="store_true",
                   help="drop episodes whose verdict is not success")
    p.add_argument("--out-dir", default=None,
                   help="default: ~/franka_data/analysis/<dataset>_rollouts/")
    args = p.parse_args()

    root = os.path.expanduser(args.outputs_root or f"~/franka_data/outputs/{args.repo_id}")
    out = Path(os.path.expanduser(args.out_dir or
               f"~/franka_data/analysis/{args.repo_id.split('/')[-1]}_rollouts"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"[demos] {args.repo_id}")
    groups = {DEMOS: D.load_real(args.repo_id), BASE: [], RESIDUAL: []}
    runs_info = []
    for run_dir, manifest in discover_runs(root, args.runs):
        if manifest["train_dataset"]["repo_id"] != args.repo_id:
            print(f"[runs] {os.path.basename(run_dir)}: trained on "
                  f"{manifest['train_dataset']['repo_id']}, skipped")
            continue
        eps, info = load_rollouts(run_dir, manifest)
        if args.successful_only:
            eps = [e for e in eps if e.get("success")]
            info["episodes"] = len(eps)
        groups[info["family"]] += eps
        runs_info.append(info)
    if not groups[BASE] and not groups[RESIDUAL]:
        raise SystemExit("no rollout episodes left to compare")

    print(f"[deviation] DTW over {len(groups[BASE]) + len(groups[RESIDUAL])} rollouts and "
          f"{len(groups[DEMOS])} demos (leave-one-out)")
    attach_deviation(groups[DEMOS], groups[BASE] + groups[RESIDUAL])
    metrics = family_metrics(groups)
    summary = summarise(groups, metrics, runs_info)
    print_summary(summary)

    label = f"rollouts vs real demos · {args.repo_id}"
    build_eef_figure(groups, summary, f"EE paths · {label}").write_html(
        out / "rollout_eef_aggregate.html", include_plotlyjs="cdn")
    build_deviation_figure(groups, metrics, summary,
                           f"Deviation and per-step deltas · {label}<br><sup>dot on each curve = "
                           f"median · 'demos' rows score each demo against the other "
                           f"{len(groups[DEMOS]) - 1} (leave-one-out)</sup>").write_html(
        out / "rollout_deviation_aggregate.html", include_plotlyjs="cdn")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'rollout_eef_aggregate.html'}\nwrote {out / 'rollout_deviation_aggregate.html'}"
          f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
