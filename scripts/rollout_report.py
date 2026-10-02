#!/usr/bin/env python3
"""PNG summary sheets for one or more rollout sweeps.

    python scripts/rollout_report.py --sweep bsp_2x
    python scripts/rollout_report.py --sweep sail_100_20 bsp_1x bsp_2x bsp_4x bsp_8x
    python scripts/rollout_report.py --sweep latest --out-dir ~/franka_ws/report

One sheet per method and suite in each sweep -- `<sweep>.png`, with `.<method>`
and `.<suite>` appended when one sweep ran several -- plus a comparison sheet per
suite holding more than one. Each sheet lists the OSC gains and step limit the
runs recorded. Runs that recorded wrist force (force_profiles.npz) add its
statistics and every episode's |F| profile overlaid. Reads only the manifests,
episodes.jsonl and force profiles that `baselines.libero_bridge.evaluate` wrote
(baselines/ROLLOUT.md); nothing here touches the robot or the GPU.

Every sheet is 16:9, 3200 x 1800 px, with no text under 9 pt.

Colour is the data-viz skill's validated palette. Success/timeout are the
categorical blue/orange pair, not green/red: green vs red measures deutan
delta-E 4.1, far under the 8 the pair needs to stay separable. On the comparison
sheet each method keeps one hue in every bar chart, and several sweeps of one
method are shades of it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.run_record import (  # noqa: E402
    DEFAULT_ROOT, force_summary, osc_settings, osc_text,
)

# Light-surface roles from the data-viz reference palette.
SURFACE = "#fcfcfb"
PANEL = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OK = "#2a78d6"      # categorical slot 1, "success"
BAD = "#eb6834"     # categorical slot 2, "timeout"
# The categorical order, for one colour per task. Validated on adjacent pairs.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# Comparison-sheet hue per method: none is success blue or timeout orange, and
# every bar carries its value, so the sub-3:1 green and amber are labelled.
METHOD_HUES = {"bspline": "#4a3aa7", "sail": "#eda100", "pi05": "#1baf7a"}
SPARE_HUES = ("#e87ba4", "#008300", "#e34948")

# Sheet geometry: 20 x 11.25 in at 160 dpi is 16:9. Layout is in inches from
# the top-left corner; EDGE is the outer margin.
SHEET_IN = (20.0, 11.25)
SHEET_DPI = 160
EDGE = 0.45
# Text sizes, pt.
FS_SHEET, FS_SHEET_SUB = 22.0, 11.5
FS_TITLE, FS_SUB = 13.5, 10.5
FS_TEXT = 10.5      # ticks, axis labels, legends
FS_TASK = 10.0      # task names down a y axis
FS_VALUE = 11.5     # a number written on a bar or a cell
FS_SMALL = 9.5      # dense lists: the run parameters, the force key
TASK_WRAP = 52      # characters on one line of a task name

# Which knob each backend is actually steered by, in the order to show it.
KNOBS = {
    "bspline": ("speed_up_times", "origin_time_scale", "degree", "n_obs_steps",
                "obs_stride", "predict_before_end", "disable_time_align"),
    "sail": ("exec_fps", "fast_fps", "slow_fps", "inf_delay", "execute_n_actions",
             "action_horizon", "slowdown_window_size", "precision_modulation", "eag"),
    "pi05": ("chunk_size",),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(root: Path, sweep: str) -> list[dict]:
    """Every run tagged `sweep`, with its episodes attached, task order preserved."""
    out = []
    for manifest in sorted(root.rglob("manifest.json")):
        try:
            doc = json.loads(manifest.read_text())
        except Exception as exc:
            print(f"  ! unreadable {manifest}: {exc}", file=sys.stderr)
            continue
        if (doc.get("run") or {}).get("sweep") != sweep:
            continue
        eps_path = manifest.parent / "episodes.jsonl"
        doc["_episodes"] = [json.loads(ln) for ln in eps_path.read_text().splitlines()
                            if ln.strip()] if eps_path.is_file() else []
        doc["_dir"] = manifest.parent
        doc["_forces"] = load_forces(manifest.parent)
        out.append(doc)
    return out


def load_forces(run_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Episode index -> (seconds, per-step |F| in N); empty for a run that recorded
    none. A sim run's time is steps / control_freq; a real run stores its own."""
    path = run_dir / "force_profiles.npz"
    if not path.is_file():
        return {}
    out = {}
    with np.load(path) as z:
        freq = float(z["control_freq"]) if "control_freq" in z.files else None
        for k in z.files:
            if not k.startswith("ee_force_"):
                continue
            ep = k.rsplit("_", 1)[1]
            norm = np.linalg.norm(z[k], axis=-1)
            t = (z[f"time_{ep}"] if f"time_{ep}" in z.files
                 else (np.arange(len(norm)) + 1) / (freq or 20.0))
            out[int(ep)] = (t, norm)
    return out


def resolve_sweep(root: Path, name: str) -> str | None:
    if name != "latest":
        return name
    ids = {s for m in root.rglob("manifest.json")
           if (s := (json.loads(m.read_text()).get("run") or {}).get("sweep"))}
    return max(ids) if ids else None


def short_task(name: str) -> str:
    """`KITCHEN_SCENE9_turn_on_the_stove` -> `KS9 turn on the stove`."""
    for long, tag in (("KITCHEN_SCENE", "KS"), ("LIVING_ROOM_SCENE", "LR"),
                      ("STUDY_SCENE", "SS")):
        if name.startswith(long):
            rest = name[len(long):]
            num, _, tail = rest.partition("_")
            return f"{tag}{num} {tail.replace('_', ' ')}"
    return name.replace("_", " ")


def stats(runs: list[dict]) -> dict:
    """Pool a sweep: counts, rates, and the per-episode series the panels need."""
    eps = [e for r in runs for e in r["_episodes"]]
    ok = [e for e in eps if e.get("success")]
    bad = [e for e in eps if not e.get("success")]
    meds = [r["summary"]["median_time_to_success_s"] for r in runs
            if r["summary"].get("median_time_to_success_s") is not None]
    verdicts: dict[str, int] = {}
    for e in eps:
        v = e.get("verdict") or "unknown"
        verdicts[v] = verdicts.get(v, 0) + 1
    infer = [e["inferences"] for e in eps if e.get("inferences")]
    steps = [e["steps"] for e in eps if e.get("steps")]
    return {
        "runs": runs, "episodes": eps, "ok": ok, "bad": bad,
        "n_tasks": len(runs), "n_eps": len(eps), "n_ok": len(ok),
        "rate": len(ok) / len(eps) if eps else None,
        "mean_median_s": sum(meds) / len(meds) if meds else None,
        "verdicts": verdicts,
        "mean_steps_ok": sum(e["steps"] for e in ok) / len(ok) if ok else None,
        "mean_steps_bad": sum(e["steps"] for e in bad) / len(bad) if bad else None,
        "sim_s": sum(e.get("wall_time_s") or 0 for e in eps),
        "clock_s": sum(e.get("clock_time_s") or 0 for e in eps),
        "plans_per_step": (sum(infer) / sum(steps)) if infer and steps else None,
        "slow_steps": sum(e.get("slow_steps") or 0 for e in eps),
        "force": force_summary(eps),
    }


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def new_sheet() -> plt.Figure:
    return plt.figure(figsize=SHEET_IN, dpi=SHEET_DPI, facecolor=SURFACE)


def rect(fig, x0: float, x1: float, top: float, bottom: float) -> list[float]:
    """Inches from the sheet's top-left corner -> an add_axes rectangle."""
    w, h = fig.get_size_inches()
    return [x0 / w, 1 - bottom / h, (x1 - x0) / w, (bottom - top) / h]


def box(fig, x0: float, x1: float, top: float, bottom: float):
    return fig.add_axes(rect(fig, x0, x1, top, bottom))


def axes_row(fig, x0, x1, top, bottom, ratios, gaps) -> list:
    """Axes side by side from x0 to x1, as wide as `ratios` say and `gaps` inches
    apart: one number for every gap, or a list with one per gap."""
    if isinstance(gaps, (int, float)):
        gaps = [gaps] * (len(ratios) - 1)
    gaps = list(gaps)[:len(ratios) - 1]
    unit = (x1 - x0 - sum(gaps)) / sum(ratios)
    axes, x = [], x0
    for r, g in zip(ratios, gaps + [0.0]):
        axes.append(box(fig, x, x + unit * r, top, bottom))
        x += unit * r + g
    return axes


def text_width(fig, labels, size: float) -> float:
    """The widest line among `labels` at `size` pt, in inches."""
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(size=size)
    lines = [ln for s in labels for ln in s.split("\n")]
    return max((renderer.get_text_width_height_descent(ln, prop, ismath=False)[0]
                for ln in lines), default=0.0) / fig.dpi


def header(fig, heading: str, sub: str) -> None:
    w, h = fig.get_size_inches()
    fig.text(EDGE / w, 1 - 0.3 / h, heading, fontsize=FS_SHEET, fontweight="bold",
             color=INK, va="top")
    fig.text(EDGE / w, 1 - 0.74 / h, sub, fontsize=FS_SHEET_SUB, color=MUTED, va="top")


def task_label(name: str) -> str:
    """short_task for a y axis: one line up to TASK_WRAP characters, else two
    about equal lines, so no word is left on a line of its own."""
    text = short_task(name)
    if len(text) <= TASK_WRAP:
        return text
    width = (len(text) + 1) // 2
    while len(lines := textwrap.wrap(text, width)) > 2:
        width += 2
    return "\n".join(lines)


def fit_names(ax, names: list[str]) -> list[str]:
    """Column names on as few lines as keep each clear of its neighbour: one, then
    split at the first space, then at every space."""
    slot = ax.get_position().width * ax.figure.get_figwidth() / max(len(names), 1)
    for fit in (names, [n.replace(" ", "\n", 1) for n in names],
                [n.replace(" ", "\n") for n in names]):
        if text_width(ax.figure, fit, FS_TEXT) <= 0.9 * slot:
            break
    return fit


def style(ax, *, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=FS_TEXT, length=3, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)


def title(ax, text, sub=None):
    ax.set_title(text, color=INK, fontsize=FS_TITLE, fontweight="bold",
                 loc="left", pad=27 if sub else 8)
    if sub:
        # A fixed 6 pt above the axes, whatever the axes' height.
        ax.text(0, 1, sub, color=MUTED, fontsize=FS_SUB, va="bottom",
                transform=offset_copy(ax.transAxes, fig=ax.figure, y=6, units="points"))


def legend_above(ax, handles=None, ncol=3):
    """Legends go on the title line: inside the axes they land on the data."""
    kw = dict(frameon=False, fontsize=FS_TEXT, labelcolor=INK2, handlelength=1.1,
              ncol=ncol, loc="lower right", bbox_to_anchor=(1.0, 1.005),
              columnspacing=1.4, handletextpad=0.5)
    return ax.legend(handles=handles, **kw) if handles else ax.legend(**kw)


def outcome_key() -> list:
    """success/timeout fills and the median/mean marks: one legend per sweep sheet."""
    return [Patch(facecolor=OK, label="success"),
            Patch(facecolor=BAD, label="timeout"),
            Line2D([], [], color=INK, lw=2, label="median"),
            Line2D([], [], color=INK, marker="D", markerfacecolor=SURFACE,
                   markersize=7, lw=0, label="mean")]


def tile(fig, rect, value, label, note=None, color=INK):
    """A stat tile: the number leads, the label explains it."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.8)
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=26,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.28, label.upper(), ha="center", va="center", fontsize=FS_SMALL,
            color=INK2, transform=ax.transAxes)
    if note:
        ax.text(0.5, 0.1, note, ha="center", va="center", fontsize=FS_SMALL - 0.5,
                color=MUTED, transform=ax.transAxes)
    return ax


def task_rows(ax, runs) -> None:
    """Task names down the y axis, first task on top, in the rows dot_panel uses."""
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([task_label(r["environment"]["task"]) for r in runs][::-1],
                       fontsize=FS_TASK, color=INK2, linespacing=1.05)
    ax.set_ylim(-0.7, len(runs) - 0.2)


def outcome_panel(ax, runs):
    """Per task: successes and timeouts as one stacked bar, counts labelled."""
    oks = [sum(1 for e in r["_episodes"] if e.get("success")) for r in runs][::-1]
    tot = [len(r["_episodes"]) for r in runs][::-1]
    bads = [t - o for t, o in zip(tot, oks)]
    y = range(len(runs))
    ax.barh(y, oks, color=OK, height=0.62, zorder=3, label="success")
    # 2px surface gap between the two fills, per the mark spec.
    ax.barh(y, bads, left=[o + 0.12 for o in oks], color=BAD, height=0.62,
            zorder=3, label="timeout")
    for i, (o, t) in enumerate(zip(oks, tot)):
        ax.text(t + max(tot) * 0.03, i, f"{100 * o / t:.0f}%", va="center",
                fontsize=FS_VALUE, color=INK, fontweight="bold")
        if t - o:
            ax.text(o + (t - o) / 2, i, f"{t - o}", va="center", ha="center",
                    fontsize=FS_TEXT, color="white", fontweight="bold", zorder=4)
    task_rows(ax, runs)
    ax.set_xlim(0, max(tot) * 1.22)
    ax.set_xlabel("episodes", fontsize=FS_TEXT, color=MUTED)
    style(ax)


def dot_panel(ax, runs, value, marks, columns, xlabel, spec):
    """One row per task: every episode as a dot, the task's median as a tick and
    mean as a diamond, and two numbers per task at the right.

    `value(episode)` places a dot (None skips it), `marks(run)` gives the
    (mean, median) to draw, and `columns` is two (header, run -> number) pairs.
    """
    rows = runs[::-1]
    allv = [v for r in runs for e in r["_episodes"] if (v := value(e)) is not None]
    lo, hi = (min(allv), max(allv)) if allv else (0.0, 1.0)
    pad = (hi - lo) * 0.04 or 1.0
    lo = max(0.0, lo - pad)
    # The dots keep the left 68%; the two number columns share the rest.
    ax.set_xlim(lo, lo + (hi + pad - lo) / 0.68)
    side = ax.get_yaxis_transform()
    for i, r in enumerate(rows):
        for e in r["_episodes"]:
            t = value(e)
            if t is None:
                continue
            good = e.get("success")
            ax.scatter(t, i + (hash(e["episode"]) % 7 - 3) * 0.035,
                       s=34, color=OK if good else BAD, alpha=0.75,
                       edgecolors=SURFACE, linewidths=0.8, zorder=3)
        mean, med = marks(r)
        if med is not None:
            ax.plot([med, med], [i - 0.33, i + 0.33], color=INK, lw=2.2,
                    zorder=5, solid_capstyle="butt")
        if mean is not None:
            ax.scatter([mean], [i], marker="D", s=36, facecolor=SURFACE,
                       edgecolors=INK, linewidths=1.6, zorder=6)
        for x, (_, col), color, weight in ((0.86, columns[0], INK, "bold"),
                                           (1.0, columns[1], MUTED, "normal")):
            v = col(r)
            ax.text(x, i, format(v, spec) if v is not None else "-", transform=side,
                    va="center", ha="right", fontsize=FS_TEXT, color=color,
                    fontweight=weight, fontfamily="monospace")
    for x, (head, _), color, weight in ((0.86, columns[0], INK, "bold"),
                                        (1.0, columns[1], MUTED, "normal")):
        ax.text(x, len(rows) - 0.45, head, transform=side, va="bottom", ha="right",
                fontsize=FS_SMALL, color=color, fontweight=weight)
    # Ticks only under the dots, not under the number columns.
    ax.set_xticks([t for t in ax.get_xticks() if lo <= t <= hi + pad])
    task_rows(ax, runs)
    ax.set_xlabel(xlabel, fontsize=FS_TEXT, color=MUTED)
    style(ax)


def duration_panel(ax, runs):
    """Every episode as a dot, with each task's mean and median beside it."""
    def mean(r):
        return r["summary"].get("mean_time_to_success_s")

    def med(r):
        return r["summary"].get("median_time_to_success_s")

    dot_panel(ax, runs, lambda e: e.get("wall_time_s") or 0,
              lambda r: (mean(r), med(r)), (("mean", mean), ("med", med)),
              "episode duration, simulated s", ".2f")


def episode_force(e: dict, key: str) -> float | None:
    return (e.get("ee_force_n") or {}).get(key)


def force_panel(ax, runs):
    """Each episode's p95 wrist force as a dot; per task, the mean and median of
    those, and the run's eval_fast-style p95 and mean |F| at the right."""
    def p95s(r):
        return [v for e in r["_episodes"] if (v := episode_force(e, "p95")) is not None]

    def marks(r):
        v = p95s(r)
        return (float(np.mean(v)), float(np.median(v))) if v else (None, None)

    def run_stat(key):
        return lambda r: (force_summary(r["_episodes"]) or {}).get(key)

    dot_panel(ax, runs, lambda e: episode_force(e, "p95"), marks,
              (("p95", run_stat("p95")), ("mean", run_stat("mean"))),
              "each episode's p95 |F|, N", ".1f")


def task_styles(runs) -> dict[str, tuple[str, str]]:
    """Task -> (colour, linestyle), in sheet order: the eight hues, then the same
    hues dashed for tasks 9-16, then grey."""
    out: dict[str, tuple[str, str]] = {}
    for r in runs:
        t = r["environment"]["task"]
        if t not in out:
            i = len(out)
            out[t] = ((SLOTS[i % len(SLOTS)], "-" if i < len(SLOTS) else "--")
                      if i < 2 * len(SLOTS) else (MUTED, "-"))
    return out


def force_overlay(ax, runs, styles) -> None:
    """Every episode's |F| profile on one axis: colour is the task, shade the
    episode (first lightest), so a task's spread reads as one hue's range."""
    tasks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for r in runs:
        tasks.setdefault(r["environment"]["task"], []).extend(
            r["_forces"][k] for k in sorted(r["_forces"]))
    for task, series in tasks.items():
        base, ls = styles[task]
        n = len(series)
        for j, (t, y) in enumerate(series):
            if not len(y):
                continue
            shade = 0.6 * (n - 1 - j) / max(n - 1, 1)
            ax.plot(t, y, lw=0.9, ls=ls, color=mix(base, SURFACE, shade), zorder=3,
                    solid_joinstyle="round")
    # Log: the gripper's ~5 N at rest, contact and impact spikes span 1 N to kN.
    ax.set_yscale("log")
    peak = max((float(y.max()) for ys in tasks.values() for _, y in ys if len(y)),
               default=10.0)
    ax.set_ylim(1.0, peak * 1.4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, None)
    ax.set_xlabel("simulated seconds since the first action", fontsize=FS_TEXT, color=MUTED)
    ax.set_ylabel("|F|, N (log scale)", fontsize=FS_TEXT, color=MUTED)
    style(ax, grid_axis="y")


def force_key(ax, runs, styles) -> None:
    """The overlay's legend, with each task's force numbers beside its swatch."""
    ax.set_facecolor(PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID); s.set_linewidth(0.8)
    episodes: dict[str, list[dict]] = {}
    for r in runs:
        episodes.setdefault(r["environment"]["task"], []).extend(r["_episodes"])
    ax.text(0.04, 0.955, "TASKS", transform=ax.transAxes, fontsize=FS_SMALL,
            color=INK, fontweight="bold", va="top")
    for x, head in ((0.80, "p95"), (0.97, "peak")):
        ax.text(x, 0.955, head, transform=ax.transAxes, fontsize=FS_SMALL, color=MUTED,
                va="top", ha="right")
    step = min(0.085, 0.84 / max(len(styles), 1))
    for i, (task, (color, ls)) in enumerate(styles.items()):
        y = 0.84 - i * step
        ax.plot([0.04, 0.09], [y, y], color=color, ls=ls, lw=2.4,
                transform=ax.transAxes, solid_capstyle="butt")
        name = short_task(task)
        ax.text(0.11, y, name if len(name) <= 34 else name[:33] + "…",
                transform=ax.transAxes, fontsize=FS_SMALL, color=INK2, va="center")
        f = force_summary(episodes[task]) or {}
        for x, key in ((0.80, "p95"), (0.97, "peak")):
            ax.text(x, y, f"{f[key]:.0f} N" if key in f else "-", transform=ax.transAxes,
                    fontsize=FS_SMALL, color=INK, va="center", ha="right",
                    fontfamily="monospace")


def mix(color: str, toward: str, frac: float) -> tuple:
    """`color` moved `frac` of the way to `toward`, in RGB."""
    a, b = np.array(to_rgb(color)), np.array(to_rgb(toward))
    return tuple(a + (b - a) * frac)


def params_panel(ax, runs, sweep):
    """What this sweep actually ran, straight out of the manifests."""
    ax.set_facecolor(PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID); s.set_linewidth(0.8)
    r0 = runs[0]
    env, pol = r0["environment"], r0["policy"]
    params = r0.get("parameters") or {}
    method = r0["run"]["method"]
    ckpt = (pol.get("checkpoint") or {}) or {}
    server = pol.get("server") or {}

    rows: list[tuple[str, str]] = [
        ("sweep id", sweep),
        ("method", method),
        ("tasks x episodes", f"{len(runs)} x {len(r0['_episodes'])}"),
        ("simulator", f"{env.get('simulator', 'libero/robosuite')}, plant {env.get('plant')}"),
        ("control rate", f"{env.get('control_freq')} Hz"),
        ("step budget", f"{env.get('max_steps')} ({env.get('max_steps', 0) / (env.get('control_freq') or 20):.0f} s sim)"),
        ("settle steps", str(env.get("settle_steps"))),
        ("render resolution", str(env.get("render_resolution"))),
        ("init states", f"suite's own {env.get('init_states')}"),
        ("control mode", str(pol.get("control_mode"))),
    ]
    osc = osc_settings(env)
    if osc is None:
        rows.append(("OSC", "not recorded"))
    else:
        rows += [("OSC kp", osc["kp"]), ("OSC damping ratio", osc["damping_ratio"]),
                 ("OSC step limit", f"{osc['step_limit_cm']:g} cm per env step")]
    for key in KNOBS.get(method, ()):
        if key in params and params[key] is not None:
            rows.append((key, str(params[key])))
    if ckpt.get("sha256"):
        rows.append(("checkpoint sha", str(ckpt["sha256"])[:16]))
    if server.get("task_instruction"):
        # pi0.5 is language-conditioned; the prompt it was given is a run parameter.
        rows.append(("prompt", f'"{server["task_instruction"]}"'))
    if server.get("train_dataset"):
        rows.append(("trained on", str(server["train_dataset"]).split("/")[0] + "/<task>"))
    git = (env.get("git") or {})
    if git.get("commit"):
        rows.append(("git", f"{str(git['commit'])[:9]}{' dirty' if git.get('dirty') else ''}"))
    rows.append(("started", str(r0["run"].get("started_at", ""))[:19].replace("T", " ")))

    ax.text(0.04, 0.965, "RUN PARAMETERS", transform=ax.transAxes, fontsize=FS_SMALL,
            color=INK, fontweight="bold", va="top")
    width = text_width(ax.figure, ["0"], FS_SMALL) * 1.2  # a monospace character
    room = ax.get_position().width * ax.figure.get_figwidth() * 0.93
    step = 0.9 / max(len(rows), 1)
    for i, (k, v) in enumerate(rows):
        y = 0.905 - i * step
        v = str(v)
        # A value that would run into its own key is cut, not overprinted.
        fit = int((room - text_width(ax.figure, [k], FS_SMALL) - 0.2) / width)
        if len(v) > fit:
            v = v[:max(fit - 1, 1)] + "…"
        ax.text(0.04, y, k, transform=ax.transAxes, fontsize=FS_SMALL, color=MUTED, va="top")
        ax.text(0.97, y, v, transform=ax.transAxes, fontsize=FS_SMALL, color=INK,
                va="top", ha="right", fontfamily="monospace")


def outcome_hist(ax, ok: list, bad: list, xlabel: str) -> None:
    """One per-episode number as a histogram, successes stacked on timeouts."""
    allv = ok + bad
    if not allv:
        return
    # Shared edges and a stacked draw: separate hist() calls bin independently,
    # which hides a small series inside a large one's range.
    lo, hi = min(allv), max(allv)
    if hi == lo:
        # One distinct value gives zero-width bins, which draw nothing.
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.linspace(lo, hi, 22)
    ax.hist([ok, bad], bins=bins, stacked=True, color=[OK, BAD],
            label=["success", "timeout"], zorder=3)
    ax.set_xlabel(xlabel, fontsize=FS_TEXT, color=MUTED)
    ax.set_ylabel("episodes", fontsize=FS_TEXT, color=MUTED)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    style(ax, grid_axis="y")


def length_panel(ax, st):
    """Episode length in steps, successes against timeouts."""
    outcome_hist(ax, [e["steps"] for e in st["ok"] if e.get("steps")],
                 [e["steps"] for e in st["bad"] if e.get("steps")],
                 "episode length, env steps")


def peak_force_panel(ax, st):
    """Each episode's largest |F|, successes against timeouts."""
    outcome_hist(ax, [v for e in st["ok"] if (v := episode_force(e, "max")) is not None],
                 [v for e in st["bad"] if (v := episode_force(e, "max")) is not None],
                 "largest |F| in the episode, N")


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def sweep_sheet(key: str, runs: list[dict], out: Path) -> Path:
    st = stats(runs)
    sweep = runs[0]["run"].get("sweep") or key
    has_force = any(r["_forces"] for r in runs)
    fig = new_sheet()
    w, h = SHEET_IN
    method = runs[0]["run"]["method"]
    knob = (runs[0].get("parameters") or {}).get("speed_up_times")
    label = f"{method}" + (f"  {knob}x" if knob not in (None, 1.0) else "")
    oscs = {osc_text(r["environment"]) for r in runs}
    osc = oscs.pop() if len(oscs) == 1 else "mixed"
    header(fig, f"{label} on {runs[0]['environment'].get('suite', 'libero_90')}",
           f"sweep {sweep}   ·   {st['n_tasks']} tasks   ·   {st['n_eps']} episodes   ·   "
           f"stock robosuite plant   ·   OSC {osc or 'not recorded'}   ·   "
           f"time in simulated seconds")
    # One key for every panel: they all draw success/timeout and median/mean alike.
    fig.legend(handles=outcome_key(), loc="upper right", ncol=4, frameon=False,
               bbox_to_anchor=((w - EDGE) / w, 1 - 0.3 / h), fontsize=FS_TEXT,
               labelcolor=INK2, handlelength=1.3, columnspacing=1.6, handletextpad=0.5)

    n_bad = st["n_eps"] - st["n_ok"]
    tiles = [
        (f"{100 * st['rate']:.1f}%" if st["rate"] is not None else "-",
         "pooled success", f"{st['n_ok']} of {st['n_eps']}", OK),
        (f"{st['mean_median_s']:.2f} s" if st["mean_median_s"] else "-",
         "mean median time", "over successes", INK),
        (str(n_bad), "failures", ", ".join(f"{k} {v}" for k, v in st["verdicts"].items()
                                          if k != "success") or "none", BAD if n_bad else INK),
        (f"{st['mean_steps_ok']:.0f}" if st["mean_steps_ok"] else "-",
         "mean steps, success",
         f"{st['mean_steps_bad']:.0f} on failure" if st["mean_steps_bad"] else "", INK),
        (f"{st['plans_per_step']:.3f}" if st["plans_per_step"] else "-",
         "inferences / step", "all episodes", INK),
        (f"{st['clock_s'] / 60:.0f} min", "wall clock",
         f"{st['sim_s'] / 60:.0f} min simulated", INK),
    ]
    f = st["force"]
    tiles.append((f"{f['mean']:.1f} N", "mean wrist |F|",
                  f"episode p95 {f['p95']:.0f} · max {f['max']:.0f} N", INK)
                 if f else ("-", "mean wrist |F|", "not recorded", MUTED))
    gap = 0.25
    tw = (w - 2 * EDGE - gap * (len(tiles) - 1)) / len(tiles)
    for i, (val, lab, note, col) in enumerate(tiles):
        x = EDGE + i * (tw + gap)
        tile(fig, rect(fig, x, x + tw, 1.08, 2.18), val, lab, note, col)

    # Top row: the per-task panels side by side, sharing one column of task names.
    # Bottom row: the histograms and the force overlay. Right: parameters and key.
    names = [task_label(r["environment"]["task"]) for r in runs]
    x_tasks = EDGE + min(text_width(fig, names, FS_TASK), 5.0) + 0.15
    split, right = 14.5, 15.0
    upper, lower = (2.95, 6.75), (7.9, 10.72)
    top = axes_row(fig, x_tasks, split, *upper,
                   [1.0, 1.3, 1.3] if has_force else [1.0, 1.3], 0.3)
    outcome_panel(top[0], runs)
    title(top[0], "Outcome per task", "timeouts counted inside each bar")
    duration_panel(top[1], runs)
    title(top[1], "Episode duration", "one dot per episode; mean and median at right")
    if has_force:
        force_panel(top[2], runs)
        title(top[2], "Wrist force per episode", "one dot per episode at its p95 |F|")
    for ax in top[1:]:
        ax.tick_params(axis="y", labelleft=False)

    params_panel(box(fig, right, w - EDGE, upper[0] - 0.58,
                     7.17 if has_force else lower[1]), runs, sweep)

    bottom = axes_row(fig, EDGE + 0.62, split, *lower,
                      [1.0, 1.0, 1.55] if has_force else [1.0], 0.8)
    length_panel(bottom[0], st)
    title(bottom[0], "Episode length", "timeouts pile up at the step budget")
    if has_force:
        peak_force_panel(bottom[1], st)
        title(bottom[1], "Peak wrist force", "each episode's largest |F|")
        styles = task_styles(runs)
        force_overlay(bottom[2], runs, styles)
        title(bottom[2], "Every force profile",
              "|F| each env step; colour is the task, earliest episode lightest")
        ax = box(fig, right, w - EDGE, *lower)
        force_key(ax, runs, styles)
        title(ax, "Key", "p95: episode p95 |F|, averaged; peak: largest")

    path = out / f"{key}.png"
    # No tight bbox: it would crop the sheet off 16:9.
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def suite_of(run: dict) -> str:
    return run["environment"].get("suite", "libero_90")


def by_method(sweep: str, runs: list[dict]) -> dict[str, list[dict]]:
    """One group per method and suite: a sheet pools its runs, so mixing methods
    would count each task once per method and label them all with the first, and
    mixing suites would title one suite's sheet with the other's name."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["run"]["method"], suite_of(r)), []).append(r)
    methods = {m for m, _ in groups}
    suites = {u for _, u in groups}
    return {".".join([sweep] + ([m] if len(methods) > 1 else [])
                     + ([u] if len(suites) > 1 else [])): rs
            for (m, u), rs in groups.items()}


def column_labels(sweeps: dict[str, list[dict]]) -> list[str]:
    """`sail`, `bspline 2x`, ...: sweep ids are too long to sit six abreast. Two
    sweeps of one method that would read the same are told apart by their OSC
    gains, and by their ids when those match too."""
    labels = []
    for runs in sweeps.values():
        method = runs[0]["run"]["method"]
        speed = (runs[0].get("parameters") or {}).get("speed_up_times")
        labels.append(f"{method} {speed:g}x" if speed is not None else method)
    for i, runs in enumerate(sweeps.values()):
        osc = osc_settings(runs[0]["environment"])
        if osc and [lb.split(" kp")[0] for lb in labels].count(labels[i]) > 1:
            labels[i] += f" kp{osc['kp']} ζ{osc['damping_ratio']}"
    return labels if len(set(labels)) == len(labels) else list(sweeps)


def osc_kp(runs: list[dict]) -> float:
    """The sweep's OSC kp, 0 when not recorded: what orders a method's gain sweeps."""
    osc = osc_settings(runs[0]["environment"])
    return float(osc["kp"].split("/")[0]) if osc else 0.0


def column_colors(sweeps: dict[str, list[dict]]) -> list[tuple]:
    """One hue per method, so a method reads the same in every panel. A method's
    sweeps are shades of its hue, lightest first: by speed-up, then kp, else sheet
    order."""
    methods = [runs[0]["run"]["method"] for runs in sweeps.values()]
    speeds = [(runs[0].get("parameters") or {}).get("speed_up_times")
              for runs in sweeps.values()]
    kps = [osc_kp(runs) for runs in sweeps.values()]
    spare = iter(SPARE_HUES)
    hue = {m: METHOD_HUES.get(m) or next(spare, MUTED) for m in dict.fromkeys(methods)}
    colors: list[tuple] = [()] * len(methods)
    for m in hue:
        idx = sorted((i for i, x in enumerate(methods) if x == m),
                     key=lambda i: (speeds[i] or 0.0, kps[i], i))
        for rank, i in enumerate(idx):
            t = rank / (len(idx) - 1) if len(idx) > 1 else 0.5
            colors[i] = (mix(hue[m], SURFACE, 0.9 * (0.5 - t)) if t < 0.5
                         else mix(hue[m], INK, 0.8 * (t - 0.5)))
    return colors


def bars(ax, vals: list, names: list, colors: list, fmt: str, ylabel: str) -> None:
    """One bar per sweep in its method's colour, its value written above it."""
    x = range(len(vals))
    ax.bar(x, vals, color=colors, width=0.62, zorder=3)
    top = max(vals + [0.0]) or 1.0
    for i, v in enumerate(vals):
        ax.text(i, v + top * 0.025, fmt.format(v), ha="center", va="bottom",
                fontsize=FS_VALUE, color=INK, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(fit_names(ax, names), fontsize=FS_TEXT,
                                               color=INK2)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel(ylabel, fontsize=FS_TEXT, color=MUTED)
    style(ax, grid_axis="y")


def task_grid(ax, sts: dict, keys: list, names: list, cell, color: str,
              light_text_above: float | None = None) -> None:
    """One row per task, one column per sweep. `cell(run)` gives (fill 0-1, text)
    for a cell, or None to leave it as a dash."""
    tasks = [r["environment"]["task"] for r in sts[keys[0]]["runs"]]
    ax.set_xlim(0, len(keys)); ax.set_ylim(0, len(tasks))
    for j, k in enumerate(keys):
        by_task = {r["environment"]["task"]: r for r in sts[k]["runs"]}
        for i, t in enumerate(tasks):
            r = by_task.get(t)
            c = cell(r) if r is not None else None
            y = len(tasks) - 1 - i
            if c is None:
                ax.text(j + 0.5, y + 0.5, "-", ha="center", va="center",
                        fontsize=FS_VALUE, color=MUTED)
                continue
            frac, text = c
            ax.add_patch(plt.Rectangle((j + 0.04, y + 0.08), 0.92, 0.84,
                                       facecolor=color, alpha=0.10 + 0.85 * frac,
                                       edgecolor=SURFACE, linewidth=1.4))
            light = light_text_above is not None and frac > light_text_above
            ax.text(j + 0.5, y + 0.5, text, ha="center", va="center",
                    fontsize=FS_VALUE, fontweight="bold", color="white" if light else INK)
    ax.set_xticks([j + 0.5 for j in range(len(keys))])
    ax.set_xticklabels(fit_names(ax, names), fontsize=FS_TEXT, color=INK2)
    ax.set_yticks([len(tasks) - 0.5 - i for i in range(len(tasks))])
    ax.set_yticklabels([task_label(t) for t in tasks], fontsize=FS_TASK, color=INK2,
                       linespacing=1.05)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)


# mean < p95 < max as the sweep's own hue, lighter, as is, and darker.
FORCE_STATS = (("mean", SURFACE, 0.55), ("p95", None, 0.0), ("max", INK, 0.35))


def force_bars(ax, sts: dict, keys: list, names: list, colors: list) -> None:
    """eval_fast's mean / p95 / max |F| per sweep, each averaged over episodes."""
    width = 0.26
    top = max((sts[k]["force"][s] for k in keys if sts[k]["force"]
               for s, _, _ in FORCE_STATS), default=1.0) or 1.0
    for s_i, (stat, toward, frac) in enumerate(FORCE_STATS):
        for i, k in enumerate(keys):
            f = sts[k]["force"]
            if not f:
                continue
            x = i + (s_i - 1) * (width + 0.02)
            ax.bar(x, f[stat], width=width, zorder=3,
                   color=colors[i] if toward is None else mix(colors[i], toward, frac))
            if stat == "p95":
                ax.text(x, f[stat] + top * 0.02, f"{f[stat]:.0f}", ha="center",
                        va="bottom", fontsize=FS_VALUE, color=INK, fontweight="bold")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(fit_names(ax, names), fontsize=FS_TEXT, color=INK2)
    ax.set_ylim(0, top * 1.16)
    ax.set_ylabel("|F| at the wrist sensor, N", fontsize=FS_TEXT, color=MUTED)
    style(ax, grid_axis="y")
    legend_above(ax, ncol=3, handles=[
        Patch(facecolor=INK2 if toward is None else mix(INK2, toward, frac), label=stat)
        for stat, toward, frac in FORCE_STATS])


def comparison_sheet(sweeps: dict[str, list[dict]], out: Path,
                     name: str = "comparison") -> Path:
    keys = list(sweeps)
    names = column_labels(sweeps)
    colors = column_colors(sweeps)
    sts = {k: stats(v) for k, v in sweeps.items()}
    has_force = any(sts[k]["force"] for k in keys)
    fig = new_sheet()
    w, h = SHEET_IN
    shapes = [f"{sts[k]['n_tasks']}x{len(sts[k]['runs'][0]['_episodes'])}" for k in keys]
    tail = "   ·   stock plant   ·   time in simulated seconds"
    sub = "  ·  ".join(f"{k} ({s})" for k, s in zip(keys, shapes))
    # Too many ids for one line: the column names say which is which.
    if text_width(fig, [sub + tail], FS_SHEET_SUB) > w - 2 * EDGE:
        prefix = os.path.commonprefix(keys)
        sub = (f"{len(keys)} sweeps{f' {prefix}*' if prefix else ''}, tasks x episodes "
               + (f"{shapes[0]} each" if len(set(shapes)) == 1 else ", ".join(shapes)))
    header(fig, "Sweep comparison", sub + tail)

    # Top row: the pooled numbers, one colour per method throughout. Bottom row:
    # per task, two grids sharing one column of task names, then the force bars.
    # A third line of column name, when one may be needed, comes out of the plots.
    spare = 0.2 if max(n.count(" ") for n in names) > 1 else 0.0
    top = axes_row(fig, EDGE + 0.65, w - EDGE, 1.65, 4.85 - spare, [1, 1, 1], 1.0)
    bars(top[0], [100 * sts[k]["rate"] if sts[k]["rate"] is not None else 0 for k in keys],
         names, colors, "{:.1f}%", "pooled success rate, %")
    title(top[0], "Success", "pooled over every task and episode")
    bars(top[1], [sts[k]["mean_median_s"] or 0 for k in keys], names, colors, "{:.2f}",
         "mean median time-to-success, s")
    title(top[1], "Speed", "lower is faster; successes only")
    bars(top[2], [sts[k]["plans_per_step"] or 0 for k in keys], names, colors, "{:.3f}",
         "inferences per env step")
    title(top[2], "Replan rate", "pooled over all episodes; timeouts replan most")

    tasks = [r["environment"]["task"] for r in sts[keys[0]]["runs"]]
    x_tasks = EDGE + min(text_width(fig, [task_label(t) for t in tasks], FS_TASK), 5.0) + 0.15
    lower = axes_row(fig, x_tasks, w - EDGE, 5.85 + spare, 10.7 - spare,
                     [1, 1, 1.2] if has_force else [1], [0.3, 1.0])

    # Per-task success, one row per task, one column per sweep. A grid of numbers
    # rather than eight grouped bars: at five sweeps the bars are unreadable.
    def success_cell(r):
        rate = r["summary"].get("success_rate") or 0
        return rate, f"{100 * rate:.0f}"

    task_grid(lower[0], sts, keys, names, success_cell, OK, light_text_above=0.72)
    title(lower[0], "Success rate per task, %", "darker is better")

    if has_force:
        p95 = {id(r): (force_summary(r["_episodes"]) or {}).get("p95")
               for k in keys for r in sts[k]["runs"]}
        vmax = max((v for v in p95.values() if v is not None), default=1.0) or 1.0

        def force_cell(r):
            v = p95[id(r)]
            return None if v is None else (v / vmax, f"{v:.0f}")

        task_grid(lower[1], sts, keys, names, force_cell, BAD, light_text_above=0.72)
        lower[1].tick_params(axis="y", labelleft=False)
        title(lower[1], "Wrist force p95 per task, N", "darker presses harder")

        force_bars(lower[2], sts, keys, names, colors)
        title(lower[2], "Wrist force", "per-episode stats, averaged")

    path = out / f"{name}.png"
    # No tight bbox: it would crop the sheet off 16:9.
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", required=True,
                   help="one or more sweep ids, or 'latest'")
    p.add_argument("--root", default=None, help=f"default {DEFAULT_ROOT}")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default <root>/reports")
    p.add_argument("--no-comparison", action="store_true",
                   help="skip the cross-sweep sheet")
    args = p.parse_args()

    root = Path(args.root or DEFAULT_ROOT).expanduser()
    out = (args.out_dir or root / "reports").expanduser()
    out.mkdir(parents=True, exist_ok=True)

    found: dict[str, list[dict]] = {}
    for name in args.sweep:
        sweep = resolve_sweep(root, name)
        if sweep is None:
            print(f"no sweep-tagged runs under {root}", file=sys.stderr)
            return 1
        runs = load_runs(root, sweep)
        if not runs:
            print(f"no runs tagged {sweep!r}", file=sys.stderr)
            continue
        found.update(by_method(sweep, runs))

    if not found:
        return 1
    for key, runs in found.items():
        path = sweep_sheet(key, runs, out)
        st = stats(runs)
        print(f"{path}  ({st['n_tasks']} tasks, {st['n_eps']} episodes, "
              f"{100 * st['rate']:.1f}% success)")
    # Compared within a suite only: the per-task grid needs the same tasks.
    by_suite: dict[str, dict[str, list[dict]]] = {}
    for key, runs in found.items():
        by_suite.setdefault(suite_of(runs[0]), {})[key] = runs
    for suite, sheets in by_suite.items():
        if len(sheets) > 1 and not args.no_comparison:
            name = "comparison" if len(by_suite) == 1 else f"comparison.{suite}"
            print(f"{comparison_sheet(sheets, out, name)}  ({len(sheets)} sheets compared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
