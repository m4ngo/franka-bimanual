#!/usr/bin/env python3
"""One LeRobot recording -> the three policies the comparison needs, from one yaml.

    python scripts/train_pipeline.py start   pipelines/<name>.yaml
    python scripts/train_pipeline.py status  [<run-dir>]      default: the newest run
    python scripts/train_pipeline.py summary [<run-dir>]      summary.png + the same table
    python scripts/train_pipeline.py stop    [<run-dir>]
    python scripts/train_pipeline.py retry   <stage> [<run-dir>] [--now]

`start` converts the recording (scripts/prepare_baseline_datasets.py) in the
foreground, then hands the three trainings -- lerobot-train for the residual
pipeline's base policy, the B-Spline and SAIL bridges in their own venvs -- to
a detached runner and returns. The trainings survive the terminal closing.
Each stage writes its own log under logs/, the runner collects the wandb links
into links.md as they appear and draws summary.png (loss curves, checkpoint
paths, links) when the last training ends. `status` reads progress out of the
logs, so it works whether or not the runner is still alive.

A stage that fails keeps the last error line of its log in pipeline.json, and
`status` prints it. `retry <stage>` launches the stage again from the last
checkpoint it left under the run (lerobot-train's checkpoints/last, B-Spline's
checkpoints/latest.ckpt, SAIL's newest model_epoch_N.pth): its log moves to
logs/<stage>.attempt<N>.log, and while other stages are still training the
retry waits for them to end (the three share one GPU; `--now` skips the wait).
A stage that ran out of GPU memory beside the others is requeued that way once
without being asked. The runner takes the request if it is alive; otherwise
`retry` starts a new one.

No training starts from scratch while an earlier run holds its work. For each
stage, `start` looks under <root> for an earlier run of the same recording
that ran the same training (STAGE_IDENTITY names the settings that count). A
finished one is linked in instead of trained again -- <stage>/ and
logs/<stage>.log point at the earlier run, and `status` says so. Otherwise
the newest checkpoint of a stopped or failed one is copied in and the
training resumes from it. `--retrain <stage>` (or `all`) trains from scratch
regardless.

One run is one directory (pipelines/example.yaml documents every knob):

    <root>/<dataset>/<timestamp>-<name>/
      config.yaml  pipeline.json  links.md  summary.png
      datasets/{sysid,sail,bspline}.hdf5
      diffusion/checkpoints/<step>/pretrained_model    run_residual.py --base-policy
      bspline/<ts>/checkpoints/latest.ckpt             bspline_rollout.sh --ckpt
      sail/<ts>/models/model_epoch_N.pth               sail_rollout.sh --ckpt
      logs/{convert,diffusion,bspline,sail,pipeline}.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from baselines.run_record import _safe_segment  # noqa: E402

DEFAULT_ROOT = Path.home() / "franka_data" / "pipeline"
STAGES = ("diffusion", "bspline", "sail")
STATE_FILE = "pipeline.json"
# `retry`'s request to the runner; the file's text names the stages to wait for.
RETRY_MARKER = "retry.{stage}"
POLL_S = 5.0
STOP_GRACE_S = 20.0
RETRY_PICKUP_S = 2 * POLL_S + 2
RETRYABLE = ("failed", "stopped", "dead (runner gone)")

_REQUIRED = object()
# Section -> key -> default. Unknown keys are rejected: a misspelt knob must
# not silently fall back to the trainer's own default.
SCHEMA: dict[str | None, dict] = {
    # `steps` is one gradient-step budget for all three trainings (lerobot-train's
    # --steps, and the bridges' --steps, which turn it into their own epoch counts).
    # A stage's own `steps` overrides it; a stage's `epochs` may not be combined with it.
    None: {"dataset": _REQUIRED, "name": _REQUIRED, "root": str(DEFAULT_ROOT), "parallel": True,
           "steps": None},
    "wandb": {"enable": True, "project": "franka-pipeline", "entity": None},
    "convert": {"episodes": None, "image_size": None, "trim_start": "auto", "max_trim": 5,
                "min_steps": 20, "skip": []},
    "diffusion": {"enabled": True, "policy_type": "diffusion", "policy_repo_id": None,
                  "batch_size": 64, "steps": None, "save_freq": 20_000, "eval_freq": 5_000,
                  "log_freq": 200, "num_workers": 12, "args": {}},
    "bspline": {"enabled": True, "epochs": None, "steps": None, "checkpoint_every": None,
                "batch_size": None, "num_workers": None, "seed": None, "device": None},
    "sail": {"enabled": True, "epochs": None, "steps": None, "epoch_every_n_steps": None,
             "save_every": None, "batch_size": None, "seed": None, "data_workers": None,
             "err_threshold": 0.005, "num_workers": 6,
             "action_key": "absolute_actions_with_precision"},
}
DIFFUSION_DEFAULT_STEPS = 200_000
# The trainers' own defaults, only so progress can be shown as n/total.
BSPLINE_DEFAULT_EPOCHS = 601
SAIL_DEFAULT_EPOCHS = 1000


# ---------------------------------------------------------------- config

def load_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: expected a mapping at the top level")
    cfg: dict = {}
    for section, defaults in SCHEMA.items():
        given = raw if section is None else (raw.get(section) or {})
        if section is not None and not isinstance(given, dict):
            raise SystemExit(f"{path}: {section}: must be a mapping")
        known = set(defaults) | (set(SCHEMA) - {None} if section is None else set())
        unknown = set(given) - known
        if unknown:
            where = section or "top level"
            raise SystemExit(f"{path}: unknown key(s) in {where}: {', '.join(sorted(unknown))}; "
                             f"see pipelines/example.yaml")
        merged = {}
        for key, default in defaults.items():
            value = given.get(key, default)
            if value is _REQUIRED:
                raise SystemExit(f"{path}: {key} is required")
            merged[key] = value
        if section is None:
            cfg.update(merged)
        else:
            cfg[section] = merged
    if not isinstance(cfg["diffusion"]["args"], dict):
        raise SystemExit(f"{path}: diffusion.args must be a mapping of lerobot-train overrides")
    for stage in ("sail", "bspline"):
        if cfg[stage]["enabled"] and stage in cfg["convert"]["skip"]:
            raise SystemExit(f"{path}: {stage} is enabled but its conversion is in convert.skip")
    # One budget, stated once. A stage's `steps` overrides the top-level one; its
    # `epochs` cannot sit beside either, or the yaml would say two lengths at once.
    for stage in ("sail", "bspline"):
        if cfg[stage]["steps"] is None:
            cfg[stage]["steps"] = cfg["steps"]
        if cfg[stage]["steps"] is not None and cfg[stage]["epochs"] is not None:
            raise SystemExit(f"{path}: {stage}: epochs and steps both set; keep one "
                             "(the top-level `steps` counts as this stage's)")
    if cfg["diffusion"]["steps"] is None:
        cfg["diffusion"]["steps"] = cfg["steps"] if cfg["steps"] is not None else DIFFUSION_DEFAULT_STEPS
    if not any(cfg[s]["enabled"] for s in STAGES):
        raise SystemExit(f"{path}: every training is disabled")
    cfg["root"] = str(Path(cfg["root"]).expanduser())
    return cfg


# ----------------------------------------------------------------- state

def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def elapsed(start: str | None, end: str | None) -> str:
    if not start:
        return ""
    t0 = datetime.fromisoformat(start)
    t1 = datetime.fromisoformat(end) if end else datetime.now()
    s = int((t1 - t0).total_seconds())
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def read_state(run_dir: Path) -> dict:
    return json.loads((run_dir / STATE_FILE).read_text())


def write_state(run_dir: Path, state: dict) -> None:
    tmp = run_dir / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(run_dir / STATE_FILE)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); it counts as gone.
    try:
        return "Z" not in Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return True


def newest_run(root: Path = DEFAULT_ROOT) -> Path:
    candidates = list(root.glob(f"*/*/{STATE_FILE}")) + list(root.glob(f"*/*/*/{STATE_FILE}"))
    if not candidates:
        raise SystemExit(f"no runs under {root}; pass the run directory")
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def resolve_run_dir(arg: str | None) -> Path:
    run_dir = Path(arg).expanduser().resolve() if arg else newest_run()
    if not (run_dir / STATE_FILE).is_file():
        raise SystemExit(f"{run_dir} is not a pipeline run (no {STATE_FILE})")
    return run_dir


# -------------------------------------------------------------- commands

def wandb_name(cfg: dict, stage: str) -> str:
    return f"{cfg['dataset'].split('/')[-1]}-{cfg['name']}-{stage}"


def child_env(cfg: dict, run_dir: Path) -> dict:
    # Unbuffered so `status` can read a live log; the group ties the three wandb runs together.
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "WANDB_RUN_GROUP": f"{cfg['dataset']}/{run_dir.name}"}
    if cfg["wandb"]["entity"]:
        env["WANDB_ENTITY"] = str(cfg["wandb"]["entity"])
    return env


def convert_cmd(cfg: dict, run_dir: Path) -> list[str]:
    c = cfg["convert"]
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "prepare_baseline_datasets.py"),
           "--source-repo-id", cfg["dataset"], "--out-dir", str(run_dir / "datasets"),
           "--trim-start", str(c["trim_start"]), "--max-trim", str(c["max_trim"]),
           "--min-steps", str(c["min_steps"])]
    if c["episodes"]:
        cmd += ["--episodes", ",".join(str(int(e)) for e in c["episodes"])]
    if c["image_size"]:
        cmd += ["--image-size", str(c["image_size"])]
    if c["skip"]:
        cmd += ["--skip", *c["skip"]]
    return cmd


_SAIL_CKPT = re.compile(r"model_epoch_(\d+)\.pth$")


def stage_checkpoint(run_dir: Path, stage: str) -> tuple[Path, int] | None:
    """(checkpoint, how far it got) of the stage's last checkpoint under the run:
    lerobot's train_config.json and its step, B-Spline's latest.ckpt and the last
    epoch its run logged (the checkpoint is at or just below it), SAIL's newest
    model_epoch_N.pth and N."""
    if stage == "diffusion":
        return diffusion_checkpoint(run_dir)
    if stage == "bspline":
        found = list((run_dir / "bspline").glob("*/checkpoints/latest.ckpt")) if (run_dir / "bspline").is_dir() else []
        if not found:
            return None

        def logged_epoch(ckpt: Path) -> int:
            epoch = 0
            for line in read_log(ckpt.parent.parent / "logs.json.txt").splitlines():
                try:
                    epoch = max(epoch, int(json.loads(line).get("epoch", 0)))
                except (ValueError, AttributeError):
                    continue
            return epoch

        # The run that got furthest, not the newest file: an aborted relaunch
        # writes an epoch-0 latest.ckpt within its first minute.
        ckpt = max(found, key=lambda p: (logged_epoch(p), p.stat().st_mtime))
        return ckpt, logged_epoch(ckpt)
    found = list((run_dir / "sail").glob("*/models/model_epoch_*.pth")) if (run_dir / "sail").is_dir() else []
    if not found:
        return None
    ckpt = max(found, key=lambda p: int(_SAIL_CKPT.search(p.name).group(1)))
    return ckpt, int(_SAIL_CKPT.search(ckpt.name).group(1))


def diffusion_checkpoint(run_dir: Path) -> tuple[Path, int] | None:
    """(train_config.json, step) of the last diffusion checkpoint, if one was saved."""
    last = run_dir / "diffusion" / "checkpoints" / "last"
    config = last / "pretrained_model" / "train_config.json"
    if not config.is_file():
        return None
    try:
        step = int(json.loads((last / "training_state" / "training_step.json").read_text())["step"])
    except (OSError, ValueError, KeyError):
        step = 0
    return config, step


def wandb_run_exists(entity: str | None, project: str, run_id: str) -> bool:
    """False only when wandb says the run is gone; any other trouble leaves the decision to lerobot."""
    import wandb
    api = wandb.Api()
    try:
        api.run(f"{entity or api.default_entity}/{project}/{run_id}")
    except wandb.errors.CommError as exc:
        return "not found" not in str(exc).lower()
    except Exception:
        return True
    return True


def diffusion_resume_extra(cfg: dict, config_path: Path) -> tuple[list[str], str | None]:
    """lerobot resumes with wandb resume="must" on the saved run id, which fails
    outright if that run was deleted; resume without wandb then."""
    if not cfg["wandb"]["enable"]:
        return [], None
    try:
        saved = json.loads(config_path.read_text()).get("wandb") or {}
    except (OSError, ValueError):
        return [], None
    run_id = saved.get("run_id")
    if run_id and not wandb_run_exists(saved.get("entity") or cfg["wandb"]["entity"], cfg["wandb"]["project"], run_id):
        return ["--wandb.enable=false"], f"wandb run {run_id} no longer exists; resuming without wandb"
    return [], None


def diffusion_cmd(cfg: dict, run_dir: Path, dataset_root: Path) -> list[str]:
    d, w = cfg["diffusion"], cfg["wandb"]
    ckpt = diffusion_checkpoint(run_dir)
    if ckpt is not None:
        # A relaunch, or a checkpoint copied from an earlier run. The saved config
        # carries every argument below; lerobot-train restores the step, optimizer
        # and wandb run from the checkpoint.
        return [sys.executable, "-m", "lerobot.scripts.lerobot_train",
                "--resume=true", f"--config_path={ckpt[0]}", f"--output_dir={run_dir / 'diffusion'}"]
    # root as well as repo_id: a recording under ~/franka_data is not in the HF
    # cache, and a bare id would send lerobot to the Hub for it.
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
           f"--dataset.repo_id={cfg['dataset']}", f"--dataset.root={dataset_root}",
           f"--policy.type={d['policy_type']}",
           f"--output_dir={run_dir / 'diffusion'}",
           f"--job_name={wandb_name(cfg, 'diffusion')}",
           f"--batch_size={d['batch_size']}", f"--steps={d['steps']}",
           f"--save_freq={d['save_freq']}", f"--eval_freq={d['eval_freq']}",
           f"--log_freq={d['log_freq']}", f"--num_workers={d['num_workers']}",
           f"--wandb.enable={'true' if w['enable'] else 'false'}",
           f"--wandb.project={w['project']}"]
    if w["entity"]:
        cmd.append(f"--wandb.entity={w['entity']}")
    if d["policy_repo_id"]:
        cmd += [f"--policy.repo_id={d['policy_repo_id']}", "--policy.push_to_hub=true"]
    else:
        cmd.append("--policy.push_to_hub=false")
    for key, value in d["args"].items():
        cmd.append(f"--{key}={value}")
    return cmd


def bspline_cmd(cfg: dict, run_dir: Path, dataset_root: Path) -> list[str]:
    b, w = cfg["bspline"], cfg["wandb"]
    cmd = [sys.executable, "-m", "baselines.bspline_bridge.train",
           str(run_dir / "datasets" / "bspline.hdf5"),
           "--output-dir", str(run_dir), "--name", "bspline"]
    ckpt = stage_checkpoint(run_dir, "bspline")
    if ckpt is not None:
        # The bridge continues that run to the epoch count its checkpoint holds.
        cmd += ["--resume", str(ckpt[0].parent.parent)]
    for key, flag in (("epochs", "--epochs"), ("steps", "--steps"),
                      ("checkpoint_every", "--checkpoint-every"),
                      ("batch_size", "--batch-size"), ("num_workers", "--num-workers"),
                      ("seed", "--seed"), ("device", "--device")):
        if b[key] is not None and not (ckpt is not None and key in ("epochs", "steps")):
            cmd += [flag, str(b[key])]
    if w["enable"]:
        cmd += ["--wandb", "--wandb-project", w["project"],
                "--wandb-name", wandb_name(cfg, "bspline")]
    return cmd


def sail_cmd(cfg: dict, run_dir: Path, dataset_root: Path) -> list[str]:
    s, w = cfg["sail"], cfg["wandb"]
    cmd = [sys.executable, "-m", "baselines.sail_bridge.train",
           str(run_dir / "datasets" / "sail.hdf5"),
           "--output-dir", str(run_dir), "--name", "sail",
           "--action-key", s["action_key"],
           "--err-threshold", str(s["err_threshold"]), "--num-workers", str(s["num_workers"])]
    ckpt = stage_checkpoint(run_dir, "sail")
    if ckpt is not None:
        cmd += ["--resume", str(ckpt[0])]
    for key, flag in (("epochs", "--epochs"), ("steps", "--steps"),
                      ("epoch_every_n_steps", "--epoch-every-n-steps"),
                      ("save_every", "--save-every"), ("batch_size", "--batch-size"),
                      ("seed", "--seed"), ("data_workers", "--data-workers")):
        if s[key] is not None:
            cmd += [flag, str(s[key])]
    if w["enable"]:
        cmd += ["--wandb", "--wandb-project", w["project"],
                "--wandb-name", wandb_name(cfg, "sail")]
        if w["entity"]:
            cmd += ["--wandb-entity", str(w["entity"])]
    return cmd


STAGE_CMDS = {"diffusion": diffusion_cmd, "bspline": bspline_cmd, "sail": sail_cmd}


def spawn(cmd: list[str], log_path: Path, env: dict) -> subprocess.Popen:
    with open(log_path, "ab") as log:
        return subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)


def tee_run(cmd: list[str], log_path: Path, env: dict) -> int:
    """Run in the foreground, echoing to the terminal and the log."""
    print("$ " + " ".join(cmd))
    with open(log_path, "ab") as log:
        log.write(("$ " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                log.write(line)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        return proc.wait()


def start_runner(run_dir: Path, env: dict) -> subprocess.Popen:
    """Launch the detached runner; it outlives the terminal."""
    log_path = run_dir / "logs" / "pipeline.log"
    with open(log_path, "ab") as log:
        runner = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_run", str(run_dir)],
                                  cwd=str(_REPO_ROOT), env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3.0)
    if runner.poll() is not None:
        raise SystemExit(f"the runner exited at once (exit {runner.returncode}); see {log_path}")
    return runner


# The diffusion settings that shape the trained policy; the others are logging.
# The settings, per section, that shape what a stage trains; worker counts,
# devices and logging are left out. None takes the whole section.
STAGE_IDENTITY = {
    "diffusion": {"diffusion": ("policy_type", "batch_size", "steps", "args")},
    "bspline": {"convert": None, "bspline": ("epochs", "steps", "checkpoint_every", "batch_size", "seed")},
    "sail": {"convert": None, "sail": ("epochs", "steps", "epoch_every_n_steps", "save_every",
                                       "batch_size", "seed", "err_threshold", "action_key")},
}


def identity(cfg: dict, stage: str) -> dict:
    return {section: cfg[section] if keys is None else {k: cfg[section][k] for k in keys}
            for section, keys in STAGE_IDENTITY[stage].items()}


def find_earlier(cfg: dict, stage: str) -> tuple[Path, int, bool] | None:
    """(run_dir, how far, finished) of the earlier run under <root>/<dataset> that got
    furthest through the same training of `stage`: the newest finished one if any,
    else the stopped, failed or dead one with the latest checkpoint. Runs still
    training, and runs that took the stage from another run, are not candidates."""
    want = identity(cfg, stage)
    runs = sorted((Path(cfg["root"]) / _safe_segment(cfg["dataset"])).glob(f"*/{STATE_FILE}"), reverse=True)
    best = None
    for state_path in runs:
        run_dir = state_path.parent
        try:
            state = json.loads(state_path.read_text())
            other = load_config(run_dir / "config.yaml")
        except (OSError, ValueError, SystemExit):
            continue
        if state.get("dataset") != cfg["dataset"] or identity(other, stage) != want:
            continue
        st = state["stages"][stage]
        over = st["state"] in ("done", "stopped", "failed") or (st["state"] == "running" and not pid_alive(st["pid"]))
        ckpt = stage_checkpoint(run_dir, stage)
        if not over or ckpt is None or (run_dir / stage).is_symlink():
            continue
        if st["state"] == "done" and not (stage == "diffusion" and ckpt[1] < int(cfg["diffusion"]["steps"])):
            return run_dir, ckpt[1], True
        if ckpt[1] > 0 and (best is None or ckpt[1] > best[1]):
            best = (run_dir, ckpt[1], False)
    return best


def link_stage(src_run: Path, run_dir: Path, stage: str) -> None:
    """The earlier run's finished stage under this run's documented paths, so
    e.g. `run_residual.py --base-policy <run>/diffusion/...` reads the same either way."""
    os.symlink(os.path.relpath(src_run / stage, run_dir), run_dir / stage)
    os.symlink(os.path.relpath(src_run / "logs" / f"{stage}.log", run_dir / "logs"), run_dir / "logs" / f"{stage}.log")


def _link_or_copy(a, b):
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def copy_checkpoint(src_run: Path, run_dir: Path, stage: str) -> Path:
    """The earlier run's last checkpoint of `stage` under this run, so the training
    resumes here and writes its own checkpoints beside it. Files a resume never
    rewrites (lerobot's step directory, SAIL's .pth) are hardlinked where the
    filesystem allows; B-Spline rewrites latest.ckpt and appends to its logs in
    place, so its run directory is copied."""
    src, _ = stage_checkpoint(src_run, stage)
    if stage == "diffusion":
        dst = run_dir / "diffusion" / "checkpoints"
        dst.mkdir(parents=True)
        last = src_run / "diffusion" / "checkpoints" / "last"
        if not last.is_symlink():
            shutil.copytree(last, dst / "last", copy_function=_link_or_copy)
            return dst / "last"
        name = os.readlink(last)
        shutil.copytree(last.parent / name, dst / name, copy_function=_link_or_copy)
        os.symlink(name, dst / "last")
        return dst / name
    if stage == "bspline":
        src_dir = src.parent.parent                       # bspline/<ts>/
        dst_dir = run_dir / "bspline" / src_dir.name
        shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("epoch=*.ckpt", "wandb"))
        return dst_dir / "checkpoints" / "latest.ckpt"
    dst = run_dir / "sail" / src.parent.parent.name / "models" / src.name   # sail/<ts>/models/model_epoch_N.pth
    dst.parent.mkdir(parents=True)
    _link_or_copy(src, dst)
    return dst


# ----------------------------------------------------------------- start

def cmd_start(args) -> int:
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    from lerobot_robot_bimanual_franka.lerobot_source import resolve_root
    try:
        dataset_root = resolve_root(cfg["dataset"])
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    run_dir = Path(cfg["root"]) / _safe_segment(cfg["dataset"]) / f"{time.strftime('%Y%m%d_%H%M%S')}-{cfg['name']}"
    retrain = set(STAGES) if "all" in (args.retrain or []) else set(args.retrain or [])
    earlier = {s: find_earlier(cfg, s) for s in STAGES if cfg[s]["enabled"] and s not in retrain}
    earlier = {s: e for s, e in earlier.items() if e is not None}
    unit = {"diffusion": "step", "bspline": "epoch", "sail": "epoch"}
    if args.dry_run:
        print(f"[dry-run] run directory {run_dir}\n[dry-run] recording {dataset_root}")
        for s, (src_run, got, finished) in earlier.items():
            if finished:
                print(f"[dry-run] {s}: reuse {src_run / s} (--retrain {s} trains it again)")
            else:
                print(f"[dry-run] {s}: resume from {unit[s]} {got} of {src_run / s} "
                      f"(--retrain {s} starts from scratch)")
        for label, cmd in (("convert", convert_cmd(cfg, run_dir)),
                           *((s, STAGE_CMDS[s](cfg, run_dir, dataset_root))
                             for s in STAGES if cfg[s]["enabled"] and not (s in earlier and earlier[s][2]))):
            print(f"[dry-run] {label}: " + " ".join(cmd))
        return 0
    logs = run_dir / "logs"
    (run_dir / "datasets").mkdir(parents=True, exist_ok=False)
    logs.mkdir()
    shutil.copy(config_path, run_dir / "config.yaml")
    state = {
        "dataset": cfg["dataset"], "dataset_root": str(dataset_root), "name": cfg["name"],
        "parallel": bool(cfg["parallel"]), "created": now(),
        "runner": {"pid": None, "started": None, "ended": None, "rc": None},
        "stages": {"convert": {"state": "pending", "log": "logs/convert.log"}},
    }
    for stage in STAGES:
        state["stages"][stage] = {
            "state": "pending" if cfg[stage]["enabled"] else "skipped",
            "pid": None, "started": None, "ended": None, "rc": None, "wandb": None,
            "log": f"logs/{stage}.log", "attempts": 0, "error": None, "queued_behind": [],
        }
    taken = []
    for s, (src_run, got, finished) in earlier.items():
        if finished:
            link_stage(src_run, run_dir, s)
            done = read_state(src_run)["stages"][s]
            state["stages"][s].update(state="reused", from_run=str(src_run), wandb=done.get("wandb"),
                                      started=done.get("started"), ended=done.get("ended"), rc=0)
            taken.append(f"{s:<10}    reusing the finished {s} of {src_run.name}; --retrain {s} trains it again\n"
                         f"              {stage_checkpoint(run_dir, s)[0]}")
        else:
            copied = copy_checkpoint(src_run, run_dir, s)
            state["stages"][s].update(from_run=str(src_run), resume_step=got if s == "diffusion" else None)
            taken.append(f"{s:<10}    resuming from {unit[s]} {got} of {src_run.name}; --retrain {s} starts "
                         f"from scratch\n              {copied}")
    write_state(run_dir, state)
    print(f"run directory {run_dir}")
    print(f"recording     {dataset_root}")
    for line in taken:
        print(line)

    env = child_env(cfg, run_dir)
    print("=== convert ===")
    state["stages"]["convert"].update(state="running", started=now())
    write_state(run_dir, state)
    rc = tee_run(convert_cmd(cfg, run_dir), logs / "convert.log", env)
    state["stages"]["convert"].update(state="done" if rc == 0 else "failed", rc=rc, ended=now())
    write_state(run_dir, state)
    if rc != 0:
        print(f"conversion failed (exit {rc}); nothing launched. Log: {logs / 'convert.log'}")
        return rc

    if args.no_train:
        print("--no-train: datasets written, no training launched")
        return 0

    runner = start_runner(run_dir, env)
    print(f"runner pid {runner.pid}, trainings launched "
          f"{'together' if cfg['parallel'] else 'one after another'}")
    print(f"\n  python scripts/train_pipeline.py status {run_dir}\n")
    print_status(run_dir)
    return 0


# ---------------------------------------------------------------- runner

def cmd_run(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    cfg = load_config(run_dir / "config.yaml")
    state = read_state(run_dir)
    previous = state["runner"]["pid"]
    if previous and previous != os.getpid() and pid_alive(previous):
        # One runner owns pipeline.json at a time. An earlier one still minding
        # its trainings cannot take requests; this one takes over when it exits.
        print(f"{now()} waiting for runner pid {previous} to exit", flush=True)
        while pid_alive(previous):
            time.sleep(POLL_S)
        state = read_state(run_dir)
    state["runner"].update(pid=os.getpid(), started=now(), ended=None, rc=None)
    write_state(run_dir, state)
    env = child_env(cfg, run_dir)

    # A fresh run has every enabled stage pending; a runner started by `retry`
    # finds none and takes its work from the marker file instead.
    pending = [s for s in STAGES if state["stages"][s]["state"] == "pending"]
    waiting: dict[str, set[str]] = {}  # stage -> stages that must end before it launches
    procs: dict[str, subprocess.Popen] = {}
    stopping = False

    dataset_root = Path(state["dataset_root"])

    def busy(stage: str) -> bool:
        st = state["stages"][stage]
        return stage in procs or (st["state"] == "running" and pid_alive(st["pid"]))

    def launch(stage: str) -> None:
        st = state["stages"][stage]
        log_path = run_dir / st["log"]
        attempt = int(st.get("attempts") or (1 if st.get("started") else 0)) + 1
        resume_step = 0
        if attempt > 1 and log_path.is_file():
            log_path.rename(log_path.with_name(f"{stage}.attempt{attempt - 1}.log"))
        ckpt = stage_checkpoint(run_dir, stage)
        if stage == "diffusion":
            if ckpt is not None:
                resume_step = ckpt[1]
            elif attempt > 1 and (run_dir / "diffusion").is_dir():
                # lerobot-train refuses an existing output_dir unless it resumes.
                (run_dir / "diffusion").rename(run_dir / f"diffusion.attempt{attempt - 1}")
        cmd = STAGE_CMDS[stage](cfg, run_dir, dataset_root)
        notes = []
        if ckpt is not None:
            notes.append(f"resuming from {ckpt[0]}")
        if resume_step:
            extra, note = diffusion_resume_extra(cfg, ckpt[0])
            cmd += extra
            notes += [note] if note else []
        with open(log_path, "ab") as log:
            log.write("".join(f"# {n}\n" for n in notes).encode() + ("$ " + " ".join(cmd) + "\n").encode())
        for n in notes:
            print(f"{now()} {stage}: {n}", flush=True)
        proc = spawn(cmd, log_path, env)
        procs[stage] = proc
        st.update(state="running", pid=proc.pid, cmd=cmd, started=now(), ended=None, rc=None,
                  wandb=None, error=None, attempts=attempt, resume_step=resume_step,
                  resume_from=str(ckpt[0]) if ckpt else None, queued_behind=[])
        notes = ([f"attempt {attempt}"] if attempt > 1 else []) + \
            ([f"resuming from step {resume_step}"] if resume_step else
             [f"resuming from {ckpt[0].parent.parent.name}/{ckpt[0].name}"] if ckpt else [])
        print(f"{now()} {stage}: pid {proc.pid}" + (f" ({', '.join(notes)})" if notes else ""), flush=True)

    def queue(stage: str, wait_for: set[str]) -> None:
        wait_for = {s for s in wait_for if s != stage and busy(s)}
        waiting[stage] = wait_for
        state["stages"][stage].update(state="queued", queued_behind=sorted(wait_for))
        print(f"{now()} {stage}: queued" + (f" behind {', '.join(sorted(wait_for))}" if wait_for else ""),
              flush=True)

    def take_requests() -> None:
        for marker in sorted(run_dir.glob(RETRY_MARKER.format(stage="*"))):
            stage = marker.name.split(".", 1)[1]
            wait_for = set(marker.read_text().split())
            marker.unlink()
            if stage not in STAGES or busy(stage) or stage in waiting or stage in pending:
                print(f"{now()} {stage}: retry request ignored (it is {state['stages'].get(stage, {}).get('state')})",
                      flush=True)
                continue
            queue(stage, wait_for)

    def launch_ready() -> None:
        while pending and (cfg["parallel"] or not procs):
            launch(pending.pop(0))
        for stage in list(waiting):
            if any(busy(s) for s in waiting[stage]) or (not cfg["parallel"] and procs):
                continue
            del waiting[stage]
            launch(stage)

    def on_signal(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f"{now()} signal {signum}: stopping", flush=True)
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    take_requests()
    launch_ready()
    write_state(run_dir, state)

    while procs or ((pending or waiting) and not stopping):
        for stage, proc in list(procs.items()):
            st = state["stages"][stage]
            if st["wandb"] is None:
                st["wandb"] = wandb_url(run_dir / st["log"])
            rc = proc.poll()
            if rc is None:
                continue
            del procs[stage]
            if stopping or rc == 0:
                st.update(state="stopped" if stopping else "done", rc=rc, ended=now())
                print(f"{now()} {stage}: {st['state']} (exit {rc})", flush=True)
                continue
            error = failure_reason(run_dir / st["log"], rc)
            st.update(state="failed", rc=rc, ended=now(), error=error)
            print(f"{now()} {stage}: failed (exit {rc}): {error or 'no error line in the log'}", flush=True)
            # The trainings share one GPU. A stage that ran out of memory beside
            # the others gets one run on its own once they have ended.
            if is_oom(error) and int(st["attempts"]) < 2 and any(busy(s) for s in STAGES if s != stage):
                queue(stage, {s for s in STAGES if s != stage})
        if not stopping:
            take_requests()
            launch_ready()
        write_state(run_dir, state)
        write_links(run_dir, state)
        if procs:
            time.sleep(POLL_S)
    for stage in pending + list(waiting):
        state["stages"][stage].update(state="stopped", queued_behind=[])
    failed = [s for s in STAGES if state["stages"][s]["state"] in ("failed", "stopped")]
    state["runner"].update(ended=now(), rc=1 if failed else 0)
    write_state(run_dir, state)
    write_links(run_dir, state)
    try:
        path = write_summary(run_dir, cfg, state)
        print(f"{now()} summary: {path}", flush=True)
    except Exception as exc:  # the figure is a convenience; the run is already on disk
        print(f"{now()} summary failed: {exc!r}", flush=True)
    return 1 if failed else 0


def write_links(run_dir: Path, state: dict) -> None:
    lines = [f"# {state['dataset']} / {run_dir.name}", "", f"run: {run_dir}", ""]
    for stage in STAGES:
        st = state["stages"][stage]
        if st["state"] == "skipped":
            continue
        lines.append(f"- {stage}: {st['wandb'] or '(no wandb link yet)'}  --  log {run_dir / st['log']}")
    (run_dir / "links.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------- log parsing

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_WANDB_URL = re.compile(r"https://wandb\.ai/\S+?/runs/[A-Za-z0-9]+")
_LEROBOT_STEP = re.compile(r"\bstep:\S+ smpl:.*?\bloss:([0-9.eE+-]+)")
_SAIL_EPOCH = re.compile(r"^Train Epoch (\d+)\s*$")
_SAIL_LOSS = re.compile(r'^\s*"Loss": ([0-9.eE+-]+)')
_SAIL_PRECISE = re.compile(r"(demo_\d+):\s+([0-9.]+)% of (\d+) steps labelled precise")
# The line a Python traceback ends on: `torch.OutOfMemoryError: CUDA out of memory ...`.
_EXC_LINE = re.compile(r"^(?:[A-Za-z_][\w.]*\.)?[A-Z]\w*(?:Error|Exception|Exit|Interrupt)(?::\s.*|\s*)$")
_FAILED_LINE = re.compile(r"\b(?:error|failed|fatal)\b", re.IGNORECASE)
_OOM = re.compile(r"out of memory|OutOfMemoryError|cudaErrorMemoryAllocation", re.IGNORECASE)


def read_log(path: Path, tail: int | None = None) -> str:
    if not path.is_file():
        return ""
    with open(path, "rb") as f:
        if tail is not None:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail))
        data = f.read()
    return _ANSI.sub("", data.decode("utf-8", errors="replace")).replace("\r", "\n")


def wandb_url(log_path: Path) -> str | None:
    m = _WANDB_URL.search(read_log(log_path))
    return m.group(0).rstrip(".,)'\"") if m else None


def failure_reason(log_path: Path, rc: int | None = None) -> str | None:
    """Why a stage failed, in one line: the last exception line of its log, else
    the last line that says so, else the signal that killed it."""
    exc = said = None
    for line in read_log(log_path, tail=500_000).splitlines():
        line = line.strip()
        if not line or line.startswith("$ "):
            continue
        if _EXC_LINE.match(line):
            exc = line
        elif _FAILED_LINE.search(line):
            said = line
    if exc is None and rc is not None and rc < 0:
        try:
            return f"killed by {signal.Signals(-rc).name}"
        except ValueError:
            return f"killed by signal {-rc}"
    return exc or said


def is_oom(reason: str | None) -> bool:
    return bool(reason and _OOM.search(reason))


def stage_logs(run_dir: Path, stage: str) -> list[Path]:
    """The stage's current log, then earlier attempts' logs, newest first."""
    earlier = sorted((run_dir / "logs").glob(f"{stage}.attempt*.log"), reverse=True)
    return [run_dir / "logs" / f"{stage}.log", *earlier]


def diffusion_losses(run_dir: Path, log_freq: int, start: int = 0) -> tuple[list[int], list[float]]:
    steps, losses = [], []
    for line in read_log(run_dir / "logs" / "diffusion.log").splitlines():
        m = _LEROBOT_STEP.search(line)
        if m:
            # lerobot prints the step rounded to thousands; every log line is log_freq
            # apart, counted from the checkpoint a relaunch resumed at.
            steps.append(start + (len(steps) + 1) * log_freq)
            losses.append(float(m.group(1)))
    return steps, losses


def bspline_losses(run_dir: Path) -> tuple[list[int], list[float]]:
    logs = sorted((run_dir / "bspline").glob("*/logs.json.txt")) if (run_dir / "bspline").is_dir() else []
    if not logs:
        return [], []
    per_epoch: dict[int, list[float]] = {}
    for line in logs[-1].read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "train_loss" in row and "epoch" in row:
            per_epoch.setdefault(int(row["epoch"]), []).append(float(row["train_loss"]))
    epochs = sorted(per_epoch)
    return epochs, [sum(per_epoch[e]) / len(per_epoch[e]) for e in epochs]


def sail_losses(run_dir: Path) -> tuple[list[int], list[float]]:
    epochs, losses, current = [], [], None
    for line in read_log(run_dir / "logs" / "sail.log").splitlines():
        m = _SAIL_EPOCH.match(line)
        if m:
            current = int(m.group(1))
            continue
        m = _SAIL_LOSS.match(line)
        if m and current is not None:
            epochs.append(current)
            losses.append(float(m.group(1)))
            current = None
    return epochs, losses


_RESOLVED_EPOCHS = re.compile(r"-> (\d+) epochs|resume: epoch \d+ of (\d+)|resuming at epoch \d+ of (\d+)")


def resolved_epochs(run_dir: Path, stage: str) -> int | None:
    """The epoch count a bridge derived from --steps, or read back from a checkpoint, as it logged it."""
    m = _RESOLVED_EPOCHS.search(read_log(run_dir / "logs" / f"{stage}.log"))
    return int(next(g for g in m.groups() if g)) if m else None


def sail_precision(run_dir: Path) -> tuple[float | None, int]:
    """(mean precise fraction weighted by steps, demos at 0%) from the labelling summary,
    which only the attempt that labelled the file printed."""
    rows: list = []
    for log in stage_logs(run_dir, "sail"):
        rows = _SAIL_PRECISE.findall(read_log(log))
        if rows:
            break
    if not rows:
        return None, 0
    total = sum(int(n) for _, _, n in rows)
    weighted = sum(float(pct) / 100 * int(n) for _, pct, n in rows) / max(total, 1)
    return weighted, sum(1 for _, pct, _ in rows if float(pct) == 0.0)


def checkpoints(run_dir: Path, stage: str) -> Path | None:
    if stage == "diffusion":
        last = run_dir / "diffusion" / "checkpoints" / "last" / "pretrained_model"
        if last.is_dir():
            return last
        found = sorted((run_dir / "diffusion" / "checkpoints").glob("*/pretrained_model")) \
            if (run_dir / "diffusion" / "checkpoints").is_dir() else []
        return found[-1] if found else None
    ckpt = stage_checkpoint(run_dir, stage)
    return ckpt[0] if ckpt else None


def progress(run_dir: Path, cfg: dict, st: dict, stage: str) -> str:
    if st["state"] == "queued":
        behind = st.get("queued_behind") or []
        return "waiting for " + ", ".join(behind) if behind else "about to launch"
    if st["state"] == "reused":
        return "finished policy"
    if stage == "diffusion":
        start = int(st.get("resume_step") or 0)
        steps, losses = diffusion_losses(run_dir, int(cfg["diffusion"]["log_freq"]), start)
        total = int(cfg["diffusion"]["steps"])
        if not steps:
            return f"resuming from step {start}" if start else "starting"
        return f"step {steps[-1]}/{total} ({100 * steps[-1] // total}%)  loss {losses[-1]:.4f}"
    if stage == "bspline":
        epochs, losses = bspline_losses(run_dir)
        total = cfg["bspline"]["epochs"] or resolved_epochs(run_dir, "bspline") or BSPLINE_DEFAULT_EPOCHS
        if not epochs:
            return "resuming" if st.get("resume_from") else "starting"
        return f"epoch {epochs[-1] + 1}/{total}  loss {losses[-1]:.4f}"
    epochs, losses = sail_losses(run_dir)
    total = cfg["sail"]["epochs"] or resolved_epochs(run_dir, "sail") or SAIL_DEFAULT_EPOCHS
    frac, zeros = sail_precision(run_dir)
    label = "" if frac is None else f"  precise {100 * frac:.0f}%" + (f" ({zeros} demos at 0%)" if zeros else "")
    if not epochs:
        return ("resuming" if st.get("resume_from") else "labelling" if frac is None else "starting") + label
    return f"epoch {epochs[-1]}/{total}  loss {losses[-1]:.4f}" + label


# ---------------------------------------------------------------- status

def shown_state(st: dict, runner_alive: bool) -> str:
    if st["state"] == "running" and not pid_alive(st["pid"]):
        return "ended (runner catching up)" if runner_alive else "dead (runner gone)"
    return st["state"]


def stage_rows(run_dir: Path, cfg: dict, state: dict) -> list[dict]:
    runner_alive = pid_alive(state["runner"]["pid"])
    rows = []
    conv = state["stages"]["convert"]
    n_files = len(list((run_dir / "datasets").glob("*.hdf5")))
    rows.append({"stage": "convert", "state": conv["state"], "pid": None,
                 "progress": f"{n_files} HDF5 file(s) in {run_dir / 'datasets'}",
                 "elapsed": elapsed(conv.get("started"), conv.get("ended")),
                 "ckpt": None, "wandb": None, "log": run_dir / conv["log"],
                 "error": failure_reason(run_dir / conv["log"], conv.get("rc")) if conv["state"] == "failed" else None,
                 "retryable": False})
    for stage in STAGES:
        st = state["stages"][stage]
        shown = shown_state(st, runner_alive)
        prog = progress(run_dir, cfg, st, stage) if st["state"] not in ("skipped", "pending") else ""
        if int(st.get("attempts") or 0) > 1:
            prog = f"attempt {st['attempts']}: {prog}"
        error = None
        if shown in RETRYABLE:
            error = st.get("error") or failure_reason(run_dir / st["log"], st.get("rc"))
        rows.append({"stage": stage, "state": shown, "pid": st["pid"] if shown == "running" else None,
                     "progress": prog,
                     "elapsed": elapsed(st.get("started"), st.get("ended")),
                     "ckpt": checkpoints(run_dir, stage),
                     "wandb": st["wandb"] or wandb_url(run_dir / st["log"]),
                     "log": run_dir / st["log"], "error": error, "retryable": shown in RETRYABLE,
                     "from_run": st.get("from_run")})
    return rows


def format_status(run_dir: Path, cfg: dict, state: dict) -> str:
    runner = state["runner"]
    runner_line = "not started" if not runner["pid"] else (
        f"finished (exit {runner['rc']})" if runner["ended"]
        else f"pid {runner['pid']} alive" if pid_alive(runner["pid"]) else f"pid {runner['pid']} gone")
    lines = [f"run      {run_dir}",
             f"dataset  {state['dataset']}   ({state.get('dataset_root', '')})",
             f"config   {run_dir / 'config.yaml'}",
             f"runner   {runner_line}, trainings {'together' if state['parallel'] else 'in order'}",
             f"links    {run_dir / 'links.md'}", ""]
    for row in stage_rows(run_dir, cfg, state):
        pid = f"pid {row['pid']}" if row["pid"] else ""
        lines.append(f"{row['stage']:<10} {row['state']:<9} {pid:<12} {row['elapsed']:>8}  {row['progress']}")
        if row["state"] == "skipped":
            continue
        if row.get("from_run"):
            lines.append(f"{'':10} from   {row['from_run']}")
        if row["ckpt"]:
            lines.append(f"{'':10} ckpt   {row['ckpt']}")
        if row["wandb"]:
            lines.append(f"{'':10} wandb  {row['wandb']}")
        lines.append(f"{'':10} log    {row['log']}")
        if row["error"]:
            lines.append(f"{'':10} error  {row['error'][:220]}")
        if row["retryable"]:
            lines.append(f"{'':10} retry  python scripts/train_pipeline.py retry {row['stage']} {run_dir}")
    if (run_dir / "summary.png").is_file():
        lines += ["", f"summary  {run_dir / 'summary.png'}"]
    return "\n".join(lines)


def print_status(run_dir: Path) -> None:
    cfg = load_config(run_dir / "config.yaml")
    print(format_status(run_dir, cfg, read_state(run_dir)))


def cmd_status(args) -> int:
    print_status(resolve_run_dir(args.run_dir))
    return 0


# --------------------------------------------------------------- summary

def write_summary(run_dir: Path, cfg: dict, state: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {
        "diffusion": ("step", diffusion_losses(run_dir, int(cfg["diffusion"]["log_freq"]),
                                               int(state["stages"]["diffusion"].get("resume_step") or 0))),
        "bspline": ("epoch", bspline_losses(run_dir)),
        "sail": ("epoch", sail_losses(run_dir)),
    }
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.25)
    fig.suptitle(f"{state['dataset']}  /  {run_dir.name}", fontsize=13, y=0.98)
    for col, stage in enumerate(STAGES):
        ax = fig.add_subplot(grid[0, col])
        xlabel, (xs, ys) = series[stage]
        st = state["stages"][stage]
        title = f"{stage}: {st['state']}"
        if ys:
            ax.plot(xs, ys, linewidth=1.0)
            if min(ys) > 0 and max(ys) / min(ys) > 10:
                ax.set_yscale("log")
            title += f"   final loss {ys[-1]:.4g}"
        else:
            ax.text(0.5, 0.5, "no loss logged", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("train loss")
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    text = format_status(run_dir, cfg, state)
    ax.text(0.0, 1.0, text, family="monospace", fontsize=8.2, va="top", ha="left",
            transform=ax.transAxes, wrap=False)
    out = run_dir / "summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def cmd_summary(args) -> int:
    run_dir = resolve_run_dir(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    state = read_state(run_dir)
    write_links(run_dir, state)
    path = write_summary(run_dir, cfg, state)
    print(format_status(run_dir, cfg, state))
    print(f"\nsummary  {path}")
    return 0


# ------------------------------------------------------------------ stop

def clear_retry_requests(run_dir: Path) -> None:
    for marker in run_dir.glob(RETRY_MARKER.format(stage="*")):
        marker.unlink()


def cmd_stop(args) -> int:
    run_dir = resolve_run_dir(args.run_dir)
    clear_retry_requests(run_dir)
    state = read_state(run_dir)
    runner_pid = state["runner"]["pid"]
    targets = {s: st["pid"] for s, st in state["stages"].items()
               if st.get("pid") and st["state"] == "running" and pid_alive(st["pid"])}
    if runner_pid and pid_alive(runner_pid):
        targets["runner"] = runner_pid
    if not targets:
        # A stage can be left "running" with nothing alive: the conversion runs
        # in `start`'s foreground, and a Ctrl-C there never reaches this file.
        stale = [s for s, st in state["stages"].items() if st["state"] in ("running", "queued")]
        for stage in stale:
            state["stages"][stage].update(state="stopped", ended=now(), queued_behind=[])
        if stale:
            write_state(run_dir, state)
            print("nothing is running; marked " + ", ".join(stale) + " stopped")
            print_status(run_dir)
        else:
            print("nothing is running")
        return 0
    print("stopping " + ", ".join(f"{k} (pid {v})" for k, v in targets.items()))
    # The runner was started in its own session, so its group holds every trainer
    # and their workers; the per-pid signals cover a runner that is already gone.
    for pid in {runner_pid, *targets.values()}:
        if not pid:
            continue
        for send in (lambda p: os.killpg(os.getpgid(p), signal.SIGTERM), lambda p: os.kill(p, signal.SIGTERM)):
            try:
                send(pid)
            except (ProcessLookupError, PermissionError):
                pass
    deadline = time.time() + STOP_GRACE_S
    while time.time() < deadline and any(pid_alive(p) for p in targets.values()):
        time.sleep(0.5)
    for name, pid in targets.items():
        if pid_alive(pid):
            print(f"{name} (pid {pid}) ignored SIGTERM; killing")
            for send in (lambda p: os.killpg(os.getpgid(p), signal.SIGKILL), lambda p: os.kill(p, signal.SIGKILL)):
                try:
                    send(pid)
                except (ProcessLookupError, PermissionError):
                    pass
    state = read_state(run_dir)
    for stage, st in state["stages"].items():
        if st["state"] in ("running", "queued"):
            st.update(state="stopped", ended=now(), queued_behind=[])
    if state["runner"]["pid"] and not state["runner"]["ended"]:
        state["runner"].update(ended=now(), rc=state["runner"]["rc"] if state["runner"]["rc"] is not None else 1)
    write_state(run_dir, state)
    print_status(run_dir)
    return 0


# ----------------------------------------------------------------- retry

def cmd_retry(args) -> int:
    run_dir = resolve_run_dir(args.run_dir)
    stage = args.stage
    cfg = load_config(run_dir / "config.yaml")
    state = read_state(run_dir)
    runner_pid = state["runner"]["pid"]
    runner_alive = pid_alive(runner_pid)
    st = state["stages"][stage]
    shown = shown_state(st, runner_alive)
    if shown not in RETRYABLE:
        hint = "; it will be marked in a few seconds" if shown.startswith("ended") else ""
        raise SystemExit(f"{stage} is {shown}{hint}; only a {', '.join(RETRYABLE)} stage can be retried")
    marker = run_dir / RETRY_MARKER.format(stage=stage)
    if marker.exists():
        raise SystemExit(f"a retry of {stage} is already requested ({marker})")
    running = [s for s in STAGES if s != stage and state["stages"][s]["state"] == "running"
               and pid_alive(state["stages"][s]["pid"])]
    wait_for = [] if args.now else running
    if st.get("error"):
        print(f"{stage} failed with: {st['error'][:220]}")
    marker.write_text(" ".join(wait_for) + "\n")
    if wait_for:
        print(f"{stage}: queued behind {', '.join(wait_for)} (they share the GPU; --now launches at once)")
    else:
        print(f"{stage}: launching")
    if runner_alive:
        print(f"the runner (pid {runner_pid}) takes the request within {POLL_S:.0f} s")
        deadline = time.time() + RETRY_PICKUP_S
        while marker.exists() and time.time() < deadline and pid_alive(runner_pid):
            time.sleep(0.5)
        if marker.exists():
            if pid_alive(runner_pid):
                # Started before `retry` existed, or hung: a new runner waits it out.
                print(f"the runner (pid {runner_pid}) did not take the request; a new runner "
                      f"takes over when it exits (its trainings are not disturbed)")
            runner_alive = False
    if not runner_alive:
        runner = start_runner(run_dir, child_env(cfg, run_dir))
        print(f"runner pid {runner.pid}")
    time.sleep(1.0)
    print()
    print_status(run_dir)
    return 0


# ------------------------------------------------------------------ main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start", help="convert the recording and launch the trainings")
    s.add_argument("config", help="a pipelines/*.yaml")
    s.add_argument("--no-train", action="store_true", help="convert only")
    s.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    s.add_argument("--retrain", action="append", choices=(*STAGES, "all"), metavar="STAGE",
                   help="train this stage from scratch even if an earlier run of this recording "
                        "finished it or left a checkpoint; repeatable, or `all`")
    s.set_defaults(func=cmd_start)
    for name, func, help_text in (("status", cmd_status, "progress, checkpoints and wandb links"),
                                  ("summary", cmd_summary, "write summary.png and print the status"),
                                  ("stop", cmd_stop, "stop the runner and every training it started")):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("run_dir", nargs="?", default=None, help=f"default: the newest run under {DEFAULT_ROOT}")
        s.set_defaults(func=func)
    s = sub.add_parser("retry", help="launch a failed, stopped or dead stage again")
    s.add_argument("stage", choices=STAGES)
    s.add_argument("run_dir", nargs="?", default=None, help=f"default: the newest run under {DEFAULT_ROOT}")
    s.add_argument("--now", action="store_true",
                   help="launch at once instead of after the stages still running have ended")
    s.set_defaults(func=cmd_retry)
    s = sub.add_parser("_run")
    s.add_argument("run_dir")
    s.set_defaults(func=cmd_run)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
