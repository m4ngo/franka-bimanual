"""Rollout bookkeeping: one self-contained directory per run, fully described.

A rollout is only useful later if you can tell what produced it. This module
owns that record, and nothing else: it imports no lerobot, no robot driver and
no policy, so the residual runner can use it as cheaply as the baseline bridges.

Layout, rooted at `~/franka_data/outputs` (data never lives in the repo):

    outputs/<train-dataset>/<timestamp>-<method>/
        manifest.json     everything known about the run
        episodes.jsonl    one line per episode, appended as it finishes
        dataset/          the LeRobotDataset recorded during the run
        videos/           one time-aligned mp4 per camera per episode

Grouping by the TRAINING dataset rather than by method is what makes the
comparison readable: every method trained on one task's demonstrations lands in
one directory, so `ls outputs/HuskyMango/pickup-bowl` is the experiment.

`episodes.jsonl` is written line by line rather than folded into the manifest so
a run interrupted at the robot still leaves every episode that finished.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

logger = logging.getLogger("baselines.run_record")

SCHEMA_VERSION = 1
DEFAULT_ROOT = Path.home() / "franka_data" / "outputs"

# Directory-name suffix per method. `multifast` rather than `multi-fast` so the
# run id stays one token either side of the timestamp separator.
METHODS = ("sail", "bspline", "multifast")

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def sha256(path: str | Path | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_provenance(path: str | Path | None) -> dict:
    """Enough to tell later whether this is the same file."""
    if not path:
        return {"path": None, "exists": False}
    p = Path(path).expanduser()
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "sha256": sha256(p) if p.is_file() else None,
        "size_bytes": st.st_size if p.is_file() else None,
        "modified_at": stamp(st.st_mtime),
    }


def describe_lerobot_dataset(root: str | Path | None) -> dict:
    """Episode/frame counts off a LeRobot dataset's own metadata.

    Read from meta/info.json rather than by opening the dataset: this is called
    while the dataset is still being written, and decoding video to count frames
    would be absurd. Note info.json carries no repo id -- that is exactly why the
    training dataset's id is stamped onto the converted HDF5 instead.
    """
    if not root:
        return {"path": None, "exists": False}
    p = Path(root).expanduser()
    info_path = p / "meta" / "info.json"
    if not info_path.is_file():
        return {"path": str(p), "exists": p.exists(), "info": None}
    try:
        info = json.loads(info_path.read_text())
    except Exception as exc:
        return {"path": str(p), "exists": True, "info_error": repr(exc)}
    return {
        "path": str(p.resolve()),
        "exists": True,
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
    }


def read_source_repo_id(hdf5_path: str | Path | None) -> str | None:
    """The training dataset id `baselines/common.py` stamped on a converted HDF5.

    Read by the policy servers, which run in the upstream conda envs -- this
    module is stdlib-only precisely so they can import it. Returns None for a
    file converted before the stamp existed; the rollout then asks for
    --train-dataset instead of guessing.
    """
    if not hdf5_path:
        return None
    p = Path(hdf5_path).expanduser()
    if not p.is_file():
        return None
    try:
        import h5py
        with h5py.File(p, "r") as f:
            value = f.attrs.get("source_repo_id")
    except Exception as exc:
        logger.warning("could not read source_repo_id from %s: %r", p, exc)
        return None
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def git_state(repo: Path = _REPO_ROOT) -> dict:
    """Commit, branch and dirtiness of the code that produced the run.

    A dirty tree is recorded, not hidden: a rollout run off uncommitted changes
    is not reproducible from the commit alone and the manifest should say so.
    """
    def run(*args, strip: bool = True):
        try:
            out = subprocess.run(("git", "-C", str(repo)) + args, capture_output=True,
                                 text=True, timeout=10)
            if out.returncode != 0:
                return None
            return out.stdout.strip() if strip else out.stdout
        except Exception:
            return None

    # NOT stripped: a porcelain line is `XY path` with XY exactly two columns,
    # and stripping eats the leading space of the first line -- which silently
    # took the first character off the first filename.
    status = run("status", "--porcelain", strip=False)
    files = [l[3:] for l in (status or "").splitlines() if len(l) > 3]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(files) if status is not None else None,
        "dirty_files": sorted(files)[:50],
    }


def machine_state() -> dict:
    return {
        "host": socket.gethostname(),
        "user": _user(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
    }


def _user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER")


def stamp(when: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(when))


def _jsonable(value):
    """Make a value writable: dataclasses, Paths, numpy scalars, enums."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    for attr in ("item", "tolist"):          # numpy scalars and arrays
        if hasattr(value, attr):
            try:
                return _jsonable(getattr(value, attr)())
            except Exception:
                pass
    if hasattr(value, "value"):              # enums
        return _jsonable(value.value)
    return str(value)


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def _safe_segment(text: str) -> str:
    """A relative path from a dataset id, preserving an org/name split.

    `HuskyMango/pickup-bowl` stays two levels deep on purpose -- that is the
    grouping the comparison is read at. Every component is sanitised and `.`
    and `..` are dropped, so an id can only ever name a directory BELOW the
    outputs root: a typo should not scatter run directories across the disk.
    """
    parts = []
    for raw in text.strip().split("/"):
        cleaned = "".join(
            ch if (ch.isalnum() or ch in "._-") else "-" for ch in raw
        ).strip(".-")
        if cleaned and cleaned not in (".", ".."):
            parts.append(cleaned)
    return "/".join(parts) or "unknown"


@dataclass
class RunDir:
    """`<root>/<train_dataset>/<timestamp>-<method>/`, created on construction."""

    train_dataset: str
    method: str
    root: Path = field(default_factory=lambda: DEFAULT_ROOT)
    when: float | None = None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        self.root = Path(self.root).expanduser()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.when))
        self.run_id = f"{stamp}-{self.method}"
        self.path = self.root / _safe_segment(self.train_dataset) / self.run_id
        try:
            # Never exist_ok: a run directory is the record of one rollout, and
            # silently merging two of them would interleave their episodes.jsonl.
            self.path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{self.path} already exists. Two runs of {self.method!r} on "
                f"{self.train_dataset!r} within the same second cannot be told "
                "apart; wait a second and start again."
            ) from exc

    @property
    def dataset_dir(self) -> Path:
        return self.path / "dataset"

    @property
    def video_dir(self) -> Path:
        return self.path / "videos"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def episodes_path(self) -> Path:
        return self.path / "episodes.jsonl"


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Accumulates the manifest and streams episodes to disk.

    Sections are filled in as they become known -- the environment before the
    arm connects, the policy after the handshake, the summary at the end -- and
    the whole manifest is rewritten after every episode so an interrupted run is
    still described.
    """

    run_dir: RunDir
    method: str
    train_dataset: dict
    sections: dict = field(default_factory=dict)
    episodes: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._started = time.time()
        self._status = "running"
        self._reason: str | None = None
        self.sections.setdefault("environment", {}).update({
            "git": git_state(),
            "machine": machine_state(),
        })

    # -- filling in -------------------------------------------------------
    def set(self, section: str, **values) -> None:
        self.sections.setdefault(section, {}).update(_jsonable(values))

    def add_episode(self, episode) -> None:
        """Record one finished episode and append it to episodes.jsonl."""
        row = _jsonable(episode if isinstance(episode, dict) else asdict(episode))
        self.episodes.append(row)
        with open(self.run_dir.episodes_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        self.write()

    def finish(self, status: str = "completed", reason: str | None = None) -> None:
        self._status, self._reason = status, reason
        self.write()

    # -- summary ----------------------------------------------------------
    def summary(self) -> dict:
        eps = self.episodes
        ok = [e for e in eps if e.get("success")]
        times = sorted(e.get("wall_time_s", 0.0) for e in ok)
        verdicts: dict[str, int] = {}
        for e in eps:
            verdicts[e.get("verdict") or "unknown"] = verdicts.get(e.get("verdict") or "unknown", 0) + 1
        return {
            "episodes": len(eps),
            "successes": len(ok),
            "success_rate": (len(ok) / len(eps)) if eps else None,
            "verdicts": verdicts,
            # Time-to-success over SUCCESSES only: a failure's duration is the
            # timeout, which says nothing about how fast the method is.
            "mean_time_to_success_s": (sum(times) / len(times)) if times else None,
            "median_time_to_success_s": times[len(times) // 2] if times else None,
            "min_time_to_success_s": times[0] if times else None,
            "max_time_to_success_s": times[-1] if times else None,
            "total_episode_time_s": sum(e.get("wall_time_s", 0.0) for e in eps),
            "total_steps": sum(e.get("steps", 0) for e in eps),
            "total_inferences": sum(e.get("inferences", 0) for e in eps),
            "total_slow_steps": sum(e.get("slow_steps", 0) for e in eps),
            "total_guided_inferences": sum(e.get("guided_inferences", 0) for e in eps),
            "aborted_episodes": sum(1 for e in eps if e.get("aborted")),
        }

    # -- output -----------------------------------------------------------
    def write(self) -> Path:
        now = time.time()
        doc = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "run_id": self.run_dir.run_id,
                "method": self.method,
                "status": self._status,
                "reason": self._reason,
                "started_at": stamp(self._started),
                "updated_at": stamp(now),
                "duration_s": round(now - self._started, 3),
            },
            "train_dataset": _jsonable(self.train_dataset),
            **{k: v for k, v in self.sections.items()},
            "outputs": {
                "run_dir": str(self.run_dir.path),
                "manifest": str(self.run_dir.manifest_path),
                "episodes": str(self.run_dir.episodes_path),
                **_jsonable(self.sections.get("outputs", {})),
            },
            "summary": self.summary(),
        }
        tmp = self.run_dir.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=False))
        tmp.replace(self.run_dir.manifest_path)
        return self.run_dir.manifest_path


# ---------------------------------------------------------------------------
# Training-dataset resolution
# ---------------------------------------------------------------------------

def resolve_train_dataset(flag: str | None, from_checkpoint: str | None,
                          checkpoint_hdf5: str | None = None) -> dict:
    """Which task's demonstrations this policy learned from.

    Resolution order is explicit flag, then the id stamped onto the converted
    HDF5 at conversion time and reported back through the policy server's `meta`.
    When both exist and disagree, the flag wins and the disagreement is recorded
    -- that is usually a checkpoint pointed at the wrong output directory, and
    silently filing the run under the wrong task is the failure this is here to
    prevent.
    """
    repo_id = flag or from_checkpoint
    if not repo_id:
        raise ValueError(
            "cannot tell which dataset this policy trained on. Pass "
            "--train-dataset <repo-id>, or re-run the converter with "
            "--source-repo-id so the checkpoint's HDF5 carries it "
            "(baselines/ROLLOUT.md, 'Provenance')."
        )
    agrees = None
    if flag and from_checkpoint:
        agrees = flag == from_checkpoint
        if not agrees:
            logger.warning(
                "--train-dataset %r disagrees with the checkpoint's own %r; "
                "filing under the flag. Check you are rolling out the policy you "
                "think you are.", flag, from_checkpoint)
    out = {
        "repo_id": repo_id,
        "resolved_from": "flag" if flag else "checkpoint",
        "flag": flag,
        "checkpoint_says": from_checkpoint,
        "agrees": agrees,
        "training_hdf5": file_provenance(checkpoint_hdf5) if checkpoint_hdf5 else None,
    }
    try:
        from lerobot_robot_bimanual_franka.lerobot_source import resolve_root
        out["local"] = describe_lerobot_dataset(resolve_root(repo_id))
    except Exception as exc:
        # The recording need not still be on this machine to roll a policy out.
        out["local"] = {"resolved": False, "reason": repr(exc)}
    return out
