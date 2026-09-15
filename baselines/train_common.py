"""Shared by the two training entrypoints. stdlib-only: they run in the
workspace venv and hand the actual training to the baseline's interpreter."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from baselines.run_record import _safe_segment

logger = logging.getLogger("baselines.train")

DEFAULT_POLICIES_ROOT = Path.home() / "franka_data" / "policies"
# Both upstreams crop 76 of an 84 px frame; the same fraction of whatever the
# converted file holds, so a full-resolution recording trains unshrunk.
CROP_FRACTION = 76 / 84


def crop_size(h: int, w: int) -> tuple[int, int]:
    return round(h * CROP_FRACTION), round(w * CROP_FRACTION)


def policies_dir(source_repo_id: str | None, override: Path | None) -> Path:
    """<policies-root>/<train-dataset>, mirroring outputs/<train-dataset>."""
    if override is not None:
        return override.expanduser()
    if not source_repo_id:
        raise SystemExit(
            "the HDF5 carries no source_repo_id (converted before the stamp existed); "
            "pass --output-dir, or reconvert with --source-repo-id"
        )
    return DEFAULT_POLICIES_ROOT / _safe_segment(source_repo_id)


def stream(cmd: list[str], cwd: Path, env: dict | None = None,
           watch: str | None = None, stdin_text: str | None = None) -> tuple[int, bool]:
    """Run, echoing output line by line; -> (returncode, whether `watch` appeared).

    `stdin_text` answers any prompt the child raises; nothing here is
    interactive, and a child blocked on input() would look like a hang.
    """
    logger.info("$ cd %s && %s", cwd, " ".join(cmd))
    seen = False
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None and proc.stdin is not None
    proc.stdin.write(stdin_text or "")
    proc.stdin.close()
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if watch and watch in line:
            seen = True
    return proc.wait(), seen
