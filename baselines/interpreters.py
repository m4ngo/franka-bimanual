"""Which python runs each baseline.

The upstream projects cannot share the workspace venv (baselines/README.md), so
each has its own interpreter and everything that launches a policy server or a
training run resolves it here. Resolution order:

    1. $SAIL_PYTHON / $BSPLINE_PYTHON      an explicit interpreter
    2. ~/franka_ws/.venv-sail/bin/python    what scripts/setup_baseline_envs.sh creates
       ~/franka_ws/.venv-bspline/bin/python
    3. conda run -n SAIL|robodiff python    upstream's own env names, if conda exists

stdlib only, so the shell wrappers can call it before anything else is set up:

    mapfile -t SAIL_PY < <(python -m baselines.interpreters sail)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# backend -> (env var, default venv, upstream conda env name)
BACKENDS = {
    "sail": ("SAIL_PYTHON", _REPO_ROOT / ".venv-sail", "SAIL"),
    "bspline": ("BSPLINE_PYTHON", _REPO_ROOT / ".venv-bspline", "robodiff"),
}


def python_for(backend: str) -> list[str]:
    """Command prefix that runs python inside the backend's environment."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    var, venv, conda_env = BACKENDS[backend]
    explicit = os.environ.get(var)
    if explicit:
        return [explicit]
    candidate = venv / "bin" / "python"
    if candidate.is_file():
        return [str(candidate)]
    if shutil.which("conda"):
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python"]
    raise SystemExit(
        f"no interpreter for the {backend} baseline: {candidate} does not exist, "
        f"${var} is unset and conda is not on PATH. Run "
        f"scripts/setup_baseline_envs.sh {backend} once (baselines/README.md)."
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="print the command that runs python in a baseline's env, "
                                            "one word per line")
    p.add_argument("backend", choices=sorted(BACKENDS))
    for word in python_for(p.parse_args().backend):
        print(word)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
