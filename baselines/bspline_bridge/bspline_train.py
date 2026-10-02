#!/usr/bin/env python3
"""bspline_policy's train.py, able to resume a run at the epoch its checkpoint holds.

    .venv-bspline/bin/python baselines/bspline_bridge/bspline_train.py <hydra overrides...>
    .venv-bspline/bin/python baselines/bspline_bridge/bspline_train.py --resume <run_dir> <hydra overrides...>

Upstream's `training.resume` restores the model, EMA, optimizer, epoch and step
from <run_dir>/checkpoints/latest.ckpt, but then trains `training.num_epochs`
further epochs and sizes the cosine learning-rate schedule to that count, so a
resumed run overshoots its epoch budget on a schedule that no longer matches its
step. This launcher reads the epoch out of the checkpoint, sets num_epochs to
what is left, and keeps the schedule sized to the run's original epoch count,
so the run ends where a fresh one would. Without --resume it is upstream's entry
point. The submodule itself is untouched.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

BRIDGE = pathlib.Path(__file__).resolve().parent
PROJECT = BRIDGE.parent / "bspline_policy" / "bspline_policy"


def load_upstream():
    # Not `import train`: this directory's own train.py would shadow it.
    spec = importlib.util.spec_from_file_location("bspline_upstream_train", PROJECT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # puts the project and diffusion_policy on sys.path
    return module


def checkpoint_epochs(run_dir: pathlib.Path) -> tuple[int, int]:
    """(epoch the checkpoint was saved at, the run's total epoch count)."""
    import dill
    import torch
    payload = torch.load(str(run_dir / "checkpoints" / "latest.ckpt"), map_location="cpu",
                         pickle_module=dill, mmap=True, weights_only=False)
    return int(dill.loads(payload["pickles"]["epoch"])), int(payload["cfg"].training.num_epochs)


def main() -> None:
    argv = sys.argv[1:]
    resume_dir = None
    if "--resume" in argv:
        i = argv.index("--resume")
        resume_dir = pathlib.Path(argv[i + 1]).resolve()
        del argv[i:i + 2]
    upstream = load_upstream()
    if resume_dir is not None:
        epoch, total = checkpoint_epochs(resume_dir)
        # The checkpoint is written after its epoch trains and before the counter
        # advances, so the loop repeats that one epoch and still ends on `total`.
        remaining = total - epoch
        if remaining <= 0:
            print(f"resume: {resume_dir} already trained all {total} epochs; nothing to do")
            return
        print(f"resume: epoch {epoch} of {total} from {resume_dir} ({remaining} to go)", flush=True)

        import diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace as workspace
        sized = workspace.get_scheduler

        def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps, **kwargs):
            # The workspace sized the schedule to `remaining` epochs; keep the run's.
            per_epoch = num_training_steps // remaining
            return sized(name, optimizer, num_warmup_steps, per_epoch * total, **kwargs)

        workspace.get_scheduler = get_scheduler
        argv += [f"hydra.run.dir={resume_dir}", f"multi_run.run_dir={resume_dir}",
                 "training.resume=true", f"training.num_epochs={remaining}"]
    sys.argv = [sys.argv[0], *argv]
    upstream.main()


if __name__ == "__main__":
    main()
