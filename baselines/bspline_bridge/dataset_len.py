"""How many samples one B-Spline training epoch draws, for a step budget.

Runs in the B-Spline venv with the same hydra arguments train.py gets, builds
the dataset the same way the workspace does, and prints one JSON line:
{"samples": N, "batch_size": B, "checkpoint_every": C}. An epoch is ceil(N / B) gradient steps.

Built rather than derived: the sampler's length comes from the B-spline fit of
each episode (chunk_bspline_trajectory counts knots the fit produced), not from
the frame count, so nothing short of the fit knows it. The zarr and sampler
caches next to the HDF5 make the second build -- training's own -- free.
"""

import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PROJECT = _REPO_ROOT / "baselines" / "bspline_policy" / "bspline_policy"
for path in (PROJECT, PROJECT.parent / "diffusion_policy"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(PROJECT / "bspline_policy" / "config"))
def main(cfg) -> None:
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(json.dumps({"samples": len(dataset), "batch_size": int(cfg.dataloader.batch_size),
                      "checkpoint_every": int(cfg.training.checkpoint_every)}), flush=True)


if __name__ == "__main__":
    main()
