#!/usr/bin/env bash

# Create the interpreters the two baselines run in: ~/franka_ws/.venv-sail and
# ~/franka_ws/.venv-bspline (uv venvs, python 3.11). Idempotent; re-run after a
# submodule update.
#
#   ./scripts/setup_baseline_envs.sh            # both
#   ./scripts/setup_baseline_envs.sh sail       # one
#
# Why not the upstream conda recipes: neither exists on this machine, and both
# pin a torch (SAIL 2.1/cu118, B-Spline 2.6) with no kernels for the RTX 5090
# (sm_120 needs torch >= 2.7 / cu128). Everything else follows their recipes --
# SAIL's installation.sh, B-Spline's conda_environment.yaml -- minus what only a
# simulator or their arm needs. The robosuite patch is not applied: it patches
# the sim, and the sim is never built here.
#
# baselines/interpreters.py is what finds these afterwards; $SAIL_PYTHON /
# $BSPLINE_PYTHON override it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYVER="${BASELINE_PYTHON_VERSION:-3.11}"
TORCH_INDEX="${BASELINE_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
AWE_REPO="${AWE_REPO:-https://github.com/lucys0/awe.git}"
PYTORCH3D_REPO="${PYTORCH3D_REPO:-https://github.com/facebookresearch/pytorch3d.git@stable}"

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

venv_python() {   # <venv-dir> -> creates the venv if needed, prints its python
    local venv="$1"
    [[ -x "$venv/bin/python" ]] || uv venv --python "$PYVER" "$venv" >&2
    echo "$venv/bin/python"
}

pip() {           # <python> args... -> uv pip install into that interpreter
    local py="$1"; shift
    uv pip install --python "$py" "$@"
}

cuda_check() {    # <python> <label>
    "$1" - "$2" <<'PY'
import sys, torch
label = sys.argv[1]
ok = torch.cuda.is_available()
print(f"[{label}] python {sys.version.split()[0]}  torch {torch.__version__}  cuda {ok}")
if not ok:
    raise SystemExit(f"[{label}] torch cannot see the GPU")
cap = torch.cuda.get_device_capability()
print(f"[{label}] {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
x = torch.randn(256, 256, device="cuda")
(x @ x).sum().item()
print(f"[{label}] GPU matmul ok")
PY
}

setup_sail() {
    local py; py="$(venv_python "$REPO_ROOT/.venv-sail")"
    echo "== SAIL env: $py"
    pip "$py" --index-url "$TORCH_INDEX" torch torchvision
    # robomimic's own install_requires, by name: installing the package with
    # deps would also build egl_probe (cmake), which only iGibson uses. Its
    # diffusers pin (0.11.1) predates its own code -- diffusion_policy.py calls
    # EMAModel(parameters=...), the 0.12+ signature -- so a version that has it.
    pip "$py" "numpy==1.26.4" h5py psutil tqdm termcolor tensorboard tensorboardX \
        imageio imageio-ffmpeg matplotlib "diffusers==0.21.4" \
        scikit-learn einops pyzmq scipy opencv-python wandb robosuite
    pip "$py" --no-deps -e "$REPO_ROOT/baselines/sail"
    # AWE: waypoint_extraction plus its top-level `utils`; its own requirements
    # list a whole sim stack we do not need. Its setup.py imports pkg_resources,
    # which the isolated build's fresh setuptools no longer ships.
    pip "$py" --no-deps --no-build-isolation "git+${AWE_REPO}"
    cuda_check "$py" sail
    "$py" - <<'PY'
import robomimic, diffusers, robosuite, sklearn, zmq, h5py, cv2
from waypoint_extraction.extract_waypoints import dp_waypoint_selection
from robomimic.algo import algo_factory
import robomimic.utils.file_utils as FileUtils
print(f"[sail] robomimic {robomimic.__version__} diffusers {diffusers.__version__} robosuite {robosuite.__version__}")
print("[sail] imports ok")
PY
}

setup_bspline() {
    local py; py="$(venv_python "$REPO_ROOT/.venv-bspline")"
    echo "== B-Spline env: $py"
    pip "$py" --index-url "$TORCH_INDEX" torch torchvision
    pip "$py" "numpy==1.26.4" scipy h5py "zarr==2.12.0" "numcodecs<0.16" imagecodecs \
        "hydra-core>=1.3,<1.4" omegaconf "einops==0.4.1" "diffusers==0.11.1" \
        "huggingface_hub<0.26" dill tqdm wandb tensorboard tensorboardX opencv-python \
        scikit-video scikit-image imageio imageio-ffmpeg termcolor threadpoolctl psutil \
        click accelerate filelock pyzmq pandas numba six
    # robomimic 0.2.0 declares deps we do not want (egl_probe again); six is
    # the one it actually imports.
    pip "$py" --no-deps "robomimic==0.2.0"
    pip "$py" -e "$REPO_ROOT/baselines/bspline_policy/bspline_policy"
    # Only pytorch3d.transforms is used (DP's RotationTransformer), which is pure
    # torch; the CUDA extensions would be built against the 13.x toolkit under
    # /usr/local/cuda and refuse the cu128 torch. CPU-only build, a few minutes.
    if ! "$py" -c "import pytorch3d.transforms" 2>/dev/null; then
        PYTORCH3D_FORCE_NO_CUDA=1 MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
            pip "$py" --no-build-isolation "git+${PYTORCH3D_REPO}"
    fi
    cuda_check "$py" bspline
    "$py" - <<'PY'
import robomimic, diffusers, zarr, hydra, dill, imagecodecs, zmq, h5py, cv2
import pytorch3d.transforms
import bspline_policy
print(f"[bspline] robomimic {robomimic.__version__} diffusers {diffusers.__version__} zarr {zarr.__version__}")
print("[bspline] imports ok")
PY
    # diffusion_policy is reached through sys.path by upstream's train.py and
    # policy server, not as an installed package; check it the way they do.
    "$py" - "$REPO_ROOT/baselines/bspline_policy" <<'PY'
import sys
root = sys.argv[1]
sys.path[:0] = [f"{root}/bspline_policy", f"{root}/diffusion_policy"]
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from bspline_policy.policy.diffusion_unet_bspline_image_policy import DiffusionUnetBSplineImagePolicy
from bspline_policy.dataset.robomimic_replay_bspline_image_dataset import RobomimicReplayBSplineImageDataset
print("[bspline] diffusion_policy + bspline_policy import ok")
PY
}

case "${1:-all}" in
    sail)    setup_sail ;;
    bspline) setup_bspline ;;
    all)     setup_sail; setup_bspline ;;
    *) echo "usage: $0 [sail|bspline|all]" >&2; exit 2 ;;
esac
echo "done. baselines/interpreters.py will pick these up; see baselines/README.md."
