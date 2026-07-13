# franka-bimanual training on Tillicum

Scope: this only containerizes the **training** step (`lerobot-train` on ACT
policies against a HuggingFace dataset repo_id). None of the teleop/hardware
code (GELLO, SpaceMouse, Franka control, cameras) is included -- there's no
robot attached to a GPU cluster node, and several of those deps (libfranka,
USB/serial device access) won't build or function in a generic container.

## Outstanding TODOs before this works end-to-end

1. **Generate `requirements.txt` from your workstation** and place it in this
   directory before building:
   ```bash
   uv pip freeze > requirements.txt
   # or, if you want a cleaner resolved lockfile-style export:
   uv export --format requirements-txt --no-hashes > requirements.txt
   ```
   This file is currently MISSING -- the Dockerfile's `COPY requirements.txt`
   step will fail until it's added. Using your actual freeze output (rather
   than re-deriving from lerobot's pyproject.toml) keeps this container
   matching your already-validated v0.5.1 + Python 3.12 environment, since
   pyproject extras have changed across lerobot versions and going from
   pyproject alone should be avoided.

2. **Verify libsvtav1 in the container's ffmpeg.** The Dockerfile installs
   Ubuntu 22.04's stock ffmpeg package, which often lacks libsvtav1 (the
   codec lerobot's dataset format prefers for TorchCodec-based video
   decoding). Build the image, then run:
   ```bash
   docker run --rm <image> ffmpeg -encoders | grep libsvtav1
   ```
   If it's missing, switch to conda-forge's ffmpeg build instead (adds a
   conda dependency) or verify your dataset was encoded in a way pyav can
   read as a fallback -- confirm this matches your workstation's behavior,
   since silent fallback to a different decoder can change data loading
   subtly.

3. **Replace placeholders in `slurm/train_act_4gpu.slurm`:**
   `YOUR_GROUP_NAME`, `HYAK_PROJECT`, dataset/policy repo_ids, batch size.

4. **Set `HF_TOKEN` and `WANDB_API_KEY`** as environment variables in your
   Tillicum shell (e.g. in a project-space dotfile, NOT your public dotfiles
   or the repo) before submitting -- interactive `hf login` / `wandb login`
   won't work in a non-interactive batch job.

5. **Batch size semantics changed for multi-GPU.** `accelerate` uses
   effective batch size = `per_gpu_batch_size * num_gpus`. If you want the
   same effective batch size you used on 1 GPU, divide by 4 before passing
   it as `$3`. LeRobot does not auto-scale learning rate for you either --
   see the comments in `train_multigpu.sh`.

## Build & deploy flow

```bash
# 1. Local build + smoke test (on your Ubuntu workstation)
docker build -t <you>/lerobot-act-train:v0.5.1 .
docker run --rm --gpus all <you>/lerobot-act-train:v0.5.1 \
  ffmpeg -encoders | grep libsvtav1   # sanity check per TODO #2
docker push <you>/lerobot-act-train:v0.5.1

# 2. On Tillicum (never on the login node -- request a debug node first)
ssh <UWNetID>@tillicum.hyak.uw.edu
salloc --qos=debug --gres=gpu:1 --cpus-per-task=8 --mem=200G --time=00:30:00
cd /gpfs/projects/<your-group>/
apptainer pull lerobot-train.sif docker://atsai06/lerobot-act-train:v0.5.1
exit   # release the debug allocation

# 3. Copy code + submit
scp scripts/train_multigpu.sh <UWNetID>@tillicum.hyak.uw.edu:/gpfs/projects/<group>/scripts/... (already baked into image, this is only needed if you iterate without rebuilding)
sbatch slurm/train_act_4gpu.slurm
squeue -u $USER
```
