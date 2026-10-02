# The SAIL and B-Spline training environments, for Tillicum
# (baselines/LIBERO_SIM.md, "Training on Tillicum"). Build from the repo root:
#
#   docker build -f baselines/tillicum/Dockerfile -t atsai06/franka-baselines:$(git describe --always --dirty) .
#
# scripts/setup_baseline_envs.sh defines both environments; this only runs it,
# pinned to the workstation's versions by *-constraints.txt. Regenerate those
# after changing either venv:
#
#   for b in sail bspline; do uv pip freeze --python .venv-$b/bin/python --exclude-editable \
#       | grep -v ' @ ' | sed 's/+cu[0-9]*$//' > baselines/tillicum/$b-constraints.txt; done

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
# git and a compiler build AWE and pytorch3d; opencv and robosuite load the GL libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential libgl1 libglib2.0-0 libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/
# apptainer runs as the submitting user, who cannot read /root.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
# torch's CUDA wheels are up to 700 MB; the default 30 s read timeout drops them.
ENV UV_HTTP_TIMEOUT=600

WORKDIR /opt/franka_ws
COPY scripts/setup_baseline_envs.sh scripts/
COPY baselines/tillicum/sail-constraints.txt baselines/tillicum/bspline-constraints.txt baselines/tillicum/
COPY baselines/sail baselines/sail
COPY baselines/bspline_policy baselines/bspline_policy
# One layer, so both venvs hardlink one copy of torch out of the uv cache.
RUN BASELINE_GPU_CHECK=0 UV_CONSTRAINT=baselines/tillicum/sail-constraints.txt \
        scripts/setup_baseline_envs.sh sail \
    && BASELINE_GPU_CHECK=0 UV_CONSTRAINT=baselines/tillicum/bspline-constraints.txt \
        scripts/setup_baseline_envs.sh bspline \
    && rm -rf /root/.cache

# After the environments, so a code change does not rebuild them.
COPY baselines/ baselines/

ENV MUJOCO_GL=egl PYTHONUNBUFFERED=1
