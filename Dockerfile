FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Prevent apt from trying to prompt interactively (e.g. tzdata's timezone/city
# picker, pulled in transitively). docker build has no TTY, so an unanswered
# prompt just hangs forever rather than failing loudly.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- System deps -----------------------------------------------------------
# Ubuntu 22.04 ships Python 3.10 by default, but the workstation env (and
# lerobot v0.5.1 as installed there) uses Python 3.12. Pull 3.12 from
# deadsnakes rather than relying on the distro default, so behavior matches
# what's already validated locally.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        xz-utils \
        git \
        cmake \
        build-essential \
        pkg-config \
    && DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Install a static ffmpeg build directly (bypassing apt's ffmpeg package).
# This avoids relying on Ubuntu 22.04's ffmpeg, which frequently lacks
# libsvtav1 support anyway, and sidesteps any apt-layer corruption issues.
# johnvansickle.com builds are the standard trusted static ffmpeg source.
RUN curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
        -o /tmp/ffmpeg-static.tar.xz \
    && tar -xf /tmp/ffmpeg-static.tar.xz -C /tmp \
    && mv /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg \
    && mv /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg-static.tar.xz /tmp/ffmpeg-*-amd64-static

# Sanity check that ffmpeg runs at all. Note: libsvtav1 (AV1 *encoding*) is
# NOT required here -- this container only trains against already-recorded
# datasets, so it only ever *decodes* video. AV1 decoding uses libdav1d,
# which this static build includes (see `--enable-libdav1d` in the ffmpeg
# -version config line). libsvtav1 would only matter if this container were
# also recording/re-encoding datasets, which it isn't.
RUN ffmpeg -version

# --- uv ----------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# --- Python env ----------------------------------------------------------
WORKDIR /workspace
RUN uv venv --python 3.12 /workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/workspace/.venv"

# --- lerobot + dependencies ------------------------------------------------
# requirements.txt should be generated on your workstation via:
#   uv pip freeze | grep -v -E "^-e file:///" > requirements.txt
# This STRIPS all editable local-path installs (lerobot itself, plus the
# hardware wrapper packages like lerobot_camera_arv, lerobot_teleoperator_gello,
# etc. -- none of which exist inside the container or are needed for training).
# Everything else (torch, transformers, wandb, etc.) stays pinned to match
# your validated workstation environment.
COPY requirements.txt /workspace/requirements.txt
RUN uv pip install -r /workspace/requirements.txt

# lerobot itself is installed explicitly here (not from the freeze file,
# since on your workstation it's an editable local clone at
# /home/franka/lerobot which doesn't exist in this build context).
# Pinned to the exact tag you validated locally. --no-deps is used because
# requirements.txt (above) already contains every dependency version lerobot
# actually resolved to on your workstation -- letting this re-resolve deps
# could silently upgrade/change torch, transformers, etc. away from your
# validated versions.
RUN uv pip install --no-deps "lerobot[training] @ git+https://github.com/huggingface/lerobot.git@v0.5.1"

RUN uv pip install wandb

# --- Your training code -----------------------------------------------------
COPY scripts/train.sh /workspace/scripts/train.sh
RUN chmod +x /workspace/scripts/train.sh

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
# No ENTRYPOINT: this lets `docker run <image> lerobot-train --help` (or any
# other command) execute directly via PATH lookup, rather than being passed
# as an argument TO bash (which was previously misinterpreting lerobot-train
# as a shell script instead of running its #!/.../python3 shebang).