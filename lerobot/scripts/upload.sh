#!/usr/bin/env bash

# Upload a local dataset directory to Hugging Face Hub.
# $1 is repo id
# $2 is path to data

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi
  conda activate lerobot >/dev/null 2>&1 || true
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <repo_id> <path/to/data>"
    exit 1
fi
hf upload "$1" "$2" . --repo-type dataset