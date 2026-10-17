#!/usr/bin/env bash
# Send one task's training data to Tillicum: SAIL's and B-Spline's HDF5s, the
# filtered recording ZAP's base diffusion policy trains on, and the code that
# job runs (sync.sh code). One password + Duo prompt covers all of it. Run on
# the workstation:
#
#   bash baselines/tillicum/send_datasets.sh <tillicum_user> ~/franka_data/baseline_prep/real/task_<dataset>
#
# The folder is what scripts/prepare_baseline_datasets.py wrote. The recording
# sent with it is the one the converters stamped into both HDF5s. Everything
# lands under /gpfs/scrubbed/<tillicum_user>; sending again resumes and skips
# what is already there.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[ $# -eq 2 ] || usage
export TILLICUM_USER=$1
DIR=$(cd "$2" && pwd)
for f in sail bspline; do
    [ -f "$DIR/$f.hdf5" ] || { echo "no $f.hdf5 in $DIR; run scripts/prepare_baseline_datasets.py first" >&2; exit 1; }
done

# The recording both files were converted from, and SAIL's precision labels.
info=$("$REPO/.venv/bin/python" - "$DIR" <<'PY'
import sys

import h5py
import numpy as np

ids = {}
for name in ("sail", "bspline"):
    with h5py.File(f"{sys.argv[1]}/{name}.hdf5", "r") as f:
        ids[name] = f.attrs.get("source_repo_id")
if None in ids.values() or ids["sail"] != ids["bspline"]:
    sys.exit(f"the two HDF5s do not name one recording: {ids}")
with h5py.File(f"{sys.argv[1]}/sail.hdf5", "r") as f:
    demos = list(f["data"].values())
    if all("absolute_actions_with_precision" in d for d in demos):
        frac = np.concatenate([d["precisions"][()].ravel() for d in demos]).mean()
        labels = f"{f['data'].attrs.get('precision_source', 'awe')}, {100 * frac:.0f}% of steps precise"
    else:
        labels = "none"
print(ids["sail"], labels)
PY
)
read -r REPO_ID LABELS <<< "$info"
echo "task:      $DIR"
echo "recording: $REPO_ID"
echo "SAIL precision labels: $LABELS"
if [ "$LABELS" = none ]; then
    echo "  SAIL will run its AWE pass on the cluster first. To label from gripper events instead:" >&2
    echo "  python -m baselines.sail_bridge.gripper_labels $DIR/sail.hdf5" >&2
fi

bash "$HERE/sync.sh" code
bash "$HERE/sync.sh" real "$DIR"
bash "$HERE/sync.sh" lerobot "$REPO_ID"

name=$(basename "$DIR")
echo
echo "Sent to ${TILLICUM_DEST:-$TILLICUM_USER@tillicum.hyak.uw.edu:/gpfs/scrubbed/$TILLICUM_USER}. Train it on the Tillicum login node:"
echo "  export WANDB_API_KEY=<your wandb key>"
echo "  bash /gpfs/scrubbed/\$USER/tillicum/submit_libero_train.sh real --tasks ${name#task_} --steps 200000 --apply \\"
echo "      -- --extra --wandb --wandb-project franka-recipe"
echo "  bash /gpfs/scrubbed/\$USER/tillicum/submit_diffusion_train.sh $REPO_ID --steps 200000 --name recipe-200k --apply \\"
echo "      -- --wandb.project=franka-recipe"
