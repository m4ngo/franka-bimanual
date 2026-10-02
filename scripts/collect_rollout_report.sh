#!/usr/bin/env bash

# Gather the PNG summary sheets and the rollout videos for a set of sweeps into
# one directory, so a run can be handed over as a single folder.
#
#   ./scripts/collect_rollout_report.sh <out-dir> <sweep-id> [<sweep-id> ...]
#
# Videos are copied as <sweep>/<task>/<original name>; the sheets and each
# sweep's rollout_summary.py table (<sweep>.summary.txt) land at the top level
# beside a manifest of what came from where.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ $# -ge 2 ]] || { echo "usage: $0 <out-dir> <sweep-id> [<sweep-id> ...]" >&2; exit 1; }
OUT="$1"; shift
mkdir -p "$OUT"

python "$REPO_ROOT/scripts/rollout_report.py" --sweep "$@" --out-dir "$OUT"
for sweep in "$@"; do
    python "$REPO_ROOT/scripts/rollout_summary.py" --sweep "$sweep" > "$OUT/$sweep.summary.txt"
done

python - "$REPO_ROOT" "$OUT" "$@" <<'PY'
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from baselines.run_record import osc_settings
out, sweeps = Path(sys.argv[2]), sys.argv[3:]
root = Path.home() / "franka_data" / "outputs"
index, total = [], 0
for m in sorted(root.rglob("manifest.json")):
    doc = json.loads(m.read_text())
    sweep = (doc.get("run") or {}).get("sweep")
    if sweep not in sweeps:
        continue
    task = doc["environment"]["task"]
    vids = sorted((m.parent / "videos").glob("*.mp4"))
    # Split by verdict: the rollout names each file with its outcome, and the
    # successful ones are what anyone actually wants to watch.
    n_ok = 0
    for v in vids:
        verdict = "success" if v.stem.endswith("_success") else "timeout"
        dest = out / "videos" / sweep / task / verdict
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v, dest / v.name)
        n_ok += verdict == "success"
    total += len(vids)
    index.append({"sweep": sweep, "method": doc["run"]["method"], "task": task,
                  "run": doc["run"]["run_id"],
                  "run_dir": str(m.parent), "osc": osc_settings(doc["environment"]),
                  "episodes": doc["summary"].get("episodes"),
                  "successes": doc["summary"].get("successes"),
                  "success_rate": doc["summary"].get("success_rate"),
                  "mean_time_to_success_s": doc["summary"].get("mean_time_to_success_s"),
                  "median_time_to_success_s": doc["summary"].get("median_time_to_success_s"),
                  "ee_force_n": doc["summary"].get("ee_force_n"),
                  "videos": len(vids), "videos_success": n_ok,
                  "videos_timeout": len(vids) - n_ok})
(out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
ok = sum(r["videos_success"] for r in index)
print(f"{len(index)} runs, {total} videos ({ok} success / {total - ok} timeout) -> {out}")
PY

echo "collected into $OUT"
ls -1 "$OUT"
