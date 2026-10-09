#!/usr/bin/env bash
# Submit the Primus smoke test.   scripts/run_smoke.sh <nodes> [KEY=VAL ...]
#   scripts/run_smoke.sh 1
#   scripts/run_smoke.sh 2 GLOBAL_BATCH_SIZE=512     # keep per-GPU batch fixed: gbs = 256 x nodes
# Prints the job id and log path. Logs and results land in <repo>/runs/.
set -eu
SK=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$SK/env.lux.sh"
N=${1:?usage: run_smoke.sh <nodes> [KEY=VAL ...]}; shift || true
for kv in "$@"; do export "$kv"; done
mkdir -p "$SK/runs"; cd "$SK/runs"
JID=$(sbatch --parsable -A "$ACCOUNT" -N "$N" -J "smoke-N$N" "$SK/scripts/submit_primus_smoke.sbatch" | tail -1)
echo "job $JID  log $SK/runs/smoke-N${N}_${JID}.log"
