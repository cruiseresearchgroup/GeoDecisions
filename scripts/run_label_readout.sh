#!/usr/bin/env bash
# Serve a base chat model with SGLang (bf16, fp32 LM head for exact label log-probabilities) and run a
# label_readout config against it; the server is stopped afterwards.
#
#   scripts/run_label_readout.sh GPU CONFIG [run.py args...]
#
# The config's model block gives repo, revision and url (http://127.0.0.1:<port>). Env: jevgeo-sglang.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?usage: run_label_readout.sh GPU CONFIG [run.py args...]}"
CONFIG="${2:?usage: run_label_readout.sh GPU CONFIG [run.py args...]}"
shift 2
ENV=(micromamba run -n jevgeo-sglang)
read -r REPO REVISION PORT < <("${ENV[@]}" python -c "
import yaml; m = yaml.safe_load(open('$CONFIG'))['model']
print(m['repo'], m['revision'], m['url'].rsplit(':', 1)[1].strip('/'))" | tail -1)
NAME=$(basename "$CONFIG" .yaml)
LOG="$ROOT/outputs/logs/${NAME}_server.log"

# Same environment fixes as scripts/run_knowline.sh: DeepGEMM (FP8 only) needs NVCC >= 12.9; SGLang's JIT kernels need
# the env's CUDA 13.0 nvcc; the env's conda compiler activation sets HOST to a target triple.
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export CUDA_HOME="$("${ENV[@]}" python -c 'import sys; print(sys.prefix)' | tail -1)"
SNAP=$("${ENV[@]}" python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$REPO', revision='$REVISION'))" | tail -1)
echo "[gpu] $GPU  [model] $REPO@$REVISION  [port] $PORT ($(date '+%F %T'))"

CUDA_VISIBLE_DEVICES=$GPU "${ENV[@]}" python -m sglang.launch_server --model-path "$SNAP" --served-model-name m \
  --tp 1 --mem-fraction-static 0.85 --enable-fp32-lm-head --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
until curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 10
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited; see $LOG" >&2; exit 1; }
done
echo "[serve] ready on :$PORT ($(date '+%F %T'))"
cd "$ROOT" && OMP_NUM_THREADS=4 "${ENV[@]}" python scripts/run.py "$CONFIG" "$@"
