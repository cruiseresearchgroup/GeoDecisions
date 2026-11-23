#!/usr/bin/env bash
# Serve KnowLine-31B-Gen1 exactly as released (serve_knowline.sh from the pinned HF snapshot: SGLang bf16 engine +
# knowline_server.py /v1/systemone front end) and run a config against it; the server is stopped afterwards.
#
#   scripts/run_knowline.sh GPU|auto [CONFIG] [run.py args...]
#
# GPU=auto waits for a GPU with <1 GiB in use on two checks 5 minutes apart (the model needs ~82 GB of a 96 GB card).
# Env: jevgeo-sglang (sglang 0.5.21, transformers 5.12.1).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?usage: run_knowline.sh GPU|auto [CONFIG] [run.py args...]}"
CONFIG="${2:-$ROOT/configs/runs/knowline-31b-gen1.yaml}"
shift $(( $# >= 2 ? 2 : 1 ))
REPO=PelaAI/KnowLine-31B-Gen1
REVISION=8f052a8130e599b53e5324c34f0f5ff90696e1c6
PORT=8080
ENV=(micromamba run -n jevgeo-sglang)

free_gpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1024 {print $1}'; }
if [[ "$GPU" == auto ]]; then
  echo "[wait] for a free GPU ($(date '+%F %T'))"
  while true; do
    first=$(free_gpus | tr '\n' ' ')
    if [[ -n "$first" ]]; then
      sleep 300
      for g in $(free_gpus); do
        if [[ " $first " == *" $g "* ]]; then GPU=$g; break 2; fi
      done
    else
      sleep 300
    fi
  done
fi
echo "[gpu] $GPU ($(date '+%F %T'))"

SNAP=$("${ENV[@]}" python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$REPO', revision='$REVISION'))" | tail -1)
LOG="$ROOT/outputs/logs/knowline_server.log"
# DeepGEMM only serves FP8 GEMMs (unused in bf16) and its JIT needs NVCC >= 12.9, which this machine lacks.
export SGLANG_ENABLE_JIT_DEEPGEMM=0
# SGLang JIT-compiles kernels with C++20: use the env's CUDA 13.0 nvcc (conda-forge cuda-nvcc), not /usr/bin/nvcc.
export CUDA_HOME="$(micromamba run -n jevgeo-sglang python -c 'import sys; print(sys.prefix)' | tail -1)"
# The HF cache stores files as symlinks into blobs/, and Python resolves the script's symlink, so the front end
# could not import knowline_layout.py next to it: run the front-end files from real copies (weights stay in $SNAP).
FRONT="$ROOT/outputs/cache/knowline_frontend/$REVISION"
mkdir -p "$FRONT"
for f in serve_knowline.sh knowline_server.py knowline_layout.py temperatures.json; do cp -L "$SNAP/$f" "$FRONT/$f"; done
# serve_knowline.sh binds the front end to $HOST, which the env's conda compiler activation sets to a target triple
# (x86_64-conda-linux-gnu); set it inside the activated env.
( cd "$FRONT" && "${ENV[@]}" env HOST=127.0.0.1 bash serve_knowline.sh "$SNAP" "$GPU" "$PORT" ) > "$LOG" 2>&1 &
SERVER=$!
trap 'pkill -P "$SERVER" 2>/dev/null; kill "$SERVER" 2>/dev/null || true' EXIT
until curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 10
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited; see $LOG" >&2; exit 1; }
done
echo "[serve] ready on :$PORT ($(date '+%F %T'))"
cd "$ROOT" && OMP_NUM_THREADS=4 "${ENV[@]}" python scripts/run.py "$CONFIG" "$@"
