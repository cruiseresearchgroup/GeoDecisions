#!/usr/bin/env bash
# Create the two micromamba environments used by this repo.
#
#   jevgeo       Python 3.11: data, metrics, reports, tests, downloads (envs/jevgeo.yml)
#   jevgeo-pplx  Python 3.12: pplx-decider-v1.2 inference (envs/pplx-decider.lock.txt, torch 2.14+cu130)
#
# The pplx-decider repo is private: run `hf auth login` (or set HF_TOKEN) with an account that has
# access first. Weights (~49 GB) download into the Hugging Face cache on first use.
#
# Usage: scripts/setup_env.sh [core|pplx|all]   (default all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHAT="${1:-all}"
MM="${MAMBA_EXE:-micromamba}"
PPLX_REPO=perplexity-ai/pplx-decider-v1.2
PPLX_REVISION=cce0f3f80033e9d89d80831d5281fd72124aacce  # snapshot the results were produced with

if [[ "$WHAT" == core || "$WHAT" == all ]]; then
  "$MM" create -y -f "$ROOT/envs/jevgeo.yml"
  "$MM" run -n jevgeo pip install -e "$ROOT" --no-deps
fi

if [[ "$WHAT" == pplx || "$WHAT" == all ]]; then
  "$MM" create -y -n jevgeo-pplx -c conda-forge python=3.12 pip
  "$MM" run -n jevgeo-pplx pip install -r "$ROOT/envs/pplx-decider.lock.txt"
  "$MM" run -n jevgeo-pplx pip install -e "$ROOT" --no-deps
  # The inference package ships inside the model snapshot (source/), not on PyPI.
  snap=$("$MM" run -n jevgeo-pplx python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$PPLX_REPO', revision='$PPLX_REVISION'))")
  "$MM" run -n jevgeo-pplx pip install --no-deps "$snap/source"
  "$MM" run -n jevgeo-pplx python -c "import torch, pplx_decider; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi
