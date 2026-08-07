#!/usr/bin/env bash
# Run TiPToP on the real Franka cell driven by franky_service.
#
# Why this wrapper exists (do not just `cd ~/tiptop && pixi run ...`):
#
#   ~/tiptop/cutamp/ is the cuTAMP SOURCE checkout and has no __init__.py, so
#   with ~/tiptop as the working directory Python treats it as a namespace
#   package and it SHADOWS the installed cuTAMP. `cutamp.__version__` then
#   disappears and tiptop's own check_cutamp_version() aborts with
#   "found <0.0.2". Running from any other directory resolves the real package.
#   This is exactly why the osmo sim eval invokes tiptop via
#   `pixi run --manifest-path ...` from $ROBOLAB_DIR.
#
# Services this expects to be up:
#   :8123  M2T2 grasp server   (cd ~/m2t2-tiptop && pixi run server -- --port 8123)
#   SAM2   runs locally in tiptop's env (no server)
#   VLM    NVIDIA inference by default; set TIPTOP_LLM_BACKEND=gemini +
#          GOOGLE_API_KEY for the paper-faithful Gemini path.
#
# Usage (then point franky_service at ws://<this-host>:8042/):
#   tiptop/franky/serve_franky.sh --plan-only     # safe first run
#   tiptop/franky/serve_franky.sh

set -euo pipefail

TIPTOP_DIR="${TIPTOP_DIR:-$HOME/tiptop}"
# NO task instruction argument: the task comes from the driver's per-frame
# `prompt`, exactly as in the cap-x real-robot service. The operator sets it at
# the robot.

# FoundationStereo is deliberately NOT used: franky_service already provides
# metric depth, matching the other baselines and our own method.
export TIPTOP_LLM_BACKEND="${TIPTOP_LLM_BACKEND:-nvidia}"
if [[ "$TIPTOP_LLM_BACKEND" == "nvidia" && -z "${NVIDIA_API_KEY:-}" ]]; then
  NVIDIA_API_KEY="$(grep -m1 '^export NVIDIA_API_KEY' "$HOME/.bashrc" | sed 's/.*="\(.*\)"/\1/')"
  export NVIDIA_API_KEY
fi

if ! (echo > /dev/tcp/127.0.0.1/8123) >/dev/null 2>&1; then
  echo "[run] FATAL: M2T2 is not listening on :8123." >&2
  echo "[run]   cd ~/m2t2-tiptop && tmux new-session -d -s m2t2 \\" >&2
  echo "[run]     'pixi run server -- --port 8123'" >&2
  exit 1
fi
echo "[run] M2T2 :8123 ready"
echo "[run] LLM backend: $TIPTOP_LLM_BACKEND"
echo "[run] task + episodes come from the driver's wire (prompt / episode_id)"

# cd to a neutral directory: see the cutamp shadowing note above.
cd /tmp
exec pixi run --manifest-path "$TIPTOP_DIR/pixi.toml" \
  python -m tiptop.franky.serve_franky \
    --output-dir "$TIPTOP_DIR/outputs/tiptop_franky" \
    "$@"
