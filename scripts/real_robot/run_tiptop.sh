#!/usr/bin/env bash
# Persistent TiPToP service for the franky_service Franka cell.
# Task + episodes come from the driver's wire (prompt / episode_id).
set -uo pipefail
export NVIDIA_API_KEY=$(grep -m1 '^export NVIDIA_API_KEY' "$HOME/.bashrc" | sed 's/.*="\(.*\)"/\1/')
# Paper-faithful spatial-reasoning model. gcp/google/gemini-2.5-flash (the
# nvidia fallback) put boxes ~90 px off the objects on this cell, which is why
# grasp association kept returning 0 grasps for the target. robotics-ER-2 lands
# within ~5-8 px. NOTE: tiptop hardcodes er-1.5-preview, which Google RETIRED;
# serve_franky.py rebinds the default to er-2-preview.
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$(grep -m1 '^export GOOGLE_API_KEY' $HOME/.bashrc | sed 's/.*="\(.*\)"/\1/')}"
export TIPTOP_LLM_BACKEND="${TIPTOP_LLM_BACKEND:-gemini}"
# Grasp depth tweak used by the sim eval (TCP offset along the approach frame).
export TIPTOP_GRASP_DEEPER_M="${TIPTOP_GRASP_DEEPER_M:-0.015}"
# Own port: 8041 belongs to the cap-x service in the driver config.
export TIPTOP_PORT="${TIPTOP_PORT:-8042}"
# :8301 is already occupied on siyi-hugo, so default to 8302 here.
export TIPTOP_MONITOR_PORT="${TIPTOP_MONITOR_PORT:-8302}"

# M2T2 must be up on :8123.
if ! (echo > /dev/tcp/127.0.0.1/8123) >/dev/null 2>&1; then
  echo "[svc] starting M2T2 on :8123"
  tmux new-session -d -s m2t2 "cd $HOME/m2t2-tiptop && $HOME/.pixi/bin/pixi run server -- --port 8123 > /tmp/m2t2_server.log 2>&1; sleep infinity"
  for i in $(seq 1 30); do
    (echo > /dev/tcp/127.0.0.1/8123) >/dev/null 2>&1 && break; sleep 5
  done
fi
(echo > /dev/tcp/127.0.0.1/8123) >/dev/null 2>&1 && echo "[svc] M2T2 :8123 ready" || { echo "[svc] FATAL: M2T2 not up"; exit 1; }

ROOT="$HOME/tiptop/outputs/tiptop_franky/$(date +%Y%m%d_%H%M%S)"
echo "[svc] TiPToP real-robot service | plan_only=${PLAN_ONLY:-false}"
echo "[svc] endpoint: ws://$(hostname -I | awk '{print $1}'):${TIPTOP_PORT}/"
echo "[svc] output:   $ROOT"
echo "[svc] dashboard: http://$(hostname -I | awk '{print $1}'):${TIPTOP_MONITOR_PORT}/"

ARGS=(--output-dir "$ROOT" --port "$TIPTOP_PORT" --monitor-port "$TIPTOP_MONITOR_PORT")
[ "${PLAN_ONLY:-false}" = "true" ] && ARGS+=(--plan-only)

# cd /tmp: ~/tiptop/cutamp/ (source checkout, no __init__.py) shadows the
# installed cuTAMP when it is the working directory.
cd /tmp
i=0
while true; do
  i=$((i+1))
  echo "[svc] ---- pass $i ($(date +%H:%M:%S)) ----"
  "$HOME/.pixi/bin/pixi" run --manifest-path "$HOME/tiptop/pixi.toml" \
    python -m tiptop.franky.serve_franky "${ARGS[@]}" 2>&1 | tee -a /tmp/tiptop_live.log
  echo "[svc] pass $i exited — restarting in 5s"
  sleep 5
done
