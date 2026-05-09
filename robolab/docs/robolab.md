# TiPToP × RoboLab-120

This directory contains everything needed to run TiPToP as a pixel-only
planner on the RoboLab-120 benchmark ([robolab_valts](https://github.com/nvlabs/robolab_valts)),
produce per-task RoboLab pass/fail metrics, and save replay videos.

TiPToP only consumes simulation **RGB**, **depth**, and derived **point
clouds** — it does **not** see RoboLab ground-truth world state. Perception
uses the canonical TiPToP stack: Gemini 2.5 Flash + SAM2 (via M2T2) for
detection, M2T2 for grasp generation, cuTAMP for TAMP.

Reported result on RoboLab-120 (external camera, +1.5 cm deeper TCP):

```text
Planning success (cuTAMP) : 52 / 120 = 43.3%
Replay success (RoboLab)  : 22 / 52  = 42.3%
End-to-end success        : 22 / 120 = 18.3%
```

See [`TIPTOP_ROBOLAB120_REPORT.md`](TIPTOP_ROBOLAB120_REPORT.md) for the
per-task table.

---

## 1. Layout

```text
tiptop/                              # upstream TiPToP
robolab/
├── install_robolab.sh              # glue installer (does not patch robolab)
├── robolab120_task_names.txt        # the 120 canonical task ids
├── scripts/
│   ├── export_robolab_tiptop_h5_rgbd.py    # RoboLab → TiPToP H5 (RGB+depth only)
│   ├── tiptop_gemini_model_patch.py        # pin Gemini 2.5 Flash
│   ├── tiptop_h5_gemini_deeper_runner.py   # single-task TiPToP H5 runner (+TCP offset)
│   ├── run_tiptop_robolab120_h5_deeper.py  # batch planner w/ mem-cap + resume
│   ├── replay_tiptop_plan_robolab.py       # single-task RoboLab replay (saves replay.mp4)
│   ├── replay_tiptop_robolab_manifest.py   # parallel, claim-locked replay driver
│   └── make_tiptop_robolab_report.py       # per-task Markdown report
└── docs/
    ├── robolab.md                          # this file
    └── TIPTOP_ROBOLAB120_REPORT.md         # headline result + per-task table
```

Nothing in RoboLab itself is patched. All of the glue lives here.

---

## 2. Prerequisites

Install these side-by-side, any layout works as long as you tell the scripts
where they are:

| Component | Repo | Needed for |
|---|---|---|
| **TiPToP** (this repo) | `TontonTremblay/tiptop` (fork) | planner, perception wrapper |
| **M2T2** | [NVlabs/M2T2](https://github.com/NVlabs/M2T2) | grasp generation (HTTP service) |
| **robolab_valts** | [nvlabs/robolab_valts](https://github.com/nvlabs/robolab_valts) | Isaac Lab simulator, the 120 tasks |
| **pixi** | https://pixi.sh | TiPToP & M2T2 environments |
| **GPU** | NVIDIA 4090 or better | M2T2 inference + Isaac sim |
| **Gemini API key** | https://ai.google.dev/gemini-api/docs/api-key | detection/grounding |

Default expected layout:

```text
<workspace>/
├── tiptop/            ← this repo (fork)
├── M2T2/
└── robolab_valts/
```

---

## 3. Install

### 3a. TiPToP

Follow the TiPToP upstream docs:

```bash
cd tiptop
pixi install
```

### 3b. M2T2 (grasp service)

```bash
git clone https://github.com/NVlabs/M2T2 ../M2T2
cd ../M2T2
pixi install
# Serve M2T2 on port 8123:
pixi run python m2t2_server.py --port 8123
```

A simpler route if you have tmux:

```bash
tmux new-session -d -s m2t2_server 'cd ../M2T2 && pixi run python m2t2_server.py --port 8123'
```

Sanity-check with `curl http://localhost:8123/health`.

### 3c. robolab_valts

Install per robolab_valts' own README. The TiPToP scripts here only need:

- a working `robolab_valts/.venv/bin/python`, and
- `robolab` importable from that venv.

No patching of the robolab_valts repo is required or performed.

### 3d. Run the bridge installer

```bash
# Default layout (tiptop, M2T2, robolab_valts side-by-side):
bash robolab/install_robolab.sh

# Or with explicit paths and Gemini key:
GOOGLE_API_KEY=sk-...   \
ROBOLAB_REPO=/abs/path/to/robolab_valts \
ROBOLAB_PYTHON=/abs/path/to/robolab_valts/.venv/bin/python \
M2T2_REPO=/abs/path/to/M2T2 \
START_M2T2=1 \
  bash robolab/install_robolab.sh
```

The installer verifies everything, writes `.env.robolab` at the TiPToP repo
root, and (with `START_M2T2=1`) starts M2T2 in a tmux session. It does not
touch the robolab_valts repo.

Then in each shell:

```bash
source .env.robolab
```

---

## 4. One-shot: run RoboLab-120 end-to-end

From the TiPToP repo root, after sourcing `.env.robolab`:

```bash
# 1. Export RoboLab observations (RGB+depth only) for all 120 tasks.
"$ROBOLAB_PYTHON" robolab/scripts/export_robolab_tiptop_h5_rgbd.py \
  --headless \
  --task $(cat robolab/robolab120_task_names.txt) \
  --output-dir tiptop_robolab120_external_h5 \
  --camera external_cam

# 2. TiPToP planning with +1.5 cm deeper TCP and 24 GB RSS cap per task.
python3 robolab/scripts/run_tiptop_robolab120_h5_deeper.py \
  --manifest tiptop_robolab120_external_h5/manifest.json \
  --output-dir tiptop_robolab120_external_deeper_outputs \
  --timeout-s 900 --max-planning-time 20 --num-particles 64 \
  --opt-steps-per-skeleton 100 --grasp-deeper-m 0.015 \
  --mem-cap-gb 24

# 3. RoboLab replay: saves replay.mp4 + result.json per task.
python3 robolab/scripts/replay_tiptop_robolab_manifest.py \
  --planning-summary tiptop_robolab120_external_deeper_outputs/summary.json \
  --output-dir tiptop_robolab120_external_deeper_replay \
  --timeout-s 900 --stride 4 --max-joint-step 0.03 \
  --gripper-steps 60 --post-steps 120 --camera both \
  --robolab-python "$ROBOLAB_PYTHON"

# 4. Combined report.
python3 robolab/scripts/make_tiptop_robolab_report.py \
  --planning-metrics tiptop_robolab120_external_deeper_outputs/metrics.json \
  --replay-metrics   tiptop_robolab120_external_deeper_replay/replay_metrics.json \
  --out TIPTOP_ROBOLAB120_REPORT.md
```

### Parallel replay

GPU permitting (~9 GB per Isaac replay, ~17 GB for two concurrent), you can
speed replay up ~2× with two workers in tmux:

```bash
tmux new-session -d -s tiptop_replay_w0 '... replay_tiptop_robolab_manifest.py ... --worker-tag w0'
tmux new-session -d -s tiptop_replay_w1 '... replay_tiptop_robolab_manifest.py ... --worker-tag w1'
```

Workers use filesystem `.claim` locks under `<output-dir>/<task>/.claim` to
avoid double-running the same task.

---

## 5. Key design decisions

- **Input constraint.** Perception reads `obs.h5` (RGB, depth, camera
  intrinsics/extrinsics) only. Ground-truth poses, names, or contact state
  from RoboLab are never read.
- **External camera.** `--camera external_cam` is preferred over wrist-cam for
  TiPToP: a single third-person RGB-D frame is enough and the grasp scene is
  less occluded.
- **+1.5 cm deeper TCP.** `TIPTOP_GRASP_DEEPER_M=0.015` is applied inside
  `tiptop_h5_gemini_deeper_runner.py` via `m2t2_to_tiptop_transform`. This
  is the only modification to grasp geometry; TiPToP's own top-scoring
  M2T2 grasp selection is unchanged.
- **Memory cap.** Pathological scenes (17+ movables) blow cuTAMP to >100 GB
  RSS. Each subprocess is wrapped in
  `systemd-run --user --scope -p MemoryMax=24GB -p MemorySwapMax=0`. Note:
  `prlimit --as=...` is **not** usable because it shrinks virtual address
  space and breaks CUDA's large VA mapping.
- **Resume.** The batch planner keeps a rolling `summary.json` and skips
  any task whose latest `metadata.json` has `planning.success=True/False`.
  Pass `--retry-failed` to force a rerun of recorded failures.
- **Replay smoothing.** `--max-joint-step 0.03` rad prevents sudden
  position-target jumps when playing back cuTAMP's sparse trajectory in Isaac
  Lab.

---

## 6. Our tiny upstream patch

`tiptop/perception/segmentation.py` — `segment_pointcloud_by_masks` used to
crash with a zero-size-array error inside
`augment_with_base_projections` whenever a masked cluster had <1 point above
the table plane. The fix: skip the object with a warning instead. This is
the only non-`robolab/` change in this fork and is safe upstream.

---

## 7. Troubleshooting

- **`ValueError: No API key was provided.`**
  `GOOGLE_API_KEY` was not set in the shell/tmux that launched the planner.
  Re-export and relaunch.
- **Random `killed / rc=137`.**
  Subprocess hit the `MemoryMax` cgroup cap. Re-run that task with
  `--mem-cap-gb 48` or `96` (check you have that much free RAM first).
- **`Warp CUDA OOM` at init.**
  You accidentally wrapped the subprocess with `prlimit --as=...` (or any
  virtual-memory limiter). Use only `systemd-run MemoryMax` as above.
- **`No result.json` after replay.**
  Isaac hit the `--timeout-s` wall. For big scenes (RubiksCubeBehindBowlTask
  etc.) bump to 1800 s, or reduce `--stride`.
- **`SKIP claimed-by-other`.**
  Another worker owns that task. Delete stale
  `<output-dir>/<task>/.claim` files if you killed a worker hard.

---

## 8. Citation

If you use this integration, please cite both TiPToP (upstream) and
robolab_valts.
