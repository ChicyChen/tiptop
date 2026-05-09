# TiPToP 🎩🤖

[![arXiv](https://img.shields.io/badge/arXiv-2603.09971-b31b1b.svg)](https://arxiv.org/abs/2603.09971)
[![Documentation](https://readthedocs.org/projects/TiPToP-robot/badge/?version=latest)](https://TiPToP-robot.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A Modular Open-Vocabulary Planning System for Robotic Manipulation**

[Website](https://tiptop-robot.github.io/) | [Documentation](https://tiptop-robot.readthedocs.io/) | [Paper](https://arxiv.org/abs/2603.09971)

<p align="center">
  <img src="docs/_static/logo-light.png" alt="TiPToP Logo" width="200">
</p>

TiPToP solves complex real-world manipulation tasks directly from raw pixels and natural-language commands by combining Task-and-Motion Planning with perception and language models through inference-time search — with zero robot training data.

## Getting Started

See the [documentation](https://tiptop-robot.readthedocs.io/) for installation, setup, and usage instructions.

## RoboLab-120 integration (this fork)

This fork adds a **self-contained TiPToP × RoboLab** bridge under [`robolab/`](robolab/).
It lets you evaluate TiPToP end-to-end on the 120-task RoboLab benchmark using
only **RGB + depth + camera calibration** from the simulator — no ground-truth
world state is read. Perception uses the canonical TiPToP stack (Gemini 2.5
Flash + SAM2 + M2T2 + cuTAMP).

Reported result on RoboLab-120 (external camera, +1.5 cm deeper TCP):

```text
Planning success (cuTAMP) : 52 / 120 = 43.3%
Replay success (RoboLab)  : 22 / 52  = 42.3%
End-to-end success        : 22 / 120 = 18.3%
```

Per-task table: [`robolab/docs/TIPTOP_ROBOLAB120_REPORT.md`](robolab/docs/TIPTOP_ROBOLAB120_REPORT.md).

### Install the bridge

Clone TiPToP (this fork), [M2T2](https://github.com/NVlabs/M2T2), and
[robolab_valts](https://github.com/nvlabs/robolab_valts) side-by-side, install
each per its own README, then:

```bash
# From the TiPToP repo root
bash robolab/install_robolab.sh
# Or with explicit paths:
GOOGLE_API_KEY=sk-... \
ROBOLAB_REPO=/abs/path/to/robolab_valts \
ROBOLAB_PYTHON=/abs/path/to/robolab_valts/.venv/bin/python \
M2T2_REPO=/abs/path/to/M2T2 \
START_M2T2=1 \
  bash robolab/install_robolab.sh

source .env.robolab
```

> The installer does **not** modify the robolab_valts repo — all bridge code
> lives in `robolab/` here.

### Run RoboLab-120 end-to-end

```bash
# 1. Export RGB+depth observations for all 120 tasks.
"$ROBOLAB_PYTHON" robolab/scripts/export_robolab_tiptop_h5_rgbd.py \
  --headless --task $(cat robolab/robolab120_task_names.txt) \
  --output-dir tiptop_robolab120_external_h5 --camera external_cam

# 2. Plan with TiPToP (+1.5cm deeper TCP, 24 GB RSS cap per task).
python3 robolab/scripts/run_tiptop_robolab120_h5_deeper.py \
  --manifest tiptop_robolab120_external_h5/manifest.json \
  --output-dir tiptop_robolab120_external_deeper_outputs \
  --timeout-s 900 --max-planning-time 20 --num-particles 64 \
  --opt-steps-per-skeleton 100 --grasp-deeper-m 0.015 --mem-cap-gb 24

# 3. Replay plans in RoboLab, saving replay.mp4 and result.json per task.
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

Full design notes, parallel-replay recipe, and troubleshooting are in
[`robolab/docs/robolab.md`](robolab/docs/robolab.md).

### The only upstream patch in this fork

`tiptop/perception/segmentation.py`: `segment_pointcloud_by_masks` now skips
objects with fewer than 10 points above the table plane instead of crashing
inside `augment_with_base_projections` with a zero-size-array error. Safe to
merge upstream.


## Building the Docs

The documentation is hosted at [tiptop-robot.readthedocs.io](https://tiptop-robot.readthedocs.io/) and automatically rebuilds on every push to `main`. To build and serve it locally for previewing changes:

```bash
pixi run docs-install   # Install doc dependencies
pixi run docs-build     # Build HTML docs
pixi run docs-serve     # Serve with live reload
```

## Contributing

See the [Contributing Guide](https://tiptop-robot.readthedocs.io/en/latest/contributing) for development setup and guidelines.

## Citation

```bibtex
@article{shen2026tiptop,
    title={{TiPToP}: A Modular Open-Vocabulary Planning System for Robotic Manipulation},
    author={Shen, William and Kumar, Nishanth and Chintalapudi, Sahit and Wang, Jie and Watson, Christopher and Hu, Edward S. and Cao, Jing and Jayaraman, Dinesh and Kaelbling, Leslie Pack and Lozano-P\'{e}rez, Tom\'{a}s},
    journal={arXiv preprint arXiv:2603.09971},
    year={2026}
}
```
