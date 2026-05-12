# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TiPToP is a Task and Motion Planning (TAMP) system for robots with a modular architecture consisting of three sequential components:

1. **Perception Module** - Processes RGB images and natural language to build object-centric 3D representations using FoundationStereo (depth), Gemini Robotics-ER 1.5 (VLM), SAM-2 (segmentation), and M2T2 (grasp detection)
2. **Planning Module** - Uses cuTAMP, a GPU-parallelized TAMP solver that optimizes thousands of candidate pick-and-place plans in parallel
3. **Execution Module** - Executes trajectories using joint impedance control

More details: https://tiptop-robot.github.io/

## Development Commands

### Installation

We use [pixi](https://pixi.prefix.dev/) for environment and dependency management:

```bash
# Install all dependencies (PyTorch, cuRobo deps, etc.)
pixi install

# Install external dependencies (ZED SDK, cuRobo, cuTAMP)
pixi run setup-all
```

### Activating the Environment

All development and CLI commands must be run inside the pixi shell:

```bash
# Activate the pixi environment
pixi shell

# Now you can run all tiptop commands directly
```

### CLI Scripts

The package provides several CLI entry points (defined in `pyproject.toml`). First activate the environment with `pixi shell`, then run commands directly:

```bash
# Setup and calibration
compute-gripper-mask       # Gemini + SAM2 gripper detection

# Visualization
viz-calibration            # Rerun visualization of camera calibration
viz-gripper-cam            # View gripper camera feed
viz-scene                  # Visualize scene

# Robot control
go-home                    # Move robot to home configuration
go-to-capture              # Move robot to image capture configuration
gripper-open               # Open gripper
gripper-close              # Close gripper
```

### Documentation

Documentation tasks can be run directly with `pixi run`:

```bash
# Install Sphinx and doc dependencies
pixi run docs-install

# Build HTML documentation
pixi run docs-build

# Serve with live reload (for development)
pixi run docs-serve

# Clean build artifacts
pixi run docs-clean
```

The documentation is automatically built and deployed via ReadTheDocs using Python 3.11.

## Architecture Overview

### Package Structure

The `tiptop/` package is organized into focused modules:

- **`config/`** - Centralized configuration management using OmegaConf
  - `tiptop.yml` contains robot, camera, and perception service settings
  - `calibration_info.json` stores camera-to-end-effector transforms
  - `tiptop_cfg()` provides lazy-loaded config with CLI override support

- **`perception/`** - Perception pipeline with microservices architecture
  - `foundation_stereo.py` - Depth estimation via HTTP to remote server
  - `m2t2.py` - Grasp detection via HTTP to remote server
  - `sam2.py` - Object segmentation (local or remote)
  - `zed_camera.py` - ZED stereo camera interface
  - All services support both sync and async (`_async`) APIs

- **`scripts/`** - CLI entry points for setup, visualization, and robot control

- **`motion_planning.py`** - Motion planning using cuRobo's MotionGen

- **`warm_start.py`** - Initializes and warm-ups IK solver and MotionGen

- **`tiptop_demo.py`** - Main end-to-end demo pipeline

### Data Flow

The system follows this pipeline:

```
1. Image Capture
   └─> ZED camera (RGB-D stereo)

2. Perception (Async where possible)
   ├─> FoundationStereo (depth refinement)
   ├─> Depth-to-3D projection (point cloud)
   ├─> Gemini (object detection from RGB)
   ├─> SAM2 (object segmentation)
   └─> M2T2 (grasp generation from point cloud)

3. Task Planning
   ├─> Create TAMP environment (objects + surfaces)
   ├─> Parse instruction with Gemini
   └─> Run cuTAMP for motion plans

4. Execution
   ├─> IK solver for grasp poses
   ├─> MotionGen for collision-aware trajectories
   └─> Bamboo Franka client for robot control

5. Visualization
   └─> Rerun for real-time 3D visualization
```

### Key Integration Points

- **Bamboo Franka Controller** (`bamboo-franka-controller`) - Joint position queries and trajectory execution with impedance control
- **cuRobo** - GPU-accelerated motion planning and collision checking
- **cuTAMP** - Task-level planning with stream-based constraint solving (external dependency, not in `pyproject.toml`)
- **Rerun** - 3D visualization of robot state, sensor data, and plans
- **OmegaConf** - Configuration management with YAML files and CLI overrides

### Coordinate Frame Conventions

Transformation matrices follow the pattern `target_from_source`:
- `world_from_cam` - Transforms camera coordinates to world coordinates
- `obj_from_grasp` - Transforms grasp frame to object frame
- `gripper_from_cam` - Transforms camera frame to gripper frame

Calibration data in `calibration_info.json` stores 4x4 homogeneous transforms for camera extrinsics.

## Microservices Architecture

TiPToP uses a microservices-based architecture where external components run as separate HTTP servers:

### M2T2 (Grasp Detection)
Generates 6-DOF grasps from point clouds. Installation:

```bash
cd $TiPToP_DIR
git clone git@github.com:williamshen-nz/m2t2-private.git M2T2
cd M2T2
conda env create -n TiPToP-m2t2 python=3.10 -y
conda activate TiPToP-m2t2

# Install PyTorch matching your CUDA version (check with: nvcc --version)
# Example for CUDA 12.8:
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install pointnet2_ops/
pip install -r requirements.txt
pip install .

# Download weights
git clone https://huggingface.co/wentao-yuan/m2t2 weights
```

### FoundationStereo (Depth Estimation)
Predicts depth maps from stereo camera images (Zed). Installation:

```bash
cd $TiPToP_DIR
git clone git@github.com:williamshen-nz/FoundationStereo-private.git FoundationStereo
cd FoundationStereo
conda env create -f environment.yml
conda run -n TiPToP-foundation_stereo pip install flash-attn
conda activate TiPToP-foundation_stereo

# Download checkpoints (non-commercial version)
pip install gdown
gdown <link-url>
```

## Coding Style and Principles

### Function-based Design
- **Prefer functions over classes**: Use standalone functions and functional programming patterns where possible
- Only use classes when managing stateful operations (e.g., `ParticleInitializer`, `RolloutFunction`, `TAMPWorld`)
- Classes that act as callable function containers should implement `__call__()` to maintain functional interface

### Documentation
- Use concise single-line docstrings for simple functions
- Only add detailed docstrings when the function behavior is non-trivial or requires explanation
- Avoid redundant documentation that simply restates the function name

### Code Organization
- Break complex logic into focused, single-purpose functions
- Keep functions cohesive - extract logical units like transformation computations or data processing
- Use descriptive function names that clearly indicate purpose (e.g., `get_world_from_gripper`, `transform_pointcloud_to_world`)
- Avoid over-fragmenting code into too many tiny functions

### Naming Conventions
- Use descriptive variable names that indicate transformations: `world_from_cam`, `obj_from_grasp`, `gripper_from_cam`
- Follow the pattern `target_from_source` for transformation matrices
- Use `_fn` suffix for higher-order functions (e.g., `grasp_to_mat4x4_fn`)

### Type Hints
- Use type hints for function parameters and return values
- Leverage `jaxtyping` for tensor dimensions (e.g., `Float[torch.Tensor, "batch dim"]`)
- Define TypedDict classes for complex return structures (see `Rollout` in `rollout.py`)

### Error Handling
- Validate inputs at the start of functions with clear error messages
- Use `raise ValueError` or `raise RuntimeError` with descriptive messages
- Include context in error messages (e.g., which object/parameter caused the issue)

### Imports
- Group imports: standard library, third-party, local imports
- Use absolute imports from `tiptop` package root
- Avoid wildcard imports
- Avoid local imports inside functions for standard modules; import at module level

### String Formatting
- Always use f-strings for string formatting; avoid `%s` or `.format()`

### Code Quality Tools
- **Ruff** configured with line length 120
- Pre-commit hooks enforce import sorting (`ruff --select I --fix`) and formatting (`ruff-format`)
- Hooks only apply to `tiptop/` directory
- `pre-commit` is a pixi dependency and is registered automatically via `pixi run setup-all`

## Working Style

- **Discuss before implementing** — When the approach isn't obvious (e.g., where to put docs, how to structure config), talk through options before writing code
- **Comments and docstrings must be accurate** — Don't write comments that describe implementation details irrelevant to the reader or are vaguely wrong. If a comment doesn't add real information, drop it
- **Don't add dead code paths** — If every case goes down the same branch, don't add the other branch "just in case." Keep what's tested, remove what isn't

## Documentation Style Guide

When working on documentation:
- Be concise without making the writing feel unnatural
- Handle edge cases appropriately but not exhaustively
- Avoid excessive verbosity
- Use MyST-Parser Markdown features (colon fences, admonitions, etc.)
- Follow the existing black & white theme aesthetic
- API documentation will be auto-generated from docstrings once source code is added

## Key Concepts

The core TAMP concepts used throughout the codebase:

- **Streams** - Generate continuous values during planning (poses, grasps, paths)
- **Environment** - Manages world state including objects and obstacles
- **Robot** - Represents robot kinematics and capabilities
- **Problem** - Combines environment, robot, and goal into a complete TAMP problem
- **Solution** - Contains success status, action sequence, and cost

The planning approach is iterative: symbolic search finds candidate task plans, motion validation checks feasibility, and refinement occurs when motions fail.

---

## Running TiPToP as a baseline on robolab tasks (NVIDIA fork — `feat/our-robolab`)

This fork (`ChicyChen/tiptop @ feat/our-robolab`) adapts upstream TiPToP to
NVIDIA's `robolab` (rather than `robolab_valts`) and adds an osmo workflow so
the planner can run as an external baseline on the same LH common-sense suite
the `vlm-orchestrator` dashboard tracks. The companion M2T2 fork is
`ChicyChen/m2t2-tiptop @ master` (adds `m2t2/__init__.py` + `pixi.toml`).

### Architecture overview

```
osmo pod (isaac-lab:2.2.0 container)
├── pixi env @ $LUSTRE_DIR/.pixi              ← tiptop (Python 3.12 + cuRobo + cuTAMP)
├── pixi env @ $LUSTRE_DIR/M2T2/.pixi          ← M2T2 (Python 3.10 + CUDA 11.7 + torch 2.0)
├── M2T2 server (port 8123, tmux session)
└── per scene × per trial:
     ├── step 1: export_robolab_tiptop_h5_rgbd.py   ($ISAAC_PY, Isaac Sim)
     ├── step 2: tiptop_h5_gemini_deeper_runner.py  (pixi run, Python 3.12)
     └── step 3: replay_tiptop_robolab_manifest.py  ($ISAAC_PY, Isaac Sim)
```

### LLM backend

Single env var swaps the perception VLM:
- **`TIPTOP_LLM_BACKEND=nvidia`** (default) — `gcp/google/gemini-2.5-flash` via
  NVIDIA inference API. Free with `NVIDIA_API_KEY`, but bboxes are noticeably
  worse on synthetic Isaac Sim renders (places boxes in empty image regions
  with no depth → 0-point object meshes downstream).
- **`TIPTOP_LLM_BACKEND=gemini`** — direct Google AI Studio API. Default model
  is `gemini-robotics-er-1.6-preview` (1.5 was retired 2026-05-12). Requires
  `GOOGLE_API_KEY` (free tier: 20 req/day quota; paid tier: ~$3.60 for full
  90-episode eval). **This backend is what produced the 3/90 LH-CS number**;
  the NVIDIA backend gave 0/90 because of bbox misplacement on bowls/containers.

### Submitting an osmo run

```bash
# osmo credential set-up (one-time)
osmo credential set google-api-key --type GENERIC --payload google_api_key=$GOOGLE_API_KEY

# Full LH-CS suite (30 tasks × 3 trials = 90 episodes, 10 batches in parallel)
cd ~/tiptop
for i in 0 1 2 3 4 5 6 7 8 9; do
  osmo workflow submit osmo/run-tiptop-lh-cs.yaml \
    --pool isaac-srl-l40-04 \
    --set-string batch_name=cs-batch-$i llm_backend=gemini \
    --set trials=3
done
```

The 10 batch tuples (Infer/Kit/Recover/Sort prefixes) mirror cap-x exactly so
the dashboard buckets remain consistent. Available `batch_name` values:
`smoke` (1 scene), `cs-batch-{0..9}`. Set `llm_backend=nvidia` to bypass the
Google API and use NVIDIA inference.

### Replay-only mode (cheap re-runs)

When step 3 needs to be re-run (e.g. after a replay-script bug fix) without
re-spending Gemini quota:

```bash
osmo workflow submit osmo/run-tiptop-lh-cs.yaml \
  --pool isaac-srl-l40-04 \
  --set-string batch_name=cs-batch-$i replay_only=true \
  --set trials=3
```

For each `(scene, trial)`, the entry script globs
`$LUSTRE_DIR/tiptop-results/<scene>/*/trial_<N>/plan/*/tiptop_plan.json`,
picks the most recent, skips steps 1+2 entirely, and replays straight from
that cached plan. Zero LLM calls.

### Post-process to dashboard

After all batches complete:

```bash
# Sync Lustre → host via osmo's syncer pod (assumes syncer-amlfs04-N is running)
bash /tmp/tiptop_postprocess.sh

# Or manually:
osmo workflow port-forward syncer-amlfs04-3 syncer --port 12224:22 &
rsync -avzP -e "ssh -p 12224 -o StrictHostKeyChecking=no" \
  root@localhost:/mnt/amlfs-04/home/$USER/tiptop-results/ ~/tiptop-results/

# Build batch JSONs (in vlm-orchestrator)
cd ~/vlm-orchestrator
python scripts/build_tiptop_lh_cs_jsons.py --tiptop-root ~/tiptop-results
python scripts/build_main_results.py
```

The aggregator picks ONE trial_dir per `trial_idx` per scene — preferring a
run_ts that contains a successful plan, then falling back to the most recent.
Replay `result.json` is also searched across all sibling run_ts dirs (replay-only
runs write a fresh timestamp). Output:

- `~/vlm-orchestrator/results/lh_cs_eval/lh_cs_tiptop_batch_{0..3}.json` — per-batch
- `~/vlm-orchestrator/results/main_results.html` — dashboard (TiPToP column on the LH-CS row)

### Adapting to a new robolab task suite

The yaml's `case "{{batch_name}}"` block at `osmo/run-tiptop-lh-cs.yaml:~280`
hard-codes the 30 LH-CS scene tuples. For a new suite (e.g. `lh_vague`,
`stacking`):

1. Add new `batch_name` entries to that case statement, each binding `SCENE_ARR`
   to a 1-3 element tuple of robolab task class names. Class names must exist
   under `~/robolab/robolab/tasks/<subdir>/` and be importable.
2. If the new tasks live under a non-default subdir (e.g.
   `long_horizon/vague_intent/`), update **two** `--task-subdirs` invocations
   in the yaml: the export call (step 1) and the replay manifest call (step 3).
   The default already includes `benchmark`, `custom`,
   `long_horizon/common_sense`.
3. In `vlm-orchestrator/scripts/build_tiptop_lh_cs_jsons.py`, update
   `PREFIX_TO_BATCH` if your new tasks need a new bucketing scheme.
4. Submit + post-process exactly as above. The dashboard will show TiPToP on
   whichever eval-set it has matching `lh_<suite>_tiptop_batch_*.json` files for.

### Things that bit us (lessons learned)

The robolab fork has drifted from what upstream TiPToP was originally built
against. Each was painful to debug on osmo (~20 min per cycle). If you see
similar errors on a new tiptop submission, these are the patches in this fork:

| Symptom | Root cause | Fix location |
|---|---|---|
| `TypeError: generate_task_env_cfg() got an unexpected keyword argument 'tasks'` | factory dropped `tasks=` filter; classes are now filtered by `get_envs(task=...)` after registration | `register()` in exporter + replay |
| `ImportError: cannot import name 'get_all_env_subtask_infos'` | renamed to `get_final_subtask_info` (singular) | top of `replay_tiptop_plan_robolab.py` (try/except) |
| `AttributeError: 'ManagerBasedRLEnv' object has no attribute 'get_env_results'` | method removed; use `terminated`/`truncated` from `env.step()` | `step_action()` in `replay_tiptop_plan_robolab.py` |
| `BlockingIOError: Unable to synchronously create file ... data.hdf5` | parallel pods racing on `<robolab>/output/data.hdf5` | `ROBOLAB_OUTPUT_DIR=$TRIAL_DIR/robolab_output` env var |
| `gym.error.NameNotFound: Environment 'X' doesn't exist` | replay subprocess inherits default subdirs only | pass `--task-subdirs benchmark custom long_horizon/<your_subdir>` to the replay manifest driver |
| `429 RESOURCE_EXHAUSTED ... limit: 20` (Gemini) | free tier preview models = 20 req/day; SDK retries but quota persists | enable paid tier on Google Cloud project (~$3.60 for 90 eps) |
| `CondaToSNonInteractiveError ... main, r` | conda 25+ requires TOS on default channels | **don't use conda** — use pixi (we already do, but if you switch a sub-env, stay on pixi) |

### Headline result so far

LH-CS (30 tasks × 3 trials = 90 episodes), Robotics-ER 1.6 backend:

```
Plans succeeded (cuTAMP)        : 12 / 90 = 13.3%
Replays succeeded (end-to-end)  :  3 / 90 =  3.3%
```

For comparison, TiPToP's own paper reports 43% planning / 18% end-to-end on
the simpler robolab-120 benchmark. The LH-CS gap is plausible — those tasks
have multi-object goals, partial-completion scenes, and bowls/containers
that stress the segmentation pipeline. The `vlm-orchestrator`-side VLA
baselines (Pass, VLM+Replan+Grasp, VLM+Replan+Tools, Tool-Chain, CaP-X) run
on the same 90 episodes and land at 11-41%.
