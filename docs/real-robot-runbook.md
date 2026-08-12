# Real-robot runbook: TiPToP on the franky_service Franka cell

How to run TiPToP as a real-robot baseline, and the failure modes that have
actually bitten us. The CaP-X half of the same cell is documented in
`cap-x/docs/real-robot-runbook.md`; both baselines run **simultaneously** on one
machine and one GPU.

Verified on `siyi-hugo` (NVIDIA L40S, 47.5 GB, driver 550).

---

## 1. Endpoints

| baseline | driver connects to | dashboard (browser) |
|---|---|---|
| **TiPToP** | `ws://<host>:8042/` | `http://<host>:8302/` |
| CaP-X | `ws://<host>:8041/` | `http://<host>:8300/` |

`8042` is a **WebSocket** port; a browser gets `HTTP 426`, which is correct. Only
`8302` is browsable. Default dashboard is 8301 upstream, moved to **8302** here
because 8301 was already taken on this host.

M2T2 serves on `8123` and is started by the launcher if absent.

---

## 2. Bring-up

```bash
tmux new-session -d -s tiptop \
  "PLAN_ONLY=false TIPTOP_PORT=8042 TIPTOP_MONITOR_PORT=8302 \
   ~/baseline-launchers/run_tiptop.sh > /tmp/tiptop.log 2>&1"
```

`PLAN_ONLY=true` plans but never moves the arm — the right first run on a new
cell. Shut down with `tmux kill-session -t tiptop; pkill -9 -f serve_franky`.

Each `tmux new-session` must be its own SSH invocation: combining a `pkill` and a
`new-session` in one remote command kills the session you just made.

**The environment cannot be built from `pip freeze`.** Use pixi:
`pixi install`, then `pixi run install-cutamp` **and** `pixi run install-curobo`
(the second is easy to forget and yields `ModuleNotFoundError: No module named
'curobo'`).

> **Never run tiptop with `~/tiptop` as the working directory.** `~/tiptop/cutamp/`
> is the cuTAMP source checkout with no `__init__.py`, so it shadows the installed
> package as a namespace package, `cutamp.__version__` disappears and
> `check_cutamp_version()` aborts with "found <0.0.2". The launcher does `cd /tmp`
> first and invokes `pixi run --manifest-path ~/tiptop/pixi.toml`.

---

## 3. The VLM must be robotics-ER

TiPToP's grounding quality decides everything downstream. Measured on a real
snapshot from this cell (672×376 px):

| model | bowl box | error |
|---|---|---|
| *(actual)* | x 340–385, y 185–230 | — |
| `gemini-robotics-er-2-preview` | x 336–381, y 189–224 | **~5 px** |
| `gcp/google/gemini-2.5-flash` | x 428–489, y 136–162 | **~90 px** |

That ~90 px error is why grasp association kept returning **0 grasps** for target
objects: the mask lands on empty table, so no grasp falls within
`contact_threshold_m` (1 cm). ER-2 also produced the correct goal
`on(banana, bowl)`.

```bash
export GOOGLE_API_KEY=...            # Google AI Studio, paid tier recommended
export TIPTOP_LLM_BACKEND=gemini     # default in the launcher
export TIPTOP_GEMINI_MODEL=gemini-robotics-er-2-preview   # optional override
```

> `tiptop/perception/gemini.py` hardcodes `model_id="gemini-robotics-er-1.5-preview"`,
> which Google has **RETIRED** ("no longer available. Please update your code").
> `serve_franky.py::apply_llm_backend_patch` rebinds the default to
> `gemini-robotics-er-2-preview` without editing shared tiptop code.

**Paper note:** the published TiPToP used ER-1.5, which no longer exists. We run
ER-2, a *newer* model. State that substitution explicitly in any write-up.

Without `GOOGLE_API_KEY` the launcher falls back to `TIPTOP_LLM_BACKEND=nvidia`
(`gcp/google/gemini-2.5-flash`), whose grounding is not good enough on this cell.
Robotics-ER is **not** reachable through the NVIDIA endpoint — our key returns
`key not allowed to access model ... models=['default-models']`.

---

## 4. What TiPToP can and cannot be asked

cuTAMP has a rich fluent set (`Holding`, `On`, `ButtonPushed`, …), but TiPToP's
wrapper exposes **only** `on(movable, surface)` and hardcodes `HandEmpty` into
every goal (`tiptop_run.py:241`). Consequences:

* ✅ `"put the banana in the bowl"` — a placement goal, expressible.
* ❌ `"pick up the object"` — no destination, so the VLM returns **no atoms**,
  cuTAMP has nothing to satisfy, and every attempt fails identically with
  `All 1 plan skeleton(s) failed particle initialization`.

This is a scope limitation of the baseline, not a bug: TiPToP is a
**rearrangement** planner. Bare pick tasks are outside it by construction.

Objects **inside containers** are the hard case even with good grounding: M2T2
proposes grasps on dominant geometry (a large bin absorbed 985 of them) while a
small object in the bin gets 0 within threshold.

---

## 5. Episode semantics

* Task and episode come from the **wire** (`prompt` and `episode_id` on every
  frame), never from a CLI flag.
* One process serves many episodes: exiting per episode would close the port, and
  `openpi_client` never reconnects.
* An episode is retried until the driver ends it. Cap with
  `--max-attempts-per-episode N` (0 = unlimited). Execution failures are bounded
  at 2 so a faulted arm is not commanded repeatedly.
* **Drivers that omit `episode_id`** get a synthetic per-connection id keyed on
  the task text (`auto<HHMMSS>-<n>`, a string so it cannot collide with a numeric
  id). Without this, a second robot's events were filed as "attempt 5" of an
  11-hour-old episode 202 from a different robot.
* Dashboard rows are keyed on `(session, episode_id)`, because drivers restart
  their counters and ids ARE re-used.

A driver that sends no depth / `camera_K` / `camera_extrinsic` / native-resolution
RGB **cannot run TiPToP at all** — the H5 snapshot cannot be built, so perception
never runs and no tool views appear. It fails loudly rather than silently.

---

## 6. Output layout

```
outputs/tiptop_franky/<stamp>/
├── episodes.jsonl                one row per attempt
├── episode_007/
│   ├── episode.json
│   └── attempt_1/
│       ├── observation.h5        the exact snapshot the planner saw
│       ├── attempt.json          status, task, timings, q_at_capture
│       └── plan/<ts>/
│           ├── tiptop_plan.json  the executable plan
│           ├── metadata.json     grounded_atoms, failure_reason, durations
│           ├── bboxes_viz.png    tiptop's own VLM-box render
│           ├── masks_viz.png     tiptop's own SAM2 render
│           ├── perception/       bboxes.json masks.npz grasps.pt pointcloud.ply
│           └── cutamp/           optimiser traces + metrics
└── no_episode_trials/
```

Statuses: `executed`, `plan_only`, `no_plan_found`, `planning_error: …`,
`execution_stopped: …`.

> **`executed` means the trajectory ran without raising — NOT that the task
> succeeded.** There is no ground-truth success signal on hardware; episodes are
> scored **manually by the operator**. `episode.json` repeats this warning.

---

## 7. Dashboard stage views

`http://<host>:8302/` shows, per attempt: the snapshot (RGB + depth heat-map +
valid-depth fraction), **VLM boxes**, **SAM2 masks**, **M2T2 grasps with
kept/discarded association** (green = within threshold, red = discarded), and the
**chosen grasp** with its approach axis.

Two failures are loud red rows rather than log noise:

* `NO GOAL PREDICATE: TiPToP only supports on(object, surface)…`
* `NO GRASPS ASSOCIATED with any object…` — usually a misplaced mask

Events live **in memory**: restart the service for a clean slate.

---

## 8. Known-bad states

| what you see | actual cause |
|---|---|
| driver: `no close frame received or sent` | service restarted (restart the driver process) or it crashed — read `/tmp/tiptop.log` |
| `atoms=0` + `failed particle initialization` (0.1 s) | the task is not expressible as `on(a, b)` |
| `No satisfying particles found` after ~60 s | a real search failure; usually the target object has 0 associated grasps |
| `Shrunk OBB for X has half extents <= 0` | the VLM labelled a small object as a placement SURFACE; retryable (it is data-dependent) |
| every object `No grasps within threshold` | masks are in the wrong place — check the VLM model is ER-2 |
