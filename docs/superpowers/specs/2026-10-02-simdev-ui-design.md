# SimDev UI: Web Front-End for the Pipeline — Design

**Status: approved in conversation, spec awaiting review.** Written 2026-10-02.

A browser app, served from the Z8, that lets the pipeline be driven from any
Windows laptop or workstation: upload CAD for new design iterations and new
driving states, queue runs from presets, watch them live, and see why one
failed — without an SSH session.

---

## 1. Why this work exists

Today a run is `simdev run cases/car/config.yaml --run-dir ~/runs/x --profile
… --set …` typed into a shell on the Z8, a new driving state is a hand-edited
YAML block plus a CAD folder copied over by hand, and "how is it going" is
`tail -f logs/log.simpleFoam`. That works from a terminal on the machine and
from nowhere else.

The user works from several Windows machines, at the lab and at home. What
they need is:

- pick a design iteration and a driving state (cornering or straight) from
  what exists, adjust a handful of values, and queue it;
- queue several runs and have them run one after another unattended;
- see the current state of every run, live, including force convergence;
- see errors in plain terms, with the log lines that matter;
- upload new Body/Wing CAD for a design iteration, and new attitude CAD for a
  driving state — and fix an upload filed under the wrong state.

**Success:** queue tonight's runs from a laptop, check on them from a
different machine at home, and understand a failure without opening a shell.

---

## 2. Decisions taken

Settled with the user before writing this; recorded so they are not
relitigated.

| Decision | Choice |
|---|---|
| Where it runs | Web app on the Z8, opened in a browser (§5) |
| Users / auth | Single user, no logins (§5.4) |
| Remote access | Tailscale only; bound to the tailnet address, never public (§5.4) |
| Per-run editing | Preset + small override set, not a full spec editor (§4.3) |
| Concurrency | One run at a time, built on a core budget so parallel runs are a setting (§5.2) |
| CAD model | Driving state = attitude parts; design = Body+Wing exported *per state* (§3) |
| v1 extras | Live force/residual plots and image gallery. Result and image comparison, notifications and sweeps are out (§9) |

---

## 3. CAD library

### 3.1 The finding that shaped it

The attitude of a driving state — body slip, steer, camber, ride height, roll
— is baked into the exported CAD, *including Body.step*. A body moves with
the state. So a design iteration cannot be one Body.step reused across
states: it must be exported once per state it will run in. Design × state is
not a free cross product; a pair exists only where the files exist.

### 3.2 Layout

Under the existing git-ignored `CAD/` directory:

```
CAD/
  states/
    <state>/
      Chassis.step
      SUS_FL.step  SUS_FR.step  SUS_RL.step  SUS_RR.step
      Tire_FL.step Tire_FR.step Tire_RL.step Tire_RR.step
      MRF_FL.step  MRF_FR.step  MRF_RL.step  MRF_RR.step
      state.yaml
  designs/
    <design>/
      <state>/
        Body.step
        Wing.step
```

Part groups, fixed in code as two constants:

| Group | Parts | Count |
|---|---|---|
| State (attitude) | Chassis, SUS_*, Tire_*, MRF_* | 13 |
| Design (aero) | Body, Wing | 2 |

Together these are the 15 part names the car case already expects.

`state.yaml` holds what today sits in one `driving_states:` entry, minus
`geometry.source_dir`:

```yaml
description: 4 m radius at 15 m/s, left
flow:    {u_inf: 15.0}
physics: {mode: cornering, corner_radius: 4.0, corner_direction: left}
domain:  {kind: annulus}
ground:  {motion: static}
```

Names (`<state>`, `<design>`) are restricted to `[a-z0-9][a-z0-9-]*` so they
are safe as directory names, in URLs and in run names.

### 3.3 Operations

`simdev/cad/library.py`, independent of the UI so the CLI uses it too.

States:

- **create** — name, 13 attitude parts, parameters. Validated (§3.4) before
  anything is written; written to a temp directory and renamed into place so
  a failed upload never leaves a half state.
- **edit parameters**, **replace parts** (any subset of the 13), **rename**.
- **delete** — refused while any design slot exists for the state; the error
  lists those slots.

Designs:

- **create** — name, then one or more *slots*, each = (state, Body, Wing).
- **add slot** to an existing design.
- **move slot** to another state — fixes an upload filed under the wrong
  state. If the target slot exists, the UI asks first and the API requires
  `overwrite=true`.
- **replace** Body and/or Wing in a slot.
- **delete slot**, **delete design**, **rename design**.

Renaming or deleting a state or design never touches existing runs: runs
carry their own copy of the geometry (§3.5).

### 3.4 Upload validation

Rejected with a message naming the file and the problem:

- a required part missing, or an unexpected file name (case-insensitive match
  is accepted and normalised, as `prepare` already does);
- a file that does not open as STEP (checked with the gmsh import already used
  by `simdev.geometry.step`);
- `state.yaml` that fails schema validation when merged over `config.yaml`
  and the `dev` profile — the same pydantic path a run takes, so a state that
  uploads is a state that resolves.

### 3.5 Assembly into a run

At `prepare`, the 13 state parts and the slot's 2 design parts are **copied**
into `<run_dir>/cad/`, and `geometry.source_dir` points there. Copying (≈7 MB
per run, negligible beside the mesh) means editing or deleting library CAD
cannot alter or break a past run. The STEP tessellation cache is keyed by
file content, so reassembling the same files does not re-tessellate.

`caseSpec.json` records `cad.design`, `cad.state` and a SHA-256 per part, so
a result is traceable to the exact Body.step it came from even after the
library has changed.

### 3.6 Changes to the case and the CLI

- `cases/car/config.yaml` loses its `driving_states:` table and
  `driving_state:` selector. A one-off script,
  `scripts/migration/split_driving_states.py`, moves the current `testcase`
  and `straight` entries to `CAD/states/<name>/state.yaml`, moves their 13
  attitude parts into the state, and their Body/Wing into
  `CAD/designs/baseline/<name>/`.
- `resolve()` gains an optional `state` layer, merged after the case file and
  before `--set` overrides. `apply_driving_state()` and its tests are removed.
- `simdev run` / `prepare` gain `--design NAME --state NAME` (and
  `--cad-root`, default `<repo>/CAD`). Given those, prepare assembles
  `<run_dir>/cad/` and the CLI behaves exactly as the app does. Without them
  the case's own `geometry.source_dir` is used, so Ahmed and other cases are
  unaffected.

---

## 4. The run lifecycle

### 4.1 A job

One job = one run directory = this command sequence, run by the worker:

```
simdev run  <case> --run-dir <dir> --profile <p> --design <d> --state <s> [--set k=v ...]
simdev images <dir>
```

`images` runs only if `run` exited and `status/post.json` exists. `run`
exiting 1 because a gate failed is not a crash: the job continues to images
and finishes as `gate_failed` (§6).

### 4.2 Run directory

`~/runs/<run-name>`, root configurable. Run name defaults to
`<design>-<state>-<profile>-<NNN>` with NNN the next free number, and is
editable. Queuing onto a name whose directory already exists is refused.

### 4.3 Overrides

The New Run form offers exactly these, pre-filled from the state and profile:

| Field | Applies when |
|---|---|
| `flow.u_inf` | always |
| `physics.corner_radius` | cornering |
| `physics.corner_direction` | cornering |
| `solve.n_ranks` | always |
| `solve.max_iterations` | always |

Only fields the user changed are sent, as `--set`, so `caseSpec.json` shows
which values were overrides. The cornering/straight mode itself is a property
of the state, not an override: a straight run of a cornering attitude is a
different state.

### 4.4 Preview

Before queuing, the form can resolve the full spec in-process (the same
`resolve()` call `prepare` makes) and show either the validation error or:
merged spec, ω = U/R, lateral g, ν_t/ν, domain kind, ground motion. Queuing
re-runs the same validation; nothing invalid enters the queue.

---

## 5. Architecture

```
Windows laptop ──Tailscale──▶ Z8 <tailnet-ip>:8000
                                │
          ┌─────────────────────┴──────────────────────┐
          │ simdev ui  (one systemd --user service)    │
          │  ├─ FastAPI: pages, htmx partials, JSON    │
          │  ├─ worker thread                          │
          │  └─ SQLite ~/.local/share/simdev-ui/ui.db  │
          └──────────┬─────────────────────────────────┘
                     │ subprocess per job, own process group
                     ▼
               simdev run … ; simdev images …
                     │ writes
                     ▼
               ~/runs/<name>/{status,logs,postProcessing,results}
```

### 5.1 Source of truth

The SQLite database holds only what the app owns: the queue and settings.
Everything about how a run is going — stages, verdicts, coefficients, images,
logs — is read live from the run directory and never copied into the
database. A run made with the CLI is therefore visible in the UI ("external"
runs, read-only, listed from the runs root).

### 5.2 Queue and worker

`jobs` table: `id, run_name, run_dir, case, design, state, profile,
overrides (json), n_ranks, position, status, pid, pgid, exit_code,
created_at, started_at, finished_at, error`.

`status` ∈ `queued, running, done, gate_failed, failed, cancelled`.

Worker loop, every 2 s:

1. Reap finished children; set status from exit code and `status/*.json`
   (§6).
2. `free = core_budget − Σ n_ranks(running)`. If
   `count(running) < max_parallel`, start the first queued job by `position`
   whose `n_ranks ≤ free`.

Settings: `core_budget = 40`, `max_parallel = 1`. With these defaults it is a
strict FIFO; raising `max_parallel` enables parallel runs with no code change.

A job is started with `start_new_session=True`; its stdout/stderr go to
`<run_dir>/logs/simdev-ui.log`. **Cancel** sends SIGTERM to the process
group (reaching mpirun and every rank), then SIGKILL after 30 s.

### 5.3 Restart

On startup, every `running` job is checked against its stored pid/pgid. If
the process group is alive, the worker adopts it (polls for exit; exit code
falls back to the status files). If not, the job becomes `failed` with
`error = "service restarted while the run was in progress"`. **Resume**
re-queues the same run directory; the pipeline's input hashes skip every
stage that already finished.

### 5.4 Access

The server binds to the Z8's Tailscale address only (configurable; default
taken from `tailscale ip -4`, refusing to start on `0.0.0.0`). There are no
accounts: anyone on the tailnet is the user. `docs/ui-setup.md` covers
installing Tailscale on the Z8 and the laptops and using the MagicDNS name.

### 5.5 Frontend

Server-rendered Jinja pages with htmx for live fragments; a charting library
loaded from a pinned CDN URL for the plots. No Node toolchain. Static files
and templates ship as package data.

---

## 6. Status and errors

A job's final status:

| Status | When |
|---|---|
| `done` | all stages `ok` |
| `gate_failed` | the pipeline completed but a stage wrote `gate_failed` (mesh quality, y+, convergence) |
| `failed` | a stage wrote `failed`, the process exited non-zero without any gate failure, or the service lost it |
| `cancelled` | the user cancelled |

How each kind is shown:

- **Validation errors** — at Preview, at queue time and at CAD upload; never
  queued, never written.
- **Gate failures** — ⚠ with the stage's `reasons[]` verbatim and the numbers
  from its `detail` (e.g. drift vs `drift_tol`).
- **Crashes** — the `StageError` message from `simdev-ui.log`, plus
  `find_fatal_errors()` applied to the failing stage's OpenFOAM log, shown
  with 50 lines of context around the first fatal line.

A failed job never stops the queue.

---

## 7. Pages

1. **Queue** (home) — running jobs with a live stage bar
   (`prepare ▸ mesh ▸ solve ▸ post ▸ images`), current iteration, s/iter and
   ETA to `max_iterations`; queued jobs with reorder, edit, cancel; the 20
   most recent finished jobs with status badges.
2. **New run** — design → state (only pairs with a slot) → profile →
   overrides → name → Preview / Queue.
3. **Run detail**
   - stage timeline with durations, verdicts and reasons;
   - **live plots**: Cd and Cl vs iteration (total; per-component toggle from
     `forceCoeffs_<part>`) with the plateau window shaded, and residuals.
     Polled every 10 s; the client sends the last iteration it has and
     receives only newer rows;
   - **image gallery** from `results/images.json` and the summary PNGs
     (`forces`, `balance`, `components_*`, `residuals`), thumbnails opening
     full size;
   - **logs** — one tab per `logs/log.*` and `simdev-ui.log`, last 200 lines,
     link to the full file;
   - actions: cancel, resume, re-run from a stage (re-queue with `--force`
     for that stage onward), strip mesh, delete run.
4. **CAD library** — states and designs with every operation in §3.3; each
   design lists its slots by state.
5. **System** — `simdev doctor`, free space on the runs root, load average,
   `core_budget` / `max_parallel`.

**Strip mesh** deletes `processor*/`, `constant/polyMesh` and the time
directories, keeping `results/`, `logs/`, `status/`, `postProcessing/`,
`caseSpec.json` and `cad/`. Both strip and delete are refused for a running
job and ask for confirmation in the page.

---

## 8. Code layout, packaging, testing

### 8.1 Layout

```
pipeline/simdev/cad/
  library.py   states, designs, slots: validate, create, move, replace, delete, assemble
pipeline/simdev/ui/
  app.py       FastAPI routes only
  queue.py     SQLite job table: enqueue, reorder, next_fitting(free_cores), transitions
  worker.py    loop: pick → spawn → watch → record; adopt on restart; cancel
  runview.py   read-only view of a run dir: stages, incremental coeffs, logs, images, errors
  templates/   pages + htmx partials
  static/
```

`runview` knows nothing of the queue; `library` nothing of the UI; `worker`
connects queue → CLI. `app.py` stays thin.

### 8.2 Packaging and service

- Optional extra `simdev[ui]`: `fastapi`, `uvicorn`, `python-multipart`.
- `simdev ui [--host] [--port 8000] [--runs-root ~/runs] [--cad-root]`.
- `scripts/simdev-ui.service`, a `systemd --user` unit; `loginctl
  enable-linger` keeps it running when logged out. Setup in
  `docs/ui-setup.md`.

### 8.3 Testing

pytest, existing conventions.

- `library` — each validation rejection; create/move/replace/delete;
  move-onto-existing needs overwrite; state delete blocked by slots;
  assembly yields exactly 15 parts with hashes recorded.
- `resolve` — the state layer's merge order (case < state < `--set`);
  migrated `testcase` resolves to the same spec as before the split, minus
  `geometry.source_dir`.
- `queue` — FIFO, reorder, core-budget fit with `max_parallel > 1`,
  transitions.
- `runview` — against small fixture run dirs: status JSON, a short
  `coefficient.dat`, incremental reads, a log with a FOAM FATAL ERROR.
- `worker` — with a stub command in place of `simdev`: success, gate
  failure, crash, cancel kills the whole process group, adopt-after-restart.
- API — FastAPI `TestClient` on New Run validation and CAD upload.
- One end-to-end test marked `openfoam`: queue a `dev` run through the API,
  wait for `done`.

---

## 9. Out of scope for v1

Result comparison tables, image comparison across runs, notifications,
parameter sweeps, accounts. Parallel runs are supported by the worker but
off by default (`max_parallel = 1`) until rank scaling is re-measured on
native Linux.
