# AvalonRacing SimDev

Development repo for the CFD simulation pipeline (OpenFOAM + supporting
automation scripts).

## Scope

This repository tracks **pipeline development only**:

- Automation/pre- and post-processing scripts (`pipeline/`)
- OpenFOAM case templates - `0/`, `constant/`, `system/` dictionaries
  (`cases/`)
- Documentation (`docs/`)

It does **not** track meshes, simulation results, logs, or any other
run output, at any stage - including once the pipeline is working as
intended. Case runs should happen in a working copy made from a
`cases/<caseName>/` template, outside of what git tracks (e.g. in an
untracked `runs/` directory, or entirely outside this repo). See
`.gitignore` for the enforced exclusions.

## Structure

```
pipeline/   automation scripts that drive meshing, running, and post-processing
cases/      OpenFOAM case templates (0/, constant/, system/ only - no mesh/results)
docs/       setup notes and pipeline usage docs
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

Solo trunk-based development on `main`. Commit directly for routine
changes; use a short-lived branch for larger pipeline rewrites you
want to test before merging.
