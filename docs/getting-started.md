# Getting Started

This repo holds the CFD pipeline's development artifacts only:
scripts and OpenFOAM case templates. No meshes, results, or run
output belong here - see the root `README.md` and `.gitignore`.

## Adding a new case template

1. Create `cases/<caseName>/` with `0/`, `constant/`, `system/`.
2. Keep only dictionary/config files - no generated mesh
   (`constant/polyMesh/`) or result directories.

## Adding a pipeline script

Add it under `pipeline/`, and note any new Python dependency in
`requirements.txt`.
