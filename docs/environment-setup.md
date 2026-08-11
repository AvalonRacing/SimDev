# Environment Setup

## 1. WSL2 + Ubuntu

From an elevated PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot when prompted, then create your Linux user.

## 2. Move the WSL disk to D:

`C:` has ~330 GB free; CFD runs will exhaust it. Create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=100GB
processors=40
```

Then relocate the distro:

```powershell
wsl --shutdown
wsl --export Ubuntu D:\wsl\ubuntu-backup.tar
wsl --unregister Ubuntu
wsl --import Ubuntu D:\wsl\Ubuntu D:\wsl\ubuntu-backup.tar
```

## 3. OpenFOAM v2412 (ESI)

```bash
curl https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get install openfoam2412-default python3-venv python3-pip
echo "source /usr/lib/openfoam/openfoam2412/etc/bashrc" >> ~/.bashrc
```

Verify:

```bash
simpleFoam -help | head -3
which surfaceFeatureExtract snappyHexMesh checkMesh
```

`surfaceFeatureExtract` must resolve. It is the ESI (openfoam.com) utility and
reads `system/surfaceFeatureExtractDict`, which is what the templates render.
If instead you find `surfaceFeatures`, you have an OpenFOAM Foundation build
(openfoam.org) — a different distribution with an incompatible, flat dictionary
format, not a newer or older version of the same thing.

### Activating the environment

The line appended to `~/.bashrc` above only takes effect in **interactive**
shells: Ubuntu's stock `~/.bashrc` opens with

```bash
case $- in
    *i*) ;;
      *) return;;
esac
```

so a non-interactive `wsl bash -c '…'` returns before ever reaching it, and
OpenFOAM is not on `PATH`. Anything that drives the pipeline programmatically
must source the environment explicitly:

```bash
wsl -d Ubuntu-24.04 -- bash -lc 'source /usr/lib/openfoam/openfoam2412/etc/bashrc; simdev run …'
```

## 4. Case location

Run cases under `~/runs` on the ext4 filesystem. **Never** under `/mnt/c` or
`/mnt/d` — bind-mounted I/O collapses OpenFOAM performance.

## 5. Core count

Use 40 ranks maximum (physical cores). OpenFOAM is memory-bandwidth bound;
the 80 logical threads reduce throughput.

## 6. The development loop

The repository lives on Windows (`C:\Users\info\Documents\AvalonRacing\SimDev`)
and OpenFOAM lives in WSL. These are the details that are easy to get wrong
and expensive to rediscover.

### Python environment

`pip install -e` against the Windows checkout **fails** from WSL: the
`simdev.egg-info` directory created by the Windows venv cannot be rewritten
across the NTFS mount (`Operation not permitted`). So the Linux side gets its
own copy of the tree and its own venv:

```bash
python3 -m venv ~/.venvs/simdev
~/.venvs/simdev/bin/pip install -e ~/SimDev
```

`python3-venv` is not installed by default on Ubuntu 24.04 — it is included
in the apt line in §3.

### Syncing Windows → WSL

`~/SimDev` is a **one-way** snapshot. Edits are made on the Windows side and
pushed across; nothing syncs back:

```bash
cd /mnt/c/Users/info/Documents/AvalonRacing/SimDev && \
  tar --exclude=.git --exclude=.venv --exclude="*.egg-info" \
      --exclude=__pycache__ --exclude=.pytest_cache \
      --exclude=benchmark_old_pipeline -cf - . | (cd ~/SimDev && tar -xf -)
```

**Editing under `~/SimDev` loses the work on the next sync.** Converting this
to a `git clone` with a remote would remove the footgun.

### Running the pipeline

Two traps, both of which produce confusing failures:

- **Do not prepend the venv to `PATH`.** `export PATH="$HOME/.venvs/simdev/bin:$PATH"`
  after sourcing the OpenFOAM bashrc leaves OpenFOAM off the path and
  `simdev doctor` reports every utility missing. Call the binary by absolute
  path instead.
- **Do not use `set -e` in a script that sources the OpenFOAM bashrc.** It
  aborts with `pop_var_context: head of shell_variables not a function
  context`.

The form that works:

```bash
wsl -d Ubuntu-24.04 -- bash -lc 'source /usr/lib/openfoam/openfoam2412/etc/bashrc
cd ~/SimDev
~/.venvs/simdev/bin/simdev run cases/ahmed/config.yaml --run-dir ~/runs/ahmed-01 --profile dev'
```

Also note that Git Bash rewrites bare Unix paths passed as `wsl.exe`
arguments (`/home/info/...` becomes `C:/Program Files/Git/home/info/...`).
Keep paths inside the quoted `bash -lc` string.

### CAD

No conversion step and no extra dependency: the CAD is exported as STL, one
file per patch, and `trimesh` (already required) reads it. The earlier
STEP-via-gmsh plan is not needed — see
`docs/superpowers/specs/2026-08-09-step-geometry-path-design.md` for why it was
dropped.

The export contract is in `docs/handbook.md` §6 "Bring in CAD". The part that
bites: parts must be exported in **assembly position**, not in their own part
frames. An export that loses the assembly transforms writes every corner of the
car to the same coordinates, and all four wheels land on top of each other at
the origin.

### Coded boundary conditions

Cornering tyres use `codedFixedValue`, which OpenFOAM compiles at run time into
`<case>/dynamicCode/`. This needs `allowSystemOperations 1` in
`etc/controlDict` — the default in v2412 — and a writable run directory, which
is another reason runs live under `~/runs` on ext4 rather than on `/mnt/c`.

A compile failure aborts the solver with `Failed wmake` and the generated
source is left in `dynamicCode/` to read.
