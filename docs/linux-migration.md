# Migrating off WSL2 to native Linux

Written 2026-08-20, from measurements taken on this machine. Everything here
was measured rather than assumed; where something is still an inference it says
so.

## 1. Why — the measured case

The solver does not get faster when given more cores. Two full 100-iteration
runs of `cases/car` at production resolution, same mesh, same case, differing
only in rank count:

| n_ranks | mesh time | solve rate |
|---|---|---|
| 40 | 876 s | **15.43 s/iter** |
| 20 | 870 s | **15.32 s/iter** |

Halving the ranks changed nothing — not the solve, not even the meshing. Twenty
of the forty ranks contribute zero.

The topology is the reason:

```
host (Windows)  2 sockets, 2 NUMA nodes   HP Z8 G4, 2 x Xeon Gold 6148
WSL2 guest      1 socket,  1 NUMA node,   40 cores / 80 threads, all "node0"
```

Windows reports 2 NUMA nodes, so this is not BIOS node-interleaving — WSL2
flattens the topology for its utility VM. Every rank sees one node, so roughly
half its memory traffic crosses the UPI link and neither the kernel nor Open MPI
can place pages near the core using them. `simpleFoam` is memory-bandwidth
bound, and a case that saturates at 20 ranks on a 40-core machine is what that
looks like. WSL 2.7.11.0 has no `.wslconfig` NUMA control; no setting there can
fix it.

### Where the 10- and 5-rank points land

A benchmark bracketing the saturation point was still running when this was
written, into `~/runs/bench-10` and `~/runs/bench-5`. The numbers survive in
the run directories - no session notes needed:

```bash
for R in ~/runs/bench-*; do
  echo -n "$(basename $R): "
  grep -oE 'ClockTime = [0-9]+ s' $R/logs/log.simpleFoam | tail -1
  grep -vc '^#' $R/postProcessing/forceCoeffs/0/coefficient.dat
done
# s/iter = ClockTime / iterations
```

Read it like this. Known: 20 ranks and 40 ranks are both 15.3-15.4 s/iter.

- **10 ranks near 30 s/iter** - saturation sits between 10 and 20 ranks. The
  case is bandwidth-bound, the NUMA story holds, migrate.
- **10 ranks also near 15 s/iter** - the ceiling is far lower than 20 ranks and
  is NOT ordinary bandwidth saturation. Something more basic is capping the VM.
  **Find that before repartitioning anything** - native Linux may not fix it.
**Honest caveat.** A CPU-burn scaling test run to confirm this came out
self-contradictory (10 -> 20 threads gained nothing, 20 -> 40 gained 1.8x,
pinned and unpinned runs disagreed). Nothing above rests on it. The claim rests
on the two OpenFOAM runs, which are clean and reproducible.

**Expected upside: see section 1b - the earlier ~2x estimate was made before
the bandwidth was measured and before the DIMM population was known.** That is an inference from the mechanism, not a measurement, and it is
the main thing the migration is buying. It turns a 20 M-cell 5000-iteration
solve from ~59 h into ~30 h. It does **not** on its own reach a 4-5 h target at
20 M cells; the iteration count has to be attacked as well.

## 1b. STOP - read this before migrating. It is a memory-bandwidth problem,
##      and the OS is only part of it.

Measured 2026-08-20 with a STREAM triad (1.9 GB working set, far out of cache):

```
 1 threads : 17.5 GB/s      8 threads : 28.9 GB/s
 2 threads : 24.5 GB/s     20 threads : 27.8 GB/s
 4 threads : 28.0 GB/s     40 threads : 28.4 GB/s   <- saturated at 4 threads
```

**~28 GB/s, saturating at four threads.** That is the whole explanation for why
10, 20 and 40 ranks all solve at 15.3-15.5 s/iter: the solver is memory bound
and the bandwidth ceiling is reached long before 10 ranks. It also explains why
meshing is flat (939 / 870 / 876 s at 10 / 20 / 40 ranks), and why a pure-ALU
CPU-burn test still scaled 17.5x - that test never touches memory.

Ruled out by measurement, so do not re-chase them:

- **MPI transport** - `/dev/shm` is 50 GB and the `vader` shared-memory BTL is
  present. Not a TCP-fallback problem.
- **Hugepages** - THP is `[madvise]` and `AnonHugePages` is 0, but forcing
  `always` changed nothing (27.8 / 28.4 / 28.3 GB/s) and AnonHugePages stayed 0.
- **CPU capacity** - pure-ALU work scales 17.5x across 40 threads.
- **NUMA placement alone** - real (WSL flattens 2 nodes into 1), but far too
  small to explain a 6x shortfall on its own.

### THE HARDWARE IS HALF THE PROBLEM, AND THE CHEAPER HALF TO FIX

```
CPU0-DIMM1  32 GB      CPU1-DIMM1  32 GB
CPU0-DIMM12 32 GB      CPU1-DIMM12 32 GB      4 DIMMs of 12 slots, 2666 MT/s
```

A Xeon Gold 6148 has **six memory channels per socket**. Two are populated per
socket - **4 of 12 channels**. At 21.3 GB/s per DDR4-2666 channel that caps the
machine at ~85 GB/s theoretical, against ~256 GB/s fully populated. A real
STREAM on this DIMM layout should still reach ~60-70 GB/s on bare metal, so:

| | bandwidth | note |
|---|---|---|
| now (WSL2, 4 channels) | **28 GB/s** | measured |
| native Linux, 4 channels | ~60-70 GB/s | inferred, ~2.3x |
| native Linux, 12 channels | ~190 GB/s | inferred, ~7x |

The last two rows are **inferences from the channel count and normal STREAM
efficiency, not measurements.** Verify by running the same STREAM binary from a
live USB before buying anything.

**Populating the other eight slots is probably the highest-leverage single
change available, and it is independent of the operating system.** Eight more
matched 32 GB DDR4-2666 ECC RDIMMs fills all twelve channels. Do not mix
capacities across channels if it can be avoided - interleaving wants identical
DIMMs.

Sequence worth following, cheapest and least destructive first:

1. Boot a Linux live USB, run the same STREAM triad. This costs nothing and
   separates 'WSL is slow' from 'this machine is slow'.
2. If bare metal gives ~60-70 GB/s, the OS is worth ~2.3x - migrate.
3. Populate the remaining channels either way. On the numbers above it is worth
   more than the migration is.
## 2. Hardware, and the three gotchas

```
Disk 0   ATA ST8000NM000A   7452 GB HDD   BusType RAID   -> D:  (7418 GB free)
Disk 1   SAMSUNG MZVLB512   477 GB NVMe   BusType NVMe   -> C:  (356 GB free)
Firmware UEFI      Secure Boot: ENABLED
C: partitions  1 System (0.1 GB) | 2 Reserved | 3 C: Basic (476 GB) | 4 Recovery (0.8 GB)
```

1. **The 8 TB disk is on an Intel SATA RAID controller** (`iaStorE` / `iaStorB`,
   C600+/C220+ chipset in RAID mode, not AHCI). Ubuntu's installer frequently
   cannot see disks in Intel RST mode. The good news: **Windows boots from the
   NVMe** via `stornvme`, which is independent of that controller, so switching
   the SATA controller to AHCI in BIOS should not break the Windows boot drive.
   Verify before touching it, and expect D: to change.
2. **The EFI System Partition is 100 MB.** Workable — Ubuntu keeps kernels in
   `/boot` on root and only the bootloader in the ESP — but it is tight, and it
   is shared with Windows. Watch it during kernel upgrades.
3. **Secure Boot is on.** Ubuntu installs fine (signed shim). Any out-of-tree
   module (NVIDIA, for instance) needs MOK enrolment at first boot.
4. **BitLocker is off** — checked 2026-08-20 with `manage-bde -status C:`:
   fully decrypted, protection disabled, no key protectors. Shrinking C: is
   unobstructed and needs no recovery key.

Recommended layout: Linux root on the **NVMe** (shrink C:, it has 356 GB free) so
the solver writes to fast storage; leave the 8 TB for bulk run archive once the
controller question is settled.

## 3. The environment to reproduce, exactly

```
Ubuntu 24.04.4 LTS
OpenFOAM v2412  from  deb [arch=amd64] https://dl.openfoam.com/repos/deb noble main
  packages: openfoam2412-default  (pulls openfoam2412, -common, -dev, -tools, -source)
  bashrc  : /usr/lib/openfoam/openfoam2412/etc/bashrc
Open MPI 4.1.6   (comes with OpenFOAM)
Python 3.12.3    python3-pip python3-venv
gmsh runtime libs: libglu1-mesa libopengl0 libxft2   <-- gmsh imports but fails
                                                         at STEP load without them
```

Python venv at `~/.venvs/simdev`, `pip install -e ~/SimDev`. Pinned versions
currently in use:

```
pydantic 2.13.4   jinja2 3.1.6   numpy 2.5.2    pandas 3.0.5   matplotlib 3.11.1
trimesh 5.0.0     scipy 1.18.0   networkx 3.6.1 pyyaml 6.0.3   gmsh 4.15.2
shapely 2.1.2     pytest 9.1.1
```

## 4. What has to move

| What | Where | Size | Notes |
|---|---|---|---|
| The repo | `C:\Users\info\Documents\AvalonRacing\SimDev` | 39 MB | **Commit first — see below** |
| `CAD/` | same, **gitignored** | 36 MB | Not in git. Must be copied by hand or it is gone |
| `~/runs` | WSL | 24 GB | Mostly regenerable meshes. The force histories are not |
| `~/.venvs` | WSL | 556 MB | Do not copy — rebuild from pyproject |

`.gitattributes` already normalises to LF (`* text=auto eol=lf`), so a fresh
clone on Linux gets correct line endings. Nothing in the pipeline is
Windows-specific; it has only ever run against Linux OpenFOAM.

**28 files are uncommitted right now**, including new modules
(`geometry/contact.py`, `geometry/mrf.py`, `geometry/decimate.py`) and their
tests. None of that reaches a fresh clone. Commit before migrating.

## 5. Once Linux is up

1. `sudo add-apt-repository` the OpenFOAM repo above, `apt install
   openfoam2412-default libglu1-mesa libopengl0 libxft2 python3-venv python3-pip`
2. `python3 -m venv ~/.venvs/simdev && ~/.venvs/simdev/bin/pip install -e ~/SimDev`
3. Copy `CAD/` back in place (it is gitignored, so the clone will not have it)
4. `~/.venvs/simdev/bin/simdev doctor` — checks every OpenFOAM utility and the
   STEP import path
5. `~/.venvs/simdev/bin/python -m pytest tests/ -q` — 408 tests
6. **Re-run the rank benchmark.** 20 vs 40 ranks on `cases/car`, 100 iterations.
   If 40 ranks is now meaningfully faster than 20, the migration delivered what
   it was for. If it is still flat, the NUMA inference was wrong and the
   bottleneck is elsewhere — find it before buying hardware.

Two WSL-era traps that stop applying, and should be deleted from
`docs/environment-setup.md` once native: the one-way `tar` sync into `~/SimDev`
(use a git clone), and the `.wslconfig` `processors=` logical-vs-physical trap.
The venv-shadows-OpenFOAM-on-PATH trap still applies — keep calling
`~/.venvs/simdev/bin/simdev` by absolute path.

## 6. Still open, unrelated to the platform

- **`solve.plateau_tol` is 0.002 and unreachable.** The cornering case converges
  in the mean (Cd 1.0635 +/- 0.35 %, Cl -1.726 +/- 0.9 % over rolling
  200-iteration windows) but carries a physical limit-cycle oscillation of
  +/-7.9 % in Cl and +/-2.2 % in Cd. Steady RANS on a massively separated flow
  has no fixed point to find. The gate needs a threshold that reflects that, or
  the run needs time-averaging.
- **Layer coverage** on `Chassis` (48 %), tyres (51-58 %) and `SUS` (25 %) is
  below the old 0.7 default and not reachable by refining — see the note in
  `cases/car/config.yaml`. y+ is the criterion that actually holds, and it
  passes: all eleven force patches at 1.96-4.90.
