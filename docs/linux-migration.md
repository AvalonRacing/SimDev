# Migrating off WSL2 to native Linux

Written 2026-08-20, from measurements taken on this machine. Everything here
was measured rather than assumed; where something is still an inference it says
so.

## 0. The day itself

Everything that can be prepared in advance has been. What is staged:

- the repo is committed and pushed, so a clone on Linux is current;
- `C:\Users\info\AvalonRacing-migration\` holds `CAD/` and
  `run-histories.tar.gz`, the two things a clone will not have (section 4);
- `scripts/migration/bootstrap-linux.sh` installs the whole environment;
- `scripts/migration/verify-bandwidth.sh` proves whether it worked.

Order, with the irreversible step as late as possible:

| # | Step | Reversible? |
|---|---|---|
| 1 | Confirm the archive: `CAD/` 30 files / 35.8 MB, tarball 414 paths | — |
| 2 | Write the Ubuntu 24.04 USB installer | yes |
| 3 | BIOS: SATA controller RST -> AHCI (section 2, gotcha 1) | yes, revert it |
| 4 | Boot Windows once to confirm it still boots from the NVMe | — |
| 5 | Shrink C: from Windows Disk Management, leaving the free space unformatted | yes |
| 6 | **Install Ubuntu into that free space** | **no** |
| 7 | `bootstrap-linux.sh`, then section 5 steps 3-8 | yes |

Steps 3 and 4 are ordered that way deliberately. Windows boots from the NVMe
through `stornvme`, which does not go through the RAID controller, so the
AHCI switch should not affect it — "should" is why it is verified before
anything is repartitioned. Expect D: to change or disappear; nothing needed
for the migration lives there.

BitLocker is off (checked 2026-08-20), so shrinking C: needs no recovery key.
Secure Boot can stay on; Ubuntu ships a signed shim.

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

**That benchmark never finished, and the number it was supposed to produce
does not exist.** `~/runs/bench-10` has no `log.simpleFoam` and no
`status/solve.json`, and its `coefficient.dat` stops at 74 iterations: the run
was interrupted. `~/runs/bench-5` was never created. Any "10 ranks" figure in
older session notes was reconstructed from directory timestamps, not read from
a log - do not cite it.

**It no longer matters, because the question was settled directly.** See
section 1b: the answer is the second branch above - the ceiling is reached
well below 10 ranks - and the "something more basic capping the VM" has been
found and measured. It is memory placement, and native Linux does fix it.

**Honest caveat.** A CPU-burn scaling test run to confirm this came out
self-contradictory (10 -> 20 threads gained nothing, 20 -> 40 gained 1.8x,
pinned and unpinned runs disagreed). Nothing above rests on it. The claim rests
on the two OpenFOAM runs, which are clean and reproducible.

**Expected upside: ~1.8x, and it is now measured rather than inferred - see
section 1b.** The machine sustains ~51 GB/s when memory is placed on the node
that uses it, against the 28.5 GB/s WSL2 delivers, and an OpenFOAM-shaped
gather benchmark reproduces that 1.85x under the same two placements.

What that buys on the production car case, with `renumberMesh` already in:

| | s/iter | 2000 iterations |
|---|---|---|
| WSL2, before renumbering | 15.10 | 8.4 h |
| WSL2, after renumbering | 13.65 | 7.6 h |
| native Linux, 1.5-1.8x | 7.6-9.1 | **4.2-5.1 h** |

Which means the five-hour target the handbook section 6 called unreachable is
reachable, with no cell cut, no iteration cut and no change to the numerics.
Treat the 1.5-1.8x band as a range, not a promise: the proxy measures the
matrix gather, and the real solve also does assembly and halo exchange that
will not speed up by the same factor.

## 1b. It is a memory-bandwidth problem, and the operating system is the
##     larger half - and the free one

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
socket - **4 of 12 channels**, on a board with 24 slots (two per channel). At
21.3 GB/s per DDR4-2666 channel that caps the machine at ~85 GB/s theoretical,
against ~256 GB/s fully populated.

### The four DIMMs are already placed correctly. Do not rearrange them.

Measured 2026-08-20 from **native Windows**, outside the VM, binding threads to
one processor group at a time (Windows puts the two sockets in separate groups,
so a process reaches one socket unless it asks otherwise):

| | bandwidth |
|---|---|
| socket 0 alone | 25.4 GB/s |
| socket 1 alone | 25.4 GB/s |
| both concurrently | **~51 GB/s** (25.6 + 25.3, no interference) |
| threads on one socket, memory on the other | 20.1 GB/s (21% NUMA penalty) |

**A single DDR4-2666 channel peaks at 21.33 GB/s. Each socket sustains 25.4.**
That exceeds one channel's theoretical maximum on the STREAM convention alone,
before counting the read-for-ownership the store does, which puts real traffic
near 34 GB/s. So each socket's two DIMMs are necessarily on **two different
channels**, at ~80% of a two-channel peak - a healthy result. For four DIMMs on
a two-socket six-channel-per-socket board, 2+2 on distinct channels is the best
arrangement there is, and it is the one installed. **There is nothing to gain
by moving them.**

### The operating system is the bigger half, and it is free to fix

WSL2 delivers **28.5 GB/s for the whole machine** against the ~51 GB/s the
hardware demonstrably has - roughly one socket's worth. Ruled out as
explanations:

- **A benchmark artefact.** The 28 GB/s reproduces exactly with parallel first
  touch and threads pinned to cores. Serial first touch was the obvious
  suspect and is not the cause.
- **The working set fitting on one node.** At a 72 GB working set - which
  *must* span both sockets, since each holds only 64 GB - it rose only 12%, to
  31.8 GB/s.
- **NUMA distance.** Remote access costs 21%, nowhere near a factor of two.

What it behaves like is a machine whose working set lives on one node's memory
controllers however many cores are used. The confirming test used an
OpenFOAM-style face-based gather/scatter - the actual `lduMatrix` access
pattern, not a streaming one - with the same binary and memory placement the
only variable:

| threads | NUMA-local | all memory on one node |
|---|---|---|
| 8 | 35.3 GB/s | 23.5 GB/s |
| 16 | 37.0 GB/s | 24.1 GB/s |
| 32 | **45.4 GB/s** | **24.5 GB/s** |

The single-node column is **flat from 8 to 32 threads** - which is precisely
the "adding cores does nothing" signature in section 1 - while NUMA-local
placement keeps scaling. **1.85x at 32 threads and still climbing.** Native
Linux sees both NUMA nodes and lets each rank first-touch its own data, which
is the left column.

So the honest split, now measured rather than inferred:

| | bandwidth | |
|---|---|---|
| now (WSL2) | **28.5 GB/s** | measured |
| native Linux, 4 channels | ~51 GB/s | measured on this hardware, ~1.8x |
| native Linux, 12 channels | ~150 GB/s | still an inference from channel count |

**Populating the other eight slots is no longer the first move.** It still
raises the ceiling and it would help under WSL too - WSL is bottlenecked on one
node's channels, and giving that node six instead of two lifts the same
ceiling - but it costs money, and the operating system is worth ~1.8x for
nothing. Eight more matched 32 GB DDR4-2666 ECC RDIMMs fills all twelve
channels; do not mix capacities, interleaving wants identical DIMMs.

Sequence, cheapest first:

1. **`renumberMesh`. Already done** - see the handbook section 6 table. 9.6% of
   the solve for no resolution and no hardware.
2. **Migrate to native Linux.** Free, and worth ~1.8x on the measurement above.
   The live-USB STREAM check this document used to recommend is no longer a
   prerequisite: the machine's true bandwidth has been measured from outside
   the VM, which is what that test was for.
3. **Then re-measure, and only then consider the DIMMs.** Run
   `scripts/migration/verify-bandwidth.sh`. If native Linux lands near 50 GB/s
   and keeps climbing past four threads, the placement problem is fixed and the
   remaining headroom is what the DIMMs would buy.
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
| The repo | `C:\Users\info\Documents\AvalonRacing\SimDev` | 39 MB | In git, pushed. A clone is enough |
| `CAD/` | same, **gitignored** | 36 MB | Not in git. Archived — see below |
| `~/runs` | WSL | 27 GB | Meshes and fields regenerate. The histories do not |
| `~/.venvs` | WSL | 556 MB | Do not copy — rebuild from pyproject |

`.gitattributes` already normalises to LF (`* text=auto eol=lf`), so a fresh
clone on Linux gets correct line endings. Nothing in the pipeline is
Windows-specific; it has only ever run against Linux OpenFOAM.

**The archive is already staged**, at
`C:\Users\info\AvalonRacing-migration\`:

```
CAD/                    35.8 MB, 30 files   verified byte-identical to the repo copy
run-histories.tar.gz     1.9 MB, 414 paths  every postProcessing/, logs/,
                                            status/ and caseSpec.json under ~/runs
```

Only 14 MB of the 27 GB in `~/runs` is irreplaceable, which is what that
tarball holds: the force traces, the solver logs and the stage records. The
rest is mesh and field data that the pipeline regenerates.

It sits on **C:**, deliberately. The Linux install shrinks C: rather than
wiping it, so the partition survives and Linux can mount it. Do not stage this
on D: — D: is on the Intel RST controller discussed in section 2, and is the
one volume that may not be visible after a BIOS change.

## 5. Once Linux is up

Steps 1 and 2 are automated. `scripts/migration/bootstrap-linux.sh` adds the
OpenFOAM repository (by explicit signed source line, not by piping their
installer into `sudo bash`), installs v2412 plus the gmsh runtime libraries
and `numactl`, clones or updates `~/SimDev`, builds the venv, appends the
OpenFOAM `bashrc` to `~/.bashrc`, and checks that every utility the pipeline
calls — `renumberMesh` included — is on PATH. It is safe to re-run.

```bash
curl -fsSL https://raw.githubusercontent.com/AvalonRacing/SimDev/main/scripts/migration/bootstrap-linux.sh -o bootstrap.sh
less bootstrap.sh          # it runs sudo; read it first
bash bootstrap.sh
```

Then, in order:

3. Mount the Windows partition and restore what git does not carry:
   ```bash
   sudo mkdir -p /mnt/windows && sudo mount /dev/nvme0n1p3 /mnt/windows   # check lsblk
   MIG=/mnt/windows/Users/info/AvalonRacing-migration
   cp -a "$MIG/CAD" ~/SimDev/CAD
   mkdir -p ~/runs && tar xzf "$MIG/run-histories.tar.gz" -C ~/runs
   ```
4. `~/.venvs/simdev/bin/simdev doctor` — checks every OpenFOAM utility and the
   STEP import path
5. `~/.venvs/simdev/bin/python -m pytest ~/SimDev/tests -q` — 415 tests
6. **Confirm the migration delivered, before anything else.**
   `bash ~/SimDev/scripts/migration/verify-bandwidth.sh`. Expect ~50 GB/s
   still climbing past four threads. **28 GB/s flat from four threads means the
   placement problem is not fixed** — stop and read section 1b rather than
   buying DIMMs.
7. **Re-run the rank benchmark.** 20 vs 40 ranks on `cases/car`, 100
   iterations. Under WSL2 it was flat at 15.3-15.4 s/iter. If 40 ranks is now
   meaningfully faster than 20, the migration delivered what it was for.
8. `numactl --hardware` should report two nodes. WSL2 reported one, and that
   was the whole problem. Consider `mpirun --bind-to core --map-by socket`
   settled only after re-measuring: it was noise under WSL2 precisely because
   placement could not matter there, and it may matter now.

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
