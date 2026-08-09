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
