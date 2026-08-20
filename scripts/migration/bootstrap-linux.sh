#!/usr/bin/env bash
# Bring a fresh Ubuntu 24.04 to a working SimDev install.
#
# Run it from anywhere; it clones or updates ~/SimDev itself. Safe to re-run.
#
#   bash bootstrap-linux.sh            # clone from origin
#   REPO=/path/to/SimDev bash ...      # use a copy already on disk
#
# What it does NOT do, because both need a human: copy CAD/ back into place
# (it is gitignored, so a clone never has it), and restore the run histories.
# The final message says where those live and what to run.
set -euo pipefail

REPO="${REPO:-$HOME/SimDev}"
VENV="${VENV:-$HOME/.venvs/simdev}"
ORIGIN="${ORIGIN:-https://github.com/AvalonRacing/SimDev.git}"
FOAM_BASHRC=/usr/lib/openfoam/openfoam2412/etc/bashrc

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "Checking the distribution"
. /etc/os-release
[ "${VERSION_ID:-}" = "24.04" ] || echo "WARNING: expected Ubuntu 24.04, found ${PRETTY_NAME:-unknown}"

say "Adding the OpenFOAM repository"
# The ESI build (openfoam.com). The Foundation's packages are a different
# lineage and do not ship surfaceFeatureExtract, which the mesh stage calls.
#
# Added by hand rather than by piping their install script into sudo bash:
# the repository line is recorded in docs/linux-migration.md section 3 and is
# worth being able to read before it runs as root.
sudo apt-get update -qq
sudo apt-get install -y -qq curl gnupg ca-certificates
curl -fsSL https://dl.openfoam.com/pubkey.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/openfoam.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/openfoam.gpg] https://dl.openfoam.com/repos/deb noble main" \
    | sudo tee /etc/apt/sources.list.d/openfoam.list >/dev/null
sudo apt-get update -qq

say "Installing OpenFOAM v2412, the gmsh runtime libraries and Python"
# libglu1-mesa / libopengl0 / libxft2: gmsh imports without them and then
# fails at STEP load, which reads as a CAD problem and is not one.
# numactl: needed to confirm the kernel sees two NUMA nodes, the thing this
# whole migration is about.
sudo apt-get install -y \
    openfoam2412-default \
    libglu1-mesa libopengl0 libxft2 \
    python3-venv python3-pip \
    git build-essential numactl

say "Getting the repository"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only
else
    git clone "$ORIGIN" "$REPO"
fi

say "Building the virtualenv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade -q pip
"$VENV/bin/pip" install -q -e "$REPO"

say "Wiring OpenFOAM into the shell"
# Sourced last so it wins on PATH. The venv must NOT shadow OpenFOAM, which
# is why simdev is always called by absolute path below.
if ! grep -q "openfoam2412/etc/bashrc" "$HOME/.bashrc"; then
    printf '\n# OpenFOAM v2412\n[ -f %s ] && source %s\n' "$FOAM_BASHRC" "$FOAM_BASHRC" >> "$HOME/.bashrc"
fi

say "Verifying"
# shellcheck disable=SC1090
source "$FOAM_BASHRC"
missing=0
for exe in blockMesh snappyHexMesh decomposePar renumberMesh checkMesh simpleFoam surfaceFeatureExtract; do
    if command -v "$exe" >/dev/null 2>&1; then
        printf '  ok      %s\n' "$exe"
    else
        printf '  MISSING %s\n' "$exe"; missing=1
    fi
done
[ "$missing" -eq 0 ] || { echo "OpenFOAM is incomplete; stopping."; exit 1; }

nodes=$(numactl --hardware | awk '/^available:/{print $2}')
printf '\n  NUMA nodes visible to the kernel: %s' "$nodes"
if [ "$nodes" = "2" ]; then
    printf '   <- correct. WSL2 reported 1, and that was the bottleneck.\n'
else
    printf '   <- EXPECTED 2. Check BIOS node interleaving before benchmarking.\n'
fi

say "Left for you"
cat <<EOF
  0. Mount the Windows NVMe partition if it is not mounted (it survives the
     install; the archive was written there before migrating):
       sudo mkdir -p /mnt/windows
       sudo mount /dev/nvme0n1p3 /mnt/windows      # confirm with lsblk
     MIG=/mnt/windows/Users/info/AvalonRacing-migration
  1. CAD/ is gitignored, so this clone does not have it. Copy it back:
       cp -a \$MIG/CAD  $REPO/CAD
  2. Restore the run histories (force traces, logs, status records):
       mkdir -p ~/runs && tar xzf \$MIG/run-histories.tar.gz -C ~/runs
  3. Check the install end to end:
       $VENV/bin/simdev doctor
       $VENV/bin/python -m pytest $REPO/tests -q
  4. Confirm the migration actually delivered - this is the point of it:
       bash $REPO/scripts/migration/verify-bandwidth.sh
     Expect ~50 GB/s still climbing past four threads, against 28.5 under WSL2.
  5. Only then re-run the rank benchmark (20 vs 40 ranks, cases/car, 100
     iterations). It was flat under WSL2; if it is still flat, stop and read
     docs/linux-migration.md section 1b before buying any hardware.

Call simdev by its absolute path ($VENV/bin/simdev). The venv on PATH shadows
OpenFOAM's own Python and that trap survives the migration.
EOF
