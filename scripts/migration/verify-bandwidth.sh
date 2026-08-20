#!/usr/bin/env bash
# Does this machine deliver its memory bandwidth? Run before and after the
# migration; the difference is the whole reason for migrating.
#
# Measured 2026-08-20, same machine, same working set:
#
#   WSL2, whole machine          28.5 GB/s   saturates at FOUR threads
#   native Windows, one socket   25.4 GB/s
#   native Windows, two sockets  ~51  GB/s   measured concurrently
#
# So the expected native-Linux result is ~50 GB/s or better, still climbing
# past four threads. Anything near 28 GB/s that goes flat at four threads
# means the migration did NOT fix the placement problem, and nothing should
# be bought until that is understood.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin="${TMPDIR:-/tmp}/membw"

gcc -O3 -march=native -fopenmp -o "$bin" "$here/membw.c"

echo "=== topology ==="
lscpu | grep -E '^(CPU\(s\)|Thread|Core|Socket|NUMA node)' || true
if command -v numactl >/dev/null 2>&1; then
    echo
    numactl --hardware | grep -E 'available|node [0-9]+ size'
else
    echo "(numactl not installed: apt install numactl - it is how you confirm"
    echo " the kernel sees two nodes, which WSL2 never did)"
fi

echo
echo "=== triad, parallel first touch, threads pinned to cores ==="
export OMP_PROC_BIND=spread OMP_PLACES=cores
for t in 1 2 4 8 16 20 32 40; do
    OMP_NUM_THREADS=$t "$bin" "${1:-6}"
done

cat <<'EOF'

Read it like this:
  climbs past 4 threads to ~50 GB/s  -> placement is correct, migration
                                        delivered, re-run the rank benchmark
  flat from 4 threads at ~28 GB/s    -> still one node's worth of bandwidth.
                                        Do NOT buy DIMMs yet; find out why.
EOF
