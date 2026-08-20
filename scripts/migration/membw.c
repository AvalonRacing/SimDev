/* STREAM-style triad, used to decide whether the platform is the bottleneck.
 *
 * Build:  gcc -O3 -march=native -fopenmp -o membw membw.c
 * Run:    OMP_PROC_BIND=spread OMP_PLACES=cores ./membw [GB]
 *
 * Reports the STREAM convention of 24 bytes per element (two reads and a
 * write). Real DRAM traffic is a third higher again, because the store does
 * a read-for-ownership first - which matters when comparing against a
 * channel's theoretical peak, and is noted where the script prints one.
 *
 * First touch is parallel and threads are pinned, deliberately. Serial first
 * touch puts every page on one NUMA node and makes any two-socket machine
 * look like a one-socket machine, which is the single easiest way to
 * misdiagnose this box. The 2026-08-20 WSL2 measurement was checked against
 * both and was not that artefact - see docs/linux-migration.md section 1b.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv)
{
    double gb = (argc > 1) ? atof(argv[1]) : 6.0;
    size_t N = (size_t)(gb * 1e9 / 24.0);   /* three arrays of 8 bytes */
    int reps = 10;

    double *a = aligned_alloc(64, N * sizeof(double));
    double *b = aligned_alloc(64, N * sizeof(double));
    double *c = aligned_alloc(64, N * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "allocation of %.1f GB failed\n", gb); return 1; }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 3.0; }

    double s = 3.0, best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; i++) a[i] = b[i] + s * c[i];
        double dt = omp_get_wtime() - t0;
        if (dt < best) best = dt;
    }

    printf("%3d threads  %6.1f GB/s   (%.2f GB working set, best %.4f s)\n",
           omp_get_max_threads(), 24.0 * N / 1e9 / best, 24.0 * N / 1e9, best);
    free(a); free(b); free(c);
    return 0;
}
