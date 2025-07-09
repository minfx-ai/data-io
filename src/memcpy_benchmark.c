// memcpy_benchmark.c
// Benchmarks copying 512 KiB of pinned (mlock‑ed) memory many times
// Prints throughput in MiB/s
// Build:  gcc -O2 -march=native -Wall -Wextra -o memcpy_benchmark memcpy_benchmark.c
// Usage:  ./memcpy_benchmark [iterations]

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

#define COPY_SIZE (512ULL * 1024 * 1024)   /* 512 MiB */

static inline double timespec_to_seconds(const struct timespec *ts) {
    return (double)ts->tv_sec + (double)ts->tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    size_t iterations = 500;             /* default */
    if (argc > 1) {
        char *end;
        iterations = strtoull(argv[1], &end, 10);
        if (end == argv[1] || iterations == 0) {
            fprintf(stderr, "Invalid iterations value.\n");
            return EXIT_FAILURE;
        }
    }

    /* Allocate 64‑byte‑aligned buffers to help the CPU */
    void *src, *dst;
    if (posix_memalign(&src, 64, COPY_SIZE) || posix_memalign(&dst, 64, COPY_SIZE)) {
        perror("posix_memalign");
        return EXIT_FAILURE;
    }

    /* Pin (lock) pages in RAM so they never swap out */
    if (mlock(src, COPY_SIZE) || mlock(dst, COPY_SIZE)) {
        perror("mlock (you may need to raise 'ulimit -l' or run as root)");
        /* Continue anyway; benchmark still works, just not pinned */
    }

    /* Prime the cache so the first measurement isn’t dominated by cold‑cache effects */
    memset(src, 0xAA, COPY_SIZE);
    memset(dst, 0x55, COPY_SIZE);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (size_t i = 0; i < iterations; ++i) {
        memcpy(dst, src, COPY_SIZE);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double seconds = timespec_to_seconds(&t1) - timespec_to_seconds(&t0);
    size_t bytes_copied = (size_t)COPY_SIZE * iterations;
    double mib_copied = (double)bytes_copied / (1024.0 * 1024.0);
    double throughput = mib_copied / seconds;

    printf("Copied %.2f MiB in %.6f seconds => %.2f MiB/s\n",
           mib_copied, seconds, throughput);

    munlock(src, COPY_SIZE);
    munlock(dst, COPY_SIZE);

    free(src);
    free(dst);
    return EXIT_SUCCESS;
}
