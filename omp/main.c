#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "mkl.h"
#include "omp.h"

#include "dgemm_omp.h"

#define MIN_REPEAT 5
#define MIN_SECONDS 1

inline double timespec_to_second(struct timespec start, struct timespec finish) {
    return (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) * 1e-9;
}

inline double get_max_ghz(int cores_per_socket) {
    assert(cores_per_socket > 0 && cores_per_socket <= 20);
    if (cores_per_socket <= 2)
        return 3.8;
    if (cores_per_socket <= 4)
        return 3.6;
    if (cores_per_socket <= 8)
        return 3.5;
    if (cores_per_socket <= 12)
        return 3.0;
    if (cores_per_socket <= 16)
        return 2.7;
    return 2.5;
}

inline int get_repeat(double seconds) {
    return MAX(MIN_REPEAT, MIN_SECONDS / seconds);
}

void test_one(int m, int n, int k, double alpha, double beta, double *A, double *B, double *C, void (*fun_ptr)(DGEMM_FUNC_SIGNITURE), char *name)
{
    struct timespec start, finish;

    // warmup doesn't count. Use this timing to get repeat times
    clock_gettime(CLOCK_MONOTONIC, &start);
    (*fun_ptr)(m, n, k, alpha, A, k, B, n, beta, C, n);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    double time_sec = timespec_to_second(start, finish);
    int repeat = get_repeat(time_sec);

    // start timing
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < repeat; iter++)
    {
        (*fun_ptr)(m, n, k, alpha, A, k, B, n, beta, C, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);

    // get max GHz
    int num_threads = mkl_get_max_threads(); // Note: this is correct only when running in a single socket
    double max_ghz = get_max_ghz(num_threads);

    const double MAX_GFLOPs = max_ghz * 2 * 2 * (512 / 64) * num_threads;
    time_sec = timespec_to_second(start, finish);
    double GFLOPs = (1.0 * m * k * n * 2 * repeat) / time_sec / 1e9;
    printf("[%4s] m=%d,n=%d,k=%d GFLOPS=%4.0f (%.0f%%)\n", name, m, n, k, GFLOPs, GFLOPs / MAX_GFLOPs * 100);
}


int main(int argc, char **argv)
{
    double *A, *B, *C;
    int m, n, k, i, j;
    int repeat;
    double alpha, beta;

    m = 768, k = 768, n = 768;
    alpha = 2.0; beta = 2.0;

    if (argc == 2) {
        m = n = k = atoi(argv[1]);
    }
    if (argc == 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    A = (double *) _mm_malloc( m * k * sizeof( double ), 64);
    B = (double *) _mm_malloc( k * n * sizeof( double ), 64);
    C = (double *) _mm_malloc( m * n * sizeof( double ), 64);

    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = (double)i;
    }

    /* Start test */
    test_one(m, n, k, alpha, beta, A, B, C, &dgemm_omp_mkl, "mkl");
    test_one(m, n, k, alpha, beta, A, B, C, &dgemm_omp_v2, "v2");
    test_one(m, n, k, alpha, beta, A, B, C, &dgemm_omp_v1, "v1");


    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
