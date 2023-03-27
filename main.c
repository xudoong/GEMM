#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dgemm.h"


void test_one(int m, int n, int k, double alpha, double beta, double *A, double *B, double *C, int repeat, void (*fun_ptr)(DGEMM_FUNC_SIGNITURE), char *name)
{
    // warmup doesn't count
    (*fun_ptr)(m, n, k, alpha, A, k, B, n, beta, C, n);

    // start timing
    clock_t start_clock = clock();
    for (int iter = 0; iter < repeat; iter++)
    {
        (*fun_ptr)(m, n, k, alpha, A, k, B, n, beta, C, n);
    }
    clock_t end_clock = clock();

    const double MAX_GFLOPs = 3.8 * 2 * 2 * (512 / 64);
    double time_sec = (double)(1.0 * end_clock - start_clock) / CLOCKS_PER_SEC;
    double GFLOPs = (1.0 * m * k * n * 2 * repeat) / time_sec / 1e9;
    printf("[%s] m=%d,n=%d,k=%d GFLOPS=%.1f (%.0f%%)\n", name, m, n, k, GFLOPs, GFLOPs / MAX_GFLOPs * 100);
}

int main(int argc, char **argv)
{
    double *A, *B, *C;
    int m, n, k, i, j;
    int repeat;
    double alpha, beta;

    m = 512, k = 512, n = 512;
    repeat = 30;
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
        C[i] = 0.0;
    }

    /* Start test */
    test_one(m, n, k, alpha, beta, A, B, C, repeat, &dgemm_mkl, "MKL");
    // test_one(m, n, k, alpha, beta, A, B, C, repeat, &dgemm_ideal, "IDEAL");
    // test_one(m, n, k, alpha, beta, A, B, C, repeat, &dgemm_vec, "VEC");
    // test_one(m, n, k, alpha, beta, A, B, C, repeat, &dgemm_mypack, "PACK");
    // test_one(m, n, k, alpha, beta, A, B, C, repeat, &dgemm_v1, "v1");
    // test_one(m, n, k, alpha, beta, A, B, C, repeat, &dgemm_v2, "v2");
    test_one(m, n, k, alpha, beta, A, B, C, repeat, &dgemm_v3, "v3");


    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
