#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dgemm.h"


void test_one(int m, int n, int k, double alpha, double beta, const double *A, const double *B, double *C, double *C_corr, void (*fun_ptr)(DGEMM_FUNC_SIGNITURE), char *name)
{    
    for (int i = 0; i < (m*n); i++) {
        C[i] = C_corr[i] = 1.0;
    }

    // warmup doesn't count
    (*fun_ptr)(m, n, k, alpha, A, k, B, n, beta, C, n);
    dgemm_mkl(m, n, k, alpha, A, k, B, n, beta, C_corr, n);

    double r = 0;
    for (int i = 0; i < m * n; i++) {
        r += (C[i] - C_corr[i]) * (C[i] - C_corr[i]);
    }
    printf("[%s] pass=%d, r=%.10f \n", name, r <= 1e-8, r);
}


int main(int argc, char** argv)
{
    double *A, *B, *C, *C_corr;
    int m, n, k, i, j;
    double alpha, beta;

    m = n = k = 128;
    alpha = 1.0; beta = 2.0;

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
    C_corr = (double *) _mm_malloc( m * n * sizeof( double ), 64);

    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(-i-1);
    }

    /* Start test */
    test_one(m, n, k, alpha, beta, A, B, C, C_corr, &dgemm_mkl, "MKL");
    // test_one(m, n, k, alpha, beta, A, B, C, C_corr, &dgemm_ideal, "ideal");
    // test_one(m, n, k, alpha, beta, A, B, C, C_corr, &dgemm_vec, "VEC");
    test_one(m, n, k, alpha, beta, A, B, C, C_corr, &dgemm_mypack, "PACK");
    test_one(m, n, k, alpha, beta, A, B, C, C_corr, &dgemm_v1, "v1");


    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
