#include "dgemm_omp.h"
#include "mkl.h"

void dgemm_omp_v1(DGEMM_FUNC_SIGNITURE) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}