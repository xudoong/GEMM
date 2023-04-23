#include "dgemm.h"
#include "mkl.h"

void dgemm_mkl(DGEMM_FUNC_SIGNITURE) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        M, N, K, alpha, A, K, B, N, beta, C, N);
}