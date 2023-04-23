#include "dgemm.h"
#include "cblas.h"

void dgemm_openblas(DGEMM_FUNC_SIGNITURE) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        M, N, K, alpha, A, K, B, N, beta, C, N);
}