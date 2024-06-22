#include "gemm.cuh"


void gemm_00_cublas(cublasHandle_t handle, GEMM_FUNC_SIGNITURE) {
    // cublas uses column-major order, while we use row-major order.
    // So we compute C^T=alpha * B * A^T + beta * C^T.
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, K,
                A, K,
                &beta,
                C, N);
}