#include "gemm.cuh"


void gemm_00_cublas(cublasHandle_t handle, GEMM_FUNC_SIGNITURE) {
    // cublas uses column-major order, while we use row-major order.
    // So we compute C^T=alpha * B^T * A^T + beta * C^T.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
        A, K, &beta, C, N);
}