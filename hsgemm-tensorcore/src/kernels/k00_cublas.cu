#include "gemm.cuh"


void gemm_00_cublas(cublasHandle_t handle, GEMM_FUNC_SIGNITURE) {
    // cublas uses column-major order, while we use row-major order.
    // So we compute C^T=alpha * B^T * A^T + beta * C^T.
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                reinterpret_cast<const __half*>(B), CUDA_R_16F, N,
                reinterpret_cast<const __half*>(A), CUDA_R_16F, K,
                &beta,
                reinterpret_cast<       float*>(C), CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}