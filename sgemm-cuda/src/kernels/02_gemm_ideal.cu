#include "gemm.cuh"

const int BLOCK_DIM = 16;

__global__ static void kernel_gemm_ideal(GEMM_FUNC_SIGNITURE) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        double tmp = 0.0;
        for (int k = 0; k < K; k++) {
            tmp += A[0] * B[0];
        }
        C[0] = alpha * tmp + beta * C[0];
    }
}

void gemm_02_ideal(GEMM_FUNC_SIGNITURE) {
    dim3 gridDim(CEIL_DIV(M, BLOCK_DIM), CEIL_DIV(N, BLOCK_DIM));
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    kernel_gemm_ideal<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
