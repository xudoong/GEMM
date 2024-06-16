#include "gemm.cuh"

const int BLOCK_DIM = 16;

__global__ static void kernel(GEMM_FUNC_SIGNITURE) {
    const uint i = blockIdx.y * blockDim.y + threadIdx.y;
    const uint j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float tmp = 0.0;
        for (int k = 0; k < K; k++) {
            tmp += float(A[i * K + k]) * float(B[k * N + j]);
        }
        C[i * N + j] = alpha * tmp + beta * C[i * N + j];
    }
}

void gemm_02_naive(GEMM_FUNC_SIGNITURE) {
    dim3 gridDim(CEIL_DIV(M, BLOCK_DIM), CEIL_DIV(N, BLOCK_DIM));
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    kernel<<<gridDim, blockDim>>>(GEMM_FUNC_PARAM);
}
